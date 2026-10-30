# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Chain vs tree speculative decoding at batch sizes x union caps (Gemma4, #721).

For every ``(cap, shape, batch)`` it runs ``batched_spec_decode`` on Spyre and
reports, per verify step: tokens emitted per request, verify ms, draft ms per
request, and throughput. Each prompt group runs once as a warmup (compiles),
then every group of ``batch`` prompts is measured.

- ``cap`` is the number of experts each MoE layer loads for the whole verify
  (``SPYRE_MOE_UNION_CAP``), shared by all ``batch * width`` rows; ``off`` is
  the per-token decode FFN. ``rows<=cap`` = no: some rows may lose their top-1
  expert, so outputs drift from greedy (see ``match_ar``).
- ``tok/s`` is batch tokens over decode wall time, drafting included (the
  drafter runs one request at a time, so this understates a batched drafter);
  ``verify tok/s`` counts verify time only (the target's ceiling).
- ``match_ar`` is the share of requests whose output equals the plain greedy
  ``generate`` of the same prompt (``--compare-ar``).

Usage (Spyre pod, #721's checkout on PYTHONPATH; start small, compiles are
per configuration)::

    SPYRE_DEVICES=0 PYTHONPATH=~/hf-adapters-721 \\
        python scripts/bench_gemma4_batched_spec_spyre.py \\
        --shapes tree16 chain8 --batch-sizes 1 4 --caps 16 32 \\
        --compare-ar --output batched.json
"""

import argparse
import importlib.util
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_BATCHES = (1, 2, 4, 8)
DEFAULT_CAPS = ("off", "16", "32", "64", "128")
DEFAULT_SHAPES = ("tree16", "chain8", "chain16")

# Distinct tasks, so a batch's requests route to different experts.
DEFAULT_PROMPTS = (
    "Write a Python function that merges two sorted lists, with a docstring "
    "and three unit tests.",
    "Explain how a refrigerator moves heat from inside to outside, step by step.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Summarize the causes and the main consequences of the French Revolution.",
    "Solve step by step: a train leaves at 9:40 at 72 km/h; a second leaves "
    "the same station at 10:15 at 90 km/h. When does the second catch up?",
    "Translate into French and German: 'The meeting is moved to Thursday "
    "because the client is travelling.'",
    "List ten practical tips for writing clear technical documentation, with "
    "one sentence of explanation each.",
    "Write a polite email declining a job offer while keeping the door open "
    "for the future.",
)


def _sibling(name: str):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name(f"{name}.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested on CPU: tests/cpu/test_bench_batched_spec.py)
# ---------------------------------------------------------------------------


def parse_cap(raw: str):
    """``"off"`` -> ``None``; otherwise a positive int."""
    if raw.lower() in ("off", "none", ""):
        return None
    cap = int(raw)
    if cap <= 0:
        raise ValueError(f"cap must be positive or 'off', got {raw!r}")
    return cap


def prompt_groups(prompts, batch):
    """Consecutive groups of ``batch`` prompts (an incomplete tail is dropped)."""
    if batch > len(prompts):
        raise ValueError(f"batch {batch} needs at least {batch} prompts")
    return [prompts[i : i + batch] for i in range(0, len(prompts) - batch + 1, batch)]


def first_difference(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


@dataclass(frozen=True)
class Row:
    shape: str
    width: int
    batch: int
    cap: str
    status: str  # ok | FAIL
    rows: int  # rows per verify: batch * width
    covers_top1: bool  # rows <= cap: every row's top-1 expert gets a slot
    tokens_per_step: float = 0.0  # per request
    verify_ms: float = 0.0  # median per step (whole batch)
    draft_ms: float = 0.0  # median per step, per request
    host_ms: float = 0.0  # median per step: accept + bookkeeping
    tokens_per_s: float = 0.0  # batch tokens / decode wall time
    verify_tokens_per_s: float = 0.0  # batch tokens / verify time only
    steps: int = 0
    requests: int = 0
    match_ar: float | None = None  # share of requests equal to plain greedy
    median_first_diff: float | None = None
    error: str = ""


def summarize(shape, width, batch, cap, runs, matches=None, diffs=None):
    """A ``Row`` from measured ``BatchStats`` runs of one configuration."""
    rows = batch * width
    emitted = [a for s in runs for a in s.emitted()]
    verify = [ms for s in runs for ms in s.verify_ms]
    draft = [ms / s.batch for s in runs for ms in s.draft_ms]
    host = [ms for s in runs for ms in s.host_ms]
    wall = sum(sum(s.draft_ms) + sum(s.verify_ms) + sum(s.host_ms) for s in runs)
    real_diffs = [d for d in (diffs or []) if d is not None]
    return Row(
        shape=shape,
        width=width,
        batch=batch,
        cap="off" if cap is None else str(cap),
        status="ok",
        rows=rows,
        covers_top1=cap is None or rows <= cap,
        tokens_per_step=round(statistics.mean(emitted), 3) if emitted else 0.0,
        verify_ms=round(statistics.median(verify), 2) if verify else 0.0,
        draft_ms=round(statistics.median(draft), 2) if draft else 0.0,
        host_ms=round(statistics.median(host), 2) if host else 0.0,
        tokens_per_s=round(1000 * sum(emitted) / max(wall, 1e-9), 2),
        verify_tokens_per_s=round(1000 * sum(emitted) / max(sum(verify), 1e-9), 2),
        steps=sum(s.steps() for s in runs),
        requests=sum(s.batch for s in runs),
        match_ar=None if matches is None else round(statistics.mean(matches), 3),
        median_first_diff=statistics.median(real_diffs) if real_diffs else None,
    )


def failed(shape, width, batch, cap, error):
    rows = batch * width
    return Row(
        shape=shape,
        width=width,
        batch=batch,
        cap="off" if cap is None else str(cap),
        status="FAIL",
        rows=rows,
        covers_top1=cap is None or rows <= cap,
        error=error,
    )


def format_table(rows):
    """Markdown table, one line per configuration."""
    header = (
        "| shape | batch | cap | rows | rows<=cap | tokens/step | verify ms "
        "| draft ms/req | tok/s | verify tok/s | match_ar | status |"
    )
    lines = [header, "|" + "---|" * 12]
    for r in rows:
        match = "-" if r.match_ar is None else f"{r.match_ar:.2f}"
        status = r.status if r.status == "ok" else f"FAIL: {r.error[:60]}"
        lines.append(
            f"| {r.shape} | {r.batch} | {r.cap} | {r.rows} "
            f"| {'yes' if r.covers_top1 else 'no'} | {r.tokens_per_step:.2f} "
            f"| {r.verify_ms:.1f} | {r.draft_ms:.1f} | {r.tokens_per_s:.1f} "
            f"| {r.verify_tokens_per_s:.1f} | {match} | {status} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Spyre driver
# ---------------------------------------------------------------------------


def _raise_recompile_limit(limit=256):
    """Each configuration compiles fresh graphs; keep dynamo from going eager."""
    import torch._dynamo

    for name in ("recompile_limit", "cache_size_limit"):
        if hasattr(torch._dynamo.config, name):
            setattr(torch._dynamo.config, name, limit)
    if hasattr(torch._dynamo.config, "accumulated_recompile_limit"):
        torch._dynamo.config.accumulated_recompile_limit = limit * 8


def _load_prompts(args, tok):
    from hf_adapters.hf_common import encode_prompts

    texts = [open(f).read() for f in args.prompt_file] or list(DEFAULT_PROMPTS)
    return [encode_prompts(tok, [t])["input_ids"][0] for t in texts]


def _references(target, prompts, max_new_tokens):
    return [
        target.generate(p[None], max_new_tokens=max_new_tokens, do_sample=False)[
            0, p.numel() :
        ].tolist()
        for p in prompts
    ]


def run_config(bsd, target, drafter, shape_name, batch, cap, prompts, refs, args):
    shape = bsd.SHAPES[shape_name]
    bsd.configure_union_cap(target, cap)
    cache = args.tree_drafters
    decoder = bsd.BatchedSpecDecoder(
        target, drafter, shape, tree_drafter=cache.get(shape_name)
    )
    cache[shape_name] = decoder.tree_drafter
    groups = prompt_groups(list(range(len(prompts))), batch)

    def run(group):
        outs = decoder.generate(
            [prompts[i] for i in group],
            args.max_new_tokens,
            max_steps=args.max_steps,
            eos_token_id=args.eos,
        )
        return outs, decoder.stats

    run(groups[0])  # warmup: compiles every graph of this configuration
    runs, matches, diffs = [], [], []
    for group in groups:
        outs, stats = run(group)
        runs.append(stats)
        for i, out in zip(group, outs):
            if refs is not None:
                ref = refs[i][: len(out)]
                diffs.append(first_difference(out, ref))
                matches.append(out == ref)
    return summarize(
        shape_name,
        shape["width"],
        batch,
        cap,
        runs,
        matches if refs is not None else None,
        diffs,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default="google/gemma-4-26B-A4B-it")
    p.add_argument("--drafter", default="UIUC-SSAIL/gemma4-26b-a4b-XPress-b16")
    p.add_argument("--prompt-file", action="append", default=[])
    p.add_argument("--shapes", nargs="+", default=list(DEFAULT_SHAPES))
    p.add_argument("--batch-sizes", nargs="+", type=int, default=list(DEFAULT_BATCHES))
    p.add_argument("--caps", nargs="+", default=list(DEFAULT_CAPS))
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--max-steps", type=int, default=64)
    p.add_argument("--compare-ar", action="store_true")
    p.add_argument("--output", help="write every row as JSON here (after each row)")
    args = p.parse_args()
    caps = [parse_cap(c) for c in args.caps]

    import torch
    from transformers import AutoTokenizer

    from hf_adapters import AutoSpyreModelForCausalLM, hf_xpress_qwen3

    bsd = _sibling("batched_spec_decode")
    unknown = set(args.shapes) - set(bsd.SHAPES)
    if unknown:
        p.error(f"unknown shapes {sorted(unknown)}; choose from {sorted(bsd.SHAPES)}")
    _raise_recompile_limit()
    tok = AutoTokenizer.from_pretrained(args.model)
    args.eos = tok.eos_token_id
    args.tree_drafters = {}  # one drafter per shape, reused across configs
    prompts = _load_prompts(args, tok)
    for batch in args.batch_sizes:
        prompt_groups(prompts, batch)  # fail fast on too few prompts

    t = time.time()
    target = AutoSpyreModelForCausalLM.from_pretrained(args.model, dtype=torch.float16)
    drafter = hf_xpress_qwen3.load_model(
        args.drafter, target_path=args.model, dtype=torch.float32
    )
    print(f"loaded in {time.time() - t:.0f} s", flush=True)
    env = {k: os.getenv(k) for k in ("SPYRE_MOE_UNION_CAP", "SPYRE_GEMMA4_FP8")}
    refs = None
    if args.compare_ar:
        bsd.configure_union_cap(target, None)
        refs = _references(target, prompts, args.max_new_tokens)
        print(f"references done ({len(refs)} prompts)", flush=True)

    rows = []
    for cap in caps:
        for shape in args.shapes:
            for batch in args.batch_sizes:
                t = time.time()
                try:
                    row = run_config(
                        bsd, target, drafter, shape, batch, cap, prompts, refs, args
                    )
                except Exception as e:  # a failed compile must not end the sweep
                    first = (str(e).strip().splitlines() or [""])[0]
                    row = failed(
                        shape,
                        bsd.SHAPES[shape]["width"],
                        batch,
                        cap,
                        f"{type(e).__name__}: {first}",
                    )
                rows.append(row)
                print(
                    json.dumps(asdict(row) | {"seconds": round(time.time() - t)}),
                    flush=True,
                )
                if args.output:
                    with open(args.output, "w") as f:
                        json.dump(
                            dict(env=env, rows=[asdict(r) for r in rows]), f, indent=1
                        )
    print(format_table(rows), flush=True)


if __name__ == "__main__":
    main()
