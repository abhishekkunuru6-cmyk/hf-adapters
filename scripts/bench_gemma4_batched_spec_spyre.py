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
- Diagnostics: ``readback ms`` is the part of ``verify ms`` that brings greedy
  ids to the host (full logits, or chunk stats with ``--device-argmax``);
  ``tok/row`` = tokens/step / width; ``active`` = share of requests still
  decoding per step (finished ones still take verify rows); ``slow`` = steps
  over 2x the median step; ``compiles`` = graphs compiled during measured
  steps (should be 0).
- ``--draft-threads N`` drafts with ``N`` torch CPU threads (labels ``+dtN``);
  the verify keeps the process default. ``--batched-drafter`` runs one
  drafter forward for the whole batch (labels ``+bdraft``).
- Shape ``plain`` is the baseline without speculation: the adapter's own
  batched ``generate`` (1 token per request per step, ``rows`` = batch). Its
  decode time is ``generate(N) - generate(1)`` per group, so prefill is left
  out; EOS is off, so every request decodes exactly N tokens. With a cap set,
  a batch of 2..cap rows takes #721's union path in plain decode too.

Usage (Spyre pod, #721's checkout on PYTHONPATH; start small, compiles are
per configuration)::

    SPYRE_DEVICES=0 PYTHONPATH=~/hf-adapters-721 \\
        python scripts/bench_gemma4_batched_spec_spyre.py \\
        --shapes tree16 chain8 --batch-sizes 1 4 --caps 16 32 \\
        --compare-ar --output batched.json

    # baseline only
    ... --shapes plain --batch-sizes 1 2 4 8 --caps off 16 32 64
"""

import argparse
import importlib.util
import json
import os
import statistics
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_BATCHES = (1, 2, 4, 8)
DEFAULT_CAPS = ("off", "16", "32", "64", "128")
DEFAULT_SHAPES = ("tree16", "chain8", "chain16")
PLAIN = "plain"  # non-speculative batched decoding baseline

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
    readback_ms: float = 0.0  # median per step: logits -> host greedy ids
    tokens_per_row: float = 0.0  # tokens/step / width
    active_share: float = 0.0  # mean share of requests still decoding per step
    mean_step_ms: float = 0.0
    median_step_ms: float = 0.0
    slow_steps: int = 0  # steps over 2x the median step
    compiles: int | None = None  # graphs compiled during the measured steps


def compile_count():
    """Graphs dynamo has compiled so far in this process."""
    from torch._dynamo.utils import counters

    return int(counters["stats"]["unique_graphs"])


def summarize(
    shape, width, batch, cap, runs, matches=None, diffs=None, *, compiles=None
):
    """A ``Row`` from measured ``BatchStats`` runs of one configuration."""
    rows = batch * width
    emitted = [a for s in runs for a in s.emitted()]
    verify = [ms for s in runs for ms in s.verify_ms]
    draft = [ms / s.batch for s in runs for ms in s.draft_ms]
    host = [ms for s in runs for ms in s.host_ms]
    wall = sum(sum(s.draft_ms) + sum(s.verify_ms) + sum(s.host_ms) for s in runs)
    real_diffs = [d for d in (diffs or []) if d is not None]
    readback = [ms for s in runs for ms in s.readback_ms]
    step = [
        d + v + h for s in runs for d, v, h in zip(s.draft_ms, s.verify_ms, s.host_ms)
    ]
    active = [
        sum(a is not None for a in st) / len(st) for s in runs for st in s.accepted
    ]
    median_step = statistics.median(step) if step else 0.0
    tokens_per_step = round(statistics.mean(emitted), 3) if emitted else 0.0
    return Row(
        shape=shape,
        width=width,
        batch=batch,
        cap="off" if cap is None else str(cap),
        status="ok",
        rows=rows,
        covers_top1=cap is None or rows <= cap,
        tokens_per_step=tokens_per_step,
        verify_ms=round(statistics.median(verify), 2) if verify else 0.0,
        draft_ms=round(statistics.median(draft), 2) if draft else 0.0,
        host_ms=round(statistics.median(host), 2) if host else 0.0,
        tokens_per_s=round(1000 * sum(emitted) / max(wall, 1e-9), 2),
        verify_tokens_per_s=round(1000 * sum(emitted) / max(sum(verify), 1e-9), 2),
        steps=sum(s.steps() for s in runs),
        requests=sum(s.batch for s in runs),
        match_ar=None if matches is None else round(statistics.mean(matches), 3),
        median_first_diff=statistics.median(real_diffs) if real_diffs else None,
        readback_ms=round(statistics.median(readback), 2) if readback else 0.0,
        tokens_per_row=round(tokens_per_step / width, 3),
        active_share=round(statistics.mean(active), 3) if active else 0.0,
        mean_step_ms=round(statistics.mean(step), 2) if step else 0.0,
        median_step_ms=round(median_step, 2),
        slow_steps=sum(t > 2 * median_step for t in step),
        compiles=compiles,
    )


def left_pad(prompts):
    """Left-padded ``(input_ids, attention_mask)`` ``[B, longest]`` of 1-D prompts."""
    import torch

    longest = max(int(p.numel()) for p in prompts)
    ids = torch.zeros(len(prompts), longest, dtype=torch.long)
    mask = torch.zeros(len(prompts), longest, dtype=torch.long)
    for b, p in enumerate(prompts):
        ids[b, longest - p.numel() :] = p
        mask[b, longest - p.numel() :] = 1
    return ids, mask


def summarize_plain(batch, cap, decode_ms, *, steps):
    """A ``Row`` for plain decoding: ``decode_ms`` per group of ``steps`` steps."""
    step_ms = [ms / steps for ms in decode_ms]
    tokens = batch * steps * len(decode_ms)
    tps = round(1000 * tokens / max(sum(decode_ms), 1e-9), 2)
    return Row(
        shape=PLAIN,
        width=1,
        batch=batch,
        cap="off" if cap is None else str(cap),
        status="ok",
        rows=batch,
        covers_top1=cap is None or batch <= cap,
        tokens_per_step=1.0,
        verify_ms=round(statistics.median(step_ms), 2),
        tokens_per_s=tps,
        verify_tokens_per_s=tps,
        steps=steps * len(decode_ms),
        requests=batch * len(decode_ms),
    )


def error_summary(e):
    """``Type: message`` of a failure, skipping wrapper header lines.

    Inductor's ``BackendCompilerFailed`` puts only ``backend='inductor'
    raised:`` on its first line; the cause is on the next one.
    """
    lines = [ln.strip() for ln in str(e).strip().splitlines() if ln.strip()]
    while len(lines) > 1 and lines[0].endswith("raised:"):
        lines = lines[1:]
    return f"{type(e).__name__}: {lines[0] if lines else ''}"


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
        "| draft ms/req | tok/s | verify tok/s | readback ms | tok/row | active "
        "| slow | compiles | match_ar | status |"
    )
    lines = [header, "|" + "---|" * 17]
    for r in rows:
        match = "-" if r.match_ar is None else f"{r.match_ar:.2f}"
        compiles = "-" if r.compiles is None else str(r.compiles)
        status = r.status if r.status == "ok" else f"FAIL: {r.error[:60]}"
        lines.append(
            f"| {r.shape} | {r.batch} | {r.cap} | {r.rows} "
            f"| {'yes' if r.covers_top1 else 'no'} | {r.tokens_per_step:.2f} "
            f"| {r.verify_ms:.1f} | {r.draft_ms:.1f} | {r.tokens_per_s:.1f} "
            f"| {r.verify_tokens_per_s:.1f} | {r.readback_ms:.1f} "
            f"| {r.tokens_per_row:.2f} | {r.active_share:.2f} | {r.slow_steps} "
            f"| {compiles} | {match} | {status} |"
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


def config_label(shape_name, args):
    """Row label: the shape plus the options that change how it runs."""
    if shape_name == PLAIN:
        return shape_name
    label = shape_name
    if getattr(args, "device_argmax", False):
        label += "+devargmax"
    threads = getattr(args, "draft_threads", 0)
    if threads:
        label += f"+dt{threads}"
    if getattr(args, "batched_drafter", False):
        label += "+bdraft"
    return label


def run_config(bsd, target, drafter, shape_name, batch, cap, prompts, refs, args):
    shape = bsd.SHAPES[shape_name]
    bsd.configure_union_cap(target, cap)
    cache = args.tree_drafters
    device_argmax = bool(getattr(args, "device_argmax", False))
    decoder = bsd.BatchedSpecDecoder(
        target,
        drafter,
        shape,
        tree_drafter=cache.get(shape_name),
        device_argmax=device_argmax,
        draft_threads=getattr(args, "draft_threads", 0),
        batched_drafter=bool(getattr(args, "batched_drafter", False)),
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
    compiled_before = compile_count()
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
        config_label(shape_name, args),
        shape["width"],
        batch,
        cap,
        runs,
        matches if refs is not None else None,
        diffs,
        compiles=compile_count() - compiled_before,
    )


def run_plain_config(bsd, target, batch, cap, prompts, args):
    """Plain batched greedy decoding: decode ms = generate(N) - generate(1)."""
    bsd.configure_union_cap(target, cap)
    n = args.max_new_tokens
    groups = prompt_groups(list(range(len(prompts))), batch)

    def timed(group, tokens):
        ids, mask = left_pad([prompts[i] for i in group])
        t0 = bsd._clock()
        target.generate(
            ids,
            attention_mask=mask,
            max_new_tokens=tokens,
            min_new_tokens=tokens,
            do_sample=False,
            eos_token_id=None,
        )
        return 1000 * (bsd._clock() - t0)

    timed(groups[0], n)  # warmup: compiles prefill and decode for this batch
    timed(groups[0], 1)
    decode_ms = [timed(g, n) - timed(g, 1) for g in groups]
    return summarize_plain(batch, cap, decode_ms, steps=n - 1)


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
    p.add_argument(
        "--device-argmax",
        action="store_true",
        help="reduce verify logits to chunk stats on the device (A8)",
    )
    p.add_argument(
        "--draft-threads",
        type=int,
        default=0,
        help="torch CPU threads while drafting (A1); 0 = the process default",
    )
    p.add_argument(
        "--batched-drafter",
        action="store_true",
        help="one drafter forward for the whole batch (A5)",
    )
    p.add_argument("--output", help="write every row as JSON here (after each row)")
    args = p.parse_args()
    caps = [parse_cap(c) for c in args.caps]
    if args.draft_threads < 0:
        p.error("--draft-threads must be >= 0")

    import torch
    from transformers import AutoTokenizer

    from hf_adapters import AutoSpyreModelForCausalLM, hf_xpress_qwen3

    bsd = _sibling("batched_spec_decode")
    known = set(bsd.SHAPES) | {PLAIN}
    unknown = set(args.shapes) - known
    if unknown:
        p.error(f"unknown shapes {sorted(unknown)}; choose from {sorted(known)}")
    if args.max_new_tokens < 2:
        p.error("--max-new-tokens must be at least 2")
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
                    if shape == PLAIN:
                        row = run_plain_config(bsd, target, batch, cap, prompts, args)
                    else:
                        row = run_config(
                            bsd, target, drafter, shape, batch, cap, prompts, refs, args
                        )
                except Exception as e:  # a failed compile must not end the sweep
                    traceback.print_exc()  # the full error stays in the log
                    row = failed(
                        config_label(shape, args),
                        1 if shape == PLAIN else bsd.SHAPES[shape]["width"],
                        batch,
                        cap,
                        error_summary(e),
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
