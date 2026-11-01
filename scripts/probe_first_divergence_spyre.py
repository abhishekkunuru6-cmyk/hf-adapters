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

"""Why speculative outputs first differ from plain greedy (Gemma4, #721).

Every batched configuration so far first differs from the adapter's plain
greedy ``generate`` at about token 17, even with the cap off. With the cap
off both are greedy over the same model, so a difference is either a near
tie that the two graphs (decode vs tree verify) round differently, or a bug.

Per prompt, at the given cap and batch size (``B`` prompts decoded together,
so their verify rows share the cap), this runs the speculative decoder, finds
the first token ``d`` where it differs from plain greedy, then runs the
target once more on the shared prefix (prompt + the ``d`` equal tokens, a
third graph: chunked prefill, never capped) and reports its logits at ``d``:

- ``margin`` = logit(plain's token) - logit(spec's token). Near 0 (fp16
  logits round at ~0.01-0.06 at these magnitudes): a near tie, numerics.
  Large, with spec's token far down the ranking: suspect a bug.
- ``prefill picks``: which of the two the third graph agrees with.
- ``top-1 gap``: logit gap between the prefill's #1 and #2, for scale.

Usage (Spyre pod; loads the target and the drafter)::

    SPYRE_DEVICES=0 PYTHONPATH=~/hf-adapters-721 \\
        python scripts/probe_first_divergence_spyre.py --shapes chain8 tree16 \\
        --output probe.json

    # cap quality at batch, as the bench ran it
    ... --shapes chain8 --cap 64 --batch-sizes 2 4 8 --device-argmax \
        --draft-threads 8 --batched-drafter --output probe_b.json
"""

import argparse
import importlib.util
import json
from pathlib import Path


def _sibling(name: str):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name(f"{name}.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def divergence_row(logits, ref_token, spec_token, k=5):
    """Target's view at the first differing position, from ``[V]`` host logits."""
    import torch

    logits = logits.float()
    values, ids = logits.topk(min(k, logits.numel()))

    def rank(t):
        return int((logits > logits[t]).sum()) + 1

    ref_logit, spec_logit = float(logits[ref_token]), float(logits[spec_token])
    top1 = int(ids[0])
    picks = "plain" if top1 == ref_token else "spec" if top1 == spec_token else "other"
    return dict(
        ref_token=int(ref_token),
        spec_token=int(spec_token),
        ref_logit=round(ref_logit, 4),
        spec_logit=round(spec_logit, 4),
        margin=round(ref_logit - spec_logit, 4),
        ref_rank=rank(ref_token),
        spec_rank=rank(spec_token),
        prefill_picks=picks,
        top1_gap=round(float(values[0] - values[1]), 4) if values.numel() > 1 else 0.0,
        top=[[int(i), round(float(v), 4)] for i, v in zip(ids, values)],
        finite=bool(torch.isfinite(logits).all()),
    )


def probe(
    decoder, prompts, refs, max_new_tokens, vocab_size, first_difference, batch=1
):
    """One row per prompt: where spec decoding (``batch`` prompts at a time)
    leaves plain greedy, and why. An incomplete tail group is dropped."""
    import torch

    if batch < 1 or batch > len(prompts):
        raise ValueError(f"batch {batch} needs 1..{len(prompts)} prompts")
    bsd = _sibling("batched_spec_decode")
    outs = {}
    for start in range(0, len(prompts) - batch + 1, batch):
        group = list(range(start, start + batch))
        decoded = decoder.generate([prompts[i] for i in group], max_new_tokens)
        outs |= dict(zip(group, decoded))
    rows = []
    for i, out in sorted(outs.items()):
        prompt, ref = prompts[i], refs[i]
        n = min(len(out), len(ref))
        d = first_difference(out[:n], ref[:n])
        row = dict(prompt=i, batch=batch, tokens=n, first_diff=d)
        if d is not None:
            prefix = torch.cat([prompt, torch.tensor(ref[:d], dtype=prompt.dtype)])
            logits = decoder.verifier.prefill([prefix], max_steps=1)
            vec = bsd._host(logits[0, -1, :vocab_size])
            row |= divergence_row(vec, ref[d], out[d])
        rows.append(row)
    return rows


def format_rows(shape, rows, decode=None):
    def tok(t):
        return repr(decode([t])) if decode else str(t)

    lines = [
        "| shape | batch | prompt | first diff | plain tok | spec tok | margin "
        "| spec rank | prefill picks | top-1 gap |",
        "|" + "---|" * 10,
    ]
    for r in rows:
        head = f"| {shape} | {r.get('batch', 1)} | {r['prompt']}"
        if r["first_diff"] is None:
            lines.append(f"{head} | none | - | - | - | - | - | - |")
            continue
        lines.append(
            f"{head} | {r['first_diff']} | {tok(r['ref_token'])} "
            f"| {tok(r['spec_token'])} | {r['margin']:.3f} | {r['spec_rank']} "
            f"| {r['prefill_picks']} | {r['top1_gap']:.3f} |"
        )
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default="google/gemma-4-26B-A4B-it")
    p.add_argument("--drafter", default="UIUC-SSAIL/gemma4-26b-a4b-XPress-b16")
    p.add_argument("--prompt-file", action="append", default=[])
    p.add_argument("--shapes", nargs="+", default=["chain8", "tree16"])
    p.add_argument("--cap", default="off", help="union cap for the spec run")
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[1])
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--device-argmax", action="store_true")
    p.add_argument("--draft-threads", type=int, default=0)
    p.add_argument("--batched-drafter", action="store_true")
    p.add_argument("--output", help="write all rows as JSON here")
    args = p.parse_args()
    if args.draft_threads < 0:
        p.error("--draft-threads must be >= 0")
    if any(b < 1 for b in args.batch_sizes):
        p.error("--batch-sizes must be >= 1")

    import torch
    from transformers import AutoTokenizer

    from hf_adapters import AutoSpyreModelForCausalLM, hf_xpress_qwen3

    bench = _sibling("bench_gemma4_batched_spec_spyre")
    bsd = _sibling("batched_spec_decode")
    unknown = set(args.shapes) - set(bsd.SHAPES)
    if unknown:
        p.error(f"unknown shapes {sorted(unknown)}; choose from {sorted(bsd.SHAPES)}")
    cap = bench.parse_cap(args.cap)
    bench._raise_recompile_limit()
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = bench._load_prompts(args, tok)
    target = AutoSpyreModelForCausalLM.from_pretrained(args.model, dtype=torch.float16)
    drafter = hf_xpress_qwen3.load_model(
        args.drafter, target_path=args.model, dtype=torch.float32
    )
    bsd.configure_union_cap(target, None)
    refs = bench._references(target, prompts, args.max_new_tokens)
    print(f"references done ({len(refs)} prompts)", flush=True)
    vocab = int(bsd.text_config(target.config).vocab_size)
    results = {}
    for shape in args.shapes:
        bsd.configure_union_cap(target, cap)
        decoder = bsd.BatchedSpecDecoder(
            target,
            drafter,
            bsd.SHAPES[shape],
            device_argmax=args.device_argmax,
            draft_threads=args.draft_threads,
            batched_drafter=args.batched_drafter,
        )
        label = bench.config_label(shape, args)
        for batch in args.batch_sizes:
            rows = probe(
                decoder,
                prompts,
                refs,
                args.max_new_tokens,
                vocab,
                bench.first_difference,
                batch=batch,
            )
            results[f"{label} B={batch}"] = rows
            print(format_rows(label, rows, tok.decode), flush=True)
        if args.output:
            with open(args.output, "w") as f:
                json.dump(dict(cap=args.cap, rows=results), f, indent=1)


if __name__ == "__main__":
    main()
