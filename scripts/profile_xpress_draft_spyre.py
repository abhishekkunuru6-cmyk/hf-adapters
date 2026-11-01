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

"""Where the XPress draft time goes, per shape (drafter only, no target).

``XPressTreeDrafter.propose`` is one drafter forward over the whole block
(``draft``: host masks + RoPE, the per-layer graphs, the copy back) followed
by a host tree search over the chunk winners. The forward is the same for
every shape; only the search depends on ``frontier`` / ``branching`` /
``depth`` (and ``search="beam"``, which runs ``propose_chain_beam``). This times both parts per step on random context rows (timing
does not depend on the values, only the search's early stop does a little).

Shapes may repeat (``chain8 tree16 chain8``) to expose order effects: each
entry builds its own drafter, as the batched bench does per shape.

``--breakdown`` then times each step of the chain search on one real draft
output (host only), once per ``--threads`` value (0 = the process default):
the candidate setup, each level's readout-row gather, refiner, readout matmul
and log-softmax + top-k, and the whole chain search. ``survivors`` is how many
of a level's candidates an exact bound (``|z . w| <= |z| |w|``) cannot rule
out; it is indicative only, since the context rows here are random.

Usage (Spyre pod; loads only the drafter and the target's embedding)::

    SPYRE_DEVICES=0 PYTHONPATH=~/hf-adapters-721 \\
        python scripts/profile_xpress_draft_spyre.py --shapes chain8 tree16 chain8

    # where the chain search time goes
    ... --shapes chain8 --breakdown --threads 0 1 2 4 8

    # one drafter forward for a whole batch (batched_xpress_drafter, A5)
    ... --shapes chain8 --batched-forward 1 2 4 8

``--threads`` also reruns the per-shape profile once per torch thread count
(0 = the process default), so the real draft loop (forward, then search,
step after step) is timed at each count, not only the hot-loop breakdown.
"""

import argparse
import importlib.util
import json
import statistics
import time
import traceback
from pathlib import Path


def _sibling(name: str):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name(f"{name}.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def median_ms(seconds):
    return 1000 * statistics.median(seconds)


def profile_shape(
    model,
    shape,
    *,
    steps,
    warmup,
    context,
    accepted,
    draft_vocab=None,
    drafter_kwargs=None,
    seed=0,
):
    """Median forward / search / propose ms over ``steps`` measured proposals.

    Each step appends ``accepted`` new context rows (as after a verify that
    accepted ``accepted - 1`` guesses) and proposes a ``width``-node tree.
    """
    import torch
    from hf_adapters._xpress_tree import XPressTreeDrafter

    bsd = _sibling("batched_spec_decode")
    clock = bsd._clock
    d = XPressTreeDrafter(
        model,
        draft_vocab=draft_vocab,
        frontier=shape["frontier"],
        branching=shape["branching"],
        depth=shape["depth"],
        **(drafter_kwargs or {}),
    )
    if not 1 <= accepted <= d.block:
        raise ValueError(f"accepted must be in [1, {d.block}]")
    g = torch.Generator().manual_seed(seed)
    width = len(d.target_layer_ids) * d.fc[0].shape[0]

    def rows(m):
        return torch.randn(m, width, generator=g)

    def token():
        return int(d.vocab[torch.randint(0, d.count, (1,), generator=g)])

    d.reset(context + (steps + warmup) * accepted)
    d.append(rows(context))
    forward, propose = [], []
    real = d.draft

    def timed(*args, **kwargs):
        t0 = clock()
        out = real(*args, **kwargs)
        forward.append(clock() - t0)
        return out

    d.draft = timed
    try:
        for _ in range(warmup + steps):
            new = rows(accepted)
            t0 = clock()
            if shape.get("search") == "beam":
                tokens, _ = bsd.propose_chain_beam(
                    d,
                    token(),
                    token(),
                    depth=shape["depth"],
                    beam=shape["frontier"],
                    branching=shape["branching"],
                    new_features=new,
                )
            else:
                tokens, _ = d.propose(
                    token(), token(), shape["width"], new_features=new
                )
            propose.append(clock() - t0)
    finally:
        del d.draft
    forward, propose = forward[warmup:], propose[warmup:]
    return dict(
        forward_ms=median_ms(forward),
        search_ms=median_ms([p - f for p, f in zip(propose, forward)]),
        propose_ms=median_ms(propose),
        steps=len(propose),
        nodes=len(tokens),
        drafter=d,
    )


def _ms(fn, repeats):
    fn()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    return 1000 * (time.perf_counter() - t0) / repeats


def _breakdown_once(d, cmax, first, hidden, anchor, predecessor, depth, repeats):
    import torch
    import torch.nn.functional as F

    cand_ids, cand_vals = d._candidates(cmax, first)
    summary = hidden.mean(dim=0, keepdim=True) @ d.r_summary.T
    hp = hidden @ d.r_hidden.T + summary
    depth = min(depth, d.block - 1)
    path, parts, survivors = [int(predecessor), int(anchor)], [], []
    for k in range(1, depth + 1):
        prev = torch.tensor([path])
        rows = d.r_readout[cand_ids[k]]

        def refiner(k=k, prev=prev):
            a = hp[None, : k + 1] + d.r_token[prev]
            c = (d.r_mix[k, : k + 1][None] * a).sum(dim=1)
            return c + (F.silu(c @ d.r_gate.T) * (c @ d.r_up.T)) @ d.r_down.T

        z = refiner()
        logits = cand_vals[k][None] + z @ rows.T
        parts.append(
            dict(
                gather_ms=_ms(lambda k=k: d.r_readout[cand_ids[k]], repeats),
                refiner_ms=_ms(refiner, repeats),
                readout_ms=_ms(lambda z=z, rows=rows: z @ rows.T, repeats),
                softmax_topk_ms=_ms(
                    lambda lg=logits: lg.log_softmax(-1).topk(1, dim=-1), repeats
                ),
                level_ms=_ms(
                    lambda k=k, prev=prev: d._slot_logprobs(
                        hp, prev, k, cand_ids, cand_vals
                    ),
                    repeats,
                ),
            )
        )
        bound = z.norm() * rows.norm(dim=-1)
        best_low = (cand_vals[k] - bound).max()
        survivors.append(int((cand_vals[k] + bound >= best_low).sum()))
        path.append(int(cand_ids[k, int(logits.argmax(-1))]))
    d.draft = lambda *a, **kw: (cmax, first, hidden)
    try:
        chain = _ms(lambda: d.propose(anchor, predecessor, depth + 1), repeats)
    finally:
        del d.draft
    row = {key: statistics.mean(p[key] for p in parts) for key in parts[0]}
    return row | dict(
        candidates_ms=_ms(lambda: d._candidates(cmax, first), repeats),
        hidden_part_ms=_ms(
            lambda: hidden @ d.r_hidden.T
            + hidden.mean(dim=0, keepdim=True) @ d.r_summary.T,
            repeats,
        ),
        chain_search_ms=chain,
        survivors=round(statistics.mean(survivors)),
        candidates=int(cand_ids.shape[1]),
    )


def search_breakdown(
    d, cmax, first, hidden, *, anchor, predecessor, depth, threads, repeats=20
):
    """Per-step host times of the chain search, once per torch thread count.

    ``d`` is a chain (frontier = branching = 1) ``XPressTreeDrafter``; ``0`` in
    ``threads`` means the process default. Thread count is restored after.
    """
    import torch

    default = torch.get_num_threads()
    rows = []
    try:
        for t in threads:
            torch.set_num_threads(t or default)
            row = _breakdown_once(
                d, cmax, first, hidden, anchor, predecessor, depth, repeats
            )
            rows.append(dict(threads=torch.get_num_threads()) | row)
    finally:
        torch.set_num_threads(default)
    return rows


def format_breakdown(rows):
    header = (
        "| threads | candidates ms | hidden ms | gather ms/level | refiner ms/level "
        "| readout ms/level | softmax+topk ms/level | level ms | chain search ms "
        "| survivors / candidates |"
    )
    lines = [header, "|" + "---|" * 10]
    for r in rows:
        lines.append(
            f"| {r['threads']} | {r['candidates_ms']:.2f} | {r['hidden_part_ms']:.2f} "
            f"| {r['gather_ms']:.2f} | {r['refiner_ms']:.2f} | {r['readout_ms']:.2f} "
            f"| {r['softmax_topk_ms']:.2f} | {r['level_ms']:.2f} "
            f"| {r['chain_search_ms']:.2f} | {r['survivors']} / {r['candidates']} |"
        )
    return "\n".join(lines)


def profile_batched(tree_drafter, batch, *, steps, warmup, context, accepted, seed=0):
    """Median ms of one batched drafter forward for ``batch`` requests.

    Each step appends ``accepted`` new rows per request, as the batched
    decoder does; prompts are ``context`` random rows each.
    """
    import torch

    bsd = _sibling("batched_spec_decode")
    bxd = _sibling("batched_xpress_drafter")
    d = tree_drafter
    g = torch.Generator().manual_seed(seed)
    width = len(d.target_layer_ids) * d.fc[0].shape[0]
    bd = bxd.BatchedXPressDrafter(d, batch)
    bd.reset(context, warmup + steps)
    bd.append_prompts([torch.randn(context, width, generator=g) for _ in range(batch)])
    times = []
    for _ in range(warmup + steps):
        anchors = [
            int(d.vocab[t]) for t in torch.randint(0, d.count, (batch,), generator=g)
        ]
        new = [torch.randn(accepted, width, generator=g) for _ in range(batch)]
        t0 = bsd._clock()
        bd.draft(anchors, new)
        times.append(bsd._clock() - t0)
    forward = median_ms(times[warmup:])
    return dict(batch=batch, forward_ms=forward, per_request_ms=forward / batch)


def best_threads(rows, shape):
    """Thread count with the lowest median propose time for ``shape``."""
    mine = [r for r in rows if r.get("shape") == shape and "propose_ms" in r]
    if not mine:
        raise ValueError(f"no profile rows for shape {shape!r}")
    return min(mine, key=lambda r: (r["propose_ms"], r["threads"]))["threads"]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default="google/gemma-4-26B-A4B-it")
    p.add_argument("--drafter", default="UIUC-SSAIL/gemma4-26b-a4b-XPress-b16")
    p.add_argument("--shapes", nargs="+", default=["chain8", "tree16", "chain8"])
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--context", type=int, default=64, help="prompt rows")
    p.add_argument("--accepted", type=int, default=4, help="new rows per step")
    p.add_argument("--breakdown", action="store_true", help="time each search step")
    p.add_argument("--threads", nargs="+", type=int, default=[0])
    p.add_argument(
        "--batched-forward",
        nargs="+",
        type=int,
        default=[],
        metavar="B",
        help="time one batched drafter forward at these batch sizes (A5)",
    )
    p.add_argument("--output", help="write the profile rows as JSON here")
    p.add_argument(
        "--pick-threads",
        metavar="JSON",
        help="print the fastest thread count for the first --shapes entry in a "
        "previous --output file, then exit (loads no model)",
    )
    args = p.parse_args()
    if args.pick_threads:
        with open(args.pick_threads) as f:
            print(best_threads(json.load(f)["rows"], args.shapes[0]))
        return
    if any(t < 0 for t in args.threads):
        p.error("--threads values must be >= 0")
    if any(b < 1 for b in args.batched_forward):
        p.error("--batched-forward values must be >= 1")

    import torch

    from hf_adapters import hf_xpress_qwen3

    shapes = _sibling("batched_spec_decode").SHAPES
    unknown = set(args.shapes) - set(shapes)
    if unknown:
        p.error(f"unknown shapes {sorted(unknown)}; choose from {sorted(shapes)}")
    model = hf_xpress_qwen3.load_model(
        args.drafter, target_path=args.model, dtype=torch.float32
    )
    print("| shape | threads | forward ms | search ms | propose ms | nodes |")
    print("|---|---|---|---|---|---|")
    chain = None
    records = []

    def save():
        if args.output:
            with open(args.output, "w") as f:
                json.dump(dict(rows=records), f, indent=1)

    default = torch.get_num_threads()
    for name in args.shapes:
        for t in args.threads:
            torch.set_num_threads(t or default)
            try:
                r = profile_shape(
                    model,
                    shapes[name],
                    steps=args.steps,
                    warmup=args.warmup,
                    context=args.context,
                    accepted=args.accepted,
                )
                used = torch.get_num_threads()
            finally:
                torch.set_num_threads(default)
            print(
                f"| {name} | {used} | {r['forward_ms']:.1f} | {r['search_ms']:.1f} "
                f"| {r['propose_ms']:.1f} | {r['nodes']} |",
                flush=True,
            )
            records.append(
                dict(shape=name, threads=used)
                | {k: r[k] for k in ("forward_ms", "search_ms", "propose_ms", "nodes")}
            )
            save()
            o = shapes[name]
            if (
                chain is None
                and o["frontier"] == o["branching"] == 1
                and "search" not in o
            ):
                chain = (r["drafter"], o["depth"])
    if args.breakdown:
        if chain is None:
            p.error("--breakdown needs a plain chain shape in --shapes")
        d, depth = chain
        anchor = int(d.vocab[0])
        cmax, first, hidden = d.draft(anchor)
        rows = search_breakdown(
            d,
            cmax,
            first,
            hidden,
            anchor=anchor,
            predecessor=anchor,
            depth=depth,
            threads=args.threads,
        )
        print(format_breakdown(rows), flush=True)
        records.extend(dict(kind="breakdown") | r for r in rows)
        save()
    if args.batched_forward:
        if chain is None:
            p.error("--batched-forward needs a plain chain shape in --shapes")
        print("| batch | forward ms | per request ms | status |")
        print("|---|---|---|---|")
        for batch in sorted(args.batched_forward):
            try:
                r = profile_batched(
                    chain[0],
                    batch,
                    steps=args.steps,
                    warmup=args.warmup,
                    context=args.context,
                    accepted=args.accepted,
                )
                line = f"| {batch} | {r['forward_ms']:.1f} | {r['per_request_ms']:.1f} | ok |"
                records.append(dict(kind="batched_forward") | r)
            except Exception as e:  # a failed compile must not hide the others
                traceback.print_exc()
                line = f"| {batch} | - | - | FAIL: {type(e).__name__}: {str(e)[:80]} |"
                records.append(dict(kind="batched_forward", batch=batch, error=str(e)))
            print(line, flush=True)
            save()


if __name__ == "__main__":
    main()
