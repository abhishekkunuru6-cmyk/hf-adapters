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
``depth``. This times both parts per step on random context rows (timing
does not depend on the values, only the search's early stop does a little).

Shapes may repeat (``chain8 tree16 chain8``) to expose order effects: each
entry builds its own drafter, as the batched bench does per shape.

Usage (Spyre pod; loads only the drafter and the target's embedding)::

    SPYRE_DEVICES=0 PYTHONPATH=~/hf-adapters-721 \\
        python scripts/profile_xpress_draft_spyre.py --shapes chain8 tree16 chain8
"""

import argparse
import importlib.util
import statistics
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

    clock = _sibling("batched_spec_decode")._clock
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
            tokens, _ = d.propose(token(), token(), shape["width"], new_features=new)
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


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default="google/gemma-4-26B-A4B-it")
    p.add_argument("--drafter", default="UIUC-SSAIL/gemma4-26b-a4b-XPress-b16")
    p.add_argument("--shapes", nargs="+", default=["chain8", "tree16", "chain8"])
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--context", type=int, default=64, help="prompt rows")
    p.add_argument("--accepted", type=int, default=4, help="new rows per step")
    args = p.parse_args()

    import torch

    from hf_adapters import hf_xpress_qwen3

    shapes = _sibling("batched_spec_decode").SHAPES
    unknown = set(args.shapes) - set(shapes)
    if unknown:
        p.error(f"unknown shapes {sorted(unknown)}; choose from {sorted(shapes)}")
    model = hf_xpress_qwen3.load_model(
        args.drafter, target_path=args.model, dtype=torch.float32
    )
    print("| shape | forward ms | search ms | propose ms | nodes |")
    print("|---|---|---|---|---|")
    for name in args.shapes:
        r = profile_shape(
            model,
            shapes[name],
            steps=args.steps,
            warmup=args.warmup,
            context=args.context,
            accepted=args.accepted,
        )
        print(
            f"| {name} | {r['forward_ms']:.1f} | {r['search_ms']:.1f} "
            f"| {r['propose_ms']:.1f} | {r['nodes']} |",
            flush=True,
        )


if __name__ == "__main__":
    main()
