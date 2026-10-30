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

"""Measure XPress acceptance length with the real Gemma4 target, on CPU (#610).

The go/no-go for the target side needs two numbers: the verify/decode cost
ratio (``scripts/spike_verify_cost_spyre.py``) and how many tokens a block
yields. This script measures the second one, with no Spyre and no
verify/rollback code, in two stages:

``dump``    Stock HF target on CPU: greedy-generate each prompt, then one
            forward over prompt + continuation with ``output_hidden_states``;
            save the token ids and the hidden states at the drafter's
            ``target_layer_ids`` (HF index ``id + tap_offset``).
``replay``  Replay greedy speculative decoding against that continuation: at an
            anchor position, propose with the context up to the anchor, accept
            the longest prefix that matches the target's greedy tokens, then
            move the anchor past the accepted drafts and the target's bonus
            token. Under greedy acceptance this is exactly what a real
            propose/verify loop produces, because the target's tokens and
            hidden states at accepted positions depend only on the prefix.

The drafter runs eagerly on CPU in fp32 through the adapter's own
``propose_block`` (``prepare_for_spyre`` with the compiled callables
unwrapped), so ``--ctx-pad 48`` measures what the Spyre path sees today and a
large ``--ctx-pad`` approximates training (full context, 2048 sliding window).

Usage (the target needs ~55 GB of host RAM in bf16)::

    python scripts/measure_xpress_acceptance.py dump --out taps.pt
    python scripts/measure_xpress_acceptance.py replay --taps taps.pt \\
        --passes 0 6 --predecessor anchor real --ctx-pad 48 1024 \\
        --verify-over-decode 7.18 --draft-ms 50.4 --decode-ms 138.9

``--verify-tokens 4 8 16`` verifies only the anchor + the first k-1 drafts (a
smaller verify touches fewer MoE experts); with ``--verify-cost-json`` from
``scripts/measure_moe_expert_usage.py report`` each k gets its own estimated
verify/decode ratio.
"""

import argparse
import itertools
import json
import statistics
import sys
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TARGET = "google/gemma-4-26B-A4B-it"
DEFAULT_DRAFTER = "UIUC-SSAIL/gemma4-26b-a4b-XPress-b16"
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_TAP_OFFSET = 1  # HF hidden_states[0] is the embedding output
PREDECESSORS = ("anchor", "real")
# "topc": the base logits' top-C per slot (paper / #677). "chunk": each
# CHUNK-id chunk's best token (#721's on-device head reduction).
CANDIDATE_RULES = ("topc", "chunk")
CHUNK = 64  # one Spyre stick of fp16
DEFAULT_PROMPTS = (
    "A store sells pencils at 3 for $0.75 and notebooks at $2.40 each. Maya buys "
    "12 pencils and 5 notebooks and pays with a $20 bill. How much change does "
    "she get? Explain step by step.",
    "Write a Python function that merges two sorted lists into one sorted list "
    "without using sort(), with a docstring and a few doctest examples.",
    "Explain how a hash map handles collisions, comparing separate chaining "
    "and open addressing, and when you would pick each.",
    "Summarize the causes and main consequences of the French Revolution in "
    "two short paragraphs.",
)


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested on CPU: tests/cpu/test_measure_xpress_acceptance.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProposeConfig:
    """One drafter setting to measure."""

    num_passes: int
    topc: int
    predecessor: str  # "anchor" (vLLM serving) or "real" (training)
    ctx_pad: int
    verify_tokens: int | None = None  # tokens the target verifies; None = block
    candidates: str = "topc"  # CANDIDATE_RULES; "chunk" ignores topc

    def label(self) -> str:
        cand = (
            f"topc={self.topc}" if self.candidates == "topc" else f"cand=chunk{CHUNK}"
        )
        label = f"K={self.num_passes} {cand} pred={self.predecessor} ctx={self.ctx_pad}"
        return (
            label if self.verify_tokens is None else f"{label} v={self.verify_tokens}"
        )


@dataclass(frozen=True)
class AcceptanceStats:
    """Greedy acceptance over all replayed blocks of one config."""

    blocks: int
    mean_matches: float  # accepted drafts per block
    tokens_per_block: float  # accepted drafts + the target's bonus token
    full_blocks: int  # blocks where every draft was accepted


def accepted_prefix(draft: Sequence[int], reference: Sequence[int]) -> int:
    """Length of the longest common prefix of the draft and the target tokens."""
    n = 0
    for d, r in zip(draft, reference):
        if d != r:
            break
        n += 1
    return n


def replay_blocks(
    tokens: Sequence[int],
    first_anchor: int,
    block_size: int,
    propose: Callable[[int], Sequence[int]],
    verify_tokens: int | None = None,
) -> list[int]:
    """Accepted-draft count per block of a greedy spec-decode replay.

    ``propose(anchor_pos)`` returns ``block_size - 1`` drafts for positions
    ``anchor_pos + 1 ...``. The target verifies the anchor plus the first
    ``verify_tokens - 1`` drafts (default: the whole block), so a block yields
    at most ``verify_tokens`` tokens. Stops once fewer than a full verify block
    of target tokens remain, so no block is cut short by the end of the
    continuation.
    """
    verify = block_size if verify_tokens is None else verify_tokens
    if not 2 <= verify <= block_size:
        raise ValueError(f"verify_tokens must be in [2, {block_size}], got {verify}")
    matches = []
    anchor = first_anchor
    while anchor + verify <= len(tokens):
        draft = list(propose(anchor))
        if len(draft) != block_size - 1:
            raise ValueError(
                f"propose returned {len(draft)} tokens, expected "
                f"{block_size - 1} draft tokens"
            )
        reference = tokens[anchor + 1 : anchor + verify]
        m = accepted_prefix(draft[: verify - 1], reference)
        matches.append(m)
        anchor += m + 1  # accepted drafts + the target's bonus token
    return matches


def summarize(matches: Sequence[int], block_size: int) -> AcceptanceStats:
    """``block_size``: the most tokens one block can yield (the verify size)."""
    if not matches:
        raise ValueError("no blocks replayed; use longer continuations")
    return AcceptanceStats(
        blocks=len(matches),
        mean_matches=statistics.fmean(matches),
        tokens_per_block=statistics.fmean(m + 1 for m in matches),
        full_blocks=sum(m == block_size - 1 for m in matches),
    )


def projected_speedup(
    tokens_per_block: float, verify_over_decode: float, draft_over_decode: float
) -> float:
    """Spec-decode tok/s over plain decode: a block costs one verify + one draft."""
    cost = verify_over_decode + draft_over_decode
    if cost <= 0:
        raise ValueError("verify + draft cost must be positive")
    return tokens_per_block / cost


def verify_ratio(
    verify_tokens: int,
    per_k: Mapping[str, float] | None,
    fallback: float | None,
) -> float | None:
    """verify/decode for a verify of ``verify_tokens``: the per-k estimate from
    ``measure_moe_expert_usage.py report`` if it has one, else the measured
    spike ratio (today's all-experts path costs the same for any k <= 64)."""
    if per_k is not None and str(verify_tokens) in per_k:
        return per_k[str(verify_tokens)]
    return fallback


def tap_indices(target_layer_ids: Sequence[int], offset: int) -> list[int]:
    """HF ``hidden_states`` indices of the drafter's target layers."""
    if offset < 0:
        raise ValueError(f"tap offset must be non-negative, got {offset}")
    return [i + offset for i in target_layer_ids]


def config_grid(
    passes: Sequence[int],
    topcs: Sequence[int],
    predecessors: Sequence[str],
    ctx_pads: Sequence[int],
    verify_tokens: Sequence[int | None] = (None,),
    candidates: Sequence[str] = ("topc",),
) -> list[ProposeConfig]:
    bad = sorted(set(predecessors) - set(PREDECESSORS))
    if bad:
        raise ValueError(f"Unknown predecessor {bad}; choose from {PREDECESSORS}")
    bad = sorted(set(candidates) - set(CANDIDATE_RULES))
    if bad:
        raise ValueError(f"Unknown candidate rule {bad}; choose from {CANDIDATE_RULES}")
    # (rule, topc) pairs: topc only varies for "topc"; "chunk" ignores it.
    rules = [(r, c) for r in candidates for c in (topcs if r == "topc" else [0])]
    return [
        ProposeConfig(k, c, p, w, v, r)
        for w, k, (r, c), p, v in itertools.product(
            ctx_pads, passes, rules, predecessors, verify_tokens
        )
    ]


def chunk_winners(base, chunk: int = CHUNK):
    """Each ``chunk``-id chunk's max logit and its first index: ``(values, ids)``.

    ``base`` is ``[..., V]``; a partial last chunk is padded with ``-inf``. The
    global argmax always survives (it is its chunk's max); a strong token that
    shares a chunk with a stronger one does not.
    """
    import torch

    vocab = base.shape[-1]
    pad = -vocab % chunk
    padded = torch.nn.functional.pad(base, (0, pad), value=float("-inf"))
    values, index = padded.reshape(*base.shape[:-1], -1, chunk).max(dim=-1)
    offsets = torch.arange(values.shape[-1]) * chunk
    return values, index + offsets


# ---------------------------------------------------------------------------
# Drafter on CPU
# ---------------------------------------------------------------------------


def force_cpu_device() -> None:
    """Point every loaded ``hf_adapters`` module's ``DEVICE`` at CPU.

    The adapter moves tensors to a module-level ``DEVICE`` that each module
    imports by name (``hf_common`` for the RoPE cache, ``_dspark_common`` for
    embeddings/context, the adapter for masks), so all copies must change.
    """
    import sys

    from hf_adapters import _dspark_common, hf_common, hf_xpress_qwen3  # noqa: F401

    for name, module in list(sys.modules.items()):
        if name.startswith("hf_adapters") and hasattr(module, "DEVICE"):
            module.DEVICE = "cpu"


def prepare_cpu_drafter(model, ctx_pad: int):
    """``prepare_for_spyre`` with the compiled callables run eagerly on CPU."""
    import torch

    from hf_adapters import hf_xpress_qwen3

    force_cpu_device()
    hf_xpress_qwen3.prepare_for_spyre(model, ctx_pad=ctx_pad)

    def eager(fn):
        return getattr(fn, "_orig_mod", getattr(fn, "_torchdynamo_orig_callable", fn))

    model._spyre_compiled_blocks = [eager(b) for b in model._spyre_compiled_blocks]
    model._spyre_xpress_fns = {k: eager(f) for k, f in model._spyre_xpress_fns.items()}
    model._spyre_rope.set_dtype(torch.float32)
    return model


def propose_at(model, taps, tokens: Sequence[int], anchor: int, cfg: ProposeConfig):
    """Drafts after ``tokens[anchor]``, with target context ``[0, anchor)``."""
    import torch

    from hf_adapters import hf_xpress_qwen3

    if anchor < 1:
        raise ValueError("the anchor needs at least one context position")
    predecessor = tokens[anchor - 1] if cfg.predecessor == "real" else None
    draft = hf_xpress_qwen3.propose_block(
        model,
        taps[None, :anchor].float(),
        torch.arange(anchor),
        int(tokens[anchor]),
        predecessor_id=None if predecessor is None else int(predecessor),
        num_passes=cfg.num_passes,
        topc=cfg.topc,
        select_candidates=chunk_winners if cfg.candidates == "chunk" else None,
    )
    return draft.tolist()


def measure_record(model, record: dict, cfg: ProposeConfig) -> list[int]:
    """Accepted-draft counts for one dumped prompt (first anchor = first new token)."""
    tokens = record["input_ids"].tolist()
    return replay_blocks(
        tokens,
        first_anchor=int(record["prompt_len"]),
        block_size=model.block_size,
        propose=lambda anchor: propose_at(model, record["taps"], tokens, anchor, cfg),
        verify_tokens=cfg.verify_tokens,
    )


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def _log(message: str) -> None:
    print(f"[xpress-acceptance] {message}", file=sys.stderr, flush=True)


def _load_prompts(path: str | None) -> list[str]:
    if path is None:
        return list(DEFAULT_PROMPTS)
    lines = Path(path).read_text().splitlines()
    prompts = [json.loads(line)["prompt"] for line in lines if line.strip()]
    if not prompts:
        raise ValueError(f"{path} has no prompts (JSONL with a 'prompt' field)")
    return prompts


def run_dump(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config

    layer_ids = Qwen3Config.from_pretrained(args.drafter_path).dflash_config[
        "target_layer_ids"
    ]
    indices = tap_indices(layer_ids, args.tap_offset)
    _log(f"loading {args.target_path} (bf16, CPU); taps at hidden_states{indices}")
    tokenizer = AutoTokenizer.from_pretrained(args.target_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.target_path, dtype=torch.bfloat16
    ).eval()

    records = []
    for i, prompt in enumerate(_load_prompts(args.prompts_file)):
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )["input_ids"]
        start = time.perf_counter()
        with torch.no_grad():
            out_ids = model.generate(
                ids, max_new_tokens=args.max_new_tokens, do_sample=False
            )
            out = model(out_ids, output_hidden_states=True)
        hidden = out.hidden_states
        if max(indices) >= len(hidden):
            raise ValueError(f"tap index {max(indices)} >= {len(hidden)} states")
        taps = torch.cat([hidden[j][0] for j in indices], dim=-1)
        # Greedy tokens re-derived from the one-shot forward; a few bf16
        # near-tie flips vs the cached generate path are expected.
        prompt_len = ids.shape[1]
        regen = out.logits[0, prompt_len - 1 : -1].argmax(-1)
        flips = int((regen != out_ids[0, prompt_len:]).sum())
        records.append(
            {
                "prompt": prompt,
                "input_ids": out_ids[0].cpu(),
                "prompt_len": prompt_len,
                "taps": taps.to(torch.bfloat16).cpu(),
                "forward_flips": flips,
            }
        )
        _log(
            f"prompt {i}: {prompt_len} + {out_ids.shape[1] - prompt_len} tokens, "
            f"{flips} forward/generate flips, {time.perf_counter() - start:.0f} s"
        )
    torch.save(
        {
            "target_path": args.target_path,
            "drafter_path": args.drafter_path,
            "target_layer_ids": list(layer_ids),
            "tap_offset": args.tap_offset,
            "records": records,
        },
        args.out,
    )
    _log(f"wrote {len(records)} records to {args.out}")


def format_table(
    rows: list[tuple[ProposeConfig, AcceptanceStats, float, float | None]],
    draft_over_decode: float,
) -> str:
    """``rows``: (config, stats, CPU s/block, verify/decode or None)."""
    header = (
        f"{'config':<42} {'blocks':>6} {'drafts':>7} {'tok/blk':>8} {'full':>5} "
        f"{'v/d':>6} {'speedup':>8} {'s/blk':>6}"
    )
    lines = [header, "-" * len(header)]
    for cfg, s, sec, ratio in rows:
        if ratio is None:
            cost = f"{'-':>6} {'-':>8}"
        else:
            speedup = projected_speedup(s.tokens_per_block, ratio, draft_over_decode)
            cost = f"{ratio:>6.2f} {speedup:>7.2f}x"
        lines.append(
            f"{cfg.label():<42} {s.blocks:>6} {s.mean_matches:>7.2f} "
            f"{s.tokens_per_block:>8.2f} {s.full_blocks:>5} {cost} {sec:>6.2f}"
        )
    return "\n".join(lines)


def run_replay(args: argparse.Namespace) -> None:
    import torch

    from hf_adapters import hf_xpress_qwen3

    # ctx_pad 48 truncates every block on purpose; that is the point of the sweep.
    warnings.filterwarnings("ignore", message="XPress context of")
    dump = torch.load(args.taps, weights_only=True)
    records = dump["records"][: args.max_prompts]
    _log(f"{len(records)} records from {args.taps} (target {dump['target_path']})")
    embed = hf_xpress_qwen3.load_target_embedding(dump["target_path"])
    lm_head = hf_xpress_qwen3.load_target_lm_head(dump["target_path"])  # None: tied
    _log(f"target lm_head: {'untied' if lm_head is not None else 'tied to embedding'}")
    config, weights = hf_xpress_qwen3.read_checkpoint(dump["drafter_path"])
    if list(config.dflash_config["target_layer_ids"]) != dump["target_layer_ids"]:
        raise ValueError("drafter target_layer_ids differ from the dumped taps")

    grid = config_grid(
        args.passes,
        args.topc,
        args.predecessor,
        args.ctx_pad,
        args.verify_tokens,
        args.candidates,
    )
    per_k = None
    if args.verify_cost_json:
        per_k = json.loads(Path(args.verify_cost_json).read_text())[
            args.verify_cost_key
        ]
    draft_over_decode = args.draft_ms / args.decode_ms

    rows = []
    for ctx_pad, cfgs in itertools.groupby(grid, key=lambda c: c.ctx_pad):
        model = prepare_cpu_drafter(
            hf_xpress_qwen3.build_model(
                config, weights, embed, torch.float32, target_lm_head_weight=lm_head
            ),
            ctx_pad,
        )
        for cfg in cfgs:
            start = time.perf_counter()
            matches = [m for r in records for m in measure_record(model, r, cfg)]
            verify = cfg.verify_tokens or model.block_size
            stats = summarize(matches, verify)
            ratio = verify_ratio(verify, per_k, args.verify_over_decode)
            sec = (time.perf_counter() - start) / stats.blocks
            rows.append((cfg, stats, sec, ratio))
            _log(f"{cfg.label()}: {stats.tokens_per_block:.2f} tokens/block")
        del model
    print(format_table(rows, draft_over_decode))
    print(
        f"\nspeedup = tokens/block / (v/d + draft/decode {draft_over_decode:.2f}); "
        "v/d = verify/decode"
        + (
            f" from {args.verify_cost_json}:{args.verify_cost_key} (falls back to --verify-over-decode)"
            if per_k is not None
            else " (--verify-over-decode)"
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="stage", required=True)

    dump = sub.add_parser("dump", help="run the target, save tokens + taps")
    dump.add_argument("--target-path", default=DEFAULT_TARGET)
    dump.add_argument("--drafter-path", default=DEFAULT_DRAFTER)
    dump.add_argument("--prompts-file", help="JSONL with a 'prompt' field per line")
    dump.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    dump.add_argument("--tap-offset", type=int, default=DEFAULT_TAP_OFFSET)
    dump.add_argument("--out", required=True)

    replay = sub.add_parser("replay", help="replay greedy spec decode on the taps")
    replay.add_argument("--taps", required=True)
    replay.add_argument("--max-prompts", type=int)
    replay.add_argument("--passes", type=int, nargs="+", default=[0, 6])
    replay.add_argument("--topc", type=int, nargs="+", default=[512])
    replay.add_argument(
        "--predecessor", nargs="+", choices=PREDECESSORS, default=["anchor"]
    )
    replay.add_argument("--ctx-pad", type=int, nargs="+", default=[48])
    replay.add_argument(
        "--candidates",
        nargs="+",
        choices=CANDIDATE_RULES,
        default=["topc"],
        help=f"per-slot candidates: top-C of the base logits (--topc), or each "
        f"{CHUNK}-id chunk's best token as in #721 (ignores --topc)",
    )
    replay.add_argument(
        "--verify-tokens",
        type=int,
        nargs="+",
        default=[None],
        help="verify only the anchor + first k-1 drafts (default: the whole block)",
    )
    replay.add_argument(
        "--verify-over-decode", type=float, help="from the verify-cost spike"
    )
    replay.add_argument(
        "--verify-cost-json",
        help="per-k verify/decode from measure_moe_expert_usage.py report --out-json",
    )
    replay.add_argument(
        "--verify-cost-key",
        default="verify_over_decode",
        choices=(
            "verify_over_decode",
            "verify_over_decode_static",
            "verify_over_decode_uniform",
        ),
        help="which cost in --verify-cost-json: mean (dynamic), per-layer or "
        "uniform static slots",
    )
    replay.add_argument("--draft-ms", type=float, default=50.4)
    replay.add_argument("--decode-ms", type=float, default=138.9)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.stage == "dump":
        run_dump(args)
    else:
        run_replay(args)


if __name__ == "__main__":
    main()
