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

"""Batched greedy speculative decoding for Gemma 4 26B-A4B on #721's layers.

#721's ``TreeSpeculativeDecoder`` is batch 1. This runs ``B`` requests in one
verify forward per step, reusing #721's per-layer verify graphs
(``Gemma4MoEBlock.tree_forward``) and XPress tree drafter unchanged:

- **Shared physical rows.** Every request's tree for a step sits in the same
  cache rows ``[pointer, pointer + width)``; ``kv_cache_update`` takes one
  index for the whole batch. The pointer advances by ``width`` every step.
- **No commit copy.** Accepted nodes stay in the rows they were verified in;
  the host records each row's logical position per request (``-1`` for
  padding and rejected nodes) and builds every mask from that. Rejected rows
  are simply never read again. A chain's accepted nodes are already a prefix,
  so for a chain this is what #721's commit does anyway.
- **Full-length caches on every layer.** Holes break the sliding ring's
  "newest rows cover the window" invariant, so sliding layers keep a
  full-length cache too and the window is a mask on positions.
- **The union cap is shared by the batch.** A layer's ``cap`` slot experts
  serve all ``B * width`` rows. Above ``B * width > cap`` a row's top-1 is not
  guaranteed a slot (#721's tree path does not check), so outputs drift.
- **One drafter, per-request state.** The drafter's request state (its
  context KV cache) is swapped per request; drafting runs one request at a
  time, so ``draft_ms`` is the sum over the batch, not a batched draft.
- **Beam chains** (``search="beam"``): the drafter's one forward is searched
  with a beam on the host, and only the path with the largest expected
  accepted length is verified, as a chain.
- **Greedy ids on the device** (``device_argmax``): instead of copying
  ``[B, width, vocab]`` logits to the host, the verify reduces them to each
  64-column chunk's max and first index (the XPress head's trick), and the
  host picks the winning chunk. ``readback_ms`` times this part of a verify.
- **Draft threads** (``draft_threads``): the drafting phase of each step (all
  ``B`` proposals) runs with this many torch CPU threads, and the previous
  count is restored before the verify (whose host argmax wants them all).
  The drafter's host search is a chain of small ops; right after a device
  forward, a large pool pays to wake every thread for each of them. 0 keeps
  the process default. Greedy outputs do not depend on it in exact
  arithmetic; on Spyre, where a target near-tie can round either way
  depending on which verify row computes the token, changed drafts can flip one.

With greedy acceptance against the verify's own predictions, each request's
output is its greedy continuation (as in #721) when the cap is off.
"""

import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
from hf_adapters._gemma4_tree import (
    SLIDING,
    _backbone,
    _final_norm,
    accept_greedy,
    tree_ancestors,
    tree_depths,
)
from hf_adapters._xpress_tree import XPressTreeDrafter

from hf_adapters import hf_common
from hf_adapters.hf_common import (
    BLOCK_SIZE,
    _mask_fill_value,
    allocate_kv_cache_tensor,
    embed_text_tokens,
    kv_cache_shapes,
    make_cache_index,
    text_config,
)

# Draft shapes: a chain is a tree with frontier 1 and branching 1.
SHAPES = {
    "tree16": dict(width=16, frontier=4, branching=4, depth=7),
    "chain8": dict(width=8, frontier=1, branching=1, depth=7),
    "chain16": dict(width=16, frontier=1, branching=1, depth=15),
    "chain4": dict(width=4, frontier=1, branching=1, depth=3),
    "chain6": dict(width=6, frontier=1, branching=1, depth=5),
    # A beam of 4 x branching 4 in the draft; one chain of 8 to verify.
    "chain8-beam": dict(width=8, frontier=4, branching=4, depth=7, search="beam"),
}

CHUNK = 64  # columns per argmax chunk (one stick)
ARGMAX_SLICE = 512 * CHUNK  # columns per reduction slice (see _xpress_tree)

# XPressTreeDrafter's per-request attributes (set by reset/append/draft).
DRAFTER_STATE = ("capacity", "keys", "values", "block_index", "rows", "last_row")


def configure_union_cap(model, cap):
    """Set every MoE layer's union cap (``None``: off) and drop its verify graphs.

    The same as ``SPYRE_MOE_UNION_CAP=cap`` at preparation. Graphs are rebuilt
    by the next ``enable_tree_verify``, with fresh code objects, so each
    ``(cap, batch, width)`` gets its own compiled entries.
    """
    blocks = list(_backbone(model).layers)
    first = blocks[0]
    num_experts = int(first.router.expert_ids.shape[0])
    if cap is not None:
        cap = hf_common.moe_union_cap(first._moe_k, num_experts, raw=str(cap))
    for block in blocks:
        block._moe_union_cap = cap
        identity = torch.eye(cap or 1, dtype=block.router.expert_ids.dtype)
        block.router.slot_identity = identity.to(hf_common.DEVICE)
        block._tree_regions = None
    return cap


def batch_masks(row_pos, node_pos, ancestors, start, window, dtype):
    """Additive ``(global, sliding)`` masks ``[B, 1, T, capacity]``.

    ``row_pos`` ``[B, capacity]``: each committed row's logical position, -1
    for rows a request must not read. ``node_pos`` ``[B, T]``: this step's
    node positions; node ``j`` of every request sits in row ``start + j``.
    A node reads committed rows (on sliding layers only positions within
    ``window`` before its own) and its own ancestors.
    """
    B, T = node_pos.shape
    committed = (row_pos >= 0)[:, None, :].expand(B, T, -1)
    in_window = row_pos[:, None, :] > node_pos[:, :, None] - window
    tree = torch.zeros(B, T, T, dtype=torch.bool)
    for b, paths in enumerate(ancestors):
        for j, path in enumerate(paths):
            tree[b, j, path] = True

    def additive(allowed):
        allowed = allowed.clone()
        allowed[:, :, start : start + T] = tree
        mask = torch.full(allowed.shape, _mask_fill_value(dtype), dtype=dtype)
        return mask.masked_fill(allowed, 0)[:, None]

    return additive(committed), additive(committed & in_window)


def pad_tree(tokens, parents, width):
    """#721's ``_pad_tree``: fill to ``width`` nodes with unused root children."""
    tokens, parents = list(tokens), list(parents)
    used = {t for t, p in zip(tokens, parents) if p == 0}
    filler = 0
    while len(tokens) < width:
        while filler in used or filler == tokens[0]:
            filler += 1
        tokens.append(filler)
        parents.append(0)
        used.add(filler)
    return tokens, parents


def propose_chain_beam(
    drafter, anchor, predecessor, *, depth, beam, branching, new_features=None
):
    """One chain ``(tokens, parents)`` from a beam search over one draft forward.

    Uses the ``XPressTreeDrafter``'s own path-conditioned slot scores. Every
    path the beam reaches is a candidate; the one with the largest expected
    accepted length ``sum_k P(first k tokens)`` is returned (ties: the first
    found). The greedy chain always stays in the beam (a plain beam can prune
    it), so the result is never worse than the greedy chain by this measure.
    With ``beam = branching = 1`` this is the drafter's greedy chain.
    """
    d = drafter
    cmax, first, hidden = d.draft(anchor, new_features)
    cand_ids, cand_vals = d._candidates(cmax, first)
    summary = hidden.mean(dim=0, keepdim=True) @ d.r_summary.T
    hidden_part = hidden @ d.r_hidden.T + summary
    beams = [((int(predecessor), int(anchor)), 0.0, 0.0)]  # path, log-prob, E
    best = greedy = beams[0]
    for k in range(1, min(depth, d.block - 1) + 1):
        prevs = torch.tensor([path for path, _, _ in beams], dtype=torch.long)
        logp = d._slot_logprobs(hidden_part, prevs, k, cand_ids, cand_vals)
        values, index = logp.topk(min(branching, logp.shape[-1]), dim=-1)
        grown, next_greedy = [], None
        for lane, (path, score, expected) in enumerate(beams):
            for j in range(values.shape[-1]):
                s = score + float(values[lane, j])
                token = int(cand_ids[k, int(index[lane, j])])
                grown.append((path + (token,), s, expected + math.exp(s)))
                if path == greedy[0] and j == 0:
                    next_greedy = grown[-1]
        for node in grown:
            if node[2] > best[2]:
                best = node
        greedy = next_greedy
        beams = sorted(grown, key=lambda node: -node[1])[:beam]
        if greedy not in beams:
            beams[-1] = greedy
    tokens = list(best[0][1:])
    return tokens, list(range(-1, len(tokens) - 1))


def vocab_bias(vocab, padded, dtype):
    """``[padded]`` additive bias: 0 on real columns, -1e4 on padding."""
    bias = torch.zeros(padded, dtype=dtype)
    bias[vocab:] = -1e4
    return bias


def chunk_argmax_stats(logits, bias, positions, slice_cols):
    """``[B, T, V]`` logits -> ``[B*T, 2*V/64]``: each chunk's max, then first index.

    Reduced in slices of ``slice_cols`` columns, as the XPress head does on
    Spyre. Indices are exact small integers in fp16.
    """
    rows = logits.shape[0] * logits.shape[1]
    flat = logits.reshape(rows, -1) + bias
    maxima, firsts = [], []
    for s in range(0, flat.shape[-1], slice_cols):
        grouped = flat[:, s : s + slice_cols].reshape(rows, -1, CHUNK)
        cmax = grouped.amax(dim=-1, keepdim=True)
        hit = (grouped == cmax).to(flat.dtype)
        first = hit * positions + (1 - hit) * float(CHUNK)
        maxima.append(cmax[..., 0])
        firsts.append(first.amin(dim=-1))
    return torch.cat([*maxima, *firsts], dim=-1)


def decode_chunk_argmax(stats, chunks):
    """Host ``[rows, 2*chunks]`` stats -> ``[rows]`` argmax (first maximum)."""
    cmax, first = stats[:, :chunks], stats[:, chunks:]
    best = cmax.argmax(dim=-1)
    return best * CHUNK + first.gather(-1, best[:, None])[:, 0].round().long()


def _host(t):
    # Copy in the device dtype, then cast on the host: a materialized
    # fp16 -> fp32 cast on Spyre permutes lanes (see hf_xpress_qwen3).
    return t.to("cpu").float()


@dataclass
class BatchState:
    """Host bookkeeping of one batch."""

    capacity: int
    keys: list
    values: list
    row_pos: torch.Tensor  # [B, capacity] logical position per row, -1 unread
    lengths: list  # per request: tokens committed (= next logical position)
    pointer: int  # next free physical row, shared by the batch
    taps: dict = field(default_factory=dict)  # prefill {layer: [B, P, H]}
    pads: list = field(default_factory=list)
    pending: bool = False


class BatchedTreeVerifier:
    """Prefill, verify and commit for ``B`` requests on a prepared MoE model."""

    def __init__(
        self, model, width, *, tap_layers=(), prefill_chunk=512, device_argmax=False
    ):
        cfg = text_config(model.config)
        if not getattr(cfg, "enable_moe_block", False):
            raise ValueError("BatchedTreeVerifier drives the Gemma 4 MoE adapter")
        if not 2 <= width <= BLOCK_SIZE:
            raise ValueError(f"tree width must be in [2, {BLOCK_SIZE}]")
        self.model = model
        self.width = int(width)
        self.layer_types = list(cfg.layer_types)
        self.window = int(cfg.sliding_window)
        self.prefill_chunk = int(prefill_chunk)
        self.tap_layers = tuple(sorted(int(i) for i in tap_layers))
        self.dtype = hf_common.get_model_dtype(_backbone(model))
        self.blocks = list(_backbone(model).layers)
        for i, block in enumerate(self.blocks):
            block.enable_tree_verify(tap=i in self.tap_layers)
        self.vocab_size = int(cfg.vocab_size)
        self.device_argmax = bool(device_argmax)
        self._argmax = None  # (compiled stats fn, bias, positions), built lazily
        self.state = None

    @torch.no_grad()
    def prefill(self, prompts, max_steps):
        """Prefill ``B`` 1-D prompts (left-padded); returns last logits ``[B, 1, V]``.

        Room is reserved for ``max_steps`` verifies of ``width`` rows each.
        """
        from hf_adapters.hf_gemma4_moe import _run_backbone_forward

        model, chunk, B = self.model, self.prefill_chunk, len(prompts)
        longest = max(int(p.numel()) for p in prompts)
        ids = torch.zeros(B, longest, dtype=torch.long)
        attention = torch.zeros(B, longest, dtype=torch.long)
        for b, p in enumerate(prompts):
            ids[b, longest - p.numel() :] = p
            attention[b, longest - p.numel() :] = 1
        norm = hf_common.normalize_generation_inputs(
            ids, attention, pad_to_multiple=chunk
        )
        padded = norm.padded_len
        pads = [int(x) for x in norm.prompt_offsets]
        capacity = hf_common.generation_cache_len(padded, max_steps * self.width)
        keys, values = [], []
        for n_kv, hd, vd in kv_cache_shapes(model):
            keys.append(allocate_kv_cache_tensor(B, n_kv, capacity, hd, self.dtype))
            values.append(allocate_kv_cache_tensor(B, n_kv, capacity, vd, self.dtype))
        kv_len = hf_common._sdpa_compatible_kv_length(padded)
        pre_k = hf_common._prefill_cache_inputs(keys, kv_len, True)
        pre_v = hf_common._prefill_cache_inputs(values, kv_len, True)
        builder = hf_common._ChunkedPrefillMaskBuilder(
            B,
            chunk,
            kv_len,
            norm.prompt_offsets,
            dtype=self.dtype,
            device=hf_common.DEVICE,
        )
        saved = (
            getattr(model, "_spyre_swa_mode", None),
            getattr(model, "_spyre_padded_prompt_len", None),
            getattr(model, "_spyre_prompt_offsets", None),
            getattr(model, "_spyre_active_prefill_chunk_size", None),
        )
        model._spyre_swa_mode = None
        model._spyre_swa_state = None
        model._spyre_padded_prompt_len = padded
        model._spyre_prompt_offsets = norm.prompt_offsets
        model._spyre_active_prefill_chunk_size = chunk
        taps = {i: [] for i in self.tap_layers}
        compiled = model._spyre_compiled_blocks
        saved_blocks = list(compiled)
        for i in self.tap_layers:

            def tapped(*args, _block=compiled[i], _rows=taps[i]):
                out = _block(*args)
                _rows.append(out[0].to("cpu"))
                return out

            compiled[i] = tapped
        try:
            for begin in range(0, padded, chunk):
                hidden = _run_backbone_forward(
                    model,
                    norm.input_ids[:, begin : begin + chunk].to(hf_common.DEVICE),
                    norm.position_ids[:, begin : begin + chunk].to(hf_common.DEVICE),
                    builder.build(begin),
                    pre_k,
                    pre_v,
                    cache_index=make_cache_index(begin, chunk, hf_common.DEVICE),
                )
        finally:
            (
                model._spyre_swa_mode,
                model._spyre_padded_prompt_len,
                model._spyre_prompt_offsets,
                model._spyre_active_prefill_chunk_size,
            ) = saved
            model._spyre_swa_state = None
            compiled[:] = saved_blocks
        logits = hf_common.run_lm_head(model, hidden, logits_to_keep=1)
        row_pos = torch.full((B, capacity), -1, dtype=torch.long)
        for b, pad in enumerate(pads):
            row_pos[b, pad:padded] = torch.arange(padded - pad)
        self.state = BatchState(
            capacity=capacity,
            keys=keys,
            values=values,
            row_pos=row_pos,
            lengths=[padded - pad for pad in pads],
            pointer=padded,
            taps={i: torch.cat(t, dim=1) for i, t in taps.items() if t},
            pads=pads,
        )
        return logits

    def prompt_features(self, b):
        """Request ``b``'s prefill tap rows ``[P_b, taps * H]`` (tap order)."""
        st = self.state
        rows = [_host(st.taps[i][b, st.pads[b] :]) for i in self.tap_layers]
        return torch.cat(rows, dim=-1)

    @torch.no_grad()
    def verify(self, trees):
        """Score one ``width``-node tree per request. Logits ``[B, width, V_padded]``."""
        st = self.state
        n = self.width
        if st.pending:
            raise RuntimeError("verify called twice without a commit")
        if len(trees) != st.row_pos.shape[0]:
            raise ValueError("one tree per request")
        if any(len(t) != n or len(p) != n for t, p in trees):
            raise ValueError(f"trees must have exactly {n} nodes")
        start = st.pointer
        if start + n > st.capacity:
            raise RuntimeError("KV cache full (raise max_steps)")
        depths = [tree_depths(p) for _, p in trees]
        ancestors = [tree_ancestors(p) for _, p in trees]
        positions = torch.tensor(
            [[length + d for d in ds] for length, ds in zip(st.lengths, depths)]
        )
        full, slid = batch_masks(
            st.row_pos, positions, ancestors, start, self.window, self.dtype
        )
        device = hf_common.DEVICE
        ids = torch.tensor([[int(x) for x in t] for t, _ in trees], dtype=torch.long)
        h = embed_text_tokens(self.model, ids.to(device))
        freqs = {lt: rope(h, positions) for lt, rope in self.model._spyre_rope.items()}
        full, slid = full.to(device), slid.to(device)
        index = make_cache_index(start, n, device)
        for i, block in enumerate(self.blocks):
            sliding = self.layer_types[i] == SLIDING
            h, st.keys[i], st.values[i] = block.tree_forward(
                h,
                freqs[self.layer_types[i]],
                slid if sliding else full,
                st.keys[i],
                st.values[i],
                index,
                block.layer_scalar,
            )
        st.pending = True
        return hf_common.run_lm_head(self.model, _final_norm(self.model, h))

    def greedy_ids(self, logits):
        """Host ``[B, T]`` argmax over the real vocabulary of verify ``logits``."""
        B, T, padded = logits.shape
        if not self.device_argmax:
            return _host(logits[:, :, : self.vocab_size]).argmax(-1)
        if padded % CHUNK:
            raise ValueError(f"padded vocabulary {padded} is not a multiple of {CHUNK}")
        if self._argmax is None or self._argmax[1].shape[0] != padded:
            device = logits.device
            bias = vocab_bias(self.vocab_size, padded, logits.dtype).to(device)
            positions = torch.arange(CHUNK, dtype=logits.dtype).to(device)
            fn = torch.compile(chunk_argmax_stats, dynamic=False, fullgraph=True)
            self._argmax = (fn, bias, positions)
        fn, bias, positions = self._argmax
        stats = _host(fn(logits, bias, positions, ARGMAX_SLICE))
        return decode_chunk_argmax(stats, padded // CHUNK).view(B, T)

    def tap_features(self):
        """``[B, width, taps * H]`` host tap rows of the last verify (node order)."""
        rows = [_host(self.blocks[i].tree_tap) for i in self.tap_layers]
        return torch.cat(rows, dim=-1)

    def commit(self, paths):
        """Keep each request's accepted ``path`` (``None``: keep nothing).

        Host only: the accepted rows become readable at their logical
        positions; the rest of this step's rows stay unread.
        """
        st = self.state
        if not st.pending:
            raise RuntimeError("commit without a verify")
        for b, path in enumerate(paths):
            if path is None:
                continue
            if not path or path[0] != 0:
                raise ValueError("an accepted path starts at the root")
            for depth, node in enumerate(path):
                st.row_pos[b, st.pointer + node] = st.lengths[b] + depth
            st.lengths[b] += len(path)
        st.pointer += self.width
        st.pending = False


class DrafterSlots:
    """One ``XPressTreeDrafter`` serving ``B`` requests by swapping request state."""

    def __init__(self, drafter):
        self.drafter = drafter
        self.states = []

    def reset(self, batch, max_rows):
        self.states = []
        for _ in range(batch):
            self.drafter.capacity = 0  # force fresh caches for this request
            self.drafter.reset(max_rows)
            self.states.append(self._save())

    def _save(self):
        return {name: getattr(self.drafter, name) for name in DRAFTER_STATE}

    @contextmanager
    def use(self, b):
        for name, value in self.states[b].items():
            setattr(self.drafter, name, value)
        try:
            yield self.drafter
        finally:
            self.states[b] = self._save()


@dataclass
class BatchStats:
    """Per-batch timings (ms) and per-request tokens emitted per step."""

    batch: int
    prefill_ms: float = 0.0
    draft_ms: list = field(default_factory=list)  # whole batch, sequential
    verify_ms: list = field(default_factory=list)  # forward + readback
    readback_ms: list = field(default_factory=list)  # logits -> host greedy ids
    host_ms: list = field(default_factory=list)  # accept + commit + taps
    accepted: list = field(default_factory=list)  # [step][request] or None

    def steps(self):
        return len(self.accepted)

    def emitted(self):
        return [a for step in self.accepted for a in step if a is not None]


def _sync():
    sync = getattr(getattr(torch, "spyre", None), "synchronize", None)
    if sync is not None:
        sync()


def _clock():
    _sync()
    return time.perf_counter()


@contextmanager
def torch_threads(n):
    """Run the block with ``n`` torch CPU threads (``0``: leave as is), then restore."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError(f"thread count must be an integer >= 0, got {n!r}")
    if n == 0:
        yield
        return
    previous = torch.get_num_threads()
    torch.set_num_threads(n)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


class BatchedSpecDecoder:
    """Batched drafter + verifier for one prepared target and one XPress drafter."""

    def __init__(
        self,
        target,
        drafter,
        shape,
        *,
        tree_drafter=None,
        draft_vocab=None,
        drafter_kwargs=None,
        device_argmax=False,
        draft_threads=0,
    ):
        """``tree_drafter``: a prebuilt ``XPressTreeDrafter`` for ``shape`` to reuse
        (keeps its compiled graphs across verifier rebuilds). ``draft_threads``:
        torch CPU threads while drafting (0 = the process default)."""
        with torch_threads(draft_threads):  # validates before any work
            pass
        self.draft_threads = draft_threads
        self.width = int(shape["width"])
        self.shape = dict(shape)
        self.taps = tuple(drafter.target_layer_ids)
        self.verifier = BatchedTreeVerifier(
            target, self.width, tap_layers=self.taps, device_argmax=device_argmax
        )
        if tree_drafter is None:
            tree_drafter = XPressTreeDrafter(
                drafter,
                draft_vocab=draft_vocab,
                frontier=shape["frontier"],
                branching=shape["branching"],
                depth=shape["depth"],
                **(drafter_kwargs or {}),
            )
        self.tree_drafter = tree_drafter
        self.slots = DrafterSlots(tree_drafter)
        self.vocab_size = int(text_config(target.config).vocab_size)
        self.stats = None

    def _propose(self, drafter, anchor, predecessor, new_rows):
        shape = self.shape
        if shape.get("search") == "beam":
            return propose_chain_beam(
                drafter,
                anchor,
                predecessor,
                depth=shape["depth"],
                beam=shape["frontier"],
                branching=shape["branching"],
                new_features=new_rows,
            )
        return drafter.propose(anchor, predecessor, self.width, new_rows)

    @torch.no_grad()
    def generate(self, prompts, max_new_tokens, *, max_steps=None, eos_token_id=None):
        """Greedy continuations (lists of ids) of ``B`` 1-D prompt tensors."""
        B = len(prompts)
        max_steps = max_steps or max_new_tokens
        eos = (
            set()
            if eos_token_id is None
            else set([eos_token_id] if isinstance(eos_token_id, int) else eos_token_id)
        )
        v, w = self.verifier, self.width
        stats = self.stats = BatchStats(batch=B)
        t0 = _clock()
        logits = v.prefill(prompts, max_steps)
        first = _host(logits[:, -1, : self.vocab_size]).argmax(-1).tolist()
        # Sized from the padded prefill, not the prompts: a size per batch would
        # recompile the drafter's graphs inside the measured steps.
        self.slots.reset(B, v.state.pointer + max_new_tokens + w)
        for b in range(B):
            with self.slots.use(b) as d:
                d.append(v.prompt_features(b))
        stats.prefill_ms = 1000 * (_clock() - t0)
        outs = [[int(t)] for t in first]
        prompt_last = [int(p[-1]) for p in prompts]
        new_rows = [None] * B

        def active(b):
            return len(outs[b]) < max_new_tokens and outs[b][-1] not in eos

        while any(active(b) for b in range(B)) and stats.steps() < max_steps:
            t0 = _clock()
            trees = []
            with torch_threads(self.draft_threads):
                for b in range(B):
                    if not active(b):
                        trees.append(pad_tree([outs[b][-1]], [-1], w))
                        continue
                    pred = outs[b][-2] if len(outs[b]) > 1 else prompt_last[b]
                    with self.slots.use(b) as d:
                        tokens, parents = self._propose(
                            d, outs[b][-1], pred, new_rows[b]
                        )
                    trees.append(pad_tree(tokens, parents, w))
            t1 = _clock()
            logits = v.verify(trees)
            t_forward = _clock()
            predictions = v.greedy_ids(logits).tolist()
            t2 = _clock()
            features = v.tap_features()
            paths, step = [], []
            for b, (tokens, parents) in enumerate(trees):
                if not active(b):
                    paths.append(None)
                    step.append(None)
                    continue
                path, bonus = accept_greedy(tokens, parents, predictions[b])
                new_rows[b] = features[b][path]
                emitted = [tokens[p] for p in path[1:]] + [bonus]
                for token in emitted:
                    if not active(b):
                        break
                    outs[b].append(token)
                paths.append(path)
                step.append(len(emitted))
            v.commit(paths)
            t3 = _clock()
            stats.draft_ms.append(1000 * (t1 - t0))
            stats.verify_ms.append(1000 * (t2 - t1))
            stats.readback_ms.append(1000 * (t2 - t_forward))
            stats.host_ms.append(1000 * (t3 - t2))
            stats.accepted.append(step)
        return [o[:max_new_tokens] for o in outs]
