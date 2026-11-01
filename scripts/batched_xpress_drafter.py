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

"""One XPress drafter forward for a whole batch of requests (A5).

``XPressTreeDrafter`` (#721) drafts one request per forward, so a batch of
``B`` requests pays ``B`` drafter forwards per step. This runs the same
layers once over ``[B, block]`` query rows, with the weights, head and host
search of a given ``XPressTreeDrafter`` (its code is not changed).

Cache layout, as in ``batched_spec_decode.BatchedTreeVerifier``: every
request writes to the same physical cache rows, so each write is one shared
``index_copy_`` that Spyre already compiles.

- The prompts go to rows ``[0, P)`` (``P`` = longest prompt rounded up to the
  append chunk); a row past a request's own prompt is a hole.
- Each draft step writes ``block`` new context rows at the next free pointer;
  request ``b`` uses the first ``m_b`` of them (its newly committed rows), the
  rest are holes. The pointer then moves by ``block``.
- The block's query rows sit after ``capacity``, as in the per-request drafter.

Logical positions (for RoPE and the sliding window) are per request and kept
on the host; masks are host-built per request, so a hole is never attended.
With the same visible rows and positions each request's block sees exactly
what the per-request drafter sees; only the summation order inside a matmul
or a softmax can differ.

**Head rows.** The head (hidden x the 262k-token vocabulary) is about two
thirds of the drafter's work and grows with rows, so batching cannot share
it. The search reads chunk stats only for slots ``0 .. depth``; the other
block rows' head output is thrown away (in the per-request drafter too). So
the head runs on the first ``head_rows = depth + 1`` rows only (half the head
for depth 7); every row's final hidden state is still returned, since the
refiner's summary averages all of them. Exact: the head is per row.
"""

import torch
import torch.nn.functional as F
from hf_adapters._xpress_tree import CHUNK, SLIDING, _rms

from hf_adapters import hf_common
from hf_adapters.hf_common import allocate_kv_cache_tensor, apply_rope_matmul

APPEND_CHUNK = 128  # as XPressTreeDrafter.append


class BatchedXPressDrafter:
    """Batched forwards for ``batch`` requests on an ``XPressTreeDrafter``'s weights.

    ``reset(prompt_rows, max_steps)``, then ``append_prompts`` once, then per
    step ``draft(anchors, new_rows, active)`` and ``search(b, ...)`` per
    request. ``new_rows[b]`` is ``None`` or the ``[m, taps * H]`` host rows
    committed since request ``b``'s previous draft (``m <= block``).
    """

    def __init__(self, tree_drafter, batch, head_rows=None):
        if batch < 1:
            raise ValueError(f"batch must be >= 1, got {batch}")
        block = tree_drafter.block
        if head_rows is None:
            head_rows = min(block, tree_drafter.options["depth"] + 1)
        if not 1 <= head_rows <= block:
            raise ValueError(f"head_rows must be in [1, {block}], got {head_rows}")
        self.d = tree_drafter
        self.batch = int(batch)
        self.head_rows = int(head_rows)
        self._fns = {}
        self.capacity = 0

    # -- device graphs ---------------------------------------------------------

    def _context_kv(self, ctx, weights, rope):
        d = self.d
        Bt, m = ctx.shape[0], ctx.shape[1]
        k = (ctx @ weights["k"]).view(Bt, m, d.n_kv, d.head_dim)
        k = _rms(k, weights["k_norm"], d.eps, 1.0).transpose(1, 2)
        v = (ctx @ weights["v"]).view(Bt, m, d.n_kv, d.head_dim).transpose(1, 2)
        return apply_rope_matmul(k, rope), v

    def _layer_body(
        self, i, h, ctx, rope_ctx, rope_blk, mask, keys, values, c_idx, b_idx
    ):
        d = self.d
        L, Bt, B = d.layers[i], h.shape[0], d.block
        x = _rms(h, L["ln_in"], d.eps)
        q = (x @ L["q"]).view(Bt, B, d.n_heads, d.head_dim)
        q = apply_rope_matmul(
            _rms(q, L["q_norm"], d.eps, 1.0).transpose(1, 2), rope_blk
        )
        kb = (x @ L["k"]).view(Bt, B, d.n_kv, d.head_dim)
        kb = apply_rope_matmul(
            _rms(kb, L["k_norm"], d.eps, 1.0).transpose(1, 2), rope_blk
        )
        vb = (x @ L["v"]).view(Bt, B, d.n_kv, d.head_dim).transpose(1, 2)
        kc, vc = self._context_kv(ctx, L, rope_ctx)
        keys.index_copy_(2, c_idx, kc)
        values.index_copy_(2, c_idx, vc)
        keys.index_copy_(2, b_idx, kb)
        values.index_copy_(2, b_idx, vb)
        o = F.scaled_dot_product_attention(
            q,
            keys,
            values,
            attn_mask=mask,
            dropout_p=0.0,
            scale=d.scaling,
            enable_gqa=True,
        )
        h = h + o.transpose(1, 2).reshape(Bt, B, -1) @ L["o"]
        y = _rms(h, L["ln_post"], d.eps)
        return h + (F.silu(y @ L["gate"]) * (y @ L["up"])) @ L["down"]

    def _head_stats(self, h):
        """``[Bt, block, H]`` -> chunk stats ``[Bt * head_rows, 2 * chunks]`` of the
        first ``head_rows`` rows, and every row's final hidden ``[Bt * block, H]``."""
        d, Bt, R = self.d, h.shape[0], self.head_rows
        hid = _rms(h, d.norm, d.eps)
        rows = Bt * R
        top = hid[:, :R].reshape(rows, -1)
        maxima, firsts = [], []
        for head, bias in zip(d.head, d.head_bias):
            logits = top @ head + bias  # [rows, slice_chunks * 64]
            grouped = logits.reshape(rows, -1, CHUNK)
            cmax = grouped.amax(dim=-1, keepdim=True)
            hit = (grouped == cmax).to(logits.dtype)
            first = hit * d.positions_in_chunk + (1 - hit) * float(CHUNK)
            maxima.append(cmax[..., 0])
            firsts.append(first.amin(dim=-1))
        stats = torch.cat([*maxima, *firsts], dim=-1)
        return stats, hid.reshape(Bt * h.shape[1], -1)

    def _fn(self, kind, i):
        key = (kind, i, self.capacity, self.batch, self.head_rows)
        fn = self._fns.get(key)
        if fn is not None:
            return fn
        d, last = self.d, i == len(self.d.layers) - 1
        if kind == "block":

            def fn(
                h, feats, rope_ctx, rope_blk, mask, keys, values, c_idx, b_idx, ctx=None
            ):
                if i == 0:
                    ctx = d._project(feats)
                h = self._layer_body(
                    i, h, ctx, rope_ctx, rope_blk, mask, keys, values, c_idx, b_idx
                )
                if last:
                    return self._head_stats(h), ctx
                return h, ctx

        else:

            def fn(feats, rope_ctx, keys, values, c_idx, ctx=None):
                if i == 0:
                    ctx = d._project(feats)
                kc, vc = self._context_kv(ctx, d.layers[i], rope_ctx)
                keys.index_copy_(2, c_idx, kc)
                values.index_copy_(2, c_idx, vc)
                return ctx

        if d._compile:
            name = f"xpress_b{self.batch}_r{self.head_rows}_{kind}{i}_c{self.capacity}"
            code = fn.__code__.replace(co_name=name)
            fn = type(fn)(
                code, fn.__globals__, code.co_name, fn.__defaults__, fn.__closure__
            )
            fn = torch.compile(fn, dynamic=False, fullgraph=True)
        self._fns[key] = fn
        return fn

    # -- request state ---------------------------------------------------------

    def reset(self, prompt_rows, max_steps):
        """Allocate caches for prompts of up to ``prompt_rows`` rows + ``max_steps`` drafts."""
        d = self.d
        if prompt_rows < 1 or max_steps < 1:
            raise ValueError("prompt_rows and max_steps must be >= 1")
        self.prompt_region = -(-prompt_rows // APPEND_CHUNK) * APPEND_CHUNK
        need = self.prompt_region + max_steps * d.block
        capacity = -(-need // hf_common.BLOCK_SIZE) * hf_common.BLOCK_SIZE
        if capacity != self.capacity:
            self.capacity = capacity
            rows = capacity + hf_common.BLOCK_SIZE
            shape = (self.batch, d.n_kv, rows, d.head_dim, d.dtype, d.device)
            self.keys = [allocate_kv_cache_tensor(*shape) for _ in d.layers]
            self.values = [allocate_kv_cache_tensor(*shape) for _ in d.layers]
            self.block_index = hf_common.make_cache_index(capacity, d.block, d.device)
        # Logical position of every physical context row per request (-1: hole).
        self.position = torch.full((self.batch, capacity), -1, dtype=torch.long)
        self.rows = [0] * self.batch  # committed context rows per request
        self.last_row = [None] * self.batch
        self.pointer = self.prompt_region

    def _rope(self, positions):
        """``[Bt, L]`` positions -> ``[Bt, L, 2, 2, D/2]`` rotations on the device."""
        d = self.d
        f = positions.double()[..., None] * d._inv_freq
        c, s = f.cos().float(), f.sin().float()
        rot = torch.stack([torch.stack([c, -s], -2), torch.stack([s, c], -2)], -3)
        return rot.to(d.device, d.dtype)

    def _split(self, rows):
        """``[Bt, L, taps * H]`` host rows -> per-tap ``[Bt, L, H]`` device tensors."""
        d = self.d
        H = rows.shape[-1] // len(d.target_layer_ids)
        return [
            rows[..., j * H : (j + 1) * H].to(d.device, d.dtype).contiguous()
            for j in range(len(d.target_layer_ids))
        ]

    @staticmethod
    def _pad(rows, width, fill):
        """``rows`` (``[m, F]``, may be empty) padded to ``width`` with ``fill`` rows."""
        m = 0 if rows is None else rows.shape[0]
        parts = [] if rows is None else [rows]
        if m < width:
            parts.append(fill.expand(width - m, -1))
        return torch.cat(parts)

    @torch.no_grad()
    def append_prompts(self, prompts):
        """Write every request's ``[P_b, taps * H]`` prompt rows (host) at rows ``[0, P_b)``."""
        d = self.d
        if len(prompts) != self.batch:
            raise ValueError(f"expected {self.batch} prompts, got {len(prompts)}")
        lengths = [int(p.shape[0]) for p in prompts]
        if min(lengths) < 1 or max(lengths) > self.prompt_region:
            raise ValueError("prompt rows must be in [1, prompt_rows]")
        for s in range(0, max(lengths), APPEND_CHUNK):
            rows = torch.stack(
                [
                    self._pad(p[s : s + APPEND_CHUNK], APPEND_CHUNK, p[-1:])
                    for p in prompts
                ]
            )
            pos = torch.arange(s, s + APPEND_CHUNK).expand(self.batch, -1)
            index = hf_common.make_cache_index(s, APPEND_CHUNK, d.device)
            rope, ctx = self._rope(pos), None
            for i in range(len(d.layers)):
                ctx = self._fn("append", i)(
                    self._split(rows), rope, self.keys[i], self.values[i], index, ctx
                )
        for b, n in enumerate(lengths):
            self.position[b, :n] = torch.arange(n)
            self.rows[b] = n
            self.last_row[b] = prompts[b][-1:]

    def _masks(self, n):
        """Per-request additive masks ``[Bt, 1, block, cols]`` for queries at ``n[b] ..``."""
        d, cap, B = self.d, self.capacity, self.d.block
        cols = cap + hf_common.BLOCK_SIZE
        n = torch.tensor(n)[:, None]  # [Bt, 1]
        pos = torch.cat(
            [self.position, torch.full((self.batch, cols - cap), -1)], dim=1
        )  # [Bt, cols]
        ctx = (pos >= 0) & (pos < n)
        idx = torch.arange(cols)[None]
        blk = (idx >= cap) & (idx < cap + B)  # [1, cols]
        query = torch.arange(B)[:, None]  # block row i attends block rows <= i
        fill = hf_common._mask_fill_value(d.dtype)
        full = (ctx | blk)[:, None, :].expand(-1, B, -1)
        masks = {"full_attention": full}
        if d.window:
            near = (ctx & (pos >= n - d.window))[:, None, :]
            causal = blk[:, None, :] & ((idx - cap)[:, None, :] <= query[None])
            masks[SLIDING] = near | causal

        def additive(ok):
            m = torch.zeros(ok.shape, dtype=d.dtype).masked_fill(~ok, fill)
            return m[:, None].to(d.device)

        return {k: additive(v) for k, v in masks.items()}

    @torch.no_grad()
    def draft(self, anchors, new_rows, active=None):
        """One forward for every request; host ``(chunk_max, chunk_first, hidden)`` per request.

        Inactive requests (``active[b]`` false) write nothing and their
        output is ``None``.
        """
        d, B, Bt = self.d, self.d.block, self.batch
        active = [True] * Bt if active is None else list(active)
        if not (len(anchors) == len(new_rows) == len(active) == Bt):
            raise ValueError(f"expected {Bt} anchors, row sets and flags")
        if self.pointer + B > self.capacity:
            raise RuntimeError("batched XPress context cache full")
        p = self.pointer
        written, start = [], list(self.rows)
        for b in range(Bt):
            rows = new_rows[b] if active[b] else None
            m = 0 if rows is None else int(rows.reshape(-1, rows.shape[-1]).shape[0])
            if m > B:
                raise ValueError(f"at most {B} new context rows per draft")
            if m:
                rows = rows.reshape(m, -1)
            written.append((rows, m))
        feats = torch.stack(
            [
                self._pad(rows, B, self.last_row[b])
                for b, (rows, _) in enumerate(written)
            ]
        )
        for b, (rows, m) in enumerate(written):
            self.position[b, p : p + m] = torch.arange(start[b], start[b] + m)
            self.rows[b] += m
            if m:
                self.last_row[b] = rows[-1:]
        n = list(self.rows)
        rope_ctx = self._rope(torch.tensor(start)[:, None] + torch.arange(B))
        rope_blk = self._rope(torch.tensor(n)[:, None] + torch.arange(B))
        ids = torch.full((Bt, B), d.mask_id, dtype=torch.long)
        ids[:, 0] = torch.tensor([int(a) for a in anchors])
        h = F.embedding(ids, d.embed_cpu).to(d.device)
        masks = self._masks(n)
        c_idx = hf_common.make_cache_index(p, B, d.device)
        ctx = None
        for i, lt in enumerate(d.layer_types):
            h, ctx = self._fn("block", i)(
                h,
                self._split(feats),
                rope_ctx,
                rope_blk,
                masks[lt],
                self.keys[i],
                self.values[i],
                c_idx,
                self.block_index,
                ctx,
            )
        self.pointer += B
        stats, hidden = h
        stats = stats.to("cpu").float().view(Bt, self.head_rows, -1)
        hidden = hidden.to("cpu").float().view(Bt, B, -1)
        c = d.chunks
        return [
            ((stats[b, :, :c], stats[b, :, c:], hidden[b]) if active[b] else None)
            for b in range(Bt)
        ]


def search(tree_drafter, outputs, anchor, predecessor, budget, propose=None):
    """The per-request host search on precomputed ``outputs`` of one request.

    Runs ``propose`` (default ``tree_drafter.propose(anchor, predecessor,
    budget)``) with the drafter's forward replaced by ``outputs``, so the
    search is exactly #721's (or the beam chain's) code.
    """
    tree_drafter.draft = lambda *a, **k: outputs
    try:
        if propose is None:
            return tree_drafter.propose(anchor, predecessor, budget)
        return propose(tree_drafter, anchor, predecessor)
    finally:
        del tree_drafter.draft
