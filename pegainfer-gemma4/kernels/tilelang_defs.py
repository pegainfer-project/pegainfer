"""TileLang definitions of Gemma 4's attention: the global family's at head
dim 512, and the sliding family's windowed reads at 256 further down.

Two kernels for the global prefill's causal read over a prompt, and two for
the decode's split-KV read over a request's whole context. The decode is a
different problem: its compute is trivial and only the bytes in flight
matter, so what makes it fast is different too, and is written at its own
definition below.

Authored here, not vendored: upstream has no kernel for this head dim on
SM90 — FlashInfer's Hopper prefill compiles 64, 128 and 256 only — so the
serving path runs a generic paged kernel that leaves most of the machine
idle at this shape.

Three things make this faster, and none of them is the loop body:

  * the grid walks a KV group's query heads across a block of query tiles
    before advancing, so the CTAs that want the same key tile stay resident
    together and step through the context in lockstep;
  * the score tile goes from shared memory straight into the value gemm,
    because copying it back into a fragment first costs a fifth of the
    runtime for a bit-identical result; and
  * one key tile is exactly one page, so its load is a single unwrapped
    copy — wrapping it in a loop, or splitting it into per-page pieces,
    forfeits the bulk path and with it about half the throughput.

The last one is why the global KV family pages at the key block rather than
at the sliding family's finer granularity.

Ragged batches are the serving case: a mixed step launches its prompt rows
as one plan with an entry per segment. The kernel adds no array of its own:
it walks the batch once per CTA to find which request owns it, summing each
request's CTA count with the tile count rounded up to a whole mapping block
so the group walk stays intact. That is a handful of shifts over a bounded
batch, and it keeps the inputs to exactly what the plan already carries. A
request that is not in the step contributes no CTAs, so it is never chosen,
and a CTA past its request's real tiles exits before it touches memory.

Each request's context length comes the same way, from how far its slice of
the page table reaches and how full its last page is, rather than as another
array restating what those already say.

Rows are packed, so the store is predicated: a partial last tile would
otherwise write over the next request's rows, which in a mixed step are the
decode rows sharing the output buffer. Keeping the bulk copy for whole tiles
and branching to the predicated store only for a request's last one measures
slower than predicating every tile, so there is no branch.
"""

import tilelang
import tilelang.language as T

DTYPE = "bfloat16"
ACC = "float"

# The fp32 output accumulator is block_M * head_dim * 4 B, so block_M is 64:
# 128 spills half the register file. block_N is the page size, and the two
# gemms want a full warpgroup pair.
BLOCK_M = 64
BLOCK_N = 64
NUM_STAGES = 1
THREADS = 256

# Query tiles per KV group before the grid advances.
QBLK = 8

PASS_CONFIGS = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}


def folded_value_bands(dim, fold_rotary):
    """Where a folded row's value operand puts each head column, as bands.

    A folded row is `[K_rot | V_identity | V_rot]`, `dim + fold_rotary` wide,
    in the permutation of `KvFormat::permute` (paged_kv.rs): with
    `rh = fold_rotary / 2` and `h = dim / 2`, head columns `[0, rh)` and
    `[h, h + rh)` are the rotated set and come first, then `[rh, h)` and
    `[h + rh, dim)`. The score operand is the row's first `dim` columns and
    the value operand its last `dim`, so the value operand's columns hold, in
    order, `[rh, h)`, `[h + rh, dim)`, then V's rotated set. Undoing that at
    the store is the whole cost of the format on the read side, and it is
    four contiguous bands `(operand column, length, head column)`, so the
    store stays four affine copies rather than one scattered one.
    """
    rh = fold_rotary // 2
    h = dim // 2
    return (
        (0, h - rh, rh),
        (h - rh, h - rh, h + rh),
        (dim - 2 * rh, rh, 0),
        (dim - rh, rh, h),
    )


def prefill_varlen(
    heads,
    groups,
    dim,
    page_size,
    max_batch,
    q_rows,
    pool_rows,
    page_table_len,
    block_M=BLOCK_M,
    block_N=BLOCK_N,
    num_stages=NUM_STAGES,
    threads=THREADS,
    qblk=QBLK,
    fold_rotary=0,
):
    """Causal GQA prefill over the paged pool, ragged across requests.

    `q_rows`, `pool_rows` and `page_table_len` are declared at the serving
    arena's maxima. They reach the generated code only as bounds guards, so a
    step smaller than the arena passes them all; the real bounds come from the
    store predicate and from the walk's own trip count. The two tensors the
    lowering reads through TMA carry their extents in the descriptors the
    launcher builds, so those are the step's own.

    `fold_rotary` is the pool's row format. Zero is the split format: a row
    is one head of K, and the layer's V rows sit one page after its K rows.
    Otherwise it is the folded format, `dim + fold_rotary` wide, whose score
    operand is the row's first `dim` columns and value operand its last
    `dim`, with the output store undoing the row's column order.
    """
    assert block_N == page_size, "one tile must be one page, or the load splits"
    head_kv = heads // groups
    q_shape = [q_rows, heads, dim]
    kv_shape = [pool_rows, head_kv, dim + fold_rotary]
    # The builder takes only TIR ranges as loop iterables, so the four bands
    # are plain integers here and four written-out loops below.
    (a_src, a_len, a_dst), (b_src, b_len, b_dst), (c_src, c_len, c_dst), (d_src, d_len, d_dst) = (
        folded_value_bands(dim, fold_rotary)
    )

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, DTYPE),
        KV: T.Tensor(kv_shape, DTYPE),
        PageIndices: T.Tensor([page_table_len], "int32"),
        PageIndptr: T.Tensor([max_batch + 1], "int32"),
        QIndptr: T.Tensor([max_batch + 1], "int32"),
        LastPageLen: T.Tensor([max_batch], "int32"),
        sm_scale: T.float32,
        total_ctas: T.int32,
        rows_per_page: T.int32,
        layer_row: T.int32,
        Output: T.Tensor(q_shape, DTYPE),
    ):
        with T.Kernel(total_ctas, threads=threads) as pid:
            # The softmax runs on exp2, so the caller's scale carries log2(e)
            # into the exponent. Gemma 4 attends unscaled and passes one;
            # baking the usual 1/sqrt(head_dim) in here would flatten every
            # distribution.
            scale = sm_scale * 1.44269504
            Q_shared = T.alloc_shared([block_M, dim], DTYPE)
            K_shared = T.alloc_shared([block_N, dim], DTYPE)
            S_shared = T.alloc_shared([block_M, block_N], DTYPE)
            V_shared = T.alloc_shared([block_N, dim], DTYPE)
            acc_s = T.alloc_fragment([block_M, block_N], ACC)
            acc_o = T.alloc_fragment([block_M, dim], ACC)
            scores_max = T.alloc_fragment([block_M], ACC)
            scores_max_prev = T.alloc_fragment([block_M], ACC)
            scores_scale = T.alloc_fragment([block_M], ACC)
            scores_sum = T.alloc_fragment([block_M], ACC)
            logsum = T.alloc_fragment([block_M], ACC)
            req = T.alloc_local([1], "int32")
            first_cta = T.alloc_local([1], "int32")
            own_ctas = T.alloc_local([1], "int32")
            walked = T.alloc_local([1], "int32")

            # Which request owns this CTA, and where its block starts. The
            # host sizes the grid with the same sum, so the two agree by
            # construction rather than through an array that could drift.
            req[0] = 0
            first_cta[0] = 0
            own_ctas[0] = 0
            walked[0] = 0
            # `total_ctas` is the host's same sum, so it also says how many
            # boundaries are real: the caller's array is as long as its own
            # batch, while `max_batch` is this kernel's ceiling.
            for i in T.serial(max_batch):
                if walked[0] < total_ctas:
                    mine = (
                        T.ceildiv(T.ceildiv(QIndptr[i + 1] - QIndptr[i], block_M), qblk)
                        * qblk
                        * heads
                    )
                    # Requests sit back to back, so the owner is the last one
                    # that both starts at or before this CTA and has any. The
                    # emptiness test is what keeps a request the step left out
                    # from claiming CTAs it has no tiles for.
                    if walked[0] <= pid and mine > 0:
                        req[0] = i
                        first_cta[0] = walked[0]
                        own_ctas[0] = mine
                    walked[0] = walked[0] + mine
            # An over-sized grid then does nothing rather than recompute
            # somebody else's tile; an under-sized one leaves a tail, which
            # the numerics gate sees.
            if pid < walked[0]:
                b = req[0]
                local = pid - first_cta[0]
                q_tiles = own_ctas[0] // heads
                per_group = q_tiles * groups
                r = local % per_group
                head = local // per_group * groups + (r % (qblk * groups)) // qblk
                q_tile = r // (qblk * groups) * qblk + r % qblk
                kv_head = head // groups
                q_start = QIndptr[b]
                q_len = QIndptr[b + 1] - q_start
                pages = PageIndptr[b + 1] - PageIndptr[b]
                kv_len = (pages - 1) * page_size + LastPageLen[b]
                offset = kv_len - q_len
                row = q_start + q_tile * block_M

                if q_tile * block_M < q_len:
                    T.copy(Q[row : row + block_M, head, :], Q_shared)
                    T.fill(acc_o, 0)
                    T.fill(logsum, 0)
                    T.fill(scores_max, -T.infinity(ACC))

                    loop_range = T.min(
                        T.ceildiv(offset + (q_tile + 1) * block_M, block_N),
                        T.ceildiv(kv_len, block_N),
                    )
                    for k in T.Pipelined(loop_range, num_stages=num_stages):
                        # One page holds every layer's K then V for `page_size`
                        # tokens, so the layer's K block starts `layer_row` into
                        # the page and its V block one page further.
                        kb = PageIndices[PageIndptr[b] + k] * rows_per_page + layer_row
                        if fold_rotary == 0:
                            T.copy(KV[kb : kb + page_size, kv_head, :], K_shared)
                        else:
                            T.copy(KV[kb : kb + page_size, kv_head, 0:dim], K_shared)
                        for i, j in T.Parallel(block_M, block_N):
                            acc_s[i, j] = T.if_then_else(
                                q_tile * block_M + i + offset < k * block_N + j, -1e9, 0
                            )
                        T.gemm(
                            Q_shared,
                            K_shared,
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullCol,
                        )

                        T.copy(scores_max, scores_max_prev)
                        T.fill(scores_max, -T.infinity(ACC))
                        T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                        for i in T.Parallel(block_M):
                            scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                        for i in T.Parallel(block_M):
                            scores_scale[i] = T.exp2(
                                scores_max_prev[i] * scale - scores_max[i] * scale
                            )
                        for i, j in T.Parallel(block_M, block_N):
                            acc_s[i, j] = T.exp2(
                                acc_s[i, j] * scale - scores_max[i] * scale
                            )
                        T.reduce_sum(acc_s, scores_sum, dim=1)
                        for i in T.Parallel(block_M):
                            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                        T.copy(acc_s, S_shared)

                        for i, j in T.Parallel(block_M, dim):
                            acc_o[i, j] *= scores_scale[i]
                        if fold_rotary == 0:
                            T.copy(
                                KV[kb + page_size : kb + 2 * page_size, kv_head, :],
                                V_shared,
                            )
                        else:
                            T.copy(
                                KV[kb : kb + page_size, kv_head, fold_rotary : fold_rotary + dim],
                                V_shared,
                            )
                        T.gemm(
                            S_shared, V_shared, acc_o, policy=T.GemmWarpPolicy.FullCol
                        )

                    for i, j in T.Parallel(block_M, dim):
                        acc_o[i, j] = acc_o[i, j] / logsum[i]

                    # Q_shared is dead once the walk ends, and the four live shared
                    # buffers already sit at the SM90 dynamic limit, so the store
                    # stages through it rather than its own.
                    T.copy(acc_o, Q_shared)
                    if fold_rotary == 0:
                        for i, d in T.Parallel(block_M, dim):
                            if q_tile * block_M + i < q_len:
                                Output[row + i, head, d] = Q_shared[i, d]
                    else:
                        for i, d in T.Parallel(block_M, a_len):
                            if q_tile * block_M + i < q_len:
                                Output[row + i, head, a_dst + d] = Q_shared[i, a_src + d]
                        for i, d in T.Parallel(block_M, b_len):
                            if q_tile * block_M + i < q_len:
                                Output[row + i, head, b_dst + d] = Q_shared[i, b_src + d]
                        for i, d in T.Parallel(block_M, c_len):
                            if q_tile * block_M + i < q_len:
                                Output[row + i, head, c_dst + d] = Q_shared[i, c_src + d]
                        for i, d in T.Parallel(block_M, d_len):
                            if q_tile * block_M + i < q_len:
                                Output[row + i, head, d_dst + d] = Q_shared[i, d_src + d]

    return main


# ---------------------------------------------------------------------------
# The sliding family's prefill: causal within a window, ragged across
# requests, over the local pool's split K|V rows at head dim 256.
#
# The shape of the global prefill with the walk narrowed to the window: a
# query tile sees the keys from the oldest one in its first row's window to
# its last row's own position, and the tile mask adds the window's lower
# bound to the causal one. Measured on GH200 over an 8192-token chunk
# against a 1024-key window, one page a tile: a 128-row query tile beats a
# 64-row one (0.78 vs 0.85 ms), the row warp partition beats the column
# one (0.63 vs 0.78), a second pipeline stage buys nothing, and a key tile
# assembled from four 16-row pages costs more than the generic kernel it
# replaces (1.1-1.4 ms against 1.06) -- so the pool pages at the tile.

LOCAL_BLOCK_M = 128
LOCAL_PREFILL_THREADS = 256


def prefill_window(
    heads,
    groups,
    dim,
    page_size,
    max_batch,
    q_rows,
    pool_rows,
    page_table_len,
    block_M=LOCAL_BLOCK_M,
    num_stages=1,
    threads=LOCAL_PREFILL_THREADS,
):
    """Windowed causal GQA prefill over the paged pool, ragged across requests.

    One key tile is one page, as in the global prefill, so the load is one
    copy. `window_left` is the inclusive key distance a query attends to.
    The bounds are the serving arena's maxima and reach the code only as
    guards; the TMA descriptors carry the step's own extents.
    """
    block_N = page_size
    head_kv = heads // groups
    q_shape = [q_rows, heads, dim]
    kv_shape = [pool_rows, head_kv, dim]

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, DTYPE),
        KV: T.Tensor(kv_shape, DTYPE),
        PageIndices: T.Tensor([page_table_len], "int32"),
        PageIndptr: T.Tensor([max_batch + 1], "int32"),
        QIndptr: T.Tensor([max_batch + 1], "int32"),
        LastPageLen: T.Tensor([max_batch], "int32"),
        sm_scale: T.float32,
        total_ctas: T.int32,
        batch: T.int32,
        rows_per_page: T.int32,
        layer_row: T.int32,
        window_left: T.int32,
        Output: T.Tensor(q_shape, DTYPE),
    ):
        with T.Kernel(total_ctas, threads=threads) as pid:
            scale = sm_scale * 1.44269504
            Q_shared = T.alloc_shared([block_M, dim], DTYPE)
            K_shared = T.alloc_shared([block_N, dim], DTYPE)
            S_shared = T.alloc_shared([block_M, block_N], DTYPE)
            V_shared = T.alloc_shared([block_N, dim], DTYPE)
            acc_s = T.alloc_fragment([block_M, block_N], ACC)
            acc_o = T.alloc_fragment([block_M, dim], ACC)
            scores_max = T.alloc_fragment([block_M], ACC)
            scores_max_prev = T.alloc_fragment([block_M], ACC)
            scores_scale = T.alloc_fragment([block_M], ACC)
            scores_sum = T.alloc_fragment([block_M], ACC)
            logsum = T.alloc_fragment([block_M], ACC)
            # Which request owns this CTA: a request owns one CTA per (query
            # tile, head), requests sit back to back, and `total_ctas` is the
            # host's same sum. The global prefill walks the boundaries in a
            # rolled loop; here that loop cost the kernel a third of its time
            # -- one CTA fits an SM, so nothing hides a serial walk -- so the
            # boundaries are read at once into registers (each index clamped
            # to the step's last real one, so the reads past `batch` stay in
            # bounds and add nothing) and the owner is the last one at or
            # before this CTA, all of it unrolled.
            bounds = T.alloc_local([max_batch + 1], "int32")
            owner = T.alloc_local([1], "int32")
            first_cta = T.alloc_local([1], "int32")
            bounds[0] = 0
            for i in T.unroll(max_batch):
                lo = QIndptr[T.min(i, batch)]
                hi = QIndptr[T.min(i + 1, batch)]
                bounds[i + 1] = bounds[i] + T.ceildiv(hi - lo, block_M) * heads
            owner[0] = 0
            first_cta[0] = 0
            for i in T.unroll(max_batch - 1):
                if pid >= bounds[i + 1]:
                    owner[0] = i + 1
                    first_cta[0] = bounds[i + 1]

            if pid < total_ctas:
                b = owner[0]
                local = pid - first_cta[0]
                # Adjacent CTAs are one query tile's heads, so a kv head's
                # tiles are read once from HBM and again from L2.
                q_tile = local // heads
                head = local % heads
                kv_head = head // groups
                q_start = QIndptr[b]
                q_len = QIndptr[b + 1] - q_start
                pages = PageIndptr[b + 1] - PageIndptr[b]
                kv_len = (pages - 1) * page_size + LastPageLen[b]
                offset = kv_len - q_len
                row = q_start + q_tile * block_M

                if q_tile * block_M < q_len:
                    T.copy(Q[row : row + block_M, head, :], Q_shared)
                    T.fill(acc_o, 0)
                    T.fill(logsum, 0)
                    T.fill(scores_max, -T.infinity(ACC))

                    # From the oldest key the tile's first row may see to the
                    # newest its last row is.
                    k_begin = T.max(q_tile * block_M + offset - window_left, 0) // block_N
                    k_end = T.min(
                        T.ceildiv(offset + (q_tile + 1) * block_M, block_N),
                        T.ceildiv(kv_len, block_N),
                    )
                    for kk in T.Pipelined(k_end - k_begin, num_stages=num_stages):
                        k = k_begin + kk
                        kb = PageIndices[PageIndptr[b] + k] * rows_per_page + layer_row
                        T.copy(KV[kb : kb + page_size, kv_head, :], K_shared)
                        for i, j in T.Parallel(block_M, block_N):
                            acc_s[i, j] = T.if_then_else(
                                (q_tile * block_M + i + offset < k * block_N + j)
                                | (k * block_N + j + window_left < q_tile * block_M + i + offset),
                                -1e9,
                                0,
                            )
                        T.gemm(
                            Q_shared,
                            K_shared,
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )

                        T.copy(scores_max, scores_max_prev)
                        T.fill(scores_max, -T.infinity(ACC))
                        T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                        for i in T.Parallel(block_M):
                            scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                        for i in T.Parallel(block_M):
                            scores_scale[i] = T.exp2(
                                scores_max_prev[i] * scale - scores_max[i] * scale
                            )
                        for i, j in T.Parallel(block_M, block_N):
                            acc_s[i, j] = T.exp2(
                                acc_s[i, j] * scale - scores_max[i] * scale
                            )
                        T.reduce_sum(acc_s, scores_sum, dim=1)
                        for i in T.Parallel(block_M):
                            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                        T.copy(acc_s, S_shared)

                        for i, j in T.Parallel(block_M, dim):
                            acc_o[i, j] *= scores_scale[i]
                        T.copy(
                            KV[kb + page_size : kb + 2 * page_size, kv_head, :],
                            V_shared,
                        )
                        T.gemm(
                            S_shared, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow
                        )

                    for i, j in T.Parallel(block_M, dim):
                        acc_o[i, j] = acc_o[i, j] / logsum[i]

                    T.copy(acc_o, Q_shared)
                    for i, d in T.Parallel(block_M, dim):
                        if q_tile * block_M + i < q_len:
                            Output[row + i, head, d] = Q_shared[i, d]

    return main


# ---------------------------------------------------------------------------
# Decode: split-KV partial and merge. A CTA is one (slot, kv head), a slot one
# chunk of one request, so every KV byte is read from HBM once and reused from
# shared memory by the group's heads. Three shapes the lowering forces:
#
#   * the gemm wants sixteen rows and a group has eight, so the query tile is
#     padded with zero rows, never stored.
#   * that tile is filled elementwise: an eight-row view into it does not
#     lower, the copy's layout is not a bijection.
#   * one key tile is one page. A half-page tile lands on a warp partition the
#     gemm refuses, a narrower block on a layout the copy cannot cover.
#
# The partial pass then reads at about 96% of the card's measured read roof at
# long context; the merge is a small second pass over each request's slots.

DECODE_M_PAD = 16
DECODE_THREADS = 256
DECODE_STAGES = 1
# The folded row is read as one 640-wide tile, 80 KB against the split
# format's two 64 KB tiles, which is what lets a second stage fit under the
# SM90 ceiling and the next page's load overlap this page's gemms.
DECODE_STAGES_FOLDED = 2
# The merge's first level folds this many slots per CTA; the second level
# walks the group heads. One level walking every slot of a 163K request is
# 639 dependent loads on 32 CTAs, and measured at 136 us against 15-20 for
# the two levels.
DECODE_MERGE_SPAN = 32


def decode_stages(fold_rotary):
    """The pipeline depth a row format's partial is lowered with."""
    return DECODE_STAGES_FOLDED if fold_rotary else DECODE_STAGES


def decode_partial(
    heads,
    groups,
    dim,
    page_size,
    max_rows,
    max_batch,
    max_slots,
    pool_rows,
    page_table_len,
    stages=DECODE_STAGES,
    threads=DECODE_THREADS,
    fold_rotary=0,
):
    """The partial pass: one (slot, kv head) per CTA over that slot's chunk.

    Writes, per slot and query head, the chunk's output normalised by its own
    row sum, and the row's log-sum-exp in the exp2 domain; the merge below
    reads both. `chunk_pages` is the chunk the tile indices count in, as
    pages, and comes from the caller so the two agree by construction.
    `fold_rotary` is the pool's row format, as for the prefill; the partial
    output is stored in head order either way, so one merge serves both.
    """
    head_kv = heads // groups
    m_pad = DECODE_M_PAD
    assert groups <= m_pad, "a group has to fit the padded query tile"
    (a_src, a_len, a_dst), (b_src, b_len, b_dst), (c_src, c_len, c_dst), (d_src, d_len, d_dst) = (
        folded_value_bands(dim, fold_rotary)
    )

    @T.prim_func
    def main(
        Q: T.Tensor([max_rows, heads, dim], DTYPE),
        KV: T.Tensor([pool_rows, head_kv, dim + fold_rotary], DTYPE),
        PageIndices: T.Tensor([page_table_len], "int32"),
        PageIndptr: T.Tensor([max_batch + 1], "int32"),
        LastPageLen: T.Tensor([max_batch], "int32"),
        RequestIndices: T.Tensor([max_slots], "int32"),
        KvTileIndices: T.Tensor([max_slots], "int32"),
        ValidMask: T.Tensor([max_slots], "uint8"),
        sm_scale: T.float32,
        rows_per_page: T.int32,
        layer_row: T.int32,
        row_offset: T.int32,
        chunk_pages: T.int32,
        padded_slots: T.int32,
        TmpV: T.Tensor([max_slots, heads, dim], DTYPE),
        TmpS: T.Tensor([max_slots, heads], ACC),
    ):
        with T.Kernel(padded_slots, head_kv, threads=threads) as (s, h):
            scale = sm_scale * 1.44269504
            Q_shared = T.alloc_shared([m_pad, dim], DTYPE)
            if fold_rotary == 0:
                K_shared = T.alloc_shared([page_size, dim], DTYPE)
                V_shared = T.alloc_shared([page_size, dim], DTYPE)
            else:
                # One tile holds the whole row; the score operand is its
                # first `dim` columns and the value operand its last.
                KV_shared = T.alloc_shared([page_size, dim + fold_rotary], DTYPE)
            S_shared = T.alloc_shared([m_pad, page_size], DTYPE)
            acc_s = T.alloc_fragment([m_pad, page_size], ACC)
            acc_o = T.alloc_fragment([m_pad, dim], ACC)
            m_cur = T.alloc_fragment([m_pad], ACC)
            m_prev = T.alloc_fragment([m_pad], ACC)
            m_scale = T.alloc_fragment([m_pad], ACC)
            row_sum = T.alloc_fragment([m_pad], ACC)
            logsum = T.alloc_fragment([m_pad], ACC)

            if ValidMask[s] != 0:
                r = RequestIndices[s]
                t = KvTileIndices[s]
                pages = PageIndptr[r + 1] - PageIndptr[r]
                kv_len = (pages - 1) * page_size + LastPageLen[r]
                first = t * chunk_pages
                n_pages = T.min(chunk_pages, pages - first)

                for i, d in T.Parallel(m_pad, dim):
                    Q_shared[i, d] = T.if_then_else(
                        i < groups, Q[row_offset + r, h * groups + i, d], 0
                    )
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(m_cur, -T.infinity(ACC))

                for p in T.Pipelined(n_pages, num_stages=stages):
                    kb = PageIndices[PageIndptr[r] + first + p] * rows_per_page + layer_row
                    if fold_rotary == 0:
                        T.copy(KV[kb : kb + page_size, h, :], K_shared)
                    else:
                        T.copy(KV[kb : kb + page_size, h, :], KV_shared)
                    # Keys past the request's length live only in its last page.
                    for i, j in T.Parallel(m_pad, page_size):
                        acc_s[i, j] = T.if_then_else(
                            (first + p) * page_size + j < kv_len, 0, -1e9
                        )
                    if fold_rotary == 0:
                        T.gemm(
                            Q_shared,
                            K_shared,
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullCol,
                        )
                    else:
                        T.gemm(
                            Q_shared,
                            KV_shared[:, 0:dim],
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullCol,
                        )
                    T.copy(m_cur, m_prev)
                    T.fill(m_cur, -T.infinity(ACC))
                    T.reduce_max(acc_s, m_cur, dim=1, clear=False)
                    for i in T.Parallel(m_pad):
                        m_cur[i] = T.max(m_cur[i], m_prev[i])
                    for i in T.Parallel(m_pad):
                        m_scale[i] = T.exp2(m_prev[i] * scale - m_cur[i] * scale)
                    for i, j in T.Parallel(m_pad, page_size):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - m_cur[i] * scale)
                    T.reduce_sum(acc_s, row_sum, dim=1)
                    for i in T.Parallel(m_pad):
                        logsum[i] = logsum[i] * m_scale[i] + row_sum[i]
                    T.copy(acc_s, S_shared)
                    for i, d in T.Parallel(m_pad, dim):
                        acc_o[i, d] *= m_scale[i]
                    if fold_rotary == 0:
                        T.copy(KV[kb + page_size : kb + 2 * page_size, h, :], V_shared)
                        T.gemm(S_shared, V_shared, acc_o, policy=T.GemmWarpPolicy.FullCol)
                    else:
                        T.gemm(
                            S_shared,
                            KV_shared[:, fold_rotary : fold_rotary + dim],
                            acc_o,
                            policy=T.GemmWarpPolicy.FullCol,
                        )

                if fold_rotary == 0:
                    for i, d in T.Parallel(m_pad, dim):
                        if i < groups:
                            TmpV[s, h * groups + i, d] = acc_o[i, d] / logsum[i]
                else:
                    # The permuted store goes through shared memory, as the
                    # prefill's does. Storing straight from the accumulator
                    # with a permuted index lowers to a loop whose thread
                    # mapping is inferred from the store rather than from the
                    # fragment, and reads the accumulator in that mapping.
                    for i, d in T.Parallel(m_pad, dim):
                        acc_o[i, d] = acc_o[i, d] / logsum[i]
                    T.copy(acc_o, Q_shared)
                    for i, d in T.Parallel(m_pad, a_len):
                        if i < groups:
                            TmpV[s, h * groups + i, a_dst + d] = Q_shared[i, a_src + d]
                    for i, d in T.Parallel(m_pad, b_len):
                        if i < groups:
                            TmpV[s, h * groups + i, b_dst + d] = Q_shared[i, b_src + d]
                    for i, d in T.Parallel(m_pad, c_len):
                        if i < groups:
                            TmpV[s, h * groups + i, c_dst + d] = Q_shared[i, c_src + d]
                    for i, d in T.Parallel(m_pad, d_len):
                        if i < groups:
                            TmpV[s, h * groups + i, d_dst + d] = Q_shared[i, d_src + d]
                for i in T.Parallel(m_pad):
                    if i < groups:
                        TmpS[s, h * groups + i] = m_cur[i] * scale + T.log2(logsum[i])

    return main


# ---------------------------------------------------------------------------
# The sliding family's decode: the same split-KV partial over the local
# pool's pages, with the window as a mask on the resident keys. A key tile
# is `tile_pages` pages copied side by side into one shared tile; a page
# past the request's is read as its last page and masked by its virtual
# position, so the copy never follows a page id the request does not own.
# The query is the last resident key while the pages hold up to a page more
# than the window, so keys older than `window_left` are masked out rather
# than released.

LOCAL_DECODE_THREADS = 128
LOCAL_DECODE_STAGES = 2


def decode_window_partial(
    heads,
    groups,
    dim,
    page_size,
    tile_pages,
    max_rows,
    max_batch,
    max_slots,
    pool_rows,
    page_table_len,
    stages=LOCAL_DECODE_STAGES,
    threads=LOCAL_DECODE_THREADS,
):
    """The windowed partial pass: one (slot, kv head) per CTA over that
    slot's chunk of the resident window, `tile_pages` pages a tile.

    Writes the same per-slot state as `decode_partial`, so the two merge
    levels serve it unchanged. `chunk_pages` has to be whole tiles, which
    the launcher refuses otherwise.
    """
    head_kv = heads // groups
    m_pad = DECODE_M_PAD
    tile = tile_pages * page_size
    assert groups <= m_pad, "a group has to fit the padded query tile"

    @T.prim_func
    def main(
        Q: T.Tensor([max_rows, heads, dim], DTYPE),
        KV: T.Tensor([pool_rows, head_kv, dim], DTYPE),
        PageIndices: T.Tensor([page_table_len], "int32"),
        PageIndptr: T.Tensor([max_batch + 1], "int32"),
        LastPageLen: T.Tensor([max_batch], "int32"),
        RequestIndices: T.Tensor([max_slots], "int32"),
        KvTileIndices: T.Tensor([max_slots], "int32"),
        ValidMask: T.Tensor([max_slots], "uint8"),
        sm_scale: T.float32,
        rows_per_page: T.int32,
        layer_row: T.int32,
        row_offset: T.int32,
        chunk_pages: T.int32,
        window_left: T.int32,
        padded_slots: T.int32,
        TmpV: T.Tensor([max_slots, heads, dim], DTYPE),
        TmpS: T.Tensor([max_slots, heads], ACC),
    ):
        with T.Kernel(padded_slots, head_kv, threads=threads) as (s, h):
            scale = sm_scale * 1.44269504
            Q_shared = T.alloc_shared([m_pad, dim], DTYPE)
            K_shared = T.alloc_shared([tile, dim], DTYPE)
            V_shared = T.alloc_shared([tile, dim], DTYPE)
            S_shared = T.alloc_shared([m_pad, tile], DTYPE)
            acc_s = T.alloc_fragment([m_pad, tile], ACC)
            acc_o = T.alloc_fragment([m_pad, dim], ACC)
            m_cur = T.alloc_fragment([m_pad], ACC)
            m_prev = T.alloc_fragment([m_pad], ACC)
            m_scale = T.alloc_fragment([m_pad], ACC)
            row_sum = T.alloc_fragment([m_pad], ACC)
            logsum = T.alloc_fragment([m_pad], ACC)

            if ValidMask[s] != 0:
                r = RequestIndices[s]
                t = KvTileIndices[s]
                pages = PageIndptr[r + 1] - PageIndptr[r]
                kv_len = (pages - 1) * page_size + LastPageLen[r]
                first = t * chunk_pages
                n_pages = T.min(chunk_pages, pages - first)
                oldest = kv_len - 1 - window_left

                for i, d in T.Parallel(m_pad, dim):
                    Q_shared[i, d] = T.if_then_else(
                        i < groups, Q[row_offset + r, h * groups + i, d], 0
                    )
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(m_cur, -T.infinity(ACC))

                n_tiles = T.ceildiv(n_pages, tile_pages)
                for tt in T.Pipelined(n_tiles, num_stages=stages):
                    for pp in T.serial(tile_pages):
                        pidx = T.min(first + tt * tile_pages + pp, pages - 1)
                        kb = PageIndices[PageIndptr[r] + pidx] * rows_per_page + layer_row
                        T.copy(
                            KV[kb : kb + page_size, h, :],
                            K_shared[pp * page_size : (pp + 1) * page_size, :],
                        )
                    for i, j in T.Parallel(m_pad, tile):
                        acc_s[i, j] = T.if_then_else(
                            ((first + tt * tile_pages) * page_size + j < kv_len)
                            & ((first + tt * tile_pages) * page_size + j >= oldest),
                            0,
                            -1e9,
                        )
                    T.gemm(
                        Q_shared,
                        K_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullCol,
                    )
                    T.copy(m_cur, m_prev)
                    T.fill(m_cur, -T.infinity(ACC))
                    T.reduce_max(acc_s, m_cur, dim=1, clear=False)
                    for i in T.Parallel(m_pad):
                        m_cur[i] = T.max(m_cur[i], m_prev[i])
                    for i in T.Parallel(m_pad):
                        m_scale[i] = T.exp2(m_prev[i] * scale - m_cur[i] * scale)
                    for i, j in T.Parallel(m_pad, tile):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - m_cur[i] * scale)
                    T.reduce_sum(acc_s, row_sum, dim=1)
                    for i in T.Parallel(m_pad):
                        logsum[i] = logsum[i] * m_scale[i] + row_sum[i]
                    T.copy(acc_s, S_shared)
                    for i, d in T.Parallel(m_pad, dim):
                        acc_o[i, d] *= m_scale[i]
                    for pp in T.serial(tile_pages):
                        pidx = T.min(first + tt * tile_pages + pp, pages - 1)
                        kb = PageIndices[PageIndptr[r] + pidx] * rows_per_page + layer_row
                        T.copy(
                            KV[kb + page_size : kb + 2 * page_size, h, :],
                            V_shared[pp * page_size : (pp + 1) * page_size, :],
                        )
                    T.gemm(S_shared, V_shared, acc_o, policy=T.GemmWarpPolicy.FullCol)

                for i, d in T.Parallel(m_pad, dim):
                    if i < groups:
                        TmpV[s, h * groups + i, d] = acc_o[i, d] / logsum[i]
                for i in T.Parallel(m_pad):
                    if i < groups:
                        TmpS[s, h * groups + i] = m_cur[i] * scale + T.log2(logsum[i])

    return main


def decode_merge_groups(heads, dim, max_batch, max_slots, span=DECODE_MERGE_SPAN, threads=DECODE_THREADS):
    """The merge's first level: one (request, query head, group of `span`
    slots) per CTA, folded in place into the group's first slot.

    The standard log-sum-exp combination of the partial pass's state over
    the group; the group's own state then stands where its first slot was,
    with the row sum folded into the log-sum-exp so the second level reads a
    normalised value and one scale, the same contract a single slot has. A
    CTA past its request's slots does nothing, so the grid can carry the
    batch's widest request.
    """

    @T.prim_func
    def main(
        TmpV: T.Tensor([max_slots, heads, dim], DTYPE),
        TmpS: T.Tensor([max_slots, heads], ACC),
        OIndptr: T.Tensor([max_batch + 1], "int32"),
        batch: T.int32,
        groups: T.int32,
    ):
        with T.Kernel(batch, heads, groups, threads=threads) as (r, hd, g):
            acc = T.alloc_fragment([dim], ACC)
            m_all = T.alloc_local([1], ACC)
            l_all = T.alloc_local([1], ACC)
            lo = OIndptr[r] + g * span
            hi = T.min(OIndptr[r] + (g + 1) * span, OIndptr[r + 1])
            if lo < hi:
                m_all[0] = -T.infinity(ACC)
                for k in T.serial(hi - lo):
                    m_all[0] = T.max(m_all[0], TmpS[lo + k, hd])
                l_all[0] = 0
                T.fill(acc, 0)
                for k in T.serial(hi - lo):
                    w = T.exp2(TmpS[lo + k, hd] - m_all[0])
                    l_all[0] += w
                    for d in T.Parallel(dim):
                        acc[d] += w * TmpV[lo + k, hd, d].astype(ACC)
            # The group folds into its own first slot, so the stores below
            # land on rows the reads above just took. Every thread reads the
            # whole of `TmpS` over the group and only its own lanes of
            # `TmpV`, so without a barrier here a thread that finished the
            # walk would overwrite state another is still reading. Both
            # `lo` and `hi` come from the block's own indices, so the barrier
            # sits outside the branch and every thread reaches it.
            T.sync_threads()
            if lo < hi:
                for d in T.Parallel(dim):
                    TmpV[lo, hd, d] = acc[d] / l_all[0]
                # One writer for the scalar: every thread holds the same
                # value, and a single store says so.
                if T.get_thread_binding() == 0:
                    TmpS[lo, hd] = m_all[0] + T.log2(l_all[0])

    return main


def decode_merge(heads, dim, max_rows, max_batch, max_slots, span=DECODE_MERGE_SPAN, threads=DECODE_THREADS):
    """The merge's second level: one (request, query head) per CTA over the
    request's group heads, `span` slots apart, into the output row.
    """

    @T.prim_func
    def main(
        TmpV: T.Tensor([max_slots, heads, dim], DTYPE),
        TmpS: T.Tensor([max_slots, heads], ACC),
        OIndptr: T.Tensor([max_batch + 1], "int32"),
        row_offset: T.int32,
        batch: T.int32,
        Output: T.Tensor([max_rows, heads, dim], DTYPE),
    ):
        with T.Kernel(batch, heads, threads=threads) as (r, hd):
            acc = T.alloc_fragment([dim], ACC)
            m_all = T.alloc_local([1], ACC)
            l_all = T.alloc_local([1], ACC)
            lo = OIndptr[r]
            hi = OIndptr[r + 1]
            n_groups = T.ceildiv(hi - lo, span)
            m_all[0] = -T.infinity(ACC)
            for k in T.serial(n_groups):
                m_all[0] = T.max(m_all[0], TmpS[lo + k * span, hd])
            l_all[0] = 0
            T.fill(acc, 0)
            for k in T.serial(n_groups):
                w = T.exp2(TmpS[lo + k * span, hd] - m_all[0])
                l_all[0] += w
                for d in T.Parallel(dim):
                    acc[d] += w * TmpV[lo + k * span, hd, d].astype(ACC)
            for d in T.Parallel(dim):
                Output[row_offset + r, hd, d] = acc[d] / l_all[0]

    return main
