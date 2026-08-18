//! The paged MLA latent KV cache — the pool-wide slab, its block tables, and
//! the free-list page allocator. Everything else about a slot's state stays in
//! [`super::buffers`].

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use half::bf16;
use pegainfer_kernels::tensor::DeviceContext;

use super::buffers::copy_rows;
use crate::config::K3_KV_LORA_RANK;
use crate::config::K3_QK_ROPE_HEAD_DIM;

/// Tokens per MLA KV page.
pub(crate) const K3_KV_PAGE_TOKENS: usize = 64;
/// One token's cached MLA latent: the post-norm kv latent and the shared
/// per-token rope half, exactly what `kv_b` expands K and V from. NoPE — no
/// rotary is ever applied, so the row is position-independent.
pub(crate) const K3_MLA_LATENT_ROW: usize = K3_KV_LORA_RANK + K3_QK_ROPE_HEAD_DIM;

/// The paged MLA latent cache: one slab per pool, shared by every MLA layer.
///
/// Layout is page-first, `[page][mla_layer][token][K3_MLA_LATENT_ROW]` bf16 —
/// all of a pool's MLA layers keep their 64-token slices at fixed offsets
/// inside one page, so a page is the transfer/interchange unit and one
/// allocation serves the whole model. Seen as rows of `K3_MLA_LATENT_ROW`,
/// layer `l`'s row for `(page, token)` is `page * mla_layers * 64 + l * 64 +
/// token`; the per-layer term is a constant row shift, which is how the write
/// path and the attention kernel address one layer's slice out of the shared
/// slab.
///
/// Pages come from a plain free-list pool: allocated when a slot's position
/// crosses a 64 boundary, freed together when the slot resets. No content
/// addressing and no reuse — this is the local pool only.
pub(crate) struct K3PagedKv {
    /// `[num_pages, mla_layers, K3_KV_PAGE_TOKENS, K3_MLA_LATENT_ROW]` bf16.
    pub(crate) slab: CudaSlice<bf16>,
    pub(crate) mla_layers: usize,
    pub(crate) num_pages: usize,
    pub(crate) max_pages_per_slot: usize,
    /// Device block table, `[rows, max_pages_per_slot]` i32, `-1` unmapped.
    pub(crate) table_dev: CudaSlice<i32>,
    /// Host mirror; the executor mutates this and re-uploads before a step.
    table_host: Vec<i32>,
    /// Free page ids; `pop` hands them out in ascending order from fresh.
    free: Vec<i32>,
}

impl K3PagedKv {
    pub(crate) fn new(
        ctx: &DeviceContext,
        rows: usize,
        max_ctx: usize,
        mla_layers: usize,
        num_pages: usize,
    ) -> Result<Self> {
        let max_pages_per_slot = max_ctx.div_ceil(K3_KV_PAGE_TOKENS);
        anyhow::ensure!(
            num_pages >= max_pages_per_slot,
            "K3 KV pool of {num_pages} pages cannot hold even one {max_ctx}-token slot"
        );
        let slab_len = num_pages * mla_layers * K3_KV_PAGE_TOKENS * K3_MLA_LATENT_ROW;
        Ok(Self {
            slab: ctx
                .stream
                .alloc_zeros::<bf16>(slab_len)
                .context("alloc K3 paged MLA latent slab")?,
            mla_layers,
            num_pages,
            max_pages_per_slot,
            table_dev: ctx
                .stream
                .clone_htod(&vec![-1i32; rows * max_pages_per_slot])
                .context("alloc K3 KV block table")?,
            table_host: vec![-1; rows * max_pages_per_slot],
            free: (0..num_pages as i32).rev().collect(),
        })
    }

    /// Rows (of `K3_MLA_LATENT_ROW`) one page spans across its layer slices.
    fn page_rows(&self) -> usize {
        self.mla_layers * K3_KV_PAGE_TOKENS
    }

    /// Elements from one page to the next — the attention kernel's page walk
    /// stride.
    pub(crate) fn page_stride(&self) -> usize {
        self.page_rows() * K3_MLA_LATENT_ROW
    }

    /// Make sure the page holding `position` is mapped for `row`, zeroing a
    /// freshly claimed page (the step's cache write is an indexed *add*, so
    /// the destination must be zero until the write).
    pub(crate) fn ensure_mapped(
        &mut self,
        ctx: &DeviceContext,
        row: usize,
        position: usize,
    ) -> Result<()> {
        let page_slot = position / K3_KV_PAGE_TOKENS;
        anyhow::ensure!(
            page_slot < self.max_pages_per_slot,
            "K3 KV row {row} position {position} exceeds its {} pages",
            self.max_pages_per_slot
        );
        let entry = row * self.max_pages_per_slot + page_slot;
        if self.table_host[entry] >= 0 {
            return Ok(());
        }
        let page = self
            .free
            .pop()
            .context("K3 KV pool is out of pages; raise kv_pages")?;
        self.table_host[entry] = page;
        let start = page as usize * self.page_rows() * K3_MLA_LATENT_ROW;
        let mut window = self
            .slab
            .slice_mut(start..start + self.page_rows() * K3_MLA_LATENT_ROW);
        ctx.stream
            .memset_zeros(&mut window)
            .context("zero a fresh K3 KV page")
    }

    /// Row index (in the `K3_MLA_LATENT_ROW`-wide view, before the per-layer
    /// shift) of `row`'s cache write at `position`. The page must be mapped.
    pub(crate) fn write_index(&self, row: usize, position: usize) -> Result<i32> {
        let page = self.table_host[row * self.max_pages_per_slot + position / K3_KV_PAGE_TOKENS];
        anyhow::ensure!(
            page >= 0,
            "K3 KV row {row} position {position} has no mapped page"
        );
        Ok(page * self.page_rows() as i32 + (position % K3_KV_PAGE_TOKENS) as i32)
    }

    /// Return every page `row` holds to the pool. Contents are not cleared —
    /// a page is zeroed when it is next claimed.
    pub(crate) fn release_row(&mut self, row: usize) {
        let base = row * self.max_pages_per_slot;
        for entry in &mut self.table_host[base..base + self.max_pages_per_slot] {
            if *entry >= 0 {
                self.free.push(*entry);
                *entry = -1;
            }
        }
    }

    /// Test hook: reverse the free list, so the next claims come from the
    /// opposite end of the pool. Physical page ids never enter the attention
    /// arithmetic — the kernel walks the block table by logical position — so
    /// any permutation must leave every logit bit-identical, and
    /// `tests/paged_kv.rs` holds the executor to exactly that.
    #[doc(hidden)]
    pub(crate) fn reverse_free_list(&mut self) {
        self.free.reverse();
    }

    /// Map fresh pages for `row` covering `tokens` positions and copy the
    /// source row's pages into them wholesale. The caller releases first.
    pub(crate) fn adopt_row(
        &mut self,
        ctx: &DeviceContext,
        source: &K3PagedKv,
        source_row: usize,
        row: usize,
        tokens: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            source.mla_layers == self.mla_layers
                && source.max_pages_per_slot == self.max_pages_per_slot,
            "K3 KV pools disagree on geometry"
        );
        let width = self.page_rows() * K3_MLA_LATENT_ROW;
        for page_slot in 0..tokens.div_ceil(K3_KV_PAGE_TOKENS) {
            self.ensure_mapped(ctx, row, page_slot * K3_KV_PAGE_TOKENS)?;
            let origin = source.table_host[source_row * source.max_pages_per_slot + page_slot];
            let target = self.table_host[row * self.max_pages_per_slot + page_slot];
            anyhow::ensure!(
                origin >= 0,
                "K3 KV adoption found no source page for slot {page_slot}"
            );
            copy_rows(
                ctx,
                &source.slab,
                origin as usize,
                &mut self.slab,
                target as usize,
                1,
                width,
            )?;
        }
        Ok(())
    }

    /// Mirror `source_row`'s block-table row into table rows `1..rows`.
    ///
    /// A prefill chunk runs one sequence as many batch rows — one row per
    /// token — and the batched attention kernel walks the block table by batch
    /// row, so every row of the chunk must see the same page chain. Causality
    /// comes from the per-row context length, not from the table.
    pub(crate) fn mirror_row_table(&mut self, source_row: usize, rows: usize) -> Result<()> {
        let width = self.max_pages_per_slot;
        ensure!(
            source_row < self.table_host.len() / width && rows * width <= self.table_host.len(),
            "K3 KV table mirror of {rows} rows is out of range"
        );
        let source = self.table_host[source_row * width..(source_row + 1) * width].to_vec();
        for row in (0..rows).filter(|row| *row != source_row) {
            self.table_host[row * width..(row + 1) * width].copy_from_slice(&source);
        }
        Ok(())
    }

    /// Upload the host block table. Cheap enough to do before every step, and
    /// outside graph capture like the rest of the step inputs.
    pub(crate) fn sync_table(&mut self, ctx: &DeviceContext) -> Result<()> {
        ctx.stream
            .memcpy_htod(&self.table_host, &mut self.table_dev)
            .map_err(|error| anyhow::anyhow!("K3 KV block-table feed failed: {error}"))
    }

    /// Write one step's latent rows into layer `mla_index`'s page slices.
    ///
    /// `kv_row` is the per-batch-row destination from [`Self::write_index`]
    /// (`-1` for rows the step does not own — the indexed write skips them).
    /// The destination is zero until this step writes it — a page is zeroed
    /// when claimed and every position is written once — so the indexed add is
    /// an exact indexed copy, taking the destination row from the device.
    pub(crate) fn append_latent(
        &mut self,
        ctx: &DeviceContext,
        mla_index: usize,
        rows: usize,
        kv_row: &CudaSlice<i32>,
        kv_norm: &CudaSlice<bf16>,
        rope: &CudaSlice<bf16>,
    ) -> Result<()> {
        ensure!(
            mla_index < self.mla_layers,
            "K3 KV layer index out of range"
        );
        ensure!(
            kv_norm.len() >= rows * K3_KV_LORA_RANK
                && rope.len() >= rows * K3_QK_ROPE_HEAD_DIM
                && kv_row.len() >= rows,
            "K3 KV append buffers too small for {rows} rows"
        );
        // Layer `mla_index`'s slice of every page starts `mla_index * 64` rows
        // into the page, so a base pointer shifted by that many rows makes the
        // layer-independent `write_index` address this layer's slice.
        let shift_rows = mla_index * K3_KV_PAGE_TOKENS;
        let out_rows = self.num_pages * self.page_rows() - shift_rows;
        let (slab_ptr, _slab_guard) = self.slab.device_ptr_mut(&ctx.stream);
        let base = slab_ptr + (shift_rows * K3_MLA_LATENT_ROW * size_of::<bf16>()) as u64;
        let (kv_row_ptr, _kv_row_guard) = kv_row.device_ptr(&ctx.stream);
        for (delta, width, column) in [
            (kv_norm, K3_KV_LORA_RANK, 0usize),
            (rope, K3_QK_ROPE_HEAD_DIM, K3_KV_LORA_RANK),
        ] {
            let (delta_ptr, _delta_guard) = delta.device_ptr(&ctx.stream);
            unsafe {
                pegainfer_kernels::ffi::scaled_add_rows_indexed_cuda(
                    delta_ptr as *const pegainfer_kernels::ffi::Half,
                    1.0,
                    kv_row_ptr as *const i32,
                    base as *mut pegainfer_kernels::ffi::Half,
                    K3_MLA_LATENT_ROW as i32,
                    column as i32,
                    width as i32,
                    rows as i32,
                    out_rows as i32,
                    pegainfer_kernels::tensor::active_cu_stream(ctx),
                )
            }
            .result()
            .map_err(|error| anyhow::anyhow!("K3 KV latent append failed: {error}"))?;
        }
        Ok(())
    }
}
