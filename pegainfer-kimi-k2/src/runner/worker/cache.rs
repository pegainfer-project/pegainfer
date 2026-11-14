use super::load::*;
use super::runtime::*;
use super::*;

impl KimiOneTokenForwardCache {
    pub(super) fn from_gpu_weights(
        ctx: &KimiRankGpuContext,
        weights: &KimiRankGpuWeights,
        names: &KimiRankWeightNames,
    ) -> Result<Self> {
        ensure!(
            weights.rank == names.rank,
            "Kimi forward cache rank mismatch: weights={}, names={}",
            weights.rank,
            names.rank
        );
        ensure!(
            names.layers.len() == KIMI_K2_LAYERS,
            "Kimi forward cache needs {} layers, got {}",
            KIMI_K2_LAYERS,
            names.layers.len()
        );

        let vocab_rows = names.plan.vocab_range.len();
        let token_embedding = GpuTensor::from_device_matrix_rows(
            raw_tensor(weights, &names.top.token_embedding)?.copy_bf16_matrix(
                ctx,
                vocab_rows,
                KIMI_K2_HIDDEN,
                "token_embedding",
            )?,
        )?;
        let final_norm = NormWeight::from_device_vec(
            raw_tensor(weights, &names.top.final_norm)?.copy_bf16_vec(
                ctx,
                KIMI_K2_HIDDEN,
                "final_norm",
            )?,
        )?;
        let lm_head = GpuTensor::from_device_matrix_rows(
            raw_tensor(weights, &names.top.lm_head)?.copy_bf16_matrix(
                ctx,
                vocab_rows,
                KIMI_K2_HIDDEN,
                "lm_head",
            )?,
        )?;
        let layers = names
            .layers
            .iter()
            .map(|layer| load_layer_forward_cache(ctx, weights, layer))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            vocab_start: names.plan.vocab_range.start,
            vocab_rows,
            token_embedding,
            final_norm,
            lm_head,
            layers,
        })
    }
}

impl KimiWorkerKvPool {
    pub(super) fn new(ctx: &DeviceContext, num_layers: usize, pool_pages: usize) -> Result<Self> {
        ensure!(num_layers > 0, "Kimi KV pool needs layers");
        ensure!(pool_pages > 0, "Kimi KV pool needs pages");
        // batch_size is irrelevant for buffer sizing (required_*_len depends
        // only on max_pages × page geometry); per-bucket arenas carry their
        // own layouts with the real batch dimension.
        let sizing = KimiMlaPagedKvLayout::separate_contiguous(pool_pages, KIMI_KV_PAGE_SIZE, 1);
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(KimiWorkerMlaLayerCache {
                ckv_cache: ctx
                    .stream
                    .alloc_zeros::<half::bf16>(sizing.required_ckv_len()?)?,
                kpe_cache: ctx
                    .stream
                    .alloc_zeros::<half::bf16>(sizing.required_kpe_len()?)?,
            });
        }
        Ok(Self { layers })
    }

    pub(super) fn layer_mut(&mut self, layer_idx: usize) -> Result<&mut KimiWorkerMlaLayerCache> {
        self.layers
            .get_mut(layer_idx)
            .ok_or_else(|| anyhow::anyhow!("Kimi KV pool layer cache {layer_idx} out of range"))
    }
}

impl KimiWorkerDecodeArena {
    pub(super) fn new(
        ctx: &DeviceContext,
        batch_size: usize,
        pool_pages: usize,
        vocab_rows: usize,
        dims: &crate::config::KimiLocalDims,
    ) -> Result<Self> {
        ensure!(
            batch_size > 0,
            "Kimi decode arena batch_size must be positive"
        );
        ensure!(pool_pages > 0, "Kimi decode arena needs a KV pool");
        let page_size = KIMI_KV_PAGE_SIZE;
        // Worst case per step: every active slot's pages (disjoint pool
        // pages, so ≤ pool_pages total) plus one padding entry per idle slot.
        let page_table_capacity = pool_pages
            .checked_add(batch_size)
            .ok_or_else(|| anyhow::anyhow!("Kimi decode arena page table capacity overflow"))?;
        let append_capacity = KIMI_MAX_REQUEST_TOKENS.max(batch_size);
        let layout = KimiMlaPagedKvLayout::separate_contiguous(pool_pages, page_size, batch_size);
        // `batch_size * (rows / batch_size)` rows, the same count in every
        // bucket.
        let kv_tiles = (KIMI_DECODE_KV_TILE_ROWS / batch_size).max(1);
        let kv_rows = batch_size * kv_tiles;
        // Placeholder table (page 0 everywhere): configure_* always rebuilds
        // it before any forward touches the pool.
        let placeholder = KimiKvStepPages::new(vec![vec![0i32]; batch_size], 0);
        let all_slots = (0..batch_size).collect::<Vec<_>>();
        let (page_indices, page_indptr, last_page_len) = build_slot_page_table(
            batch_size,
            page_size,
            &placeholder,
            &all_slots,
            &vec![1usize; batch_size],
        )?;
        let mut page_indices_padded = vec![0i32; page_table_capacity];
        page_indices_padded[..page_indices.len()].copy_from_slice(&page_indices);
        let batch_indices_padded = {
            let mut padded = vec![0i32; append_capacity];
            for (idx, value) in padded.iter_mut().enumerate().take(batch_size) {
                *value = idx as i32;
            }
            padded
        };
        let positions_padded = vec![0i32; append_capacity];
        // Idle plan: no row carries KV until the first configure_* call.
        let idle_plan = plan_decode_kv_tiles(batch_size, kv_tiles, &[], &[])?;
        let token_ids = vec![0u32; batch_size];
        let (cos_host, sin_host) = build_yarn_rope_cache(KIMI_MAX_REQUEST_TOKENS);

        Ok(Self {
            batch_size,
            page_size,
            pool_pages,
            page_table_capacity,
            append_capacity,
            layout,
            kv_tiles,
            kv_rows,
            page_indices_d: ctx.stream.clone_htod(&page_indices_padded)?,
            page_indptr_d: ctx.stream.clone_htod(&page_indptr)?,
            last_page_len_d: ctx.stream.clone_htod(&last_page_len)?,
            batch_indices_d: ctx.stream.clone_htod(&batch_indices_padded)?,
            positions_d: ctx.stream.clone_htod(&positions_padded)?,
            row_slots_d: ctx.stream.clone_htod(&idle_plan.row_slots)?,
            row_tiles_d: ctx.stream.clone_htod(&idle_plan.row_tiles)?,
            tile_tokens_d: ctx.stream.clone_htod(&[idle_plan.tile_tokens])?,
            slot_indptr_d: ctx.stream.clone_htod(&idle_plan.slot_indptr)?,
            row_valid_d: ctx.stream.clone_htod(&idle_plan.row_valid)?,
            partial_ckv_d: ctx
                .stream
                .alloc_zeros::<half::bf16>(kv_rows * dims.local_heads * KIMI_K2_MLA_KV_LORA_RANK)?,
            partial_lse_d: ctx.stream.alloc_zeros::<f32>(kv_rows * dims.local_heads)?,
            token_ids_d: ctx.stream.clone_htod(&token_ids)?,
            cos_d: ctx.stream.clone_htod(&cos_host)?,
            sin_d: ctx.stream.clone_htod(&sin_host)?,
            scratch: KimiWorkerDecodeScratch::new(ctx, batch_size, dims)?,
            logits: HiddenStates::zeros(ctx, vocab_rows, batch_size)?,
            graph: CudaGraphState::new(),
        })
    }

    fn upload_page_table(
        &mut self,
        ctx: &DeviceContext,
        page_indices: &[i32],
        page_indptr: &[i32],
        last_page_len: &[i32],
    ) -> Result<()> {
        ensure!(
            page_indices.len() <= self.page_table_capacity,
            "Kimi page table length {} exceeds arena capacity {}",
            page_indices.len(),
            self.page_table_capacity
        );
        ensure!(
            page_indptr.len() == self.batch_size + 1 && last_page_len.len() == self.batch_size,
            "Kimi page table metadata must cover all {} slots: indptr={}, last_page_len={}",
            self.batch_size,
            page_indptr.len(),
            last_page_len.len()
        );
        {
            let mut page_indices_d = self.page_indices_d.slice_mut(0..page_indices.len());
            ctx.stream.memcpy_htod(page_indices, &mut page_indices_d)?;
        }
        ctx.stream
            .memcpy_htod(page_indptr, &mut self.page_indptr_d)?;
        ctx.stream
            .memcpy_htod(last_page_len, &mut self.last_page_len_d)?;
        Ok(())
    }

    /// Configure the page table and append positions for one prefill step.
    /// `cached_tokens > 0` is a prefix-cache hit: the page table covers the
    /// cached prefix plus the `seq_len`-token suffix, and the suffix appends
    /// at absolute positions `cached_tokens..cached_tokens + seq_len`.
    /// Returns the slot's offset into the uploaded page-index array so the
    /// forward pass can gather the cached prefix from pool pages.
    pub(super) fn configure_slot_prefill(
        &mut self,
        ctx: &DeviceContext,
        slot: usize,
        seq_len: usize,
        cached_tokens: usize,
        kv_pages: &KimiKvStepPages,
    ) -> Result<usize> {
        ensure!(seq_len > 0, "Kimi prefill KV write requires tokens");
        ensure!(
            slot < self.batch_size,
            "Kimi prefill slot {slot} exceeds batch_size {}",
            self.batch_size
        );
        let kv_tokens = cached_tokens + seq_len;
        ensure!(
            kv_tokens <= KIMI_MAX_REQUEST_TOKENS,
            "Kimi prefill kv tokens {kv_tokens} exceed per-request KV capacity {KIMI_MAX_REQUEST_TOKENS}"
        );
        ensure!(
            kv_pages.rows() == 1,
            "Kimi prefill expects one KV page row, got {}",
            kv_pages.rows()
        );
        self.ensure_pool_page(kv_pages.padding_page)?;
        let (page_indices, page_indptr, last_page_len) = build_slot_page_table(
            self.batch_size,
            self.page_size,
            kv_pages,
            &[slot],
            &[kv_tokens],
        )?;
        let slot_pages_start = usize::try_from(page_indptr[slot])
            .map_err(|_| anyhow::anyhow!("Kimi prefill slot page offset must be non-negative"))?;
        self.ensure_pool_pages(&page_indices)?;
        self.upload_page_table(ctx, &page_indices, &page_indptr, &last_page_len)?;
        let batch_indices = vec![slot as i32; seq_len];
        let positions = (cached_tokens as i32..kv_tokens as i32).collect::<Vec<_>>();
        {
            let mut batch_indices_d = self.batch_indices_d.slice_mut(0..seq_len);
            ctx.stream
                .memcpy_htod(&batch_indices, &mut batch_indices_d)?;
        }
        {
            let mut positions_d = self.positions_d.slice_mut(0..seq_len);
            ctx.stream.memcpy_htod(&positions, &mut positions_d)?;
        }
        Ok(slot_pages_start)
    }

    pub(super) fn configure_batch_decode(
        &mut self,
        ctx: &DeviceContext,
        slots: &[usize],
        append_positions: &[usize],
        kv_pages: &KimiKvStepPages,
    ) -> Result<()> {
        ensure!(!slots.is_empty(), "Kimi batch decode requires active slots");
        ensure!(
            slots.len() == append_positions.len(),
            "Kimi batch decode slots/positions mismatch: slots={}, positions={}",
            slots.len(),
            append_positions.len()
        );
        ensure!(
            slots.len() <= self.batch_size,
            "Kimi batch decode active slots {} exceeds batch_size {}",
            slots.len(),
            self.batch_size
        );
        ensure!(
            kv_pages.rows() == slots.len(),
            "Kimi batch decode KV page rows {} must match active slots {}",
            kv_pages.rows(),
            slots.len()
        );
        self.ensure_pool_page(kv_pages.padding_page)?;
        let mut batch_indices = vec![0i32; self.batch_size];
        let mut row_positions = vec![0i32; self.batch_size];
        let mut occupied_slots = vec![false; self.batch_size];
        // KV tokens each active row holds after this step's append.
        let mut kv_tokens = Vec::with_capacity(slots.len());

        for (row, (&slot, &position)) in slots.iter().zip(append_positions.iter()).enumerate() {
            ensure!(
                slot < self.batch_size,
                "Kimi batch decode slot {slot} exceeds batch_size {}",
                self.batch_size
            );
            ensure!(
                !occupied_slots[slot],
                "Kimi batch decode slot {slot} appears more than once"
            );
            ensure!(
                position < KIMI_MAX_REQUEST_TOKENS,
                "Kimi decode append_position {position} exceeds per-request KV capacity {KIMI_MAX_REQUEST_TOKENS}"
            );
            occupied_slots[slot] = true;
            batch_indices[row] = slot as i32;
            row_positions[row] = position as i32;
            kv_tokens.push(position + 1);
        }
        // CUDA-graph padding rows reference idle slots, whose page lists are
        // the pool padding page: their dummy append/read lands on a page no
        // live request owns.
        let free_slots = occupied_slots
            .iter()
            .enumerate()
            .filter_map(|(slot, occupied)| (!occupied).then_some(slot))
            .collect::<Vec<_>>();
        for row in slots.len()..self.batch_size {
            batch_indices[row] = free_slots[row - slots.len()] as i32;
        }

        let (page_indices, page_indptr, last_page_len) =
            build_slot_page_table(self.batch_size, self.page_size, kv_pages, slots, &kv_tokens)?;
        self.ensure_pool_pages(&page_indices)?;
        self.upload_page_table(ctx, &page_indices, &page_indptr, &last_page_len)?;
        {
            let mut batch_indices_d = self.batch_indices_d.slice_mut(0..self.batch_size);
            ctx.stream
                .memcpy_htod(&batch_indices, &mut batch_indices_d)?;
        }
        {
            let mut positions_d = self.positions_d.slice_mut(0..self.batch_size);
            ctx.stream.memcpy_htod(&row_positions, &mut positions_d)?;
        }
        let plan = plan_decode_kv_tiles(self.batch_size, self.kv_tiles, slots, &kv_tokens)?;
        ctx.stream
            .memcpy_htod(&plan.row_slots, &mut self.row_slots_d)?;
        ctx.stream
            .memcpy_htod(&plan.slot_indptr, &mut self.slot_indptr_d)?;
        ctx.stream
            .memcpy_htod(&plan.row_valid, &mut self.row_valid_d)?;
        ctx.stream
            .memcpy_htod(&[plan.tile_tokens], &mut self.tile_tokens_d)?;
        Ok(())
    }

    fn ensure_pool_page(&self, page: i32) -> Result<()> {
        ensure!(
            page >= 0 && (page as usize) < self.pool_pages,
            "Kimi KV page {page} outside pool of {} pages",
            self.pool_pages
        );
        Ok(())
    }

    fn ensure_pool_pages(&self, pages: &[i32]) -> Result<()> {
        for &page in pages {
            self.ensure_pool_page(page)?;
        }
        Ok(())
    }

    pub(super) fn upload_batch_tokens(
        &mut self,
        ctx: &DeviceContext,
        token_ids: &[u32],
    ) -> Result<()> {
        ensure!(
            token_ids.len() <= self.batch_size,
            "Kimi batch token upload length {} exceeds batch_size {}",
            token_ids.len(),
            self.batch_size
        );
        let mut tokens = vec![0u32; self.batch_size];
        tokens[..token_ids.len()].copy_from_slice(token_ids);
        ctx.stream.memcpy_htod(&tokens, &mut self.token_ids_d)?;
        Ok(())
    }

    pub(super) fn append_prefill_layer_kv(
        &mut self,
        ctx: &DeviceContext,
        kv_pool: &mut KimiWorkerKvPool,
        layer_idx: usize,
        compressed_normed: &GpuTensor<KIMI_K2_MLA_KV_LORA_RANK>,
        append_kpe: &GpuTensor<KIMI_K2_MLA_ROPE_DIM>,
    ) -> Result<()> {
        ensure!(
            compressed_normed.seq_len <= self.append_capacity,
            "Kimi prefill append seq_len {} exceeds metadata capacity {}",
            compressed_normed.seq_len,
            self.append_capacity
        );
        let layer_cache = kv_pool.layer_mut(layer_idx)?;
        kimi_mla_paged_kv_append(
            ctx,
            &mut layer_cache.ckv_cache,
            &mut layer_cache.kpe_cache,
            self.layout,
            &self.page_indices_d,
            &self.page_indptr_d,
            &self.last_page_len_d,
            compressed_normed,
            append_kpe,
            &self.batch_indices_d,
            &self.positions_d,
        )
    }
}

/// Build the slot-indexed page table (FlashInfer CSR over all `batch_size`
/// slots) from a step's row-major page CSR. `slots[i]` is the slot that
/// `kv_pages` row `i` belongs to; that slot holds `kv_tokens[i]` KV tokens
/// after this step's append. Slots without a row ride the padding page.
pub(super) fn build_slot_page_table(
    batch_size: usize,
    page_size: usize,
    kv_pages: &KimiKvStepPages,
    slots: &[usize],
    kv_tokens: &[usize],
) -> Result<(Vec<i32>, Vec<i32>, Vec<i32>)> {
    ensure!(page_size > 0, "Kimi KV page_size must be positive");
    ensure!(
        slots.len() == kv_tokens.len() && slots.len() == kv_pages.rows(),
        "Kimi page table row mismatch: slots={}, kv_tokens={}, page rows={}",
        slots.len(),
        kv_tokens.len(),
        kv_pages.rows()
    );
    let padding_row = [kv_pages.padding_page];
    let mut slot_rows: Vec<&[i32]> = vec![&padding_row; batch_size];
    let mut slot_tokens = vec![1usize; batch_size];
    for (row, (&slot, &tokens)) in slots.iter().zip(kv_tokens.iter()).enumerate() {
        ensure!(
            slot < batch_size,
            "Kimi page table slot {slot} exceeds batch_size {batch_size}"
        );
        slot_rows[slot] = kv_pages.row(row)?;
        slot_tokens[slot] = tokens;
    }

    let mut page_indices = Vec::with_capacity(kv_pages.pages.len() + batch_size);
    let mut page_indptr = Vec::with_capacity(batch_size + 1);
    let mut last_page_len = Vec::with_capacity(batch_size);
    page_indptr.push(0i32);
    for slot in 0..batch_size {
        let tokens = slot_tokens[slot];
        ensure!(tokens > 0, "Kimi page table slot {slot} has zero KV tokens");
        let needed = tokens.div_ceil(page_size);
        // Exact match required: the scheduler's block accounting and the
        // kernel's view of the sequence must agree, or KV silently lands on
        // (or reads from) pages the request does not own.
        ensure!(
            slot_rows[slot].len() == needed,
            "Kimi page table slot {slot} has {} pages for {} KV tokens, needs {needed}",
            slot_rows[slot].len(),
            tokens
        );
        page_indices.extend_from_slice(slot_rows[slot]);
        page_indptr.push(page_indices.len() as i32);
        last_page_len.push((((tokens - 1) % page_size) + 1) as i32);
    }
    Ok((page_indices, page_indptr, last_page_len))
}

/// One step's split-KV layout: row `row * tiles + tile` is tile `tile` of
/// decode row `row`, `slot_indptr` keeps a request's rows contiguous for the
/// merge, and rows it does not fill stay masked so the grid stays bucket-sized.
pub(super) struct KimiDecodeTilePlan {
    pub(super) tile_tokens: i32,
    pub(super) slot_indptr: Vec<i32>,
    pub(super) row_slots: Vec<i32>,
    pub(super) row_tiles: Vec<i32>,
    pub(super) row_valid: Vec<u8>,
}

pub(super) fn plan_decode_kv_tiles(
    batch_size: usize,
    tiles: usize,
    slots: &[usize],
    kv_tokens: &[usize],
) -> Result<KimiDecodeTilePlan> {
    ensure!(tiles > 0, "Kimi decode tile plan needs at least one tile");
    ensure!(
        slots.len() == kv_tokens.len(),
        "Kimi decode tile plan slots/KV mismatch: slots={}, KV={}",
        slots.len(),
        kv_tokens.len()
    );
    ensure!(
        slots.len() <= batch_size,
        "Kimi decode tile plan rows {} exceed batch size {batch_size}",
        slots.len()
    );
    let longest = kv_tokens.iter().copied().max().unwrap_or(0);
    let tile_tokens = longest.div_ceil(tiles).max(KIMI_DECODE_MIN_KV_TILE_TOKENS);
    let rows = batch_size * tiles;
    // Tile index depends on the row index alone, so the arena uploads it once.
    let row_tiles = (0..rows)
        .map(|row| (row % tiles) as i32)
        .collect::<Vec<_>>();
    let mut row_slots = vec![0i32; rows];
    let mut row_valid = vec![0u8; rows];
    let mut slot_indptr = Vec::with_capacity(batch_size + 1);
    slot_indptr.push(0i32);
    let mut used_rows = 0i32;
    for (row, (&slot, &kv)) in slots.iter().zip(kv_tokens.iter()).enumerate() {
        ensure!(
            slot < batch_size,
            "Kimi decode tile plan slot {slot} exceeds batch size {batch_size}"
        );
        let base = row * tiles;
        let used = kv.div_ceil(tile_tokens).min(tiles);
        for tile in 0..tiles {
            row_slots[base + tile] = slot as i32;
            row_valid[base + tile] = u8::from(tile < used);
        }
        used_rows += used as i32;
        slot_indptr.push(used_rows);
    }
    // Idle rows still launch (the grid is bucket-sized) but own no tile.
    for _ in slots.len()..batch_size {
        slot_indptr.push(used_rows);
    }
    Ok(KimiDecodeTilePlan {
        tile_tokens: tile_tokens as i32,
        slot_indptr,
        row_slots,
        row_tiles,
        row_valid,
    })
}

impl KimiWorkerDecodeScratch {
    fn new(
        ctx: &DeviceContext,
        batch_size: usize,
        dims: &crate::config::KimiLocalDims,
    ) -> Result<Self> {
        let marlin_block_size = kimi_marlin_block_size(batch_size);
        let marlin_route_workspace =
            KimiMarlinRouteWorkspace::new(ctx, batch_size, marlin_block_size)?;
        let marlin_workspace = KimiMarlinWna16Workspace::new(
            ctx,
            marlin_route_workspace.max_m_blocks,
            KIMI_K2_HIDDEN,
            marlin_block_size,
        )?;
        Ok(Self {
            mla: crate::typed_scratch::MlaDecodeScratch::new(ctx, batch_size, dims)?,
            dense_mlp: crate::typed_scratch::DenseMlpDecodeScratch::new(ctx, batch_size, dims)?,
            shared_expert: crate::typed_scratch::SharedExpertDecodeScratch::new(
                ctx, batch_size, dims,
            )?,
            router: crate::typed_scratch::RouterScratch::new(ctx, batch_size)?,
            marlin: crate::typed_scratch::MarlinExpertScratch::new(ctx, batch_size)?,
            marlin_route_workspace,
            marlin_workspace,
            comm: crate::typed_scratch::CommScratch::new(ctx, batch_size)?,
            sampling: crate::typed_scratch::SamplingScratch::new(ctx, batch_size)?,
        })
    }
}
