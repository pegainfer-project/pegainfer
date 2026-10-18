//! Logical KV block pool, moved here from `pegainfer_kv_cache::pool` as a
//! strangler migration: `pegainfer-kv-store` is becoming the single owner of
//! the logical-layer contract, so new wiring imports it from this crate and
//! talks to kvbm-logical directly instead of going through
//! `pegainfer-kv-cache`. The original stays in place for its existing
//! consumers until they migrate. Two parts were dropped in the move because
//! the store never calls them: the model-forward `KvView` constructors
//! (`prefill_view`/`decode_view`/`speculative_view`) and the opt-in KV-event
//! feed (`with_events`); Reservation/Probe do not depend on either.

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicI64;
use std::sync::atomic::Ordering;

use dynamo_kv_hashing::compute_salt_hash;
use kvbm_logical::SequenceHash;
use kvbm_logical::blocks::ImmutableBlock;
use kvbm_logical::blocks::MutableBlock;
use kvbm_logical::integrations::DecodeOutcome;
use kvbm_logical::integrations::SchedulableSequence;
use kvbm_logical::integrations::ScheduleError;
use kvbm_logical::manager::BlockManager;
use kvbm_logical::pools::BlockDuplicationPolicy;
use kvbm_logical::registry::BlockRegistry;
use pegainfer_frontend::engine::KvPrefix;

/// Logical KV block pool: a `BlockManager` plus the reserved padding block.
///
/// Owns no GPU memory — the physical layout (full-attention `KvBuffer`,
/// MLA dual ckv/kpe buffers, ...) lives with the consumer and is indexed
/// by the block IDs this pool hands out.
pub struct BlockPool {
    block_manager: BlockManager<()>,
    block_size: usize,
    padding_block_id: usize,
    /// Blocks promised to admitted requests but not yet drawn
    /// (Σ admitted `lifetime_blocks - resident_blocks`); maintained by
    /// [`entitlement::EntitledSeq`], consumed by the
    /// [`Self::reserve_loaded_blocks`] floor.
    entitled: Arc<AtomicI64>,
    /// Serializes every move that trades availability against the floor:
    /// opportunistic reserves ([`Self::reserve_loaded_blocks`]), entitlement
    /// declarations ([`RequestKv::try_admit`]), and admitted draws/returns
    /// (`EntitledSeq::with_draw` — a returned block and its entitlement
    /// refund must become visible together, or a reserve lands in between
    /// and over-commits the pool).
    reserve_gate: Arc<Mutex<()>>,
}

impl BlockPool {
    pub fn new(block_size: usize, num_blocks: usize) -> Self {
        // kvbm validates the same two things three levels down; state them at
        // the argument boundary instead.
        assert!(num_blocks >= 2, "need at least 2 blocks (1 for padding)");
        assert!(
            block_size.is_power_of_two() && block_size <= 1024,
            "block_size must be a power of two in 2..=1024, got {block_size}"
        );

        let block_manager = BlockManager::builder()
            .block_count(num_blocks)
            .block_size(block_size)
            .registry(BlockRegistry::builder().build())
            .duplication_policy(BlockDuplicationPolicy::Allow)
            .with_lru_backend()
            .build()
            .expect("BlockManager build cannot fail past the asserts above");

        // Reserve block 0 as CUDA Graph padding slot.
        let padding_blocks = block_manager
            .allocate_blocks(1)
            .expect("a fresh pool always has a block for padding");
        let padding_block_id = padding_blocks[0].block_id();
        let padding_complete = padding_blocks
            .into_iter()
            .next()
            .unwrap()
            .stage(SequenceHash::default(), block_size)
            .expect("staging an empty hash on a fresh block cannot fail");
        // Register so it stays alive (ImmutableBlock RAII keeps it out of the
        // free pool), then leak — padding lives for the lifetime of the engine.
        std::mem::forget(block_manager.register_block(padding_complete));

        Self {
            block_manager,
            block_size,
            padding_block_id,
            entitled: Arc::new(AtomicI64::new(0)),
            reserve_gate: Arc::new(Mutex::new(())),
        }
    }

    pub(crate) fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn available_blocks(&self) -> usize {
        self.block_manager.available_blocks()
    }

    /// Blocks promised to admitted requests ([`RequestKv::try_admit`]) but
    /// not yet drawn. [`Self::reserve_loaded_blocks`] keeps this many out of
    /// opportunistic reach so an admitted request's page-crossing draw can
    /// never fail.
    pub fn entitled_blocks(&self) -> usize {
        self.entitled.load(Ordering::Acquire).max(0) as usize
    }

    pub fn total_blocks(&self) -> usize {
        self.block_manager.total_blocks()
    }

    pub fn padding_block_id(&self) -> i32 {
        self.padding_block_id as i32
    }

    /// Evict every cached-but-unused block from the GPU prefix cache (drain the
    /// inactive pool). In-use blocks are untouched. A cold-cache flush — and,
    /// for the offload path, the way to force a prefix out of HBM so a
    /// subsequent request must restore it from the CPU tier.
    pub fn evict_inactive(&self) {
        self.block_manager.evict_inactive();
    }

    /// `lora_name` scopes the prefix cache: blocks registered under one
    /// adapter (or the base model, `None`) never match a request running
    /// under a different adapter — the name is folded into the block-hash
    /// chain as a salt, so K/V computed with different weights can't be
    /// silently reused.
    pub fn new_request(
        &self,
        prompt_tokens: Vec<u32>,
        max_output_tokens: usize,
        lora_name: Option<&str>,
    ) -> RequestKv {
        self.new_request_with_cache_salt(prompt_tokens, max_output_tokens, None, lora_name)
    }

    /// Construct a request whose prefix-cache identity is additionally scoped
    /// by `cache_salt`. Use this when identical token blocks can produce
    /// different KV bytes because of state outside that block. `lora_name`
    /// remains a separate weight identity; callers must not overload it with
    /// request-local cache semantics.
    pub fn new_request_with_cache_salt(
        &self,
        prompt_tokens: Vec<u32>,
        max_output_tokens: usize,
        cache_salt: Option<&str>,
        lora_name: Option<&str>,
    ) -> RequestKv {
        let salt_hash = compute_salt_hash(cache_salt, lora_name)
            .expect("compute_salt_hash is infallible for string cache/lora identities");
        let lifetime_blocks = (prompt_tokens.len() + max_output_tokens).div_ceil(self.block_size);
        let seq = SchedulableSequence::new(
            prompt_tokens,
            max_output_tokens,
            self.block_size as u32,
            None,
            Some(salt_hash),
        );
        RequestKv {
            inner: entitlement::EntitledSeq::new(
                seq,
                Arc::clone(&self.entitled),
                Arc::clone(&self.reserve_gate),
                lifetime_blocks,
            ),
        }
    }

    // ── KV-offload prefetch (CPU-tier load before prefill) ─────────────

    /// [`Self::probe_prefix`] with the same additional cache scope accepted by
    /// [`Self::new_request_with_cache_salt`]. The producer request and every
    /// restore probe must derive the identical salt.
    pub(crate) fn probe_prefix_with_cache_salt(
        &self,
        prompt_tokens: Vec<u32>,
        cache_salt: Option<&str>,
        lora_name: Option<&str>,
    ) -> PrefixProbe {
        let num_input = prompt_tokens.len();
        let rkv = self.new_request_with_cache_salt(prompt_tokens, 0, cache_salt, lora_name);
        let seq_hashes = rkv.inner.seq().inner().sequence().all_sequence_hashes();
        // match_and_add_prefix leaves >=1 prompt token uncached, so a request
        // can reuse at most this many leading blocks — the CPU load must not
        // exceed it, or the trailing loaded block would never be re-matched.
        let cacheable = num_input.saturating_sub(1) / self.block_size;
        let gpu_guard = self.block_manager.match_blocks(&seq_hashes);
        let gpu_hit = gpu_guard.len();
        PrefixProbe {
            seq_hashes,
            gpu_hit,
            cacheable,
            held: gpu_guard,
        }
    }

    /// Reserve `count` mutable blocks as the GPU destinations for a CPU→GPU
    /// load. Returns `None` under block pressure (the caller then skips the
    /// prefetch and prefills from scratch, or retries under its deadline).
    /// The reservation's [`LoadReservation::page_ids`] feed the connector's
    /// load; on completion hand it to
    /// [`commit_loaded_blocks`](Self::commit_loaded_blocks).
    ///
    /// Opportunistic by contract: the reserve stays above the entitled floor
    /// ([`Self::entitled_blocks`]) — blocks promised to admitted requests are
    /// out of reach even while they sit in the free/inactive pools.
    pub fn reserve_loaded_blocks(&self, count: usize) -> Option<LoadReservation> {
        let _gate = self.reserve_gate.lock().expect("reserve gate poisoned");
        if self.available_blocks() < count + self.entitled_blocks() {
            return None;
        }
        let blocks = self.block_manager.allocate_blocks(count)?;
        Some(LoadReservation { blocks })
    }

    /// Stage + register the freshly-loaded blocks under the probe's
    /// continuation hashes (`seq_hashes[gpu_hit .. gpu_hit + reserved]`) and
    /// fold them into the probe's held set, so a following
    /// `new_request().match_and_add_prefix()` reuses the full GPU+CPU prefix.
    ///
    /// The probe keeps holding every registered block until the request
    /// prefills, closing the eviction window between registration and re-match.
    pub(crate) fn commit_loaded_blocks(
        &self,
        probe: &mut PrefixProbe,
        reservation: LoadReservation,
    ) {
        let start = probe.gpu_hit;
        for (i, block) in reservation.blocks.into_iter().enumerate() {
            let hash = probe.seq_hashes[start + i];
            let complete = block
                .stage(hash, self.block_size)
                .expect("loaded block stage: block_size invariant violated");
            probe.held.push(self.block_manager.register_block(complete));
        }
    }
}

/// A prompt's prefix resolved against the GPU cache, ready to drive a CPU-tier
/// prefetch. Holds every GPU-hit (and, after commit, CPU-loaded) block so they
/// can't be evicted while the load is in flight and before the request prefills.
pub struct PrefixProbe {
    /// Content hashes of every complete prompt block, in order (native form).
    seq_hashes: Vec<SequenceHash>,
    /// Length of the contiguous GPU-resident prefix.
    gpu_hit: usize,
    /// Reuse cap: blocks past this are never matched (the final chunk forwards).
    cacheable: usize,
    /// Strong refs keeping matched/loaded blocks resident until prefill.
    held: Vec<ImmutableBlock<()>>,
}

impl PrefixProbe {
    /// Blocks already resident in GPU HBM (the existing prefix-cache hit).
    #[cfg(test)]
    fn gpu_hit_blocks(&self) -> usize {
        self.gpu_hit
    }

    /// Total blocks this probe holds: the GPU-hit prefix plus any committed from
    /// a CPU-tier load. They are already out of the free pool and become the
    /// request's cached prefix at prefill, so admission credits them against the
    /// request's block need (avoiding a double-count against `available_blocks`).
    pub(crate) fn held_blocks(&self) -> usize {
        self.held.len()
    }

    /// Content hashes to query the CPU tier with: the blocks past the GPU hit,
    /// capped at the reuse boundary. Empty when the GPU hit already covers
    /// every reusable block (nothing to load — prefill normally).
    pub(crate) fn cpu_query_hashes(&self) -> Vec<Vec<u8>> {
        if self.gpu_hit >= self.cacheable {
            return Vec::new();
        }
        self.seq_hashes[self.gpu_hit..self.cacheable]
            .iter()
            .map(|h| sequence_hash_bytes(h).to_vec())
            .collect()
    }

    /// Number of blocks [`Self::cpu_query_hashes`] covers, without
    /// materializing the hash bytes. Callers substituting their own key
    /// scheme (vLLM-compat P/D) slice their chain to exactly this window.
    pub fn cpu_query_window(&self) -> usize {
        self.cacheable.saturating_sub(self.gpu_hit)
    }

    /// Drop the anti-eviction pins on every held block past the first
    /// `blocks` (they fall to the inactive — evictable, still matchable —
    /// pool). `held` is in prefix order, so the surviving pins cover exactly
    /// the leading `blocks` blocks of the prefix.
    pub(crate) fn truncate_held(&mut self, blocks: usize) {
        self.held.truncate(blocks);
    }

    /// Reuse cap in blocks; see [`Self::include_final_block`].
    pub(crate) fn cacheable_blocks(&self) -> usize {
        self.cacheable
    }

    /// Physical page ids of the held blocks, in prefix order. The native
    /// admission reads the last one as its boundary copy-on-restore source;
    /// the probe's pins keep it resident until the hold drops.
    fn held_page_ids(&self) -> Vec<i32> {
        self.held.iter().map(|b| b.block_id() as i32).collect()
    }

    /// Lift the reuse cap to every complete block of the chain
    /// ([`crate::ResolvePolicy::full_pages`]): a pad-aligned handoff chain
    /// never prefills, so its boundary block is part of the required hit.
    pub(crate) fn include_final_block(&mut self) {
        self.cacheable = self.seq_hashes.len();
    }
}

/// An opaque strong pin on one registered KV block. While held it keeps the
/// block in the active pool (out of the free/inactive pools), so the physical
/// slot cannot be reallocated. Used to hold a block stable across an in-flight
/// async offload save; cheap to clone/drop (one `Arc` bump). See
/// [`RequestKv::assigned_block_guards`].
///
/// The inner guard is never read — it exists purely for its `Drop`, which
/// releases the pin. Holding the value *is* the contract.
pub struct KvBlockGuard(#[allow(dead_code)] ImmutableBlock<()>);

/// GPU destination blocks reserved for a CPU→GPU load, consumed by
/// [`BlockPool::commit_loaded_blocks`] once the DMA lands.
pub struct LoadReservation {
    blocks: Vec<MutableBlock<()>>,
}

impl LoadReservation {
    /// Physical page ids the connector loads the leased CPU blocks into, in
    /// lease order (the i-th leased block lands in `page_ids()[i]`).
    pub(crate) fn page_ids(&self) -> Vec<i32> {
        self.blocks.iter().map(|b| b.block_id() as i32).collect()
    }

    /// Number of reserved destination blocks.
    // `is_empty` was dead code (hawk sweep, #743); `len` stands alone.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.blocks.len()
    }
}

/// Out-of-vocab sentinel filling a handoff naming chain to the page
/// boundary. Both sides of a handoff must derive the identical padded
/// chain, so this is a store constant, not a caller parameter.
pub const PAD_TOKEN_ID: u32 = u32::MAX;

/// Page ids pinned by a resolved prefix, in prefix order — empty for
/// [`KvPrefix::none`] or a hold this store did not mint.
pub fn resolved_page_ids(prefix: &KvPrefix) -> Vec<i32> {
    prefix
        .hold_any()
        .and_then(|hold| hold.downcast_ref::<PrefixProbe>())
        .map(PrefixProbe::held_page_ids)
        .unwrap_or_default()
}

/// The only door to `&mut SchedulableSequence`. Every mutation settles the
/// resident-block delta against the pool's entitled counter in the same
/// call, so a block-drawing path added later cannot skip the accounting —
/// the raw field is private to this module and the compiler is the enforcer.
mod entitlement {
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::sync::atomic::AtomicI64;
    use std::sync::atomic::Ordering;

    use kvbm_logical::integrations::SchedulableSequence;

    pub(super) struct EntitledSeq {
        seq: SchedulableSequence<()>,
        /// The owning pool's entitled counter (`BlockPool::entitled`).
        counter: Arc<AtomicI64>,
        /// The owning pool's reserve gate: an admitted mutation and its
        /// counter settlement hold it together, so a concurrent reserve
        /// never sees returned blocks before their entitlement refund.
        gate: Arc<Mutex<()>>,
        /// Set by [`Self::admit`]; only admitted requests move the counter.
        admitted: bool,
        /// Input-plus-output page capacity, frozen at construction. The
        /// sequence may later reclassify an input token as generated
        /// (external-prefill anchor adoption), which must not shrink the
        /// reservation: the promoted token still occupies its sequence
        /// position and the dangling token at the end of the sequence still
        /// provisions a page beyond it.
        lifetime_blocks: usize,
    }

    impl EntitledSeq {
        pub(super) fn new(
            seq: SchedulableSequence<()>,
            counter: Arc<AtomicI64>,
            gate: Arc<Mutex<()>>,
            lifetime_blocks: usize,
        ) -> Self {
            Self {
                seq,
                counter,
                gate,
                admitted: false,
                lifetime_blocks,
            }
        }

        fn resident(&self) -> i64 {
            (self.seq.assigned_blocks() + self.seq.staged_blocks() + self.seq.unassigned_blocks())
                as i64
        }

        /// The un-drawn lifetime remainder this request is owed.
        pub(super) fn remaining(&self) -> i64 {
            self.lifetime_blocks as i64 - self.resident()
        }

        pub(super) fn is_admitted(&self) -> bool {
            self.admitted
        }

        /// Declare this request authoritative: from now until retire, the
        /// pool keeps its remainder out of opportunistic reach. Idempotent.
        pub(super) fn admit(&mut self) {
            if self.admitted {
                return;
            }
            self.admitted = true;
            self.counter.fetch_add(self.remaining(), Ordering::AcqRel);
        }

        /// Run a sequence mutation; the resident delta settles against the
        /// counter (draws shrink it, returned blocks grow it back). Admitted
        /// requests mutate under the reserve gate: a return path
        /// (revert/rejected drafts) frees blocks inside `f`, and without the
        /// gate a reserve could take them before the refund lands — the
        /// entitled floor would then exceed what remains. Unadmitted
        /// requests settle nothing and skip the lock.
        pub(super) fn with_draw<T>(
            &mut self,
            f: impl FnOnce(&mut SchedulableSequence<()>) -> T,
        ) -> T {
            if !self.admitted {
                return f(&mut self.seq);
            }
            let gate = Arc::clone(&self.gate);
            let _gate = gate.lock().expect("reserve gate poisoned");
            let before = self.resident();
            let out = f(&mut self.seq);
            self.counter
                .fetch_add(before - self.resident(), Ordering::AcqRel);
            out
        }

        /// Terminal refund of the remainder, BEFORE the blocks themselves
        /// return: while resident they were never entitled, and afterwards
        /// they are plain availability. Idempotent; `Drop` is the backstop.
        pub(super) fn retire(&mut self) {
            if !self.admitted {
                return;
            }
            self.admitted = false;
            self.counter.fetch_sub(self.remaining(), Ordering::AcqRel);
        }

        pub(super) fn seq(&self) -> &SchedulableSequence<()> {
            &self.seq
        }

        pub(super) fn lifetime_blocks(&self) -> usize {
            self.lifetime_blocks
        }
    }

    impl Drop for EntitledSeq {
        fn drop(&mut self) {
            self.retire();
        }
    }
}

/// Per-request KV state wrapping `SchedulableSequence`.
///
/// Lifecycle: `schedule_prefill → forward over step_page_indices →
/// apply_prefill`, then either `schedule_decode → forward → apply_decode` or
/// `schedule_speculative → forward → apply_speculative` in a loop
/// (`revert_schedule` undoes a reservation whose step failed). The forward
/// pass and its page-table view belong to the model, not to the logical
/// store.
pub struct RequestKv {
    inner: entitlement::EntitledSeq,
}

impl RequestKv {
    /// Declare this request admission-authoritative — atomically with the
    /// physical re-check, under the pool's reserve gate. On success the pool
    /// keeps this request's un-drawn lifetime remainder (`lifetime_blocks -
    /// resident_blocks`) out of opportunistic
    /// [`BlockPool::reserve_loaded_blocks`] reach until release/drop, so a
    /// mid-flight page-crossing draw can never fail.
    ///
    /// Returns false without admitting when the pool cannot cover every
    /// admitted remainder plus this one — i.e. a concurrent reserve won the
    /// race for these pages between the caller's budget check and this call;
    /// the caller defers and retries. `credited_blocks` are prefix-held pages
    /// that fold into this request's resident set on match: already out of
    /// `available_blocks`, soon to shrink the remainder, so counting them
    /// avoids a double charge. Idempotent once admitted.
    pub fn try_admit(&mut self, pool: &BlockPool, credited_blocks: usize) -> bool {
        if self.inner.is_admitted() {
            return true;
        }
        let _gate = pool.reserve_gate.lock().expect("reserve gate poisoned");
        let remaining = self.inner.remaining().max(0) as usize;
        if pool.available_blocks() + credited_blocks < pool.entitled_blocks() + remaining {
            return false;
        }
        self.inner.admit();
        true
    }

    // ── Prefix cache ───────────────────────────────────────────────────

    /// Match the prompt's full blocks against registered blocks and skip
    /// their prefill. Returns the number of cached tokens; `kv_position()`
    /// advances by the same amount. Must be called on a fresh request,
    /// before the first `schedule_prefill`.
    ///
    /// Matching always leaves at least one prompt token uncached so the
    /// final prefill chunk can emit the first generated token.
    pub fn match_and_add_prefix(&mut self, pool: &BlockPool) -> anyhow::Result<usize> {
        let blocks = self
            .inner
            .with_draw(|seq| seq.match_and_add_prefix(&pool.block_manager))
            .map_err(|e| anyhow::anyhow!("match_and_add_prefix: {e}"))?;
        Ok(blocks * self.inner.seq().block_size())
    }

    // ── Scheduling (allocates blocks) ──────────────────────────────────

    pub fn schedule_prefill(
        &mut self,
        num_tokens: usize,
        pool: &BlockPool,
    ) -> Result<(), ScheduleError> {
        self.inner
            .with_draw(|seq| seq.schedule_prefill(num_tokens, &pool.block_manager))
    }

    pub fn schedule_decode(&mut self, pool: &BlockPool) -> Result<(), ScheduleError> {
        self.inner
            .with_draw(|seq| seq.schedule_decode(&pool.block_manager))
    }

    /// Reserve KV for a speculative verify step covering `num_draft_tokens`
    /// consecutive positions (current dangling token + draft candidates).
    /// [`Self::apply_speculative`] commits the accepted prefix; on any failure
    /// [`Self::revert_schedule`] returns the reservation.
    pub fn schedule_speculative(
        &mut self,
        num_draft_tokens: usize,
        pool: &BlockPool,
    ) -> Result<(), ScheduleError> {
        self.inner
            .with_draw(|seq| seq.schedule_speculative(num_draft_tokens, &pool.block_manager))
    }

    // ── Apply (register blocks, advance position) ──────────────────────

    pub fn apply_prefill(&mut self, token: u32, pool: &BlockPool) -> anyhow::Result<()> {
        self.inner
            .with_draw(|seq| seq.apply_prefill(Some(token), &pool.block_manager))
            .map_err(|e| anyhow::anyhow!("apply_prefill: {e}"))
    }

    /// Apply a non-final prefill chunk: registers the chunk's blocks and
    /// advances `kv_position` without emitting a generated token. The final
    /// chunk must go through [`Self::apply_prefill`] instead.
    pub fn apply_prefill_chunk(&mut self, pool: &BlockPool) -> anyhow::Result<()> {
        self.inner
            .with_draw(|seq| seq.apply_prefill(None, &pool.block_manager))
            .map_err(|e| anyhow::anyhow!("apply_prefill_chunk: {e}"))
    }

    /// Complete the trailing partial block with [`PAD_TOKEN_ID`] and register
    /// it, so the handoff seal saves it under the padded hash. Naming only:
    /// no compute, no allocation, `kv_position` untouched. No-op when
    /// `kv_position` sits on a page boundary. Returns the pads appended.
    pub fn pad_to_boundary(&mut self, pool: &BlockPool) -> anyhow::Result<usize> {
        self.inner
            .with_draw(|seq| seq.pad_tail_block(PAD_TOKEN_ID, &pool.block_manager))
            .map_err(|e| anyhow::anyhow!("pad_to_boundary: {e}"))
    }

    /// Convert the one uncomputed final input token left by an external
    /// prefill restore into the normal dangling-token state expected by
    /// decode/speculative scheduling. Does not advance `kv_position`.
    pub fn adopt_external_prefill_anchor(&mut self) -> anyhow::Result<()> {
        self.inner
            .with_draw(kvbm_logical::SchedulableSequence::adopt_external_prefill_anchor)
            .map_err(|e| anyhow::anyhow!("adopt_external_prefill_anchor: {e}"))
    }

    pub fn apply_decode(&mut self, token: u32, pool: &BlockPool) -> anyhow::Result<DecodeOutcome> {
        self.inner
            .with_draw(|seq| seq.apply_decode(token, &pool.block_manager))
            .map_err(|e| anyhow::anyhow!("apply_decode: {e}"))
    }

    /// Commit the accepted prefix of a speculative verify step. kvbm keeps the
    /// `accepted_tokens` KV and LIFO-releases the rejected draft blocks.
    pub fn apply_speculative(
        &mut self,
        accepted_tokens: &[u32],
        pool: &BlockPool,
    ) -> anyhow::Result<DecodeOutcome> {
        self.inner
            .with_draw(|seq| seq.apply_speculative(accepted_tokens, &pool.block_manager))
            .map_err(|e| anyhow::anyhow!("apply_speculative: {e}"))
    }

    /// Undo a scheduled-but-unapplied KV reservation (e.g. a speculative
    /// schedule whose forward or apply failed) and return its blocks to the pool.
    pub fn revert_schedule(&mut self) -> anyhow::Result<()> {
        self.inner
            .with_draw(kvbm_logical::SchedulableSequence::revert_schedule)
            .map_err(|e| anyhow::anyhow!("revert_schedule: {e}"))
    }

    pub fn release(&mut self) -> anyhow::Result<()> {
        // Refund the entitlement remainder before the blocks return: while
        // resident they were never entitled, afterwards they are plain
        // availability.
        self.inner.retire();
        self.inner
            .with_draw(kvbm_logical::SchedulableSequence::release)
            .map_err(|e| anyhow::anyhow!("release: {e}"))
    }

    /// Mark every assigned block's canonical primary to reset on release.
    ///
    /// A duplicate resets itself but keeps its primary alive; marking the
    /// primary prevents that hidden block from entering the inactive cache on
    /// final drop.
    #[cfg(test)]
    fn mark_blocks_reset_on_release(&self) {
        for (_, block) in self.inner.seq().inner().assignments().assigned_iter() {
            block.set_primary_reset_on_release(true);
        }
    }

    // ── Queries ────────────────────────────────────────────────────────

    /// Tokens with KV already computed.
    pub fn kv_position(&self) -> usize {
        self.inner.seq().kv_position()
    }

    #[cfg(test)]
    fn is_complete(&self) -> bool {
        self.inner.seq().is_complete()
    }

    #[cfg(test)]
    fn generated_tokens(&self) -> usize {
        self.inner.seq().generated_tokens()
    }

    /// Full input-plus-output page capacity fixed when this request was
    /// created. Admission uses this value for already-active requests so any
    /// internal tokens added by a protocol remain accounted for.
    pub fn lifetime_blocks(&self) -> usize {
        self.inner.lifetime_blocks()
    }

    /// Physical blocks currently held by this request, including registered,
    /// staged, and eagerly allocated dangling blocks.
    pub fn resident_blocks(&self) -> usize {
        let seq = self.inner.seq();
        seq.assigned_blocks() + seq.staged_blocks() + seq.unassigned_blocks()
    }

    /// Physical page IDs assigned to this request, in sequence order.
    /// Includes every block the request currently holds — which can be one
    /// more than the KV tokens need (see `step_page_indices`).
    fn page_indices(&self) -> Vec<i32> {
        self.inner
            .seq()
            .inner()
            .assignments()
            .all_block_ids()
            .map(|&id| id as i32)
            .collect()
    }

    /// Physical pages covering the KV tokens currently committed to this
    /// request, in logical sequence order.
    pub fn current_page_indices(&self) -> Vec<i32> {
        let mut pages = self.page_indices();
        pages.truncate(
            self.inner
                .seq()
                .kv_position()
                .div_ceil(self.inner.seq().block_size()),
        );
        pages
    }

    /// Page IDs covering exactly the KV tokens present after this step
    /// appends `new_tokens` (`kv_position + new_tokens`). `page_indices()`
    /// can hold one block more: kvbm's `schedule_decode` eagerly allocates
    /// the next generation block whenever this step's token fills the last
    /// slot of the current block. Page tables built from the raw list make
    /// the kernel see a longer sequence than exists — use this for any
    /// per-step page row handed to a forward pass.
    pub fn step_page_indices(&self, new_tokens: usize) -> Vec<i32> {
        assert!(new_tokens > 0, "a forward step appends at least one token");
        let kv_tokens = self.inner.seq().kv_position() + new_tokens;
        let mut pages = self.page_indices();
        pages.truncate(kv_tokens.div_ceil(self.inner.seq().block_size()));
        pages
    }

    // ── KV offload bridge ──────────────────────────────────────────────

    /// Content hashes of every *full* prompt block, in prompt order.
    ///
    /// These are the keys the KV-offload connector queries the CPU tier with,
    /// so they must be identical across any two requests sharing a prefix.
    /// They are kvbm's lineage-based [`SequenceHash`], which is exactly that:
    /// position + content + parent fragment, so block `i` of prompt `P` hashes
    /// the same no matter which request computed it.
    #[cfg(test)]
    fn prompt_block_hashes(&self) -> Vec<[u8; 16]> {
        self.inner
            .seq()
            .inner()
            .sequence()
            .all_sequence_hashes()
            .iter()
            .map(sequence_hash_bytes)
            .collect()
    }

    /// `(page_id, content_hash)` for every block currently assigned to this
    /// request, in prompt order. Drives the offload save once a block seals;
    /// the first [`prefix_matched_blocks`](Self::prefix_matched_blocks) entries
    /// are GPU-hit reuse (already resident) and are normally skipped.
    pub fn assigned_block_hashes(&self) -> Vec<(i32, [u8; 16])> {
        self.inner
            .seq()
            .inner()
            .assignments()
            .assigned_iter()
            .map(|(&id, block)| (id as i32, sequence_hash_bytes(&block.sequence_hash())))
            .collect()
    }

    /// Strong pins for every block currently assigned to this request, aligned
    /// 1:1 (same order) with [`assigned_block_hashes`](Self::assigned_block_hashes).
    ///
    /// An offload save's GPU→CPU copy runs asynchronously after the save is
    /// submitted; holding the matching [`KvBlockGuard`] keeps that block out of
    /// the free/inactive pool until the copy lands, so a later request can't be
    /// allocated the same slot and overwrite it mid-copy. Drop the guard once
    /// the save reports done.
    pub(crate) fn assigned_block_guards(&self) -> Vec<KvBlockGuard> {
        self.inner
            .seq()
            .inner()
            .assignments()
            .assigned_iter()
            .map(|(_, block)| KvBlockGuard(block.clone()))
            .collect()
    }

    /// Number of leading blocks reused from the GPU prefix cache.
    pub(crate) fn prefix_matched_blocks(&self) -> usize {
        self.inner.seq().inner().prefix_matched_blocks()
    }
}

/// Pack a kvbm [`SequenceHash`] (lineage hash) into the 16-byte content key the
/// offload tier addresses blocks by. Big-endian for a stable on-wire ordering.
fn sequence_hash_bytes(hash: &SequenceHash) -> [u8; 16] {
    hash.as_u128().to_be_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The offload CPU-tier query keys are `prompt_block_hashes`. The whole
    /// load path is built on these being identical for any two requests that
    /// share a prefix (and diverging the moment content does) — otherwise a
    /// warm block saved by one request would never match the next. Guard it.
    #[test]
    fn prompt_block_hashes_stable_across_shared_prefix() {
        let pool = BlockPool::new(16, 256);
        let shared: Vec<u32> = (0..48).map(|i| 1000 + i).collect(); // 3 full blocks
        let mut a_tokens = shared.clone();
        a_tokens.extend((0..16).map(|i| 7000 + i)); // 4th block diverges
        let mut b_tokens = shared.clone();
        b_tokens.extend((0..16).map(|i| 9000 + i));

        let a = pool.new_request(a_tokens, 8, None);
        let b = pool.new_request(b_tokens, 8, None);
        let ha = a.prompt_block_hashes();
        let hb = b.prompt_block_hashes();

        assert_eq!(ha.len(), 4, "64 tokens / 16 = 4 full blocks");
        assert_eq!(hb.len(), 4);
        assert_eq!(ha[..3], hb[..3], "shared prefix must hash identically");
        assert_ne!(ha[3], hb[3], "divergent block must hash differently");
        assert!(ha.iter().all(|h| *h != [0u8; 16]), "hashes are non-trivial");

        // A different LoRA salt must poison the match — same tokens, new keys.
        let c = pool.new_request(shared, 8, Some("adapter-x"));
        assert_ne!(
            c.prompt_block_hashes()[0],
            ha[0],
            "salt (lora) must scope the prefix cache"
        );
    }

    #[test]
    fn cache_salt_isolates_request_and_probe_prefixes() {
        let pool = BlockPool::new(16, 32);
        let prompt = vec![7u32; 48]; // 3 full blocks

        let mut first =
            pool.new_request_with_cache_salt(prompt.clone(), 4, Some("native-mtp-prompt-a"), None);
        first
            .schedule_prefill(48, &pool)
            .expect("first prefill schedule");
        first.apply_prefill(42, &pool).expect("first prefill apply");
        first.release().expect("first release");

        let other_probe =
            pool.probe_prefix_with_cache_salt(prompt.clone(), Some("native-mtp-prompt-b"), None);
        assert_eq!(
            other_probe.gpu_hit_blocks(),
            0,
            "the offload probe must use the same cache-salt isolation"
        );
        drop(other_probe);

        let repeated_probe =
            pool.probe_prefix_with_cache_salt(prompt.clone(), Some("native-mtp-prompt-a"), None);
        assert_eq!(repeated_probe.gpu_hit_blocks(), 3);
        drop(repeated_probe);

        let mut other =
            pool.new_request_with_cache_salt(prompt.clone(), 4, Some("native-mtp-prompt-b"), None);
        assert_eq!(
            other.match_and_add_prefix(&pool).expect("other match"),
            0,
            "a distinct cache salt must isolate identical token blocks"
        );

        let mut repeated =
            pool.new_request_with_cache_salt(prompt, 4, Some("native-mtp-prompt-a"), None);
        assert_eq!(
            repeated
                .match_and_add_prefix(&pool)
                .expect("repeated match"),
            32,
            "the same cache salt must preserve ordinary prefix reuse"
        );
    }

    #[test]
    fn request_reports_the_lifetime_capacity_it_was_created_with() {
        let pool = BlockPool::new(16, 8);
        let req = pool.new_request(vec![1; 16], 17, None);

        assert_eq!(req.lifetime_blocks(), 3);
    }

    #[test]
    fn external_prefill_anchor_promotion_keeps_lifetime_capacity() {
        let pool = BlockPool::new(16, 8);
        // 16 restored tokens + 1 anchor tail + 16 output positions. The
        // anchor promotion reclassifies one input token as generated; the
        // reservation must not shrink — the promoted token still occupies a
        // sequence position and the dangling token at the end of the
        // sequence still provisions a page beyond it.
        let mut req = pool.new_request(vec![1; 17], 16, None);
        assert_eq!(req.lifetime_blocks(), 3);
        req.schedule_prefill(16, &pool).expect("restored chunk");
        req.apply_prefill_chunk(&pool).expect("restored apply");
        req.adopt_external_prefill_anchor()
            .expect("anchor adoption");
        assert_eq!(
            req.lifetime_blocks(),
            3,
            "promoting the external anchor to generated must not shrink the reservation"
        );
    }

    fn complete_non_retained_speculative_request(
        pool: &BlockPool,
        prompt: &[u32],
        max_output_tokens: usize,
    ) {
        const MAX_PREFILL_CHUNK: usize = 1024;
        const MAX_VERIFY_SPAN: usize = 17;

        let mut request = pool.new_request(prompt.to_vec(), max_output_tokens, None);
        let mut prefilled = 0;
        while prompt.len() - prefilled > MAX_PREFILL_CHUNK {
            request
                .schedule_prefill(MAX_PREFILL_CHUNK, pool)
                .expect("schedule prefill chunk");
            request
                .apply_prefill_chunk(pool)
                .expect("apply prefill chunk");
            prefilled += MAX_PREFILL_CHUNK;
        }
        request
            .schedule_prefill(prompt.len() - prefilled, pool)
            .expect("schedule final prefill chunk");
        request
            .apply_prefill(70_000, pool)
            .expect("apply final prefill chunk");

        while !request.is_complete() {
            let span = (max_output_tokens - request.generated_tokens()).min(MAX_VERIFY_SPAN);
            request
                .schedule_speculative(span, pool)
                .expect("schedule speculative verify");
            let accepted = (0..span)
                .map(|offset| 80_000 + offset as u32)
                .collect::<Vec<_>>();
            request
                .apply_speculative(&accepted, pool)
                .expect("apply speculative verify");
        }

        request.mark_blocks_reset_on_release();
    }

    /// CPU-only shape of issue #681: 160 request pages, a 1643-token prompt
    /// split into 1024 + 619 prefill chunks, 512 output tokens, and 17-token
    /// speculative verify spans. A completed request must leave enough clean
    /// pages for the identical request to finish again.
    #[test]
    fn issue_681_profiled_capacity_reuses_blocks_across_requests() {
        // BlockPool reserves one additional CUDA-graph padding page.
        let pool = BlockPool::new(16, 161);
        let baseline = pool.available_blocks();
        let prompt = (0..1643).map(|i| 30_000 + i).collect::<Vec<_>>();

        for round in 0..2 {
            complete_non_retained_speculative_request(&pool, &prompt, 512);
            assert_eq!(
                pool.available_blocks(),
                baseline,
                "issue #681 request {round} did not release its KV pages"
            );
        }
    }

    /// kvbm's `schedule_decode` allocates the next generation block when the
    /// appended token fills the current block (`need = pending + 1`), so the
    /// raw `page_indices()` exceeds `ceil(kv_tokens / block_size)` at every
    /// block boundary. `step_page_indices` must hand the forward pass an
    /// exact page row at every step — this deadlocked Kimi DP8 on H200 when
    /// the raw list reached the worker's exact-match page-table check, and
    /// made qwen3's FlashInfer metadata read one garbage page past the
    /// sequence at every block boundary (#291).
    #[test]
    fn step_page_indices_exact_at_block_boundaries() {
        let mut raw_overshoots = 0usize;
        for prompt_len in [1usize, 15, 16, 17, 31, 32, 33, 40, 47, 48] {
            let pool = BlockPool::new(16, 256);
            let mut kv =
                pool.new_request((0..prompt_len as u32).map(|i| 100 + i).collect(), 24, None);
            kv.schedule_prefill(prompt_len, &pool).unwrap();
            assert_eq!(
                kv.step_page_indices(prompt_len).len(),
                prompt_len.div_ceil(16),
                "prefill page row P={prompt_len}"
            );
            kv.apply_prefill(1000, &pool).unwrap();
            for step in 0..23u32 {
                kv.schedule_decode(&pool).unwrap();
                let need = (kv.kv_position() + 1).div_ceil(16);
                assert_eq!(
                    kv.step_page_indices(1).len(),
                    need,
                    "decode page row P={prompt_len} step={step}"
                );
                raw_overshoots += usize::from(kv.page_indices().len() > need);
                kv.apply_decode(2000 + step, &pool).unwrap();
            }
        }
        assert!(
            raw_overshoots > 0,
            "kvbm no longer over-allocates the generation block; \
             step_page_indices and this test can be retired"
        );
    }

    /// The maintained counter audited against recomputation from request
    /// truth at every lifecycle stage — draws shrink it, returned blocks
    /// (revert) grow it back, release refunds the remainder exactly.
    #[test]
    fn entitlement_tracks_the_undrained_remainder_through_the_lifecycle() {
        let pool = BlockPool::new(16, 64);
        let audit = |pool: &BlockPool, kv: &RequestKv, stage: &str| {
            assert_eq!(
                pool.entitled_blocks(),
                kv.lifetime_blocks() - kv.resident_blocks(),
                "counter vs recomputed remainder at {stage}"
            );
        };

        // 32-token prompt + 33 outputs: lifetime = ceil(65/16) = 5.
        let mut kv = pool.new_request((0..32).map(|i| 100 + i).collect(), 33, None);
        assert_eq!(pool.entitled_blocks(), 0, "construction entitles nothing");
        assert!(kv.try_admit(&pool, 0));
        audit(&pool, &kv, "admit");

        kv.schedule_prefill(32, &pool).unwrap();
        audit(&pool, &kv, "schedule_prefill");
        kv.apply_prefill(1000, &pool).unwrap();
        audit(&pool, &kv, "apply_prefill");

        for step in 0..16u32 {
            kv.schedule_decode(&pool).unwrap();
            audit(&pool, &kv, "schedule_decode");
            kv.apply_decode(2000 + step, &pool).unwrap();
            audit(&pool, &kv, "apply_decode");
        }

        // The shrink path: a reverted reservation returns its blocks while
        // the request lives on, so the entitlement grows back.
        kv.schedule_speculative(8, &pool).unwrap();
        audit(&pool, &kv, "schedule_speculative");
        kv.revert_schedule().unwrap();
        audit(&pool, &kv, "revert_schedule");

        kv.release().unwrap();
        assert_eq!(pool.entitled_blocks(), 0, "release refunds the remainder");
    }

    #[test]
    fn entitlement_refunds_on_drop_without_release() {
        let pool = BlockPool::new(16, 64);
        let mut kv = pool.new_request(vec![1; 16], 16, None);
        assert!(kv.try_admit(&pool, 0));
        kv.schedule_prefill(16, &pool).unwrap();
        assert!(pool.entitled_blocks() > 0);
        drop(kv);
        assert_eq!(pool.entitled_blocks(), 0, "Drop is the retire backstop");
    }

    #[test]
    fn unadmitted_requests_never_move_the_counter() {
        let pool = BlockPool::new(16, 64);
        let mut kv = pool.new_request(vec![1; 16], 16, None);
        kv.schedule_prefill(16, &pool).unwrap();
        kv.apply_prefill(1000, &pool).unwrap();
        assert_eq!(pool.entitled_blocks(), 0);
        kv.release().unwrap();
        let probe = pool.probe_prefix_with_cache_salt(vec![1; 16], None, None);
        assert_eq!(pool.entitled_blocks(), 0, "probe requests entitle nothing");
        drop(probe);
        assert_eq!(pool.entitled_blocks(), 0);
    }

    /// The floor itself: an opportunistic reserve may only take what is left
    /// above every admitted request's un-drawn remainder — even though those
    /// blocks sit in the free pool.
    #[test]
    fn reserve_loaded_blocks_declines_into_the_entitled_floor() {
        // 8 blocks minus padding = 7 available.
        let pool = BlockPool::new(16, 8);
        assert_eq!(pool.available_blocks(), 7);

        // 16-token prompt + 65 outputs: lifetime = ceil(81/16) = 6.
        let mut kv = pool.new_request(vec![1; 16], 65, None);
        assert!(kv.try_admit(&pool, 0));
        assert_eq!(pool.entitled_blocks(), 6);

        assert!(
            pool.reserve_loaded_blocks(2).is_none(),
            "2 + entitled 6 exceeds available 7"
        );
        let held = pool
            .reserve_loaded_blocks(1)
            .expect("1 + entitled 6 fits available 7");

        // The entitled party draws past the floor without contention.
        kv.schedule_prefill(16, &pool).unwrap();
        assert_eq!(pool.entitled_blocks(), 5);
        assert!(
            pool.reserve_loaded_blocks(1).is_none(),
            "the draw shrank available and entitlement together"
        );

        kv.revert_schedule().unwrap();
        assert_eq!(pool.entitled_blocks(), 6, "the reverted draw grows it back");
        kv.release().unwrap();
        assert_eq!(pool.entitled_blocks(), 0);
        drop(held);
        assert!(
            pool.reserve_loaded_blocks(7).is_some(),
            "with no admitted requests the whole pool is reservable"
        );
    }

    /// Both orders of the admit-vs-reserve race resolve safely under the
    /// shared gate: whichever move lands second sees the first and yields.
    #[test]
    fn try_admit_and_reserve_yield_to_whichever_landed_first() {
        // 8 blocks minus padding = 7 available; lifetime = ceil(81/16) = 6.
        let pool = BlockPool::new(16, 8);

        // Reserve first: admission must defer instead of over-committing.
        let held = pool.reserve_loaded_blocks(2).expect("empty floor");
        let mut kv = pool.new_request(vec![1; 16], 65, None);
        assert!(
            !kv.try_admit(&pool, 0),
            "5 available cannot cover a remainder of 6"
        );
        assert_eq!(pool.entitled_blocks(), 0, "a lost race entitles nothing");
        drop(held);
        assert!(kv.try_admit(&pool, 0), "the freed pages admit the retry");
        assert!(kv.try_admit(&pool, 0), "idempotent once admitted");

        // Admit first: the reserve must decline into the new floor.
        assert!(pool.reserve_loaded_blocks(2).is_none());
        kv.release().unwrap();
    }

    /// Prefix-held pages are already out of `available` and shrink the
    /// remainder on match — crediting them keeps a cache-hit request
    /// admittable at exact capacity.
    #[test]
    fn try_admit_credits_prefix_held_blocks() {
        let pool = BlockPool::new(16, 8);
        let held = pool.reserve_loaded_blocks(2).expect("empty floor");
        let mut kv = pool.new_request(vec![1; 16], 65, None);
        assert!(!kv.try_admit(&pool, 0));
        assert!(
            kv.try_admit(&pool, 1),
            "5 available + 1 credited page covers the remainder of 6"
        );
        drop(held);
    }

    /// Concurrency fuzz for the Codex-flagged race: reserver threads hammer
    /// the pool while a scheduler thread runs admit → draw → release
    /// lifecycles. The invariant under test is the whole point of the
    /// mechanism — an admitted request's draw NEVER fails, no matter how the
    /// reserves interleave — plus counter integrity at the end.
    #[test]
    fn fuzz_concurrent_reserves_never_starve_admitted_draws() {
        use std::sync::atomic::AtomicBool;

        // Small pool for real contention: 16 usable blocks, lifetimes of 5,
        // reserves of 1..=6.
        let pool = Arc::new(BlockPool::new(16, 17));
        let stop = Arc::new(AtomicBool::new(false));

        // Each reserver holds at most one reservation of <=4 blocks and
        // drops before re-grabbing: peak held = 12 of 16 blocks, so the
        // smallest requests (lifetime 3) always find room eventually while
        // the bigger ones exercise the defer path under real contention.
        let reservers: Vec<_> = (0..3u64)
            .map(|seed| {
                let pool = Arc::clone(&pool);
                let stop = Arc::clone(&stop);
                std::thread::spawn(move || {
                    let mut rng = 0x9e37_79b9_7f4a_7c15_u64.wrapping_add(seed);
                    let mut held: Option<LoadReservation> = None;
                    while !stop.load(Ordering::Acquire) {
                        rng = rng
                            .wrapping_mul(6_364_136_223_846_793_005)
                            .wrapping_add(1_442_695_040_888_963_407);
                        let n = (rng >> 33) as usize % 4 + 1;
                        drop(held.take());
                        held = pool.reserve_loaded_blocks(n);
                    }
                })
            })
            .collect();

        // Scheduler thread (this one): full lifecycles; a deferred admit is
        // legal, a failed draw after admission never is.
        let mut admitted_rounds = 0u32;
        let mut rng = 0xdead_beef_cafe_f00d_u64;
        for round in 0..20_000u32 {
            rng = rng
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let max_output = 32 + ((rng >> 13) as usize % 4) * 16; // lifetime 3..=6
            let prompt: Vec<u32> = (0..16).map(|i| round.wrapping_mul(31) + i).collect();
            let mut kv = pool.new_request(prompt, max_output, None);
            if !kv.try_admit(&pool, 0) {
                continue;
            }
            admitted_rounds += 1;
            kv.schedule_prefill(16, &pool)
                .expect("an admitted prefill draw must never fail");
            kv.apply_prefill(70_000, &pool)
                .expect("apply after a successful schedule");
            rng = rng
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let decode_steps = (rng >> 33) as usize % (max_output - 4);
            for step in 0..decode_steps {
                kv.schedule_decode(&pool)
                    .expect("an admitted decode draw must never fail");
                kv.apply_decode(80_000 + step as u32, &pool)
                    .expect("apply after a successful schedule");
            }
            // The return path: a reverted speculative reservation frees
            // blocks and refunds entitlement — racing reserves must see
            // both moves together or the floor over-commits.
            kv.schedule_speculative(3, &pool)
                .expect("an admitted speculative draw must never fail");
            kv.revert_schedule().unwrap();
            kv.mark_blocks_reset_on_release();
            kv.release().unwrap();
            assert_eq!(
                pool.entitled_blocks(),
                0,
                "round {round} leaked entitlement"
            );
        }
        stop.store(true, Ordering::Release);
        for t in reservers {
            t.join().unwrap();
        }
        assert_eq!(pool.entitled_blocks(), 0);
        assert!(
            admitted_rounds > 100,
            "contention starved admission entirely ({admitted_rounds} rounds) — \
             the fuzz lost its subject"
        );
    }
}
