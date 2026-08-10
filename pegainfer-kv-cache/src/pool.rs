use std::sync::Arc;

use dynamo_kv_hashing::compute_salt_hash;
use kvbm_logical::KvbmSequenceHashProvider;
use kvbm_logical::SequenceHash;
use kvbm_logical::blocks::ImmutableBlock;
use kvbm_logical::blocks::MutableBlock;
use kvbm_logical::events::EventsManager;
use kvbm_logical::events::KvCacheEvent;
use kvbm_logical::integrations::DecodeOutcome;
use kvbm_logical::integrations::SchedulableSequence;
use kvbm_logical::integrations::ScheduleError;
use kvbm_logical::manager::BlockManager;
use kvbm_logical::pools::BlockDuplicationPolicy;
use kvbm_logical::registry::BlockRegistry;
use tokio::sync::broadcast;

use crate::view::KvView;

/// Broadcast capacity for the opt-in KV-event feed. Generous: the scheduler
/// drains it every step, but a step can register/evict many blocks, and a
/// dropped event silently desyncs the router's prefix tree. Only allocated on
/// the [`BlockPool::with_events`] path.
const KV_EVENT_CHANNEL_CAPACITY: usize = 65_536;

/// Logical KV block pool: a `BlockManager` plus the reserved padding block.
///
/// Owns no GPU memory — the physical layout (full-attention `KvBuffer`,
/// MLA dual ckv/kpe buffers, ...) lives with the consumer and is indexed
/// by the block IDs this pool hands out.
pub struct BlockPool {
    block_manager: BlockManager<()>,
    block_size: usize,
    padding_block_id: usize,
}

impl BlockPool {
    pub fn new(block_size: usize, num_blocks: usize) -> anyhow::Result<Self> {
        Self::build(block_size, num_blocks, BlockRegistry::builder().build())
    }

    /// Like [`new`](Self::new), but the pool also emits block store/remove
    /// events for an out-of-band cache-aware router, returned as a raw broadcast
    /// receiver to drain synchronously. The stream carries `Create` events too
    /// (the emission policy can't separate them) — the consumer ignores `Create`
    /// and sources richer store events from the seal site, and maps the lineage
    /// hash in each `Remove` back to the router's `sequence_hash`. OFF by
    /// default: plain single-machine serving uses [`new`](Self::new) and pays
    /// neither the per-block event attachment nor the channel.
    pub(crate) fn with_events(
        block_size: usize,
        num_blocks: usize,
    ) -> anyhow::Result<(Self, broadcast::Receiver<KvCacheEvent>)> {
        // Default emission policy is AllEventsPolicy (every block tracked).
        let events = Arc::new(
            EventsManager::builder()
                .channel_capacity(KV_EVENT_CHANNEL_CAPACITY)
                .build(),
        );
        let rx = events.subscribe_receiver();
        let registry = BlockRegistry::builder().event_manager(events).build();
        let pool = Self::build(block_size, num_blocks, registry)?;
        Ok((pool, rx))
    }

    fn build(
        block_size: usize,
        num_blocks: usize,
        registry: BlockRegistry,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(num_blocks >= 2, "need at least 2 blocks (1 for padding)");

        let block_manager = BlockManager::builder()
            .block_count(num_blocks)
            .block_size(block_size)
            .registry(registry)
            .duplication_policy(BlockDuplicationPolicy::Allow)
            .with_lru_backend()
            .build()
            .map_err(|e| anyhow::anyhow!("BlockManager build failed: {e}"))?;

        // Reserve block 0 as CUDA Graph padding slot.
        let padding_blocks = block_manager
            .allocate_blocks(1)
            .ok_or_else(|| anyhow::anyhow!("failed to allocate padding block"))?;
        let padding_block_id = padding_blocks[0].block_id();
        let padding_complete = padding_blocks
            .into_iter()
            .next()
            .unwrap()
            .stage(SequenceHash::default(), block_size)
            .map_err(|e| anyhow::anyhow!("padding block stage failed: {e}"))?;
        // Register so it stays alive (ImmutableBlock RAII keeps it out of the
        // free pool), then leak — padding lives for the lifetime of the engine.
        std::mem::forget(block_manager.register_block(padding_complete));

        Ok(Self {
            block_manager,
            block_size,
            padding_block_id,
        })
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn available_blocks(&self) -> usize {
        self.block_manager.available_blocks()
    }

    pub fn total_blocks(&self) -> usize {
        self.block_manager.total_blocks()
    }

    pub fn padding_block_id(&self) -> i32 {
        self.padding_block_id as i32
    }

    /// Maximum blocks a single request can consume (total minus padding).
    pub fn max_request_blocks(&self) -> usize {
        self.block_manager.total_blocks().saturating_sub(1)
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
    fn new_request_with_cache_salt(
        &self,
        prompt_tokens: Vec<u32>,
        max_output_tokens: usize,
        cache_salt: Option<&str>,
        lora_name: Option<&str>,
    ) -> RequestKv {
        let salt_hash = compute_salt_hash(cache_salt, lora_name)
            .expect("compute_salt_hash is infallible for string cache/lora identities");
        let seq = SchedulableSequence::new(
            prompt_tokens,
            max_output_tokens,
            self.block_size as u32,
            None,
            Some(salt_hash),
        );
        RequestKv {
            seq,
            emitted_blocks: 0,
        }
    }

    // ── KV-offload prefetch (CPU-tier load before prefill) ─────────────

    /// Resolve `prompt_tokens` against the GPU prefix cache *without* creating
    /// a request, returning a [`PrefixProbe`] that holds the GPU-hit prefix
    /// blocks alive so an async CPU-tier load can extend it. The connector
    /// queries the probe's [`PrefixProbe::cpu_query_hashes`] against the host
    /// tier, then [`reserve_loaded_blocks`](Self::reserve_loaded_blocks) +
    /// load + [`commit_loaded_blocks`](Self::commit_loaded_blocks).
    ///
    /// `lora_name` must match the request's adapter — it salts the block
    /// hashes, so probing under the wrong adapter would query unrelated keys.
    pub fn probe_prefix(&self, prompt_tokens: Vec<u32>, lora_name: Option<&str>) -> PrefixProbe {
        self.probe_prefix_with_cache_salt(prompt_tokens, None, lora_name)
    }

    /// [`Self::probe_prefix`] with the same additional cache scope accepted by
    /// [`Self::new_request_with_cache_salt`]. The producer request and every
    /// restore probe must derive the identical salt.
    fn probe_prefix_with_cache_salt(
        &self,
        prompt_tokens: Vec<u32>,
        cache_salt: Option<&str>,
        lora_name: Option<&str>,
    ) -> PrefixProbe {
        let num_input = prompt_tokens.len();
        let rkv = self.new_request_with_cache_salt(prompt_tokens, 0, cache_salt, lora_name);
        let seq_hashes = rkv.seq.inner().sequence().all_sequence_hashes();
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
    /// prefetch and prefills from scratch). The reservation's
    /// [`LoadReservation::page_ids`] feed the connector's load; on completion
    /// hand it to [`commit_loaded_blocks`](Self::commit_loaded_blocks).
    pub fn reserve_loaded_blocks(&self, count: usize) -> Option<LoadReservation> {
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
    pub fn commit_loaded_blocks(&self, probe: &mut PrefixProbe, reservation: LoadReservation) {
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
    pub fn gpu_hit_blocks(&self) -> usize {
        self.gpu_hit
    }

    /// Total blocks this probe holds: the GPU-hit prefix plus any committed from
    /// a CPU-tier load. They are already out of the free pool and become the
    /// request's cached prefix at prefill, so admission credits them against the
    /// request's block need (avoiding a double-count against `available_blocks`).
    pub fn held_blocks(&self) -> usize {
        self.held.len()
    }

    /// Content hashes to query the CPU tier with: the blocks past the GPU hit,
    /// capped at the reuse boundary. Empty when the GPU hit already covers
    /// every reusable block (nothing to load — prefill normally).
    pub fn cpu_query_hashes(&self) -> Vec<Vec<u8>> {
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
    pub fn page_ids(&self) -> Vec<i32> {
        self.blocks.iter().map(|b| b.block_id() as i32).collect()
    }

    /// Number of reserved destination blocks.
    // `is_empty` was dead code (hawk sweep, #743); `len` stands alone.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.blocks.len()
    }
}

/// Per-request KV state wrapping `SchedulableSequence`.
///
/// Lifecycle: `schedule_prefill → prefill_view/pages → forward → apply_prefill`,
/// then either `schedule_decode → decode_view → forward → apply_decode` or
/// `schedule_speculative → speculative_view → forward → apply_speculative` in a
/// loop (`revert_schedule` undoes a reservation whose step failed).
pub struct RequestKv {
    seq: SchedulableSequence<()>,
    /// Cursor for [`Self::take_newly_registered_blocks`]: how many of this
    /// request's sequence blocks have already been surfaced as KV-router store
    /// events. Starts past the prefix-cache hit (those were stored by whoever
    /// first sealed them — this assumes GPU-resident reuse, i.e. KV offload off,
    /// which holds wherever the event feed is enabled). Untouched on the plain
    /// path where the feed is off.
    emitted_blocks: usize,
}

impl RequestKv {
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
            .seq
            .match_and_add_prefix(&pool.block_manager)
            .map_err(|e| anyhow::anyhow!("match_and_add_prefix: {e}"))?;
        // Prefix-hit blocks are already in the router's tree (whoever first
        // sealed them stored them, and a GPU hit means they were never evicted),
        // so the store-event cursor skips them.
        self.emitted_blocks = self.seq.assigned_blocks();
        Ok(blocks * self.seq.block_size())
    }

    // ── Scheduling (allocates blocks) ──────────────────────────────────

    pub fn schedule_prefill(
        &mut self,
        num_tokens: usize,
        pool: &BlockPool,
    ) -> Result<(), ScheduleError> {
        self.seq.schedule_prefill(num_tokens, &pool.block_manager)
    }

    pub fn schedule_decode(&mut self, pool: &BlockPool) -> Result<(), ScheduleError> {
        self.seq.schedule_decode(&pool.block_manager)
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
        self.seq
            .schedule_speculative(num_draft_tokens, &pool.block_manager)
    }

    // ── Views (for forward pass) ───────────────────────────────────────

    /// Build an immutable `KvView` for prefill.
    ///
    /// `prompt_len` tokens will be appended starting at `kv_position()`.
    /// The view's seq_len = kv_position + prompt_len (post-advance state
    /// that FlashInfer attention metadata expects). The page row is exact
    /// (`step_page_indices`), never the raw block holdings — the raw list
    /// can carry an eagerly-allocated next block the kernel must not see.
    pub fn prefill_view(&self, prompt_len: usize) -> KvView {
        let target_seq_len = self.seq.kv_position() + prompt_len;
        KvView::new(
            self.step_page_indices(prompt_len),
            target_seq_len,
            self.seq.block_size(),
        )
    }

    /// Build an immutable `KvView` for decode (one new token). Same exact
    /// page-row contract as `prefill_view`.
    pub fn decode_view(&self) -> KvView {
        KvView::new(
            self.step_page_indices(1),
            self.seq.kv_position() + 1,
            self.seq.block_size(),
        )
    }

    /// Build an immutable `KvView` for speculative verification: the verifier
    /// forwards `num_draft_tokens` consecutive positions (current dangling token
    /// followed by draft candidates). The view covers the post-step KV extent;
    /// [`Self::apply_speculative`] later commits only the accepted prefix and
    /// releases excess draft capacity. Same exact page-row contract as
    /// [`Self::prefill_view`].
    pub fn speculative_view(&self, num_draft_tokens: usize) -> KvView {
        KvView::new(
            self.step_page_indices(num_draft_tokens),
            self.seq.kv_position() + num_draft_tokens,
            self.seq.block_size(),
        )
    }

    // ── Apply (register blocks, advance position) ──────────────────────

    pub fn apply_prefill(&mut self, token: u32, pool: &BlockPool) -> anyhow::Result<()> {
        self.seq
            .apply_prefill(Some(token), &pool.block_manager)
            .map_err(|e| anyhow::anyhow!("apply_prefill: {e}"))
    }

    /// Apply a non-final prefill chunk: registers the chunk's blocks and
    /// advances `kv_position` without emitting a generated token. The final
    /// chunk must go through [`Self::apply_prefill`] instead.
    pub fn apply_prefill_chunk(&mut self, pool: &BlockPool) -> anyhow::Result<()> {
        self.seq
            .apply_prefill(None, &pool.block_manager)
            .map_err(|e| anyhow::anyhow!("apply_prefill_chunk: {e}"))
    }

    pub fn apply_decode(&mut self, token: u32, pool: &BlockPool) -> anyhow::Result<DecodeOutcome> {
        self.seq
            .apply_decode(token, &pool.block_manager)
            .map_err(|e| anyhow::anyhow!("apply_decode: {e}"))
    }

    /// Commit the accepted prefix of a speculative verify step. kvbm keeps the
    /// `accepted_tokens` KV and LIFO-releases the rejected draft blocks.
    pub fn apply_speculative(
        &mut self,
        accepted_tokens: &[u32],
        pool: &BlockPool,
    ) -> anyhow::Result<DecodeOutcome> {
        self.seq
            .apply_speculative(accepted_tokens, &pool.block_manager)
            .map_err(|e| anyhow::anyhow!("apply_speculative: {e}"))
    }

    /// Undo a scheduled-but-unapplied KV reservation (e.g. a speculative
    /// schedule whose forward or apply failed) and return its blocks to the pool.
    pub fn revert_schedule(&mut self) -> anyhow::Result<()> {
        self.seq
            .revert_schedule()
            .map_err(|e| anyhow::anyhow!("revert_schedule: {e}"))
    }

    pub fn release(&mut self) -> anyhow::Result<()> {
        self.seq
            .release()
            .map_err(|e| anyhow::anyhow!("release: {e}"))
    }

    /// Mark every assigned block's canonical primary to reset on release.
    ///
    /// A duplicate resets itself but keeps its primary alive; marking the
    /// primary prevents that hidden block from entering the inactive cache on
    /// final drop.
    pub fn mark_blocks_reset_on_release(&self) {
        for (_, block) in self.seq.inner().assignments().assigned_iter() {
            block.set_primary_reset_on_release(true);
        }
    }

    // ── Queries ────────────────────────────────────────────────────────

    /// Tokens with KV already computed.
    pub fn kv_position(&self) -> usize {
        self.seq.kv_position()
    }

    pub fn is_complete(&self) -> bool {
        self.seq.is_complete()
    }

    pub fn generated_tokens(&self) -> usize {
        self.seq.generated_tokens()
    }

    /// Physical blocks currently held by this request, including registered,
    /// staged, and eagerly allocated dangling blocks.
    pub fn resident_blocks(&self) -> usize {
        self.seq.assigned_blocks() + self.seq.staged_blocks() + self.seq.unassigned_blocks()
    }

    /// Physical page IDs assigned to this request, in sequence order.
    /// Includes every block the request currently holds — which can be one
    /// more than the KV tokens need (see `step_page_indices`).
    fn page_indices(&self) -> Vec<i32> {
        self.seq
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
        pages.truncate(self.seq.kv_position().div_ceil(self.seq.block_size()));
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
        let kv_tokens = self.seq.kv_position() + new_tokens;
        let mut pages = self.page_indices();
        pages.truncate(kv_tokens.div_ceil(self.seq.block_size()));
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
        self.seq
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
        self.seq
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
    pub fn assigned_block_guards(&self) -> Vec<KvBlockGuard> {
        self.seq
            .inner()
            .assignments()
            .assigned_iter()
            .map(|(_, block)| KvBlockGuard(block.clone()))
            .collect()
    }

    /// Number of leading blocks reused from the GPU prefix cache.
    pub fn prefix_matched_blocks(&self) -> usize {
        self.seq.inner().prefix_matched_blocks()
    }

    /// Blocks this request has registered (made cacheable) since the last call,
    /// in the u64 hash space a KV-router consumes plus the kvbm lineage hash.
    ///
    /// A block becomes reusable by other requests once it is *registered*
    /// (assigned an `ImmutableBlock`), which lags sealing the token block by a
    /// step. Diffs the registered count against an internal cursor and returns
    /// the new run in sequence order; prefix-cache hits are skipped (see
    /// [`Self::emitted_blocks`]). Empty when nothing new registered this step.
    pub fn take_newly_registered_blocks(&mut self) -> Vec<RegisteredBlock> {
        let registered = self.seq.assigned_blocks();
        if registered <= self.emitted_blocks {
            return Vec::new();
        }
        let blocks = self.seq.inner().sequence().blocks();
        debug_assert!(
            registered <= blocks.len(),
            "registered blocks ({registered}) exceed sealed token blocks ({})",
            blocks.len()
        );
        let new = blocks[self.emitted_blocks..registered]
            .iter()
            .map(|b| RegisteredBlock {
                plh: b.kvbm_sequence_hash().as_u128(),
                sequence_hash: b.sequence_hash(),
                tokens_hash: b.block_hash(),
                parent_sequence_hash: b.parent_sequence_hash(),
            })
            .collect();
        self.emitted_blocks = registered;
        new
    }
}

/// One full KV block a request just registered, ready to become a KV-router
/// store event. `sequence_hash`/`tokens_hash`/`parent_sequence_hash` are the
/// router's u64 hashes (`dynamo_tokens::TokenBlock` accessors, identical by
/// construction to what the router recomputes); `plh` is the engine's 128-bit
/// lineage hash, kept only to correlate an eviction event (which carries the
/// lineage hash) back to `sequence_hash`.
#[derive(Clone, Copy, Debug)]
pub struct RegisteredBlock {
    pub plh: u128,
    pub sequence_hash: u64,
    pub tokens_hash: u64,
    pub parent_sequence_hash: Option<u64>,
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
        let pool = BlockPool::new(16, 256).unwrap();
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
        let pool = BlockPool::new(16, 32).unwrap();
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
        let pool = BlockPool::new(16, 161).unwrap();
        assert_eq!(pool.max_request_blocks(), 160);
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
    /// sequence at every block boundary (#291). `prefill_view`/`decode_view`
    /// must carry the same exact row.
    #[test]
    fn step_page_indices_exact_at_block_boundaries() {
        let mut raw_overshoots = 0usize;
        for prompt_len in [1usize, 15, 16, 17, 31, 32, 33, 40, 47, 48] {
            let pool = BlockPool::new(16, 256).unwrap();
            let mut kv =
                pool.new_request((0..prompt_len as u32).map(|i| 100 + i).collect(), 24, None);
            kv.schedule_prefill(prompt_len, &pool).unwrap();
            assert_eq!(
                kv.step_page_indices(prompt_len).len(),
                prompt_len.div_ceil(16),
                "prefill page row P={prompt_len}"
            );
            assert_eq!(
                kv.prefill_view(prompt_len).num_pages(),
                prompt_len.div_ceil(16),
                "prefill view P={prompt_len}"
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
                let view = kv.decode_view();
                assert_eq!(
                    view.num_pages(),
                    need,
                    "decode view P={prompt_len} step={step}"
                );
                assert_eq!(
                    (view.num_pages() - 1) * 16 + view.last_page_len(),
                    kv.kv_position() + 1,
                    "kernel-derived length P={prompt_len} step={step}"
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
}
