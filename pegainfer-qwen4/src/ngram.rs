//! Host-side n-gram addressing for the per-layer-embedding (PLE) block.
//!
//! The PLE block injects a hashed n-gram embedding into one decoder layer. Its
//! first half is pure integer arithmetic: turn a token sequence into one table
//! row per (position, head). That half lives here, on the host, in a crate with
//! no CUDA dependency — the second half, the 95 GiB table gather those rows
//! address, is a device concern and consumes what this module produces.
//!
//! The semantics are `Qwen4ExpTextNGramEmbedding.forward`:
//!
//! 1. **An EOS-padded history.** A whole-sequence forward has no cache state, so
//!    the history is [`NgramHasher::context`] copies of `eos_token_id` followed
//!    by the sequence. The padding is what makes the first rows' windows legal
//!    rather than truncated, and it is why the reference drops those warm-up
//!    rows from its output.
//! 2. **An EOS-resetting context window** (`_shift_right_ignore_eos`). For a
//!    window `shift` positions wide, position `p` reads `history[p - shift]` —
//!    unless that read would reach across an EOS boundary or off the front of
//!    the history, in which case it reads `eos_token_id` instead. The boundary
//!    is the cummax over EOS positions: an n-gram never spans two segments.
//! 3. **A mul-xor fold per (row, order).** Each window position's token is
//!    multiplied by that position's `layer_multipliers` entry and the products
//!    are XOR-folded. One fold per order, so an order-`k` n-gram folds the
//!    `k` widest windows down to the narrowest.
//! 4. **A per-head prime remainder plus a per-head offset.** Each head reduces
//!    its order's fold modulo its own prime vocabulary size and adds its own row
//!    offset into the flat table, so the heads occupy disjoint row ranges and
//!    one id names one (head, row) pair.
//!
//! Heads are laid out **(order, head), order-major**: the `heads_per_ngram`
//! bigram heads first, then the trigram heads, and so on up to `ngram_size`.
//! The table index a head's remainder is taken against is therefore the same
//! number as the column that head's id lands in, which is what makes the layout
//! worth stating explicitly — a transposed (head, order) layout keeps every id
//! in range and produces a completely different embedding.
//!
//! Every id this module returns is a row index into the flat n-gram table whose
//! geometry the config derives: `heads * head_dim` wide rows, `sum(head_vocab_sizes)`
//! of them before padding. A gather turns `ids` into `rows x heads x head_dim`
//! activation, and nothing here needs the table itself to exist.
//!
//! Pinned by `pegainfer-qwen4/tests/ple_ngram_replay.rs` against a golden
//! recorded from the reference at the frozen checkpoint's real geometry: the ids
//! are compared with i64 equality at zero tolerance, so a wrong window, fold,
//! modulus, offset or layout fails rather than drifting.

use anyhow::Result;
use anyhow::ensure;

/// The n-gram hash and its inputs.
///
/// Everything here is checkpoint data, not geometry: the reference derives the
/// per-head primes and their offsets from config (`_find_nth_prime_after` over
/// `ngram_vocab_size_base`) and also stores them, so a loader can build this
/// from either and cross-check. Constructing one validates the inputs, because
/// a bad modulus is a panic in `rem_euclid` and a short multiplier run is a
/// silent read past the window.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NgramHasher {
    eos_token_id: i64,
    /// One odd multiplier per context position, `ngram_size` of them. Position
    /// 0 is the token at the hashed row itself, so the fold always uses
    /// `multipliers[0]` and an order-`k` n-gram adds `k - 1` more.
    layer_multipliers: Vec<i64>,
    /// Per-head prime modulus, `(ngram_size - 1) * heads_per_ngram` of them.
    head_vocab_sizes: Vec<i64>,
    /// Per-head row offset into the flat table, same length and laid out the
    /// same way.
    head_offsets: Vec<i64>,
    heads_per_ngram: usize,
}

impl NgramHasher {
    /// Build a hasher from the checkpoint's own vectors.
    ///
    /// `layer_multipliers` must carry one entry per context position
    /// (`ngram_size` of them, at least 2 — an n-gram needs a context token),
    /// and both per-head vectors one entry per head. The reference generates its
    /// multipliers as positive odd integers bounded by `(2^63 - 1) /
    /// vocab_size`, which is what keeps every `token * multiplier` product and
    /// their XOR fold non-negative; positivity is required here so that
    /// `rem_euclid` and a truncated remainder cannot disagree on a fold that a
    /// different checkpoint produced some other way.
    ///
    /// # Errors
    ///
    /// If the vectors are not the lengths the head layout implies, if any
    /// multiplier is not positive, if any head modulus is not at least 2 (a
    /// modulus of 1 hashes every token to one row, and 0 does not divide), if any
    /// offset is negative, or if a head's `offset + modulus` overflows i64 — the
    /// table row count is a sum of these, so an overflowing range is a table that
    /// cannot be sized. Contiguity of the offsets is *not* checked here: that is a
    /// property of the table the config derives, not of the hash.
    pub fn new(
        eos_token_id: i64,
        layer_multipliers: Vec<i64>,
        head_vocab_sizes: Vec<i64>,
        head_offsets: Vec<i64>,
        heads_per_ngram: usize,
    ) -> Result<Self> {
        ensure!(
            heads_per_ngram > 0,
            "qwen4_exp n-gram: heads_per_ngram must be positive"
        );
        ensure!(
            layer_multipliers.len() >= 2,
            "qwen4_exp n-gram: layer_multipliers must carry one entry per context position \
             (ngram_size, at least 2), got {}",
            layer_multipliers.len()
        );
        let orders = layer_multipliers.len() - 1;
        let heads = orders * heads_per_ngram;
        ensure!(
            head_vocab_sizes.len() == heads && head_offsets.len() == heads,
            "qwen4_exp n-gram: {orders} orders x {heads_per_ngram} heads needs {heads} entries in \
             both per-head vectors, got {} moduli and {} offsets",
            head_vocab_sizes.len(),
            head_offsets.len()
        );
        for (position, multiplier) in layer_multipliers.iter().enumerate() {
            ensure!(
                *multiplier > 0,
                "qwen4_exp n-gram: layer_multipliers[{position}] must be positive, got \
                 {multiplier}"
            );
        }
        for head in 0..heads {
            let size = head_vocab_sizes[head];
            let offset = head_offsets[head];
            ensure!(
                size >= 2,
                "qwen4_exp n-gram: head {head} has modulus {size}, which cannot address a table"
            );
            ensure!(
                offset >= 0,
                "qwen4_exp n-gram: head {head} has a negative table offset {offset}"
            );
            ensure!(
                offset.checked_add(size).is_some(),
                "qwen4_exp n-gram: head {head}'s range {offset}..+{size} overflows i64"
            );
        }
        Ok(Self {
            eos_token_id,
            layer_multipliers,
            head_vocab_sizes,
            head_offsets,
            heads_per_ngram,
        })
    }

    /// The token that both pads the warm-up rows and closes a segment.
    pub fn eos_token_id(&self) -> i64 {
        self.eos_token_id
    }

    /// `ngram_size - 1`: the number of order blocks in a row of
    /// [`ids`](Self::ids), and — the same number, which is why
    /// [`context`](Self::context) exists rather than a second derivation — of
    /// warm-up rows the history is padded with.
    pub fn orders(&self) -> usize {
        self.layer_multipliers.len() - 1
    }

    /// Heads per order block.
    pub fn heads_per_ngram(&self) -> usize {
        self.heads_per_ngram
    }

    /// Total heads, and the width of one row of [`NgramHasher::ids`].
    pub fn heads(&self) -> usize {
        self.orders() * self.heads_per_ngram
    }

    /// Warm-up rows the history is padded with: one per context token an
    /// n-gram's window can reach back over. Always [`orders`](Self::orders).
    pub fn context(&self) -> usize {
        self.orders()
    }

    /// The order an output column's block carries: slot 0 is the bigram, slot 1
    /// the trigram, and so on.
    pub fn order_of_slot(&self, slot: usize) -> usize {
        slot + 2
    }

    /// The table head an (order, head) pair's modulus and offset belong to.
    ///
    /// Under the reference's order-major layout this is also the *output column*
    /// the id lands in, `slot * heads_per_ngram + head` for `slot = order - 2`.
    /// That identity is worth stating because it is what makes a transposed
    /// (head, order) layout silently wrong: every id still names a real table
    /// row, so nothing goes out of range and only a value comparison catches it.
    pub fn table_head_of(&self, order: usize, head: usize) -> usize {
        (order - 2) * self.heads_per_ngram + head
    }

    /// The history a whole-sequence forward hashes: [`context`](Self::context)
    /// EOS rows, then the sequence.
    ///
    /// A decode step instead carries its own `ngram_size - 1` token window as
    /// state and hands that in; this is the no-state case, which is what the
    /// reference's prefill and what the golden record.
    pub fn padded_history(&self, input_ids: &[i64]) -> Vec<i64> {
        let mut history = vec![self.eos_token_id; self.context()];
        history.extend_from_slice(input_ids);
        history
    }

    /// For every position of `history`, the first position of the EOS-delimited
    /// segment that contains it.
    ///
    /// This is the cummax over EOS positions that `_shift_right_ignore_eos`
    /// performs: `segment_starts[p]` is one past the last EOS *strictly before*
    /// `p`. An EOS token starts the **next** segment, it does not belong to the
    /// one it closes, which is why the search excludes `p` itself. Position 0 is
    /// always the start of its own segment.
    ///
    /// It is returned as data rather than folded into
    /// [`shifted_context`](Self::shifted_context) because it is the piece a
    /// resumed decode has to carry: a window that reaches back over a segment
    /// boundary has to be reset, and the boundary is a property of the history,
    /// not of the shift.
    pub fn segment_starts(&self, history: &[i64]) -> Vec<usize> {
        let mut starts = vec![0usize; history.len()];
        let mut segment_start = 0usize;
        for (p, &token) in history.iter().enumerate() {
            starts[p] = segment_start;
            if token == self.eos_token_id {
                segment_start = p + 1;
            }
        }
        starts
    }

    /// The reference's `_shift_right_ignore_eos` on one history: the token
    /// `shift` positions back, or `eos_token_id` where the segment does not
    /// reach that far.
    ///
    /// `segment_starts` is [`segment_starts`](Self::segment_starts) of the same
    /// history. Position `p` reads `history[p - shift]` when the read stays
    /// inside `p`'s segment (`p - segment_starts[p] >= shift`) and inside the
    /// history (`p >= shift`), and the padding token otherwise. `shift` 0 is the
    /// identity, so the windows for orders 2..=`ngram_size` are shifts
    /// 0..`ngram_size`.
    ///
    /// # Panics
    ///
    /// If `segment_starts` is shorter than `history`.
    pub fn shifted_context(
        &self,
        history: &[i64],
        shift: usize,
        segment_starts: &[usize],
    ) -> Vec<i64> {
        assert!(
            segment_starts.len() >= history.len(),
            "a shifted context needs one segment start per history position: {} for {}",
            segment_starts.len(),
            history.len()
        );
        if shift == 0 {
            return history.to_vec();
        }
        (0..history.len())
            .map(|p| {
                let in_segment = p - segment_starts[p];
                if in_segment >= shift && p >= shift {
                    history[p - shift]
                } else {
                    self.eos_token_id
                }
            })
            .collect()
    }

    /// Every window a whole-sequence forward needs, indexed by shift: entry `k`
    /// is [`shifted_context`](Self::shifted_context) at shift `k`, for `k` in
    /// 0..`ngram_size`.
    pub fn windows(&self, history: &[i64], segment_starts: &[usize]) -> Vec<Vec<i64>> {
        (0..self.layer_multipliers.len())
            .map(|shift| self.shifted_context(history, shift, segment_starts))
            .collect()
    }

    /// One (row, order) mul-XOR fold: `windows[0][position] * multipliers[0]`,
    /// XOR-ed with `windows[k][position] * multipliers[k]` for `1 <= k < order`.
    ///
    /// `position` is an index into the *history*, not into the output rows: row
    /// `r` of a whole-sequence forward is at `r + `[`context`](Self::context).
    /// The multiply wraps, which is faithful rather than load-bearing — the
    /// reference bounds each multiplier so the products cannot overflow, and an
    /// XOR of non-negative i64s keeps the sign bit clear, so the fold is never
    /// negative for a hasher that passed [`new`](Self::new).
    ///
    /// # Panics
    ///
    /// If `order` exceeds the window count, or `position` exceeds the windows'
    /// length.
    pub fn fold(&self, windows: &[&[i64]], position: usize, order: usize) -> i64 {
        assert!(
            order >= 1 && order <= windows.len(),
            "an order-{order} fold needs {order} windows, got {}",
            windows.len()
        );
        let mut mixed = windows[0][position].wrapping_mul(self.layer_multipliers[0]);
        for (window, multiplier) in windows[1..order]
            .iter()
            .zip(&self.layer_multipliers[1..order])
        {
            mixed ^= window[position].wrapping_mul(*multiplier);
        }
        mixed
    }

    /// One head's table row for a fold: the fold modulo that head's prime, plus
    /// that head's offset.
    ///
    /// `rem_euclid`, not `%`: the reference uses `torch.remainder`, a floored
    /// remainder, which only differs from Rust's truncated one on a negative
    /// fold — and a hasher that passed [`new`](Self::new) cannot produce one.
    /// Using the Euclidean form keeps the two equal by construction rather than
    /// by the accident of the multiplier bound.
    ///
    /// # Panics
    ///
    /// If `table_head` is out of range.
    pub fn head_id(&self, fold: i64, table_head: usize) -> i64 {
        fold.rem_euclid(self.head_vocab_sizes[table_head]) + self.head_offsets[table_head]
    }

    /// Hash a whole sequence: one row of [`heads`](Self::heads) table indices per
    /// input token, row-major.
    ///
    /// Column `slot * heads_per_ngram + head` holds the id of the head at
    /// (`order_of_slot(slot)`, `head`), so the bigram block occupies the first
    /// `heads_per_ngram` columns and the trigram block the next. The warm-up
    /// rows are dropped: the output has one row per `input_ids` entry, read from
    /// history position `row + `[`context`](Self::context).
    pub fn ids(&self, input_ids: &[i64]) -> Vec<i64> {
        let history = self.padded_history(input_ids);
        let starts = self.segment_starts(&history);
        let owned = self.windows(&history, &starts);
        let windows: Vec<&[i64]> = owned.iter().map(Vec::as_slice).collect();
        let total = self.heads();
        let mut ids = vec![0i64; input_ids.len() * total];
        for slot in 0..self.orders() {
            let order = self.order_of_slot(slot);
            for (row, id_row) in ids.chunks_mut(total).enumerate() {
                let mixed = self.fold(&windows, row + self.context(), order);
                for head in 0..self.heads_per_ngram {
                    id_row[slot * self.heads_per_ngram + head] =
                        self.head_id(mixed, self.table_head_of(order, head));
                }
            }
        }
        ids
    }
}
