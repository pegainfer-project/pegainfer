//! Expert parallelism: the startup handshake one rank needs, and what makes a
//! group of them safe to run without a coordinator.
//!
//! ## Free-running ranks
//!
//! Every rank is an autonomous engine with its own scheduler thread, its own
//! slots and its own requests. There is no coordinator, no cross-rank host
//! protocol, and nothing to negotiate. The **only** runtime coupling is inside
//! a step, and it is a compile-time constant: every rank launches the same
//! sequence, at the same shapes, on every step it takes. A rank that has
//! nothing to serve takes a padding step — a real step with every row marked
//! padding — rather than skipping one.
//!
//! ## The transport
//!
//! Routed experts go through the fused MegaMoE kernel, which does the whole
//! cross-rank forward itself: it dispatches over NVLink into its peers'
//! symmetric slabs, computes every expert this rank owns for whoever sent it
//! work, and combines each token back into the rank that owns it, pairing the
//! world with its own device-side barriers. The host issues **no collective at
//! all** — a step is one kernel launch per MoE layer on the rank's own stream,
//! and the write-then-launch ordering the inputs need is stream order on that
//! same stream.
//!
//! So this module is small on purpose. All it does is the startup handshake:
//!
//! * every rank publishes its slab's base pointer and device ordinal once its
//!   allocation has been zeroed and synchronised;
//! * every rank blocks on the full table before its first launch, which makes
//!   the rendezvous itself the group's startup barrier — no launch can precede
//!   the last allocation;
//! * every rank confirms it can address each peer's device (the pairs were
//!   opened before the slabs were allocated, because a memory-pool access grant
//!   does not reliably reach allocations that predate it).
//!
//! An idle rank still launches every layer: the kernel serves this rank's
//! experts for its peers' tokens and joins every barrier at zero local tokens.
//! And a failed step is still fatal to the process — a rank that skips a launch
//! leaves its peers inside a barrier nothing will ever satisfy.
//!
//! *History*: this used to be a fixed four-collective-per-layer NCCL chain
//! (allgather the dispatch, run the masked chain over the fleet's batch through
//! an expert window, scatter to entry-major staging, sum all-reduce, combine).
//! It proved the sharding scheme was bitwise-equal to single rank, and the
//! fused kernel inherited both that criterion and the free-running structure
//! above. `tests/ep_mega_oracle.rs` is the living gate. Two constraints died
//! with it: `ep_size x max_batch <= masked_cap` (the fleet's whole batch had to
//! fit one masked tile) and the per-step collective ledger — plain NCCL has no
//! device-side timeout, so a mispaired chain was a silent wrong answer and the
//! ledger was the only detector. The fused kernel's barrier times out at 60 s
//! and asserts, so what remains of the ledger is a launch count on the slab
//! (`K3MegaScratch::begin_step`), kept only so the rank that fell behind names
//! itself instead of leaving its peers to time out anonymously.

use std::sync::Arc;
use std::sync::Condvar;
use std::sync::Mutex;
use std::time::Duration;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use pegainfer_kernels::ops::k3_mega_open_peer_access;

/// How long a rank waits for its peers to publish their slabs before it gives
/// up. Every rank publishes at construction, so this only ever expires when a
/// peer died on the way there.
const SLAB_RENDEZVOUS_TIMEOUT: Duration = Duration::from_secs(300);

/// One rank's contribution to the startup exchange: where its symmetric slab
/// lives and which device that is, so its peers can open access to it.
#[derive(Clone, Copy, Debug)]
struct K3EpSlab {
    base: i64,
    device_ordinal: usize,
}

/// The in-process handshake an EP group's ranks pair through.
///
/// Every rank publishes its symmetric-slab base pointer at construction —
/// after the allocation has been zeroed and synchronised — and reads the full
/// table back on its first step. Waiting for that table IS the startup barrier,
/// so a rank cannot launch before every peer slab exists and is zeroed.
///
/// Deliberately in-process: an EP group is one process with one thread per
/// rank. A multi-node group would replace this with a real out-of-band exchange
/// (and with exported IPC handles rather than bare pointers) and nothing else
/// about the transport would move.
#[derive(Debug)]
pub struct K3EpRendezvous {
    ranks: usize,
    slabs: Mutex<Vec<Option<K3EpSlab>>>,
    ready: Condvar,
}

impl K3EpRendezvous {
    /// A rendezvous for a `ranks`-wide EP group.
    #[must_use]
    pub fn new(ranks: usize) -> Arc<Self> {
        Arc::new(Self {
            ranks,
            slabs: Mutex::new(vec![None; ranks]),
            ready: Condvar::new(),
        })
    }

    pub(crate) fn ranks(&self) -> usize {
        self.ranks
    }

    /// Publish this rank's zeroed slab. Never blocks: the ranks are constructed
    /// one after another on one thread, so a publish that waited for its peers
    /// would deadlock before they exist.
    fn publish_slab(&self, rank: usize, slab: K3EpSlab) -> Result<()> {
        let mut slabs = self.slabs.lock().expect("K3 EP rendezvous poisoned");
        ensure!(rank < self.ranks, "K3 EP rank {rank} is outside the group");
        ensure!(
            slabs[rank].is_none(),
            "K3 EP rank {rank} published its symmetric slab twice"
        );
        slabs[rank] = Some(slab);
        self.ready.notify_all();
        Ok(())
    }

    /// Block until every rank has published, then return the world's slabs in
    /// rank order.
    fn slabs(&self, rank: usize) -> Result<Vec<K3EpSlab>> {
        let mut slabs = self.slabs.lock().expect("K3 EP rendezvous poisoned");
        while slabs.iter().any(Option::is_none) {
            let (guard, timeout) = self
                .ready
                .wait_timeout(slabs, SLAB_RENDEZVOUS_TIMEOUT)
                .expect("K3 EP rendezvous poisoned");
            slabs = guard;
            if timeout.timed_out() && slabs.iter().any(Option::is_none) {
                let missing: Vec<usize> = slabs
                    .iter()
                    .enumerate()
                    .filter_map(|(peer, slab)| slab.is_none().then_some(peer))
                    .collect();
                bail!(
                    "K3 EP rank {rank} waited {}s for its peers' symmetric slabs; ranks \
                     {missing:?} never published",
                    SLAB_RENDEZVOUS_TIMEOUT.as_secs()
                );
            }
        }
        Ok(slabs
            .iter()
            .map(|slab| slab.expect("checked above"))
            .collect())
    }
}

/// One rank of an expert-parallel group.
///
/// It owns no buffers and issues nothing per step. Its whole job is the startup
/// handshake: publish this rank's symmetric slab, and — once, on the stepping
/// thread — collect the world's table and confirm peer access to every other
/// device so the kernel's cross-rank stores land.
pub(crate) struct K3EpRuntime {
    rendezvous: Arc<K3EpRendezvous>,
    rank: usize,
    device_ordinal: usize,
    ready: bool,
}

impl K3EpRuntime {
    /// Publish this rank's slab. The caller must already have synchronised the
    /// allocation's zeroing: a peer that sees this entry is entitled to assume
    /// the memory behind it is live and zeroed.
    pub(crate) fn new(
        rendezvous: Arc<K3EpRendezvous>,
        rank: usize,
        base: i64,
        device_ordinal: usize,
    ) -> Result<Self> {
        let ranks = rendezvous.ranks();
        ensure!(
            ranks > 1 && rank < ranks,
            "K3 EP rank {rank} is not part of a {ranks}-rank group"
        );
        rendezvous.publish_slab(
            rank,
            K3EpSlab {
                base,
                device_ordinal,
            },
        )?;
        Ok(Self {
            rendezvous,
            rank,
            device_ordinal,
            ready: false,
        })
    }

    /// Resolve the world's base-pointer table, exactly once. Returns the table
    /// the first time through and `None` afterwards.
    ///
    /// This blocks until every peer has published, which is the group's startup
    /// barrier: a rank publishes only after its slab is allocated, zeroed and
    /// synchronised, so no launch can precede the last allocation.
    ///
    /// The device pairs were opened before the slabs were allocated (the
    /// memory-pool grant has to precede the allocation it covers), so all that
    /// is left here is confirming that the ranks the group actually contains
    /// are among the ones this rank opened — the call is idempotent, and it
    /// fails with the peer's ordinal in the message if that device turned out
    /// to be unreachable.
    pub(crate) fn ensure_ready(&mut self) -> Result<Option<Vec<i64>>> {
        if self.ready {
            return Ok(None);
        }
        let slabs = self.rendezvous.slabs(self.rank)?;
        for peer in &slabs {
            k3_mega_open_peer_access(self.device_ordinal, peer.device_ordinal).with_context(
                || {
                    format!(
                        "K3 EP rank {} cannot address rank {}'s slab",
                        self.rank, peer.device_ordinal
                    )
                },
            )?;
        }
        self.ready = true;
        log::info!(
            "K3 EP rank {} paired with {} ranks over peer access (devices {:?})",
            self.rank,
            slabs.len(),
            slabs.iter().map(|s| s.device_ordinal).collect::<Vec<_>>()
        );
        Ok(Some(slabs.iter().map(|slab| slab.base).collect()))
    }
}

/// A step that fails under expert parallelism has already left the group out of
/// phase: this rank owed its peers one kernel launch per MoE layer and did not
/// make it, so every peer is now inside a device barrier this rank will never
/// reach. There is nothing to recover, so the rank takes the process down
/// instead of returning into the scheduler's fail-the-batch-and-keep-serving
/// path (which stays, for single-rank).
// Exiting the process is the point, not a shortcut: there is no state from
// which this group can serve a correct next token.
#[allow(clippy::exit)]
pub(crate) fn ep_fatal(rank: usize, phase: &str, error: &anyhow::Error) -> ! {
    let reason = format!(
        "K3 EP rank {rank} failed during {phase}: {error:#}. The EP group cannot recover from a \
         missed step — every peer is waiting on a launch that will not come — so this process is \
         exiting."
    );
    log::error!("{reason}");
    // The log goes nowhere when nobody installed a logger, and this call takes
    // the process down: a fatal that leaves no trace is worse than noisy.
    eprintln!("{reason}");
    std::process::exit(1);
}
