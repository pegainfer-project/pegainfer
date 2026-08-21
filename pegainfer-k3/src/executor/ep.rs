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
//! So this module is small on purpose. All it does is the startup handshake,
//! in one of two shapes:
//!
//! * **In-process** (one machine, one process, one thread per rank): every
//!   rank publishes its slab's base pointer and device ordinal, blocks on the
//!   full table before its first launch, and confirms peer access to each
//!   peer's device. The rendezvous itself is the group's startup barrier.
//! * **Fleet** (a `--k3-ranks start..end` slice of a wider world): the slabs
//!   are `CU_MEM_HANDLE_TYPE_FABRIC` allocations, and what travels between
//!   processes is each rank's 64-byte fabric handle, exchanged once over a
//!   minimal TCP bootstrap (mirroring GLM5.2's `rendezvous.rs`): the process
//!   hosting rank 0 binds `--k3-rendezvous` and collects every process's
//!   handles, then serves the world's table back; every process imports the
//!   handles it does not host and maps them into its own address space. After
//!   that one exchange the engines never talk again — the NVLink domain spans
//!   the rack (NVL72 + IMEX), so the kernel's cross-rank stores work exactly
//!   as they do in-process.
//!
//! An idle rank still launches every layer: the kernel serves this rank's
//! experts for its peers' tokens and joins every barrier at zero local tokens.
//! And a failed step is still fatal to the process — a rank that skips a launch
//! leaves its peers inside a barrier nothing will ever satisfy. That
//! fail-stop is also the fleet's restart story: handles do not survive a
//! process, so a lost rank means the whole fleet relaunches together.
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

use std::io::Read as _;
use std::io::Write as _;
use std::net::TcpListener;
use std::net::TcpStream;
use std::ops::Range;
use std::sync::Arc;
use std::sync::Condvar;
use std::sync::Mutex;
use std::time::Duration;
use std::time::Instant;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use pegainfer_kernels::ops::K3_MEGA_FABRIC_HANDLE_BYTES;
use pegainfer_kernels::ops::k3_mega_fabric_slab_import;
use pegainfer_kernels::ops::k3_mega_open_peer_access;

/// How long a rank waits for its peers to publish their slabs before it gives
/// up. Every rank publishes at construction, so this only ever expires when a
/// peer died on the way there.
const SLAB_RENDEZVOUS_TIMEOUT: Duration = Duration::from_secs(300);

/// Bootstrap wire version: bump when the handshake shape changes.
const BOOTSTRAP_VERSION: u32 = 1;
/// `"K3EP"`, so a stray connection to the wrong port fails loudly.
const BOOTSTRAP_MAGIC: u32 = u32::from_le_bytes(*b"K3EP");
/// The rank-0 process binds the bootstrap address on its first step, which is
/// after its own (possibly 1.5 TB) weight load; peers retry for a window that
/// survives that.
const BOOTSTRAP_CONNECT_RETRY: Duration = Duration::from_secs(2);
const BOOTSTRAP_CONNECT_TIMEOUT: Duration = Duration::from_secs(3600);
/// A peer's table reply lands only when the SLOWEST process has loaded and
/// checked in, so the read timeout is a fleet-load bound, not an IO bound.
const BOOTSTRAP_REPLY_TIMEOUT: Duration = Duration::from_secs(3600);
/// Writing a hello is genuinely just IO.
const BOOTSTRAP_IO_TIMEOUT: Duration = Duration::from_secs(30);

/// One rank's contribution to the startup exchange. `base` is only
/// dereferenceable inside the publishing process; `fabric` is what makes the
/// slab reachable from any other process in the NVLink domain.
#[derive(Clone, Copy, Debug)]
struct K3EpSlab {
    base: i64,
    device_ordinal: usize,
    /// Present exactly in fleet mode: the slab's export handle and its
    /// pre-rounding byte size, the pair a peer process needs to import it.
    fabric: Option<K3FabricSlab>,
}

/// A fabric-exportable slab's wire identity.
#[derive(Clone, Copy, Debug)]
pub(crate) struct K3FabricSlab {
    pub(crate) handle: [u8; K3_MEGA_FABRIC_HANDLE_BYTES],
    pub(crate) num_bytes: usize,
}

/// What the fleet bootstrap exchanges per rank.
#[derive(Clone, Copy, Debug)]
struct WireSlab {
    num_bytes: u64,
    handle: [u8; K3_MEGA_FABRIC_HANDLE_BYTES],
}

/// The fleet exchange's lifecycle, driven by the first local rank to step.
enum FleetState {
    NotStarted,
    InProgress,
    Ready(Arc<Vec<i64>>),
    Failed(String),
}

/// The handshake an EP group's ranks pair through.
///
/// In-process ([`K3EpRendezvous::new`]): the table covers the whole world and
/// every rank publishes into it directly. Fleet ([`K3EpRendezvous::fleet`]):
/// the table covers this process's `--k3-ranks` slice; the rest of the world
/// arrives as fabric handles over the TCP bootstrap and is imported once,
/// process-wide.
///
/// Waiting for the completed table IS the startup barrier, so a rank cannot
/// launch before every peer slab exists and is zeroed.
pub struct K3EpRendezvous {
    ep_size: usize,
    /// The global ranks this process hosts. `0..ep_size` in-process.
    local_ranks: Range<usize>,
    /// One slot per LOCAL rank.
    slabs: Mutex<Vec<Option<K3EpSlab>>>,
    ready: Condvar,
    /// Present exactly in fleet mode.
    fleet: Option<FleetBootstrap>,
}

struct FleetBootstrap {
    /// `host:port` the rank-0 process binds and everyone else connects to.
    addr: String,
    state: Mutex<FleetState>,
    done: Condvar,
}

impl std::fmt::Debug for K3EpRendezvous {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("K3EpRendezvous")
            .field("ep_size", &self.ep_size)
            .field("local_ranks", &self.local_ranks)
            .field("fleet", &self.fleet.as_ref().map(|fleet| &fleet.addr))
            .finish_non_exhaustive()
    }
}

impl K3EpRendezvous {
    /// An in-process rendezvous for a `ranks`-wide EP group: one process, one
    /// thread per rank, bare pointers.
    #[must_use]
    pub fn new(ranks: usize) -> Arc<Self> {
        Arc::new(Self {
            ep_size: ranks,
            local_ranks: 0..ranks,
            slabs: Mutex::new(vec![None; ranks]),
            ready: Condvar::new(),
            fleet: None,
        })
    }

    /// A fleet rendezvous: this process hosts `local_ranks` of an
    /// `ep_size`-wide world, slabs are fabric allocations, and `addr` is the
    /// bootstrap the world exchanges handles through (bound by the process
    /// hosting rank 0).
    pub fn fleet(ep_size: usize, local_ranks: Range<usize>, addr: String) -> Result<Arc<Self>> {
        ensure!(
            !local_ranks.is_empty() && local_ranks.end <= ep_size,
            "K3 fleet ranks {local_ranks:?} do not fit an ep_size of {ep_size}"
        );
        ensure!(
            local_ranks.len() < ep_size,
            "K3 fleet ranks {local_ranks:?} cover the whole {ep_size}-rank world; use the \
             in-process rendezvous"
        );
        Ok(Arc::new(Self {
            ep_size,
            local_ranks: local_ranks.clone(),
            slabs: Mutex::new(vec![None; local_ranks.len()]),
            ready: Condvar::new(),
            fleet: Some(FleetBootstrap {
                addr,
                state: Mutex::new(FleetState::NotStarted),
                done: Condvar::new(),
            }),
        }))
    }

    /// The EP world size.
    pub(crate) fn ranks(&self) -> usize {
        self.ep_size
    }

    /// Whether this group's slabs must be fabric-exportable (multi-process).
    pub(crate) fn is_fleet(&self) -> bool {
        self.fleet.is_some()
    }

    /// Publish this rank's slab. Never blocks: the ranks are constructed
    /// one after another on one thread, so a publish that waited for its peers
    /// would deadlock before they exist.
    fn publish_slab(&self, rank: usize, slab: K3EpSlab) -> Result<()> {
        ensure!(
            self.local_ranks.contains(&rank),
            "K3 EP rank {rank} is outside this process's ranks {:?}",
            self.local_ranks
        );
        ensure!(
            self.fleet.is_none() || slab.fabric.is_some(),
            "K3 EP rank {rank} is part of a fleet but published a slab without a fabric handle"
        );
        let index = rank - self.local_ranks.start;
        let mut slabs = self.slabs.lock().expect("K3 EP rendezvous poisoned");
        ensure!(
            slabs[index].is_none(),
            "K3 EP rank {rank} published its symmetric slab twice"
        );
        slabs[index] = Some(slab);
        self.ready.notify_all();
        Ok(())
    }

    /// Block until every LOCAL rank has published, then return their slabs in
    /// local order.
    fn local_slabs(&self, rank: usize) -> Result<Vec<K3EpSlab>> {
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
                    .filter_map(|(index, slab)| {
                        slab.is_none().then_some(self.local_ranks.start + index)
                    })
                    .collect();
                bail!(
                    "K3 EP rank {rank} waited {}s for its process-local peers' symmetric slabs; \
                     ranks {missing:?} never published",
                    SLAB_RENDEZVOUS_TIMEOUT.as_secs()
                );
            }
        }
        Ok(slabs
            .iter()
            .map(|slab| slab.expect("checked above"))
            .collect())
    }

    /// The world's base-pointer table as addressed from this process, fleet
    /// mode. The first caller runs the whole exchange (serve or fetch, then
    /// import); everyone else blocks on its outcome. Any failure is terminal
    /// for every rank — there is no world without the table.
    fn fleet_table(&self, rank: usize, device_ordinal: usize) -> Result<Arc<Vec<i64>>> {
        let bootstrap = self
            .fleet
            .as_ref()
            .expect("fleet_table is only called in fleet mode");
        {
            let mut state = bootstrap.state.lock().expect("K3 EP bootstrap poisoned");
            loop {
                match &*state {
                    FleetState::Ready(table) => return Ok(table.clone()),
                    FleetState::Failed(message) => {
                        bail!("K3 EP bootstrap already failed: {message}")
                    }
                    FleetState::InProgress => {
                        state = bootstrap
                            .done
                            .wait(state)
                            .expect("K3 EP bootstrap poisoned");
                    }
                    FleetState::NotStarted => {
                        *state = FleetState::InProgress;
                        break;
                    }
                }
            }
        }
        let outcome = self.run_fleet_exchange(rank, device_ordinal);
        let mut state = bootstrap.state.lock().expect("K3 EP bootstrap poisoned");
        match outcome {
            Ok(table) => {
                let table = Arc::new(table);
                *state = FleetState::Ready(table.clone());
                bootstrap.done.notify_all();
                Ok(table)
            }
            Err(error) => {
                *state = FleetState::Failed(format!("{error:#}"));
                bootstrap.done.notify_all();
                Err(error)
            }
        }
    }

    /// The exchange itself: local slabs -> wire records -> serve or fetch the
    /// world's table -> import every remote handle.
    fn run_fleet_exchange(&self, rank: usize, device_ordinal: usize) -> Result<Vec<i64>> {
        let bootstrap = self.fleet.as_ref().expect("fleet mode");
        let local = self.local_slabs(rank)?;
        let local_wire: Vec<WireSlab> = local
            .iter()
            .map(|slab| {
                let fabric = slab.fabric.expect("publish_slab enforced the handle");
                WireSlab {
                    num_bytes: fabric.num_bytes as u64,
                    handle: fabric.handle,
                }
            })
            .collect();

        let world: Vec<WireSlab> = if self.local_ranks.start == 0 {
            serve_bootstrap(
                &bootstrap.addr,
                self.ep_size,
                self.local_ranks.clone(),
                &local_wire,
            )
            .with_context(|| format!("K3 EP bootstrap serve on {}", bootstrap.addr))?
        } else {
            fetch_bootstrap(
                &bootstrap.addr,
                self.ep_size,
                self.local_ranks.clone(),
                &local_wire,
            )
            .with_context(|| format!("K3 EP bootstrap fetch from {}", bootstrap.addr))?
        };

        // Every rank derives its slab size from the same layout arithmetic, so
        // a size that differs is a mixed-build fleet — refuse it before the
        // kernel walks a wrong layout.
        let expected = local_wire[0].num_bytes;
        for (peer, slab) in world.iter().enumerate() {
            ensure!(
                slab.num_bytes == expected,
                "K3 EP rank {peer} published a {}-byte slab, this process allocates {expected}; \
                 the fleet is not running one build",
                slab.num_bytes
            );
        }

        let mut table = vec![0i64; self.ep_size];
        for (peer, wire) in world.iter().enumerate() {
            if self.local_ranks.contains(&peer) {
                table[peer] = local[peer - self.local_ranks.start].base;
            } else {
                table[peer] = k3_mega_fabric_slab_import(
                    &wire.handle,
                    usize::try_from(wire.num_bytes)?,
                    device_ordinal,
                )
                .with_context(|| format!("import K3 EP rank {peer}'s fabric slab"))?;
            }
        }
        log::info!(
            "K3 EP bootstrap complete: {} ranks paired over the NVLink fabric (this process \
             hosts {:?})",
            self.ep_size,
            self.local_ranks
        );
        Ok(table)
    }
}

// ── The TCP bootstrap ───────────────────────────────────────────────────
//
// Mirrors GLM5.2's rendezvous: the rank-0 process binds and serves, everyone
// else connects once, and the exchange is the process's whole cross-node
// control plane. The difference is the payload — every rank's fabric handle
// travels, not one shared id — so the root must COLLECT before it can answer,
// and every connection stays open until the world is complete.

/// Root side: bind `addr`, accept until every non-local rank has checked in,
/// then answer every held connection with the world's table. Returns that
/// table. The listener then keeps serving the completed table until the
/// process dies — but a fresh hello for an already-filled slot is refused,
/// because its handles cannot match the ones the world already imported
/// (restarts are fleet-wide by design).
fn serve_bootstrap(
    addr: &str,
    ep_size: usize,
    local_ranks: Range<usize>,
    local: &[WireSlab],
) -> Result<Vec<WireSlab>> {
    // Bind the port on every interface rather than whatever the hostname in
    // `addr` resolves to locally: /etc/hosts commonly maps a machine's own
    // name to 127.0.1.1, which would put the listener on loopback while the
    // peers dial the fabric address — connection refused until their 1 h
    // timeout, with nothing in the root's log.
    let port = addr
        .rsplit_once(':')
        .map(|(_, port)| port)
        .with_context(|| format!("K3 EP rendezvous address {addr} carries no port"))?;
    let listener = TcpListener::bind(("0.0.0.0", port.parse::<u16>()?))
        .with_context(|| format!("bind 0.0.0.0:{port} (rendezvous {addr})"))?;
    log::info!("K3 EP bootstrap root listening on 0.0.0.0:{port} for {ep_size} ranks");
    let mut world: Vec<Option<WireSlab>> = vec![None; ep_size];
    for (index, slab) in local.iter().enumerate() {
        world[local_ranks.start + index] = Some(*slab);
    }
    let mut held: Vec<TcpStream> = Vec::new();
    while world.iter().any(Option::is_none) {
        let (mut stream, peer) = listener.accept().context("accept")?;
        stream.set_read_timeout(Some(BOOTSTRAP_IO_TIMEOUT))?;
        stream.set_write_timeout(Some(BOOTSTRAP_IO_TIMEOUT))?;
        match read_hello(&mut stream, ep_size) {
            Ok((ranks, slabs)) => {
                let occupied: Vec<usize> = ranks
                    .clone()
                    .filter(|rank| world[*rank].is_some())
                    .collect();
                if !occupied.is_empty() {
                    let message = format!(
                        "ranks {occupied:?} already checked in; a K3 fleet restarts whole, not \
                         one process at a time"
                    );
                    let _ = write_reply_error(&mut stream, &message);
                    bail!("peer {peer}: {message}");
                }
                log::info!("K3 EP bootstrap: peer {peer} checked in ranks {ranks:?}");
                for (offset, slab) in slabs.into_iter().enumerate() {
                    world[ranks.start + offset] = Some(slab);
                }
                held.push(stream);
            }
            Err(error) => {
                log::warn!("K3 EP bootstrap: rejected a connection from {peer}: {error:#}");
                let _ = write_reply_error(&mut stream, &format!("{error:#}"));
            }
        }
    }
    let table: Vec<WireSlab> = world
        .into_iter()
        .map(|slab| slab.expect("loop ran until complete"))
        .collect();
    for mut stream in held {
        write_reply_table(&mut stream, &table).context("answer a held peer")?;
    }
    // Keep answering (completed-table) fetches until the process dies, so a
    // peer that connects moments after completion still gets an answer — an
    // error answer, since its slot is filled, which is the honest one.
    let table_for_thread = table.clone();
    let ep_size_for_thread = ep_size;
    std::thread::Builder::new()
        .name("k3-ep-bootstrap".into())
        .spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { continue };
                let _ = stream.set_read_timeout(Some(BOOTSTRAP_IO_TIMEOUT));
                let _ = stream.set_write_timeout(Some(BOOTSTRAP_IO_TIMEOUT));
                let _ = match read_hello(&mut stream, ep_size_for_thread) {
                    Ok((ranks, _)) => write_reply_error(
                        &mut stream,
                        &format!(
                            "ranks {ranks:?} arrived after the fleet paired; restart the whole \
                             fleet"
                        ),
                    ),
                    Err(error) => write_reply_error(&mut stream, &format!("{error:#}")),
                };
            }
            drop(table_for_thread);
        })
        .context("spawn the K3 EP bootstrap thread")?;
    Ok(table)
}

/// Peer side: connect (with retry — the root may still be loading weights),
/// send this process's hello, and block until the root answers with the
/// world's table.
fn fetch_bootstrap(
    addr: &str,
    ep_size: usize,
    local_ranks: Range<usize>,
    local: &[WireSlab],
) -> Result<Vec<WireSlab>> {
    let started = Instant::now();
    let mut stream = loop {
        match TcpStream::connect(addr) {
            Ok(stream) => break stream,
            Err(error) => {
                ensure!(
                    started.elapsed() < BOOTSTRAP_CONNECT_TIMEOUT,
                    "K3 EP bootstrap {addr} unreachable after {}s: {error}",
                    BOOTSTRAP_CONNECT_TIMEOUT.as_secs()
                );
                log::info!("K3 EP bootstrap {addr} not ready ({error}); retrying");
                std::thread::sleep(BOOTSTRAP_CONNECT_RETRY);
            }
        }
    };
    stream.set_write_timeout(Some(BOOTSTRAP_IO_TIMEOUT))?;
    // The reply lands when the SLOWEST process checks in; that is a weight
    // load, not an IO hiccup.
    stream.set_read_timeout(Some(BOOTSTRAP_REPLY_TIMEOUT))?;
    write_hello(&mut stream, ep_size, local_ranks.clone(), local)?;
    let table = read_reply(&mut stream, ep_size)?;
    log::info!(
        "K3 EP bootstrap: fetched the {ep_size}-rank table from {addr} (ranks {local_ranks:?})"
    );
    Ok(table)
}

fn write_hello(
    stream: &mut TcpStream,
    ep_size: usize,
    ranks: Range<usize>,
    slabs: &[WireSlab],
) -> Result<()> {
    let mut hello = Vec::with_capacity(20 + slabs.len() * (8 + K3_MEGA_FABRIC_HANDLE_BYTES));
    hello.extend_from_slice(&BOOTSTRAP_MAGIC.to_le_bytes());
    hello.extend_from_slice(&BOOTSTRAP_VERSION.to_le_bytes());
    hello.extend_from_slice(&(ep_size as u32).to_le_bytes());
    hello.extend_from_slice(&(ranks.start as u32).to_le_bytes());
    hello.extend_from_slice(&(ranks.end as u32).to_le_bytes());
    for slab in slabs {
        hello.extend_from_slice(&slab.num_bytes.to_le_bytes());
        hello.extend_from_slice(&slab.handle);
    }
    stream.write_all(&hello).context("send the hello")
}

fn read_hello(stream: &mut TcpStream, ep_size: usize) -> Result<(Range<usize>, Vec<WireSlab>)> {
    let mut head = [0u8; 20];
    stream.read_exact(&mut head).context("read the hello")?;
    let word = |bytes: &[u8]| u32::from_le_bytes(bytes.try_into().expect("4-byte word"));
    let (magic, version, peer_ep) = (word(&head[0..4]), word(&head[4..8]), word(&head[8..12]));
    ensure!(magic == BOOTSTRAP_MAGIC, "not a K3 EP bootstrap hello");
    ensure!(
        version == BOOTSTRAP_VERSION && peer_ep as usize == ep_size,
        "bootstrap mismatch: peer version={version} ep_size={peer_ep}, expected \
         version={BOOTSTRAP_VERSION} ep_size={ep_size}"
    );
    let (start, end) = (word(&head[12..16]) as usize, word(&head[16..20]) as usize);
    ensure!(
        start < end && end <= ep_size,
        "bootstrap hello names ranks {start}..{end}, outside 0..{ep_size}"
    );
    let mut slabs = Vec::with_capacity(end - start);
    for _ in start..end {
        let mut num_bytes = [0u8; 8];
        stream
            .read_exact(&mut num_bytes)
            .context("read a slab size")?;
        let mut handle = [0u8; K3_MEGA_FABRIC_HANDLE_BYTES];
        stream.read_exact(&mut handle).context("read a handle")?;
        slabs.push(WireSlab {
            num_bytes: u64::from_le_bytes(num_bytes),
            handle,
        });
    }
    Ok((start..end, slabs))
}

fn write_reply_table(stream: &mut TcpStream, table: &[WireSlab]) -> Result<()> {
    let mut reply = Vec::with_capacity(4 + table.len() * (8 + K3_MEGA_FABRIC_HANDLE_BYTES));
    reply.extend_from_slice(&0u32.to_le_bytes());
    for slab in table {
        reply.extend_from_slice(&slab.num_bytes.to_le_bytes());
        reply.extend_from_slice(&slab.handle);
    }
    stream.write_all(&reply).context("send the table")
}

fn write_reply_error(stream: &mut TcpStream, message: &str) -> Result<()> {
    stream.write_all(&1u32.to_le_bytes())?;
    stream.write_all(message.as_bytes())?;
    Ok(())
}

fn read_reply(stream: &mut TcpStream, ep_size: usize) -> Result<Vec<WireSlab>> {
    let mut status = [0u8; 4];
    stream.read_exact(&mut status).context("read the reply")?;
    if u32::from_le_bytes(status) != 0 {
        let mut message = String::new();
        let _ = stream.read_to_string(&mut message);
        bail!("K3 EP bootstrap rejected: {message}");
    }
    let mut table = Vec::with_capacity(ep_size);
    for _ in 0..ep_size {
        let mut num_bytes = [0u8; 8];
        stream
            .read_exact(&mut num_bytes)
            .context("read the table")?;
        let mut handle = [0u8; K3_MEGA_FABRIC_HANDLE_BYTES];
        stream.read_exact(&mut handle).context("read the table")?;
        table.push(WireSlab {
            num_bytes: u64::from_le_bytes(num_bytes),
            handle,
        });
    }
    Ok(table)
}

// ── One rank's runtime ──────────────────────────────────────────────────

/// One rank of an expert-parallel group.
///
/// It owns no buffers and issues nothing per step. Its whole job is the startup
/// handshake: publish this rank's symmetric slab, and — once, on the stepping
/// thread — collect the world's table and make every peer slab addressable
/// (peer-access confirmation in-process; nothing at all in fleet mode, where
/// the imported fabric mappings carry their own access grants).
pub(crate) struct K3EpRuntime {
    rendezvous: Arc<K3EpRendezvous>,
    rank: usize,
    device_ordinal: usize,
    ready: bool,
}

impl K3EpRuntime {
    /// Publish this rank's slab. The caller must already have synchronised the
    /// allocation's zeroing: a peer that sees this entry is entitled to assume
    /// the memory behind it is live and zeroed. `fabric` is required exactly
    /// when the rendezvous is a fleet.
    pub(crate) fn new(
        rendezvous: Arc<K3EpRendezvous>,
        rank: usize,
        base: i64,
        device_ordinal: usize,
        fabric: Option<K3FabricSlab>,
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
                fabric,
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
    /// This blocks until every peer has published (and, in fleet mode, until
    /// the whole world's bootstrap has completed), which is the group's
    /// startup barrier: a rank publishes only after its slab is allocated,
    /// zeroed and synchronised, so no launch can precede the last allocation.
    ///
    /// In-process, the device pairs were opened before the slabs were
    /// allocated (the memory-pool grant has to precede the allocation it
    /// covers), so all that is left is confirming that the ranks the group
    /// actually contains are among the ones this rank opened. In fleet mode
    /// there is nothing to open: local slabs granted every local device at
    /// allocation, and imported mappings carry their grants.
    pub(crate) fn ensure_ready(&mut self) -> Result<Option<Vec<i64>>> {
        if self.ready {
            return Ok(None);
        }
        let table = if self.rendezvous.is_fleet() {
            let table = self
                .rendezvous
                .fleet_table(self.rank, self.device_ordinal)?;
            self.ready = true;
            log::info!(
                "K3 EP rank {} paired with {} ranks over the NVLink fabric",
                self.rank,
                table.len()
            );
            table.as_ref().clone()
        } else {
            let slabs = self.rendezvous.local_slabs(self.rank)?;
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
            slabs.iter().map(|slab| slab.base).collect()
        };
        Ok(Some(table))
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

#[cfg(test)]
mod tests {
    use super::*;

    fn wire(fill: u8) -> WireSlab {
        WireSlab {
            num_bytes: 4096,
            handle: [fill; K3_MEGA_FABRIC_HANDLE_BYTES],
        }
    }

    /// Root + one peer over localhost: the peer's handles land in the root's
    /// table and both sides read back the same completed world.
    #[test]
    fn bootstrap_round_trips_the_world_table() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("local addr").to_string();
        drop(listener);

        let root_addr = addr.clone();
        let root = std::thread::spawn(move || {
            serve_bootstrap(&root_addr, 4, 0..2, &[wire(0), wire(1)]).expect("serve")
        });
        let peer = std::thread::spawn(move || {
            fetch_bootstrap(&addr, 4, 2..4, &[wire(2), wire(3)]).expect("fetch")
        });
        let root_table = root.join().expect("root thread");
        let peer_table = peer.join().expect("peer thread");
        for (rank, table) in [&root_table, &peer_table].into_iter().enumerate() {
            assert_eq!(table.len(), 4, "table {rank}");
            for (peer_rank, slab) in table.iter().enumerate() {
                assert_eq!(slab.handle, [peer_rank as u8; K3_MEGA_FABRIC_HANDLE_BYTES]);
                assert_eq!(slab.num_bytes, 4096);
            }
        }
    }

    /// An ep_size mismatch is a configuration error, reported to the peer.
    #[test]
    fn bootstrap_rejects_a_mismatched_world() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("local addr").to_string();
        drop(listener);

        let root_addr = addr.clone();
        let root = std::thread::spawn(move || {
            // The root still completes: the second, correct peer supplies the
            // missing ranks after the mismatched one is refused.
            serve_bootstrap(&root_addr, 4, 0..2, &[wire(0), wire(1)]).expect("serve")
        });
        let bad_addr = addr.clone();
        let bad = std::thread::spawn(move || {
            fetch_bootstrap(&bad_addr, 8, 2..4, &[wire(9), wire(9)])
                .expect_err("an ep_size mismatch must be refused")
        });
        let error = bad.join().expect("bad peer thread");
        assert!(error.to_string().contains("rejected"), "{error:#}");
        let good = std::thread::spawn(move || {
            fetch_bootstrap(&addr, 4, 2..4, &[wire(2), wire(3)]).expect("fetch")
        });
        good.join().expect("good peer thread");
        root.join().expect("root thread");
    }

    /// A fleet rendezvous refuses shapes that make no sense.
    #[test]
    fn fleet_shapes_are_validated() {
        assert!(K3EpRendezvous::fleet(16, 4..8, "unused".into()).is_ok());
        assert!(K3EpRendezvous::fleet(16, 0..16, "unused".into()).is_err());
        assert!(K3EpRendezvous::fleet(16, 12..17, "unused".into()).is_err());
        assert!(K3EpRendezvous::fleet(16, 4..4, "unused".into()).is_err());
    }
}
