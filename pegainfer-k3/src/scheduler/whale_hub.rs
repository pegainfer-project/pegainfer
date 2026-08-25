//! Transports for the whale rendezvous ([`super::whale`]): the pure state
//! machines exchange [`WhaleToSequencer`]/[`WhaleToMember`] values; a hub
//! moves them between processes and holds the sequencer.
//!
//! * [`LocalWhaleHub`] — one process hosts the whole world: the sequencer
//!   lives behind a mutex and outbound messages land straight in per-rank
//!   mailboxes. This is the in-process degenerate case of the fleet protocol,
//!   and what the scheduler's mock-fleet tests drive.
//! * [`TcpWhaleHub`] — the fleet: the process hosting global rank 0 runs the
//!   sequencer and listens; every other process keeps one connection open
//!   (the same lifetime as the EP bootstrap link, `executor/ep.rs`) and
//!   routes its local ranks' traffic over it. Messages are hand-framed
//!   little-endian, magic-tagged, and versioned, like the EP bootstrap —
//!   whales are sparse and small (a 256k-token prompt is 1 MiB), so there is
//!   nothing here worth a serialization dependency.
//!
//! Either hub presents the same two calls to the scheduler: `send` a message
//! toward the sequencer, `drain` this rank's pending messages at a launch
//! boundary. Neither waits on another rank's progress: `drain` empties a
//! local mailbox and a host's `send` is a channel put; a peer's `send`
//! writes the frame inline, bounded by the write timeout, not by a peer.

use std::collections::VecDeque;
use std::io::Read;
use std::io::Write;
use std::net::TcpListener;
use std::net::TcpStream;
use std::sync::Arc;
use std::sync::Condvar;
use std::sync::Mutex;
use std::sync::mpsc;
use std::time::Duration;

use anyhow::Context as _;
use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use pegainfer_kernels::ops::K3_MEGA_FABRIC_HANDLE_BYTES;

use super::whale::GlobalRank;
use super::whale::WhaleDescriptor;
use super::whale::WhaleOutbound;
use super::whale::WhaleSeq;
use super::whale::WhaleSequencer;
use super::whale::WhaleToMember;
use super::whale::WhaleToSequencer;
use crate::executor::whale_gang::K3WhaleSlabWire;

/// `"K3WH"`: a stray connection to the wrong port dies loudly.
const WHALE_MAGIC: u32 = u32::from_le_bytes(*b"K3WH");
const WHALE_VERSION: u32 = 1;
/// A gathering whale whose member never replies is cancelled after this and
/// the poster falls back to a local prefill. A genuinely dead member kills
/// the fleet through the collective's own deadline; this bound only keeps
/// the whale queue from wedging behind a slow joiner.
const WHALE_GATHER_DEADLINE: Duration = Duration::from_secs(30);
/// Socket writes only; reads are unbounded (a link is engine-lifetime) except
/// the one startup table read, which waits on the whole fleet's load.
const WHALE_WRITE_TIMEOUT: Duration = Duration::from_secs(30);
const WHALE_CONNECT_RETRY: Duration = Duration::from_secs(2);
/// The sequencer process binds after its (possibly enormous) weight load;
/// peers retry for a window that survives it, exactly like the EP bootstrap.
const WHALE_CONNECT_TIMEOUT: Duration = Duration::from_secs(3600);
/// A prompt is at most the serving ceiling; anything claiming more than this
/// many tokens is a framing bug, not a request.
const WHALE_MAX_PROMPT_TOKENS: u32 = 1 << 22;
/// No fleet is bigger than this; larger counts in a frame are framing bugs.
const WHALE_MAX_WORLD: usize = 1024;

/// Per-rank inbox of sequencer messages, drained at launch boundaries.
type Mailboxes = Arc<Mutex<Vec<VecDeque<WhaleToMember>>>>;

fn deliver_local(mailboxes: &Mailboxes, first_local: GlobalRank, outbound: &WhaleOutbound) -> bool {
    let mut boxes = mailboxes.lock().expect("whale mailboxes poisoned");
    let Some(slot) = outbound.to.checked_sub(first_local) else {
        return false;
    };
    let Some(inbox) = boxes.get_mut(slot) else {
        return false;
    };
    inbox.push_back(outbound.message.clone());
    true
}

fn drain_mailbox(
    mailboxes: &Mailboxes,
    first_local: GlobalRank,
    rank: GlobalRank,
) -> Vec<WhaleToMember> {
    let mut boxes = mailboxes.lock().expect("whale mailboxes poisoned");
    rank.checked_sub(first_local)
        .and_then(|slot| boxes.get_mut(slot))
        .map(|inbox| inbox.drain(..).collect())
        .unwrap_or_default()
}

/// The transport a whale serving lane holds, whichever process shape the
/// deployment has: one process hosting the whole world ([`LocalWhaleHub`]) or
/// a fleet peer/host ([`TcpWhaleHub`]).
#[derive(Clone)]
pub enum K3WhaleHub {
    Local(Arc<LocalWhaleHub>),
    Tcp(Arc<TcpWhaleHub>),
}

impl K3WhaleHub {
    pub fn send(&self, message: WhaleToSequencer) -> Result<()> {
        match self {
            Self::Local(hub) => hub.send(message),
            Self::Tcp(hub) => hub.send(message),
        }
    }

    pub fn drain(&self, rank: GlobalRank) -> Vec<WhaleToMember> {
        match self {
            Self::Local(hub) => hub.drain(rank),
            Self::Tcp(hub) => hub.drain(rank),
        }
    }
}

impl std::fmt::Debug for K3WhaleHub {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Local(_) => "K3WhaleHub::Local",
            Self::Tcp(_) => "K3WhaleHub::Tcp",
        })
    }
}

// ---------------------------------------------------------------------------
// Local hub
// ---------------------------------------------------------------------------

/// The whole world in one process: `send` feeds the sequencer directly and
/// its outbound lands in the mailboxes before the call returns.
pub struct LocalWhaleHub {
    sequencer: Mutex<WhaleSequencer>,
    mailboxes: Mailboxes,
}

impl LocalWhaleHub {
    pub fn new(world: usize, chunk_tokens: usize) -> Arc<Self> {
        Arc::new(Self {
            sequencer: Mutex::new(WhaleSequencer::new(world, chunk_tokens)),
            mailboxes: Arc::new(Mutex::new(vec![VecDeque::new(); world])),
        })
    }

    pub fn send(&self, message: WhaleToSequencer) -> Result<()> {
        let outbound = self
            .sequencer
            .lock()
            .expect("whale sequencer poisoned")
            .on_message(message)?;
        for out in outbound {
            deliver_local(&self.mailboxes, 0, &out);
        }
        Ok(())
    }

    pub fn drain(&self, rank: GlobalRank) -> Vec<WhaleToMember> {
        drain_mailbox(&self.mailboxes, 0, rank)
    }
}

// ---------------------------------------------------------------------------
// Wire format
// ---------------------------------------------------------------------------

fn put_u32(writer: &mut impl Write, value: u32) -> Result<()> {
    writer
        .write_all(&value.to_le_bytes())
        .context("whale frame write")
}

fn put_u64(writer: &mut impl Write, value: u64) -> Result<()> {
    writer
        .write_all(&value.to_le_bytes())
        .context("whale frame write")
}

fn put_rank(writer: &mut impl Write, rank: GlobalRank) -> Result<()> {
    put_u32(writer, u32::try_from(rank).context("whale rank")?)
}

fn put_tokens(writer: &mut impl Write, tokens: &[u32]) -> Result<()> {
    put_u32(
        writer,
        u32::try_from(tokens.len()).context("whale prompt length")?,
    )?;
    let mut bytes = Vec::with_capacity(tokens.len() * 4);
    for &token in tokens {
        bytes.extend_from_slice(&token.to_le_bytes());
    }
    writer.write_all(&bytes).context("whale frame write")
}

fn get_u32(reader: &mut impl Read) -> Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf).context("whale frame read")?;
    Ok(u32::from_le_bytes(buf))
}

fn get_u64(reader: &mut impl Read) -> Result<u64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf).context("whale frame read")?;
    Ok(u64::from_le_bytes(buf))
}

fn get_rank(reader: &mut impl Read) -> Result<GlobalRank> {
    Ok(get_u32(reader)? as GlobalRank)
}

fn get_tokens(reader: &mut impl Read) -> Result<Arc<[u32]>> {
    let len = get_u32(reader)?;
    ensure!(
        len <= WHALE_MAX_PROMPT_TOKENS,
        "whale frame claims {len} prompt tokens — framing bug"
    );
    let mut bytes = vec![0u8; len as usize * 4];
    reader.read_exact(&mut bytes).context("whale frame read")?;
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("chunks_exact(4)")))
        .collect())
}

const KIND_HELLO: u32 = 0;
const KIND_REQUEST: u32 = 1;
const KIND_READY: u32 = 2;
const KIND_GATHER: u32 = 3;
const KIND_COMMIT: u32 = 4;
const KIND_CANCEL: u32 = 5;
/// Startup only, peer → host: one local rank's whale-slab fabric identity.
const KIND_SLAB: u32 = 6;
/// Startup only, host → peer: the world's completed slab table.
const KIND_TABLE: u32 = 7;

fn write_header(writer: &mut impl Write, kind: u32) -> Result<()> {
    put_u32(writer, WHALE_MAGIC)?;
    put_u32(writer, WHALE_VERSION)?;
    put_u32(writer, kind)
}

fn read_header(reader: &mut impl Read) -> Result<u32> {
    let magic = get_u32(reader)?;
    ensure!(magic == WHALE_MAGIC, "whale link: bad magic {magic:#x}");
    let version = get_u32(reader)?;
    ensure!(
        version == WHALE_VERSION,
        "whale link: version {version}, expected {WHALE_VERSION}"
    );
    get_u32(reader)
}

fn write_to_sequencer(writer: &mut impl Write, message: &WhaleToSequencer) -> Result<()> {
    match message {
        WhaleToSequencer::Request {
            request,
            poster,
            prompt,
        } => {
            write_header(writer, KIND_REQUEST)?;
            put_u64(writer, *request)?;
            put_rank(writer, *poster)?;
            put_tokens(writer, prompt)?;
        }
        WhaleToSequencer::Ready { seq, rank, count } => {
            write_header(writer, KIND_READY)?;
            put_u64(writer, *seq)?;
            put_rank(writer, *rank)?;
            put_u64(writer, *count)?;
        }
    }
    writer.flush().context("whale link flush")
}

fn read_to_sequencer(reader: &mut impl Read, kind: u32) -> Result<WhaleToSequencer> {
    match kind {
        KIND_REQUEST => {
            let request = get_u64(reader)?;
            let poster = get_rank(reader)?;
            let prompt = get_tokens(reader)?;
            Ok(WhaleToSequencer::Request {
                request,
                poster,
                prompt,
            })
        }
        KIND_READY => {
            let seq = get_u64(reader)?;
            let rank = get_rank(reader)?;
            let count = get_u64(reader)?;
            Ok(WhaleToSequencer::Ready { seq, rank, count })
        }
        other => bail!("whale link: kind {other} is not a sequencer-bound message"),
    }
}

fn write_to_member(writer: &mut impl Write, to: GlobalRank, message: &WhaleToMember) -> Result<()> {
    let kind = match message {
        WhaleToMember::Gather { .. } => KIND_GATHER,
        WhaleToMember::Commit { .. } => KIND_COMMIT,
        WhaleToMember::Cancel { .. } => KIND_CANCEL,
    };
    write_header(writer, kind)?;
    put_rank(writer, to)?;
    match message {
        WhaleToMember::Gather { descriptor } => {
            put_u64(writer, descriptor.seq)?;
            put_u64(writer, descriptor.request)?;
            put_rank(writer, descriptor.poster)?;
            put_u32(
                writer,
                u32::try_from(descriptor.gang.len()).context("whale gang size")?,
            )?;
            for &member in descriptor.gang.iter() {
                put_rank(writer, member)?;
            }
            put_u64(writer, descriptor.prompt_hash)?;
            put_tokens(writer, &descriptor.prompt)?;
        }
        WhaleToMember::Commit { seq, launch } => {
            put_u64(writer, *seq)?;
            put_u64(writer, *launch)?;
        }
        WhaleToMember::Cancel { seq } => put_u64(writer, *seq)?,
    }
    writer.flush().context("whale link flush")
}

fn read_to_member(reader: &mut impl Read, kind: u32) -> Result<(GlobalRank, WhaleToMember)> {
    let to = get_rank(reader)?;
    let message = match kind {
        KIND_GATHER => {
            let seq = get_u64(reader)?;
            let request = get_u64(reader)?;
            let poster = get_rank(reader)?;
            let gang_len = get_u32(reader)? as usize;
            ensure!(
                gang_len <= WHALE_MAX_WORLD,
                "whale link: gang of {gang_len} ranks"
            );
            let gang: Arc<[GlobalRank]> = (0..gang_len)
                .map(|_| get_rank(reader))
                .collect::<Result<_>>()?;
            let prompt_hash = get_u64(reader)?;
            let prompt = get_tokens(reader)?;
            let descriptor = WhaleDescriptor {
                seq,
                request,
                poster,
                gang,
                prompt,
                prompt_hash,
            };
            descriptor.verify()?;
            WhaleToMember::Gather { descriptor }
        }
        KIND_COMMIT => {
            let seq = get_u64(reader)?;
            let launch = get_u64(reader)?;
            WhaleToMember::Commit { seq, launch }
        }
        KIND_CANCEL => WhaleToMember::Cancel {
            seq: get_u64(reader)?,
        },
        other => bail!("whale link: kind {other} is not a member-bound message"),
    };
    Ok((to, message))
}

// ---------------------------------------------------------------------------
// TCP hub
// ---------------------------------------------------------------------------

/// What one peer process announces on connect: the global ranks it hosts,
/// and how many slab frames follow (0 = no data-plane exchange, or exactly
/// one per hosted rank).
struct Hello {
    ranks: std::ops::Range<GlobalRank>,
    slabs: usize,
}

fn write_hello(writer: &mut impl Write, hello: &Hello) -> Result<()> {
    write_header(writer, KIND_HELLO)?;
    put_rank(writer, hello.ranks.start)?;
    put_u32(
        writer,
        u32::try_from(hello.ranks.len()).context("whale hello count")?,
    )?;
    put_u32(
        writer,
        u32::try_from(hello.slabs).context("whale hello slabs")?,
    )?;
    writer.flush().context("whale link flush")
}

fn put_slab_body(writer: &mut impl Write, slab: &K3WhaleSlabWire) -> Result<()> {
    put_u64(
        writer,
        u64::try_from(slab.num_bytes).context("whale slab size")?,
    )?;
    writer
        .write_all(&slab.handle)
        .context("whale slab handle write")
}

fn get_slab_body(reader: &mut impl Read) -> Result<K3WhaleSlabWire> {
    let num_bytes = usize::try_from(get_u64(reader)?).context("whale slab size")?;
    let mut handle = [0u8; K3_MEGA_FABRIC_HANDLE_BYTES];
    reader
        .read_exact(&mut handle)
        .context("whale slab handle read")?;
    Ok(K3WhaleSlabWire { handle, num_bytes })
}

fn write_slab(writer: &mut impl Write, rank: GlobalRank, slab: &K3WhaleSlabWire) -> Result<()> {
    write_header(writer, KIND_SLAB)?;
    put_rank(writer, rank)?;
    put_slab_body(writer, slab)?;
    writer.flush().context("whale link flush")
}

fn read_slab(reader: &mut impl Read) -> Result<(GlobalRank, K3WhaleSlabWire)> {
    Ok((get_rank(reader)?, get_slab_body(reader)?))
}

fn write_table(writer: &mut impl Write, table: &[K3WhaleSlabWire]) -> Result<()> {
    write_header(writer, KIND_TABLE)?;
    put_u32(
        writer,
        u32::try_from(table.len()).context("whale table size")?,
    )?;
    for slab in table {
        put_slab_body(writer, slab)?;
    }
    writer.flush().context("whale link flush")
}

fn read_table(reader: &mut impl Read) -> Result<Vec<K3WhaleSlabWire>> {
    let world = get_u32(reader)? as usize;
    ensure!(
        world <= WHALE_MAX_WORLD,
        "whale link: table of {world} ranks"
    );
    (0..world).map(|_| get_slab_body(reader)).collect()
}

/// The startup slab allgather's shared state on the host: one slot per world
/// rank, completed when every process has checked its slabs in.
struct SlabBoard {
    table: Mutex<Vec<Option<K3WhaleSlabWire>>>,
    done: Condvar,
}

impl SlabBoard {
    fn insert(&self, rank: GlobalRank, slab: K3WhaleSlabWire) -> Result<()> {
        let mut table = self.table.lock().expect("whale slab board poisoned");
        let slot = table
            .get_mut(rank)
            .with_context(|| format!("whale slab for rank {rank}, outside the world"))?;
        ensure!(slot.is_none(), "whale slab for rank {rank} arrived twice");
        *slot = Some(slab);
        self.done.notify_all();
        Ok(())
    }

    /// Block until every rank's slab is in, then return the completed table.
    fn wait_complete(&self, timeout: Duration) -> Result<Vec<K3WhaleSlabWire>> {
        let deadline = std::time::Instant::now() + timeout;
        let mut table = self.table.lock().expect("whale slab board poisoned");
        loop {
            if table.iter().all(Option::is_some) {
                return Ok(table.iter().map(|slot| slot.expect("checked")).collect());
            }
            let now = std::time::Instant::now();
            ensure!(
                now < deadline,
                "whale slab exchange timed out: {} of {} ranks checked in",
                table.iter().filter(|slot| slot.is_some()).count(),
                table.len()
            );
            let (next, _) = self
                .done
                .wait_timeout(table, deadline - now)
                .expect("whale slab board poisoned");
            table = next;
        }
    }
}

/// The fleet transport. Construct with [`TcpWhaleHub::host`] on the process
/// hosting global rank 0 (it runs the sequencer) or [`TcpWhaleHub::connect`]
/// everywhere else. Both sides serve `send`/`drain` for their local ranks;
/// the reader/sequencer threads live for the process's lifetime, like the
/// engine threads they serve — a dead link is engine-fatal, reported at the
/// next `send`.
pub struct TcpWhaleHub {
    first_local: GlobalRank,
    mailboxes: Mailboxes,
    outbound: HubRole,
    /// The first error any background thread hit; `send` reports it.
    failed: Arc<Mutex<Option<String>>>,
}

enum HubRole {
    Host {
        to_sequencer: mpsc::Sender<WhaleToSequencer>,
    },
    Peer {
        to_host: Mutex<TcpStream>,
    },
}

impl TcpWhaleHub {
    /// Host the sequencer for a `world`-rank fleet, serving `local` ranks
    /// `first_local..first_local + local` from this process. Binds `addr` and
    /// accepts peer processes for the hub's whole lifetime.
    ///
    /// `local_slabs` arms the startup data-plane exchange: this process's
    /// slab identities, one per local rank in rank order. When non-empty,
    /// every peer must check in its own (`connect` with slabs), the call
    /// blocks until the world's table is complete — the fleet's startup
    /// barrier, like the EP bootstrap — and the table comes back alongside
    /// the hub. Empty runs the pure rendezvous with no exchange.
    pub fn host(
        addr: &str,
        world: usize,
        chunk_tokens: usize,
        first_local: GlobalRank,
        local: usize,
        local_slabs: Vec<K3WhaleSlabWire>,
    ) -> Result<(Arc<Self>, Vec<K3WhaleSlabWire>)> {
        ensure!(
            local_slabs.is_empty() || local_slabs.len() == local,
            "whale hub: {} slabs for {local} local ranks",
            local_slabs.len()
        );
        let board = if local_slabs.is_empty() {
            None
        } else {
            let mut table = vec![None; world];
            for (offset, slab) in local_slabs.into_iter().enumerate() {
                table[first_local + offset] = Some(slab);
            }
            Some(Arc::new(SlabBoard {
                table: Mutex::new(table),
                done: Condvar::new(),
            }))
        };
        // Bind the port on every interface rather than whatever the hostname
        // in `addr` resolves to locally: /etc/hosts commonly maps a machine's
        // own name to 127.0.1.1, which would put the listener on loopback
        // while the peers dial the fabric address — connection refused with
        // nothing in the sequencer's log (the EP bootstrap hit exactly this).
        let port = addr
            .rsplit_once(':')
            .map(|(_, port)| port)
            .with_context(|| format!("whale hub: address {addr} carries no port"))?;
        let listener = TcpListener::bind(("0.0.0.0", port.parse::<u16>()?))
            .with_context(|| format!("whale hub: bind 0.0.0.0:{port} ({addr})"))?;
        let mailboxes: Mailboxes = Arc::new(Mutex::new(vec![VecDeque::new(); local]));
        let failed: Arc<Mutex<Option<String>>> = Arc::default();
        let (inbound_tx, inbound_rx) = mpsc::channel::<WhaleToSequencer>();
        // Peer connections register their writer half here, keyed by the rank
        // range from their hello.
        type PeerLinks = Arc<Mutex<Vec<(std::ops::Range<GlobalRank>, TcpStream)>>>;
        let peers: PeerLinks = Arc::default();

        let hub = Arc::new(Self {
            first_local,
            mailboxes: mailboxes.clone(),
            outbound: HubRole::Host {
                to_sequencer: inbound_tx.clone(),
            },
            failed: failed.clone(),
        });

        // One connection thread per peer process: the slab exchange blocks
        // until the whole world checks in, so a serial acceptor deadlocks any
        // world with more than one peer process. The writer registers only
        // after the exchange, or the sequencer could interleave a commit
        // with the table frame on one socket. A peer that dies mid-exchange
        // only logs here; the world then times out `wait_complete` at
        // WHALE_CONNECT_TIMEOUT — that is the error the operator sees.
        {
            let peers = peers.clone();
            let failed = failed.clone();
            let board = board.clone();
            std::thread::Builder::new()
                .name("k3-whale-accept".into())
                .spawn(move || {
                    for connection in listener.incoming() {
                        let Ok(mut socket) = connection else { continue };
                        let _ = socket.set_nodelay(true);
                        let peers = peers.clone();
                        let board = board.clone();
                        let inbound = inbound_tx.clone();
                        let failed = failed.clone();
                        std::thread::Builder::new()
                            .name("k3-whale-conn".into())
                            .spawn(move || {
                                let hello = match read_header(&mut socket).and_then(|kind| {
                                    ensure!(
                                        kind == KIND_HELLO,
                                        "whale hub: first frame kind {kind}"
                                    );
                                    let first = get_rank(&mut socket)?;
                                    let count = get_u32(&mut socket)? as usize;
                                    let slabs = get_u32(&mut socket)? as usize;
                                    ensure!(
                                        count <= WHALE_MAX_WORLD,
                                        "whale hub: hello claims {count} ranks"
                                    );
                                    Ok(Hello {
                                        ranks: first..first + count,
                                        slabs,
                                    })
                                }) {
                                    Ok(hello) => hello,
                                    Err(error) => {
                                        log::warn!("K3 whale hub: rejected connection: {error:#}");
                                        return;
                                    }
                                };
                                if let Err(error) =
                                    serve_slab_exchange(&mut socket, &hello, board.as_ref())
                                {
                                    log::warn!(
                                        "K3 whale hub: slab exchange with ranks {:?} failed: \
                                         {error:#}",
                                        hello.ranks
                                    );
                                    return;
                                }
                                let Ok(writer) = socket.try_clone() else {
                                    return;
                                };
                                // The sequencer thread writes commits through
                                // this handle while holding the peer table; a
                                // wedged peer must fail the hub, not park the
                                // sequencer forever.
                                let _ = writer.set_write_timeout(Some(WHALE_WRITE_TIMEOUT));
                                peers
                                    .lock()
                                    .expect("whale peers poisoned")
                                    .push((hello.ranks, writer));
                                loop {
                                    let message = read_header(&mut socket)
                                        .and_then(|kind| read_to_sequencer(&mut socket, kind));
                                    match message {
                                        Ok(message) => {
                                            if inbound.send(message).is_err() {
                                                return;
                                            }
                                        }
                                        Err(error) => {
                                            fail(&failed, format!("peer link died: {error:#}"));
                                            return;
                                        }
                                    }
                                }
                            })
                            .expect("spawn whale connection");
                    }
                })
                .expect("spawn whale acceptor");
        }

        // Sequencer: consumes the inbound channel, routes outbound locally or
        // to the owning peer link. The recv timeout doubles as the gather
        // deadline tick.
        {
            let mailboxes = mailboxes.clone();
            let failed = failed.clone();
            std::thread::Builder::new()
                .name("k3-whale-seq".into())
                .spawn(move || {
                    let mut sequencer = WhaleSequencer::new(world, chunk_tokens);
                    let mut gathering: Option<(WhaleSeq, std::time::Instant)> = None;
                    loop {
                        let step = match inbound_rx.recv_timeout(WHALE_GATHER_DEADLINE / 4) {
                            Ok(message) => sequencer.on_message(message),
                            Err(mpsc::RecvTimeoutError::Timeout) => {
                                if gathering.is_some_and(|(_, start)| {
                                    start.elapsed() > WHALE_GATHER_DEADLINE
                                }) {
                                    sequencer.on_gather_timeout()
                                } else {
                                    continue;
                                }
                            }
                            Err(mpsc::RecvTimeoutError::Disconnected) => return,
                        };
                        let outbound = match step {
                            Ok(outbound) => outbound,
                            Err(error) => {
                                fail(&failed, format!("sequencer refused a message: {error:#}"));
                                return;
                            }
                        };
                        // The deadline clock belongs to one seq: a commit can
                        // pump the next gather within the same transition, and
                        // the new whale deserves a fresh window.
                        gathering = sequencer.gathering_seq().map(|seq| match gathering {
                            Some((previous, start)) if previous == seq => (seq, start),
                            _ => (seq, std::time::Instant::now()),
                        });
                        for out in outbound {
                            if deliver_local(&mailboxes, first_local, &out) {
                                continue;
                            }
                            let mut links = peers.lock().expect("whale peers poisoned");
                            let Some((_, writer)) =
                                links.iter_mut().find(|(ranks, _)| ranks.contains(&out.to))
                            else {
                                fail(
                                    &failed,
                                    format!("no link hosts whale member rank {}", out.to),
                                );
                                return;
                            };
                            if let Err(error) = write_to_member(writer, out.to, &out.message) {
                                fail(&failed, format!("peer write failed: {error:#}"));
                                return;
                            }
                        }
                    }
                })
                .expect("spawn whale sequencer");
        }
        let table = match board {
            None => Vec::new(),
            Some(board) => board
                .wait_complete(WHALE_CONNECT_TIMEOUT)
                .context("whale hub: the world never completed its slab exchange")?,
        };
        Ok((hub, table))
    }

    /// Join the fleet's whale hub as the process serving `local` ranks
    /// `first_local..`, connecting to the sequencer at `addr` (retrying
    /// through its weight load, like the EP bootstrap). Non-empty
    /// `local_slabs` join the startup data-plane exchange and block until
    /// the host serves the world's completed table back.
    pub fn connect(
        addr: &str,
        first_local: GlobalRank,
        local: usize,
        local_slabs: Vec<K3WhaleSlabWire>,
    ) -> Result<(Arc<Self>, Vec<K3WhaleSlabWire>)> {
        ensure!(
            local_slabs.is_empty() || local_slabs.len() == local,
            "whale hub: {} slabs for {local} local ranks",
            local_slabs.len()
        );
        let deadline = std::time::Instant::now() + WHALE_CONNECT_TIMEOUT;
        let mut socket = loop {
            match TcpStream::connect(addr) {
                Ok(socket) => break socket,
                Err(error) => {
                    ensure!(
                        std::time::Instant::now() < deadline,
                        "whale hub: could not reach the sequencer at {addr}: {error}"
                    );
                    std::thread::sleep(WHALE_CONNECT_RETRY);
                }
            }
        };
        socket.set_nodelay(true).ok();
        socket
            .set_write_timeout(Some(WHALE_WRITE_TIMEOUT))
            .context("whale hub: set write timeout")?;
        write_hello(
            &mut socket,
            &Hello {
                ranks: first_local..first_local + local,
                slabs: local_slabs.len(),
            },
        )?;
        let table = if local_slabs.is_empty() {
            Vec::new()
        } else {
            for (offset, slab) in local_slabs.iter().enumerate() {
                write_slab(&mut socket, first_local + offset, slab)?;
            }
            // The table lands only when the SLOWEST process has loaded and
            // checked in, so this read is a fleet-load bound, not an IO one.
            socket
                .set_read_timeout(Some(WHALE_CONNECT_TIMEOUT))
                .context("whale hub: set table read timeout")?;
            let kind = read_header(&mut socket)?;
            ensure!(
                kind == KIND_TABLE,
                "whale hub: expected the slab table, got frame kind {kind}"
            );
            let table = read_table(&mut socket)?;
            socket
                .set_read_timeout(None)
                .context("whale hub: clear table read timeout")?;
            table
        };
        let mailboxes: Mailboxes = Arc::new(Mutex::new(vec![VecDeque::new(); local]));
        let failed: Arc<Mutex<Option<String>>> = Arc::default();
        let reader_boxes = mailboxes.clone();
        let reader_failed = failed.clone();
        let mut reader = socket.try_clone().context("whale hub: clone socket")?;
        std::thread::Builder::new()
            .name("k3-whale-read".into())
            .spawn(move || {
                loop {
                    let message =
                        read_header(&mut reader).and_then(|kind| read_to_member(&mut reader, kind));
                    match message {
                        Ok((to, message)) => {
                            let out = WhaleOutbound { to, message };
                            if !deliver_local(&reader_boxes, first_local, &out) {
                                fail(
                                    &reader_failed,
                                    format!("sequencer addressed rank {to}, not hosted here"),
                                );
                                return;
                            }
                        }
                        Err(error) => {
                            fail(&reader_failed, format!("sequencer link died: {error:#}"));
                            return;
                        }
                    }
                }
            })
            .expect("spawn whale reader");
        Ok((
            Arc::new(Self {
                first_local,
                mailboxes,
                outbound: HubRole::Peer {
                    to_host: Mutex::new(socket),
                },
                failed,
            }),
            table,
        ))
    }

    pub fn send(&self, message: WhaleToSequencer) -> Result<()> {
        if let Some(reason) = self
            .failed
            .lock()
            .expect("whale hub failure poisoned")
            .clone()
        {
            bail!("whale hub failed: {reason}");
        }
        match &self.outbound {
            HubRole::Host { to_sequencer } => to_sequencer
                .send(message)
                .map_err(|_| anyhow::anyhow!("whale sequencer thread is gone")),
            HubRole::Peer { to_host } => {
                let mut socket = to_host.lock().expect("whale link poisoned");
                // A failed (or timed-out partial) write leaves the stream
                // mid-frame: poison the hub so a sibling rank's next send
                // cannot append into the abandoned frame.
                write_to_sequencer(&mut *socket, &message)
                    .inspect_err(|error| fail(&self.failed, format!("host link died: {error:#}")))
            }
        }
    }

    pub fn drain(&self, rank: GlobalRank) -> Vec<WhaleToMember> {
        drain_mailbox(&self.mailboxes, self.first_local, rank)
    }
}

/// The host side of one connection's startup slab exchange: read the peer's
/// slab frames onto the board, wait for the world to complete, and serve the
/// table back. A no-op when neither side armed the exchange; an error when
/// they disagree.
fn serve_slab_exchange(
    socket: &mut TcpStream,
    hello: &Hello,
    board: Option<&Arc<SlabBoard>>,
) -> Result<()> {
    let Some(board) = board else {
        ensure!(
            hello.slabs == 0,
            "the peer sent slabs, but this host has no data-plane exchange armed"
        );
        return Ok(());
    };
    ensure!(
        hello.slabs == hello.ranks.len(),
        "the peer hosts {} ranks but sent {} slabs",
        hello.ranks.len(),
        hello.slabs
    );
    for _ in 0..hello.slabs {
        let kind = read_header(socket)?;
        ensure!(kind == KIND_SLAB, "expected a slab frame, got kind {kind}");
        let (rank, slab) = read_slab(socket)?;
        ensure!(
            hello.ranks.contains(&rank),
            "a slab for rank {rank} from the process hosting {:?}",
            hello.ranks
        );
        board.insert(rank, slab)?;
    }
    let table = board.wait_complete(WHALE_CONNECT_TIMEOUT)?;
    write_table(socket, &table)
}

fn fail(failed: &Arc<Mutex<Option<String>>>, reason: String) {
    let mut slot = failed.lock().expect("whale hub failure poisoned");
    if slot.is_none() {
        log::error!("K3 whale hub: {reason}");
        *slot = Some(reason);
    }
}

#[cfg(test)]
mod tests {
    use super::super::whale::WhaleDuty;
    use super::super::whale::WhaleMember;
    use super::*;

    const CHUNK: usize = 16896;

    fn prompt_of(total: usize) -> Arc<[u32]> {
        (0..total as u32).collect()
    }

    #[test]
    fn local_hub_runs_a_whale_end_to_end() {
        let hub = LocalWhaleHub::new(4, CHUNK);
        hub.send(WhaleToSequencer::Request {
            request: 1,
            poster: 3,
            prompt: prompt_of(12288),
        })
        .unwrap();
        let mut members: Vec<WhaleMember> = (0..4).map(WhaleMember::new).collect();
        // The local hub is synchronous: gathers already sit in the mailboxes.
        for rank in 0..4 {
            for message in hub.drain(rank) {
                if let Some(reply) = members[rank].on_message(message, 7).unwrap() {
                    hub.send(reply).unwrap();
                }
            }
        }
        // ...and so do the commits after the last ready.
        let mut launches = Vec::new();
        for (rank, member) in members.iter_mut().enumerate() {
            for message in hub.drain(rank) {
                member.on_message(message, 8).unwrap();
            }
            let launch = (8..64)
                .find(|&count| matches!(member.at_launch(count).unwrap(), WhaleDuty::Enter(_)))
                .expect("member never entered");
            launches.push(launch);
        }
        assert!(
            launches.iter().all(|&launch| launch == launches[0]),
            "{launches:?}"
        );
    }

    #[test]
    fn tcp_hub_commits_across_a_loopback_peer() {
        // Host process serves ranks 0..2 and the sequencer; the peer process
        // serves ranks 2..4 and posts the whale. One simulated launch period
        // per loop pass keeps the loopback latency far below it, which is the
        // commit slack's stated assumption.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        drop(listener);
        let (host, _) = TcpWhaleHub::host(&addr, 4, CHUNK, 0, 2, Vec::new()).unwrap();
        let (peer, _) = TcpWhaleHub::connect(&addr, 2, 2, Vec::new()).unwrap();
        peer.send(WhaleToSequencer::Request {
            request: 5,
            poster: 2,
            prompt: prompt_of(12288),
        })
        .unwrap();
        let hubs: [&Arc<TcpWhaleHub>; 4] = [&host, &host, &peer, &peer];
        let mut members: Vec<WhaleMember> = (0..4).map(WhaleMember::new).collect();
        let mut launches: [Option<u64>; 4] = [None; 4];
        let deadline = std::time::Instant::now() + Duration::from_secs(20);
        let mut count = 0u64;
        while launches.iter().any(Option::is_none) {
            assert!(
                std::time::Instant::now() < deadline,
                "the whale never committed over the loopback: {launches:?}"
            );
            for rank in 0..4 {
                for message in hubs[rank].drain(rank) {
                    if let Some(reply) = members[rank].on_message(message, count).unwrap() {
                        hubs[rank].send(reply).unwrap();
                    }
                }
                if let WhaleDuty::Enter(whale) = members[rank].at_launch(count).unwrap() {
                    assert_eq!(whale.descriptor.cp_rank_of(rank), Some(whale.cp_rank));
                    launches[rank] = Some(count);
                }
            }
            count += 1;
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(
            launches.iter().all(|&launch| launch == launches[0]),
            "{launches:?}"
        );
    }

    #[test]
    fn slab_exchange_serves_every_process_the_same_world_table() {
        fn slab(seed: u8) -> K3WhaleSlabWire {
            K3WhaleSlabWire {
                handle: [seed; K3_MEGA_FABRIC_HANDLE_BYTES],
                num_bytes: 4096 + seed as usize,
            }
        }
        // The host serves ranks 0..2 and blocks until the world checks in, so
        // the peer joins from another thread.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        drop(listener);
        let peer_addr = addr.clone();
        let peer = std::thread::spawn(move || {
            TcpWhaleHub::connect(&peer_addr, 2, 2, vec![slab(2), slab(3)]).unwrap()
        });
        let (_host, host_table) =
            TcpWhaleHub::host(&addr, 4, CHUNK, 0, 2, vec![slab(0), slab(1)]).unwrap();
        let (_peer, peer_table) = peer.join().expect("peer joins the exchange");
        assert_eq!(host_table.len(), 4);
        assert_eq!(peer_table.len(), 4);
        for (rank, (host_slab, peer_slab)) in host_table.iter().zip(&peer_table).enumerate() {
            assert_eq!(host_slab.handle, [rank as u8; K3_MEGA_FABRIC_HANDLE_BYTES]);
            assert_eq!(host_slab.handle, peer_slab.handle);
            assert_eq!(host_slab.num_bytes, peer_slab.num_bytes);
        }
    }

    #[test]
    fn slab_exchange_completes_with_more_than_one_peer_process() {
        fn slab(seed: u8) -> K3WhaleSlabWire {
            K3WhaleSlabWire {
                handle: [seed; K3_MEGA_FABRIC_HANDLE_BYTES],
                num_bytes: 4096 + seed as usize,
            }
        }
        // Regression: with a serial acceptor the first accepted peer's
        // exchange blocked in `wait_complete` until the whole world checked
        // in, so the second peer was never accepted — a startup deadlock at
        // any world with more than one peer process (worked at two processes,
        // wedged the CP16 4-tray fleet). Three processes here: host ranks
        // 0..2, peers 2..4 and 4..6.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        drop(listener);
        let peers: Vec<_> = [2usize, 4]
            .into_iter()
            .map(|first| {
                let addr = addr.clone();
                std::thread::spawn(move || {
                    let slabs = vec![slab(first as u8), slab(first as u8 + 1)];
                    TcpWhaleHub::connect(&addr, first, 2, slabs).unwrap()
                })
            })
            .collect();
        let (_host, host_table) =
            TcpWhaleHub::host(&addr, 6, CHUNK, 0, 2, vec![slab(0), slab(1)]).unwrap();
        let mut tables = vec![host_table];
        for peer in peers {
            let (_peer, table) = peer.join().expect("peer joins the exchange");
            tables.push(table);
        }
        for table in &tables {
            assert_eq!(table.len(), 6);
            for (rank, entry) in table.iter().enumerate() {
                assert_eq!(entry.handle, [rank as u8; K3_MEGA_FABRIC_HANDLE_BYTES]);
                assert_eq!(entry.num_bytes, 4096 + rank);
            }
        }
    }
}
