//! Whale rendezvous: how a fleet of free-running EP ranks agrees to run one
//! long prompt's context-parallel prefill together — across processes.
//!
//! The in-process gang ([`super::gang`]) levels its members through a shared
//! board: every member re-posts its launch count until the whole gang sits at
//! the board's maximum. That works because the board is one `Mutex` away from
//! every member. A fleet gang spans processes on different machines, and the
//! free-running discipline (`docs/models/glm52/free-running-dp.md`) forbids
//! replacing the board with anything that waits: no rank ever stops, so the
//! agreement must ride on facts that are already true while everyone keeps
//! stepping.
//!
//! Two such facts carry the whole protocol:
//!
//! * **The launch count is a global clock.** Every launch is a two-sided
//!   paired collective, so the whole world's counts stay within one launch of
//!   each other. "Launch `L`" therefore names one global moment, and it
//!   arrives on its own — nobody has to be woken up for it.
//! * **Whales are sparse.** A 32k+ prompt is an event, not the steady state,
//!   so the coordination state may exist only around one whale and cost one
//!   host round-trip; there is no standing cross-process protocol.
//!
//! The rendezvous is a two-phase broadcast over a host side-channel (the
//! fleet's bootstrap TCP link, kept alive), sequenced by one rank so that
//! concurrent whales get one global order:
//!
//! 1. **Gather.** The poster sends the prompt to the sequencer, which picks
//!    the gang (widest width the segment floor admits — the wider the gang,
//!    the shorter the whole fleet's stall) and broadcasts the descriptor to
//!    the members. A member replies with its current launch count and *arms*:
//!    from here until the commit it starts no multi-launch operation, so its
//!    count advances by exactly one per step and any launch after its reply
//!    is one it can hit exactly. It keeps stepping the whole time.
//! 2. **Commit.** When every member has replied, the sequencer broadcasts
//!    `L = max(replies) + slack`. Every member's count is below `L` (counts
//!    advanced at most one per launch period while the commit was in flight,
//!    and the slack covers that), so each one steps normally until its count
//!    reaches `L` and enters the CP superstep there. All members enter at the
//!    same absolute launch or none do; a member that dies instead of entering
//!    is caught by the exchange deadline (fleet-fatal), not here.
//!
//! Ranks outside the gang never hear about the whale: their launch `L` is an
//! ordinary step, and the mega collective pairs it against the gang's
//! superstep exactly as it pairs any other heterogeneous step.
//!
//! The state machines below are pure — one event in, the messages to send
//! out — so the transports ([`LocalWhaleHub`],
//! [`super::whale_hub::TcpWhaleHub`]) stay I/O-only. Width and gang policy
//! live with the data plane ([`crate::executor::cp`]); this module only
//! applies them.
//!
//! [`LocalWhaleHub`]: super::whale_hub::LocalWhaleHub

use std::collections::VecDeque;
use std::sync::Arc;

use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;

use crate::executor::cp::k3_whale_gang;
use crate::executor::cp::k3_whale_width;

/// A rank's identity across the whole EP world (not its process-local index).
pub type GlobalRank = usize;
/// The absolute launch count — the fleet's global clock.
pub type LaunchCount = u64;
/// A whale's position in the sequencer's global order.
pub type WhaleSeq = u64;

/// Launches of slack the commit adds over the highest gathered count. An
/// armed member cannot stop launching (free-running), so the commit must name
/// a launch it has not yet passed: the slack is the number of launch periods
/// the reply-to-commit host round trip may take. Launch periods are tens of
/// milliseconds and up, the management-TCP round trip is sub-millisecond, so
/// four is a wide margin — and a commit that still arrives late dies loudly
/// at the member instead of mispairing a collective.
const K3_WHALE_COMMIT_SLACK: LaunchCount = 4;

/// One whale's identity and shape. Everything a member needs to run its
/// segment is a deterministic function of this descriptor, so agreeing on the
/// descriptor and the launch is agreeing on everything.
#[derive(Clone, Debug)]
pub struct WhaleDescriptor {
    pub seq: WhaleSeq,
    /// The scheduler-side request this whale answers, echoed back so the
    /// poster can pair the commit with its pending admission.
    pub request: u64,
    /// The gang's owner: the rank whose slot the prompt lands in. Always the
    /// last CP rank, so the final KDA state and the full MLA context end up
    /// on the rank that decodes the sequence.
    pub poster: GlobalRank,
    /// Gang members in CP order (`gang.last() == poster`).
    pub gang: Arc<[GlobalRank]>,
    pub prompt: Arc<[u32]>,
    /// FNV-1a over the prompt tokens: a descriptor that crossed the wire and
    /// disagrees with its own hash must die loudly, not compute garbage.
    pub prompt_hash: u64,
}

impl WhaleDescriptor {
    /// This rank's CP rank in the gang, if it is a member.
    pub fn cp_rank_of(&self, rank: GlobalRank) -> Option<usize> {
        self.gang.iter().position(|&member| member == rank)
    }

    pub fn verify(&self) -> Result<()> {
        ensure!(
            k3_whale_prompt_hash(&self.prompt) == self.prompt_hash,
            "whale {} prompt hash mismatch — the descriptor was corrupted in transit",
            self.seq
        );
        ensure!(
            self.gang.last() == Some(&self.poster),
            "whale {} gang does not end at its poster",
            self.seq
        );
        Ok(())
    }
}

/// FNV-1a, the token stream fed as little-endian bytes. Not cryptographic —
/// the threat is corruption and framing bugs, not an adversary on the fabric
/// management network.
fn k3_whale_prompt_hash(prompt: &[u32]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &token in prompt {
        for byte in token.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

/// Member -> sequencer.
#[derive(Clone, Debug)]
pub enum WhaleToSequencer {
    /// A rank asks for a whale prefill of `prompt` (it becomes the poster).
    Request {
        request: u64,
        poster: GlobalRank,
        prompt: Arc<[u32]>,
    },
    /// A member's gather reply: its launch count at the reply, from which on
    /// it is armed (stepping one launch at a time until the commit lands).
    Ready {
        seq: WhaleSeq,
        rank: GlobalRank,
        count: LaunchCount,
    },
}

/// Sequencer -> one member (the transport routes by rank).
#[derive(Clone, Debug)]
pub enum WhaleToMember {
    Gather {
        descriptor: WhaleDescriptor,
    },
    Commit {
        seq: WhaleSeq,
        launch: LaunchCount,
    },
    /// The whale will not run (a member never replied, or admission failed
    /// after sequencing). Members disarm; the poster answers the request by
    /// falling back to a local prefill.
    Cancel {
        seq: WhaleSeq,
    },
}

/// One outbound message with its destination.
#[derive(Clone, Debug)]
pub struct WhaleOutbound {
    pub to: GlobalRank,
    pub message: WhaleToMember,
}

fn broadcast(gang: &[GlobalRank], message: &WhaleToMember) -> Vec<WhaleOutbound> {
    gang.iter()
        .map(|&to| WhaleOutbound {
            to,
            message: message.clone(),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Sequencer
// ---------------------------------------------------------------------------

/// One whale mid-gather.
struct Gathering {
    descriptor: WhaleDescriptor,
    /// Reply per gang member, in gang order.
    replies: Vec<Option<LaunchCount>>,
}

/// The fleet's single whale sequencer, hosted next to global rank 0. Pure
/// state machine: transitions return the messages to deliver. One gather runs
/// at a time — a second request queues behind it — so commits leave here in
/// one global order with strictly increasing launches, which is what keeps
/// two whales from interleaving their exchange windows.
pub struct WhaleSequencer {
    world: usize,
    chunk_tokens: usize,
    next_seq: WhaleSeq,
    /// The launch the last committed whale runs at. The next commit must land
    /// strictly after it: the superstep at `L` is the whale's, whole.
    committed_floor: Option<LaunchCount>,
    gathering: Option<Gathering>,
    queue: VecDeque<(u64, GlobalRank, Arc<[u32]>)>,
}

impl WhaleSequencer {
    pub fn new(world: usize, chunk_tokens: usize) -> Self {
        Self {
            world,
            chunk_tokens,
            next_seq: 0,
            committed_floor: None,
            gathering: None,
            queue: VecDeque::new(),
        }
    }

    /// Feed one inbound message; returns what to send.
    pub fn on_message(&mut self, message: WhaleToSequencer) -> Result<Vec<WhaleOutbound>> {
        match message {
            WhaleToSequencer::Request {
                request,
                poster,
                prompt,
            } => {
                self.queue.push_back((request, poster, prompt));
                self.start_next_gather()
            }
            WhaleToSequencer::Ready { seq, rank, count } => {
                let Some(gathering) = self.gathering.as_mut() else {
                    // A reply to a whale that was cancelled under it: stale,
                    // not an error — the member already got the cancel.
                    return Ok(Vec::new());
                };
                if gathering.descriptor.seq != seq {
                    return Ok(Vec::new());
                }
                let Some(cp_rank) = gathering.descriptor.cp_rank_of(rank) else {
                    bail!("whale {seq}: ready from rank {rank}, which is not in the gang");
                };
                ensure!(
                    gathering.replies[cp_rank].is_none(),
                    "whale {seq}: rank {rank} replied ready twice"
                );
                gathering.replies[cp_rank] = Some(count);
                self.try_commit()
            }
        }
    }

    /// The whale currently gathering, if any. The transport owns the clock:
    /// it tracks how long one seq has been gathering and drives
    /// [`WhaleSequencer::on_gather_timeout`].
    pub fn gathering_seq(&self) -> Option<WhaleSeq> {
        self.gathering
            .as_ref()
            .map(|gathering| gathering.descriptor.seq)
    }

    /// The sequencer's gather deadline fired (transport-driven): cancel the
    /// gathering whale, if any, and move on. The poster falls back to a local
    /// prefill; a member that is genuinely dead kills the fleet through the
    /// collective's own deadline, not through this one.
    pub fn on_gather_timeout(&mut self) -> Result<Vec<WhaleOutbound>> {
        let Some(gathering) = self.gathering.take() else {
            return Ok(Vec::new());
        };
        let seq = gathering.descriptor.seq;
        log::warn!("K3 whale {seq}: gather timed out; cancelling");
        let mut outbound = broadcast(&gathering.descriptor.gang, &WhaleToMember::Cancel { seq });
        outbound.extend(self.start_next_gather()?);
        Ok(outbound)
    }

    /// Start the next gatherable queued whale, if none is gathering. A
    /// refused whale must not strand the ones queued behind it: no later
    /// message is guaranteed to ever arrive.
    fn start_next_gather(&mut self) -> Result<Vec<WhaleOutbound>> {
        let mut outbound = Vec::new();
        while self.gathering.is_none() {
            let Some((request, poster, prompt)) = self.queue.pop_front() else {
                break;
            };
            let seq = self.next_seq;
            self.next_seq += 1;
            let Some(width) = k3_whale_width(prompt.len(), self.world, self.chunk_tokens) else {
                // Too short for even the narrowest gang, or too long for the
                // widest: the poster prefills locally. Sequenced requests are
                // pre-screened by the poster, so this is a race (config
                // changed) or a bug — either way, refuse rather than wedge.
                log::warn!(
                    "K3 whale {seq}: {} tokens admit no gang width in a {}-rank world; \
                     cancelling",
                    prompt.len(),
                    self.world
                );
                outbound.push(WhaleOutbound {
                    to: poster,
                    message: WhaleToMember::Cancel { seq },
                });
                continue;
            };
            let gang = k3_whale_gang(poster, width, self.world);
            let prompt_hash = k3_whale_prompt_hash(&prompt);
            let descriptor = WhaleDescriptor {
                seq,
                request,
                poster,
                gang: gang.into(),
                prompt,
                prompt_hash,
            };
            outbound.extend(broadcast(
                &descriptor.gang,
                &WhaleToMember::Gather {
                    descriptor: descriptor.clone(),
                },
            ));
            self.gathering = Some(Gathering {
                replies: vec![None; descriptor.gang.len()],
                descriptor,
            });
        }
        Ok(outbound)
    }

    fn try_commit(&mut self) -> Result<Vec<WhaleOutbound>> {
        let Some(gathering) = self.gathering.as_ref() else {
            return Ok(Vec::new());
        };
        let Some(highest) = gathering
            .replies
            .iter()
            .copied()
            .try_fold(0, |max: LaunchCount, reply| {
                reply.map(|count| max.max(count))
            })
        else {
            return Ok(Vec::new());
        };
        let gathering = self.gathering.take().expect("checked above");
        // Strictly after both every member's armed count and the previous
        // whale's superstep. Armed members advance one launch per step, so
        // any launch above their replies is reachable exactly.
        let launch = (highest + K3_WHALE_COMMIT_SLACK)
            .max(self.committed_floor.map_or(0, |floor| floor + 1));
        self.committed_floor = Some(launch);
        let seq = gathering.descriptor.seq;
        let mut outbound = broadcast(
            &gathering.descriptor.gang,
            &WhaleToMember::Commit { seq, launch },
        );
        outbound.extend(self.start_next_gather()?);
        Ok(outbound)
    }
}

// ---------------------------------------------------------------------------
// Member
// ---------------------------------------------------------------------------

/// A committed whale this member will enter.
#[derive(Clone, Debug)]
pub struct CommittedWhale {
    pub descriptor: WhaleDescriptor,
    pub launch: LaunchCount,
    pub cp_rank: usize,
}

/// What the scheduler must do at a launch boundary.
#[derive(Debug)]
pub enum WhaleDuty {
    /// Nothing scheduled here: run a normal step. While the member is not
    /// [`WhaleMember::is_quiet`] that step must stay single-launch.
    Free,
    /// This launch is a committed whale's superstep: enter `prefill_cp` now.
    Enter(Box<CommittedWhale>),
}

/// One rank's side of the rendezvous. Pure: the scheduler feeds it inbound
/// messages and its own launch count, and acts on what comes back. The
/// scheduler must consult [`WhaleMember::at_launch`] at *every* launch
/// boundary — the protocol's soundness is exactly that an armed member's
/// count advances one launch at a time.
pub struct WhaleMember {
    rank: GlobalRank,
    /// Gathered, replied-to whales awaiting their commit. While any is
    /// pending the member is armed: no multi-launch operations, so the
    /// gathered reply stays an exact floor on reachable launches.
    armed: VecDeque<WhaleDescriptor>,
    /// Committed whales in launch order.
    committed: VecDeque<CommittedWhale>,
}

impl WhaleMember {
    pub fn new(rank: GlobalRank) -> Self {
        Self {
            rank,
            armed: VecDeque::new(),
            committed: VecDeque::new(),
        }
    }

    /// Feed one message from the sequencer; optionally returns the reply.
    pub fn on_message(
        &mut self,
        message: WhaleToMember,
        count: LaunchCount,
    ) -> Result<Option<WhaleToSequencer>> {
        match message {
            WhaleToMember::Gather { descriptor } => {
                descriptor.verify()?;
                ensure!(
                    descriptor.cp_rank_of(self.rank).is_some(),
                    "whale {} gathered rank {}, which is not in its gang",
                    descriptor.seq,
                    self.rank
                );
                let seq = descriptor.seq;
                self.armed.push_back(descriptor);
                Ok(Some(WhaleToSequencer::Ready {
                    seq,
                    rank: self.rank,
                    count,
                }))
            }
            WhaleToMember::Commit { seq, launch } => {
                let Some(position) = self.armed.iter().position(|armed| armed.seq == seq) else {
                    bail!("whale {seq}: commit for a whale this member never gathered");
                };
                let descriptor = self.armed.remove(position).expect("position just found");
                ensure!(
                    launch > count,
                    "whale {seq}: committed to launch {launch}, but rank {} is already at \
                     {count} — the commit slack failed to cover the arming window",
                    self.rank
                );
                if let Some(last) = self.committed.back() {
                    ensure!(
                        launch > last.launch,
                        "whale {seq}: commit at {launch} does not follow whale {} at {}",
                        last.descriptor.seq,
                        last.launch
                    );
                }
                let cp_rank = descriptor
                    .cp_rank_of(self.rank)
                    .expect("membership checked at gather");
                self.committed.push_back(CommittedWhale {
                    descriptor,
                    launch,
                    cp_rank,
                });
                Ok(None)
            }
            WhaleToMember::Cancel { seq } => {
                self.armed.retain(|armed| armed.seq != seq);
                // A committed whale cannot be cancelled: members enter at its
                // launch unconditionally (unanimity is the whole point), and
                // an unwanted result is dropped, not prevented.
                Ok(None)
            }
        }
    }

    /// What to do at launch boundary `count`. Must be consulted at every
    /// boundary, including between the chunks of a local chunked prefill.
    pub fn at_launch(&mut self, count: LaunchCount) -> Result<WhaleDuty> {
        if let Some(front) = self.committed.front() {
            ensure!(
                front.launch >= count,
                "rank {} is at launch {count}, past whale {}'s superstep at {} — a launch \
                 boundary went unconsulted",
                self.rank,
                front.descriptor.seq,
                front.launch
            );
            if front.launch == count {
                let whale = self.committed.pop_front().expect("front just observed");
                return Ok(WhaleDuty::Enter(Box::new(whale)));
            }
        }
        Ok(WhaleDuty::Free)
    }

    /// Whether nothing is gathered or committed here. A member that is not
    /// quiet must step one launch at a time: an armed reply is an exact floor
    /// on reachable launches, and a committed superstep must be hit exactly.
    pub fn is_quiet(&self) -> bool {
        self.armed.is_empty() && self.committed.is_empty()
    }

    /// Whether `seq` is armed here (gathered, commit pending). The poster
    /// uses this to tell a helper-side cancel from the width refusal of its
    /// own not-yet-gathered post.
    pub fn is_armed(&self, seq: WhaleSeq) -> bool {
        self.armed.iter().any(|armed| armed.seq == seq)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The mega row ceiling (#962): CP16 x one chunk covers 256k in a single
    /// superstep, which is exactly why the protocol was raised to 16896.
    const CHUNK: usize = 16896;

    fn prompt_of(total: usize) -> Arc<[u32]> {
        (0..total as u32).collect()
    }

    #[test]
    fn prompt_hash_sees_order_and_length() {
        assert_ne!(
            k3_whale_prompt_hash(&[1, 2, 3]),
            k3_whale_prompt_hash(&[3, 2, 1])
        );
        assert_ne!(k3_whale_prompt_hash(&[]), k3_whale_prompt_hash(&[0]));
    }

    // ---- protocol: deterministic exchanges ---------------------------------

    fn commit_of(outbound: &WhaleOutbound) -> Option<(WhaleSeq, LaunchCount)> {
        match outbound.message {
            WhaleToMember::Commit { seq, launch } => Some((seq, launch)),
            _ => None,
        }
    }

    #[test]
    fn happy_path_commits_every_member_to_one_launch() {
        let mut sequencer = WhaleSequencer::new(4, CHUNK);
        let gathers = sequencer
            .on_message(WhaleToSequencer::Request {
                request: 9,
                poster: 2,
                prompt: prompt_of(12288),
            })
            .unwrap();
        assert_eq!(gathers.len(), 4);
        let mut members: Vec<WhaleMember> = (0..4).map(WhaleMember::new).collect();
        let mut commits = Vec::new();
        for gather in gathers {
            // Counts 10/11: the fleet's real ±1 launch skew.
            let count = 10 + gather.to as LaunchCount % 2;
            let reply = members[gather.to]
                .on_message(gather.message, count)
                .unwrap()
                .expect("a gather demands a ready");
            commits.extend(sequencer.on_message(reply).unwrap());
        }
        assert_eq!(commits.len(), 4);
        let (seq, launch) = commit_of(&commits[0]).expect("commit");
        assert_eq!(seq, 0);
        assert!(launch >= 11 + K3_WHALE_COMMIT_SLACK);
        for commit in commits {
            members[commit.to].on_message(commit.message, 12).unwrap();
        }
        // Every member free-runs to the launch and enters exactly there.
        for (rank, member) in members.iter_mut().enumerate() {
            let mut entered = None;
            for count in 12..=launch {
                match member.at_launch(count).unwrap() {
                    WhaleDuty::Enter(whale) => entered = Some((count, whale)),
                    WhaleDuty::Free => assert!(!member.is_quiet()),
                }
            }
            let (count, whale) = entered.expect("member never entered");
            assert_eq!(count, launch);
            assert_eq!(whale.descriptor.cp_rank_of(rank), Some(whale.cp_rank));
            assert!(member.is_quiet());
        }
    }

    #[test]
    fn late_reply_from_a_mid_operation_member_still_clears_its_count() {
        let mut sequencer = WhaleSequencer::new(4, CHUNK);
        let gathers = sequencer
            .on_message(WhaleToSequencer::Request {
                request: 1,
                poster: 0,
                prompt: prompt_of(12288),
            })
            .unwrap();
        let mut members: Vec<WhaleMember> = (0..4).map(WhaleMember::new).collect();
        // Rank 0 is three launches deep in a chunked local prefill and only
        // drains its inbox at 23; everyone else replies at 20.
        let mut commits = Vec::new();
        for gather in gathers {
            let count = if gather.to == 0 { 23 } else { 20 };
            let reply = members[gather.to]
                .on_message(gather.message, count)
                .unwrap()
                .unwrap();
            commits.extend(sequencer.on_message(reply).unwrap());
        }
        let (_, launch) = commit_of(&commits[0]).expect("commit");
        assert!(launch >= 23 + K3_WHALE_COMMIT_SLACK);
        // The straggler receives the commit one launch later still.
        let to_zero = commits.iter().find(|commit| commit.to == 0).unwrap();
        members[0].on_message(to_zero.message.clone(), 24).unwrap();
    }

    #[test]
    fn concurrent_whales_serialize_with_strictly_increasing_launches() {
        let mut sequencer = WhaleSequencer::new(8, CHUNK);
        let first_gathers = sequencer
            .on_message(WhaleToSequencer::Request {
                request: 1,
                poster: 1,
                prompt: prompt_of(20480),
            })
            .unwrap();
        assert_eq!(first_gathers.len(), 8);
        // The second request queues behind the first gather: silence.
        let queued = sequencer
            .on_message(WhaleToSequencer::Request {
                request: 2,
                poster: 5,
                prompt: prompt_of(20480),
            })
            .unwrap();
        assert!(queued.is_empty());
        let mut members: Vec<WhaleMember> = (0..8).map(WhaleMember::new).collect();
        let mut second_wave = Vec::new();
        for gather in first_gathers {
            let reply = members[gather.to]
                .on_message(gather.message, 5)
                .unwrap()
                .unwrap();
            second_wave.extend(sequencer.on_message(reply).unwrap());
        }
        // The last ready commits whale 0 and pumps whale 1's gather in one
        // transition; per-destination FIFO delivers them in that order.
        let launch_a = second_wave
            .iter()
            .find_map(commit_of)
            .expect("first commit")
            .1;
        let mut commits_b = Vec::new();
        for outbound in second_wave {
            match outbound.message {
                WhaleToMember::Commit { .. } => {
                    members[outbound.to]
                        .on_message(outbound.message, 6)
                        .unwrap();
                }
                WhaleToMember::Gather { .. } => {
                    let reply = members[outbound.to]
                        .on_message(outbound.message, 7)
                        .unwrap()
                        .unwrap();
                    commits_b.extend(sequencer.on_message(reply).unwrap());
                }
                WhaleToMember::Cancel { .. } => panic!("unexpected cancel"),
            }
        }
        let (seq_b, launch_b) = commits_b.iter().find_map(commit_of).expect("second commit");
        assert_eq!(seq_b, 1);
        assert!(
            launch_b > launch_a,
            "whale 1 at {launch_b} must follow whale 0 at {launch_a}"
        );
        // A member in both gangs accepts the commits in order.
        for commit in commits_b {
            members[commit.to].on_message(commit.message, 8).unwrap();
        }
    }

    #[test]
    fn gather_timeout_cancels_and_the_stale_ready_is_ignored() {
        let mut sequencer = WhaleSequencer::new(4, CHUNK);
        let gathers = sequencer
            .on_message(WhaleToSequencer::Request {
                request: 3,
                poster: 1,
                prompt: prompt_of(12288),
            })
            .unwrap();
        let straggler = gathers[3].to;
        let mut member = WhaleMember::new(straggler);
        let reply = member
            .on_message(gathers[3].message.clone(), 4)
            .unwrap()
            .unwrap();
        for gather in &gathers[..2] {
            sequencer
                .on_message(WhaleToSequencer::Ready {
                    seq: 0,
                    rank: gather.to,
                    count: 4,
                })
                .unwrap();
        }
        let cancels = sequencer.on_gather_timeout().unwrap();
        assert_eq!(cancels.len(), 4);
        assert!(
            cancels
                .iter()
                .all(|c| matches!(c.message, WhaleToMember::Cancel { seq: 0 }))
        );
        // The straggler's ready limps in afterward: stale, silently dropped.
        assert!(sequencer.on_message(reply).unwrap().is_empty());
        // The cancel disarms the member entirely.
        member
            .on_message(WhaleToMember::Cancel { seq: 0 }, 6)
            .unwrap();
        assert!(member.is_quiet());
        assert!(matches!(member.at_launch(7).unwrap(), WhaleDuty::Free));
        // And the sequencer lives on: the next whale gathers normally.
        let next = sequencer
            .on_message(WhaleToSequencer::Request {
                request: 4,
                poster: 0,
                prompt: prompt_of(12288),
            })
            .unwrap();
        assert_eq!(next.len(), 4);
    }

    #[test]
    fn inadmissible_request_is_refused_to_the_poster_alone() {
        let mut sequencer = WhaleSequencer::new(4, CHUNK);
        let out = sequencer
            .on_message(WhaleToSequencer::Request {
                request: 8,
                poster: 2,
                prompt: prompt_of(1024),
            })
            .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].to, 2);
        assert!(matches!(out[0].message, WhaleToMember::Cancel { .. }));
    }

    #[test]
    fn a_refused_whale_does_not_strand_the_queue_behind_it() {
        // Regression (found by the fuzzer): a runt queued ahead of a real
        // whale used to be refused without pumping further, and no later
        // message was guaranteed to arrive — the real whale sat in the
        // sequencer queue forever.
        let mut sequencer = WhaleSequencer::new(4, CHUNK);
        let gathers = sequencer
            .on_message(WhaleToSequencer::Request {
                request: 1,
                poster: 0,
                prompt: prompt_of(12288),
            })
            .unwrap();
        for runt_poster in [1, 2] {
            let queued = sequencer
                .on_message(WhaleToSequencer::Request {
                    request: 2,
                    poster: runt_poster,
                    prompt: prompt_of(64),
                })
                .unwrap();
            assert!(queued.is_empty(), "queued behind the gather");
        }
        let queued = sequencer
            .on_message(WhaleToSequencer::Request {
                request: 4,
                poster: 3,
                prompt: prompt_of(12288),
            })
            .unwrap();
        assert!(queued.is_empty());
        // The last ready commits whale 0, refuses both runts, and gathers the
        // real whale queued behind them — all in one transition.
        let mut tail = Vec::new();
        for gather in &gathers {
            tail = sequencer
                .on_message(WhaleToSequencer::Ready {
                    seq: 0,
                    rank: gather.to,
                    count: 3,
                })
                .unwrap();
        }
        let commits = tail
            .iter()
            .filter(|out| matches!(out.message, WhaleToMember::Commit { .. }))
            .count();
        let cancels: Vec<GlobalRank> = tail
            .iter()
            .filter(|out| matches!(out.message, WhaleToMember::Cancel { .. }))
            .map(|out| out.to)
            .collect();
        let gathers_after = tail
            .iter()
            .filter(|out| matches!(out.message, WhaleToMember::Gather { .. }))
            .count();
        assert_eq!(commits, 4);
        assert_eq!(cancels, vec![1, 2]);
        assert_eq!(gathers_after, 4, "the whale behind the runts must gather");
        assert_eq!(sequencer.gathering_seq(), Some(3));
    }

    // ---- protocol: things that must die loudly -----------------------------

    fn one_gather(sequencer: &mut WhaleSequencer, poster: GlobalRank) -> Vec<WhaleOutbound> {
        sequencer
            .on_message(WhaleToSequencer::Request {
                request: 1,
                poster,
                prompt: prompt_of(12288),
            })
            .unwrap()
    }

    #[test]
    fn corrupted_descriptor_dies_loudly() {
        let mut sequencer = WhaleSequencer::new(4, CHUNK);
        let gathers = one_gather(&mut sequencer, 0);
        let WhaleToMember::Gather { mut descriptor } = gathers[0].message.clone() else {
            panic!("expected gather");
        };
        descriptor.prompt_hash ^= 1;
        let to = gathers[0].to;
        let error = WhaleMember::new(to)
            .on_message(WhaleToMember::Gather { descriptor }, 0)
            .unwrap_err();
        assert!(error.to_string().contains("hash"), "{error}");
    }

    #[test]
    fn gather_addressed_to_a_non_member_dies_loudly() {
        let mut sequencer = WhaleSequencer::new(8, CHUNK);
        // Width-4 whale gangs ranks 0..4; rank 7 must refuse its descriptor.
        let gathers = sequencer
            .on_message(WhaleToSequencer::Request {
                request: 1,
                poster: 0,
                prompt: prompt_of(9000),
            })
            .unwrap();
        assert!(gathers.iter().all(|gather| gather.to != 7));
        let error = WhaleMember::new(7)
            .on_message(gathers[0].message.clone(), 0)
            .unwrap_err();
        assert!(error.to_string().contains("not in its gang"), "{error}");
    }

    #[test]
    fn duplicate_ready_dies_loudly() {
        let mut sequencer = WhaleSequencer::new(4, CHUNK);
        let gathers = one_gather(&mut sequencer, 0);
        let rank = gathers[0].to;
        sequencer
            .on_message(WhaleToSequencer::Ready {
                seq: 0,
                rank,
                count: 3,
            })
            .unwrap();
        let error = sequencer
            .on_message(WhaleToSequencer::Ready {
                seq: 0,
                rank,
                count: 4,
            })
            .unwrap_err();
        assert!(error.to_string().contains("twice"), "{error}");
    }

    #[test]
    fn an_unconsulted_launch_boundary_is_detected() {
        let (mut member, launch) = committed_member();
        let error = member.at_launch(launch + 1).unwrap_err();
        assert!(error.to_string().contains("unconsulted"), "{error}");
    }

    #[test]
    fn a_commit_arriving_past_its_launch_is_detected() {
        let mut sequencer = WhaleSequencer::new(4, CHUNK);
        let gathers = one_gather(&mut sequencer, 0);
        let rank = gathers[0].to;
        let mut member = WhaleMember::new(rank);
        member.on_message(gathers[0].message.clone(), 10).unwrap();
        // The member somehow launched past the committed slot before the
        // commit arrived: the slack failed, and silence would mispair a
        // collective — this must be the loud path.
        let error = member
            .on_message(
                WhaleToMember::Commit {
                    seq: 0,
                    launch: 10 + K3_WHALE_COMMIT_SLACK,
                },
                10 + K3_WHALE_COMMIT_SLACK,
            )
            .unwrap_err();
        assert!(error.to_string().contains("slack"), "{error}");
    }

    /// A member with a whale committed a few launches out.
    fn committed_member() -> (WhaleMember, LaunchCount) {
        let mut sequencer = WhaleSequencer::new(4, CHUNK);
        let gathers = one_gather(&mut sequencer, 0);
        let rank = gathers[0].to;
        let mut member = WhaleMember::new(rank);
        let reply = member
            .on_message(gathers[0].message.clone(), 10)
            .unwrap()
            .unwrap();
        let mut commits = sequencer.on_message(reply).unwrap();
        for gather in &gathers[1..] {
            commits.extend(
                sequencer
                    .on_message(WhaleToSequencer::Ready {
                        seq: 0,
                        rank: gather.to,
                        count: 10,
                    })
                    .unwrap(),
            );
        }
        let (_, launch) = commits.iter().find_map(commit_of).expect("commit");
        let mine = commits.iter().find(|commit| commit.to == rank).unwrap();
        member.on_message(mine.message.clone(), 11).unwrap();
        (member, launch)
    }

    #[test]
    fn a_pending_whale_pins_the_member_until_its_superstep() {
        // Committed but before the launch: free steps, not quiet — the
        // scheduler keeps a non-quiet member single-launch.
        let (mut member, launch) = committed_member();
        assert!(matches!(
            member.at_launch(launch - 3).unwrap(),
            WhaleDuty::Free
        ));
        assert!(!member.is_quiet());
        assert!(matches!(
            member.at_launch(launch).unwrap(),
            WhaleDuty::Enter(_)
        ));
        assert!(member.is_quiet());
        // Gathered alone (no commit yet) pins too: the pending commit may
        // name any launch above the reply.
        let mut fresh = WhaleMember::new(2);
        let gathers = one_gather(&mut WhaleSequencer::new(4, CHUNK), 0);
        let mine = gathers.into_iter().find(|gather| gather.to == 2).unwrap();
        fresh.on_message(mine.message, 0).unwrap();
        assert!(matches!(fresh.at_launch(1).unwrap(), WhaleDuty::Free));
        assert!(!fresh.is_quiet());
    }
}

/// Seeded fleet fuzz: a simulated free-running world where every rank
/// launches once per round (the mega pairing's ±1 pinning), messages ride
/// per-destination FIFO queues with 0–1 rounds of latency (TCP ordering,
/// latency below one launch period — the slack's stated assumption), and
/// unconstrained ranks start multi-launch operations that skip boundary
/// consultation exactly as a local chunked prefill does. The invariants are
/// the protocol's whole contract: every admissible whale enters unanimously —
/// full gang, one launch, distinct CP ranks — launches follow sequence order,
/// no transition errors, and the fleet quiesces.
#[cfg(test)]
mod fuzz {
    use std::collections::HashMap;

    use super::*;
    use crate::executor::cp::K3_WHALE_SEGMENT_FLOOR;

    struct Rng(u64);

    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed | 1)
        }

        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }

        fn below(&mut self, bound: u64) -> u64 {
            self.next() % bound
        }
    }

    struct SimRank {
        member: WhaleMember,
        count: LaunchCount,
        inbox: VecDeque<(u64, WhaleToMember)>,
        /// Launches left in the multi-launch operation in flight; boundaries
        /// inside it go unconsulted and the inbox stays undrained.
        busy: u64,
    }

    #[test]
    fn fuzzed_fleets_reach_unanimous_entries() {
        for seed in 1..=48u64 {
            fuzz_one(&mut Rng::new(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15)));
        }
    }

    fn fuzz_one(rng: &mut Rng) {
        const WORLD: usize = 8;
        const CHUNK: usize = 4224;
        let mut sequencer = WhaleSequencer::new(WORLD, CHUNK);
        let mut ranks: Vec<SimRank> = (0..WORLD)
            .map(|rank| SimRank {
                member: WhaleMember::new(rank),
                count: 0,
                inbox: VecDeque::new(),
                busy: 0,
            })
            .collect();
        let mut to_sequencer: VecDeque<(u64, WhaleToSequencer)> = VecDeque::new();
        let mut entered: Vec<(WhaleSeq, LaunchCount, usize, usize)> = Vec::new();
        let mut cancels = 0usize;
        let whales = 4 + rng.below(8) as usize;
        let mut posted = 0usize;
        let mut admissible = 0usize;
        let mut round = 0u64;

        loop {
            round += 1;
            assert!(round < 4000, "the fleet failed to quiesce");

            // Sometimes a rank posts a whale — usually a real one, sometimes
            // a runt the sequencer must refuse without wedging the queue.
            if posted < whales && rng.below(3) == 0 {
                let poster = rng.below(WORLD as u64) as usize;
                let span = (WORLD * CHUNK - 2 * K3_WHALE_SEGMENT_FLOOR) as u64;
                let total = if rng.below(4) != 0 {
                    2 * K3_WHALE_SEGMENT_FLOOR + rng.below(span) as usize
                } else {
                    1 + rng.below(K3_WHALE_SEGMENT_FLOOR as u64) as usize
                };
                if k3_whale_width(total, WORLD, CHUNK).is_some() {
                    admissible += 1;
                }
                posted += 1;
                to_sequencer.push_back((
                    round + rng.below(2),
                    WhaleToSequencer::Request {
                        request: posted as u64,
                        poster,
                        prompt: (0..total as u32).collect(),
                    },
                ));
            }

            // Rank phase: everyone launches exactly once, in a rotated order.
            let offset = rng.below(WORLD as u64) as usize;
            for i in 0..WORLD {
                let rank_id = (i + offset) % WORLD;
                let rank = &mut ranks[rank_id];
                if rank.busy > 0 {
                    rank.busy -= 1;
                    rank.count += 1;
                    continue;
                }
                while rank.inbox.front().is_some_and(|&(at, _)| at <= round) {
                    let (_, message) = rank.inbox.pop_front().expect("front just observed");
                    if matches!(message, WhaleToMember::Cancel { .. }) {
                        cancels += 1;
                    }
                    if let Some(reply) = rank
                        .member
                        .on_message(message, rank.count)
                        .expect("member transition")
                    {
                        to_sequencer.push_back((round + rng.below(2), reply));
                    }
                }
                match rank.member.at_launch(rank.count).expect("launch boundary") {
                    WhaleDuty::Enter(whale) => {
                        entered.push((
                            whale.descriptor.seq,
                            rank.count,
                            whale.cp_rank,
                            whale.descriptor.gang.len(),
                        ));
                        rank.count += 1;
                    }
                    WhaleDuty::Free => {
                        // Start an operation spanning 1..=cap launches; the
                        // boundaries inside it go unconsulted. A non-quiet
                        // member must stay single-launch — that pin is the
                        // production contract this fuzz drives.
                        let cap = if rank.member.is_quiet() { 3 } else { 1 };
                        rank.busy = rng.below(cap);
                        rank.count += 1;
                    }
                }
            }

            // Sequencer phase: drain what has arrived, route the outbound.
            while to_sequencer.front().is_some_and(|&(at, _)| at <= round) {
                let (_, message) = to_sequencer.pop_front().expect("front just observed");
                for out in sequencer.on_message(message).expect("sequencer transition") {
                    ranks[out.to]
                        .inbox
                        .push_back((round + rng.below(2), out.message));
                }
            }

            let quiet = posted == whales
                && to_sequencer.is_empty()
                && ranks
                    .iter()
                    .all(|rank| rank.inbox.is_empty() && rank.member.is_quiet());
            if quiet {
                break;
            }
        }

        // Unanimity: every admissible whale entered with its full gang at one
        // launch, CP ranks a permutation of 0..width.
        let mut by_seq: HashMap<WhaleSeq, Vec<(LaunchCount, usize, usize)>> = HashMap::new();
        for (seq, launch, cp_rank, width) in entered {
            by_seq
                .entry(seq)
                .or_default()
                .push((launch, cp_rank, width));
        }
        assert_eq!(by_seq.len(), admissible, "every admissible whale must run");
        assert_eq!(
            cancels,
            posted - admissible,
            "every runt must be refused once"
        );
        let mut launches: Vec<(WhaleSeq, LaunchCount)> = Vec::new();
        for (seq, entries) in &by_seq {
            let width = entries[0].2;
            assert_eq!(entries.len(), width, "whale {seq}: partial entry");
            let launch = entries[0].0;
            assert!(
                entries.iter().all(|&(at, _, _)| at == launch),
                "whale {seq}: split entry {entries:?}"
            );
            let mut cp_ranks: Vec<usize> = entries.iter().map(|&(_, cp_rank, _)| cp_rank).collect();
            cp_ranks.sort_unstable();
            assert_eq!(cp_ranks, (0..width).collect::<Vec<_>>(), "whale {seq}");
            launches.push((*seq, launch));
        }
        launches.sort_unstable();
        assert!(
            launches.windows(2).all(|pair| pair[0].1 < pair[1].1),
            "commit order must follow sequence order: {launches:?}"
        );
    }
}
