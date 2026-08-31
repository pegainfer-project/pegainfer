//! The CP prefill gang: how independent scheduler threads agree to run one
//! prompt's context-parallel prefill together.
//!
//! Every scheduler partition steps free-running — the EP mega launches pair
//! two-sided by absolute index across ranks, so a step only completes once
//! every rank has queued its paired launch, and no rank can ever run more
//! than one step ahead. A CP prefill is the one moment the partitions must
//! act as one gang: all of them inside [`StepExecutor::prefill_cp`] for the
//! same prompt, entering at the **same absolute launch count** — the chunk
//! step's exchange windows sync the local stream mid-step, and entered
//! unequal, the leading rank would wait there on peer launches the others,
//! parked at the window's CPU barrier, will never queue.
//!
//! The gang is a posted-job board plus one discipline: **a member never
//! waits quietly; it pumps padding steps until the whole gang is provably
//! level, then computes.** Waiting quietly deadlocks — a peer may be blocked
//! inside its own step's device sync, which only completes once this rank
//! queues the matching launch. Pumping cannot run away either: the pump's
//! own sync is the same two-sided pairing, so it blocks until every peer
//! steps. Each member re-posts its launch count on the board every pass;
//! once every member has checked in and this member's count equals the
//! board's maximum, everyone else is either at the same count or pumping up
//! to it, and the member enters the compute.
//!
//! Jobs run in posting order on every partition — one global order, or two
//! concurrent gangs could interleave their exchange windows.

use std::sync::Arc;
use std::sync::Mutex;

use anyhow::Context as _;
use anyhow::Result;

use super::executor::SlotId;
use super::executor::StepExecutor;
use crate::executor::cp::K3CpGroup;

/// One posted CP prefill.
struct GangJob {
    id: u64,
    prompt: Arc<[u32]>,
    /// Partition that posted the job — the gang's owner (CP rank `size-1`).
    poster: usize,
    /// The poster's slot the prompt lands in. Meaningful only on the
    /// poster's executor; the other partitions run stateless segments.
    slot: SlotId,
    /// Per-partition entry counts: `None` until the partition reaches this
    /// job at its board check, then the launch count it is at (or is pumping
    /// toward). The gang computes once every entry equals the max.
    entries: Vec<Option<u64>>,
    /// Which partitions have finished their `prefill_cp` for it.
    executed: Vec<bool>,
}

#[derive(Default)]
struct GangBoard {
    next_id: u64,
    /// Pending jobs in posting order. A job leaves the board when the last
    /// partition finishes executing it.
    jobs: Vec<GangJob>,
}

impl GangBoard {
    /// The entry for `id`. A job leaves the board only after every partition
    /// executed it, so a participant mid-protocol always finds it.
    fn job_mut(&mut self, id: u64) -> &mut GangJob {
        self.jobs
            .iter_mut()
            .find(|job| job.id == id)
            .expect("a job leaves the board only after everyone executed it")
    }
}

/// The shared gang handle: one per engine, cloned into every scheduler.
pub struct K3CpGang {
    group: Arc<K3CpGroup>,
    size: usize,
    board: Mutex<GangBoard>,
}

impl std::fmt::Debug for K3CpGang {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("K3CpGang")
            .field("size", &self.size)
            .finish_non_exhaustive()
    }
}

impl K3CpGang {
    pub fn new(group: Arc<K3CpGroup>) -> Arc<Self> {
        let size = group.cp_size();
        Arc::new(Self {
            group,
            size,
            board: Mutex::new(GangBoard::default()),
        })
    }

    pub fn size(&self) -> usize {
        self.size
    }

    /// The CP rank partition `p` takes in a gang owned by `poster`: the
    /// poster is the owner (last rank), the rest follow in ascending
    /// partition order.
    fn cp_rank(&self, partition: usize, poster: usize) -> usize {
        if partition == poster {
            self.size - 1
        } else {
            partition - usize::from(partition > poster)
        }
    }

    /// Post `prompt` as partition `poster` and serve the board until the
    /// posted job has run here. The poster is the job's owner, so its own
    /// `prefill_cp` ingests the prompt into `slot` and returns the boundary
    /// token. Jobs posted earlier are served on the way — the global order
    /// is what keeps two concurrent gangs from crossing exchanges.
    pub fn post_and_run<E>(
        &self,
        poster: usize,
        slot: SlotId,
        prompt: Arc<[u32]>,
        executor: &mut E,
    ) -> Result<u32>
    where
        E: StepExecutor + ?Sized,
    {
        let own_id = {
            let mut board = self.board.lock().expect("gang board poisoned");
            let id = board.next_id;
            board.next_id += 1;
            board.jobs.push(GangJob {
                id,
                prompt,
                poster,
                slot,
                entries: vec![None; self.size],
                executed: vec![false; self.size],
            });
            id
        };
        self.run_jobs(poster, Some(own_id), executor)?
            .context("the owner's CP prefill must sample the boundary token")
    }

    /// Serve every job this partition has not run yet, in posting order.
    /// Called at the top of every scheduler step; returns immediately when
    /// nothing is pending.
    pub fn serve<E>(&self, partition: usize, executor: &mut E) -> Result<()>
    where
        E: StepExecutor + ?Sized,
    {
        self.run_jobs(partition, None, executor).map(|_| ())
    }

    /// Run this partition's pending jobs in posting order, up to a horizon:
    /// `own` (the poster's just-posted job), or for a serving pass the last
    /// job already on the board. Jobs posted while this runs wait for the
    /// next pass — without the horizon a partition could be trapped serving
    /// a stream of peers' postings while its own decode stalls. Returns the
    /// boundary token of `own`, when given.
    fn run_jobs<E>(
        &self,
        partition: usize,
        own: Option<u64>,
        executor: &mut E,
    ) -> Result<Option<u32>>
    where
        E: StepExecutor + ?Sized,
    {
        let mut token = None;
        let mut board = self.board.lock().expect("gang board poisoned");
        let Some(horizon) = own.or_else(|| board.jobs.last().map(|job| job.id)) else {
            return Ok(None);
        };
        // The lowest-id job this partition has not executed, each pass. Every
        // partition picks jobs this way, so the execution order is the
        // posting order everywhere.
        while let Some(id) = board
            .jobs
            .iter()
            .find(|job| job.id <= horizon && !job.executed[partition])
            .map(|job| job.id)
        {
            // Level up: pump until every partition has checked in and this
            // one's launch count is the board's maximum. An entry posted
            // before a pump names the count the pump reaches, so the maximum
            // a member reads always covers its peers' in-flight pumps and the
            // whole gang converges on one entry count.
            loop {
                let count = executor.step_count();
                let job = board.job_mut(id);
                if job.entries[partition].is_none() {
                    log::info!("K3 CP gang job {id}: partition {partition} arrived at {count}");
                }
                job.entries[partition] = Some(count);
                let level = job
                    .entries
                    .iter()
                    .copied()
                    .try_fold(0u64, |max, entry| entry.map(|entry| max.max(entry)));
                if level == Some(count) {
                    break;
                }
                job.entries[partition] = Some(count + 1);
                drop(board);
                executor.pump_step()?;
                anyhow::ensure!(
                    executor.step_count() > count,
                    "pump_step did not advance the launch count; the gang cannot level"
                );
                board = self.board.lock().expect("gang board poisoned");
            }
            let job = board.job_mut(id);
            let prompt = job.prompt.clone();
            let cp_rank = self.cp_rank(partition, job.poster);
            let owner = job.poster == partition;
            let slot = if owner { job.slot } else { 0 };
            drop(board);
            log::info!(
                "K3 CP gang job {id}: partition {partition} computing as cp_rank {cp_rank} \
                 ({} tokens, {} launches)",
                prompt.len(),
                executor.step_count(),
            );
            let sampled = executor.prefill_cp(slot, &prompt, &self.group, cp_rank);
            // Mark executed even on error: the scheduler's step fails on the
            // propagated error and the engine winds down, but a peer partition
            // still serving the board must not find a permanently unrunnable
            // job wedging every later one.
            board = self.board.lock().expect("gang board poisoned");
            board.job_mut(id).executed[partition] = true;
            board
                .jobs
                .retain(|job| !job.executed.iter().all(|&executed| executed));
            if owner {
                token = Some(sampled?.context("the owner's CP prefill returned no token")?);
            } else {
                sampled?;
            }
        }
        Ok(token)
    }
}
