//! Runtime self-pricing for the hedge ladder (`PEGAINFER_SPEC_HEDGE_AUTO`).
//!
//! Instead of a hand-picked chain count, the lane measures its own operating
//! point: a windowed explore-then-commit controller cycles the chain count
//! `C ∈ {0..=Cmax}` (prefixes of `PEGAINFER_SPEC_HEDGE_POSITIONS`), scores
//! each window by delivered **committed tokens per second** — the objective
//! itself, so bucket quantization, eager penalties, and host overhead are all
//! priced in automatically — and commits to the argmax. A commit expires after
//! [`COMMIT_ROUNDS`] and re-explores, so the choice follows workload drift
//! (hard traffic converges to the ladder, easy traffic to `C = 0`). Per
//! candidate the last [`MAX_WINDOWS`] windows are kept, unweighted: after a
//! long commit the incumbent's windows are all recent while the challengers'
//! are not, which biases re-exploration toward the incumbent. Round timing uses inter-round host timestamps; gaps above
//! [`ROUND_GAP_CUTOFF`] (prefill, admission stalls) are treated as window
//! boundaries and never attributed to a chain count.

use std::time::Duration;
use std::time::Instant;

/// Rounds per exploration window. Each candidate C holds the lane for this
/// many verify rounds before rotating; the first rounds of a fresh shape pay
/// graph warm-up, which the window length amortizes.
const WINDOW_ROUNDS: u32 = 64;
/// Full rotations over all candidates before committing.
const EXPLORE_CYCLES: u32 = 3;
/// Rounds to hold a committed choice before re-exploring.
const COMMIT_ROUNDS: u32 = 4096;
/// Cheaper re-exploration after the first commit (per-candidate window).
const REEXPLORE_WINDOW_ROUNDS: u32 = 32;
/// Inter-round gaps above this are scheduling boundaries, not round cost.
const ROUND_GAP_CUTOFF: Duration = Duration::from_millis(500);

/// Extra throughput a larger chain count must show over the best smaller one
/// (median window rate) before it is preferred. Below this the controller
/// takes the cheaper configuration — on a flat landscape (easy traffic) the
/// argmax is noise, and the noise of a 64-round window runs several percent.
const ADD_CHAIN_MARGIN: f64 = 1.03;
/// Completed windows kept per candidate (older ones age out).
const MAX_WINDOWS: usize = 12;

#[derive(Default, Clone)]
struct CandidateStats {
    /// Completed exploration windows, `(tokens, secs)` each. Scored by the
    /// MEDIAN of per-window rates — one polluted window (graph warm-up, a
    /// straggling request) cannot swing the commit the way pooled sums can.
    windows: Vec<(f64, f64)>,
}

impl CandidateStats {
    fn median_rate(&self) -> Option<f64> {
        let mut rates: Vec<f64> = self
            .windows
            .iter()
            .filter(|(_, secs)| *secs > 0.0)
            .map(|(tokens, secs)| tokens / secs)
            .collect();
        if rates.is_empty() {
            return None;
        }
        rates.sort_by(f64::total_cmp);
        Some(rates[rates.len() / 2])
    }
}

pub(super) struct AutoHedge {
    stats: Vec<CandidateStats>,
    cur_tokens: f64,
    cur_secs: f64,
    /// Executed chain count of the in-progress window. Capacity guards can
    /// force a round to run below the candidate; a window only scores if
    /// every round in it executed the same count, so those rounds are booked
    /// to the configuration that actually ran.
    window_c: usize,
    cur_c: usize,
    committed: Option<usize>,
    rounds_in_window: u32,
    window_rounds: u32,
    cycles_done: u32,
    commit_left: u32,
    last_tick: Option<Instant>,
}

impl AutoHedge {
    pub(super) fn new(max_chains: usize) -> Self {
        Self {
            stats: vec![CandidateStats::default(); max_chains + 1],
            cur_tokens: 0.0,
            cur_secs: 0.0,
            window_c: 0,
            cur_c: 0,
            committed: None,
            rounds_in_window: 0,
            window_rounds: WINDOW_ROUNDS,
            cycles_done: 0,
            commit_left: 0,
            last_tick: None,
        }
    }

    /// Chain count for the next draft round (0 = no hedging this round).
    pub(super) fn current_c(&self) -> usize {
        self.committed.unwrap_or(self.cur_c)
    }

    /// Record one verify round's outcome. `executed_c` is the chain count
    /// the round actually ran — 0 when a guard disabled hedging, whatever
    /// [`Self::current_c`] requested.
    pub(super) fn tick(&mut self, executed_c: usize, committed_tokens: usize) {
        let now = Instant::now();
        let Some(prev) = self.last_tick.replace(now) else {
            return;
        };
        let elapsed = now - prev;
        if elapsed > ROUND_GAP_CUTOFF {
            // Documented boundary: a stall (prefill, admission) severs the
            // window. Continuing it would mix measurements across whatever
            // the traffic looked like on either side of the stall.
            self.cur_tokens = 0.0;
            self.cur_secs = 0.0;
            self.rounds_in_window = 0;
            return;
        }
        if executed_c >= self.stats.len() {
            return;
        }
        if self.rounds_in_window == 0 {
            self.window_c = executed_c;
        } else if executed_c != self.window_c {
            self.cur_tokens = 0.0;
            self.cur_secs = 0.0;
            self.rounds_in_window = 0;
            self.window_c = executed_c;
        }
        self.cur_tokens += committed_tokens as f64;
        self.cur_secs += elapsed.as_secs_f64();
        self.rounds_in_window += 1;
        let window_done = self.rounds_in_window >= self.window_rounds;
        if window_done {
            let c = self.window_c;
            self.stats[c].windows.push((self.cur_tokens, self.cur_secs));
            if self.stats[c].windows.len() > MAX_WINDOWS {
                self.stats[c].windows.remove(0);
            }
            self.cur_tokens = 0.0;
            self.cur_secs = 0.0;
            self.rounds_in_window = 0;
        }

        if let Some(held) = self.committed {
            self.commit_left = self.commit_left.saturating_sub(1);
            if self.commit_left == 0 {
                self.committed = None;
                self.cur_c = 0;
                self.cur_tokens = 0.0;
                self.cur_secs = 0.0;
                self.rounds_in_window = 0;
                self.cycles_done = 0;
                self.window_rounds = REEXPLORE_WINDOW_ROUNDS;
                log::info!("spec hedge auto: re-exploring (held C={held})");
            }
            return;
        }
        if !window_done {
            return;
        }
        if self.cur_c + 1 < self.stats.len() {
            self.cur_c += 1;
            return;
        }
        self.cur_c = 0;
        self.cycles_done += 1;
        if self.cycles_done < EXPLORE_CYCLES {
            return;
        }
        // Commit. Walk candidates from cheapest up; a larger chain count must
        // beat the running best's median window rate by ADD_CHAIN_MARGIN to
        // be preferred — uncertainty resolves toward fewer chains, so a flat
        // landscape (easy traffic) settles at the cheap end instead of the
        // noise argmax.
        let mut best = 0usize;
        let mut best_rate = self.stats[0].median_rate().unwrap_or(0.0);
        for (c, s) in self.stats.iter().enumerate().skip(1) {
            let Some(rate) = s.median_rate() else {
                continue;
            };
            if rate > best_rate * ADD_CHAIN_MARGIN {
                best = c;
                best_rate = rate;
            }
        }
        let rates: Vec<String> = self
            .stats
            .iter()
            .map(|s| {
                s.median_rate()
                    .map_or("-".to_string(), |r| format!("{r:.0}"))
            })
            .collect();
        log::info!(
            "spec hedge auto: committed C={best} (median window tok/s by C: [{}])",
            rates.join(", ")
        );
        self.committed = Some(best);
        self.commit_left = COMMIT_ROUNDS;
        self.cycles_done = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn drive(auto: &mut AutoHedge, rounds: u32, tokens_for: impl Fn(usize) -> usize) {
        for _ in 0..rounds {
            auto.last_tick = Instant::now().checked_sub(Duration::from_millis(50));
            let c = auto.current_c();
            auto.tick(c, tokens_for(c));
        }
    }

    #[test]
    fn explores_all_candidates_then_commits_to_best() {
        let mut auto = AutoHedge::new(3);
        // C=2 yields the most tokens per (constant) round time.
        drive(
            &mut auto,
            WINDOW_ROUNDS * 4 * EXPLORE_CYCLES + 1,
            |c| match c {
                2 => 30,
                3 => 28,
                _ => 20,
            },
        );
        assert_eq!(auto.committed, Some(2));
    }

    #[test]
    fn commit_expires_into_reexploration() {
        let mut auto = AutoHedge::new(1);
        drive(&mut auto, WINDOW_ROUNDS * 2 * EXPLORE_CYCLES + 1, |c| {
            if c == 1 { 30 } else { 20 }
        });
        assert_eq!(auto.committed, Some(1));
        drive(&mut auto, COMMIT_ROUNDS, |_| 30);
        assert_eq!(auto.committed, None);
        assert_eq!(auto.window_rounds, REEXPLORE_WINDOW_ROUNDS);
    }

    #[test]
    fn long_gaps_sever_the_window() {
        let mut auto = AutoHedge::new(1);
        // Seed a half-built window, then stall: the gap round must not be
        // attributed AND the pre-stall partial window must be discarded.
        drive(&mut auto, 10, |_| 25);
        assert!(auto.cur_tokens > 0.0);
        auto.last_tick = Instant::now().checked_sub(Duration::from_secs(2));
        auto.tick(0, 30);
        assert!(auto.cur_tokens.abs() < f64::EPSILON);
        assert_eq!(auto.rounds_in_window, 0);
    }

    #[test]
    fn guard_disabled_rounds_are_booked_to_executed_config() {
        let mut auto = AutoHedge::new(2);
        // Every round runs C=0 regardless of the candidate (guard always
        // trips): only stats[0] may accumulate windows.
        drive_executed(&mut auto, WINDOW_ROUNDS * 3 * EXPLORE_CYCLES + 1, 0, 25);
        assert!(auto.stats[1].windows.is_empty());
        assert!(auto.stats[2].windows.is_empty());
        assert_eq!(auto.committed, Some(0));
    }

    fn drive_executed(auto: &mut AutoHedge, rounds: u32, executed: usize, tokens: usize) {
        for _ in 0..rounds {
            auto.last_tick = Instant::now().checked_sub(Duration::from_millis(50));
            auto.tick(executed, tokens);
        }
    }

    #[test]
    fn flat_landscape_settles_cheap() {
        let mut auto = AutoHedge::new(3);
        drive(&mut auto, WINDOW_ROUNDS * 4 * EXPLORE_CYCLES + 1, |_| 25);
        assert_eq!(auto.committed, Some(0));
    }
}
