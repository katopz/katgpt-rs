//! Plan 621 Phase 4 — the healer-lane Super-GOAT eval engine (T4.1–T4.4).
//!
//! A deterministic, seed-pinned simulation of the refine-style healer lane's
//! fix-miss decision, comparing three arms over ONE planted corpus and ONE
//! shared per-state outcome stream (paired comparison — policies differ only
//! in their decision rules, never in their draws):
//!
//! 1. **never-escalate floor** — draws local candidates to `N_MAX`, never
//!    escalates. The G1 control: the corpus must contain states whose only
//!    rescue is escalation, so the floor must lose real rescues (a gate that
//!    never fires is indistinguishable from this arm — T4.1's failure mode).
//! 2. **incumbent (`modelless_cap` alone)** — the standard modelless pool
//!    (`POOL` candidates) is the fix-miss event; on a miss, escalate while
//!    the escalation-budget cap allows. No probe, no extended local draws.
//! 3. **probe-gated (the fusion)** — draws adaptively in `POOL`-sized
//!    batches, probing the accumulated all-fail pool after each batch; a
//!    pass at any point IS the local rescue (the loop exits — a verifying
//!    candidate is the fixed fix). `Settled(KnowledgeLike)` escalates;
//!    `CappedUndetermined` takes the conservative non-escalating stop. The
//!    T4.3 cost line arms the escalation: if the probe's accumulated
//!    evaluations exceed `PROBE_COST_FRAC` of the escalation cost, the gate
//!    FALLS BACK to the incumbent decision (escalate) and counts the breach
//!    loudly — never a silent budget overrun.
//!
//! Cost model (units of one draft+verify evaluation, `C_PROBE = 1.0`): a
//! local draw is one candidate generated + verified (the healer's compile
//! gate); an escalation is the hosted-expert call (`c_depth`, swept — the
//! verdict is asserted across the sweep, never at one tuned constant).
//!
//! Determinism: every f64 op is sequential IEEE-754 (no threading, no
//! transcendentals beyond the kernel's own `ln`/`sqrt`), the LCG is the
//! T2-fixture constants, and the comparison table serializes through
//! `f64::to_bits` — the table's blake3 is replayable across runs AND
//! across profiles (debug test / release bench), which `t44` pins.
//!
//! Eval-only harness code: never a shipped API, never compiled outside the
//! gated targets that include it.

#![cfg(feature = "escalation_probe_gate")]
// Each harness consumes a subset (the plan437_stage0_common convention):
// the test suite reads the per-state results/kinds; the bench asserts on
// the aggregates only.
#![allow(dead_code)]

use katgpt_core::escalation_probe_gate::{EscalationOption, GateDecision, GateInput, gate};
use katgpt_core::state_probe::{
    AdaptiveClassifier, AdaptiveDecision, ClassFlip, ProbeClass, ProbeInput, probe,
};

// ── the pinned constants (recorded in .benchmarks; the sweep owns c_depth) ──

/// The two-sided 95% Wilson critical value (the T1/T2 fixture value).
pub const Z95: f64 = 1.959_963_984_540_054;
/// The execution-reachability bar (T2.2's ε).
pub const EPSILON: f64 = 0.1;
/// The corpus-distance penalty (FlyBy's 0.1; the recorded λ=0.2 failure is
/// the ceiling). Single-depth ladder: λ shapes the reported utility, never
/// the decision (no second depth to re-rank onto).
pub const LAMBDA: f64 = 0.1;
/// The escalation success bar for the declared prior.
pub const R_MIN: f64 = 0.5;
/// One local draw (candidate draft + verify) — the spend unit.
pub const C_PROBE: f64 = 1.0;
/// The incumbent's standard modelless pool — the fix-miss event when it
/// all-fails. Every arm pays this pool (the common heal attempt).
pub const POOL: u32 = 8;
/// The local-draw cap for the arms that keep drawing (floor + probe).
pub const N_MAX: u32 = 64;
/// T4.3: the probe's accumulated evaluations may cost at most this fraction
/// of the gated (escalation) decision before the loud fallback fires.
pub const PROBE_COST_FRAC: f64 = 0.10;
/// The lane's offline-measured escalation prior (what the consumer declares
/// on the ladder — NOT the per-state true rate, which the outcome stream
/// draws so escalations can honestly fail).
pub const DECLARED_PRIOR: f64 = 0.7;
/// The corpus seed (the plan-numbered 621 family).
pub const CORPUS_SEED: u64 = 0x0621_0400;

/// How many states of each class the planted corpus carries (T4.1: BOTH
/// classes present, plus the ε-straddle band that exercises the
/// conservative stop in a budget-capped posture).
pub const N_EXEC: usize = 100;
pub const N_KNOWLEDGE: usize = 100;
pub const N_MIDDLE: usize = 50;

// ── the corpus ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// True per-candidate verify rate ∈ [0.15, 0.45] — locally reachable.
    Exec,
    /// True rate ∈ [0.0, 0.03] — the fix needs external knowledge.
    Knowledge,
    /// True rate ∈ [0.08, 0.13] — straddles ε; the honest middle band.
    Middle,
}

#[derive(Debug, Clone, Copy)]
pub struct State {
    pub kind: Kind,
    /// True per-candidate verify rate (the local channel's rate).
    pub p_local: f64,
    /// True escalation solve rate (the outcome draws make escalations
    /// honestly fail against it; the LADDER's declared prior stays fixed).
    pub r_esc: f64,
    /// Corpus distance ∈ [0, 1] — consumer-supplied in production; drawn
    /// here so the reported utilities carry a real distance axis.
    pub corpus_distance: f64,
}

pub struct Corpus {
    pub states: Vec<State>,
}

/// Deterministic uniform [0,1) — the T2-fixture LCG constants, no global
/// state. (A 12-line test-engine copy of the fixture's generator; the
/// alternative — a shared test-support module — is heavier than the
/// duplication. The constants are pinned in both files.)
pub struct Lcg(u64);
impl Lcg {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 40) as f64 / (1u64 << 24) as f64
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
}

/// Generate the planted corpus. Exec: p ~ U(0.15, 0.45); Knowledge:
/// p ~ U(0.0, 0.03); Middle: p ~ U(0.08, 0.13). Every state draws its
/// escalation solve rate r ~ U(0.55, 0.9) and distance Ĉ ~ U(0, 1).
pub fn gen_corpus() -> Corpus {
    let mut rng = Lcg::new(CORPUS_SEED);
    let mut states = Vec::with_capacity(N_EXEC + N_KNOWLEDGE + N_MIDDLE);
    for _ in 0..N_EXEC {
        states.push(State {
            kind: Kind::Exec,
            p_local: rng.range(0.15, 0.45),
            r_esc: rng.range(0.55, 0.9),
            corpus_distance: rng.unit(),
        });
    }
    for _ in 0..N_KNOWLEDGE {
        states.push(State {
            kind: Kind::Knowledge,
            p_local: rng.range(0.0, 0.03),
            r_esc: rng.range(0.55, 0.9),
            corpus_distance: rng.unit(),
        });
    }
    for _ in 0..N_MIDDLE {
        states.push(State {
            kind: Kind::Middle,
            p_local: rng.range(0.08, 0.13),
            r_esc: rng.range(0.55, 0.9),
            corpus_distance: rng.unit(),
        });
    }
    Corpus { states }
}

/// The per-state OUTCOME stream — pre-drawn once so every arm replays the
/// SAME draws (paired comparison): 64 local-candidate uniforms (pass iff
/// `u < p_local`) + one escalation-outcome uniform (rescued iff
/// `u < r_esc`). The escalation uniform sits at a fixed stream offset, so
/// an arm's draw path can never move it.
pub struct OutcomeStream {
    /// `[state_index][draw_index]` — 64 local draws per state.
    local: Vec<[f64; N_MAX as usize]>,
    esc: Vec<f64>,
}

impl OutcomeStream {
    pub fn draw(corpus: &Corpus) -> Self {
        let n = corpus.states.len();
        let mut local = Vec::with_capacity(n);
        let mut esc = Vec::with_capacity(n);
        for (i, s) in corpus.states.iter().enumerate() {
            let mut rng = Lcg::new(CORPUS_SEED ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let mut row = [0.0f64; N_MAX as usize];
            for slot in &mut row {
                *slot = rng.unit();
            }
            // Each draw passes iff u < p — threshold ONCE here so the arms
            // consume booleans (identical semantics, no per-arm re-deriving).
            local.push(row);
            esc.push(rng.unit());
            let _ = s;
        }
        Self { local, esc }
    }

    /// Did candidate `draw_index` (0-based) of state `i` verify?
    #[inline]
    pub fn candidate_passed(&self, i: usize, draw_index: u32, p_local: f64) -> bool {
        self.local[i][draw_index as usize] < p_local
    }

    /// Did the escalation of state `i` rescue the fix?
    #[inline]
    pub fn escalation_rescued(&self, i: usize, r_esc: f64) -> bool {
        self.esc[i] < r_esc
    }
}

// ── the arms ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// A drafted candidate verified — the fix landed locally.
    LocalRescue,
    /// The gate (or the incumbent's cap rule) sent the state to the hosted
    /// expert. `breach_fallback = true` marks a T4.3 fallback.
    Escalate { breach_fallback: bool },
    /// Conservative non-escalating stop (capped-undetermined, or the
    /// incumbent's cap refused): the fix stays missed.
    Stay,
}

#[derive(Debug, Clone, Copy)]
pub struct ActionResult {
    pub action: Action,
    /// Local draws consumed (the pool counts — every arm pays ≥ POOL).
    pub draws_used: u32,
    pub rescued: bool,
    pub spend: f64,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct ArmOutcome {
    pub escalations: u32,
    pub breach_fallbacks: u32,
    pub rescued: u32,
    pub spend: f64,
    /// Probe observations taken (probe arm only) — the coverage denominator.
    pub probe_observations: u64,
    /// Observations where the true rate sat inside the Wilson interval.
    pub coverage_hits: u64,
    /// Local draws consumed across the run (the probe's own cost axis).
    pub total_draws: u64,
}

/// One arm run over the whole corpus: per-state results in corpus order +
/// the aggregate outcome.
pub struct ArmRun {
    pub results: Vec<ActionResult>,
    pub outcome: ArmOutcome,
}

fn finish(results: Vec<ActionResult>, mut outcome: ArmOutcome) -> ArmRun {
    for r in &results {
        if matches!(r.action, Action::Escalate { .. }) {
            outcome.escalations += 1;
        }
        if r.rescued {
            outcome.rescued += 1;
        }
        outcome.spend += r.spend;
        outcome.total_draws += u64::from(r.draws_used);
    }
    ArmRun { results, outcome }
}

/// Arm 1 — the never-escalate floor: draw to `N_MAX`, take the first pass,
/// never escalate. The G1 control (a gate that never fires scores this).
pub fn run_floor(corpus: &Corpus, draws: &OutcomeStream) -> ArmRun {
    let mut results = Vec::with_capacity(corpus.states.len());
    for (i, s) in corpus.states.iter().enumerate() {
        let mut used = 0u32;
        let mut rescued = false;
        while used < N_MAX {
            if draws.candidate_passed(i, used, s.p_local) {
                rescued = true;
                used += 1;
                break;
            }
            used += 1;
        }
        let r = used as f64 * C_PROBE;
        results.push(ActionResult {
            action: if rescued {
                Action::LocalRescue
            } else {
                Action::Stay
            },
            draws_used: used,
            rescued,
            spend: r,
        });
    }
    finish(results, ArmOutcome::default())
}

/// Arm 2 — the incumbent (`modelless_cap` alone): the standard pool is the
/// fix-miss event; on a miss, escalate while `cap` (the escalation budget,
/// `None` = unlimited) allows; a refused escalation leaves the fix missed.
pub fn run_incumbent_cap(
    corpus: &Corpus,
    draws: &OutcomeStream,
    c_depth: f64,
    cap: Option<f64>,
) -> ArmRun {
    let mut results = Vec::with_capacity(corpus.states.len());
    let mut esc_spend = 0.0;
    for (i, s) in corpus.states.iter().enumerate() {
        let mut passed = false;
        for d in 0..POOL {
            if draws.candidate_passed(i, d, s.p_local) {
                passed = true;
                break;
            }
        }
        let (action, rescued, spend) = if passed {
            (Action::LocalRescue, true, POOL as f64 * C_PROBE)
        } else {
            let within_cap = cap.is_none_or(|c| esc_spend + c_depth <= c);
            if within_cap {
                esc_spend += c_depth;
                let rescued = draws.escalation_rescued(i, s.r_esc);
                (
                    Action::Escalate {
                        breach_fallback: false,
                    },
                    rescued,
                    POOL as f64 * C_PROBE + c_depth,
                )
            } else {
                (Action::Stay, false, POOL as f64 * C_PROBE)
            }
        };
        results.push(ActionResult {
            action,
            rescued,
            draws_used: POOL,
            spend,
        });
    }
    finish(results, ArmOutcome::default())
}

/// Arm 3 — the probe-gated fusion: adaptive local draws with a probe
/// observation on the accumulated all-fail pool after each batch; a pass
/// exits as the local rescue; `Settled(KnowledgeLike)` escalates (T4.3-armed);
/// `CappedUndetermined` takes the conservative stop.
///
/// `histogram: &[]` is deliberate: fix candidates carry no answer classes —
/// the healer lane reads the pass-rate interval, the entropy axis stays at
/// its answers-not-tracked zero (the kernel's documented contract).
pub fn run_probe_gated(corpus: &Corpus, draws: &OutcomeStream, c_depth: f64, n_max: u32) -> ArmRun {
    let mut results = Vec::with_capacity(corpus.states.len());
    let mut outcome = ArmOutcome::default();
    let mut flip_log = [ClassFlip {
        at_n: 0,
        from: ProbeClass::Undetermined,
        to: ProbeClass::Undetermined,
        interval_width: 0.0,
    }; 8];

    for (i, s) in corpus.states.iter().enumerate() {
        let mut cl = AdaptiveClassifier::new(EPSILON, n_max, &mut flip_log);
        let mut action = Action::Stay;
        let mut rescued = false;
        let mut done = false;
        let mut n = 0u32;
        while !done && n < n_max {
            // One batch of POOL candidates.
            for _ in 0..POOL.min(n_max - n) {
                if draws.candidate_passed(i, n, s.p_local) {
                    action = Action::LocalRescue;
                    rescued = true;
                    n += 1;
                    done = true;
                    break;
                }
                n += 1;
            }
            if done {
                break;
            }
            // Probe the accumulated all-fail pool.
            let est = probe(
                &ProbeInput {
                    pass_count: 0,
                    n,
                    histogram: &[],
                },
                Z95,
            );
            outcome.probe_observations += 1;
            if s.p_local >= est.wilson_lo && s.p_local <= est.wilson_hi {
                outcome.coverage_hits += 1;
            }
            match cl.observe(&est, n) {
                AdaptiveDecision::Settled(ProbeClass::KnowledgeLike) => {
                    // T4.3: the cost line arms the escalation decision.
                    let probe_cost = f64::from(n) * C_PROBE;
                    let breach = probe_cost > PROBE_COST_FRAC * c_depth;
                    if breach {
                        // Loud fallback to the incumbent DECISION (escalate
                        // the fix-miss) — the budget is disclosed, never
                        // silently eaten.
                        outcome.breach_fallbacks += 1;
                    } else {
                        // The full gate: escalate iff the utility says so
                        // (the corpus distance rides the state).
                        let ladder_state = [EscalationOption {
                            depth: 1,
                            success_prior: DECLARED_PRIOR,
                            corpus_distance: s.corpus_distance,
                        }];
                        let g = gate(
                            &GateInput {
                                probe: est,
                                options: &ladder_state,
                                lambda: LAMBDA,
                            },
                            EPSILON,
                            R_MIN,
                        );
                        match g {
                            GateDecision::Escalate { .. } => {}
                            GateDecision::StayLocal => {
                                // λ discounted the whole ladder out: the
                                // conservative stay (no escalation spend).
                                action = Action::Stay;
                                done = true;
                            }
                        }
                    }
                    if !done {
                        rescued = draws.escalation_rescued(i, s.r_esc);
                        action = Action::Escalate {
                            breach_fallback: breach,
                        };
                        done = true;
                    }
                }
                AdaptiveDecision::Settled(ProbeClass::Productive) => {
                    // Unreachable on all-fail pools (wilson_lo = 0 < ε) —
                    // handled conservatively, never a panic.
                    action = Action::Stay;
                    done = true;
                }
                AdaptiveDecision::CappedUndetermined => {
                    action = Action::Stay;
                    done = true;
                }
                AdaptiveDecision::Settled(ProbeClass::Undetermined) => {
                    // Unreachable by construction (Settled only fires on
                    // decisive classes) — handled conservatively, never a
                    // panic.
                    action = Action::Stay;
                    done = true;
                }
                AdaptiveDecision::Continue => {}
            }
        }
        let spend = f64::from(n) * C_PROBE
            + if matches!(action, Action::Escalate { .. }) {
                c_depth
            } else {
                0.0
            };
        results.push(ActionResult {
            action,
            rescued,
            draws_used: n,
            spend,
        });
    }
    finish(results, outcome)
}

// ── T4.2: the comparison table ──────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TableRow {
    pub name: ArmName,
    pub escalations: u32,
    pub rescued: u32,
    /// Rounded to 1e-6 for the table bytes (spend is a sum of products of
    /// one-decimal draws and c_depth; 1e-6 is far below any decision-relevant
    /// delta and stable under IEEE summation order — which is fixed).
    pub spend_micros: u64,
    pub breach_fallbacks: u32,
    pub coverage: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArmName {
    NeverEscalateFloor,
    IncumbentCap,
    ProbeGated,
}

impl ArmName {
    pub const fn label(self) -> &'static str {
        match self {
            Self::NeverEscalateFloor => "never-escalate floor",
            Self::IncumbentCap => "incumbent (modelless_cap)",
            Self::ProbeGated => "probe-gated (fusion)",
        }
    }
}

pub fn table_row(name: ArmName, run: &ArmRun) -> TableRow {
    TableRow {
        name,
        escalations: run.outcome.escalations,
        rescued: run.outcome.rescued,
        spend_micros: (run.outcome.spend * 1_000_000.0).round() as u64,
        breach_fallbacks: run.outcome.breach_fallbacks,
        coverage: if run.outcome.probe_observations > 0 {
            Some(run.outcome.coverage_hits as f64 / run.outcome.probe_observations as f64)
        } else {
            None
        },
    }
}

/// The full default-posture table (T4.2) — floor, incumbent, probe-gated.
pub fn default_table(corpus: &Corpus, draws: &OutcomeStream, c_depth: f64) -> [TableRow; 3] {
    let floor = run_floor(corpus, draws);
    let incumbent = run_incumbent_cap(corpus, draws, c_depth, None);
    let probe = run_probe_gated(corpus, draws, c_depth, N_MAX);
    [
        table_row(ArmName::NeverEscalateFloor, &floor),
        table_row(ArmName::IncumbentCap, &incumbent),
        table_row(ArmName::ProbeGated, &probe),
    ]
}

/// Canonical blake3 over the table bytes (arm order fixed; per row: name
/// discriminant, escalations, rescued, spend_micros, breach_fallbacks,
/// coverage bits or a 0-marker) — the T4.4 determinism pin.
pub fn table_blake3(rows: &[TableRow]) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    for r in rows {
        h.update(&[match r.name {
            ArmName::NeverEscalateFloor => 0u8,
            ArmName::IncumbentCap => 1,
            ArmName::ProbeGated => 2,
        }]);
        h.update(&r.escalations.to_be_bytes());
        h.update(&r.rescued.to_be_bytes());
        h.update(&r.spend_micros.to_be_bytes());
        h.update(&r.breach_fallbacks.to_be_bytes());
        match r.coverage {
            Some(c) => {
                h.update(&[1u8]);
                h.update(&c.to_bits().to_be_bytes());
            }
            None => {
                h.update(&[0u8]);
            }
        }
    }
    *h.finalize().as_bytes()
}

/// The matched-spend search (T4.2/Super-GOAT): the smallest escalation cap
/// whose incumbent mean spend is ≥ `target_spend` — i.e. the incumbent given
/// EXACTLY the probe arm's budget. Spend is monotone non-decreasing in the
/// cap, so a binary search over the cap in spend units is exact to 1e-6.
pub fn incumbent_cap_for_spend(
    corpus: &Corpus,
    draws: &OutcomeStream,
    c_depth: f64,
    target_spend: f64,
) -> (f64, ArmRun) {
    let mut lo = 0.0f64;
    let mut hi = c_depth * corpus.states.len() as f64 + 1.0;
    // Invariant: spend(lo) < target ≤ spend(hi) (cap 0 escalates nothing;
    // the hi cap escalates everything).
    for _ in 0..64 {
        let mid = 0.5 * (lo + hi);
        let run = run_incumbent_cap(corpus, draws, c_depth, Some(mid));
        if run.outcome.spend < target_spend {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let run = run_incumbent_cap(corpus, draws, c_depth, Some(hi));
    (hi, run)
}

/// Print the T4.2 table (every G1 run's disclosure duty).
pub fn print_table(rows: &[TableRow], corpus_n: usize, c_depth: f64) {
    println!(
        "── Plan 621 Phase 4 T4.2 — healer-lane comparison (corpus {corpus_n} states, c_depth {c_depth:.0}) ──"
    );
    println!(
        "{:<26} {:>7} {:>8} {:>9} {:>12} {:>8}",
        "arm", "esc", "esc-rate", "rescued", "spend", "breach"
    );
    for r in rows {
        println!(
            "{:<26} {:>7} {:>8.1}% {:>9}/{} {:>12} {:>8}",
            r.name.label(),
            r.escalations,
            100.0 * f64::from(r.escalations) / corpus_n as f64,
            r.rescued,
            corpus_n,
            format!("{:.0}", r.spend_micros as f64 / 1_000_000.0),
            r.breach_fallbacks,
        );
    }
    for r in rows {
        if let Some(c) = r.coverage {
            println!("  CI coverage (probe arm): {c:.4} over the run's observations (95% nominal)");
        }
    }
}
