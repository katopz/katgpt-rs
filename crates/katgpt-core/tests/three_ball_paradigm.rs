//! Plan 597 Phase 4 — three-ball paradigm integration test (G1 qualitative
//! ablation, arms a/b/c), behind the `bmr` feature (double protection with
//! the required-features row — the green-zero rule).
//!
//! # Paradigm (our instantiation of the paper's three-ball task)
//!
//! An agent inhabits a 3×3×3 grid world (27 cells = the 3 **location**
//! factors). Each step it may move one cell, **gaze** (a known-noise visual
//! modality — present structurally, carries no task information in this
//! instantiation), or **pick** a choice 0/1/2 (the **choice** factor). The
//! feedback modality has 3 outcomes {none, reward, penalty} with
//! log-preferences c = [0, 2, −6].
//!
//! One hidden rule `(X*, Y*)` — one of the 81 partial rules from
//! [`katgpt_core::bmr::enumerate_isomorphic_rules`] — makes one cell `X*`
//! special: picking `Y*` there yields reward, picking anything else there
//! yields penalty, and picks in every other cell yield none. Rule discovery
//! = accumulating feedback counts over the 81 (cell × pick) columns and
//! watching the posterior over the 81-rule space concentrate.
//!
//! # Arms (qualitative ablation, per plan T4.1)
//!
//! All scoring arms share the same pragmatic term at the same policy
//! precision β = 0.25 (expected log-preference scaled by β — a model
//! parameter; β = 1 lets the −6 penalty dominate the epistemic term and the
//! agent never returns to an ambiguous special cell). Arms differ ONLY in
//! their epistemic terms:
//!
//! - **(a) full info gain** — pragmatic + the Eq-10 expected information
//!   gain over models ([`katgpt_core::bmr::ModelSpace::efe_model_gain`]) at
//!   epistemic precision λ = 4; moves scored by one-step lookahead (best
//!   pick at the destination). The λ > 1 precision is required by the
//!   a-priori outcome marginals ({78/81 none, 1/81 reward, 2/81 penalty}
//!   make every fresh pick's expected preference ≈ −0.03 — the plan-T4.3
//!   "anticipated-outcome normalization" failure class).
//! - **(b) states+params only** — pragmatic + a state-novelty bonus for
//!   unvisited destinations + a param-novelty term (entropy of the touched
//!   feedback column under the accumulated counts). NO model term.
//! - **(c) no info gain (random)** — uniform random action each step.
//!
//! Candidate actions are ordered gazes → moves → picks and ties resolve to
//! the LAST maximum, so equal scores favor acting on the current cell.
//!
//! After each trial the agent computes its Occam statistic; crossing 16 nats
//! commits it to the argmax model and it exploits thereafter (the ×512
//! novelty-suppression convention's freeze equivalent — see the bmr module
//! header). Committing to a wrong model is a premature commit.
//!
//! # Gates (plan T4.2 — qualitative, not paper-number clones)
//!
//! - (a) discovers the true rule (Occam > 16 and argmax == true) in ≥ 90%
//!   of 64 seeds within 40 trials; KL(posterior ‖ true) → 0 on discovery.
//! - (b) leaves ≥ 2 plausible models (posterior > 0.01) in a majority.
//! - (c) fails to reach Occam > 4 in most runs.
//! - Premature-commit count recorded (a tunable, not a bug).
//!
//! Run with `--nocapture` to see the summary table; `TB_TRACE=1` traces
//! seed-wise actions (stderr).
#![cfg(feature = "bmr")]

use katgpt_core::bmr::{
    self, Counts, EfeScratch, FactorLayout, ModelSpace, enumerate_isomorphic_rules,
    occam_log_bayes_factor,
};

/// Grid edge per location factor.
const EDGE: usize = 3;
/// Feedback outcome rows: none / reward / penalty.
const ROW_NONE: usize = 0;
const ROW_REWARD: usize = 1;
const ROW_PENALTY: usize = 2;
/// Log-preferences over feedback outcomes (c = [0, 2, −6]).
const PREF: [f64; 3] = [0.0, 2.0, -6.0];
/// Policy precision on the pragmatic term (shared by all scoring arms).
const BETA_PRAG: f64 = 0.25;
/// Full-prior pseudocount per entry (the Dirichlet concentration of the
/// unconstrained model). T4.3 prior tuning: the sustained per-observation
/// evidence rate scales as 2(ã−SHRINKAGE)·ln N, so ã = 1 saturates far below
/// the 16-nat commit line within the 240-step budget while ã = 4 clears it
/// in ~70–120 picks. First-pick kill magnitude is ã-independent (~4.45 nats).
const PRIOR_FILL: f64 = 4.0;
/// Epistemic precision on the Eq-10 model-gain term (arm a).
const W_MODEL_GAIN: f64 = 4.0;
/// Param-novelty weight (arm b).
const W_PARAM_NOVELTY: f64 = 0.5;
/// State-novelty bonus for unvisited destinations (arm b).
const W_STATE_NOVELTY: f64 = 2.0;
/// Steps per trial / trials per run / seeds per arm.
const STEPS: usize = 6;
const TRIALS: usize = 40;
const SEEDS: usize = 64;
/// Occam commit threshold (nats).
const COMMIT_OCCAM: f64 = 16.0;
/// Plausible-model posterior threshold (arm b gate).
const PLAUSIBLE_MASS: f64 = 0.01;

/// Position as a 3-factor tuple; cell index = x*9 + y*3 + z.
type Pos = [usize; 3];

fn pos_to_cell(p: Pos) -> usize {
    p[0] * EDGE * EDGE + p[1] * EDGE + p[2]
}

fn cell_to_pos(cell: usize) -> Pos {
    [cell / (EDGE * EDGE), (cell / EDGE) % EDGE, cell % EDGE]
}

/// The six axis moves as (axis, delta).
const MOVES: [(usize, i64); 6] = [(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)];

fn apply_move(p: Pos, mv: (usize, i64)) -> Option<Pos> {
    let (axis, d) = mv;
    let v = p[axis] as i64 + d;
    if (0..EDGE as i64).contains(&v) {
        let mut q = p;
        q[axis] = v as usize;
        Some(q)
    } else {
        None
    }
}

fn manhattan(a: Pos, b: Pos) -> i64 {
    (0..3).map(|i| (a[i] as i64 - b[i] as i64).abs()).sum()
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Arm {
    /// Full info gain: states + params + models.
    Full,
    /// States + params only.
    StatesParams,
    /// No info gain: random.
    Random,
}

/// Per-run metrics.
#[derive(Clone, Debug, Default)]
struct RunOutcome {
    discovered: bool,
    occam_final: f64,
    kl_final: f64,
    plausible: usize,
    premature_commit: bool,
    committed: bool,
    score: f64,
}

/// Environment feedback for pick `k` at cell `c` under true rule `(x, y)`.
fn feedback(c: usize, k: usize, x: usize, y: usize) -> usize {
    if c == x {
        if k == y {
            ROW_REWARD
        } else {
            ROW_PENALTY
        }
    } else {
        ROW_NONE
    }
}

/// Marginal outcome probabilities the agent anticipates for pick `k` at
/// `cell` under model posterior `post` (rule index = x*3 + y, the
/// enumerator's documented ordering).
fn outcome_marginals(post: &[f64], cell: usize, k: usize) -> [f64; 3] {
    let mut q = [0.0f64; 3];
    for (m, &mass) in post.iter().enumerate() {
        if mass > 0.0 {
            q[feedback(cell, k, m / 3, m % 3)] += mass;
        }
    }
    q
}

/// Shannon entropy of a normalized distribution (nats).
fn entropy(p: &[f64]) -> f64 {
    p.iter().filter(|&&v| v > 0.0).map(|&v| -v * v.ln()).sum()
}

/// Anticipated-outcome triples for pick `k` at `cell` (the Eq-10 action).
fn pick_action(cell: usize, k: usize, post: &[f64]) -> Vec<(usize, usize, f64)> {
    let q = outcome_marginals(post, cell, k);
    let col = cell * 3 + k;
    [(ROW_NONE, q[0]), (ROW_REWARD, q[1]), (ROW_PENALTY, q[2])]
        .into_iter()
        .filter(|&(_, p)| p > 0.0)
        .map(|(row, p)| (col, row, p))
        .collect()
}

/// β-scaled expected log-preference of pick `k` at `cell` under `post`.
fn pick_pragmatic(cell: usize, k: usize, post: &[f64]) -> f64 {
    let q = outcome_marginals(post, cell, k);
    BETA_PRAG * (q[0] * PREF[0] + q[1] * PREF[1] + q[2] * PREF[2])
}

/// Param-novelty: entropy of the touched feedback column under the
/// accumulated counts (the states+params epistemic axis; no model posterior).
fn pick_param_novelty(engine: &ModelSpace, cell: usize, k: usize) -> f64 {
    let c = engine.post().col(cell * 3 + k);
    let s: f64 = c.iter().sum();
    if s <= 0.0 {
        return 0.0;
    }
    entropy(&c.iter().map(|&v| v / s).collect::<Vec<_>>())
}

/// KL(posterior ‖ one-hot(true)) — conservative reading: mass not on the
/// true model is measured against a uniform spread over the other models.
fn kl_vs_true(post: &[f64], true_rule: usize) -> f64 {
    let p = post[true_rule];
    let rest = 1.0 - p;
    if rest <= 0.0 {
        return 0.0;
    }
    let uniform = rest / (post.len() - 1) as f64;
    let mut kl = p.ln();
    for (m, &q) in post.iter().enumerate() {
        if m != true_rule && q > 0.0 {
            kl += q * (q / uniform).ln();
        }
    }
    kl.max(0.0)
}

/// One scored candidate action. kind: 0 = move, 1 = gaze, 2 = pick.
#[derive(Clone, Copy)]
struct Candidate {
    score: f64,
    kind: u8,
    idx: usize,
}

/// Run one seed of one arm.
fn run(seed: u64, arm: Arm) -> RunOutcome {
    let trace = std::env::var("TB_TRACE").is_ok();
    let mut rng = fastrand::Rng::with_seed(seed);
    let true_rule = rng.usize(..81);
    let (x_star, y_star) = (true_rule / 3, true_rule % 3);

    let mut layout = FactorLayout::new(&[EDGE, EDGE, EDGE], 3, 3, ROW_REWARD, ROW_PENALTY);
    layout.unconstrained_fill = PRIOR_FILL;
    let models = enumerate_isomorphic_rules(&layout);
    let prior = Counts::filled(3, 81, PRIOR_FILL);
    let mut engine = ModelSpace::new(prior.clone(), models, prior);
    let mut scratch = EfeScratch::new();

    let mut pos: Pos = [0, 0, 0];
    let mut visited = [false; 27];
    let mut committed: Option<usize> = None;
    let mut premature = false;
    let mut score = 0.0;
    let mut post_buf = [0.0f64; bmr::MAX_MODELS];
    let mut post: Vec<f64> = vec![1.0 / 81.0; 81];

    'trials: for _trial in 0..TRIALS {
        for _step in 0..STEPS {
            visited[pos_to_cell(pos)] = true;
            if let Some(m) = committed {
                // Exploitation under the committed model: pick its choice
                // when in its cell, else step toward it.
                let (cx, cy) = (m / 3, m % 3);
                if pos_to_cell(pos) == cx {
                    let fb = feedback(cx, cy, x_star, y_star);
                    engine.accumulate(cx * 3 + cy, fb, 1.0);
                    score += PREF[fb];
                } else {
                    let target = cell_to_pos(cx);
                    let mut stepped = false;
                    for &mv in MOVES.iter() {
                        if let Some(dest) = apply_move(pos, mv)
                            && manhattan(dest, target) < manhattan(pos, target) {
                                pos = dest;
                                stepped = true;
                                break;
                            }
                    }
                    if !stepped {
                        break 'trials; // boxed in — impossible on this grid
                    }
                }
                continue;
            }

            engine.posterior_into(&mut post_buf);
            post.copy_from_slice(&post_buf[..81]);

            // Gaze candidates first (score 0), then moves, then picks —
            // ties resolve to the last max, favoring action over abstention.
            let mut candidates: Vec<Candidate> = Vec::with_capacity(12);
            for g in 0..3 {
                candidates.push(Candidate { score: 0.0, kind: 1, idx: g });
            }
            match arm {
                Arm::Full => {
                    for (i, &mv) in MOVES.iter().enumerate() {
                        if let Some(dest) = apply_move(pos, mv) {
                            let cell = pos_to_cell(dest);
                            let mut s = f64::NEG_INFINITY;
                            for k in 0..3 {
                                let action = pick_action(cell, k, &post);
                                let v = pick_pragmatic(cell, k, &post)
                                    + W_MODEL_GAIN
                                        * engine.efe_model_gain(&action, &mut scratch);
                                s = s.max(v);
                            }
                            candidates.push(Candidate { score: s, kind: 0, idx: i });
                        }
                    }
                    for k in 0..3 {
                        let cell = pos_to_cell(pos);
                        let action = pick_action(cell, k, &post);
                        candidates.push(Candidate {
                            score: pick_pragmatic(cell, k, &post)
                                + W_MODEL_GAIN * engine.efe_model_gain(&action, &mut scratch),
                            kind: 2,
                            idx: k,
                        });
                    }
                }
                Arm::StatesParams => {
                    for (i, &mv) in MOVES.iter().enumerate() {
                        if let Some(dest) = apply_move(pos, mv) {
                            let cell = pos_to_cell(dest);
                            let mut s = f64::NEG_INFINITY;
                            for k in 0..3 {
                                s = s.max(pick_pragmatic(cell, k, &post));
                            }
                            if !visited[cell] {
                                s += W_STATE_NOVELTY;
                            }
                            candidates.push(Candidate { score: s, kind: 0, idx: i });
                        }
                    }
                    for k in 0..3 {
                        let cell = pos_to_cell(pos);
                        candidates.push(Candidate {
                            score: pick_pragmatic(cell, k, &post)
                                + W_PARAM_NOVELTY * pick_param_novelty(&engine, cell, k),
                            kind: 2,
                            idx: k,
                        });
                    }
                }
                Arm::Random => {
                    for (i, &mv) in MOVES.iter().enumerate() {
                        if apply_move(pos, mv).is_some() {
                            candidates.push(Candidate { score: rng.f64(), kind: 0, idx: i });
                        }
                    }
                    for k in 0..3 {
                        candidates.push(Candidate { score: rng.f64(), kind: 2, idx: k });
                    }
                    // Random gazes already pushed (score 0) — give them a
                    // random score too so they compete honestly.
                    for c in candidates.iter_mut().take(3) {
                        c.score = rng.f64();
                    }
                }
            }

            let best = candidates
                .iter()
                .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap())
                .copied()
                .unwrap_or(Candidate { score: 0.0, kind: 1, idx: 0 });
            if trace && _trial < 2 {
                eprintln!(
                    "seed{seed} t{_trial} s{_step} pos={pos:?} best={}?{} score={:.4}",
                    ["m", "g", "p"][best.kind as usize],
                    best.idx,
                    best.score
                );
            }
            match best.kind {
                0 => pos = apply_move(pos, MOVES[best.idx]).unwrap(),
                1 => {
                    let _visual = rng.usize(..3); // known noise — no update
                }
                2 => {
                    let cell = pos_to_cell(pos);
                    let fb = feedback(cell, best.idx, x_star, y_star);
                    engine.accumulate(cell * 3 + best.idx, fb, 1.0);
                    score += PREF[fb];
                }
                _ => unreachable!(),
            }
        }

        // Post-trial Occam commit check.
        if committed.is_none() {
            engine.posterior_into(&mut post_buf);
            let p = &post_buf[..81];
            if occam_log_bayes_factor(p) > COMMIT_OCCAM {
                let argmax = argmax_of(p);
                committed = Some(argmax);
                if argmax != true_rule {
                    premature = true;
                }
            }
        }
    }

    let mut final_post = vec![0.0f64; 81];
    match committed {
        Some(m) => final_post[m] = 1.0,
        None => {
            engine.posterior_into(&mut post_buf);
            final_post.copy_from_slice(&post_buf[..81]);
        }
    }
    let occam = occam_log_bayes_factor(&final_post);
    let discovered = argmax_of(&final_post) == true_rule && occam > COMMIT_OCCAM;
    if trace {
        eprintln!(
            "seed{seed} FINAL true={true_rule} argmax={} occam={occam:.3} discovered={discovered} plausible={} premature={premature} score={score:.1}",
            argmax_of(&final_post),
            final_post.iter().filter(|&&v| v > PLAUSIBLE_MASS).count()
        );
    }
    RunOutcome {
        discovered,
        occam_final: occam,
        kl_final: kl_vs_true(&final_post, true_rule),
        plausible: final_post.iter().filter(|&&v| v > PLAUSIBLE_MASS).count(),
        premature_commit: premature,
        committed: committed.is_some(),
        score,
    }
}

fn argmax_of(v: &[f64]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

#[test]
fn three_ball_ablation_gates() {
    let mut lines = Vec::new();
    for arm in [Arm::Full, Arm::StatesParams, Arm::Random] {
        let mut discovered = 0;
        let mut premature = 0;
        let mut plausible_ge2 = 0;
        let mut occam_le4 = 0;
        let mut committed_n = 0;
        let mut mean_score = 0.0;
        let mut mean_kl = 0.0;
        let mut mean_occam = 0.0;
        for seed in 0..SEEDS as u64 {
            let out = run(seed, arm);
            discovered += out.discovered as usize;
            premature += out.premature_commit as usize;
            plausible_ge2 += (out.plausible >= 2) as usize;
            occam_le4 += (out.occam_final <= 4.0) as usize;
            committed_n += out.committed as usize;
            mean_score += out.score;
            mean_kl += out.kl_final;
            mean_occam += out.occam_final;
        }
        let n = SEEDS as f64;
        lines.push(format!(
            "{arm:?}: discovered {discovered}/{SEEDS} ({:.0}%), premature {premature}, \
             committed {committed_n}, plausible>=2 {plausible_ge2}/{SEEDS}, occam<=4 {occam_le4}/{SEEDS}, \
             mean score {:+.1}, mean KL {:.4}, mean Occam {:+.2}",
            100.0 * discovered as f64 / n,
            mean_score / n,
            mean_kl / n,
            mean_occam / n,
        ));

        match arm {
            Arm::Full => assert!(
                discovered * 10 >= SEEDS * 9,
                "arm (a) must discover in >=90% of seeds: {discovered}/{SEEDS}"
            ),
            Arm::StatesParams => assert!(
                plausible_ge2 * 2 > SEEDS,
                "arm (b) must leave >=2 plausible models in a majority: {plausible_ge2}/{SEEDS}"
            ),
            Arm::Random => assert!(
                occam_le4 * 2 > SEEDS,
                "arm (c) must fail Occam > 4 in most runs: {occam_le4}/{SEEDS}"
            ),
        }
    }
    println!(
        "three-ball ablation ({} seeds, {} trials):\n  {}",
        SEEDS,
        TRIALS,
        lines.join("\n  ")
    );
}
