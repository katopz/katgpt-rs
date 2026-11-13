//! Discrete format-native authoring arm (Issue 1209; Research 668 — the
//! Tree-Ising distill, arXiv:2611.14210). Sibling of
//! [`crate::weight_edit_overlay::lp_authoring`] (the Issue-993 continuous
//! arm): authors carrier repairs by **discrete optimization over
//! format-native moves** — sign flips and ×2^k scale shifts, each EXACT
//! in binary floats — instead of continuous LP coefficients.
//!
//! Objective (the paper's forward-statistics form, Eq. 6 shape): one dense
//! QUBO per output row over the candidate move set — a linear dose term
//! along the repair direction `u` (γ-weighted, computed on the region's
//! mean input profile `μ_ℛ`), quadratic input-collateral (self terms +
//! same-column pairs from the region Gram `C_𝒫`), and an L0 count penalty
//! λ. The builder consumes `(u, μ_ℛ, C_𝒫, W, λ)` and derives every move's
//! signed region-profile effect `s_a = (f_a − 1)·W[r, j_a]·μ_ℛ[j_a]` —
//! no forward pass of its own.
//!
//! # Move set + exactness boundary
//!
//! `{SignFlip, Scale2k(k)}` per candidate weight. Sign flips and power-of-
//! two scalings are exact on any binary float — EXCEPT at the exponent
//! edges: a scaled candidate that overflows to infinity or underflows to
//! subnormal in f32's exponent range is NOT admissible. The predicate
//! refuses it loudly (the exactness boundary named in Issue 1209; the
//! apply-path behaviour — FTZ where the kernel flushes — is the declared
//! format's business, never assumed away here). The ternary arm's route
//! flips carry no exponent axis and are out of this lane's v1 move set.
//!
//! # Emission (the existing carrier apply path, verbatim)
//!
//! Each accepted move on weight `(r, j)` emits one
//! [`EditForm::DirectionSubtraction`] spec — `columns: [j]`,
//! `direction: e_r`, `strength: −(f−1)·W[r, j]` — the COO posture: the
//! carrier's `forward_delta_into` composes specs additively and applies
//! `y[r] -= strength·h[j]` = `ΔW[r,j]·h[j]`, the dequant-domain delta
//! (never a requantize; the Issue-980 retention law rides along: zero
//! carrier mass stays a bit-exact no-op).
//!
//! # Determinism
//!
//! Greedy descent is deterministic by construction (strictly-better
//! compare, first-index ties, sequential candidate order). The SA arm
//! pins a per-call SplitMix64 (no global RNG, the Issue-809 law) and runs
//! fixed-replica sequential sweeps (the Issue-940 fold law): same seed ⇒
//! bit-identical emitted spec.
//!
//! # Format-bit accounting (REPORT-ONLY)
//!
//! Per accepted move: a sign flip rewrites the format's sign bit (1 bit);
//! a ×2^k shift rewrites the exponent field (the BF16 state-tree formula:
//! 8 exponent bits, the shift re-encodes at most the leading exponent
//! run — counted as `min(|k|, 8)` here, parameterized by the declared
//! format). On the f32 fixtures the accounting is ILLUSTRATIVE: the moves
//! are exact and format-native in kind, but a fixture-scale bits win is
//! not a deployment gain until a packed-checkpoint patcher exists (the
//! T6 deferred rung; the Bench-976 serving veto stands).
//!
//! Instrument-grade: fixture-scale authoring arm; never claims model
//! quality (the Issue-990 precedent).
//!
//! # Demotion law (pre-registered in Issue 1209)
//!
//! If the discrete arm fails battery arms that hand-authored (980) or
//! LP-authored (993) edits pass: record the negative (the 990/Issue-565
//! precedent), strike the arm from the module + docs (no feature flag
//! exists to remove), close the issue NEGATIVE. The lane's hand/LP paths
//! are untouched either way. No zombie arm.

use crate::weight_edit_overlay::{EditForm, WeightEditOverlay, WeightEditSpec};

/// One format-native candidate move on weight `(row, col)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiscreteMove {
    /// `f = −1` (exact on every binary float, subnormals included — the
    /// sign bit alone flips).
    SignFlip,
    /// `f = 2^k` (exact within the format's normal exponent range; the
    /// admissibility predicate refuses the overflow/subnormal edges).
    /// `k > 0` scales up, `k < 0` down; `k == 0` is the identity (never a
    /// candidate).
    Scale2k(i32),
}

impl DiscreteMove {
    /// The exact multiplicative factor on binary floats.
    pub fn factor(self) -> f32 {
        match self {
            DiscreteMove::SignFlip => -1.0,
            DiscreteMove::Scale2k(k) => match k.signum() {
                1 => {
                    let mut f = 1.0f32;
                    for _ in 0..k {
                        f *= 2.0;
                    }
                    f
                }
                -1 => {
                    let mut f = 1.0f32;
                    for _ in 0..(-k) {
                        f /= 2.0;
                    }
                    f
                }
                _ => 1.0,
            },
        }
    }

    /// The move's weight-domain delta factor: `(f − 1)`.
    pub fn delta_factor(self) -> f32 {
        self.factor() - 1.0
    }

    /// Report-only format-bit cost (see the module doc; ILLUSTRATIVE on
    /// f32 fixtures — the BF16 state-tree formula named here).
    pub fn format_bits(self) -> u32 {
        match self {
            // The sign bit, always exactly one.
            DiscreteMove::SignFlip => 1,
            // BF16 exponent field = 8 bits; the shift re-encodes the
            // leading exponent run — bounded by the field width.
            DiscreteMove::Scale2k(k) => (k.unsigned_abs()).min(8),
        }
    }
}

/// f32's normal exponent range (the exactness boundary's axis): powers of
/// two inside `[2^-126, 2^127)` scale exactly; outside it the result
/// overflows/underflows and the move is refused.
const F32_MAX_EXP: i32 = 127;
const F32_MIN_NORMAL_EXP: i32 = -126;

/// The exactness predicate: is `f` exact on `w` in f32? Sign flips always;
/// Scale2k(k) iff the result stays normal (never infinite, never
/// subnormal). Refuses loudly via `false` — never assumes exactness.
pub fn move_admissible(m: DiscreteMove, w: f32) -> bool {
    if w == 0.0 || !w.is_finite() {
        // Zero stays zero under any exact scaling; non-finite weights are
        // outside the authoring domain entirely.
        return w == 0.0;
    }
    match m {
        DiscreteMove::SignFlip => true,
        DiscreteMove::Scale2k(k) => {
            // The weight's unbiased exponent (f32 bit decomposition: the
            // biased field minus 127; normals only — subnormal weights are
            // outside the candidate ranking's domain).
            let exp = (((w.to_bits() >> 23) & 0xFF) as i32) - 127;
            let shifted = exp + k;
            (F32_MIN_NORMAL_EXP..=F32_MAX_EXP).contains(&shifted)
        }
    }
}

/// Per-row candidate move with its precomputed signed region-profile
/// effect (the QUBO's `δ_a` payload).
#[derive(Clone, Copy, Debug)]
pub struct CandidateMove {
    pub row: usize,
    pub col: usize,
    pub mv: DiscreteMove,
    /// Signed region-profile effect `s = (f−1)·W[r,c]·μ_ℛ[c]`.
    pub effect: f32,
    /// The precomputed energy delta of taking this move ALONE from the
    /// empty set (the greedy seed order's key).
    pub solo_energy: f64,
}

/// Authoring configuration (the issue's T2 config rows: the conclude
/// direction arm rides `beta`).
#[derive(Clone, Debug)]
pub struct DiscreteAuthoringConfig {
    /// K candidate moves per output row, ranked by the dose-projection
    /// score `|(uᵀW_:j)·μ_j|`.
    pub k_candidates: usize,
    /// Dose weight γ (the objective's linear-term scale).
    pub gamma: f32,
    /// L0 penalty per moved weight.
    pub lambda: f64,
    /// The conclude-direction arm: `u = u_loop − β·u_end` is supplied by
    /// the caller — this field records which arm the solve ran so the
    /// emitted artifact can name it (β ∈ {0, 0.5} pre-registered).
    pub beta: f32,
    /// Cap on accepted moves per output row (the sparsity axis).
    pub max_moves_per_row: usize,
    /// SA: outer sweeps per replica.
    pub sa_sweeps: usize,
    /// SA: fixed replica count (16 — pre-registered).
    pub sa_replicas: usize,
    /// SA: initial temperature.
    pub sa_temp0: f64,
    /// SA: geometric cooling factor per sweep (0 < c < 1).
    pub sa_cool: f64,
    /// Pinned per-call seed (the SA arm; no global RNG).
    pub seed: u64,
}

impl Default for DiscreteAuthoringConfig {
    fn default() -> Self {
        Self {
            k_candidates: 8,
            gamma: 1.0,
            lambda: 0.05,
            beta: 0.0,
            max_moves_per_row: 4,
            sa_sweeps: 24,
            sa_replicas: 16,
            sa_temp0: 1.0,
            sa_cool: 0.85,
            seed: 0x1209_D157,
        }
    }
}

/// The objective's statistics bundle — everything the QUBO builder
/// consumes (the issue's "(u, μ_ℛ, C_𝒫 rows, δ_a, λ)" intake).
pub struct ObjectiveStats {
    /// Repair direction in the weight-row space, length `out_dim`
    /// (already conclude-combined by the caller: `u = u_loop − β·u_end`).
    pub u: Vec<f32>,
    /// The region's mean input profile `μ_ℛ`, length `in_dim`.
    pub mu_region: Vec<f32>,
    /// The input-collateral Gram `C_𝒫`, `in_dim × in_dim` row-major —
    /// taken over the AUTHORED-FOR distribution (the fixture's honest
    /// analogue: push ∪ keep rows; the collateral the repair must pay).
    pub gram_region: Vec<f32>,
    /// The base weights, `out_dim × in_dim` row-major (the moves' subject).
    pub weights: Vec<f32>,
    pub out_dim: usize,
    pub in_dim: usize,
    /// Optional retention axis (the lane's keep law as an authoring
    /// constraint — the discrete twin of the LP arm's ε=0 keep
    /// constraints): per-column mean |input| over the SILENCE class.
    /// When supplied with `keep_tol`, a candidate move is admissible only
    /// if its off-region response stays within tolerance:
    /// `|ΔW[r,j]|·keep_mean_abs[j] ≤ keep_tol`. `None` = unconstrained
    /// (the paper's bare posture).
    pub keep_mean_abs: Option<Vec<f32>>,
    /// The silence tolerance paired with `keep_mean_abs`.
    pub keep_tol: Option<f32>,
}

/// The authored result: accepted moves + the report-only accounting.
#[derive(Clone, Debug)]
pub struct DiscreteAuthoringSolution {
    pub moves: Vec<CandidateMove>,
    /// Final QUBO energy of the accepted set (the solver's objective).
    pub energy: f64,
    /// Solver that produced it (`greedy` | `sa`).
    pub solver: &'static str,
    /// Report-only format-bit total (ILLUSTRATIVE on f32 fixtures).
    pub format_bits: u32,
}

/// Pinned SplitMix64 (the SA arm's only randomness; the Issue-809 law —
/// no global RNG anywhere in this module).
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        ((self.next_u64() >> 33) as usize) % n.max(1)
    }
}

/// Enumerate + rank the candidate moves for one output row: for every
/// weight column j, both move kinds (admissible only), ranked by the
/// dose-projection score `|(u_r·W[r,j])·μ_j|`, truncated to K.
fn row_candidates(
    stats: &ObjectiveStats,
    row: usize,
    cfg: &DiscreteAuthoringConfig,
) -> Vec<CandidateMove> {
    let ObjectiveStats {
        u,
        mu_region,
        weights,
        in_dim,
        ..
    } = stats;
    let mut cands: Vec<(f64, CandidateMove)> = Vec::with_capacity(*in_dim * 2);
    let u_r = u[row] as f64;
    for j in 0..*in_dim {
        let w = weights[row * in_dim + j];
        let mu_j = mu_region[j];
        let score = (u_r * w as f64 * mu_j as f64).abs();
        if score == 0.0 || w == 0.0 {
            continue;
        }
        for mv in [
            DiscreteMove::SignFlip,
            DiscreteMove::Scale2k(1),
            DiscreteMove::Scale2k(-1),
        ] {
            if !move_admissible(mv, w) {
                continue;
            }
            // The retention axis (when supplied): the move's off-region
            // response must stay within the silence tolerance — the
            // discrete twin of the LP arm's ε=0 keep constraints.
            if let (Some(keep), Some(tol)) = (&stats.keep_mean_abs, stats.keep_tol) {
                let resp = mv.delta_factor().abs() * w.abs() * keep.get(j).copied().unwrap_or(0.0);
                if resp > tol {
                    continue;
                }
            }
            let effect = mv.delta_factor() * w * mu_j;
            cands.push((
                score,
                CandidateMove {
                    row,
                    col: j,
                    mv,
                    effect,
                    solo_energy: 0.0,
                },
            ));
        }
    }
    // Stable descending rank by score; ties keep the enumeration order
    // (first-index tie rule).
    cands.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    cands.truncate(cfg.k_candidates);
    cands.into_iter().map(|(_, m)| m).collect()
}

/// The QUBO energy of a move multiset over one row's candidates: linear
/// dose (reward, γ·u_r·s), quadratic collateral (self C_jj·s² + pairwise
/// C_jajb·s_a·s_b on same-column candidates), L0 count (λ each).
fn row_energy(
    stats: &ObjectiveStats,
    row: usize,
    cfg: &DiscreteAuthoringConfig,
    picks: &[usize],
    cands: &[CandidateMove],
) -> f64 {
    let u_r = stats.u[row] as f64;
    let mut e = 0.0f64;
    for (n, &ai) in picks.iter().enumerate() {
        let a = &cands[ai];
        e -= cfg.gamma as f64 * u_r * a.effect as f64;
        e += stats.gram_region[a.col * stats.in_dim + a.col] as f64 * (a.effect * a.effect) as f64;
        for &bi in picks.iter().skip(n + 1) {
            let b = &cands[bi];
            if a.col == b.col {
                e += stats.gram_region[a.col * stats.in_dim + b.col] as f64
                    * (a.effect * b.effect) as f64;
            }
        }
        e += cfg.lambda;
    }
    e
}

/// Fill each candidate's `solo_energy` (the empty-set delta — the greedy
/// seed order's key).
fn fill_solo(
    stats: &ObjectiveStats,
    row: usize,
    cfg: &DiscreteAuthoringConfig,
    cands: &mut [CandidateMove],
) {
    for i in 0..cands.len() {
        cands[i].solo_energy = row_energy(stats, row, cfg, &[i], cands);
    }
}

/// Solve one output row's QUBO greedily (single-flip descent): from the
/// empty set, repeatedly take the single addition whose energy delta is
/// the most negative; stop when no addition improves. Deterministic:
/// sequential candidate order, strictly-better compare (first-index ties).
fn solve_row_greedy(
    stats: &ObjectiveStats,
    row: usize,
    cfg: &DiscreteAuthoringConfig,
    cands: &mut [CandidateMove],
) -> Vec<usize> {
    fill_solo(stats, row, cfg, cands);
    let mut picks: Vec<usize> = Vec::new();
    loop {
        if picks.len() >= cfg.max_moves_per_row {
            break;
        }
        let cur = row_energy(stats, row, cfg, &picks, cands);
        let mut best: Option<(usize, f64)> = None;
        for i in 0..cands.len() {
            if picks.contains(&i) {
                continue;
            }
            // One spin per weight: any same-column candidate is the
            // same weight — exclusive by construction (the T4
            // colliding-move refusal pins this).
            if picks.iter().any(|&p| cands[p].col == cands[i].col) {
                continue;
            }
            let mut next_picks = picks.clone();
            next_picks.push(i);
            let delta = row_energy(stats, row, cfg, &next_picks, cands) - cur;
            if delta < -1e-12 && best.is_none_or(|(_, bd)| delta < bd) {
                best = Some((i, delta));
            }
        }
        match best {
            Some((i, _)) => picks.push(i),
            None => break,
        }
    }
    picks
}

/// The seeded simulated-annealing solve over one row's candidates: fixed
/// replica count, sequential sweeps per replica (the Issue-940 fold law —
/// fixed order, no parallel reshuffle), geometric temperature schedule,
/// pinned per-replica seeds derived from the config seed. Same config ⇒
/// bit-identical picks.
fn solve_row_sa(
    stats: &ObjectiveStats,
    row: usize,
    cfg: &DiscreteAuthoringConfig,
    cands: &mut [CandidateMove],
) -> Vec<usize> {
    fill_solo(stats, row, cfg, cands);
    let mut global_best: Option<(f64, Vec<usize>)> = None;
    for replica in 0..cfg.sa_replicas {
        let mut rng = SplitMix64(cfg.seed ^ (0xD15C_0000 + replica as u64));
        let mut picks: Vec<usize> = Vec::new();
        let mut temp = cfg.sa_temp0;
        for _sweep in 0..cfg.sa_sweeps {
            // One sweep: propose |cands| single-element moves in pinned order.
            for i in 0..cands.len() {
                if picks.len() >= cfg.max_moves_per_row && !picks.contains(&i) {
                    continue;
                }
                let cur = row_energy(stats, row, cfg, &picks, cands);
                let (next, proposed) = if picks.contains(&i) {
                    // Propose REMOVAL (always admissible; frees capacity).
                    let mut v = picks.clone();
                    v.retain(|&p| p != i);
                    (row_energy(stats, row, cfg, &v, cands), v)
                } else {
                    if picks.iter().any(|&p| cands[p].col == cands[i].col) {
                        continue;
                    }
                    let mut v = picks.clone();
                    v.push(i);
                    (row_energy(stats, row, cfg, &v, cands), v)
                };
                let delta = next - cur;
                let accept = delta <= 0.0 || {
                    let u01 = (rng.next_u64() >> 40) as f64 / (1u64 << 24) as f64;
                    u01 < ((-delta) / temp).exp()
                };
                if accept {
                    picks = proposed;
                }
            }
            temp *= cfg.sa_cool;
            if temp < 1e-6 {
                temp = 1e-6;
            }
        }
        let e = row_energy(stats, row, cfg, &picks, cands);
        let better = match &global_best {
            Some((be, _)) => e < *be,
            None => true,
        };
        if better {
            global_best = Some((e, picks));
        }
    }
    global_best.map(|(_, p)| p).unwrap_or_default()
}

/// Full authoring pass: enumerate candidates per row, solve each row's
/// QUBO with the named solver, and emit the COO overlay on the existing
/// carrier apply path.
///
/// `solver`: `"greedy"` (deterministic incumbent) or `"sa"` (seeded
/// annealing). Anything else refuses loud.
pub fn author_discrete_overlay(
    stats: &ObjectiveStats,
    cfg: &DiscreteAuthoringConfig,
    solver: &str,
) -> Result<DiscreteAuthoringSolution, &'static str> {
    if stats.u.len() != stats.out_dim
        || stats.mu_region.len() != stats.in_dim
        || stats.gram_region.len() != stats.in_dim * stats.in_dim
        || stats.weights.len() != stats.out_dim * stats.in_dim
    {
        return Err("objective stats dimensions disagree");
    }
    if stats.out_dim == 0 || stats.in_dim == 0 {
        return Err("empty weight matrix");
    }
    let solve = match solver {
        "greedy" => {
            solve_row_greedy
                as fn(
                    &ObjectiveStats,
                    usize,
                    &DiscreteAuthoringConfig,
                    &mut [CandidateMove],
                ) -> Vec<usize>
        }
        "sa" => solve_row_sa,
        _ => return Err("unknown solver (greedy | sa)"),
    };
    let mut all: Vec<CandidateMove> = Vec::new();
    let mut energy = 0.0f64;
    let mut bits = 0u32;
    for row in 0..stats.out_dim {
        let mut cands = row_candidates(stats, row, cfg);
        if cands.is_empty() {
            continue;
        }
        let picks = solve(stats, row, cfg, &mut cands);
        if picks.is_empty() {
            continue;
        }
        energy += row_energy(stats, row, cfg, &picks, &cands);
        for &pi in &picks {
            let m = cands[pi];
            bits += m.mv.format_bits();
            all.push(m);
        }
    }
    Ok(DiscreteAuthoringSolution {
        moves: all,
        energy,
        solver: match solver {
            "greedy" => "greedy",
            _ => "sa",
        },
        format_bits: bits,
    })
}

/// Emit the accepted moves as the COO overlay on the EXISTING carrier
/// apply path: one [`EditForm::DirectionSubtraction`] spec per moved
/// element (`columns: [j]`, `direction: e_r`,
/// `strength: −(f−1)·W[r,j]`), the dequant-domain delta (never a
/// requantize). The carrier's `forward_delta_into` composes specs
/// additively; the Issue-980 retention law rides along unchanged.
pub fn emit_overlay(
    sol: &DiscreteAuthoringSolution,
    stats: &ObjectiveStats,
    layer: u8,
) -> Result<WeightEditOverlay, &'static str> {
    let mut specs = Vec::with_capacity(sol.moves.len());
    for m in &sol.moves {
        let w = stats.weights[m.row * stats.in_dim + m.col];
        let strength = -(m.mv.delta_factor() * w);
        if strength == 0.0 {
            continue;
        }
        let mut direction = vec![0.0f32; stats.out_dim];
        direction[m.row] = 1.0;
        specs.push(WeightEditSpec {
            layer,
            form: EditForm::DirectionSubtraction {
                columns: vec![m.col as u32],
                direction,
                strength,
            },
        });
    }
    WeightEditOverlay::new(specs).ok_or("emitted overlay failed carrier validation")
}

/// Norm-matched random-move control (the T4 falsifier's engine): per row,
/// draw the same NUMBER of moves as the authored arm at the same candidate
/// columns, but with a seeded pseudo-random move kind + shifted 2^k —
/// norm-matched in the count axis, randomized in the geometry axis.
pub fn random_control_moves(
    stats: &ObjectiveStats,
    authored: &DiscreteAuthoringSolution,
    seed: u64,
) -> Vec<CandidateMove> {
    let mut rng = SplitMix64(seed);
    let per_row: Vec<(usize, usize)> = {
        let mut counts = vec![0usize; stats.out_dim];
        for m in &authored.moves {
            counts[m.row] += 1;
        }
        counts.into_iter().enumerate().collect()
    };
    let mut out = Vec::new();
    for (row, n) in per_row {
        // Randomly draw n moves across the row's columns, kinds shifted
        // away from the authored geometry (norm-matched in count, random
        // in column and kind).
        for _ in 0..n {
            let j = rng.below(stats.in_dim);
            let w = stats.weights[row * stats.in_dim + j];
            if w == 0.0 {
                continue;
            }
            let mv = match rng.below(4) {
                0 => DiscreteMove::SignFlip,
                1 => DiscreteMove::Scale2k(2),
                2 => DiscreteMove::Scale2k(-2),
                _ => DiscreteMove::Scale2k(1),
            };
            if !move_admissible(mv, w) {
                continue;
            }
            out.push(CandidateMove {
                row,
                col: j,
                mv,
                effect: mv.delta_factor() * w * stats.mu_region[j],
                solo_energy: 0.0,
            });
        }
    }
    out
}

/// The random control's overlay (the battery's no-op null comparator).
pub fn emit_control_overlay(
    moves: &[CandidateMove],
    stats: &ObjectiveStats,
    layer: u8,
) -> Result<WeightEditOverlay, &'static str> {
    let sol = DiscreteAuthoringSolution {
        moves: moves.to_vec(),
        energy: 0.0,
        solver: "control",
        format_bits: 0,
    };
    emit_overlay(&sol, stats, layer)
}
