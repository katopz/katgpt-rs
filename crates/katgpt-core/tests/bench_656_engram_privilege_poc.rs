//! Issue 656 T1 — planted-drift PoC for counterfactual privilege gating.
//!
//! Falsifies or defends the claim that a per-slot counterfactual-advantage
//! gate recovers utility the similarity-only gate cannot.
//!
//! # Verdict rule (from the issue)
//!
//! > T1 falsifies or defends. If planted-drift shows no recoverable penalty
//! > (the similarity gate already suffices at realistic poisoning rates),
//! > close as negative result.
//!
//! Falsifiable target: **the gate recovers ≥ half the poisoned-entry penalty**.
//!
//! # Why four regimes and not one
//!
//! A single hand-picked regime would only prove the mechanism can be made to
//! look good. Each regime is a different answer to "when does the similarity
//! gate already suffice?":
//!
//! - **A · sign-opposed drift** — poison has *identical* cosine to the query as
//!   the good patterns but the opposite utility projection. The similarity gate
//!   is provably blind (same gate, opposite sign). The regime the gate exists
//!   for.
//! - **B · similarity-separable drift** (control) — poison is anti-aligned with
//!   the query, so `σ(dot/τ)` already suppresses it. If privilege gating adds
//!   little here, the honest reading is "the similarity gate is enough for
//!   *this* kind of corruption."
//! - **C · clean table** (G1) — 0% poison. The gate must be ≈ a no-op.
//! - **D · class-conditional utility** (the scope limit) — the LOPD F2 shape
//!   proper: one entry that genuinely *helps* query class 0 and *hurts* query
//!   class 1, with good entries staying good in both. Built from three mutually
//!   orthogonal directions (`u` query, `w₀`/`w₁` per-class utility) so the
//!   split entry is similarity-identical to a good one. A per-slot **scalar**
//!   `Δ` cannot represent class-dependent utility — this measures what it does
//!   instead.
//!
//! # Arms
//!
//! - `oracle` — poison slots left **empty**. The achievable ideal: what the
//!   consumer would get if the drifted entries had been evicted. Note this is
//!   *not* "poison replaced by good patterns" — a veto can zero a bad entry's
//!   contribution, it cannot conjure a good one. Scoring against the
//!   replace-with-good baseline would cap recovery at exactly 0.5 by
//!   construction and make the target unreachable-with-margin.
//! - `naive` — poisoned table, shipped `fuse_into_hidden_state`.
//! - `priv-exact` — poisoned table, privileged fuse, ledger fed **exact
//!   per-slot marginal δ** (`K+1` scorer calls per update).
//! - `priv-aggregate` — same, but fed one aggregate δ split by
//!   `CreditAssignment::GateWeighted` (2 scorer calls per update). Expected to
//!   fail in regime A; included because the cheap path is the one a cost-
//!   sensitive host reaches for first, and its failure mode should be measured
//!   rather than asserted.
//!
//! # Error metric (scale-invariant, per-path)
//!
//! Each arm is compared against **its own uncorrupted self**, because the
//! privileged path applies a uniform attenuation (`p < 1` even for earned
//! slots) that has nothing to do with poison:
//!
//! ```text
//! rel_err(path) = |S(path, poisoned) − S(path, clean)| / |S(path, clean)|
//! recovery      = (rel_err_naive − rel_err_priv) / rel_err_naive
//! ```
//!
//! Comparing raw scores across paths would confound "vetoed the poison" with
//! "attenuated everything," which is the easiest way to fake this result.
//!
//! # Run
//!
//! ```bash
//! cargo test -p katgpt-core --features engram \
//!     --test bench_656_engram_privilege_poc --release -- --nocapture
//! ```

#![cfg(feature = "engram")]

use katgpt_core::engram::{
    CreditAssignment, EngramConfig, EngramHash, EngramTable, EngramTableBuilder, K_MAX,
    PrivilegeConfig, PrivilegeLedger, PrivilegeTrace, fuse_into_hidden_state,
    fuse_into_hidden_state_privileged, sigmoid_fuse_scaled_into,
};
use std::time::Instant;

// ─── Parameters ─────────────────────────────────────────────────────────────

const D: usize = 32;
const N_SLOTS: usize = 64;
/// Heads 0..K_MAX map to slots 0..K_MAX (keys are the slot ids, N_SLOTS > K_MAX).
const N_ACTIVE: usize = K_MAX;
const TRAIN_ROUNDS: usize = 400;
const EVAL_QUERIES: usize = 64;
/// Poison fraction of the active slots. 25% is deliberately below the "half the
/// table is rotten" strawman.
const POISON_FRAC: f32 = 0.25;
/// Recovery bar from the issue: "recovers ≥ half the poisoned-entry penalty".
const RECOVERY_BAR: f64 = 0.5;
/// Retrieval events available to the sparse-update sweep. Held fixed across
/// update periods so the sweep reads as quality-at-a-cost, not quality-vs-time.
const EVENT_BUDGET: usize = 1_600;
/// Update cadences swept. Shared by the cost ladder and the quality sweep so
/// the two tables can be read as one curve.
const PERIODS: [usize; 5] = [1, 4, 16, 64, 256];

// ─── Deterministic RNG (splitmix64) ─────────────────────────────────────────

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [-1, 1).
    fn signed(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / 8_388_608.0 - 1.0
    }
}

// ─── Geometry ───────────────────────────────────────────────────────────────

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn normalize(v: &mut [f32]) {
    let n = dot(v, v).sqrt().max(1e-12);
    for x in v.iter_mut() {
        *x /= n;
    }
}

/// One query direction `u` and **two** per-class utility directions `w[0]`,
/// `w[1]`, all mutually orthogonal.
///
/// Regimes A/B/C are single-objective: `w[0] == w[1]`, so the class index is
/// inert. Regime D makes them genuinely orthogonal, which is the only way to
/// build an entry whose utility flips across query classes while its cosine to
/// the query stays identical to a good entry's.
struct Geometry {
    u: Vec<f32>,
    w: [Vec<f32>; 2],
}

impl Geometry {
    fn new(two_class: bool, rng: &mut Rng) -> Self {
        let mut u: Vec<f32> = (0..D).map(|_| rng.signed()).collect();
        normalize(&mut u);

        let mut w0: Vec<f32> = (0..D).map(|_| rng.signed()).collect();
        orthogonalize(&mut w0, &[&u]);
        normalize(&mut w0);

        let w1 = if two_class {
            let mut w1: Vec<f32> = (0..D).map(|_| rng.signed()).collect();
            orthogonalize(&mut w1, &[&u, &w0]);
            normalize(&mut w1);
            w1
        } else {
            w0.clone()
        };
        Self { u, w: [w0, w1] }
    }

    /// `a·u + b0·w₀ + b1·w₁ + noise`, with noise projected out of
    /// span(u, w₀, w₁) so it perturbs neither cosine-to-query nor utility.
    fn pattern(&self, a: f32, b0: f32, b1: f32, noise: f32, rng: &mut Rng) -> Vec<f32> {
        let mut n: Vec<f32> = (0..D).map(|_| rng.signed() * noise).collect();
        orthogonalize(&mut n, &[&self.u, &self.w[0], &self.w[1]]);
        (0..D)
            .map(|j| a * self.u[j] + b0 * self.w[0][j] + b1 * self.w[1][j] + n[j])
            .collect()
    }

    /// Consumer score for a query class: how far the state points along that
    /// class's utility direction.
    fn score(&self, state: &[f32], class: usize) -> f32 {
        dot(state, &self.w[class & 1])
    }
}

/// Gram-Schmidt `v` against an already-orthogonal basis.
fn orthogonalize(v: &mut [f32], basis: &[&Vec<f32>]) {
    for b in basis {
        let p = dot(v, b);
        for (j, vj) in v.iter_mut().enumerate() {
            *vj -= p * b[j];
        }
    }
}

// ─── Regimes ────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
enum Regime {
    /// Poison: same cosine-to-query, opposite utility. Similarity gate blind.
    SignOpposed,
    /// Poison: anti-aligned with the query. Similarity gate already suppresses.
    SimilaritySeparable,
    /// No poison.
    Clean,
    /// Poison helps query class 0 and hurts class 1 (equally frequent); good
    /// entries help both. The LOPD F2 shape.
    ClassConditional,
}

impl Regime {
    fn name(self) -> &'static str {
        match self {
            Self::SignOpposed => "A · sign-opposed drift",
            Self::SimilaritySeparable => "B · similarity-separable (control)",
            Self::Clean => "C · clean table (G1)",
            Self::ClassConditional => "D · class-conditional (scope limit)",
        }
    }
    fn poison_count(self) -> usize {
        match self {
            Self::Clean => 0,
            _ => (N_ACTIVE as f32 * POISON_FRAC).round() as usize,
        }
    }
    /// Only regime D has two distinct per-class utility directions.
    fn two_class(self) -> bool {
        matches!(self, Regime::ClassConditional)
    }
    /// Which utility direction query `idx` is scored against.
    fn query_class(self, idx: usize) -> usize {
        if self.two_class() { idx % 2 } else { 0 }
    }
}

/// Build the active-slot patterns. `include_poison = false` yields the oracle
/// table (poison slots simply absent).
fn build_table(
    geo: &Geometry,
    regime: Regime,
    include_poison: bool,
    rng: &mut Rng,
) -> impl EngramTable + use<> {
    let n_poison = regime.poison_count();
    let mut b = EngramTableBuilder::new(N_SLOTS, D);
    for slot in 0..N_ACTIVE {
        let is_poison = slot < n_poison;
        if is_poison && !include_poison {
            continue; // oracle: the drifted entry was evicted
        }
        // (a, b0, b1) — `a` sets cosine-to-query, `b0`/`b1` set per-class utility.
        let (a, b0, b1) = match (is_poison, regime) {
            // Good in a single-objective regime: aligned, positive utility.
            (false, Regime::SignOpposed | Regime::SimilaritySeparable | Regime::Clean) => {
                (1.0, 0.6, 0.0)
            }
            // Good in regime D: helps BOTH classes.
            (false, Regime::ClassConditional) => (1.0, 0.6, 0.6),
            // A: identical cosine-to-query, flipped utility.
            (true, Regime::SignOpposed) => (1.0, -0.6, 0.0),
            // B: anti-aligned with the query — the similarity gate can see it.
            (true, Regime::SimilaritySeparable) => (-1.0, -0.6, 0.0),
            // D: similarity-identical to a good entry, but helps class 0 only.
            (true, Regime::ClassConditional) => (1.0, 0.6, -0.6),
            (true, Regime::Clean) => unreachable!("clean regime has no poison"),
        };
        b.add_pattern(EngramHash(slot as u64), &geo.pattern(a, b0, b1, 0.10, rng));
    }
    b.build()
}

fn make_query(geo: &Geometry, idx: usize, rng: &mut Rng) -> Vec<f32> {
    // Query ≈ u, perturbed off the utility plane so retrieval is non-degenerate
    // without changing the utility structure.
    let mut q = geo.pattern(1.0, 0.0, 0.0, 0.08, rng);
    for (j, v) in q.iter_mut().enumerate() {
        *v += 0.01 * ((idx as f32 + j as f32) * 0.7).sin();
    }
    q
}

fn keys() -> [EngramHash; K_MAX] {
    let mut k = [EngramHash(0); K_MAX];
    for (i, slot) in k.iter_mut().enumerate() {
        *slot = EngramHash(i as u64);
    }
    k
}

// ─── Scratch ────────────────────────────────────────────────────────────────

struct Scratch {
    lookup: Vec<f32>,
    norm: Vec<f32>,
    out: Vec<f32>,
    state: Vec<f32>,
    contrib: Vec<f32>,
}

impl Scratch {
    fn new() -> Self {
        Self {
            lookup: vec![0.0; K_MAX * D],
            norm: vec![0.0; D],
            out: vec![0.0; D],
            state: vec![0.0; D],
            contrib: vec![0.0; D],
        }
    }
}

// ─── Arms ───────────────────────────────────────────────────────────────────

fn naive_score(
    table: &dyn EngramTable,
    geo: &Geometry,
    query: &[f32],
    class: usize,
    cfg: &EngramConfig,
    s: &mut Scratch,
) -> f32 {
    s.state.iter_mut().for_each(|x| *x = 0.0);
    fuse_into_hidden_state(
        &mut s.state,
        query,
        table,
        &keys(),
        cfg,
        &mut s.lookup,
        &mut s.norm,
        &mut s.out,
    );
    geo.score(&s.state, class)
}

#[allow(clippy::too_many_arguments)] // mirrors the primitive's own hot-path signature
fn privileged_score(
    table: &dyn EngramTable,
    geo: &Geometry,
    query: &[f32],
    class: usize,
    cfg: &EngramConfig,
    ledger: &PrivilegeLedger,
    trace: &mut PrivilegeTrace,
    s: &mut Scratch,
) -> f32 {
    s.state.iter_mut().for_each(|x| *x = 0.0);
    fuse_into_hidden_state_privileged(
        &mut s.state,
        query,
        table,
        &keys(),
        cfg,
        ledger,
        trace,
        &mut s.lookup,
        &mut s.out,
    );
    geo.score(&s.state, class)
}

/// Multiplicative outcome noise on the measured δ.
///
/// `amp = 0.0` is the noise-free ideal the headline numbers are measured at.
/// Anything above that models a host whose outcome verification is imperfect —
/// the case that decides whether "7 updates suffice" generalizes.
struct Noise {
    amp: f32,
    rng: Rng,
}

impl Noise {
    fn none() -> Self {
        Self {
            amp: 0.0,
            rng: Rng::new(1),
        }
    }
    fn new(amp: f32, seed: u64) -> Self {
        Self {
            amp,
            rng: Rng::new(seed),
        }
    }
    #[inline]
    fn perturb(&mut self, delta: f32) -> f32 {
        if self.amp == 0.0 {
            delta
        } else {
            delta + self.amp * delta.abs() * self.rng.signed()
        }
    }
}

/// One exact-marginal update: `K+1` scorer calls (one base + one per populated
/// head). Returns the number of scorer calls consumed.
#[allow(clippy::too_many_arguments)] // mirrors the host-side update loop
fn update_exact(
    table: &dyn EngramTable,
    geo: &Geometry,
    query: &[f32],
    class: usize,
    cfg: &EngramConfig,
    ledger: &mut PrivilegeLedger,
    s: &mut Scratch,
    // Outcome noise as a multiple of |δ|. `0.0` is the noise-free ideal; a real
    // host's outcome labels are never that clean, and the update count needed
    // to converge is what degrades.
    noise: &mut Noise,
) -> usize {
    let ks = keys();
    table.lookup_into(&ks, &mut s.lookup);
    // Base state is the pre-fuse latent (zero here) — δ_k is head k's *marginal*
    // effect on the consumer, which for a linear scorer equals its leave-one-out
    // effect exactly.
    s.state.iter_mut().for_each(|x| *x = 0.0);
    let s_base = geo.score(&s.state, class);
    let mut calls = 1usize;

    for (k, key) in ks.iter().enumerate() {
        let e_k: Vec<f32> = s.lookup[k * D..(k + 1) * D].to_vec();
        if e_k.iter().all(|&x| x == 0.0) {
            continue;
        }
        // Unscaled contribution — what head k would add at full privilege.
        sigmoid_fuse_scaled_into(query, &e_k, &e_k, &mut s.contrib, &cfg.fusion, 1.0);
        for j in 0..D {
            s.state[j] += s.contrib[j];
        }
        let delta = noise.perturb(geo.score(&s.state, class) - s_base);
        calls += 1;
        for j in 0..D {
            s.state[j] -= s.contrib[j];
        }
        // A = +1: the synthetic outcome is always correctly verified. A host
        // with noisy outcome labels would pass its own confidence here.
        ledger.observe((key.0 as usize) % table.num_slots(), 1.0, delta);
    }
    calls
}

/// One aggregate update: 2 scorer calls, split by `GateWeighted`.
fn update_aggregate(
    table: &dyn EngramTable,
    geo: &Geometry,
    query: &[f32],
    class: usize,
    cfg: &EngramConfig,
    ledger: &mut PrivilegeLedger,
    s: &mut Scratch,
) -> usize {
    let mut trace = PrivilegeTrace::new();
    s.state.iter_mut().for_each(|x| *x = 0.0);
    let s_base = geo.score(&s.state, class);
    // Snapshot the ledger's gating for this fuse, then measure the aggregate.
    let snapshot = ledger.clone();
    fuse_into_hidden_state_privileged(
        &mut s.state,
        query,
        table,
        &keys(),
        cfg,
        &snapshot,
        &mut trace,
        &mut s.lookup,
        &mut s.out,
    );
    let delta = geo.score(&s.state, class) - s_base;
    ledger.observe_trace(&trace, 1.0, delta, CreditAssignment::GateWeighted);
    2
}

/// Sum of |contribution·w_class| over fused heads — the utility mass the fuse
/// moved, regardless of direction. The denominator of purity.
fn utility_mass(
    table: &dyn EngramTable,
    geo: &Geometry,
    query: &[f32],
    class: usize,
    cfg: &EngramConfig,
    ledger: Option<&PrivilegeLedger>,
    s: &mut Scratch,
) -> f32 {
    let ks = keys();
    table.lookup_into(&ks, &mut s.lookup);
    let mut mass = 0.0f32;
    for (k, key) in ks.iter().enumerate() {
        let e_k: Vec<f32> = s.lookup[k * D..(k + 1) * D].to_vec();
        if e_k.iter().all(|&x| x == 0.0) {
            continue;
        }
        let slot = (key.0 as usize) % table.num_slots();
        let p = ledger.map_or(1.0, |l| l.privilege(slot));
        sigmoid_fuse_scaled_into(query, &e_k, &e_k, &mut s.contrib, &cfg.fusion, p);
        mass += geo.score(&s.contrib, class).abs();
    }
    mass
}

// ─── Result plumbing ────────────────────────────────────────────────────────

struct ArmResult {
    rel_err: f64,
    purity: f64,
}

/// Mean over the eval queries of `|S_poisoned − S_clean| / |S_clean|`, plus the
/// mean purity `S / Σ|contribution·w|` (scale-invariant "did the fuse point the
/// right way").
fn evaluate<F>(geo: &Geometry, regime: Regime, mut score_triple: F) -> ArmResult
where
    F: FnMut(usize, usize, &[f32]) -> (f32, f32, f32),
{
    let mut rng = Rng::new(0xE1A5_5E55);
    let mut sum_rel = 0.0f64;
    let mut sum_pur = 0.0f64;
    let mut n = 0usize;
    for q in 0..EVAL_QUERIES {
        let query = make_query(geo, q, &mut rng);
        let class = regime.query_class(q);
        let (s_poisoned, s_clean, mag) = score_triple(q, class, &query);
        let (sp, sc, m) = (s_poisoned as f64, s_clean as f64, mag as f64);
        if sc.abs() < 1e-9 {
            continue;
        }
        sum_rel += (sp - sc).abs() / sc.abs();
        sum_pur += if m > 1e-9 { sp / m } else { 0.0 };
        n += 1;
    }
    let n = n.max(1) as f64;
    ArmResult {
        rel_err: sum_rel / n,
        purity: sum_pur / n,
    }
}

/// Mean privilege factor over the good slots and over the poison slots — the
/// most direct read on "did the gate actually discriminate?"
fn privilege_split(ledger: &PrivilegeLedger, regime: Regime) -> (f64, f64) {
    let n_poison = regime.poison_count();
    let mean = |range: std::ops::Range<usize>| -> f64 {
        let n = range.len();
        match n {
            0 => f64::NAN,
            _ => range.map(|s| ledger.privilege(s) as f64).sum::<f64>() / n as f64,
        }
    };
    (mean(n_poison..N_ACTIVE), mean(0..n_poison))
}

// ─── Sparse-update sweep (the amortization question) ────────────────────────

/// Issue 656 design point 4: "amortize via the per-slot EMA (score sparsely,
/// decay between)". That is a *claim about quality under sparsity*, and the
/// fuse-only cost ratio cannot test it — it measures the floor with zero
/// updates, i.e. a ledger that never learns anything.
///
/// This sweep fixes the retrieval-event budget and varies how often an update
/// fires, so the reported curve is quality-at-a-cost rather than quality and
/// cost measured on two different configurations.
struct SparsityPoint {
    noise_amp: f32,
    period: usize,
    updates: usize,
    recovery: f64,
    p_good: f64,
    p_poison: f64,
}

/// Regime A only — the regime the gate exists for. Returns one point per
/// update period, all sharing the same `EVENT_BUDGET` of retrieval events.
fn sparsity_sweep(periods: &[usize], noise_amp: f32) -> Vec<SparsityPoint> {
    let regime = Regime::SignOpposed;
    let mut geo_rng = Rng::new(0x6560_0001);
    let geo = Geometry::new(regime.two_class(), &mut geo_rng);
    let cfg = EngramConfig::for_dim(D);
    let mut s = Scratch::new();

    let mut t_rng = Rng::new(0x6560_0002);
    let poisoned = build_table(&geo, regime, true, &mut t_rng);
    let mut t_rng = Rng::new(0x6560_0002);
    let clean = build_table(&geo, regime, false, &mut t_rng);

    let pcfg = PrivilegeConfig {
        scale: 0.03,
        margin: 0.0,
        ..PrivilegeConfig::for_delta_scale(0.3)
    };

    // Naive reference is period-independent.
    let naive = {
        let mut sc = Scratch::new();
        evaluate(&geo, regime, |_, class, q| {
            let sp = naive_score(&poisoned, &geo, q, class, &cfg, &mut sc);
            let scl = naive_score(&clean, &geo, q, class, &cfg, &mut sc);
            let mass = utility_mass(&poisoned, &geo, q, class, &cfg, None, &mut sc);
            (sp, scl, mass)
        })
    };

    periods
        .iter()
        .map(|&period| {
            let mut train = |table: &dyn EngramTable| -> (PrivilegeLedger, usize) {
                let mut ledger = PrivilegeLedger::new(table.num_slots(), pcfg);
                let mut rng = Rng::new(0x6560_0003);
                let noise = &mut Noise::new(noise_amp, 0x6560_0004 + period as u64);
                let mut updates = 0usize;
                for r in 0..EVENT_BUDGET {
                    let query = make_query(&geo, r, &mut rng);
                    let class = regime.query_class(r);
                    // Every round is a retrieval event; only every `period`-th
                    // one pays for counterfactual scoring.
                    if r % period == 0 {
                        update_exact(table, &geo, &query, class, &cfg, &mut ledger, &mut s, noise);
                        updates += 1;
                    } else {
                        let mut tr = PrivilegeTrace::new();
                        privileged_score(
                            table, &geo, &query, class, &cfg, &ledger, &mut tr, &mut s,
                        );
                    }
                }
                (ledger, updates)
            };
            let (lp, updates) = train(&poisoned);
            let (lc, _) = train(&clean);
            let (p_good, p_poison) = privilege_split(&lp, regime);

            let mut sc = Scratch::new();
            let mut tr = PrivilegeTrace::new();
            let arm = evaluate(&geo, regime, |_, class, q| {
                let sp = privileged_score(&poisoned, &geo, q, class, &cfg, &lp, &mut tr, &mut sc);
                let scl = privileged_score(&clean, &geo, q, class, &cfg, &lc, &mut tr, &mut sc);
                let mass = utility_mass(&poisoned, &geo, q, class, &cfg, Some(&lp), &mut sc);
                (sp, scl, mass)
            });
            SparsityPoint {
                noise_amp,
                period,
                updates,
                recovery: (naive.rel_err - arm.rel_err) / naive.rel_err,
                p_good,
                p_poison,
            }
        })
        .collect()
}

// ─── Regime driver ──────────────────────────────────────────────────────────

struct RegimeReport {
    regime: Regime,
    naive: ArmResult,
    priv_exact: ArmResult,
    priv_aggregate: ArmResult,
    recovery_exact: f64,
    recovery_aggregate: f64,
    exact_calls: usize,
    aggregate_calls: usize,
    p_good: f64,
    p_poison: f64,
    /// |p_poison after an even-length training run − after an odd-length run|.
    /// Large ⇒ the EMA is latching onto whichever query class it saw last
    /// instead of converging.
    recency_latch: f64,
}

fn run_regime(regime: Regime) -> RegimeReport {
    let mut geo_rng = Rng::new(0x6560_0001);
    let geo = Geometry::new(regime.two_class(), &mut geo_rng);
    let cfg = EngramConfig::for_dim(D);
    let mut s = Scratch::new();

    // Two tables per regime: the poisoned one and its oracle (poison evicted).
    let mut t_rng = Rng::new(0x6560_0002);
    let poisoned = build_table(&geo, regime, true, &mut t_rng);
    let mut t_rng = Rng::new(0x6560_0002); // same stream → identical good slots
    let clean = build_table(&geo, regime, false, &mut t_rng);

    // δ magnitudes here land around 0.3; tune the gate to that scale and
    // sharpen it (s ≈ 0.1·typical) so a sustained-negative slot is genuinely
    // vetoed rather than merely suppressed.
    let pcfg = PrivilegeConfig {
        scale: 0.03,
        margin: 0.0,
        ..PrivilegeConfig::for_delta_scale(0.3)
    };

    let mut train =
        |table: &dyn EngramTable, exact: bool, rounds: usize| -> (PrivilegeLedger, usize) {
            let mut ledger = PrivilegeLedger::new(table.num_slots(), pcfg);
            let mut rng = Rng::new(0x6560_0003);
            let mut calls = 0usize;
            for r in 0..rounds {
                let query = make_query(&geo, r, &mut rng);
                let class = regime.query_class(r);
                calls += if exact {
                    update_exact(
                        table,
                        &geo,
                        &query,
                        class,
                        &cfg,
                        &mut ledger,
                        &mut s,
                        &mut Noise::none(),
                    )
                } else {
                    update_aggregate(table, &geo, &query, class, &cfg, &mut ledger, &mut s)
                };
            }
            (ledger, calls)
        };

    let (led_poisoned_exact, exact_calls) = train(&poisoned, true, TRAIN_ROUNDS);
    let (led_clean_exact, _) = train(&clean, true, TRAIN_ROUNDS);
    let (led_poisoned_agg, aggregate_calls) = train(&poisoned, false, TRAIN_ROUNDS);
    let (led_clean_agg, _) = train(&clean, false, TRAIN_ROUNDS);
    // One extra round flips which query class the EMA saw last.
    let (led_odd, _) = train(&poisoned, true, TRAIN_ROUNDS + 1);

    let (p_good, p_poison) = privilege_split(&led_poisoned_exact, regime);
    let (_, p_poison_odd) = privilege_split(&led_odd, regime);
    let recency_latch = if p_poison.is_nan() {
        0.0
    } else {
        (p_poison - p_poison_odd).abs()
    };

    // ── Evaluate ────────────────────────────────────────────────────────────
    let naive = {
        let mut sc = Scratch::new();
        evaluate(&geo, regime, |_, class, q| {
            let sp = naive_score(&poisoned, &geo, q, class, &cfg, &mut sc);
            let scl = naive_score(&clean, &geo, q, class, &cfg, &mut sc);
            let mass = utility_mass(&poisoned, &geo, q, class, &cfg, None, &mut sc);
            (sp, scl, mass)
        })
    };

    let eval_priv = |lp: &PrivilegeLedger, lc: &PrivilegeLedger| {
        let mut sc = Scratch::new();
        let mut tr = PrivilegeTrace::new();
        evaluate(&geo, regime, |_, class, q| {
            let sp = privileged_score(&poisoned, &geo, q, class, &cfg, lp, &mut tr, &mut sc);
            let scl = privileged_score(&clean, &geo, q, class, &cfg, lc, &mut tr, &mut sc);
            let mass = utility_mass(&poisoned, &geo, q, class, &cfg, Some(lp), &mut sc);
            (sp, scl, mass)
        })
    };
    let priv_exact = eval_priv(&led_poisoned_exact, &led_clean_exact);
    let priv_aggregate = eval_priv(&led_poisoned_agg, &led_clean_agg);

    let recovery = |arm: &ArmResult| -> f64 {
        if naive.rel_err > 1e-9 {
            (naive.rel_err - arm.rel_err) / naive.rel_err
        } else {
            f64::NAN
        }
    };

    RegimeReport {
        regime,
        recovery_exact: recovery(&priv_exact),
        recovery_aggregate: recovery(&priv_aggregate),
        naive,
        priv_exact,
        priv_aggregate,
        exact_calls,
        aggregate_calls,
        p_good,
        p_poison,
        recency_latch,
    }
}

// ─── Cost ───────────────────────────────────────────────────────────────────

struct CostReport {
    ratio_fuse_only: f64,
    ratios: Vec<(usize, f64)>,
}

/// Wall-clock of `REPS` retrieval events, plain vs. privileged, with an exact
/// update every `period` events. `ratio ≤ 2.0` is the issue's T1 bar; `≤ 1.20`
/// is T3 G2.
///
/// # Estimator: interleaved min-of-N, not median-of-N
///
/// An earlier draft took the median of 3 sequential trials at 4k reps and read
/// 0.977× / 1.122× / 1.201× on **identical code** — straddling the 1.20× gate,
/// so the verdict was a coin flip on machine load. Two things were wrong:
///
/// 1. **Median tracks contention.** Scheduler noise, thermal drift, and
///    competing compiles only ever *add* time to a timed loop; they never
///    subtract. The minimum over trials is therefore the least-contaminated
///    estimator of the true cost, and is standard for latency microbenchmarks.
///    A median deliberately keeps a noise-inflated sample.
/// 2. **The baseline was measured once, up front.** Every privileged variant
///    was then divided by that single `t_naive`, so any load drift between the
///    baseline and a variant went straight into the ratio. Trials are now
///    **interleaved**: each trial times the baseline and every variant
///    back-to-back, so both sides see the same machine.
///
/// Reps are also 5× higher, putting each timed loop in the tens-of-ms range
/// where timer granularity is irrelevant.
fn measure_cost() -> CostReport {
    const REPS: usize = 20_000;
    const TRIALS: usize = 5;

    let mut geo_rng = Rng::new(0x6560_0001);
    let geo = Geometry::new(false, &mut geo_rng);
    let cfg = EngramConfig::for_dim(D);
    let mut t_rng = Rng::new(0x6560_0002);
    let table = build_table(&geo, Regime::SignOpposed, true, &mut t_rng);
    let mut s = Scratch::new();
    let mut rng = Rng::new(7);
    let queries: Vec<Vec<f32>> = (0..64).map(|i| make_query(&geo, i, &mut rng)).collect();

    let mut ledger = PrivilegeLedger::new(table.num_slots(), PrivilegeConfig::default());
    let mut trace = PrivilegeTrace::new();

    // Warm up both paths so trial 1 isn't paying for cold caches or first-call
    // code paths — which would otherwise become the minimum's competition.
    for q in &queries {
        naive_score(&table, &geo, q, 0, &cfg, &mut s);
        privileged_score(&table, &geo, q, 0, &cfg, &ledger, &mut trace, &mut s);
    }

    // `None` = the plain baseline; `Some(period)` = privileged, updating every
    // `period` events. Index 0 is the baseline, 1 is privileged-with-no-updates,
    // the rest are the cadences.
    let variants: Vec<Option<Option<usize>>> =
        std::iter::once(None) // baseline
            .chain(std::iter::once(Some(None))) // privileged, no updates
            .chain(PERIODS.iter().map(|&p| Some(Some(p))))
            .collect();
    let mut best = vec![f64::INFINITY; variants.len()];

    for _ in 0..TRIALS {
        for (vi, variant) in variants.iter().enumerate() {
            let t0 = Instant::now();
            let mut acc = 0.0f32;
            for i in 0..REPS {
                let q = &queries[i % queries.len()];
                match variant {
                    None => acc += naive_score(&table, &geo, q, 0, &cfg, &mut s),
                    Some(period) => {
                        acc +=
                            privileged_score(&table, &geo, q, 0, &cfg, &ledger, &mut trace, &mut s);
                        if let Some(p) = period
                            && i % p == 0
                        {
                            update_exact(
                                &table,
                                &geo,
                                q,
                                0,
                                &cfg,
                                &mut ledger,
                                &mut s,
                                &mut Noise::none(),
                            );
                        }
                    }
                }
            }
            let t = t0.elapsed().as_nanos() as f64;
            std::hint::black_box(acc);
            best[vi] = best[vi].min(t);
        }
    }

    let t_naive = best[0];
    CostReport {
        ratio_fuse_only: best[1] / t_naive,
        ratios: PERIODS
            .into_iter()
            .enumerate()
            .map(|(i, p)| (p, best[i + 2] / t_naive))
            .collect(),
    }
}

// ─── Report ─────────────────────────────────────────────────────────────────

fn main() {
    println!("\n════ Issue 656 T1 — planted-drift PoC (counterfactual privilege gating) ════");
    println!(
        "D={D}  slots={N_SLOTS}  active={N_ACTIVE}  poison={:.0}%  train_rounds={TRAIN_ROUNDS}  eval_queries={EVAL_QUERIES}",
        POISON_FRAC * 100.0
    );

    let reports: Vec<RegimeReport> = [
        Regime::SignOpposed,
        Regime::SimilaritySeparable,
        Regime::Clean,
        Regime::ClassConditional,
    ]
    .into_iter()
    .map(run_regime)
    .collect();

    println!("\n── Quality (priv-exact: K+1 scorer calls per update) ─────────────────────");
    println!(
        "{:<38} {:>10} {:>10} {:>10} {:>9}",
        "regime", "err_naive", "err_exact", "recovery", "purity↑"
    );
    for r in &reports {
        println!(
            "{:<38} {:>10.4} {:>10.4} {:>9.1}% {:>9.3}",
            r.regime.name(),
            r.naive.rel_err,
            r.priv_exact.rel_err,
            r.recovery_exact * 100.0,
            r.priv_exact.purity
        );
    }

    println!("\n── Did the gate discriminate? (mean privilege factor by slot class) ──────");
    println!(
        "{:<38} {:>10} {:>10} {:>10} {:>14}",
        "regime", "p_good", "p_poison", "ratio", "recency_latch"
    );
    for r in &reports {
        let ratio = if r.p_poison > 1e-9 {
            format!("{:.1}×", r.p_good / r.p_poison)
        } else {
            "∞".to_string()
        };
        println!(
            "{:<38} {:>10.4} {:>10.4} {:>10} {:>14.4}",
            r.regime.name(),
            r.p_good,
            r.p_poison,
            ratio,
            r.recency_latch
        );
    }

    println!("\n── Cheap aggregate attribution (GateWeighted, 2 scorer calls) ───────────");
    println!(
        "{:<38} {:>10} {:>10} {:>12} {:>10}",
        "regime", "err_agg", "recovery", "calls_exact", "calls_agg"
    );
    for r in &reports {
        println!(
            "{:<38} {:>10.4} {:>9.1}% {:>12} {:>10}",
            r.regime.name(),
            r.priv_aggregate.rel_err,
            r.recovery_aggregate * 100.0,
            r.exact_calls,
            r.aggregate_calls
        );
    }

    println!("\n── Purity: naive vs priv-exact ──────────────────────────────────────────");
    for r in &reports {
        println!(
            "{:<38} naive {:>7.3}   priv-exact {:>7.3}",
            r.regime.name(),
            r.naive.purity,
            r.priv_exact.purity
        );
    }

    let cost = measure_cost();
    println!("\n── Cost (interleaved min of 5 × 20k reps; wall-clock ratio vs. plain fuse) ──");
    println!(
        "  fuse only (no updates)        {:.3}×",
        cost.ratio_fuse_only
    );
    for (p, r) in &cost.ratios {
        println!("  + exact update every {p:>3}      {r:.3}×");
    }

    // The joint curve. Quality and cost measured at the SAME cadence — the
    // fuse-only ratio is a floor for a ledger that never learns, so quoting it
    // as "the cost of privilege gating" would be measuring the wrong thing.
    let sweep = sparsity_sweep(&PERIODS, 0.0);
    let cost_at = |period: usize| -> f64 {
        cost.ratios
            .iter()
            .find(|(p, _)| *p == period)
            .map_or(f64::NAN, |(_, r)| *r)
    };
    println!(
        "\n── Amortization: quality AND cost at the same cadence (regime A, {EVENT_BUDGET} events) ──"
    );
    println!(
        "{:>8} {:>9} {:>10} {:>9} {:>10} {:>8}",
        "period", "updates", "recovery", "cost", "p_good", "p_poison"
    );
    for sp in &sweep {
        println!(
            "{:>8} {:>9} {:>9.1}% {:>8.3}× {:>10.4} {:>8.4}",
            sp.period,
            sp.updates,
            sp.recovery * 100.0,
            cost_at(sp.period),
            sp.p_good,
            sp.p_poison
        );
    }

    // The sweep above is measured on a NOISE-FREE δ, which is why "7 updates
    // suffice" falls out of it. A host's outcome verification is never that
    // clean, and the honest question is how fast that headline degrades.
    println!(
        "\n── Outcome-noise sensitivity (regime A; recovery %, noise as a multiple of |δ|) ──"
    );
    let noisy: Vec<Vec<SparsityPoint>> = [0.0f32, 1.0, 3.0, 8.0]
        .iter()
        .map(|&amp| sparsity_sweep(&PERIODS, amp))
        .collect();
    let header: Vec<String> = noisy
        .iter()
        .map(|run| {
            let amp = run.first().map_or(0.0, |sp| sp.noise_amp);
            format!("{:>10}", format!("noise {amp:.0}×"))
        })
        .collect();
    println!("{:>8} {}", "period", header.join(" "));
    for (i, period) in PERIODS.iter().enumerate() {
        let cells: Vec<String> = noisy
            .iter()
            .map(|run| format!("{:>9.1}%", run[i].recovery * 100.0))
            .collect();
        println!("{period:>8} {}", cells.join(" "));
    }

    // ── Verdict ─────────────────────────────────────────────────────────────
    println!("\n── Verdict ──────────────────────────────────────────────────────────────");
    let a = &reports[0];
    let b = &reports[1];
    let c = &reports[2];
    let d = &reports[3];

    let mut failures: Vec<String> = Vec::new();
    let mut gate = |ok: bool, line: String, fail_msg: String| {
        println!("  [{}] {line}", pf(ok));
        if !ok {
            failures.push(fail_msg);
        }
    };

    // T1 primary: regime A must clear the recovery bar (updates every round).
    gate(
        a.recovery_exact >= RECOVERY_BAR,
        format!(
            "T1  regime A recovery {:.1}% ≥ {:.0}%",
            a.recovery_exact * 100.0,
            RECOVERY_BAR * 100.0
        ),
        format!(
            "T1: sign-opposed recovery {:.1}% < {:.0}%",
            a.recovery_exact * 100.0,
            RECOVERY_BAR * 100.0
        ),
    );

    // T1 cost: "recovers ≥ half the penalty AT ≤ 2× retrieval-event cost" —
    // both conditions at ONE cadence, not cherry-picked from two tables.
    let t1_joint = sweep
        .iter()
        .filter(|sp| sp.recovery >= RECOVERY_BAR && cost_at(sp.period) <= 2.0)
        .min_by(|x, y| {
            cost_at(x.period)
                .partial_cmp(&cost_at(y.period))
                .expect("cost ratios are finite")
        });
    gate(
        t1_joint.is_some(),
        match t1_joint {
            Some(sp) => format!(
                "T1  joint: recovery {:.1}% ≥ 50% AND cost {:.3}× ≤ 2.0× at period {} ({} updates)",
                sp.recovery * 100.0,
                cost_at(sp.period),
                sp.period,
                sp.updates
            ),
            None => "T1  no cadence clears recovery ≥ 50% and cost ≤ 2.0× together".to_string(),
        },
        "T1: no cadence achieves recovery ≥ 50% at ≤ 2× retrieval cost".into(),
    );

    // G1: clean table — the gate must not manufacture error.
    gate(
        c.priv_exact.rel_err <= 1e-3,
        format!(
            "G1  clean-table rel_err {:.2e} ≤ 1e-3 (gate is a no-op without poison)",
            c.priv_exact.rel_err
        ),
        format!(
            "G1: clean-table rel_err {:.2e} > 1e-3",
            c.priv_exact.rel_err
        ),
    );

    // G2 (the issue's wording: "amortized ≤ +20% at retrieval events"). Gate on
    // the AMORTIZED cost at a cadence that still clears the recovery bar — not
    // on the fuse-only floor, which is the cost of a ledger that never learns.
    let g2 = sweep
        .iter()
        .filter(|sp| sp.recovery >= RECOVERY_BAR && cost_at(sp.period) <= 1.20)
        .min_by_key(|sp| sp.period);
    gate(
        g2.is_some(),
        match g2 {
            Some(sp) => format!(
                "G2  amortized {:.3}× ≤ 1.20× at period {} while holding recovery {:.1}%",
                cost_at(sp.period),
                sp.period,
                sp.recovery * 100.0
            ),
            None => "G2  no cadence holds recovery ≥ 50% under 1.20×".to_string(),
        },
        "G2: no cadence holds recovery ≥ 50% under +20% amortized cost".into(),
    );
    println!(
        "         (hot-path floor, zero updates: {:.3}× — reported, NOT the gate; \
         a ledger that never updates never learns)",
        cost.ratio_fuse_only
    );

    println!("\n  Scope findings (reported, not gated):");
    println!(
        "    · Control B: naive err {:.4} vs regime A's {:.4} ({:.0}× smaller) — the \
         similarity gate already does nearly all of the work on similarity-separable \
         drift; privilege gating only polishes the remainder ({:.1}% of an \
         already-tiny penalty).",
        b.naive.rel_err,
        a.naive.rel_err,
        a.naive.rel_err / b.naive.rel_err.max(1e-12),
        b.recovery_exact * 100.0
    );
    println!(
        "    · Scope limit D (class-conditional utility): recovery {:.1}%, p_good {:.3} vs \
         p_poison {:.3}, recency_latch {:.4}. A per-slot SCALAR Δ averages over query \
         classes — it cannot represent \"helps class 0, hurts class 1\". Query-conditional \
         gating would need the query in the ledger key.",
        d.recovery_exact * 100.0,
        d.p_good,
        d.p_poison,
        d.recency_latch
    );
    println!(
        "    · Cheap aggregate attribution on regime A: recovery {:.1}% (vs {:.1}% exact) — \
         unsigned weights cannot split a near-zero aggregate δ across sign-opposed slots. \
         The 8.5× scorer-call saving buys nothing in the regime that motivates the gate.",
        a.recovery_aggregate * 100.0,
        a.recovery_exact * 100.0
    );
    let sparse_noisy = noisy.last().and_then(|run| run.last());
    println!(
        "    · The \"{} updates suffice\" headline is measured on a NOISE-FREE δ. At 8× \
         outcome noise and the same sparse cadence it reads {:.1}% — sparse updating and \
         noisy outcomes are not independently safe choices. Budget updates against the \
         host's actual outcome-label quality, not against this fixture.",
        sweep.last().map_or(0, |sp| sp.updates),
        sparse_noisy.map_or(f64::NAN, |sp| sp.recovery) * 100.0
    );

    if failures.is_empty() {
        println!("\n  ALL GATES PASS\n")
    } else {
        println!("\n  FAILURES:");
        for f in &failures {
            println!("    · {f}");
        }
        println!();
        std::process::exit(1);
    }
}

fn pf(b: bool) -> &'static str {
    if b { "PASS" } else { "FAIL" }
}
