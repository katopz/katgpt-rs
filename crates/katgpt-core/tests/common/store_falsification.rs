//! Store falsification arms — the Issue 984 standing GOAT-gate pattern for
//! memory-input primitives (Research 656, arXiv:2610.21375 Memorilla).
//!
//! The instrument: a memory read whose input is garbage must fail loudly.
//! Memorilla's noise-store proof (their FactKG −5.93 vs the documents-only
//! variant's +0.04 under a Gaussian-noise store) shows a read that does not
//! condition on evidence is NOISE-INVARIANT — absent work wearing numbers,
//! the timed-region loud-zero defense one abstraction up. These arms ask
//! "does this primitive actually consume its store?" of any store-backed
//! read, alongside (never replacing) the caller's own fixture controls
//! (e.g. Bench 845's W2 negative-control world).
//!
//! # The four store conditions
//!
//! 1. **real** — the store as shipped.
//! 2. **noise** — BLAKE3-seeded iid Gaussian matched to the store's GLOBAL
//!    entry scale, rows unit-normalized (see
//!    `store_transforms::noise_store`). Destroys internal structure AND the
//!    query binding; preserves overall energy. No RNG drift: the seed is
//!    BLAKE3 over a domain label + the store's own bit pattern, so the same
//!    store reproduces the same noise across runs and platforms.
//!    ⚠ Measured lesson (Issue 984, first catch — the arms caught
//!    THEMSELVES): an earlier arm matched PER-DIMENSION marginals instead.
//!    Per-dim std is dominated by between-cluster variance, so the noise
//!    rows concentrate near the cluster-center subspace — and a max-sim
//!    read recovers structure right through that bias (Bench 845 full
//!    scale: noise delta −0.0003 AURC, I1 fail while the real read was
//!    fine). For max-similarity reads, per-dim marginal matching is NOT
//!    structure destruction; match global energy only.
//! 3. **shuffled** — one fixed BLAKE3-seeded coordinate permutation applied
//!    to EVERY row (see `store_transforms::shuffled_store`). For a set-max
//!    read with no per-exemplar provenance emission (the
//!    CorpusDistanceGate max-cosine confidence, for example), a ROW
//!    permutation is the identity on the read — the observable binding is
//!    the shared basis. A coordinate permutation is orthogonal, so it
//!    conserves every pairwise store cosine exactly (internal geometry
//!    intact) while destroying the query↔store channel. This is the honest
//!    instantiation of Memorilla's "permuted embedding↔provenance binding"
//!    for scalar-confidence reads; a primitive that EMITS per-exemplar
//!    provenance should override with a true binding shuffle.
//! 4. **empty** — no primitive / no store: the caller supplies the
//!    no-memory floor signals (for Bench 845, the model's reported score —
//!    the score-only arm). Memorilla's 0.153-vs-0.186 lesson: a bad memory
//!    arm can sit BELOW this floor.
//!
//! # Preregistered inequalities (AURC, lower is better)
//!
//! Any violation = the primitive injects phantom structure or ignores
//! evidence:
//!
//! - **I1 evidence-dependence** — `AURC(noise) ≥ AURC(real) + margin`:
//!   the real store must beat matched-statistics noise by a margin.
//! - **I2 structure-hallucination** — `AURC(noise) ≥ AURC(empty) − tol`:
//!   a noise store must not beat the no-memory floor.
//! - **I3 binding matters** — `AURC(shuffled) ≥ AURC(real) + margin`.
//!
//! # The canary (G-noise liveness)
//!
//! A store-blind stub read (constant signal) produces identical metrics
//! under every store condition, so I1's delta is exactly 0 < margin — the
//! stub MUST fail the inequalities. `canary_holds` asserts that polarity:
//! the real read passes AND the stub fails; a stub that passes proves the
//! gate is blind, which is itself a failure.
//!
//! # Usage (the standing convention for NEW memory-input primitives, T5)
//!
//! ```ignore
//! #[path = "common/store_falsification.rs"]
//! mod store_falsification;
//! use store_falsification::{run_store_arms, canary_holds, StoreArmMargins, StoreWorld};
//!
//! let worlds: Vec<StoreWorld<D>> = /* queries, errors, empty-floor signals */;
//! let verdict = run_store_arms(&real_store, &worlds, |store: &[[f32; D]]| {
//!     let gate = MyMemoryRead::new(store);
//!     let reader: StoreReader<D> = Box::new(move |queries: &[[f32; D]]| {
//!         queries.iter().map(|q| gate.confidence(q)).collect()
//!     });
//!     reader
//! }, &StoreArmMargins::default());
//! assert!(verdict.pass, "real read must consume its store");
//! let stub = run_store_arms(&real_store, &worlds, |_| {
//!     let stub: StoreReader<D> = Box::new(|queries| vec![0.5; queries.len()]);
//!     stub
//! }, &StoreArmMargins::default());
//! assert!(canary_holds(verdict.pass, stub.pass), "gate is live");
//! ```
//!
//! Feature-neutral: generic over the reader, std + blake3 (dev-dep) only.
//! The metric fns here are PRIVATE twins of `bench845_worlds`' public ones:
//! this module is included by consumers that never include the worlds
//! module (the Issue 984 named row), and `#[path]` double-includes are
//! duplicate-symbol errors — one home per include graph, not per codebase.

#[path = "store_transforms.rs"]
mod transforms;
// Re-exported: consumers of THIS module reach the transforms through it
// (a direct transforms include alongside this module would double-load the
// file — module-file includes must be single per crate).
pub use transforms::{noise_store, shuffled_store};

/// A store-parameterized reader: queries → confidence signals. Boxed so
/// callers can return owned-state readers (a gate built from the store) or
/// store-blind stubs without the `impl Fn` opaque-type identity friction
/// (a generic `R: Fn` param will not unify with an `impl Fn` opaque —
/// found while landing Issue 984).
pub type StoreReader<const D: usize> = Box<dyn Fn(&[[f32; D]]) -> Vec<f32>>;

/// The deep-abstention rate the readouts quote (Bench 845's ρ=30% — the
/// regime where the OOD mass is reachable).
pub const SEL30_RATE: f32 = 0.30;

/// One evaluation world: queries in the real-store basis, the drawn
/// outcomes as errors, and the no-memory floor signals (for Bench 845, the
/// model's reported confidence — the score-only arm).
pub struct StoreWorld<const D: usize> {
    pub queries: Vec<[f32; D]>,
    pub errors: Vec<bool>,
    pub empty_signals: Vec<f32>,
}

/// Preregistered margins (AURC scale). Defaults are the Issue 984
/// preregistration: evidence/binding ≥ 0.005 AURC, hallucination
/// tolerance 0.001 (equality-guard, not a free pass — a noise store
/// beating empty by more than 0.001 AURC is a red).
pub struct StoreArmMargins {
    pub evidence: f64,
    pub hallucination_tol: f64,
    pub binding: f64,
}

impl Default for StoreArmMargins {
    fn default() -> Self {
        Self {
            evidence: 0.005,
            hallucination_tol: 0.001,
            binding: 0.005,
        }
    }
}

/// Risk–coverage AURC: order by signal DESCENDING (most confident retained
/// first), average the prefix error rate over every coverage level. Lower
/// is better. O(n log n), one alloc — measurement-side only. (Moved here
/// from Bench 845 at Issue 984; private twin of `bench845_worlds::aurc` —
/// see the module doc for why there are two homes.)
fn aurc(signals: &[f32], errors: &[bool]) -> f32 {
    let mut idx: Vec<usize> = (0..signals.len()).collect();
    idx.sort_by(|a, b| signals[*b].total_cmp(&signals[*a]));
    let mut err_sum = 0usize;
    let mut total = 0.0f32;
    for (k, i) in idx.iter().enumerate() {
        err_sum += usize::from(errors[*i]);
        total += err_sum as f32 / (k + 1) as f32;
    }
    total / signals.len() as f32
}

/// Selective accuracy at matched abstain rate ρ: abstain the ρ-fraction of
/// LOWEST signal, return accuracy on the retained set. (Private twin of
/// `bench845_worlds::selective_accuracy_at`.)
fn selective_accuracy_at(signals: &[f32], errors: &[bool], rate: f32) -> f32 {
    let mut sorted: Vec<f32> = signals.to_vec();
    sorted.sort_by(f32::total_cmp);
    let cut = ((rate * signals.len() as f32) as usize).min(signals.len() - 1);
    let threshold = sorted[cut];
    let mut ok = 0usize;
    let mut kept = 0usize;
    for (s, e) in signals.iter().zip(errors) {
        if *s >= threshold {
            ok += usize::from(!*e);
            kept += 1;
        }
    }
    ok as f32 / kept as f32
}

/// Per-arm replicate-mean metrics under one store condition.
#[derive(Clone, Copy)]
struct ArmMean {
    aurc: f64,
    sel30: f64,
}

impl ArmMean {
    fn empty() -> Self {
        Self {
            aurc: 0.0,
            sel30: 0.0,
        }
    }
    fn push(&mut self, signals: &[f32], errors: &[bool]) {
        self.aurc += aurc(signals, errors) as f64;
        self.sel30 += selective_accuracy_at(signals, errors, SEL30_RATE) as f64;
    }
    fn finish(&mut self, n: usize) {
        self.aurc /= n as f64;
        self.sel30 /= n as f64;
    }
}

/// The verdict: per-arm means, the noise delta (the Memorilla +0.04-style
/// instrument readout, in BOTH directions), and the three inequalities.
pub struct StoreArmsVerdict {
    pub real: ArmOut,
    pub noise: ArmOut,
    pub shuffled: ArmOut,
    pub empty: ArmOut,
    /// Quality-direction readout: `sel30(real) − sel30(noise)` (≥ margin-ish
    /// means the real store is doing the work; Memorilla's +0.04 shape).
    pub noise_delta_sel30: f64,
    /// AURC-direction readout: `AURC(noise) − AURC(real)` (positive =
    /// evidence-dependence).
    pub noise_delta_aurc: f64,
    /// I1 evidence / I2 no-hallucination / I3 binding, in order.
    pub inequalities: [bool; 3],
    /// All three inequalities hold.
    pub pass: bool,
}

/// Public per-arm readout (replicate means).
#[derive(Clone, Copy)]
pub struct ArmOut {
    pub aurc: f64,
    pub sel30: f64,
}

/// Run the four store arms over `worlds` with a store-parameterized reader.
///
/// `make_reader` builds a reader from a store; it is called once per store
/// condition (real / noise / shuffled). The reader maps a world's queries
/// to confidence signals. Replicate means are taken over `worlds` in order.
pub fn run_store_arms<const D: usize, RB>(
    real_store: &[[f32; D]],
    worlds: &[StoreWorld<D>],
    make_reader: RB,
    margins: &StoreArmMargins,
) -> StoreArmsVerdict
where
    RB: Fn(&[[f32; D]]) -> StoreReader<D>,
{
    let noise_rows = noise_store(real_store);
    let shuffled_rows = shuffled_store(real_store);
    let read_real = make_reader(real_store);
    let read_noise = make_reader(&noise_rows);
    let read_shuffled = make_reader(&shuffled_rows);
    assert!(!worlds.is_empty(), "at least one replicate world required");

    let mut real = ArmMean::empty();
    let mut noise = ArmMean::empty();
    let mut shuffled = ArmMean::empty();
    let mut empty = ArmMean::empty();
    for w in worlds {
        assert_eq!(w.queries.len(), w.errors.len(), "world shape");
        assert_eq!(w.queries.len(), w.empty_signals.len(), "world shape");
        real.push(&read_real(&w.queries), &w.errors);
        noise.push(&read_noise(&w.queries), &w.errors);
        shuffled.push(&read_shuffled(&w.queries), &w.errors);
        empty.push(&w.empty_signals, &w.errors);
    }
    let n = worlds.len();
    for arm in [&mut real, &mut noise, &mut shuffled, &mut empty] {
        arm.finish(n);
    }
    let out = |a: &ArmMean| ArmOut {
        aurc: a.aurc,
        sel30: a.sel30,
    };
    let (real, noise, shuffled, empty) = (out(&real), out(&noise), out(&shuffled), out(&empty));

    let noise_delta_sel30 = real.sel30 - noise.sel30;
    let noise_delta_aurc = noise.aurc - real.aurc;
    // AURC lower-is-better: evidence and binding need a real-store WIN by
    // margin; the noise store must not beat the empty floor by more than tol.
    let i1 = noise.aurc >= real.aurc + margins.evidence;
    let i2 = noise.aurc >= empty.aurc - margins.hallucination_tol;
    let i3 = shuffled.aurc >= real.aurc + margins.binding;
    let pass = i1 && i2 && i3;
    StoreArmsVerdict {
        real,
        noise,
        shuffled,
        empty,
        noise_delta_sel30,
        noise_delta_aurc,
        inequalities: [i1, i2, i3],
        pass,
    }
}

/// The canary polarity: the REAL read must pass and a store-blind stub must
/// FAIL. A stub that passes proves the arms cannot detect evidence-blindness
/// (gate blind); a real read that fails is the primitive's own red. Either
/// way the canary pair is broken.
pub fn canary_holds(real_pass: bool, stub_pass: bool) -> bool {
    real_pass && !stub_pass
}

/// Print the verdict table (the instrument readout — every number quoted
/// beside its inequality).
pub fn print_verdict_table(name: &str, v: &StoreArmsVerdict) {
    println!("  {name:<10} AURC      sel30     noise Δ");
    let row = |label: &str, a: &ArmOut, delta: Option<f64>| {
        println!(
            "  {label:<10} {:-9.4} {:-9.4} {}",
            a.aurc,
            a.sel30,
            delta
                .map(|d| format!("  {:+.4}", d))
                .unwrap_or_else(|| "  —".to_string())
        );
    };
    row("real", &v.real, None);
    row("noise", &v.noise, Some(v.noise_delta_sel30));
    row("shuffled", &v.shuffled, None);
    row("empty", &v.empty, None);
    let [i1, i2, i3] = v.inequalities;
    println!(
        "  I1 evidence (noise ≥ real + m): {}   I2 no-hallucination (noise ≥ empty − t): {}   I3 binding (shuffled ≥ real + m): {}",
        if i1 { "OK" } else { "FAIL" },
        if i2 { "OK" } else { "FAIL" },
        if i3 { "OK" } else { "FAIL" }
    );
}

// NOTE: no #[cfg(test)] mod here — this file is #[path]-included by
// harness=false benches, where #[test] fns are stripped and their uses
// vanish (dead-code warnings). The transform tests live in
// tests/memory_store_falsification_gate.rs (the Issue 984 named row).
