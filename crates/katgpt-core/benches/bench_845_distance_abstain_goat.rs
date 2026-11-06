//! Bench 845 — corpus-distance abstain T1.6 gate (Proposal 014 T1.6 /
//! Issue 863; the Research 576 §2.1 `SalesRLAgent` extraction).
//!
//! The claim under test: a distance-derived confidence (max cosine to the
//! registered corpus) buys selective accuracy over the score-threshold
//! ABSTAIN baseline ([`CalibratedActionBridge::should_abstain`]) — but ONLY
//! when the failure mode is out-of-corpus. The bench separates the two
//! worlds on ONE geometry, which is the honest form of the claim:
//!
//! - **W1 (OOD-blind errors — the arena's world):** the model's reported
//!   score carries NO out-of-distribution information; queries far from the
//!   corpus collapse to near-chance accuracy while reporting the same
//!   confident scores. This is the hallucination regime both `SalesRLAgent`
//!   (arXiv:2503.23303) and the confidence-routing literature
//!   (arXiv:2510.01237) name. Distance carries error information the score
//!   lacks → the fused gate (score OR distance) must win.
//! - **W2 (score-only errors — negative control):** identical geometry,
//!   identical scores, identical OOD query share — but accuracy is driven
//!   by the score everywhere, OOD included. Distance now carries NO error
//!   information → distance-only must NOT beat the score gate, and the
//!   fused advantage must vanish. This is the fixture-discrimination
//!   control: a distance arm that won in W2 would prove the fixture rigged,
//!   not the mechanism real.
//!
//! Arms: `score` (the baseline — abstain low p), `dist` (abstain low
//! corpus-confidence), `fused` (the deployable OR, swept as `min` of the
//! two batch quantiles — rank fusion), `oracle` (W1 only, ranks by true
//! p — the labeled ceiling, printed never gated). Metrics: risk–coverage
//! AURC (lower is better) + selective accuracy at matched abstain rates
//! ρ ∈ {5, 10, 20, 30}%.
//!
//! Statistics: R world replicates (independent arrival + outcome draws over
//! the shared corpus geometry); every gate reads a replicate MEAN plus a
//! sign count, because a single-world selective-accuracy delta at these
//! operating points is ~1σ noise (measured: a one-off +0.63 pp at ρ=30%
//! is not a verdict; the mean over 8 replicates is).
//!
//! Gates:
//! - **G1** mechanism sanity (the module's unit tests carry the heavy half;
//!   here: every confidence in (0,1), no NaN across every replicate).
//! - **G2** W1 win (replicate means): AURC(fused) < AURC(score);
//!   mean Δ(sel-acc, fused−score) ≥ −0.2 pp at every ρ (non-inferiority —
//!   shallow abstention has no OOD headroom); mean Δ ≥ +0.2 pp at ρ=20%
//!   and ≥ +0.3 pp at ρ=30% (the deep-abstention regime where the OOD mass
//!   is reachable); Δ(ρ=30%) > 0 in at least 6 of 8 replicates.
//! - **G3** W2 control: mean AURC(dist) > AURC(score) AND mean
//!   AURC(fused-W2) > AURC(score) — the fused advantage must VANISH when
//!   distance carries no error information; mean Δ(dist−score) ≤ +0.2 pp
//!   at every ρ.
//! - **G4** zero-alloc query path (Issue-741 predicate).
//! - **G5** per-query latency: max-similarity over K=128, D=64 under a
//!   round budget. The budget is a REGRESSION CEILING on a SHARED box
//!   (measured spread on this M3: ~2.2 µs quiet-ish vs ~14.9 µs under
//!   sibling cargo load — the AGENTS.md box-state rule; the print is the
//!   measurement, the budget only catches gross de-optimization).
//! - **G6** memory-store falsification arms (Issue 984, the standing
//!   convention for NEW memory-input primitives): the four store
//!   conditions — real / BLAKE3-seeded matched-statistics noise /
//!   basis-shuffled (binding) / score-only empty floor — with the
//!   preregistered inequalities real-beats-noise, noise-never-beats-empty,
//!   shuffled-never-beats-real, plus the store-blind canary that must
//!   FAIL (a stub that passes proves the arms blind). The shared harness
//!   (`tests/common/store_falsification.rs`) is what every future
//!   memory-input primitive's GOAT gate inherits; W2 above remains the
//!   fixture-level control, G6 the store-level one.
//!
//! # Run
//!
//! ```bash
//! CARGO_TARGET_DIR=/tmp/bench845 cargo bench -p katgpt-core \
//!   --features distance_abstain --no-default-features \
//!   --bench bench_845_distance_abstain_goat -- --nocapture
//! ```

#![cfg(feature = "distance_abstain")]

#[path = "../tests/common/mod.rs"]
mod common;
#[path = "../tests/common/store_falsification.rs"]
mod store_falsification;
counting_allocator!();

use katgpt_core::distance_abstain::CorpusDistanceGate;
use std::hint::black_box;
use store_falsification::{
    StoreArmMargins, StoreWorld, canary_holds, print_verdict_table, run_store_arms,
};

// The W1/W2 worlds moved to the shared module at Issue 985 T6 (the
// soft-read second-signal cell re-runs the SAME worlds — never a fork).
// Moved functions, byte-identical behavior: same seeds, same draws.
#[path = "../tests/common/bench845_worlds.rs"]
mod worlds;
use worlds::{
    Arrival, D, K_EXEMPLARS, MID, N_TEST, OOD_SHARE, SCALE, aurc, build_corpus, build_world,
    selective_accuracy_at, w1_true_p, w2_true_p,
};

const N_REPS: usize = 8;
const RATES: [f32; 4] = [0.05, 0.10, 0.20, 0.30];

/// Empirical quantile of each value within its own batch (rank / n, in the
/// value's original order). Rank fusion puts the two signals on ONE scale
/// before the `min()` — without it, min(p, `d_conf`) degenerates to whichever
/// signal sits lower on the number line (measured: raw `d_conf` < p
/// everywhere, so the raw-min sweep collapsed to distance-only). Production
/// reads the same composition from running quantiles / an outcome
/// calibrator — the batch CDF here is the validation-grade stand-in, not an
/// oracle: it uses no outcome information.
fn ranks(values: &[f32]) -> Vec<f32> {
    let n = values.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|a, b| values[*a].total_cmp(&values[*b]));
    let mut out = vec![0.0f32; n];
    for (pos, i) in idx.iter().enumerate() {
        out[*i] = (pos + 1) as f32 / n as f32;
    }
    out
}

/// Risk–coverage AURC and selective accuracy live in the shared
/// store-falsification harness (`tests/common/store_falsification.rs`,
/// moved there at Issue 984) — one metric implementation for the T1.6
/// gates and the G6 store arms.
/// Per-arm metric row for one world replicate.
#[derive(Clone, Copy)]
struct ArmRow {
    aurc: f32,
    sel_acc: [f32; RATES.len()],
}

fn evaluate(signals: &[f32], world: &[Arrival]) -> ArmRow {
    let errors: Vec<bool> = world.iter().map(|a| !a.correct).collect();
    ArmRow {
        aurc: aurc(signals, &errors),
        sel_acc: std::array::from_fn(|i| selective_accuracy_at(signals, &errors, RATES[i])),
    }
}

fn distance_signals(gate: &CorpusDistanceGate<D>, world: &[Arrival]) -> Vec<f32> {
    world
        .iter()
        .map(|a| gate.abstain_confidence(&a.q))
        .collect()
}

fn fused_rank_signals(dists: &[f32], world: &[Arrival]) -> Vec<f32> {
    let rq_p = ranks(&world.iter().map(|a| a.p).collect::<Vec<f32>>());
    let rq_d = ranks(dists);
    rq_p.iter().zip(&rq_d).map(|(a, b)| a.min(*b)).collect()
}

/// Running mean over replicate rows.
struct MeanRow {
    aurc: f64,
    sel_acc: [f64; RATES.len()],
    n: usize,
}

impl MeanRow {
    fn new() -> Self {
        Self {
            aurc: 0.0,
            sel_acc: [0.0; RATES.len()],
            n: 0,
        }
    }
    fn push(&mut self, r: &ArmRow) {
        self.aurc += r.aurc as f64;
        for (acc, v) in self.sel_acc.iter_mut().zip(&r.sel_acc) {
            *acc += *v as f64;
        }
        self.n += 1;
    }
    fn aurc(&self) -> f64 {
        self.aurc / self.n as f64
    }
    fn sel(&self, i: usize) -> f64 {
        self.sel_acc[i] / self.n as f64
    }
    /// Per-rate means as values, so callers zip instead of indexing.
    fn sel_means(&self) -> [f64; RATES.len()] {
        self.sel_acc.map(|a| a / self.n as f64)
    }
}

fn print_arm(name: &str, m: &MeanRow) {
    print!("  {name:<8} AURC {:-7.4}   sel-acc:", m.aurc());
    for (r, s) in RATES.iter().zip(m.sel_means()) {
        print!("  ρ={:>4.0}% {:.4}", r * 100.0, s);
    }
    println!();
}

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Bench 845 — corpus-distance abstain T1.6 gate");
    println!(
        "  (Proposal 014 T1.6 / Issue 863; K={K_EXEMPLARS}, D={D}, n={N_TEST}, reps {N_REPS}, OOD {:.0}%)",
        OOD_SHARE * 100.0
    );
    println!("═══════════════════════════════════════════════════════════════");

    let corpus = build_corpus();
    let gate = CorpusDistanceGate::<D>::new(&corpus, MID, SCALE);

    // G1 — mechanism sanity across every arrival in every replicate.
    let mut g1 = true;

    let mut w1_score = MeanRow::new();
    let mut w1_dist = MeanRow::new();
    let mut w1_fused = MeanRow::new();
    let mut w1_oracle = MeanRow::new();
    let mut w1_delta30_positive = 0usize;

    let mut w2_score = MeanRow::new();
    let mut w2_dist = MeanRow::new();
    let mut w2_fused = MeanRow::new();

    for rep in 0..N_REPS {
        // W1 — OOD-blind errors: distance carries information the score lacks.
        let w1 = build_world(0x845_0100 + rep as u64, &corpus, w1_true_p);
        let dists = distance_signals(&gate, &w1);
        let s_score = evaluate(&w1.iter().map(|a| a.p).collect::<Vec<f32>>(), &w1);
        let s_dist = evaluate(&dists, &w1);
        let s_fused = evaluate(&fused_rank_signals(&dists, &w1), &w1);
        let s_oracle = evaluate(
            &w1.iter()
                .map(|a| w1_true_p(a.p, a.ood))
                .collect::<Vec<f32>>(),
            &w1,
        );
        for a in &w1 {
            let c = gate.abstain_confidence(&a.q);
            g1 &= c.is_finite() && c > 0.0 && c < 1.0;
        }
        if s_fused.sel_acc[3] > s_score.sel_acc[3] {
            w1_delta30_positive += 1;
        }
        w1_score.push(&s_score);
        w1_dist.push(&s_dist);
        w1_fused.push(&s_fused);
        w1_oracle.push(&s_oracle);

        // W2 — score-only errors: distance carries nothing (negative control).
        let w2 = build_world(0x845_0200 + rep as u64, &corpus, w2_true_p);
        let dists2 = distance_signals(&gate, &w2);
        let t_score = evaluate(&w2.iter().map(|a| a.p).collect::<Vec<f32>>(), &w2);
        let t_dist = evaluate(&dists2, &w2);
        let t_fused = evaluate(&fused_rank_signals(&dists2, &w2), &w2);
        for a in &w2 {
            let c = gate.abstain_confidence(&a.q);
            g1 &= c.is_finite() && c > 0.0 && c < 1.0;
        }
        w2_score.push(&t_score);
        w2_dist.push(&t_dist);
        w2_fused.push(&t_fused);
    }
    println!(
        "G1 — confidence in (0,1), NaN-free over 2×{N_TEST}×{N_REPS} queries: {}",
        if g1 { "OK" } else { "FAIL" }
    );

    println!();
    println!("W1 — OOD-blind errors (the arena's world), replicate means:");
    print_arm("score", &w1_score);
    print_arm("dist", &w1_dist);
    print_arm("fused", &w1_fused);
    print_arm("oracle*", &w1_oracle);
    println!("  (*oracle ranks by true p — labeled ceiling, never a gate)");

    println!();
    println!("W2 — score-only errors (negative control), replicate means:");
    print_arm("score", &w2_score);
    print_arm("dist", &w2_dist);
    print_arm("fused", &w2_fused);

    // ── G2 — the T1.6 win gate (W1, replicate means) ────────────────────
    println!();
    println!("G2 — W1 fused win vs the score-threshold baseline (means over {N_REPS} reps):");
    let g2_aurc = w1_fused.aurc() < w1_score.aurc();
    println!(
        "  AURC fused {:.4} < score {:.4} : {}",
        w1_fused.aurc(),
        w1_score.aurc(),
        if g2_aurc { "OK" } else { "FAIL" }
    );
    // Non-inferiority at every ρ (mean), deep-regime wins, sign consistency.
    let mut g2_noninf = true;
    for ((r, f), s) in RATES
        .iter()
        .zip(w1_fused.sel_means())
        .zip(w1_score.sel_means())
    {
        let delta = f - s;
        let ok = delta >= -0.002;
        g2_noninf &= ok;
        println!(
            "  ρ={:>4.0}%  mean Δ(fused−score) {:+.4}  {}",
            r * 100.0,
            delta,
            if ok { "ok" } else { "FAIL" }
        );
    }
    let mean20 = w1_fused.sel(2) - w1_score.sel(2);
    let mean30 = w1_fused.sel(3) - w1_score.sel(3);
    let g2_margins = mean20 >= 0.002 && mean30 >= 0.003;
    println!(
        "  deep-regime means: ρ=20% {:+.4} (≥+0.002), ρ=30% {:+.4} (≥+0.003) : {}",
        mean20,
        mean30,
        if g2_margins { "OK" } else { "FAIL" }
    );
    let g2_sign = w1_delta30_positive >= 6;
    println!(
        "  Δ(ρ=30%) > 0 in {w1_delta30_positive}/{N_REPS} replicates (≥6) : {}",
        if g2_sign { "OK" } else { "FAIL" }
    );
    let g2 = g2_aurc && g2_noninf && g2_margins && g2_sign;

    // ── G3 — the fixture-discrimination control (W2) ────────────────────
    println!();
    println!("G3 — W2 control (the fused advantage must VANISH when errors are score-driven):");
    let g3_dist_aurc = w2_dist.aurc() > w2_score.aurc();
    let g3_fused_aurc = w2_fused.aurc() > w2_score.aurc();
    println!(
        "  AURC dist {:.4} > score {:.4} : {}",
        w2_dist.aurc(),
        w2_score.aurc(),
        if g3_dist_aurc { "OK" } else { "FAIL" }
    );
    println!(
        "  AURC fused-W2 {:.4} > score {:.4} : {}",
        w2_fused.aurc(),
        w2_score.aurc(),
        if g3_fused_aurc { "OK" } else { "FAIL" }
    );
    let mut g3_rates = true;
    for ((r, d), s) in RATES
        .iter()
        .zip(w2_dist.sel_means())
        .zip(w2_score.sel_means())
    {
        let delta = d - s;
        let ok = delta <= 0.002;
        g3_rates &= ok;
        println!(
            "  ρ={:>4.0}%  mean Δ(dist−score) {:+.4} (≤+0.002)  {}",
            r * 100.0,
            delta,
            if ok { "ok" } else { "FAIL" }
        );
    }
    let g3 = g3_dist_aurc && g3_fused_aurc && g3_rates;

    // ── G4 — zero-alloc query path (Issue-741 predicate) ────────────────
    use std::sync::atomic::Ordering;
    {
        let _probe: Vec<u8> = vec![0u8; 64];
        black_box(&_probe);
        let live = ALLOC_COUNT.load(Ordering::Relaxed) > 0;
        assert!(live, "CountingAllocator not installed — G4 vacuous");
    }
    let w1 = build_world(0x845_0100, &corpus, w1_true_p);
    let before = ALLOC_COUNT.load(Ordering::Relaxed);
    let mut sink = 0.0f32;
    for a in w1.iter() {
        sink += gate.abstain_confidence(&a.q);
        black_box(gate.fused_should_abstain(&a.q, a.p, 0.75, 0.5));
    }
    black_box(sink);
    let g4_delta = ALLOC_COUNT.load(Ordering::Relaxed).saturating_sub(before);
    let g4 = g4_delta == 0;
    println!();
    println!(
        "G4 — query-path allocations over {N_TEST} arrivals: {g4_delta} {}",
        if g4 { "(OK)" } else { "(FAIL)" }
    );

    // ── G5 — per-query latency (K×D fold; regression ceiling, the print
    //        is the measurement; best-of across reps against box load) ──
    let reps = 9usize;
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t0 = std::time::Instant::now();
        let mut acc = 0.0f32;
        for a in w1.iter() {
            acc += gate.max_similarity(&a.q);
        }
        black_box(acc);
        let dt = t0.elapsed().as_secs_f64();
        let per_q = dt / (N_TEST as f64) * 1e9;
        if per_q < best {
            best = per_q;
        }
    }
    let g5 = best < 20_000.0;
    println!(
        "G5 — max_similarity latency: best {best:.0} ns/query (reps {reps}, ceiling 20000 ns) {}",
        if g5 { "OK" } else { "FAIL" }
    );

    // ── G6 — memory-store falsification arms (Issue 984; the T2 arms) ──
    // Does the gate actually CONSUME its store? Four conditions (real /
    // BLAKE3-seeded matched noise / basis-shuffled / score-only empty
    // floor) + the store-blind canary that must FAIL. AURC lower-is-better.
    let margins = StoreArmMargins::default();
    let worlds: Vec<StoreWorld<D>> = (0..N_REPS)
        .map(|rep| {
            let w1 = build_world(0x845_0100 + rep as u64, &corpus, w1_true_p);
            StoreWorld {
                queries: w1.iter().map(|a| a.q).collect(),
                errors: w1.iter().map(|a| !a.correct).collect(),
                empty_signals: w1.iter().map(|a| a.p).collect(),
            }
        })
        .collect();
    let gate_reader = |store: &[[f32; D]]| -> store_falsification::StoreReader<D> {
        let gate = CorpusDistanceGate::<D>::new(store, MID, SCALE);
        Box::new(move |queries: &[[f32; D]]| {
            queries.iter().map(|q| gate.abstain_confidence(q)).collect()
        })
    };
    let stub_reader = |_store: &[[f32; D]]| -> store_falsification::StoreReader<D> {
        Box::new(|queries: &[[f32; D]]| vec![0.5f32; queries.len()])
    };
    let arms = run_store_arms(&corpus, &worlds, gate_reader, &margins);
    let stub = run_store_arms(&corpus, &worlds, stub_reader, &margins);
    println!();
    println!(
        "G6 — memory-store falsification arms (Issue 984; AURC lower-is-better, sel30 = selective accuracy at ρ=30%):"
    );
    println!("  CorpusDistanceGate under four store conditions:");
    print_verdict_table("gate", &arms);
    println!(
        "  noise deltas: sel30 {:+.4} · AURC {:+.4} (the Memorilla +0.04-style readout)",
        arms.noise_delta_sel30, arms.noise_delta_aurc
    );
    println!("  canary — store-blind stub (must FAIL the arms):");
    print_verdict_table("stub", &stub);
    let g6_canary = canary_holds(arms.pass, stub.pass);
    println!(
        "  canary polarity (real pass ∧ stub fail): {}",
        if g6_canary {
            "OK"
        } else {
            "FAIL — gate blind"
        }
    );
    let g6 = arms.pass && g6_canary;

    // ── Verdict ─────────────────────────────────────────────────────────
    println!();
    let all = g1 && g2 && g3 && g4 && g5 && g6;
    println!("GATES: G1 {g1} · G2 {g2} · G3 {g3} · G4 {g4} · G5 {g5} · G6 {g6}");
    println!(
        "{}",
        if all {
            "✓ Bench 845 T1.6 GATES PASSED — corpus-distance abstain earns the opt-in lane"
        } else {
            "✗ Bench 845 T1.6 GATES FAILED"
        }
    );
    assert!(all, "T1.6 gates must pass");
}
