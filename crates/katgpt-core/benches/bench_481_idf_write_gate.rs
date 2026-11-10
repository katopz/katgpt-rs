//! TF-IDF write-gate GOAT bench (Issue 650 / Research 481).
//!
//! Measures the **G1 interference/retention gate** — the load-bearing
//! defend-wrong PoC for TF-IDF slot selection ("Continual Learning via
//! Sparse Memory Finetuning", arXiv:2510.15103 §6 Fig 6):
//!
//! - **Task**: write fact set A, then fact set B (shared table, overlapping
//!   key space — both drawn from the same broad distribution). Measure
//!   recall(A) after B.
//! - **Arms** (matched candidate pool `POOL=16`, matched write width `T=4`):
//!   - **IDF** — `write_idf` with real `BackgroundAccessStats` built from a
//!     background query corpus (selection = weight × idf).
//!   - **TF-only** — `write_idf` with degenerate all-zero stats (uniform idf
//!     → selection = top-`t` by weight). This is the baseline arm.
//!   - **Random-t** — pool retrieved, `t` slots picked by a seeded LCG,
//!     applied via `write_selected` (the non-interference control).
//! - **Matched learning(B)**: each arm's `gate` swept, the gate whose final
//!   recall(B) is closest to the target (`min(0.85, TF's best)`) is chosen —
//!   the paper's regime assumes TF-only "learns comparably"; at our tiny
//!   t=4 write set TF may be capacity-bound at hot slots, in which case the
//!   target ratchets down to what TF can reach and the honest asymmetry is
//!   reported.
//! - **PASS**: IDF recall(A) ≥ TF-only recall(A) + 10pp absolute, at matched
//!   recall(B).
//!
//! - **G2 (latency)**: per-write overhead of the idf fold (O(k) multiplies +
//!   one top-t selection) vs a plain `write` — report ns/write both arms.
//! - **G4 (alloc-free)**: 0 allocations across 1000 steady-state `write_idf`
//!   calls (stats table preallocated; scratch reused).
//!
//! **Recall metric**: a fact is retained if max-cos(value-row, target) over
//! the fact's own top-16 retrieved neighborhood ≥ 0.90 (the paper's read is
//! full attention; a top-16 neighborhood read is the tractable analog).
//!
//! # Run
//!
//! ```bash
//! CARGO_TARGET_DIR=/tmp/pkm650 cargo bench -p katgpt-core \
//!   --features product_key_memory_episodic --bench bench_481_idf_write_gate -- --nocapture
//! ```

use katgpt_core::product_key_memory::{
    BackgroundAccessStats, PkmEpisodicStore, PkmScratch, ProductKeyMemory, ScoreFn,
};
use std::hint::black_box;
use std::time::Instant;

#[path = "../tests/common/mod.rs"]
mod common;
counting_allocator!();

// ── Constants ───────────────────────────────────────────────────────────────

/// Table dimensions. SQRT_N=32 → N=1024 slots. D_K=8 (halves 4-dim, enough
/// for 32-row codebooks to discriminate). D_V=4.
const SQRT_N: usize = 32;
const N_SLOTS: usize = SQRT_N * SQRT_N;
const D_K: usize = 8;
const D_V: usize = 4;
/// Per-codebook top-k. K=4 → final pool k ≤ K*K = 16.
const K: usize = 4;
/// Candidate pool per write (query_into k).
const POOL: usize = 16;
/// Write width (top-t selected from the pool).
const T: usize = 4;

/// Background corpus: 512 queries in batches of 16 → |B| = 32 batches.
const N_BG: usize = 512;
const BG_BATCH: usize = 16;

/// Facts per set (A then B — both from the same broad distribution, so their
/// candidate pools overlap at the generally-hot slots).
const N_FACTS: usize = 128;

/// Recall threshold: max-cos(value, target) over the top-16 neighborhood.
const RECALL_COS: f32 = 0.90;

/// G1 target: IDF recall(A) − TF recall(A) ≥ +10pp absolute.
const G1_MARGIN_TARGET: f32 = 0.10;

/// Sweep for the matched-learning gate tuning.
const GATES: [f32; 6] = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

// ── Deterministic splitmix64 PRNG (mirrors bench_408 benches) ──────────────

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
        }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn next_gauss(&mut self) -> f32 {
        let u1 = (self.next_u64() >> 40) as f32 / ((1u32 << 24) as f32);
        let u2 = (self.next_u64() >> 40) as f32 / ((1u32 << 24) as f32);
        let r = (-2.0f32 * (u1.max(1e-12)).ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        r * theta.cos()
    }
    /// Small deterministic LCG for the random-t control arm.
    fn next_below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Cosine similarity of two equal-length slices (0-norm safe).
fn cos(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    dot / (na.max(1e-12).sqrt() * nb.max(1e-12).sqrt())
}

type Pair = ([f32; D_K], [f32; D_V]);

/// Table regime for the G1 gate.
///
/// - `Organic` — plain `from_random` (untrained). Key norms have only
///   chi-fluctuation spread, so no slot is retrieved by ~all background
///   batches — the hot-slot pathology is mostly absent. IDF ≈ TF here is
///   the **no-harm control**: IDF must not damage recall when the
///   pathology it fixes is absent.
/// - `NormRamp` — key rows scaled by a linear norm ramp (1.0 → 1.0+spread).
///   Models the trained-table precondition: under dot scoring, magnitude
///   inflation concentrates retrieval on high-norm rows (the documented
///   `ScoreFn::Dot` magnitude-sensitivity — the exact pathology
///   `ScoreFn::Idw` exists to avoid on the read side). Background counts
///   concentrate on the hot rows; TF-only writes pile onto them; IDF
///   downweights them. This is the paper's regime (their Fig 6 measures
///   forgetting on a PKM trained by pretraining, which develops exactly
///   this activation concentration).
#[derive(Clone, Copy, PartialEq, Debug)]
enum Regime {
    Organic,
    NormRamp,
}

impl Regime {
    fn label(&self) -> &'static str {
        match self {
            Self::Organic => "organic (untrained table — no-harm control)",
            Self::NormRamp => "norm-ramp (trained-analog — hot-slot pathology)",
        }
    }
}

/// Norm-ramp spread: row 0 unscaled, row SQRT_N-1 scaled by 1+spread.
const NORM_SPREAD: f32 = 0.8;

fn build_table(regime: Regime) -> ProductKeyMemory<SQRT_N, D_K, D_V> {
    let mut table = ProductKeyMemory::from_random(42);
    if let Regime::NormRamp = regime {
        let half = D_K / 2;
        for i in 0..SQRT_N {
            let s = 1.0 + NORM_SPREAD * (i as f32 / (SQRT_N - 1) as f32);
            for x in table.keys_1[i * half..(i + 1) * half].iter_mut() {
                *x *= s;
            }
            for x in table.keys_2[i * half..(i + 1) * half].iter_mut() {
                *x *= s;
            }
        }
    }
    table
}

fn gen_pairs(seed: u64, n: usize) -> Vec<Pair> {
    let mut rng = Rng::new(seed);
    let mut pairs = Vec::with_capacity(n);
    for _ in 0..n {
        let mut q = [0.0f32; D_K];
        for x in q.iter_mut() {
            *x = rng.next_gauss();
        }
        let mut target = [0.0f32; D_V];
        for x in target.iter_mut() {
            *x = rng.next_gauss();
        }
        pairs.push((q, target));
    }
    pairs
}

fn gen_queries(seed: u64, n: usize) -> Vec<[f32; D_K]> {
    gen_pairs(seed, n).into_iter().map(|(q, _)| q).collect()
}

// ── Arms ────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Debug)]
enum Arm {
    Idf,
    Tf,
    Random,
}

impl Arm {
    fn label(&self) -> &'static str {
        match self {
            Self::Idf => "IDF   ",
            Self::Tf => "TF    ",
            Self::Random => "Random",
        }
    }
}

/// One fact write under `arm` at `gate` (the shared per-fact write policy).
#[allow(clippy::too_many_arguments)]
fn write_fact(
    arm: Arm,
    gate: f32,
    store: &mut PkmEpisodicStore<SQRT_N, D_K, D_V>,
    rng: &mut Rng,
    q: &[f32; D_K],
    target: &[f32; D_V],
    stats: &BackgroundAccessStats<N_SLOTS>,
    tf_stats: &BackgroundAccessStats<N_SLOTS>,
    out: &mut [(usize, f32)],
    scratch: &mut PkmScratch<SQRT_N, K>,
) {
    match arm {
        Arm::Idf => {
            store.write_idf(q, target, gate, ScoreFn::Dot, POOL, T, stats, out, scratch);
        }
        Arm::Tf => {
            store.write_idf(
                q,
                target,
                gate,
                ScoreFn::Dot,
                POOL,
                T,
                tf_stats,
                out,
                scratch,
            );
        }
        Arm::Random => {
            let n = store
                .working()
                .query_into(q, ScoreFn::Dot, POOL, out, scratch);
            let mut picked = [(0usize, 0.0f32); T];
            for p in picked.iter_mut() {
                *p = out[rng.next_below(n)];
            }
            store.write_selected(&picked, target, gate);
        }
    }
}

/// One full A-then-B run under `arm` at `gate`. Returns
/// `(recall_a_pre, recall_a_post, recall_b)`.
fn run_arm(
    regime: Regime,
    arm: Arm,
    gate: f32,
    stats: &BackgroundAccessStats<N_SLOTS>,
) -> (f32, f32, f32) {
    let facts_a = gen_pairs(100, N_FACTS);
    let facts_b = gen_pairs(200, N_FACTS);

    let mut store = PkmEpisodicStore::new(build_table(regime));
    let mut out = [(0usize, 0.0f32); POOL];
    let mut scratch = PkmScratch::<SQRT_N, K>::new();
    let mut rng = Rng::new(777);

    // Degenerate stats for the TF arm: |B|=1, all counts zero → uniform idf.
    let mut tf_stats = BackgroundAccessStats::<N_SLOTS>::new();
    tf_stats.record_batch(&[]);

    for (q, target) in &facts_a {
        write_fact(
            arm,
            gate,
            &mut store,
            &mut rng,
            q,
            target,
            stats,
            &tf_stats,
            &mut out,
            &mut scratch,
        );
    }
    let recall_a_pre = recall(&store, &facts_a);

    for (q, target) in &facts_b {
        write_fact(
            arm,
            gate,
            &mut store,
            &mut rng,
            q,
            target,
            stats,
            &tf_stats,
            &mut out,
            &mut scratch,
        );
    }
    let recall_a_post = recall(&store, &facts_a);
    let recall_b = recall(&store, &facts_b);
    (recall_a_pre, recall_a_post, recall_b)
}

/// Fraction of `facts` whose target is recoverable from the fact's own
/// top-16 retrieved neighborhood (max-cos ≥ RECALL_COS).
fn recall(store: &PkmEpisodicStore<SQRT_N, D_K, D_V>, facts: &[Pair]) -> f32 {
    let mut out = [(0usize, 0.0f32); POOL];
    let mut scratch = PkmScratch::<SQRT_N, K>::new();
    let mut hits = 0usize;
    for (q, target) in facts {
        let n = store
            .working()
            .query_into(q, ScoreFn::Dot, POOL, &mut out, &mut scratch);
        let mut best = f32::NEG_INFINITY;
        for &(idx, _) in &out[..n] {
            let c = cos(store.working().value(idx), target);
            if c > best {
                best = c;
            }
        }
        if best >= RECALL_COS {
            hits += 1;
        }
    }
    hits as f32 / facts.len() as f32
}

fn regime_short(r: Regime) -> &'static str {
    match r {
        Regime::Organic => "organic",
        Regime::NormRamp => "ramp  ",
    }
}

// ── Main ────────────────────────────────────────────────────────────────────

fn main() {
    println!("═══ Issue 650 / Research 481 — TF-IDF Write-Gate GOAT ═══");
    println!();
    println!("Configuration:");
    println!("  PKM:      SQRT_N={SQRT_N} (N={N_SLOTS} slots), D_K={D_K}, D_V={D_V}, K={K}");
    println!(
        "  Write:    pool={POOL}, t={T} (smallest write-set regime — the paper's widest-gap regime)"
    );
    println!(
        "  Background: {} queries × batch {} → |B|={} batches",
        N_BG,
        BG_BATCH,
        N_BG / BG_BATCH
    );
    println!(
        "  Facts:    A={N_FACTS} then B={N_FACTS} (same broad distribution — overlapping pools)"
    );
    println!("  Recall:   max-cos over top-16 neighborhood ≥ {RECALL_COS:.2}");
    println!("  Gates:    {GATES:?}");
    println!("  NormRamp spread: {NORM_SPREAD:.1}");
    println!();

    // ── Build stats + sweep per regime; G1 gates on NormRamp ──────────────
    let regimes = [Regime::NormRamp, Regime::Organic];
    let arms = [Arm::Idf, Arm::Tf, Arm::Random];
    // results[regime][arm][gate] = (pre, post, rb).
    let mut results = [[[(0.0f32, 0.0f32, 0.0f32); GATES.len()]; 3]; 2];
    let mut regime_stats: [BackgroundAccessStats<N_SLOTS>; 2] =
        [BackgroundAccessStats::new(), BackgroundAccessStats::new()];

    for (ri, &regime) in regimes.iter().enumerate() {
        let bg_queries = gen_queries(300, N_BG);
        let table = build_table(regime);
        let mut bg_out = [(0usize, 0.0f32); POOL];
        let mut bg_scratch = PkmScratch::<SQRT_N, K>::new();
        let mut batch_slots = [0usize; BG_BATCH * POOL];
        let t0 = Instant::now();
        let stats = BackgroundAccessStats::<N_SLOTS>::build_background_stats(
            &table,
            &bg_queries,
            BG_BATCH,
            ScoreFn::Dot,
            POOL,
            &mut bg_out,
            &mut batch_slots,
            &mut bg_scratch,
        );
        let stats_build = t0.elapsed();
        drop(table);

        let mut hist = [0usize; 8];
        let mut idf_min = f32::INFINITY;
        let mut idf_max = f32::NEG_INFINITY;
        for i in 0..N_SLOTS {
            let c = stats.slot_batch_count(i);
            let bucket = ((c as f32 / (N_BG as f32 / BG_BATCH as f32) * 8.0) as usize).min(7);
            hist[bucket] += 1;
            idf_min = idf_min.min(stats.idf(i));
            idf_max = idf_max.max(stats.idf(i));
        }
        println!("── Regime: {} ──", regime.label());
        println!(
            "  stats built in {:?}: |B|={}, idf range [{:.3}, {:.3}]",
            stats_build,
            stats.n_batches(),
            idf_min,
            idf_max
        );
        println!("  slot count histogram (count/|B| buckets 0,⅛,…,1): {hist:?}");
        println!();

        for (ai, &arm) in arms.iter().enumerate() {
            for (gi, &gate) in GATES.iter().enumerate() {
                let (pre, post, rb) = run_arm(regime, arm, gate, &stats);
                results[ri][ai][gi] = (pre, post, rb);
                println!(
                    "  {} gate={:.2}:  recall(A)_pre={:.3}  recall(A)_post={:.3}  recall(B)={:.3}",
                    arm.label(),
                    gate,
                    pre,
                    post,
                    rb
                );
            }
            println!();
        }
        regime_stats[ri] = stats;
    }

    // ── G1 verdict per regime ─────────────────────────────────────────────
    println!("── G1 Interference/Retention Gate ───────────────────────────────────");
    println!(
        "  target: IDF recall(A)_post − TF recall(A)_post ≥ +{G1_MARGIN_TARGET:.2} at matched recall(B)"
    );
    let mut g1_verdicts = [false; 2];
    for (ri, &regime) in regimes.iter().enumerate() {
        // Matched-learning target: min(0.85, TF's best recall(B)).
        let tf_best_rb = results[ri][1]
            .iter()
            .map(|m| m.2)
            .fold(f32::NEG_INFINITY, f32::max);
        let target_rb = 0.85f32.min(tf_best_rb);

        let mut chosen: [(f32, (f32, f32, f32)); 3] = [(0.0, (0.0, 0.0, 0.0)); 3];
        for (ai, &arm) in arms.iter().enumerate() {
            let mut best_gi = 0usize;
            let mut best_dist = f32::INFINITY;
            for (gi, m) in results[ri][ai].iter().enumerate() {
                let dist = (m.2 - target_rb).abs();
                if dist < best_dist {
                    best_dist = dist;
                    best_gi = gi;
                }
            }
            chosen[ai] = (GATES[best_gi], results[ri][ai][best_gi]);
            println!(
                "  [{}] {} @ gate={:.2}: recall(B)={:.3} (target {:.3}), recall(A)_pre={:.3}, recall(A)_post={:.3}",
                regime_short(regime),
                arm.label(),
                chosen[ai].0,
                chosen[ai].1.2,
                target_rb,
                chosen[ai].1.0,
                chosen[ai].1.1
            );
        }
        let (_, idf_m) = chosen[0];
        let (_, tf_m) = chosen[1];
        let (rand_gate, rand_m) = chosen[2];
        let rb_gap = (idf_m.2 - tf_m.2).abs();
        let margin = idf_m.1 - tf_m.1;
        // G1 proper gates on NormRamp (the pathology-present regime).
        let pass = margin >= G1_MARGIN_TARGET && rb_gap <= 0.02;
        g1_verdicts[ri] = pass;
        println!(
            "  [{}] IDF recall(A)={:.3} vs TF recall(A)={:.3} → margin {:+.3}; recall(B) gap {:.3}; Random control recall(A)={:.3} @ gate={:.2}",
            regime_short(regime),
            idf_m.1,
            tf_m.1,
            margin,
            rb_gap,
            rand_m.1,
            rand_gate
        );
        if ri == 0 {
            println!(
                "  [{}] G1 verdict: {} (this is the gated regime)",
                regime_short(regime),
                if pass { "✅ PASS" } else { "❌ FAIL" }
            );
        } else {
            println!(
                "  [{}] no-harm control: margin {:+.3} (informational — IDF should not damage recall when the pathology is absent)",
                regime_short(regime),
                margin
            );
        }
        println!();
    }
    let g1_pass = g1_verdicts[0];
    // Recompute the ramp margin for the final verdict line.
    let ramp_margin = {
        let tf_best_rb = results[0][1]
            .iter()
            .map(|m| m.2)
            .fold(f32::NEG_INFINITY, f32::max);
        let target_rb = 0.85f32.min(tf_best_rb);
        let pick = |ai: usize| {
            let mut best_gi = 0usize;
            let mut best_dist = f32::INFINITY;
            for (gi, m) in results[0][ai].iter().enumerate() {
                let dist = (m.2 - target_rb).abs();
                if dist < best_dist {
                    best_dist = dist;
                    best_gi = gi;
                }
            }
            results[0][ai][best_gi]
        };
        pick(0).1 - pick(1).1
    };

    // ── G2: per-write latency (ramp regime — the gated regime) ───────────
    println!("── G2 Latency Gate ──────────────────────────────────────────────────");
    let stats = &regime_stats[0];
    let facts = gen_pairs(100, 1000);
    let mut store = PkmEpisodicStore::new(build_table(Regime::NormRamp));
    let mut out = [(0usize, 0.0f32); POOL];
    let mut scratch = PkmScratch::<SQRT_N, K>::new();

    // Warm up both paths.
    for (q, target) in facts.iter().take(64) {
        store.write_idf(
            q,
            target,
            0.8,
            ScoreFn::Dot,
            POOL,
            T,
            stats,
            &mut out,
            &mut scratch,
        );
        store.write(q, target, 0.8, ScoreFn::Dot, T, &mut out, &mut scratch);
    }

    let mut t_idf = std::time::Duration::ZERO;
    for (q, target) in black_box(facts.iter().take(1000)) {
        let t0 = Instant::now();
        black_box(store.write_idf(
            q,
            target,
            0.8,
            ScoreFn::Dot,
            POOL,
            T,
            stats,
            &mut out,
            &mut scratch,
        ));
        t_idf += t0.elapsed();
    }
    let ns_idf = t_idf.as_nanos() as f64 / 1000.0;

    let mut t_plain = std::time::Duration::ZERO;
    for (q, target) in black_box(facts.iter().take(1000)) {
        let t0 = Instant::now();
        black_box(store.write(q, target, 0.8, ScoreFn::Dot, T, &mut out, &mut scratch));
        t_plain += t0.elapsed();
    }
    let ns_plain = t_plain.as_nanos() as f64 / 1000.0;

    let overhead = ns_idf - ns_plain;
    println!("  write_idf (pool={POOL}, t={T}): {ns_idf:>8.0} ns/write");
    println!("  write     (k={T} plain):   {ns_plain:>8.0} ns/write");
    println!(
        "  idf-fold overhead:        {:>8.0} ns/write ({:.1}× plain)",
        overhead,
        ns_idf / ns_plain.max(1e-9)
    );
    println!(
        "  G2 verdict: {} (µs-scale; informational — O(k) multiplies + one top-t selection)",
        if overhead < 5000.0 {
            "✅ PASS"
        } else {
            "❌ FAIL"
        }
    );
    println!();

    // ── G4: alloc-free steady state ───────────────────────────────────────
    println!("── G4 Alloc-Free Gate ───────────────────────────────────────────────");
    let (_, allocs) = alloc_delta(|| {
        for (q, target) in facts.iter().take(1000) {
            black_box(store.write_idf(
                q,
                target,
                0.8,
                ScoreFn::Dot,
                POOL,
                T,
                stats,
                &mut out,
                &mut scratch,
            ));
        }
    });
    println!("  allocations across 1000 steady-state write_idf calls: {allocs}");
    let g4_pass = allocs == 0;
    println!(
        "  G4 verdict: {}",
        if g4_pass { "✅ PASS" } else { "❌ FAIL" }
    );
    println!();

    // ── Final ─────────────────────────────────────────────────────────────
    let pass = g1_pass && g4_pass;
    if pass {
        println!(
            "═══ Issue 650 GOAT: ✅ PASS — G1 (norm-ramp) margin {ramp_margin:+.3}, G4 {allocs} allocs ═══"
        );
    } else {
        println!(
            "═══ Issue 650 GOAT: ❌ FAIL — G1 {} (norm-ramp margin {:+.3}), G4 {} allocs ═══",
            if g1_pass { "PASS" } else { "FAIL" },
            ramp_margin,
            allocs
        );
        std::process::exit(1);
    }
}
