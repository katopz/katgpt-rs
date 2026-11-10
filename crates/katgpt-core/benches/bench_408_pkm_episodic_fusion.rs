//! Product Key Memory — δ-rule write gate fusion bench (Plan 408 Phase 5).
//!
//! Measures the **G4 fusion gate** that decides whether PKM-scaled δ-rule
//! writes beat the rank-r `DeltaMemoryState` substrate at equal write budget:
//!
//! - **Task**: synthetic associative recall. Generate `N_PAIRS` random
//!   `(q, target)` pairs (Gaussian, L2-normalized for the δ-Mem path). Store
//!   all pairs into both a `PkmEpisodicStore` and a `DeltaMemoryState`. Then
//!   query each `q`, measure reconstruction MSE vs `target`.
//! - **PKM**: `SQRT_N=32` (N=1024 slots), `D_K=8`, `D_V=4`. Retrieval top-`k=4`.
//!   `gate=0.5` (EMA-style consolidation). Two variants measured: `write`
//!   (unweighted — literal Plan 408 T5.1) and `write_weighted` (scales the
//!   per-slot update by the softmax retrieval weight).
//! - **δ-Mem**: `rank=4` (matches `D_V`). 4×4=16 state params.
//! - **Target**: PKM MSE ≤ 0.5 × δ-Mem MSE (≥2× lower reconstruction MSE).
//!
//! # Why PKM should win
//!
//! PKM has `SQRT_N² × D_V = 4096` value params vs δ-Mem's `rank² = 16` state
//! params — a 256× capacity advantage. The question the gate answers: does the
//! √N retrieval factorization lose enough information to throw away that
//! capacity advantage, or does PKM's sparse-write locality preserve it?
//!
//! # Run
//!
//! ```bash
//! CARGO_TARGET_DIR=/tmp/pkm_phase5 cargo bench -p katgpt-core \
//!   --features product_key_memory_episodic --bench bench_408_pkm_episodic_fusion -- --nocapture
//! ```
//!
//! Or, working around the intermittent macOS dyld/trustd launch stall:
//!
//! ```bash
//! CARGO_TARGET_DIR=/tmp/pkm_phase5 target/release/deps/bench_408_pkm_episodic_fusion-* --nocapture
//! ```

use katgpt_core::delta_mem::{DeltaMemoryConfig, DeltaMemoryState};
use katgpt_core::product_key_memory::{PkmEpisodicStore, PkmScratch, ProductKeyMemory, ScoreFn};
use std::hint::black_box;
use std::time::Instant;

#[path = "../tests/common/mod.rs"]
mod common;
counting_allocator!();

// ── Constants ───────────────────────────────────────────────────────────────

/// PKM table dimensions. SQRT_N=32 → N=1024 slots (4× the pair count, so the
/// table has headroom for distinct slot assignment). D_K=8 (halves are 4-dim,
/// enough for √N=32 codebook rows to discriminate). D_V=4 (matches δ-Mem rank).
const SQRT_N: usize = 32;
const D_K: usize = 8;
const D_V: usize = 4;
/// Per-codebook top-k for PKM retrieval. Final k can be up to K*K=16.
const K: usize = 4;

/// δ-Mem rank — set to D_V so both memories produce 4-dim outputs.
const DM_RANK: usize = D_V;

/// Number of (q, target) pairs to store + recall.
const N_PAIRS: usize = 200;

/// δ-rule gate for PKM writes. 0.5 = EMA-style consolidation (each slot moves
/// 50% toward target on first write, 25% on a second write to the same slot,
/// etc.). High enough to consolidate in one pass, low enough to avoid
/// last-write-wins collapse when two queries share a top-k slot.
const GATE: f32 = 0.5;

/// G4 fusion gate target: PKM MSE / δ-Mem MSE ≤ 0.5 (≥2× lower).
const G4_MSE_RATIO_TARGET: f64 = 0.5;

// ── Deterministic splitmix64 PRNG (mirrors bench_408_pkm_goat) ─────────────

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
    /// Standard-normal-ish sample via Box-Muller (deterministic, no deps).
    fn next_gauss(&mut self) -> f32 {
        // Two uniforms → one normal. We use the first; discard the second.
        let u1 = (self.next_u64() >> 40) as f32 / ((1u32 << 24) as f32);
        let u2 = (self.next_u64() >> 40) as f32 / ((1u32 << 24) as f32);
        let r = (-2.0f32 * (u1.max(1e-12)).ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        r * theta.cos()
    }
    /// Fill a fixed-size array with Gaussian samples.
    fn fill_gauss<const N: usize>(&mut self, out: &mut [f32; N]) {
        for x in out.iter_mut() {
            *x = self.next_gauss();
        }
    }
}

/// L2-normalize a vector in place. Required by `DeltaMemoryState::write` (and
/// applied to both PKM and δ-Mem inputs for fairness).
fn l2_normalize(v: &mut [f32]) {
    let mut norm_sq = 0.0f32;
    for &x in v.iter() {
        norm_sq += x * x;
    }
    let norm = norm_sq.sqrt().max(1e-12);
    for x in v.iter_mut() {
        *x /= norm;
    }
}

// ── MSE helpers ────────────────────────────────────────────────────────────

/// Mean squared error between two equal-length slices: mean Σ (a_i - b_i)².
fn mse(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut sum = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = x - y;
        sum += d * d;
    }
    sum / a.len().max(1) as f32
}

// ── PKM recall path ────────────────────────────────────────────────────────

/// Build a PKM table with random keys (for retrieval) but ZERO values (fair
/// vs δ-Mem's zero-init state — both start as "empty memory").
///
/// Random values would give PKM an unfair disadvantage: after a gate=0.5
/// write, `V = 0.5*random + 0.5*target` retains residual random variance that
/// δ-Mem (zero-init) doesn't have. Zero values isolate the retrieval-quality
/// variable.
fn pkm_table_zero_values(seed: u64) -> ProductKeyMemory<SQRT_N, D_K, D_V> {
    let mut table = ProductKeyMemory::<SQRT_N, D_K, D_V>::from_random(seed);
    for v in table.values.iter_mut() {
        *v = 0.0;
    }
    table
}

/// Run the PKM episodic recall task. Returns the mean per-query recall MSE.
///
/// Both variants start from a zero-values table (fair vs δ-Mem's zero state),
/// store all `N_PAIRS` pairs (normalized targets), then recall.
///
/// `top_k` controls how many slots each write touches. k=1 minimizes
/// collisions (each query writes to a single slot); k=4 provides redundancy
/// but increases inter-query interference.
fn pkm_recall(pairs: &[([f32; D_K], [f32; D_V])], weighted: bool, top_k: usize) -> f32 {
    let table = pkm_table_zero_values(2024);
    let mut store = PkmEpisodicStore::new(table);
    let mut scratch = PkmScratch::<SQRT_N, K>::new();
    let mut out = [(0usize, 0.0f32); K];

    // Normalized target buffer (δ-Mem requires normalized values; we apply
    // the same normalization to PKM targets for an apples-to-apples recall).
    let mut norm_target = [0.0f32; D_V];

    // Phase 1: store all pairs with normalized targets.
    for (q, target) in pairs {
        norm_target.copy_from_slice(target);
        l2_normalize(&mut norm_target);
        if weighted {
            store.write_weighted(
                q,
                &norm_target,
                GATE,
                ScoreFn::Dot,
                top_k,
                &mut out,
                &mut scratch,
            );
        } else {
            store.write(
                q,
                &norm_target,
                GATE,
                ScoreFn::Dot,
                top_k,
                &mut out,
                &mut scratch,
            );
        }
    }

    // Phase 2: recall each q, retrieve top-1 value, compute MSE vs normalized target.
    let mut total_mse = 0.0f32;
    let mut count = 0usize;
    for (q, target) in pairs {
        norm_target.copy_from_slice(target);
        l2_normalize(&mut norm_target);
        let n = store
            .working()
            .query_into(q, ScoreFn::Dot, 1, &mut out, &mut scratch);
        if n == 0 {
            continue;
        }
        let top1_idx = out[0].0;
        let recalled = store.working().value(top1_idx);
        total_mse += mse(recalled, &norm_target);
        count += 1;
    }
    if count == 0 {
        return f32::INFINITY;
    }
    total_mse / count as f32
}

// ── δ-Mem recall path ──────────────────────────────────────────────────────

/// Run the δ-Mem recall task. Returns the mean per-query MSE.
///
/// δ-Mem's `read` returns a `D_V`-dim vector (its rank). We feed the normalized
/// `D_K`-dim query projected down to `D_V` dims by truncation (the first `D_V`
/// components of the normalized query). This is a deterministic projection —
/// the fairest input for a rank-r associative memory that cannot see the full
/// `D_K`-dim query.
fn delta_mem_recall(pairs: &[([f32; D_K], [f32; D_V])]) -> f32 {
    let mut dm = DeltaMemoryState::new(DeltaMemoryConfig {
        rank: DM_RANK,
        ..Default::default()
    });

    // Phase 1: store all pairs.
    // Project D_K-dim query to D_V-dim key via truncation + L2-normalize.
    // δ-Mem's write requires normalized key AND value.
    let mut key_buf = [0.0f32; D_V];
    let mut val_buf = [0.0f32; D_V];
    for (q, target) in pairs {
        key_buf.copy_from_slice(&q[..D_V]);
        l2_normalize(&mut key_buf);
        val_buf.copy_from_slice(target);
        l2_normalize(&mut val_buf);
        dm.write(&key_buf, &val_buf);
    }

    // Phase 2: recall.
    let mut out_buf = vec![0.0f32; DM_RANK];
    let mut total_mse = 0.0f32;
    let mut count = 0usize;
    for (q, target) in pairs {
        key_buf.copy_from_slice(&q[..D_V]);
        l2_normalize(&mut key_buf);
        dm.read_into(&key_buf, &mut out_buf);
        // Compare the δ-Mem output (rank-dim) to the normalized target.
        val_buf.copy_from_slice(target);
        l2_normalize(&mut val_buf);
        total_mse += mse(&out_buf, &val_buf);
        count += 1;
    }
    if count == 0 {
        return f32::INFINITY;
    }
    total_mse / count as f32
}

// ── Main ───────────────────────────────────────────────────────────────────

fn main() {
    println!("═══ Plan 408 Phase 5 — PKM × δ-Mem Fusion Gate (G4) ═══");
    println!();
    println!("Configuration:");
    println!(
        "  PKM:   SQRT_N={} (N={} slots), D_K={}, D_V={}, K={}, gate={}, k∈{{1,4}}",
        SQRT_N,
        SQRT_N * SQRT_N,
        D_K,
        D_V,
        K,
        GATE
    );
    println!(
        "  δ-Mem: rank={} ({} state params)",
        DM_RANK,
        DM_RANK * DM_RANK
    );
    println!("  Pairs: {N_PAIRS} (equal write budget for both)");
    println!();

    // ── Generate pairs ──────────────────────────────────────────────────────
    // Queries are L2-normalized (required by δ-Mem, applied to PKM for fairness).
    // Targets are raw Gaussian — both PKM and δ-Mem normalize them on write
    // and recall against the normalized target (apples-to-apples).
    let mut rng = Rng::new(42);
    let mut pairs: Vec<([f32; D_K], [f32; D_V])> = Vec::with_capacity(N_PAIRS);
    for _ in 0..N_PAIRS {
        let mut q = [0.0f32; D_K];
        rng.fill_gauss(&mut q);
        l2_normalize(&mut q);
        let mut target = [0.0f32; D_V];
        rng.fill_gauss(&mut target);
        pairs.push((q, target));
    }
    // black_box the pair generation so the compiler doesn't elide it.
    let pairs = black_box(pairs);

    // ── PKM k=4 unweighted ──────────────────────────────────────────────────
    let t0 = Instant::now();
    let pkm_mse_uw_k4 = pkm_recall(&pairs, false, 4);
    let t_pkm_uw_k4 = t0.elapsed();
    println!("PKM (unweighted write, k=4):");
    println!("  recall MSE = {pkm_mse_uw_k4:.6}");
    println!("  wall time  = {t_pkm_uw_k4:?}");
    println!();

    // ── PKM k=4 weighted ─────────────────────────────────────────────────────
    let t0 = Instant::now();
    let pkm_mse_w_k4 = pkm_recall(&pairs, true, 4);
    let t_pkm_w_k4 = t0.elapsed();
    println!("PKM (weighted write, k=4):");
    println!("  recall MSE = {pkm_mse_w_k4:.6}");
    println!("  wall time  = {t_pkm_w_k4:?}");
    println!();

    // ── PKM k=1 unweighted (minimal collision) ──────────────────────────────
    let t0 = Instant::now();
    let pkm_mse_uw_k1 = pkm_recall(&pairs, false, 1);
    let t_pkm_uw_k1 = t0.elapsed();
    println!("PKM (unweighted write, k=1 — minimal collision):");
    println!("  recall MSE = {pkm_mse_uw_k1:.6}");
    println!("  wall time  = {t_pkm_uw_k1:?}");
    println!();

    // ── δ-Mem ───────────────────────────────────────────────────────────────
    let t0 = Instant::now();
    let dm_mse = delta_mem_recall(&pairs);
    let t_dm = t0.elapsed();
    println!("δ-Mem (rank={DM_RANK}):");
    println!("  recall MSE = {dm_mse:.6}");
    println!("  wall time  = {t_dm:?}");
    println!();

    // ── G4 verdict ───────────────────────────────────────────────────────────
    println!("── G4 Fusion Gate ────────────────────────────────────────────────────");
    println!(
        "  target:  PKM MSE / δ-Mem MSE ≤ {}  (≥{:.1}× lower)",
        G4_MSE_RATIO_TARGET,
        1.0 / G4_MSE_RATIO_TARGET
    );
    let ratios = [
        ("unweighted k=4", pkm_mse_uw_k4),
        ("weighted   k=4", pkm_mse_w_k4),
        ("unweighted k=1", pkm_mse_uw_k1),
    ];
    let mut best_ratio = f64::INFINITY;
    let mut best_label = "";
    for (label, pkm_mse) in &ratios {
        let ratio = *pkm_mse as f64 / dm_mse.max(1e-12) as f64;
        let verdict = if ratio <= G4_MSE_RATIO_TARGET {
            "✅ PASS"
        } else {
            "❌ FAIL"
        };
        println!("  {label}:  MSE={pkm_mse:.6}  ratio={ratio:.4}  →  {verdict}");
        if ratio < best_ratio {
            best_ratio = ratio;
            best_label = label;
        }
    }
    println!();

    // ── Alloc check (informational — Phase 5 is NOT zero-alloc-gated) ──────
    // Unlike the Phase 3 G4 (zero-alloc retrieval), Phase 5 has no zero-alloc
    // requirement (the write path legitimately allocates via query_into's
    // softmax + the δ-Mem comparison allocates Vecs). We report the alloc
    // count for transparency but do not gate on it.
    let alloc_before = ALLOC_COUNT.load(std::sync::atomic::Ordering::Relaxed);
    let (_r, alloc_during) = alloc_delta(|| pkm_recall(&pairs, false, 4));
    let _ = alloc_before;
    println!("  (informational) PKM unweighted k=4 write+recall allocs: {alloc_during}");
    println!();

    // ── Final verdict ──────────────────────────────────────────────────────────
    let pass = best_ratio <= G4_MSE_RATIO_TARGET;
    if pass {
        println!(
            "═══ G4: ✅ PASS — best variant '{best_label}' ratio={best_ratio:.4} ≤ {G4_MSE_RATIO_TARGET} ═══"
        );
        // Exit 0 (cargo bench success).
    } else {
        println!(
            "═══ G4: ❌ FAIL — best variant '{best_label}' ratio={best_ratio:.4} > {G4_MSE_RATIO_TARGET} ═══"
        );
        std::process::exit(1);
    }
}
