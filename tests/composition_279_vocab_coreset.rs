//! Plan 279 T2.5 — Composition test: Manifold Power Iteration Router × Vocab Coreset.
//!
//! Validates that **MPI router conditioning** (Plan 279) and **vocab_coreset
//! top-p selection** (Plan 181) compose cleanly — the two systems deliver
//! their respective gains **additively, without blocking each other**
//! (research note §2.5 Fusion B).
//!
//! # Scope
//!
//! This test validates **composition correctness**, not MPI quality. MPI's
//! own quality claims (λ alignment gain, MaxVio reduction, etc.) are locked
//! in by `tests/bench_279_manifold_power_iter_goat.rs`. Here we only verify
//! that the two systems interoperate without contract violations.
//!
//! # Orthogonality claim
//!
//! The two systems operate on **different axes**:
//!
//! | Feature       | Acts on            | Knob             | Effect                       |
//! |---------------|--------------------|------------------|------------------------------|
//! | MPI router    | router row `R[i]`  | `iters`, `C'`    | per-expert score *direction* |
//! | vocab_coreset | score distribution | `p` (top-p mass) | coreset *size* adaptivity    |
//!
//! `vocab_coreset` is **score-shape-agnostic**: it works on any non-negative
//! score distribution, whether MPI-conditioned or not. MPI changes *which*
//! experts score highly; `vocab_coreset` decides *how many* to keep. The two
//! never alias memory, never share state.
//!
//! # What this test proves
//!
//! 1. **G1 — Real interaction:** MPI changes the score distribution (if it
//!    didn't, there'd be no composition to test). Locks in that the
//!    composition is non-trivial.
//! 2. **G2 — `vocab_coreset` contract preserved under MPI scores:** for both
//!    `R` and `R'`, coreset size is monotone non-decreasing in `p`,
//!    `p=1.0` selects all experts, and every coreset is non-empty and bounded.
//! 3. **G3 — Well-formed outputs:** no panics, no NaNs, all scores in
//!    sigmoid range `[0, 1]`.
//! 4. **G4 — Determinism:** byte-identical coreset across repeated runs with
//!    the same seed (sync/quorum-safe composition).
//! 5. **G5 — Sigmoid discipline:** every score is `σ(β·x·R[i]^T)`, never
//!    softmax (AGENTS.md constraint — independent per-expert).
//!
//! Run:
//! ```bash
//! cargo test --features "manifold_power_iter_router vocab_coreset" \
//!            --test composition_279_vocab_coreset -- --nocapture
//! ```

#![cfg(all(feature = "manifold_power_iter_router", feature = "vocab_coreset"))]

use katgpt_rs::manifold_power_iter_router::{
    compute_expert_gram_into, gate_sigmoid_topk, manifold_power_iter_router,
};
use katgpt_rs::speculative::vocab_coreset::vocab_coreset;
use katgpt_rs::spectral_retract::PowerRetractScratch;

// ── Deterministic PRNG (xorshift64, matches the GOAT test convention) ─────

fn seeded_vec(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed;
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        v.push(((s & 0xFFFF) as f32 / 0x8000 as f32) - 1.0);
    }
    v
}

fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

// ── Synthetic MoE fixture ─────────────────────────────────────────────────
//
// Per-expert gate weights W_g[i] are rank-1 with a KNOWN dominant right-
// singular vector u[i]. This gives MPI a clean signal to act on (so the
// composition is non-trivial) without making any quality claims that depend
// on `iters` count or starting router state.

const N_EXPERTS: usize = 8;
const D_MODEL: usize = 16; // small for fast tests; composition is dimension-independent
const K_CONTEXTS: usize = 4; // K+1 draft positions in dMoE terminology

/// Build a rank-1 D×D Gram whose dominant right-singular vector is `u`.
fn build_rank1_gram(sigma: f32, u: &[f32], d: usize) -> Vec<f32> {
    let un = norm(u);
    let scale = sigma / (un * un);
    let mut w = vec![0.0f32; d * d];
    for i in 0..d {
        for j in 0..d {
            w[i * d + j] = u[i] * u[j] * scale;
        }
    }
    let mut g = vec![0.0f32; d * d];
    compute_expert_gram_into(&w, d, &mut g);
    g
}

struct MoeFixture {
    /// Unconditioned router R (N × D, row-major). Random seed 42.
    router_r: Vec<f32>,
    /// Per-expert grams (each D × D, row-major).
    grams: Vec<Vec<f32>>,
    /// K input contexts (each length D).
    contexts: Vec<Vec<f32>>,
}

fn build_fixture() -> MoeFixture {
    let mut grams = Vec::with_capacity(N_EXPERTS);
    for i in 0..N_EXPERTS {
        let mut u = seeded_vec(1000 + i as u64, D_MODEL);
        let nu = norm(&u);
        for x in &mut u {
            *x /= nu;
        }
        let sigma = 3.0 + (i as f32) * 0.5;
        grams.push(build_rank1_gram(sigma, &u, D_MODEL));
    }

    let router_r = seeded_vec(42, N_EXPERTS * D_MODEL);
    let contexts = (0..K_CONTEXTS)
        .map(|k| seeded_vec(7_000 + k as u64, D_MODEL))
        .collect();

    MoeFixture {
        router_r,
        grams,
        contexts,
    }
}

/// Compute per-expert sigmoid scores `σ(β·x·R[i]^T)` for one context.
fn score_context(x: &[f32], r: &[f32], beta: f32, scores_buf: &mut [f32]) {
    gate_sigmoid_topk(x, r, N_EXPERTS, D_MODEL, beta, N_EXPERTS, scores_buf);
}

/// Per-context scores + vocab_coreset selection for one router variant.
struct CompositionResult {
    scores_per_ctx: Vec<Vec<f32>>,
    coreset_mask: Vec<bool>,
    coreset_size: usize,
}

fn run_composition(router: &[f32], contexts: &[Vec<f32>], p: f32) -> CompositionResult {
    let beta = 1.0;
    let mut scores_per_ctx = Vec::with_capacity(contexts.len());
    for x in contexts {
        let mut scores = vec![0.0f32; N_EXPERTS];
        score_context(x, router, beta, &mut scores);
        scores_per_ctx.push(scores);
    }

    let marginals: Vec<&[f32]> = scores_per_ctx.iter().map(|v| v.as_slice()).collect();
    let mut coreset_mask = vec![false; N_EXPERTS];
    let coreset_size = vocab_coreset(&marginals, p, &mut coreset_mask);

    CompositionResult {
        scores_per_ctx,
        coreset_mask,
        coreset_size,
    }
}

/// Construct the MPI-conditioned router R' from the fixture's R + grams.
fn build_r_prime(fixture: &MoeFixture) -> Vec<f32> {
    let grams_ref: Vec<&[f32]> = fixture.grams.iter().map(|g| g.as_slice()).collect();
    let mut r_prime = fixture.router_r.clone();
    let mut scratch = PowerRetractScratch::new(D_MODEL);
    manifold_power_iter_router(
        &mut r_prime,
        &grams_ref,
        N_EXPERTS,
        D_MODEL,
        1.0, // c_prime
        1,   // iters — paper default (G8 of the underlying GOAT gate)
        &mut scratch,
    );
    r_prime
}

// ── Tests ─────────────────────────────────────────────────────────────────

/// G1 — the composition is non-trivial: MPI changes the score distribution.
///
/// If MPI produced byte-identical scores to vanilla R, there would be no
/// composition to test. This locks in that the two systems actually interact.
///
/// We don't claim MPI *improves* scores (that's the GOAT gate's job) — only
/// that it *changes* them, making the composition meaningful.
#[test]
fn g1_mpi_changes_score_distribution() {
    let fixture = build_fixture();
    let r_prime = build_r_prime(&fixture);

    let res_r = run_composition(&fixture.router_r, &fixture.contexts, 0.9);
    let res_rp = run_composition(&r_prime, &fixture.contexts, 0.9);

    let mut any_diff = false;
    for (k, (sr, srp)) in res_r
        .scores_per_ctx
        .iter()
        .zip(res_rp.scores_per_ctx.iter())
        .enumerate()
    {
        for i in 0..N_EXPERTS {
            let delta = (sr[i] - srp[i]).abs();
            if delta > 1e-6 {
                any_diff = true;
                eprintln!(
                    "✓ context {k} expert {i}: R={:.6} → R'={:.6} (Δ={delta:.6})",
                    sr[i],
                    srp[i]
                );
                break;
            }
        }
        if any_diff {
            break;
        }
    }
    assert!(any_diff, "G1 FAIL: MPI produced identical scores — composition is trivial");
}

/// G2 + G3 — `vocab_coreset` contract is respected for both router variants,
/// and all outputs are well-formed (no NaNs, scores in sigmoid range).
#[test]
fn g2_g3_coreset_contract_and_well_formed_outputs() {
    let fixture = build_fixture();
    let r_prime = build_r_prime(&fixture);

    let ps = [0.5_f32, 0.7, 0.9, 1.0];

    for (label, router) in [("R", fixture.router_r.as_slice()), ("R'", r_prime.as_slice())] {
        // G3: well-formed scores for every context (no NaN, in [0,1]).
        let probe = run_composition(router, &fixture.contexts, 0.9);
        for (k, scores) in probe.scores_per_ctx.iter().enumerate() {
            for (i, &s) in scores.iter().enumerate() {
                assert!(
                    s.is_finite() && (0.0..=1.0).contains(&s),
                    "G3 FAIL [{label}] context {k} expert {i}: score out of range = {s}"
                );
            }
        }

        // G2: coreset contract — monotone in p, p=1.0 → full set, bounded.
        let mut sizes = Vec::with_capacity(ps.len());
        for &p in &ps {
            let res = run_composition(router, &fixture.contexts, p);
            assert!(
                res.coreset_size > 0 && res.coreset_size <= N_EXPERTS,
                "G2a FAIL [{label}] p={p}: coreset_size out of range = {}",
                res.coreset_size
            );
            sizes.push(res.coreset_size);
        }

        for w in sizes.windows(2) {
            assert!(
                w[1] >= w[0],
                "G2b FAIL [{label}]: coreset size must be monotone in p: got {sizes:?}"
            );
        }

        assert_eq!(
            sizes.last(),
            Some(&(N_EXPERTS as usize)),
            "G2c FAIL [{label}]: p=1.0 must select all experts, got {sizes:?}"
        );

        eprintln!("✓ G2+G3 [{label}] contract respected, sizes={sizes:?}");
    }
}

/// G4 — determinism: same seed → byte-identical coreset across repeated runs.
///
/// Composition determinism follows from each component being a pure function
/// of its inputs (G5 of the underlying MPI GOAT gate, and `vocab_coreset`'s
/// stable sort). This test locks in that the *composition* is also pure.
#[test]
fn g4_composition_deterministic_across_runs() {
    let fixture = build_fixture();

    let run_once = || -> (Vec<Vec<f32>>, Vec<bool>, usize) {
        let r_prime = build_r_prime(&fixture);
        let res = run_composition(&r_prime, &fixture.contexts, 0.85);
        (res.scores_per_ctx.clone(), res.coreset_mask.clone(), res.coreset_size)
    };

    let (scores_a, mask_a, size_a) = run_once();
    let (scores_b, mask_b, size_b) = run_once();

    assert_eq!(scores_a, scores_b, "G4 FAIL: scores differ across runs");
    assert_eq!(mask_a, mask_b, "G4 FAIL: coreset masks differ across runs");
    assert_eq!(size_a, size_b, "G4 FAIL: coreset sizes differ across runs");
    eprintln!("✓ G4 byte-identical composition across runs (size={size_a})");
}

/// G5 — sigmoid discipline: scores are independent per-expert sigmoids, never
/// softmax-normalized. Verified on both R and R'.
#[test]
fn g5_sigmoid_discipline_not_softmax() {
    let fixture = build_fixture();
    let r_prime = build_r_prime(&fixture);

    for (label, router) in [("R", fixture.router_r.as_slice()), ("R'", r_prime.as_slice())] {
        let mut scores = vec![0.0f32; N_EXPERTS];
        score_context(&fixture.contexts[0], router, 1.0, &mut scores);

        let sum: f32 = scores.iter().copied().sum();
        for (i, &s) in scores.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&s),
                "G5a FAIL [{label}]: score out of [0,1] at expert {i}: {s}"
            );
        }
        assert_ne!(
            sum, 1.0,
            "G5b FAIL [{label}]: scores sum to 1.0 (softmax); independent sigmoids don't"
        );
        eprintln!("✓ G5 [{label}] sigmoid scores sum={sum:.4} (not softmax)");
    }
}
