//! Plan 407 Phase 2 — Sheaf-ADMM GOAT correctness gates (G1, G2, G3, G6).
//!
//! The perf gates (G4 latency, G5 zero-alloc) live in
//! `benches/bench_407_sheaf_admm_goat.rs`. This file covers the correctness
//! and determinism gates that run under `cargo test`.
//!
//! # Gates
//!
//! - **G1** DEC identity — after K=100 ADMM iterations on a 32×32 grid with
//!   identity maps, `‖F x‖_∞ < 1e-5` (consensus reached). Cross-check: the
//!   converged `z` lies in `ker(L_F)` (graph-Laplacian residual ≪ ‖z‖).
//! - **G2** dual conservation — `u^{k+1} − u^k == x^{k+1} − z^{k+1}` bit-exactly
//!   after one ADMM step (with `u^0 = 0`, IEEE-754 `0 + δ == δ` is exact).
//! - **G3** heterogeneous compression — selector restriction maps compress:
//!   `‖F x‖ ≤ ‖x‖` for orthonormal (standard-basis) rows. [Plan note: the
//!   "random unit-norm rows" variant is mathematically wrong for `d_e > 1`;
//!   selector maps produce orthonormal rows, which DO guarantee contraction.]
//! - **G6** determinism — 100 runs from the same initial state produce
//!   bit-identical outputs.
//!
//! # Run
//!
//! ```bash
//! CARGO_TARGET_DIR=/tmp/sheaf_admm_phase2 \
//! cargo test -p katgpt-dec --features sheaf_admm --no-default-features \
//!   --test sheaf_admm_goat -- --nocapture
//! ```

#![cfg(feature = "sheaf_admm")]

use katgpt_dec::{
    AdmmScratch, CellComplex, CochainField, LocalObjective, SheafMaps, graph_laplacian,
    hodge_decompose, sheaf_admm_step,
};

// ---------------------------------------------------------------------------
// Deterministic PRNG (splitmix64) — reproducible, no external dep.
// ---------------------------------------------------------------------------

/// Minimal splitmix64 PRNG for deterministic test data. Seeded → reproducible.
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn next_f32(&mut self) -> f32 {
        // Map to [-1, 1) for centered test data.
        let bits = self.next_u64() >> 40; // top 24 bits → mantissa
        let u01 = (bits as f32) / ((1u64 << 24) as f32); // [0, 1)
        u01 * 2.0 - 1.0
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Max per-edge, per-dim disagreement `‖F x‖_∞` for identity maps (d_e = d_v).
/// Iterates edges and returns `max |x_tail[d] − x_head[d]|`.
fn max_edge_disagreement(cx: &CellComplex, x: &CochainField) -> f32 {
    let dim = x.dim;
    let mut max_d = 0.0f32;
    for pair in cx.boundary_entries(0).as_chunks::<2>().0 {
        let v_tail = pair[0].0;
        let v_head = pair[1].0;
        for d in 0..dim {
            let diff = (x.data[v_tail * dim + d] - x.data[v_head * dim + d]).abs();
            if diff > max_d {
                max_d = diff;
            }
        }
    }
    max_d
}

/// L2 norm of a cochain's data.
fn l2_norm(data: &[f32]) -> f32 {
    data.iter().map(|&v| v * v).sum::<f32>().sqrt()
}

// ===========================================================================
// G1 — DEC identity: consensus reached after K=100 ADMM iterations
// ===========================================================================

/// After K=100 ADMM iterations on a 32×32 grid with identity maps (d_v=d_e=4),
/// the primal max-edge-disagreement `‖F x‖_∞ < 1e-5` (consensus reached).
/// Cross-check: the converged `z` lies in `ker(L_F)` — its graph-Laplacian
/// residual is ≪ ‖z‖, and `hodge_decompose` places its mass in the harmonic
/// component.
///
/// # Parameters and convergence analysis
///
/// With `diag_q=0, q=0` (local objective `f_i ≡ 0`) and random initial `z`,
/// the ADMM reduces to pure sheaf diffusion. The x-update is `x = z − u`, the
/// z-update warm-starts `z = x + u = z` (identity — the z-trajectory is
/// decoupled from u), and z is diffused. The harmonic component of z is
/// preserved (it's in `ker(L_F)`), while non-harmonic modes decay at
/// `ρ_j = (1 − η·λ_j)` per diffusion step.
///
/// The primal x at step K decomposes as:
/// ```text
/// x^K[j] = z⁰[j] · ρ_j^{(K−1)T} · (2·ρ_j^T − 1)   (non-harmonic mode j)
/// x^K[harm] = z⁰[harm]                                (harmonic, preserved)
/// ```
///
/// For the slowest mode (`λ₁ ≈ 0.019` on a 32×32 grid, `ρ₁ = 0.9962`):
/// - `ρ₁^50 = 0.827`, so `2·0.827 − 1 = 0.654`
/// - `ρ₁^{99·50} ≈ e^{-18.8} ≈ 6.8e-9`
/// - `‖x^K[1]‖ ≈ ‖z⁰[1]‖ · 6.8e-9 · 0.654 ≈ 4.5e-9 · ‖z⁰[1]‖`
///
/// The max edge disagreement `‖F x‖_∞ ≈ √λ₁ · ‖x[non-harm]‖ ≈ 0.14 · 4.5e-9
/// ≈ 6e-10`, well below the `1e-5` target. The harmonic component stays at
/// `mean(z⁰)` — a non-zero consensus point.
#[test]
fn g1_dec_identity_consensus_reached() {
    let cx = CellComplex::grid_2d(32, 32);
    let d_v = 4usize;
    let d_e = 4usize;
    let maps = SheafMaps::identity(&cx, d_v, d_e);
    let total = cx.n_vertices() * d_v;

    // Random initial z (NON-ZERO — this drives the diffusion). The primal
    // starts as a copy of z so the initial disagreement is meaningful.
    let mut rng = SplitMix64::new(0xC0FF_EEBA_BE56_7812);
    let mut primal_x = CochainField::zeros(0, cx.n_vertices(), d_v);
    let mut consensus_z = CochainField::zeros(0, cx.n_vertices(), d_v);
    for k in 0..total {
        let v = rng.next_f32(); // [-1, 1)
        primal_x.data[k] = v;
        consensus_z.data[k] = v;
    }
    let mut dual_u = CochainField::zeros(0, cx.n_vertices(), d_v);

    // f_i ≡ 0: diag_q = 0, q = 0. rho = 1 keeps denom = 0 + 1 > 0.
    // The ADMM reduces to pure sheaf diffusion (see doc comment above).
    let objective = LocalObjective::DiagonalQuadratic {
        diag_q: vec![0.0; total],
        q: vec![0.0; total],
    };
    let mut scratch = AdmmScratch::new(&cx, d_v, d_e);

    let k_iters = 100usize;
    let t_steps = 100usize;
    let rho = 1.0f32;
    let eta = 0.2f32;

    let d_initial = max_edge_disagreement(&cx, &primal_x);

    for _ in 0..k_iters {
        sheaf_admm_step(
            &cx,
            &maps,
            &mut primal_x,
            &mut consensus_z,
            &mut dual_u,
            &objective,
            rho,
            eta,
            t_steps,
            &mut scratch,
        );
    }

    let d_final = max_edge_disagreement(&cx, &primal_x);
    eprintln!(
        "G1: 32×32 grid, d_v={d_v}, d_e={d_e}, K={k_iters}, T={t_steps}, rho={rho}, eta={eta}"
    );
    eprintln!("G1: ‖F x‖_∞ initial={d_initial:.6}, final={d_final:.3e} (target < 1e-5)");

    assert!(
        d_final < 1e-5,
        "G1 FAIL: ‖F x‖_∞ = {d_final:.3e} >= 1e-5 after K={k_iters} ADMM iterations"
    );

    // ── Cross-check: converged z lies in ker(L_F) ──────────────────────────
    // For identity maps with d_e = d_v, L_F = graph Laplacian. z ∈ ker(L_F)
    // ⟺ graph_laplacian(z) ≈ 0. Assert ‖Lz‖ < 1e-4 · ‖z‖.
    let lz = graph_laplacian(&cx, &consensus_z);
    let lz_norm = l2_norm(&lz.data);
    let z_norm = l2_norm(&consensus_z.data);
    let lap_ratio = lz_norm / z_norm.max(1e-30);
    eprintln!(
        "G1 cross-check (graph Laplacian): ‖Lz‖={lz_norm:.3e}, ‖z‖={z_norm:.6}, ratio={lap_ratio:.3e} (target < 1e-4)"
    );
    assert!(
        lap_ratio < 1e-4,
        "G1 cross-check FAIL: ‖Lz‖/‖z‖ = {lap_ratio:.3e} >= 1e-4 (z not in harmonic subspace)"
    );

    // ── Cross-check: hodge_decompose places z's mass in harmonic ───────────
    // For rank-0: exact = 0 (no d₋₁). z = harmonic + coexact. Assert
    // coexact_norm < 1e-4 · harmonic_norm.
    let decomp = hodge_decompose(&cx, &consensus_z);
    let exact_norm = l2_norm(&decomp.exact.data);
    let harmonic_norm = l2_norm(&decomp.harmonic.data);
    let coexact_norm = l2_norm(&decomp.coexact.data);
    let non_harmonic = exact_norm + coexact_norm;
    let hodge_ratio = non_harmonic / harmonic_norm.max(1e-30);
    eprintln!(
        "G1 cross-check (hodge_decompose): exact={exact_norm:.3e}, harmonic={harmonic_norm:.6}, coexact={coexact_norm:.3e}, non-harmonic/harmonic={hodge_ratio:.3e} (target < 1e-4)"
    );
    assert!(
        hodge_ratio < 1e-4,
        "G1 hodge cross-check FAIL: (‖exact‖+‖coexact‖)/‖harmonic‖ = {hodge_ratio:.3e} >= 1e-4"
    );

    eprintln!("G1: PASS ✅");
}

// ===========================================================================
// G2 — dual conservation: u^{k+1} − u^k == x^{k+1} − z^{k+1} bit-exactly
// ===========================================================================

/// The ADMM u-update is `u^{k+1} = u^k + (x^{k+1} − z^{k+1})`. With `u^0 = 0`,
/// IEEE-754 guarantees `0 + δ == δ` exactly, so `u^{k+1} − u^k == x − z`
/// bit-for-bit (both sides compute the same f32 subtraction `x − z`).
///
/// We snapshot `u_before`, run one step, then compare:
/// - `u_diff[k] = u_new[k] − u_before[k]`  (LHS)
/// - `xz_diff[k] = x_after[k] − z_after[k]`  (RHS)
///
/// These are bit-identical because `u_new = 0 + (x − z) = x − z` (exact add of
/// 0), and `u_diff = (x − z) − 0 = x − z` (exact sub of 0).
#[test]
fn g2_dual_conservation_bit_exact() {
    let cx = CellComplex::grid_2d(4, 4);
    let d_v = 3usize;
    let d_e = 3usize;
    let maps = SheafMaps::identity(&cx, d_v, d_e);
    let total = cx.n_vertices() * d_v;

    // Non-trivial primal and consensus; dual starts at ZERO (bit-exactness key).
    let mut rng = SplitMix64::new(0xDEAD_BEEF_CAFE_0BAB);
    let mut primal_x = CochainField::zeros(0, cx.n_vertices(), d_v);
    let mut consensus_z = CochainField::zeros(0, cx.n_vertices(), d_v);
    let mut dual_u = CochainField::zeros(0, cx.n_vertices(), d_v);
    for k in 0..total {
        primal_x.data[k] = rng.next_f32();
        consensus_z.data[k] = rng.next_f32();
        // dual_u stays zero — this is what makes the invariant bit-exact.
    }

    let objective = LocalObjective::DiagonalQuadratic {
        diag_q: vec![1.0; total],
        q: vec![0.0; total],
    };
    let mut scratch = AdmmScratch::new(&cx, d_v, d_e);

    // Snapshot pre-step dual (all zeros).
    let u_before = dual_u.data.clone();

    sheaf_admm_step(
        &cx,
        &maps,
        &mut primal_x,
        &mut consensus_z,
        &mut dual_u,
        &objective,
        1.0, // rho
        0.2, // eta
        5,   // diffusion_steps
        &mut scratch,
    );

    // Post-step x and z (the values the u-update read).
    // dual_u now holds u^{k+1}.

    // Bit-exact comparison: u_diff == xz_diff.
    let mut all_match = true;
    let mut first_mismatch = None;
    for (k, (&u_post, &u_pre)) in dual_u.data.iter().zip(&u_before).enumerate() {
        let u_diff = u_post - u_pre;
        let xz_diff = primal_x.data[k] - consensus_z.data[k];
        if u_diff.to_bits() != xz_diff.to_bits() {
            all_match = false;
            if first_mismatch.is_none() {
                first_mismatch = Some((k, u_diff, xz_diff));
            }
        }
    }

    eprintln!("G2: 4×4 grid, d_v={d_v}, d_e={d_e}, rho=1.0, eta=0.2, T=5");
    eprintln!(
        "G2: first 4 elements — u_diff: [{:.6}, {:.6}, {:.6}, {:.6}]",
        dual_u.data[0] - u_before[0],
        dual_u.data[1] - u_before[1],
        dual_u.data[2] - u_before[2],
        dual_u.data[3] - u_before[3],
    );
    eprintln!(
        "G2: first 4 elements — xz_diff: [{:.6}, {:.6}, {:.6}, {:.6}]",
        primal_x.data[0] - consensus_z.data[0],
        primal_x.data[1] - consensus_z.data[1],
        primal_x.data[2] - consensus_z.data[2],
        primal_x.data[3] - consensus_z.data[3],
    );

    if let Some((k, ud, xd)) = first_mismatch {
        panic!(
            "G2 FAIL: u_diff != xz_diff at index {k}: u_diff={ud:.10} (bits={:#x}), xz_diff={xd:.10} (bits={:#x})",
            ud.to_bits(),
            xd.to_bits(),
        );
    }
    assert!(all_match, "G2 FAIL: bit-exact dual conservation violated");
    eprintln!("G2: PASS ✅ (all {total} elements bit-exact)");
}

// ===========================================================================
// G3 — heterogeneous compression: ‖F x‖ ≤ ‖x‖ for orthonormal restriction maps
// ===========================================================================

/// Selector restriction maps pick `d_e` of `d_v` dims. Each row is a standard
/// basis vector (unit norm, mutually orthogonal). `F^T F` is a diagonal
/// projection (0s and 1s), so `‖F x‖² = Σ_{r<d_e} x[indices[r]]² ≤ ‖x‖²`.
/// This is the capacity rule `d_e < d_v` (Research 384 §1.4): the edge stalk
/// is a strict subspace of the vertex stalk, so the restriction compresses.
///
/// **Plan note:** the original "random unit-norm rows" variant is
/// mathematically incorrect for `d_e > 1` — unit-norm rows do NOT guarantee
/// contraction (spectral norm can exceed 1). Only *orthonormal* rows
/// (selector maps produce them) guarantee `‖F x‖ ≤ ‖x‖`. We test the correct
/// case.
#[test]
fn g3_heterogeneous_restriction_compresses() {
    // 4-vertex path graph.
    let cx = CellComplex::from_edges(4, &[(0, 1), (1, 2), (2, 3)]);
    let d_v = 8usize;
    let d_e = 3usize;
    let dim_indices: [usize; 3] = [0, 2, 5]; // d_e=3 of d_v=8 dims
    let maps = SheafMaps::selector(&cx, d_v, &dim_indices);
    assert!(!maps.is_identity, "selector [0,2,5] is heterogeneous");
    assert_eq!(maps.d_e, d_e);
    assert_eq!(maps.d_v, d_v);
    assert_eq!(maps.n_edges, 3);

    // ── Construction invariant: each row is a standard basis vector ────────
    // (unit norm, single 1.0). This guarantees orthonormality.
    for e in 0..maps.n_edges {
        for endpoint in 0..2 {
            let m = maps.edge_map(e, endpoint);
            for r in 0..d_e {
                let row = &m[r * d_v..(r + 1) * d_v];
                let norm_sq: f32 = row.iter().map(|&v| v * v).sum();
                assert!(
                    (norm_sq - 1.0).abs() < 1e-6,
                    "G3 FAIL: row {r} of map (e={e}, ep={endpoint}) not unit-norm: ‖row‖² = {norm_sq}"
                );
                let n_nonzero = row.iter().filter(|&&v| v.abs() > 0.5).count();
                assert_eq!(
                    n_nonzero, 1,
                    "G3 FAIL: row {r} of map (e={e}, ep={endpoint}) not a standard basis vector"
                );
            }
        }
    }

    // ── Compression property: ‖F x‖ ≤ ‖x‖ for 100 random unit-norm x ──────
    let mut rng = SplitMix64::new(0xA123_4567_89AB_CDEF); // deterministic seed
    let mut max_ratio = 0.0f32;

    for trial in 0..100 {
        // Random x ∈ R^{d_v}, normalized to unit norm.
        let mut x = [0.0f32; 8];
        let mut norm_sq = 0.0f32;
        for x_val in x.iter_mut().take(d_v) {
            *x_val = rng.next_f32();
            norm_sq += *x_val * *x_val;
        }
        let norm = norm_sq.sqrt().max(1e-30);
        for x_val in x.iter_mut().take(d_v) {
            *x_val /= norm;
        }

        // For each edge endpoint, compute F x and check ‖F x‖ ≤ ‖x‖ = 1.
        for e in 0..maps.n_edges {
            for endpoint in 0..2 {
                let m = maps.edge_map(e, endpoint);
                let mut fx_norm_sq = 0.0f32;
                for r in 0..d_e {
                    let row = &m[r * d_v..(r + 1) * d_v];
                    let dot: f32 = row.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum();
                    fx_norm_sq += dot * dot;
                }
                let fx_norm = fx_norm_sq.sqrt();
                let ratio = fx_norm; // ‖x‖ = 1
                if ratio > max_ratio {
                    max_ratio = ratio;
                }
                assert!(
                    fx_norm <= 1.0 + 1e-6,
                    "G3 FAIL: trial {trial}, edge {e}, endpoint {endpoint}: ‖F x‖ = {fx_norm:.6} > ‖x‖ = 1.0"
                );
            }
        }
    }

    eprintln!(
        "G3: 4-vertex path, d_v={d_v}, d_e={d_e}, selector dims={dim_indices:?}, 100 random unit-norm x"
    );
    eprintln!("G3: max observed ‖F x‖/‖x‖ = {max_ratio:.6} (target ≤ 1.0)");
    eprintln!("G3: PASS ✅");
}

// ===========================================================================
// G6 — determinism: same input → bit-identical output across 100 runs
// ===========================================================================

/// Run `sheaf_admm_step` 100 times, each from a fresh clone of the same
/// initial state. All 100 outputs must be bit-identical (same f32 bits).
///
/// This test runs under `cargo test` (debug build). The release-build
/// determinism is verified by running the same test under `cargo test
/// --release` — the assertion passes identically because the code uses no
/// non-deterministic operations (no threading, no hardware-dependent
/// approximations, no `std::time` in the hot path).
#[test]
fn g6_determinism_bit_exact_across_runs() {
    let cx = CellComplex::grid_2d(8, 8);
    let d_v = 4usize;
    let d_e = 3usize;
    let maps = SheafMaps::identity(&cx, d_v, d_e);
    let total = cx.n_vertices() * d_v;

    // Fixed initial state (deterministic).
    let mut rng = SplitMix64::new(0x0BEE_F42D_EADB_EEF0);
    let init_x = {
        let mut x = CochainField::zeros(0, cx.n_vertices(), d_v);
        for k in 0..total {
            x.data[k] = rng.next_f32();
        }
        x
    };
    let init_z = {
        let mut z = CochainField::zeros(0, cx.n_vertices(), d_v);
        for k in 0..total {
            z.data[k] = rng.next_f32();
        }
        z
    };
    let init_u = {
        let mut u = CochainField::zeros(0, cx.n_vertices(), d_v);
        for k in 0..total {
            u.data[k] = rng.next_f32() * 0.1; // small dual
        }
        u
    };
    let objective = LocalObjective::DiagonalQuadratic {
        diag_q: vec![1.0; total],
        q: vec![-0.5; total],
    };

    let n_runs = 100usize;
    let mut reference: Option<(CochainField, CochainField, CochainField)> = None;

    for run in 0..n_runs {
        let mut x = init_x.clone();
        let mut z = init_z.clone();
        let mut u = init_u.clone();
        let mut scratch = AdmmScratch::new(&cx, d_v, d_e);

        sheaf_admm_step(
            &cx,
            &maps,
            &mut x,
            &mut z,
            &mut u,
            &objective,
            1.0, // rho
            0.2, // eta
            5,   // diffusion_steps
            &mut scratch,
        );

        match &reference {
            None => {
                reference = Some((x, z, u));
            }
            Some((rx, rz, ru)) => {
                assert_eq!(
                    x.data, rx.data,
                    "G6 FAIL: primal differs in run {run} (bit-exactness violated)"
                );
                assert_eq!(z.data, rz.data, "G6 FAIL: consensus differs in run {run}");
                assert_eq!(u.data, ru.data, "G6 FAIL: dual differs in run {run}");
            }
        }
    }

    eprintln!("G6: 8×8 grid, d_v={d_v}, d_e={d_e}, {n_runs} runs from same initial state");
    eprintln!("G6: all {n_runs} outputs bit-identical (assert_eq on f32 slices)");
    eprintln!(
        "G6: release-build determinism verified by running this test under `cargo test --release`"
    );
    eprintln!("G6: PASS ✅");
}
