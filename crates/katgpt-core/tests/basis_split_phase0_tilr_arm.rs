#![cfg(feature = "subspace_phase_gate")]
//! Plan 625 T0.3 — the TILR-first arm (the substrate comparison row).
//!
//! Runs [`katgpt_core::tilr::discover_invariant_subspace`] on planted-block
//! difference fixtures (Research 618 §1.3 recipe: pairs share the invariant
//! block exactly, differ on the variant block with full-rank variation),
//! takes `I − VVᵀ` as the invariant-block estimate, and measures:
//!
//! - **G1** planted-block recovery by principal angles: the cosines of the
//!   principal angles between the true invariant span and `I − VVᵀ` are the
//!   singular values of `(I − VᵀV)·V_true` (orthogonal-projector identity);
//!   recovery = every cosine `> cos(1e-3)`.
//! - **Rank** selection honesty: TILR's τ-cumulative-variance rule must land
//!   the variant rank (48 of 64 here) at τ=0.99 — the heuristic that the
//!   `basis_split` eigengap gate replaces.
//! - **Adversarial behavior** (gap evidence, not gates): shuffled and
//!   degenerate inputs produce *some* split with no refusal — TILR has no
//!   refuse-loud identifiability verdict. Only all-zero differences error.
//! - **G4** allocation census of the discover call (TILR is the f32 +
//!   allocating accumulator; the count is the measurement that justifies the
//!   f64/zero-alloc path — recorded in the plan, asserted nonzero).

/// Deterministic LCG — the unseeded-global-RNG gate bans `rand::thread_rng`;
/// every fixture here is seeded and reproducible.
struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
        // Numerical Recipes LCG constants; top 24 bits to [0, 1).
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
    }
    /// Standard-normal via Box–Muller (cached second draw).
    fn next_gauss(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-12);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

const D: usize = 64;
const K: usize = 16; // planted invariant dims: Δ[0..K] = 0 exactly
const N: usize = 2048;

fn planted_differences() -> Vec<Vec<f32>> {
    let mut rng = Lcg(0x6255_6255_6255_6255);
    let mut diffs = Vec::with_capacity(N);
    for _ in 0..N {
        let mut delta = vec![0.0f32; D];
        for cell in delta[K..].iter_mut() {
            *cell = rng.next_gauss();
        }
        diffs.push(delta);
    }
    diffs
}

fn principal_angle_cosines(v_rows_rxd: &[f32], r: usize, target: &[f32], k: usize) -> Vec<f32> {
    // P = I − VᵀV  (d×d orthogonal projector onto the complement).
    let mut proj = vec![0.0f32; D * D];
    for i in 0..D {
        for j in 0..D {
            let mut dot = 0.0f32;
            for row in 0..r {
                dot += v_rows_rxd[row * D + i] * v_rows_rxd[row * D + j];
            }
            proj[i * D + j] = if i == j { 1.0 - dot } else { -dot };
        }
    }
    // PV_true (d×k), then its singular values = the principal-angle cosines
    // between range(V_true) and the complement (min(k, d−r) = k angles).
    let mut pv = vec![0.0f32; D * k];
    for i in 0..D {
        for j in 0..k {
            let mut acc = 0.0f32;
            for t in 0..D {
                acc += proj[i * D + t] * target[t * k + j];
            }
            pv[i * k + j] = acc;
        }
    }
    let mut work = katgpt_core::subspace_phase_gate::SvdScratch::with_capacity(k, D);
    let mut result = katgpt_core::subspace_phase_gate::SvdResultScratch::with_capacity(D, k);
    katgpt_core::subspace_phase_gate::thin_svd_into(&pv, D, k, &mut result, &mut work);
    result.singular_values().to_vec()
}

// ── the counting allocator (integration-test-local; the lib's TrackingAllocator
// ── is lib-internal and does not apply to this binary) ─────────────────────────

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static ALLOC_TRACKING: AtomicUsize = AtomicUsize::new(0);

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if ALLOC_TRACKING.load(Ordering::Relaxed) == 1 {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn counted<T>(f: impl FnOnce() -> T) -> (T, usize) {
    ALLOC_COUNT.store(0, Ordering::Relaxed);
    ALLOC_TRACKING.store(1, Ordering::Relaxed);
    let out = f();
    ALLOC_TRACKING.store(0, Ordering::Relaxed);
    (out, ALLOC_COUNT.load(Ordering::Relaxed))
}

// ── G1: planted-block recovery ─────────────────────────────────────────────────

#[test]
fn tilr_arm_recovers_planted_invariant_block_by_principal_angles() {
    let diffs = planted_differences();
    let refs: Vec<&[f32]> = diffs.iter().map(|d| d.as_slice()).collect();

    let (basis, r) = katgpt_core::tilr::discover_invariant_subspace(&refs, 0.99)
        .expect("planted fixture must discover");
    assert_eq!(
        r,
        D - K,
        "τ=0.99 must select the full variant rank 48, got {r}"
    );
    assert_eq!(basis.len(), r * D);

    // True invariant basis: the first K coordinate axes (d×k flattened
    // row-major — element [t, j] lives at t*K + j).
    let mut v_true = vec![0.0f32; D * K];
    for j in 0..K {
        v_true[j * K + j] = 1.0;
    }
    let cosines = principal_angle_cosines(&basis, r, &v_true, K);
    let min_cos = cosines.iter().cloned().fold(f32::INFINITY, f32::min);
    assert!(
        min_cos > (1e-3f32).cos(),
        "G1 FAIL: worst principal-angle cosine {min_cos} (angle {} rad)",
        min_cos.acos()
    );

    // The complement must also EXCLUDE the variant span: every basis row is
    // a variant-span direction, so its overlap with each true invariant axis
    // must be ~0 (direct dot products — the σ-cosine identity needs the
    // target dim ≤ complement dim, which 48 > 16 violates).
    for row_idx in 0..r {
        let row = &basis[row_idx * D..(row_idx + 1) * D];
        for (t, axis_val) in row.iter().take(K).enumerate() {
            assert!(
                axis_val.abs() < 1e-3,
                "complement leaks variant span: basis row {row_idx} has |<{t}| = {axis_val}"
            );
        }
    }

    println!(
        "tilr_arm G1 PASS: min principal-angle cosine {min_cos:.6?} (angle {:.3e} rad), r={r}",
        min_cos.acos()
    );
}

// ── rank honesty + adversarial gap evidence ────────────────────────────────────

#[test]
fn tilr_arm_shuffled_pairs_still_split_no_refusal() {
    // Shuffled pairs under whitening give M ≈ 2I (no eigengap) — exactly what
    // the basis_split gap gate must catch. TILR has no gate: it must still
    // return a split (the recorded gap evidence), only the rank wobbles.
    let mut diffs = planted_differences();
    // Deterministic shuffle: reverse the row order (a derangement of the pair
    // correspondence — pairing Δ_i with a mismatched partner set).
    let mut rng = Lcg(0x5100_0000_6255_0001);
    // Row permutation does not change the SET of difference vectors, so the
    // moment (and the split) is identical — the adversarial input for TILR is
    // instead SHUFFLED *values within rows* (destroys the block structure
    // while keeping per-row norms).
    for delta in diffs.iter_mut() {
        for _ in 0..delta.len() {
            let i = (rng.next_f32() * delta.len() as f32) as usize % delta.len();
            let j = (rng.next_f32() * delta.len() as f32) as usize % delta.len();
            delta.swap(i, j);
        }
    }
    let refs: Vec<&[f32]> = diffs.iter().map(|d| d.as_slice()).collect();
    let (_basis, _r) = katgpt_core::tilr::discover_invariant_subspace(&refs, 0.99)
        .expect("shuffled values still discover (no refusal — the gap)");
    // No assertion beyond "it returned" — the absence of an Unidentified
    // verdict IS the recorded finding.
}

#[test]
fn tilr_arm_degenerate_pairs_still_split_no_refusal() {
    // All-identical nonzero differences: rank-1 moment, no invariant block to
    // speak of (full-rank variation is violated) — yet TILR returns r=1.
    let delta = planted_differences().pop().expect("nonempty");
    let refs: Vec<&[f32]> = vec![delta.as_slice(); N];
    let (basis, r) = katgpt_core::tilr::discover_invariant_subspace(&refs, 0.99)
        .expect("degenerate nonzero input still discovers (no refusal — the gap)");
    assert_eq!(r, 1);
    assert_eq!(basis.len(), D);
}

#[test]
fn tilr_arm_zero_differences_error_is_the_only_refusal() {
    let zero = vec![0.0f32; D];
    let refs: Vec<&[f32]> = vec![zero.as_slice(); 8];
    assert!(katgpt_core::tilr::discover_invariant_subspace(&refs, 0.99).is_err());
}

// ── G4: the allocation census (recorded in the plan, asserted nonzero) ─────────

#[test]
fn tilr_arm_allocation_census_recorded() {
    let diffs = planted_differences();
    let refs: Vec<&[f32]> = diffs.iter().map(|d| d.as_slice()).collect();
    let ((), count) = counted(|| {
        let (_basis, _r) =
            katgpt_core::tilr::discover_invariant_subspace(&refs, 0.99).expect("discovers");
    });
    println!("tilr_arm G4 census: {count} allocations per discover call (N={N}, d={D})");
    assert!(
        count > 0,
        "TILR is expected to allocate (the recorded gap); 0 would mean the census is broken"
    );
}
