//! SIMD lattice-edge utility op (Plan 335 Phase 5).
//!
//! Per-edge scalar utility for game-lattice traversal — "how much does this
//! NPC want to traverse each edge?" Computed as a bridge from the eggshell
//! cochain lanes (raw `&[f32]`) and the NPC's HLA affective state
//! (latent → raw bridge via [`HlaToCohainWeights`]) into a single utility
//! scalar per edge.
//!
//! # Leaf-clean raw-slice API
//!
//! The plan (T5.1) originally specified a `view: &ValidatedZoneView,
//! hla: &HlaState` signature. That was a **layering bug**: `ValidatedZoneView`
//! lives in `riir-ai` (Phase 3), which depends on `katgpt-core` — not the other
//! way around. `katgpt-core` is the leaf and cannot reference types from repos
//! above it (circular dep), and `HlaState` is not a real type in the tree.
//!
//! The signature here is **leaf-clean**: it takes raw `&[f32]` lanes and raw
//! `&[u32]` index buffers. The typed `ValidatedZoneView` → raw-slice bridging
//! happens at the `riir-ai` call site (Phase 3/4), per the AGENTS.md bridge
//! pattern (raw physics stays raw; semantic projection is a local bridge that
//! only emits scalar coefficients across the sync boundary). This mirrors how
//! the existing DEC operators (`exterior_derivative`, `hodge_laplacian`) take
//! `&CochainField` (a katgpt-core type), not game-runtime types.
//!
//! # Sigmoid polarity (critical)
//!
//! Utility uses the **standard** sigmoid polarity `σ(x) = 1 / (1 + e^{-x})`
//! (high raw → high utility). This is the **opposite** polarity from
//! `dec::terrain_cochains::sigmoid`, which uses `1 / (1 + e^{+x})` so that
//! high danger → low safety. We therefore do **not** reuse the terrain sigmoid;
//! instead we route through [`crate::simd::simd_sigmoid_inplace`], whose
//! scalar fallback is [`crate::simd::fast_sigmoid`] (`1 / (1 + e^{-x})`).
//!
//! # SIMD strategy
//!
//! The per-edge computation is a **gather** pattern: for each edge we read
//! `interest[edge_src_vertex_idx[edge]]`, `safety[...]`, `destruction[...]`
//! (scattered vertex lookups) plus `occupancy[edge_face_idx[edge]]` (scattered
//! face lookup) and the contiguous `threat[edge]`. Explicit SIMD intrinsics
//! over a gather are non-portable and hard to keep correct across
//! x86_64/aarch64/wasm32. Instead:
//!
//! 1. The raw-utility accumulation loop is written as a branch-free scalar
//!    loop with hoisted broadcast coefficients. LLVM lowers the contiguous
//!    `threat[edge]` and `out_edge_utility[edge]` accesses to vector
//!    load/store and the scattered vertex/face reads to vector gather
//!    instructions on targets that have them (AVX2 `vgatherdps`, NEON
//!    `tbl`-based emulation).
//! 2. The sigmoid is then applied in one vectorized contiguous pass via the
//!    existing [`crate::simd::simd_sigmoid_inplace`], which relies on LLVM
//!    auto-vectorization (NEON / AVX2 / wasm32 simd128 / scalar).
//!
//! This keeps the leaf portable (wasm32 + native) and reuses the
//! already-validated sigmoid kernels instead of forking a new one.

use crate::simd::simd_sigmoid_inplace;

// ---------------------------------------------------------------------------
// HlaToCohainWeights (T5.2)
// ---------------------------------------------------------------------------

/// Bridge weights mapping HLA affective slots to cochain utility coefficients.
///
/// **Latent → raw bridge** (per AGENTS.md §Latent vs raw space rules). The HLA
/// state lives in latent space (per-NPC affective embedding); this struct
/// projects the four relevant affective axes onto the four scalar coefficients
/// the leaf-level utility op consumes. The projection is computed at the
/// `riir-ai` call site and only the resulting four scalars cross the sync
/// boundary — never the full HLA embedding.
///
/// Slot → cochain mapping grounded in **R144 Functional Emotions**:
/// - `curiosity_w`   ← valence lane      → interest multiplier (approach fame)
/// - `calm_w`        ← calm lane         → safety·occupancy multiplier (seek safety)
/// - `fear_w`        ← fear lane         → threat multiplier (avoid threat)
/// - `desperation_w` ← desperation lane  → destruction tolerance (risk-tolerance)
///
/// These weights are **deterministic**, not trained — consistent with the
/// modelless-first mandate. They are produced by a sigmoid projection of the
/// NPC's HLA affective scalars at the call site (latent → raw bridge), never
/// by gradient descent.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct HlaToCohainWeights {
    /// valence lane     → interest multiplier (curiosity for fame/notability).
    pub curiosity_w: f32,
    /// calm lane        → safety·occupancy multiplier (safety-seeking).
    pub calm_w: f32,
    /// fear lane        → threat multiplier (threat-avoidance).
    pub fear_w: f32,
    /// desperation lane → destruction tolerance (risk-tolerance under pressure).
    pub desperation_w: f32,
}

impl Default for HlaToCohainWeights {
    /// Deterministic neutral defaults (not trained).
    ///
    /// Curiosity and calm slightly higher than desperation — a baseline NPC
    /// mildly favors exploration and safety over risk-taking. Fear matches
    /// curiosity/calm so the default threat-avoidance is balanced against
    /// approach drive. Sum = 1.0 so the four coefficients form a normalized
    /// weighting when all lanes are O(1).
    #[inline]
    fn default() -> Self {
        Self {
            curiosity_w: 0.3,
            calm_w: 0.3,
            fear_w: 0.3,
            desperation_w: 0.1,
        }
    }
}

impl HlaToCohainWeights {
    /// All-zero weights. Useful for tests: with zero weights the raw utility
    /// accumulator is identically zero, so `sigmoid(0) = 0.5` for every edge.
    #[inline]
    pub const fn zero() -> Self {
        Self {
            curiosity_w: 0.0,
            calm_w: 0.0,
            fear_w: 0.0,
            desperation_w: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// lattice_edge_utility_into (T5.1, T5.3)
// ---------------------------------------------------------------------------

/// Compute per-edge traversal utility, writing one `f32` per edge into
/// `out_edge_utility`.
///
/// For each edge `e` with source vertex `src = edge_src_vertex_idx[e]` and
/// adjacent face `face = edge_face_idx[e]`:
///
/// ```text
/// raw_utility = interest[src] · curiosity_w
///             + safety[src]  · occupancy[face] · calm_w
///             - threat[e]    · fear_w
///             + destruction[src] · desperation_w
///
/// utility     = 1 / (1 + e^{-raw_utility})     // standard polarity
/// ```
///
/// High raw → high utility (NPC wants to traverse). Desperate NPCs
/// (`desperation_w` high) tolerate destroyed terrain; fearful NPCs
/// (`fear_w` high) avoid threat-heavy edges; calm NPCs favor safe + occupied
/// edges; curious NPCs favor interesting (fame/notability) edges.
///
/// # Arguments
///
/// | Lane                  | Cochain rank | Indexed by          |
/// |-----------------------|--------------|---------------------|
/// | `interest_lane`       | rank-0       | vertex (`src`)      |
/// | `safety_lane`         | rank-0       | vertex (`src`)      |
/// | `occupancy_lane`      | rank-2       | face                |
/// | `threat_lane`         | rank-1       | edge                |
/// | `destruction_lane`    | rank-0       | vertex (`src`)      |
/// | `edge_src_vertex_idx` | —            | edge → src vertex   |
/// | `edge_face_idx`       | —            | edge → adjacent face|
///
/// The two index buffers must have equal length; that length defines the edge
/// count `n_edges`. `out_edge_utility` must have length `>= n_edges`. Only the
/// first `n_edges` entries of `out_edge_utility` are written (and read back
/// for the sigmoid pass).
///
/// # Bounds
///
/// `debug_assert!` validates `out_edge_utility.len() >= edge_src_vertex_idx.len()`
/// and `edge_src_vertex_idx.len() == edge_face_idx.len()`. Index values inside
/// `edge_src_vertex_idx` / `edge_face_idx` are bounds-checked against the lane
/// slices via `debug_assert!` only — the hot loop uses `get_unchecked` to stay
/// branch-free. Run with debug assertions during integration testing; in
/// release the caller is responsible for valid indices (the `riir-ai`
/// `ValidatedZoneView` upstream guarantees this).
///
/// # Zero allocation
///
/// Zero-alloc by construction: the function only reads input slices and writes
/// into the caller-provided `out_edge_utility` slice. No `Vec`, `Box`, or
/// other heap allocation appears in the body. (Per AGENTS.md hot-loop rules.)
///
/// # SIMD
///
/// See the module docs: the raw-utility loop relies on LLVM
/// auto-vectorization of the chunked f32 loop (gather on targets that support
/// it); the sigmoid pass reuses [`crate::simd::simd_sigmoid_inplace`] (scalar loop,
/// auto-vectorized by LLVM where available). Explicit SIMD intrinsics are
/// avoided in
/// the accumulation loop to keep the leaf portable.
#[inline]
pub fn lattice_edge_utility_into(
    interest_lane: &[f32],
    safety_lane: &[f32],
    occupancy_lane: &[f32],
    threat_lane: &[f32],
    destruction_lane: &[f32],
    edge_src_vertex_idx: &[u32],
    edge_face_idx: &[u32],
    hla_weights: &HlaToCohainWeights,
    out_edge_utility: &mut [f32],
) {
    let n_edges = edge_src_vertex_idx.len();
    debug_assert_eq!(
        edge_face_idx.len(),
        n_edges,
        "lattice_edge_utility_into: edge_src_vertex_idx and edge_face_idx must have equal length"
    );
    debug_assert!(
        out_edge_utility.len() >= n_edges,
        "lattice_edge_utility_into: out_edge_utility.len() ({}) < n_edges ({})",
        out_edge_utility.len(),
        n_edges
    );
    if n_edges == 0 {
        return;
    }

    // Bounds-check index buffers against lanes (debug only). The hot loop
    // below uses get_unchecked, so a malformed index is UB in release — the
    // upstream ValidatedZoneView (riir-ai) is the source of truth in prod.
    #[cfg(debug_assertions)]
    {
        for e in 0..n_edges {
            let s = edge_src_vertex_idx[e] as usize;
            let f = edge_face_idx[e] as usize;
            debug_assert!(
                s < interest_lane.len() && s < safety_lane.len() && s < destruction_lane.len(),
                "lattice_edge_utility_into: edge {} src vertex idx {} out of lane range",
                e,
                s
            );
            debug_assert!(
                f < occupancy_lane.len(),
                "lattice_edge_utility_into: edge {} face idx {} out of occupancy lane range",
                e,
                f
            );
            debug_assert!(
                e < threat_lane.len(),
                "lattice_edge_utility_into: edge {} out of threat lane range (len {})",
                e,
                threat_lane.len()
            );
        }
    }

    // Hoist broadcast coefficients out of the loop so the inner loop body is
    // a branch-free f32 FMA chain with scalar loads/gathers. LLVM lowers the
    // contiguous threat[e] / out[e] pair to vector load/store and the
    // scattered vertex/face loads to gather instructions where available.
    let curiosity_w = hla_weights.curiosity_w;
    let calm_w = hla_weights.calm_w;
    let fear_w = hla_weights.fear_w;
    let desperation_w = hla_weights.desperation_w;

    for e in 0..n_edges {
        // SAFETY: indices are bounds-checked above in debug builds, and in
        // release the upstream ValidatedZoneView guarantees validity.
        // n_edges <= out_edge_utility.len() is asserted above.
        unsafe {
            let src = *edge_src_vertex_idx.get_unchecked(e) as usize;
            let face = *edge_face_idx.get_unchecked(e) as usize;

            let interest = *interest_lane.get_unchecked(src);
            let safety = *safety_lane.get_unchecked(src);
            let destruction = *destruction_lane.get_unchecked(src);
            let occupancy = *occupancy_lane.get_unchecked(face);
            let threat = *threat_lane.get_unchecked(e);

            // raw_utility = interest·curiosity + safety·occupancy·calm
            //              - threat·fear + destruction·desperation
            //
            // Written as a straightforward f32 arithmetic chain so LLVM can
            // emit fused-multiply-add (FMA) on targets that have it.
            let raw = interest * curiosity_w + safety * occupancy * calm_w - threat * fear_w
                + destruction * desperation_w;

            *out_edge_utility.get_unchecked_mut(e) = raw;
        }
    }

    // Apply standard-polarity sigmoid (1/(1+e^{-x})) in one vectorized pass.
    // simd_sigmoid_inplace has NEON / AVX2 / wasm32 simd128 / scalar paths.
    simd_sigmoid_inplace(&mut out_edge_utility[..n_edges]);
}

// ---------------------------------------------------------------------------
// Tests (T5.4)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// σ(0) = 0.5 exactly in f32 (1/(1+1)).
    const HALF: f32 = 0.5;
    /// High positive raw → utility saturates near 1.
    const NEAR_ONE: f32 = 0.999;
    /// High negative raw → utility saturates near 0.
    const NEAR_ZERO: f32 = 0.001;

    /// Build a trivial 2-edge lattice: both edges share vertex 0 / face 0.
    /// Lanes are vertex/edge/face arrays of length 2.
    struct TwoEdgeLattice {
        interest: Vec<f32>,
        safety: Vec<f32>,
        occupancy: Vec<f32>,
        threat: Vec<f32>,
        destruction: Vec<f32>,
        src_idx: Vec<u32>,
        face_idx: Vec<u32>,
    }

    impl TwoEdgeLattice {
        fn new() -> Self {
            Self {
                interest: vec![0.0, 0.0],
                safety: vec![0.0, 0.0],
                occupancy: vec![0.0, 0.0],
                threat: vec![0.0, 0.0],
                destruction: vec![0.0, 0.0],
                src_idx: vec![0, 0],
                face_idx: vec![0, 0],
            }
        }

        fn run(&self, weights: &HlaToCohainWeights) -> [f32; 2] {
            let mut out = [0.0f32; 2];
            lattice_edge_utility_into(
                &self.interest,
                &self.safety,
                &self.occupancy,
                &self.threat,
                &self.destruction,
                &self.src_idx,
                &self.face_idx,
                weights,
                &mut out,
            );
            out
        }
    }

    #[test]
    fn test_zero_weights_give_half_utility() {
        // All weights zero → raw accumulates to 0 → sigmoid(0) = 0.5.
        let lat = TwoEdgeLattice::new();
        let out = lat.run(&HlaToCohainWeights::zero());
        for u in out.iter() {
            assert!((u - HALF).abs() < 1e-6, "expected sigmoid(0)=0.5, got {u}");
        }
    }

    #[test]
    fn test_high_fear_zeros_threat_edges() {
        // fear_w=1.0, others 0, threat high → -threat·1.0 is large negative →
        // sigmoid → near 0.
        let mut lat = TwoEdgeLattice::new();
        lat.threat = vec![10.0, 20.0]; // large threat
        let weights = HlaToCohainWeights {
            fear_w: 1.0,
            curiosity_w: 0.0,
            calm_w: 0.0,
            desperation_w: 0.0,
        };
        let out = lat.run(&weights);
        for (e, u) in out.iter().enumerate() {
            assert!(
                *u < NEAR_ZERO,
                "edge {}: expected threat-suppressed utility < {}, got {}",
                e,
                NEAR_ZERO,
                u
            );
        }
    }

    #[test]
    fn test_high_desperation_tolerates_destruction() {
        // desperation_w=1.0, destruction high, others 0 → +destruction·1.0
        // large positive → sigmoid → near 1.
        let mut lat = TwoEdgeLattice::new();
        lat.destruction = vec![10.0, 20.0];
        let weights = HlaToCohainWeights {
            desperation_w: 1.0,
            curiosity_w: 0.0,
            calm_w: 0.0,
            fear_w: 0.0,
        };
        let out = lat.run(&weights);
        for (e, u) in out.iter().enumerate() {
            assert!(
                *u > NEAR_ONE,
                "edge {}: expected destruction-tolerant utility > {}, got {}",
                e,
                NEAR_ONE,
                u
            );
        }
    }

    #[test]
    fn test_high_curiosity_amplifies_interest() {
        // curiosity_w=1.0, interest high, others 0 → +interest·1.0 large
        // positive → sigmoid → near 1.
        let mut lat = TwoEdgeLattice::new();
        lat.interest = vec![10.0, 20.0];
        let weights = HlaToCohainWeights {
            curiosity_w: 1.0,
            calm_w: 0.0,
            fear_w: 0.0,
            desperation_w: 0.0,
        };
        let out = lat.run(&weights);
        for (e, u) in out.iter().enumerate() {
            assert!(
                *u > NEAR_ONE,
                "edge {}: expected curiosity-amplified utility > {}, got {}",
                e,
                NEAR_ONE,
                u
            );
        }
    }

    #[test]
    fn test_high_calm_amplifies_safe_occupied() {
        // calm_w=1.0, safety·occupancy high, others 0 → +safety·occupancy·1.0
        // large positive → sigmoid → near 1.
        let mut lat = TwoEdgeLattice::new();
        lat.safety = vec![1.0, 1.0];
        lat.occupancy = vec![10.0, 20.0];
        let weights = HlaToCohainWeights {
            calm_w: 1.0,
            curiosity_w: 0.0,
            fear_w: 0.0,
            desperation_w: 0.0,
        };
        let out = lat.run(&weights);
        for (e, u) in out.iter().enumerate() {
            assert!(
                *u > NEAR_ONE,
                "edge {}: expected calm-amplified utility > {}, got {}",
                e,
                NEAR_ONE,
                u
            );
        }
    }

    #[test]
    fn test_output_length_matches_edge_count() {
        // n_edges = 2; out_edge_utility must receive exactly 2 written values,
        // and a longer out buffer must have only the first 2 touched.
        let mut lat = TwoEdgeLattice::new();
        lat.interest = vec![5.0, 5.0];
        let weights = HlaToCohainWeights {
            curiosity_w: 1.0,
            calm_w: 0.0,
            fear_w: 0.0,
            desperation_w: 0.0,
        };
        // Sentinel-fill a longer buffer; entries beyond n_edges must survive.
        let mut out = [-1.0f32; 5];
        lattice_edge_utility_into(
            &lat.interest,
            &lat.safety,
            &lat.occupancy,
            &lat.threat,
            &lat.destruction,
            &lat.src_idx,
            &lat.face_idx,
            &weights,
            &mut out,
        );
        // First 2 written (both interest=5, curiosity=1 → σ(5)≈0.9933).
        assert!(
            out[0] > 0.99 && out[1] > 0.99,
            "first two entries should be written, got {:?}",
            &out[..2]
        );
        // Entries beyond n_edges untouched.
        for v in &out[2..] {
            assert_eq!(
                *v, -1.0,
                "entries beyond n_edges must be untouched, got {v}"
            );
        }
    }

    #[test]
    fn test_default_weights_are_normalized() {
        // Sanity: Default sums to 1.0 (documented invariant).
        let w = HlaToCohainWeights::default();
        let sum = w.curiosity_w + w.calm_w + w.fear_w + w.desperation_w;
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "default weights should sum to 1.0, got {sum}"
        );
    }

    #[test]
    #[should_panic(expected = "edge_face_idx")]
    fn test_bounds_check_debug_asserts_mismatched_index_lengths() {
        // debug_assert_eq! on edge_src_vertex_idx.len() == edge_face_idx.len()
        // panics in debug builds. Under release this test would not panic on
        // the debug_assert, so it is only meaningful in debug; that matches the
        // DoD "test_bounds_check_debug_asserts".
        let lat = TwoEdgeLattice::new();
        let mismatched_face_idx: [u32; 1] = [0];
        let mut out = [0.0f32; 2];
        lattice_edge_utility_into(
            &lat.interest,
            &lat.safety,
            &lat.occupancy,
            &lat.threat,
            &lat.destruction,
            &lat.src_idx,         // len 2
            &mismatched_face_idx, // len 1 — mismatch
            &HlaToCohainWeights::default(),
            &mut out,
        );
    }

    #[test]
    fn test_empty_edge_set_is_noop() {
        // n_edges = 0 → returns immediately, no panic, no writes.
        let mut out = [99.0f32; 4];
        lattice_edge_utility_into(
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &HlaToCohainWeights::default(),
            &mut out,
        );
        for v in &out {
            assert_eq!(*v, 99.0, "empty edge set must not touch out buffer");
        }
    }

    #[test]
    fn test_polarity_is_standard() {
        // Standard polarity: positive raw → utility > 0.5; negative raw → < 0.5.
        // curiosity_w=1.0, interest=+5 → σ(5) > 0.5.
        // curiosity_w=1.0, interest=-5 → σ(-5) < 0.5.
        let mut lat_pos = TwoEdgeLattice::new();
        lat_pos.interest = vec![5.0, 5.0];
        let weights = HlaToCohainWeights {
            curiosity_w: 1.0,
            ..HlaToCohainWeights::zero()
        };
        let out_pos = lat_pos.run(&weights);
        for u in &out_pos {
            assert!(*u > HALF, "positive raw should give utility > 0.5, got {u}");
        }

        let mut lat_neg = TwoEdgeLattice::new();
        lat_neg.interest = vec![-5.0, -5.0];
        let out_neg = lat_neg.run(&weights);
        for u in &out_neg {
            assert!(*u < HALF, "negative raw should give utility < 0.5, got {u}");
        }
    }

    #[test]
    fn test_terms_combine_linearly_in_raw() {
        // Two opposing terms cancel: curiosity_w=1 interest=+5, fear_w=1 threat=5
        // → raw = 5 - 5 = 0 → σ(0)=0.5.
        let mut lat = TwoEdgeLattice::new();
        lat.interest = vec![5.0, 5.0];
        lat.threat = vec![5.0, 5.0];
        let weights = HlaToCohainWeights {
            curiosity_w: 1.0,
            fear_w: 1.0,
            calm_w: 0.0,
            desperation_w: 0.0,
        };
        let out = lat.run(&weights);
        for u in &out {
            assert!(
                (u - HALF).abs() < 1e-6,
                "interest(+5)·1 - threat(5)·1 = 0 → σ(0)=0.5, got {u}"
            );
        }
    }
}
