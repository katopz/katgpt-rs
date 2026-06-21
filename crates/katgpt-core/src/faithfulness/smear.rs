//! SmearClassifier — ternary classification of latent mass distribution.
//!
//! Plan 298, Research 277 (arXiv:2606.20560 — Engels et al., "How Transparent
//! is DiffusionGemma?", DeepMind, Jun 2026).
//!
//! Extends Plan 278's binary `FaithfulnessProbe` (faithful / unfaithful) with
//! a vocabulary for *how* latent mass is distributed across hypotheses/sites:
//!
//! - [`SmearClass::CoherentSingle`] — one dominant hypothesis at one site.
//!   The faithful single-hypothesis case.
//! - [`SmearClass::TokenSmear`] — high mass on one direction spread across
//!   `span` adjacent sites. Benign positional uncertainty (paper §5.2.1).
//!   Faithful.
//! - [`SmearClass::SequenceSmear`] — mass split across ≥2 semantically
//!   distinct directions at one site. Potentially unfaithful multi-hypothesis
//!   superposition (paper §5.2.2).
//!
//! **Phase 1 only** — standalone classifier. Phase 2 (Plan 298) will wire this
//! into `DefaultFaithfulnessProbe` as an optional diagnostic.
//!
//! # Determinism
//!
//! Same input → bit-identical [`SmearReport`]. No RNG, no hash-map iteration;
//! all loops walk slice indices `0..k` in fixed order. Safe for deterministic
//! replay and quorum commit of the `#[repr(u8)]` class byte.
//!
//! # Hot-path contract
//!
//! [`SmearClassifier::classify`] takes a caller-provided `&mut [f32]` scratch
//! buffer of length `k + k*(k-1)/2` (norms + pairwise cosines). Zero
//! allocation in the hot path — no `Vec`, no `Box`, no `format!`.

use crate::simd::simd_dot_f32;

/// Smear classification of a latent mass distribution (Research 277, Plan 298).
///
/// `#[repr(u8)]` per AGENTS.md — 1-byte sync-friendly output, safe to emit
/// alongside raw sync blocks without bloating wire format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SmearClass {
    /// Mass concentrated on a single direction at a single site.
    /// The "faithful single-hypothesis" case.
    CoherentSingle = 0,
    /// Mass on one direction spread across `span` adjacent sites.
    /// Benign positional uncertainty (paper §5.2.1). Faithful.
    TokenSmear = 1,
    /// Mass split across ≥2 semantically distinct directions at one site.
    /// Potentially unfaithful multi-hypothesis superposition (paper §5.2.2).
    SequenceSmear = 2,
}

/// Per-classification detail (diagnostics + GOAT gate evidence).
///
/// `PartialEq` (no `Eq` — `semantic_distance: f32`) so callers can assert
/// bit-identical reports for determinism checks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SmearReport {
    /// Ternary verdict.
    pub class: SmearClass,
    /// Number of distinct sites carrying significant mass (≥1 for TokenSmear,
    /// always 1 for SequenceSmear since it is by definition at one site).
    pub span: u8,
    /// Number of distinct semantic directions at the dominant site
    /// (≥2 for SequenceSmear, always 1 for TokenSmear).
    pub n_hypotheses: u8,
    /// Max pairwise cosine distance among the significant directions.
    /// Higher = more semantically distinct = more concerning.
    /// Range: `[0.0, 2.0]` (cosine distance can hit 2.0 for antipodal vectors).
    pub semantic_distance: f32,
}

/// Ternary smear classifier trait.
///
/// Classifies a `[k][d]` row-major slice of MUX superposition weights (Plan 178)
/// or BoM K-hypothesis beliefs (Plan 281) into one of three smear classes.
///
/// # Arguments
///
/// - `weights` — flat `[k * d]` slice, row-major (k rows of d elements).
/// - `k` — number of hypotheses/sites. **Capped at 16** per trait contract.
///   The pairwise cosine loop is `O(k²)`; for `k > 16`, subsample or use a
///   cheaper proxy (max-norm hypothesis). Implementations may debug-assert
///   this cap.
/// - `d` — dimensionality of each hypothesis/site.
/// - `scratch` — caller-allocated, length `k + k*(k-1)/2`.
///   - `scratch[0..k]` receives per-row L2 norms.
///   - `scratch[k..k+pairs]` receives pairwise cosines (survivor pairs only;
///     entries for filtered pairs are left untouched).
///   - Reused across calls; zero-allocation in the hot path.
///
/// # Determinism
///
/// Same `(weights, k, d)` → bit-identical [`SmearReport`]. Implementations
/// MUST NOT use RNG or iterate unordered collections.
pub trait SmearClassifier {
    /// Classify the given `[k][d]` row-major weight slice.
    fn classify(&self, weights: &[f32], k: usize, d: usize, scratch: &mut [f32]) -> SmearReport;
}

/// Default classifier: max pairwise cosine distance among significant
/// hypotheses, compared against `tau_same`.
///
/// Defaults: `epsilon = 1e-3`, `tau_same = 0.1` (near-parallel threshold).
pub struct CosineSmearClassifier {
    /// L2 norm threshold below which a hypothesis is treated as insignificant.
    pub epsilon: f32,
    /// Max pairwise cosine distance for TokenSmear. Pairs with distance `<=`
    /// `tau_same` are treated as positional variants of one direction.
    pub tau_same: f32,
}

impl Default for CosineSmearClassifier {
    #[inline]
    fn default() -> Self {
        Self {
            epsilon: 1e-3,
            tau_same: 0.1,
        }
    }
}

impl CosineSmearClassifier {
    /// Construct with explicit thresholds.
    #[inline]
    pub const fn new(epsilon: f32, tau_same: f32) -> Self {
        Self { epsilon, tau_same }
    }
}

impl SmearClassifier for CosineSmearClassifier {
    /// Decision logic (paper §5.2.1 vs §5.2.2, operationalized):
    ///
    /// 1. Compute norms `‖w_i‖` into `scratch[0..k]`.
    /// 2. Filter: drop rows with `norm < epsilon`. Count survivors `S`.
    /// 3. If `S <= 1` → [`SmearClass::CoherentSingle`].
    /// 4. Compute pairwise cosine `cos(w_i, w_j) = dot / (‖w_i‖·‖w_j‖)` for
    ///    survivor pairs into `scratch[k..]`. `denom >= epsilon² > 0` for
    ///    survivors — no div-by-zero possible. A paranoid `denom > 0.0`
    ///    guard is kept as a defence-in-depth.
    /// 5. `semantic_distance = max over survivor pairs of (1 - cosine)`.
    /// 6. If `semantic_distance <= tau_same` → [`SmearClass::TokenSmear`]
    ///    with `span = S`. Else → [`SmearClass::SequenceSmear`] with
    ///    `n_hypotheses = S`.
    ///
    /// All loops walk slice indices in fixed order — deterministic, safe for
    /// replay/quorum.
    #[inline]
    fn classify(&self, weights: &[f32], k: usize, d: usize, scratch: &mut [f32]) -> SmearReport {
        // Trait contract cap. O(k²) pairwise loop is uneconomical past this.
        debug_assert!(k <= 16, "SmearClassifier k cap is 16; got {k}; subsample");

        // Pairs count is k*(k-1)/2; saturating arithmetic avoids underflow at k=0.
        let pairs = k.saturating_mul(k.saturating_sub(1)) / 2;
        let scratch_needed = k + pairs;
        debug_assert!(
            scratch.len() >= scratch_needed,
            "scratch too small: need {scratch_needed} (k={k}, pairs={pairs}), got {}",
            scratch.len()
        );
        debug_assert!(
            weights.len() >= k * d,
            "weights too small: need {}, got {}",
            k * d,
            weights.len()
        );

        // Step 1: per-row L2 norms into scratch[0..k].
        // ‖w‖ = sqrt(w · w); simd_dot_f32 picks NEON/AVX2/scalar per target.
        for i in 0..k {
            let row = &weights[i * d..(i + 1) * d];
            let norm_sq = simd_dot_f32(row, row, d);
            scratch[i] = norm_sq.sqrt();
        }

        // Step 2: count survivors. Single pass, deterministic.
        let mut survivor_count: u8 = 0;
        for i in 0..k {
            if scratch[i] >= self.epsilon {
                survivor_count += 1;
            }
        }

        // Step 3: S ≤ 1 → CoherentSingle. Covers k=0 and single-dominant cases.
        if survivor_count <= 1 {
            return SmearReport {
                class: SmearClass::CoherentSingle,
                span: survivor_count,
                n_hypotheses: survivor_count,
                semantic_distance: 0.0,
            };
        }

        // Steps 4 & 5: pairwise cosines among survivors; track max distance.
        // denom = ‖w_i‖·‖w_j‖ ≥ epsilon² > 0 for survivor pairs, so cosine is
        // always finite. The explicit `denom > 0.0` guard is defence-in-depth.
        let mut max_distance: f32 = 0.0;
        let mut pair_idx = k;
        for i in 0..k {
            if scratch[i] < self.epsilon {
                continue;
            }
            let row_i = &weights[i * d..(i + 1) * d];
            let norm_i = scratch[i];
            for j in (i + 1)..k {
                if scratch[j] < self.epsilon {
                    continue;
                }
                let row_j = &weights[j * d..(j + 1) * d];
                let norm_j = scratch[j];
                let dot = simd_dot_f32(row_i, row_j, d);
                let denom = norm_i * norm_j;
                let cosine = if denom > 0.0 { dot / denom } else { 0.0 };
                scratch[pair_idx] = cosine;
                pair_idx += 1;
                let distance = 1.0 - cosine;
                if distance > max_distance {
                    max_distance = distance;
                }
            }
        }

        // Step 6: decision. `<=` is the inclusive boundary (paper-faithful).
        if max_distance <= self.tau_same {
            SmearReport {
                class: SmearClass::TokenSmear,
                span: survivor_count,
                n_hypotheses: 1,
                semantic_distance: max_distance,
            }
        } else {
            SmearReport {
                class: SmearClass::SequenceSmear,
                span: 1,
                n_hypotheses: survivor_count,
                semantic_distance: max_distance,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Plain float equality (no `approx` dep). 1e-5 tolerance covers f32 noise.
    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    /// Helper: allocate a fresh scratch buffer of the contract size for `k`.
    fn fresh_scratch(k: usize) -> Vec<f32> {
        let pairs = k.saturating_mul(k.saturating_sub(1)) / 2;
        vec![0.0; k + pairs]
    }

    /// T1.4.1 — single non-zero hypothesis → CoherentSingle.
    #[test]
    fn coherent_single_one_dominant_direction() {
        let clf = CosineSmearClassifier::default();
        // k=1, d=4, one significant row.
        let weights = [1.0, 0.0, 0.0, 0.0];
        let mut scratch = fresh_scratch(1);
        let report = clf.classify(&weights, 1, 4, &mut scratch);
        assert_eq!(report.class, SmearClass::CoherentSingle);
        assert_eq!(report.span, 1);
        assert_eq!(report.n_hypotheses, 1);
        assert!(approx_eq(report.semantic_distance, 0.0));
    }

    /// T1.4.2 — three parallel rows (cosine = 1.0, distance = 0.0 ≤ tau_same)
    /// → TokenSmear { span: 3, n_hypotheses: 1 }.
    #[test]
    fn token_smear_parallel_directions_across_sites() {
        let clf = CosineSmearClassifier::default();
        // k=3, d=2, all rows identical → perfectly parallel.
        let weights = [
            1.0, 0.0, // row 0
            1.0, 0.0, // row 1
            1.0, 0.0, // row 2
        ];
        let mut scratch = fresh_scratch(3);
        let report = clf.classify(&weights, 3, 2, &mut scratch);
        assert_eq!(report.class, SmearClass::TokenSmear);
        assert_eq!(report.span, 3);
        assert_eq!(report.n_hypotheses, 1);
        assert!(approx_eq(report.semantic_distance, 0.0));
    }

    /// T1.4.3 — two orthogonal rows at one site → SequenceSmear
    /// { n_hypotheses: 2, semantic_distance ≈ 1.0 }.
    #[test]
    fn sequence_smear_orthogonal_directions_one_site() {
        let clf = CosineSmearClassifier::default();
        // k=2, d=2, rows orthogonal: cosine = 0, distance = 1.0.
        let weights = [
            1.0, 0.0, // row 0
            0.0, 1.0, // row 1
        ];
        let mut scratch = fresh_scratch(2);
        let report = clf.classify(&weights, 2, 2, &mut scratch);
        assert_eq!(report.class, SmearClass::SequenceSmear);
        assert_eq!(report.span, 1);
        assert_eq!(report.n_hypotheses, 2);
        assert!(approx_eq(report.semantic_distance, 1.0));
    }

    /// T1.4.4 — sub-epsilon norms are dropped before classification.
    /// Three rows: two parallel significant, one near-zero. Should behave
    /// like the two-row parallel case → TokenSmear { span: 2 }.
    #[test]
    fn epsilon_filters_low_norm_hypotheses() {
        let clf = CosineSmearClassifier::default();
        // Row 2 has norm 1e-4 < epsilon (1e-3); must be filtered out.
        let weights = [
            1.0, 0.0, // row 0 — significant
            1.0, 0.0, // row 1 — significant
            1e-4, 0.0, // row 2 — sub-epsilon, dropped
        ];
        let mut scratch = fresh_scratch(3);
        let report = clf.classify(&weights, 3, 2, &mut scratch);
        assert_eq!(report.class, SmearClass::TokenSmear);
        assert_eq!(report.span, 2, "low-norm row must not count toward span");
        assert_eq!(report.n_hypotheses, 1);
    }

    /// T1.4.5 — distance exactly at tau_same → TokenSmear (inclusive `<=`).
    /// Constructed so the math is bit-exact: orthogonal rows give cosine = 0.0
    /// exactly (1*0 + 0*1 = 0.0 in IEEE 754), distance = 1.0 exactly. Set
    /// tau_same = 1.0 and verify the inclusive boundary fires TokenSmear.
    #[test]
    fn tau_same_boundary() {
        // Orthogonal rows → cosine = 0.0 exactly, distance = 1.0 exactly.
        let weights = [
            1.0, 0.0, // row 0
            0.0, 1.0, // row 1
        ];
        let clf_at_boundary = CosineSmearClassifier::new(1e-3, 1.0);
        let mut scratch = fresh_scratch(2);
        let report = clf_at_boundary.classify(&weights, 2, 2, &mut scratch);
        assert_eq!(
            report.class,
            SmearClass::TokenSmear,
            "distance == tau_same must classify as TokenSmear (inclusive <=)"
        );

        // Sanity: just above the boundary → SequenceSmear.
        let clf_above = CosineSmearClassifier::new(1e-3, 0.99);
        let report_above = clf_above.classify(&weights, 2, 2, &mut scratch);
        assert_eq!(report_above.class, SmearClass::SequenceSmear);
    }

    /// T1.4.6 — same input → bit-identical SmearReport.
    /// SmearReport derives PartialEq; f32 fields compare bit-exactly, so this
    /// catches any nondeterminism (RNG, HashMap iteration, etc.).
    #[test]
    fn deterministic_for_fixed_input() {
        let clf = CosineSmearClassifier::default();
        // Mixed layout: row 0 and row 1 near-parallel, row 2 orthogonal.
        let weights = [
            1.0, 0.0, 0.0, // row 0
            0.99, 0.01, 0.0, // row 1 — near-parallel to row 0
            0.0, 0.0, 1.0, // row 2 — orthogonal
        ];
        let mut scratch_a = fresh_scratch(3);
        let mut scratch_b = fresh_scratch(3);
        let a = clf.classify(&weights, 3, 3, &mut scratch_a);
        let b = clf.classify(&weights, 3, 3, &mut scratch_b);
        assert_eq!(a, b, "SmearReport must be bit-identical for fixed input");
    }
}
