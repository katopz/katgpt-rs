//! ConvexTok optimality certification — compare achieved compression vs LP-proven lower bound.
//!
//! **Source:** Tempus et al. (2026). Tokenisation via Convex Relaxations. arXiv:2605.22821

use super::convex_types::{LpSolution, OptimalityCert, RoundedVocabulary};

/// Certifier for tokeniser optimality.
pub struct Certifier;

impl Certifier {
    /// Certify how close a rounded vocabulary is to LP-proven optimal.
    ///
    /// # Arguments
    /// * `lp_solution` — The LP relaxation solution (provides lower bound)
    /// * `rounded` — The rounded vocabulary (provides actual compression)
    pub fn certify(lp_solution: &LpSolution, rounded: &RoundedVocabulary) -> OptimalityCert {
        let lp_lower_bound = lp_solution.lp_value;
        let actual_compression = rounded.compression_value;

        // Gap: (actual - lp) / lp * 100%
        // If lp_lower_bound is ~0 (degenerate), gap is 0%
        let gap_percent = if lp_lower_bound.abs() > 1e-10 {
            (actual_compression - lp_lower_bound) / lp_lower_bound * 100.0
        } else {
            0.0
        };

        OptimalityCert {
            lp_lower_bound,
            actual_compression,
            gap_percent,
            within_one_percent: gap_percent <= 1.0,
            integrality_fraction: lp_solution.integrality_fraction(),
        }
    }

    /// Certify an arbitrary tokenizer (e.g., BPE) against the LP bound.
    ///
    /// # Arguments
    /// * `lp_solution` — LP solution providing the lower bound
    /// * `tokenizer_compression` — Compression achieved by the external tokenizer
    pub fn certify_external(
        lp_solution: &LpSolution,
        tokenizer_compression: f64,
    ) -> OptimalityCert {
        let gap_percent = if lp_solution.lp_value.abs() > 1e-10 {
            (tokenizer_compression - lp_solution.lp_value) / lp_solution.lp_value * 100.0
        } else {
            0.0
        };

        OptimalityCert {
            lp_lower_bound: lp_solution.lp_value,
            actual_compression: tokenizer_compression,
            gap_percent,
            within_one_percent: gap_percent <= 1.0,
            integrality_fraction: 0.0, // unknown for external tokenizer
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::convex_types::{LpSolution, RoundedVocabulary, RoundingScheme};
    use super::Certifier;

    fn make_lp(lp_value: f64, c: Vec<f64>) -> LpSolution {
        let budget_k = c.len();
        LpSolution {
            f: vec![],
            p: vec![],
            c,
            lp_value,
            budget_k,
        }
    }

    fn make_rounded(compression_value: f64, n_selected: usize) -> RoundedVocabulary {
        RoundedVocabulary {
            selected_colours: vec![],
            selected_bytes: vec![],
            n_selected,
            compression_value,
            rounding_scheme: RoundingScheme::Det,
        }
    }

    #[test]
    fn perfect_match_zero_gap() {
        let lp = make_lp(2.5, vec![1.0, 0.0, 0.5]);
        let rounded = make_rounded(2.5, 2);

        let cert = Certifier::certify(&lp, &rounded);

        assert!((cert.gap_percent - 0.0).abs() < 1e-10, "gap should be 0%");
        assert!(cert.within_one_percent, "should be within one percent");
        assert!((cert.lp_lower_bound - 2.5).abs() < 1e-10);
        assert!((cert.actual_compression - 2.5).abs() < 1e-10);
    }

    #[test]
    fn small_gap_within_one_percent() {
        // lp=1.0, actual=1.005 → gap=0.5%, within_one_percent=true
        let lp = make_lp(1.0, vec![0.5, 0.5]);
        let rounded = make_rounded(1.005, 2);

        let cert = Certifier::certify(&lp, &rounded);

        assert!((cert.gap_percent - 0.5).abs() < 1e-10, "gap should be 0.5%");
        assert!(
            cert.within_one_percent,
            "0.5% gap should be within one percent"
        );
    }

    #[test]
    fn large_gap_outside_one_percent() {
        // lp=1.0, actual=1.5 → gap=50%, within_one_percent=false
        let lp = make_lp(1.0, vec![0.5, 0.5]);
        let rounded = make_rounded(1.5, 2);

        let cert = Certifier::certify(&lp, &rounded);

        assert!((cert.gap_percent - 50.0).abs() < 1e-10, "gap should be 50%");
        assert!(
            !cert.within_one_percent,
            "50% gap should NOT be within one percent"
        );
    }

    #[test]
    fn certify_external_same_logic() {
        // certify_external should produce the same gap as certify for identical values
        let lp = make_lp(3.0, vec![1.0, 0.0, 1.0]);
        let rounded = make_rounded(3.15, 2);

        let cert_internal = Certifier::certify(&lp, &rounded);
        let cert_external = Certifier::certify_external(&lp, rounded.compression_value);

        assert!(
            (cert_internal.gap_percent - cert_external.gap_percent).abs() < 1e-10,
            "gap should match between certify and certify_external"
        );
        assert_eq!(
            cert_internal.within_one_percent,
            cert_external.within_one_percent
        );
        // integrality_fraction differs: internal uses LP data, external is 0.0
        assert!((cert_external.integrality_fraction - 0.0).abs() < 1e-10);
        assert!(
            cert_internal.integrality_fraction > 0.0,
            "internal should use LP data"
        );
    }

    #[test]
    fn zero_lp_bound_handles_gracefully() {
        // lp_value=0.0 should not panic or produce NaN/Inf
        let lp = make_lp(0.0, vec![0.5]);
        let rounded = make_rounded(1.0, 1);

        let cert = Certifier::certify(&lp, &rounded);

        assert!(
            cert.gap_percent.is_finite(),
            "gap should be finite with zero LP bound"
        );
        assert!(
            (cert.gap_percent - 0.0).abs() < 1e-10,
            "gap should be 0% for degenerate LP"
        );

        // Also test certify_external with zero bound
        let cert_ext = Certifier::certify_external(&lp, 1.0);
        assert!(
            cert_ext.gap_percent.is_finite(),
            "external gap should be finite with zero LP bound"
        );
    }
}
