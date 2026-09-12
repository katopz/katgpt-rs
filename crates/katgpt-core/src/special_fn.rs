// special_fn — shared special-function substrate (Lanczos ln_gamma, g=7,
// f64). Extracted from best_belief.rs (Plan 597 T1.1, 2026-09-12) so bmr
// consumes the SAME kernel instead of duplicating it — one ln_gamma per
// crate, the substrate-first rule. UNGATED and pub(crate): zero-cost when
// unused (an uninstantiated fn), invisible on the public API surface.
// Behavior is bit-identical to the best_belief original (same coefficients,
// same reflection branch) — best_belief's G1 fixtures pin it.

/// ln Γ(x) via the Lanczos approximation (g=7, 9 coefficients), f64.
/// Reflection formula for x < 0.5: Γ(x)Γ(1-x) = π / sin(πx).
pub(crate) fn ln_gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.5203681218851,
        -1259.1392167224028,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507343278686905,
        -0.13857109526572012,
        9.984_369_578_019_572e-6,
        1.5056327351493116e-7,
    ];
    if x < 0.5 {
        // Reflection: Γ(x)Γ(1-x) = π / sin(πx).
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin().abs()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = C[0];
    let t = x + G + 0.5;
    for (i, c) in C.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

#[cfg(test)]
mod tests {
    use super::ln_gamma;

    /// Integers: Γ(n) = (n-1)! — check via ln to avoid overflow.
    #[test]
    fn ln_gamma_integers_match_factorial() {
        let cases: [(f64, f64); 6] = [
            (1.0, 0.0),        // Γ(1) = 0! = 1
            (2.0, 0.0),        // Γ(2) = 1! = 1
            (3.0, 2.0_f64.ln()),  // Γ(3) = 2!
            (4.0, 6.0_f64.ln()),  // Γ(4) = 3!
            (5.0, 24.0_f64.ln()), // Γ(5) = 4!
            (6.0, 120.0_f64.ln()), // Γ(6) = 5!
        ];
        for (x, want) in cases {
            let got = ln_gamma(x);
            assert!(
                (got - want).abs() < 1e-10,
                "ln_gamma({x}) = {got}, want {want}"
            );
        }
    }

    /// Half-integers: Γ(1/2)=√π, Γ(3/2)=√π/2, Γ(5/2)=3√π/4.
    #[test]
    fn ln_gamma_half_integers() {
        let sqrt_pi = std::f64::consts::PI.sqrt();
        let cases = [
            (0.5, sqrt_pi.ln()),
            (1.5, (sqrt_pi / 2.0).ln()),
            (2.5, (3.0 * sqrt_pi / 4.0).ln()),
            (3.5, (15.0 * sqrt_pi / 8.0).ln()),
        ];
        for (x, want) in cases {
            let got = ln_gamma(x);
            assert!(
                (got - want).abs() < 1e-10,
                "ln_gamma({x}) = {got}, want {want}"
            );
        }
    }

    /// Monotone increasing for x >= 2 (ψ(x) > 0 there).
    #[test]
    fn ln_gamma_monotone_for_large_x() {
        let mut prev = ln_gamma(2.0);
        for i in 3..40 {
            let cur = ln_gamma(f64::from(i));
            assert!(cur > prev, "ln_gamma not monotone at {i}");
            prev = cur;
        }
    }

    /// x -> 1 gives 0 (Γ(1) = 1).
    #[test]
    fn ln_gamma_at_one_is_zero() {
        assert!(ln_gamma(1.0).abs() < 1e-12);
    }

    /// Reflection branch sanity: Γ(x)Γ(1-x) = π/sin(πx) at x = 0.25.
    #[test]
    fn ln_gamma_reflection_identity() {
        let x = 0.25;
        let lhs = ln_gamma(x) + ln_gamma(1.0 - x);
        let pi = std::f64::consts::PI;
        let rhs = (pi / (pi * x).sin()).ln();
        assert!((lhs - rhs).abs() < 1e-10, "{lhs} vs {rhs}");
    }
}
