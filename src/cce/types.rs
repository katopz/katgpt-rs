//! CCE primitive data types — occupation measure, deviation, deviation class,
//! payoff tensor (Plan 295 Phase 1, Research 274).
//!
//! These types are the wire/algorithm surface for the LP-CCE formulation.
//! They are intentionally generic over `<const N: usize, const A: usize>`
//! (state-space size, action-space size) so the same runtime can be applied
//! to 2-action chicken (N=4, A=2), 3-action RPS (N=3, A=3), 4-action emission
//! abatement (N=4, A=4), etc.

use core::marker::PhantomData;

/// Marker type for a finite state space of size `N`.
///
/// Not used directly in function signatures (everything is generic over
/// `<const N: usize>`), but provided for type-level documentation and any
/// future type-state extensions.
pub struct StateSpace<const N: usize>(PhantomData<[(); N]>);

/// Marker type for a finite action space of size `A`. See [`StateSpace`].
pub struct ActionSpace<const A: usize>(PhantomData<[(); A]>);

/// Occupation measure `ρ ∈ P(S × A)` — a probability distribution over
/// state-action pairs.
///
/// Stored row-major: `entries[s * A + a] == ρ(s, a)`. Length is `N · A`.
///
/// ## Invariant
///
/// On construction via [`OccupationMeasure::new`]: length is `N·A`, every
/// entry is `≥ -1e-6` (small negative tolerance for float noise), and the sum
/// is `1.0 ± 1e-5`.
///
/// Direct field construction (`OccupationMeasure { entries }`) bypasses
/// validation — used internally by trusted builders (e.g. simplex projections,
/// deviation application) where the invariant holds by construction.
#[derive(Clone, Debug)]
pub struct OccupationMeasure<const N: usize, const A: usize> {
    /// Flattened `N·A` probability entries, row-major: index `s·A + a`.
    pub entries: Vec<f32>,
}

/// Construction/validation errors for [`OccupationMeasure`].
#[derive(Debug)]
pub enum OccupationMeasureError {
    /// Entry vector length was not `N·A`.
    WrongLength { expected: usize, got: usize },
    /// Sum of entries was not within `1e-5` of `1.0`.
    NotNormalized { sum: f32 },
    /// An entry was below `-1e-6`.
    NegativeEntry { index: usize, value: f32 },
}

impl<const N: usize, const A: usize> OccupationMeasure<N, A> {
    /// Validate and construct an occupation measure from raw entries.
    pub fn new(entries: Vec<f32>) -> Result<Self, OccupationMeasureError> {
        let expected = N * A;
        if entries.len() != expected {
            return Err(OccupationMeasureError::WrongLength {
                expected,
                got: entries.len(),
            });
        }
        for (i, &v) in entries.iter().enumerate() {
            if v < -1e-6 {
                return Err(OccupationMeasureError::NegativeEntry {
                    index: i,
                    value: v,
                });
            }
        }
        let sum: f32 = entries.iter().copied().sum();
        if (sum - 1.0).abs() > 1e-5 {
            return Err(OccupationMeasureError::NotNormalized { sum });
        }
        Ok(Self { entries })
    }

    /// Uniform distribution over `S × A`: every entry `= 1 / (N·A)`.
    pub fn uniform() -> Self {
        let p = 1.0 / (N * A) as f32;
        Self {
            entries: vec![p; N * A],
        }
    }

    /// Dirac distribution on a single `(state, action)` pair.
    pub fn dirac(state: usize, action: usize) -> Self {
        let mut e = vec![0.0; N * A];
        e[state * A + action] = 1.0;
        Self { entries: e }
    }

    /// `ρ(state, action)`.
    #[inline]
    pub fn at(&self, state: usize, action: usize) -> f32 {
        self.entries[state * A + action]
    }

    /// `ρ(state, action) = value` (no re-validation).
    #[inline]
    pub fn set(&mut self, state: usize, action: usize, value: f32) {
        self.entries[state * A + action] = value;
    }

    /// Marginal probability of `state`: `μ(s) = Σ_a ρ(s, a)`.
    #[inline]
    pub fn marginal_state(&self, state: usize) -> f32 {
        let base = state * A;
        self.entries[base..base + A].iter().copied().sum()
    }

    /// Convert `(state, action)` to flat row-major index `s·A + a`.
    #[inline]
    pub fn flat_index(state: usize, action: usize) -> usize {
        state * A + action
    }

    /// Inverse of [`Self::flat_index`]: flat index → `(state, action)`.
    #[inline]
    pub fn unflat_index(flat: usize) -> (usize, usize) {
        (flat / A, flat % A)
    }

    /// Trusted constructor used by simplex projections and deviation
    /// application where the invariant holds by construction.
    #[inline]
    pub(crate) fn from_entries_trusted(entries: Vec<f32>) -> Self {
        debug_assert_eq!(
            entries.len(),
            N * A,
            "from_entries_trusted: wrong length"
        );
        Self { entries }
    }
}

impl<const N: usize, const A: usize> Default for OccupationMeasure<N, A> {
    fn default() -> Self {
        Self::uniform()
    }
}

/// A deviation `κ : S → P(A)` — a fixed alternative policy.
///
/// `kernel[s]` is a probability distribution over `A` for state `s`. The
/// intended reading: "when the mediator would have you play `a` at state `s`,
/// instead sample from `kernel[s]`."
#[derive(Clone, Debug)]
pub struct Deviation<const N: usize, const A: usize> {
    /// Opaque identifier (caller-assigned). Used for logging / deduplication.
    pub id: u32,
    /// `kernel[s][a] = Pr(play a | state s)` under this deviation.
    pub kernel: [[f32; A]; N],
}

impl<const N: usize, const A: usize> Deviation<N, A> {
    /// Constant deviation: always play `action` regardless of state.
    ///
    /// `kernel[s][action] = 1` for every `s`.
    pub fn constant(id: u32, action: usize) -> Self {
        let mut kernel = [[0.0f32; A]; N];
        for s in 0..N {
            kernel[s][action] = 1.0;
        }
        Self { id, kernel }
    }

    /// Identity deviation: play the recommended action. Requires `N == A`.
    ///
    /// `kernel[s][s] = 1` for every `s` (honest mediator: recommendation = action).
    pub fn identity(id: u32) -> Self {
        assert!(
            N == A,
            "identity deviation requires N == A (got N={N}, A={A})"
        );
        let mut kernel = [[0.0f32; A]; N];
        for s in 0..N {
            kernel[s][s] = 1.0;
        }
        Self { id, kernel }
    }

    /// Build from a raw kernel (caller validates that each `kernel[s]` is a
    /// probability distribution).
    pub fn from_kernel(id: u32, kernel: [[f32; A]; N]) -> Self {
        Self { id, kernel }
    }

    /// `Pr(play action | state)` under this deviation.
    #[inline]
    pub fn prob(&self, state: usize, action: usize) -> f32 {
        self.kernel[state][action]
    }
}

/// A finite class of deviations `D = {κ₁, …, κ_K}`.
///
/// The CCE constraint set is parameterized by `D`. The trait also provides
/// a default [`DeviationClass::apply`] that returns the deviated occupation
/// measure `ρ'(s, a') = μ(s) · κ(s)[a']` where `μ(s) = Σ_a ρ(s, a)` is the
/// state marginal under the original measure.
pub trait DeviationClass<const N: usize, const A: usize> {
    /// Slice of all deviations in the class.
    fn deviations(&self) -> &[Deviation<N, A>];

    /// Apply `κ` to `ρ`: redistribute each state's action mass according to
    /// `κ(s)`. Result `ρ'(s, a') = μ(s) · κ(s)[a']` is normalized because
    /// both `ρ` and `κ` are probability distributions.
    fn apply(
        &self,
        kappa: &Deviation<N, A>,
        rho: &OccupationMeasure<N, A>,
    ) -> OccupationMeasure<N, A> {
        let mut entries = vec![0.0; N * A];
        for s in 0..N {
            let mu_s = rho.marginal_state(s);
            if mu_s == 0.0 {
                continue;
            }
            let base = s * A;
            for a in 0..A {
                entries[base + a] = mu_s * kappa.kernel[s][a];
            }
        }
        OccupationMeasure::from_entries_trusted(entries)
    }
}

/// Payoff / cost tensor for the LP-CCE formulation.
///
/// ## Cost convention
///
/// `gamma` is the **cost** functional (paper minimizes `Γ₀`). The CCE LP is:
///
/// ```text
/// minimize   gamma0(ρ)
/// subject to gamma(ρ) ≤ gamma_dev(ρ, κ)   for all κ ∈ D
///            sum_{s,a} ρ(s,a) = 1, ρ ≥ 0
/// ```
///
/// `reward_follow(s, a) = cost(s, a)` is the per-index cost of following the
/// mediator's recommendation at `(s, a)`. `gamma(ρ)` and `gamma_dev(ρ, κ)`
/// have default implementations in terms of `reward_follow`; override only
/// when an impl has a closed form (e.g., a precomputed cost matrix).
pub trait PayoffTensor<const N: usize, const A: usize> {
    /// Per-index cost of following: `cost(s, a)`.
    fn reward_follow(&self, state: usize, action: usize) -> f32;

    /// Per-state expected cost of deviating to `κ`:
    /// `Σ_{a'} κ(s)[a'] · cost(s, a')`.
    ///
    /// Default impl: dot product of `κ(s)` with `cost(s, ·)`.
    fn reward_deviate(&self, state: usize, kappa: &Deviation<N, A>) -> f32 {
        let mut g = 0.0;
        for a in 0..A {
            g += kappa.kernel[state][a] * self.reward_follow(state, a);
        }
        g
    }

    /// Cost of following the recommendation under `ρ`:
    /// `Γ(ρ) = Σ_{s,a} ρ(s, a) · cost(s, a)`.
    ///
    /// Default impl: linear in `ρ` via `reward_follow`.
    fn gamma(&self, rho: &OccupationMeasure<N, A>) -> f32 {
        let mut g = 0.0;
        for s in 0..N {
            for a in 0..A {
                g += rho.at(s, a) * self.reward_follow(s, a);
            }
        }
        g
    }

    /// Cost of deviating to `κ` under `ρ`:
    /// `Γ_dev(ρ, κ) = Σ_s μ(s) · reward_deviate(s, κ)`
    /// where `μ(s) = Σ_a ρ(s, a)` is the state marginal.
    ///
    /// Default impl: linear in `μ` via `reward_deviate`.
    fn gamma_dev(
        &self,
        rho: &OccupationMeasure<N, A>,
        kappa: &Deviation<N, A>,
    ) -> f32 {
        let mut g = 0.0;
        for s in 0..N {
            let mu_s = rho.marginal_state(s);
            if mu_s == 0.0 {
                continue;
            }
            g += mu_s * self.reward_deviate(s, kappa);
        }
        g
    }

    /// Moderator objective `Γ₀(ρ)` — the world-level cost the LP minimizes
    /// (e.g., expected emission, expected economic loss). Implementer-defined.
    fn gamma0(&self, rho: &OccupationMeasure<N, A>) -> f32;

    /// Per-index coefficient of the moderator objective, assuming `Γ₀` is
    /// linear: `Γ₀(ρ) = Σ_{s,a} ρ(s,a) · gamma0_coeff(s, a)`.
    ///
    /// Default: `gamma0_coeff = reward_follow` (i.e. `Γ₀ = Γ`, the moderator
    /// objective equals the player's cost of following). Override when the
    /// moderator has a separate objective (e.g., world-level welfare ≠ player
    /// cost). Used by `CceLp::solve` (Phase 2) to build the LP objective row.
    fn gamma0_coeff(&self, state: usize, action: usize) -> f32 {
        self.reward_follow(state, action)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn occupation_measure_uniform_sums_to_one() {
        let rho = OccupationMeasure::<3, 4>::uniform();
        assert_eq!(rho.entries.len(), 12);
        let sum: f32 = rho.entries.iter().copied().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum = {sum}");
        for &v in &rho.entries {
            assert!((v - 1.0 / 12.0).abs() < 1e-6);
        }
    }

    #[test]
    fn occupation_measure_dirac_is_canonical() {
        let rho = OccupationMeasure::<3, 2>::dirac(1, 1);
        assert_eq!(rho.entries, vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        assert_eq!(rho.marginal_state(0), 0.0);
        assert_eq!(rho.marginal_state(1), 1.0);
        assert_eq!(rho.marginal_state(2), 0.0);
    }

    #[test]
    fn occupation_measure_new_rejects_bad_inputs() {
        // Wrong length.
        let err = OccupationMeasure::<2, 2>::new(vec![0.5, 0.5]).unwrap_err();
        assert!(matches!(err, OccupationMeasureError::WrongLength { expected: 4, got: 2 }));

        // Negative entry.
        let err = OccupationMeasure::<2, 1>::new(vec![1.5, -0.5]).unwrap_err();
        assert!(matches!(err, OccupationMeasureError::NegativeEntry { index: 1, .. }));

        // Not normalized (sum = 0.4).
        let err = OccupationMeasure::<2, 2>::new(vec![0.1, 0.1, 0.1, 0.1]).unwrap_err();
        assert!(matches!(err, OccupationMeasureError::NotNormalized { .. }));

        // Valid.
        let rho = OccupationMeasure::<2, 2>::new(vec![0.3, 0.2, 0.1, 0.4]).unwrap();
        assert_eq!(rho.at(0, 1), 0.2);
        assert_eq!(rho.at(1, 0), 0.1);
    }

    #[test]
    fn occupation_measure_flat_index_roundtrip() {
        for s in 0..4 {
            for a in 0..3 {
                let flat = OccupationMeasure::<4, 3>::flat_index(s, a);
                assert_eq!(OccupationMeasure::<4, 3>::unflat_index(flat), (s, a));
            }
        }
    }

    #[test]
    fn deviation_constant_is_valid_kernel() {
        let kappa = Deviation::<3, 4>::constant(7, 2);
        assert_eq!(kappa.id, 7);
        for s in 0..3 {
            let row_sum: f32 = kappa.kernel[s].iter().copied().sum();
            assert!((row_sum - 1.0).abs() < 1e-6, "row {s} sum = {row_sum}");
            assert_eq!(kappa.prob(s, 2), 1.0);
            assert_eq!(kappa.prob(s, 0), 0.0);
        }
    }

    #[test]
    fn deviation_identity_requires_square() {
        let kappa = Deviation::<3, 3>::identity(0);
        for s in 0..3 {
            assert_eq!(kappa.prob(s, s), 1.0);
            assert_eq!(kappa.prob(s, (s + 1) % 3), 0.0);
        }
    }

    #[test]
    fn deviation_class_apply_redistributes_mass() {
        struct TwoDevs {
            v: Vec<Deviation<2, 2>>,
        }
        impl DeviationClass<2, 2> for TwoDevs {
            fn deviations(&self) -> &[Deviation<2, 2>] {
                &self.v
            }
        }
        let d = TwoDevs {
            v: vec![Deviation::<2, 2>::constant(0, 1)],
        };
        // ρ = [[0.4, 0.1], [0.3, 0.2]] — μ = [0.5, 0.5].
        let rho = OccupationMeasure::<2, 2>::new(vec![0.4, 0.1, 0.3, 0.2]).unwrap();
        let kappa = &d.deviations()[0]; // always play action 1.
        let rho_prime = d.apply(kappa, &rho);
        // ρ'(s, 1) = μ(s) · 1 = μ(s); ρ'(s, 0) = 0.
        assert_eq!(rho_prime.at(0, 0), 0.0);
        assert!((rho_prime.at(0, 1) - 0.5).abs() < 1e-6);
        assert_eq!(rho_prime.at(1, 0), 0.0);
        assert!((rho_prime.at(1, 1) - 0.5).abs() < 1e-6);
        let sum: f32 = rho_prime.entries.iter().copied().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn payoff_tensor_default_impls_are_consistent() {
        // Custom 2x2 cost tensor: cost(s,a) = (s+1)*(a+1).
        struct QuadCost;
        impl PayoffTensor<2, 2> for QuadCost {
            fn reward_follow(&self, s: usize, a: usize) -> f32 {
                ((s + 1) as f32) * ((a + 1) as f32)
            }
            fn gamma0(&self, rho: &OccupationMeasure<2, 2>) -> f32 {
                self.gamma(rho)
            }
        }
        let p = QuadCost;
        // ρ = [[0.1, 0.2], [0.3, 0.4]].
        let rho = OccupationMeasure::<2, 2>::new(vec![0.1, 0.2, 0.3, 0.4]).unwrap();
        // cost(s,a) = (s+1)*(a+1):
        //   cost(0,0)=1, cost(0,1)=2, cost(1,0)=2, cost(1,1)=4.
        // Γ(ρ) = 0.1·1 + 0.2·2 + 0.3·2 + 0.4·4 = 0.1 + 0.4 + 0.6 + 1.6 = 2.7.
        assert!((p.gamma(&rho) - 2.7).abs() < 1e-6);

        let kappa = Deviation::<2, 2>::constant(0, 1); // always play a=1.
        // reward_deviate(0, κ) = κ(0)[1]·cost(0,1) = 1·2 = 2.
        assert!((p.reward_deviate(0, &kappa) - 2.0).abs() < 1e-6);
        // reward_deviate(1, κ) = κ(1)[1]·cost(1,1) = 1·4 = 4.
        assert!((p.reward_deviate(1, &kappa) - 4.0).abs() < 1e-6);
        // μ(0) = 0.1+0.2 = 0.3, μ(1) = 0.3+0.4 = 0.7.
        // Γ_dev(ρ, κ) = 0.3·2 + 0.7·4 = 0.6 + 2.8 = 3.4.
        assert!((p.gamma_dev(&rho, &kappa) - 3.4).abs() < 1e-6);
    }
}
