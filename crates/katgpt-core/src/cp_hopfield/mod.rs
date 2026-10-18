//! cp_hopfield — Top-eigenvector associative memory recall on `CP^(d-1) = SU(d)/U(d-1)`.
//!
//! Distilled from Victor Galitski, *High-Capacity Generalized Hopfield Networks*
//! (alphaXiv 2607.hopfield-networks, JQI/UMD, 2026-07-31), with the Lie-algebraic
//! linearization from Galitski, *Phys. Rev. A* **84**, 012118 (2011).
//!
//! # The mechanism
//!
//! A neuron is a normalized `d`-dimensional complex vector `|ξ⟩` (a "qudit")
//! defined modulo a `U(1)` phase — a point on the complex projective space
//! `CP^(d-1)`. It embeds into `R^(d²-1)` as a generalized Bloch magnetization
//! `s_a = ⟨ξ|λ_a|ξ⟩`, where `{λ_a}` are the SU(d) generators (Pauli for `d=2`,
//! Gell-Mann for `d=3`, generalized Gell-Mann for `d>3`).
//!
//! Recall is **not** vector alignment. The `i`-th neuron's active energy is a
//! Rayleigh quotient over a `d×d` Hermitian *memory kernel*:
//!
//! ```text
//! E_active[s_i] = −2 ⟨s_i | K_i | s_i⟩
//! K_i = Σ_μ O_μ^(i) |ξ_i^μ⟩⟨ξ_i^μ|
//! O_μ^(i) = (2/N) Σ_{j≠i} (|⟨s_j | ξ_j^μ⟩|² − 1/d)      (Mattis overlap, excl. i)
//! ```
//!
//! so recall = align `|s_i⟩` with the **top eigenvector** of `K_i`.
//!
//! # Why capacity inverts
//!
//! Near a stored memory the kernel decomposes as `K ≈ m|1⟩⟨1| + √α · C(d) · G`
//! with `G` a GUE random matrix. The signal is a rank-1 *spike*; the crosstalk is
//! a semicircle bulk of half-width `2√α C(d)`. The top eigenvector stays pinned to
//! the spike until the spike merges with the bulk — the **Baik-Ben Arous-Péché
//! (BBP) transition**. `C(d)` decays as `1/d²`, so the bulk shrinks and capacity
//! *grows* with `d`, where sphere-based vector alignment is gapless and decays:
//!
//! | Manifold | `α_c` (asymptotic in `N`) | Mechanism |
//! |---|---|---|
//! | `S^(n-1)` | `4/(27n)` | vector alignment (gapless) |
//! | `CP^1 = S^2` (`d=2`) | 0.05 | top-eigenvector ≡ vector alignment |
//! | `CP^2` (`d=3`, qutrit) | 0.62 | top-eigenvector (BBP-gapped) |
//! | `CP^3` (`d=4`) | 2.41 | top-eigenvector |
//! | `CP^7` (`d=8`) | ~40 | top-eigenvector |
//!
//! **These numbers are asymptotic in `N`.** See [`capacity`] for the measured
//! finite-`N` values at the `N` this codebase actually uses (8, 64); the honest
//! finite-`N` correction is large and is the reason this primitive ships opt-in.
//!
//! # Modelless contract
//!
//! Everything here is deterministic construction plus numerical linear algebra:
//! the SU(d) basis and its structure constants are built in closed form, the
//! kernel is a Hebbian outer-product sum, and recall is Rayleigh-quotient ascent
//! (power iteration). **No gradient descent, no training, no learned parameters.**
//! This is what lets the primitive be loaded from a frozen snapshot (freeze/thaw
//! Path 1) rather than trained — the "stable fixed points" come from the geometry,
//! not from tuned recurrent weights.
//!
//! # Two recall paths
//!
//! - [`CpHopfieldRecaller::recall_step`] — the discrete asynchronous rule: build
//!   `K_i`, take its top eigenvector, project back to Bloch coordinates.
//! - [`llg`] — the continuous dissipative flow (generalized Landau-Lifshitz-
//!   Gilbert). Precession conserves energy; Gilbert damping lowers it monotonically
//!   (`Ė = −λ Σ |s_i ×_f B_i|² ≤ 0`). Recall = flow to a fixed point.
//!
//! # Const generics
//!
//! `D` is the complex dimension `d`; `D2` is the real Bloch dimension `d²−1`.
//! Stable Rust cannot express `[f32; D*D-1]`, so `D2` is a second const
//! parameter checked against `D` at construction. Use the [`CpHopfield2`],
//! [`CpHopfield3`], [`CpHopfield4`], [`CpHopfield8`] aliases rather than spelling
//! the pair out.
//!
//! # Scope — what this module deliberately omits
//!
//! The source paper's quantum extension (§VI–VII) is a *negative* result: it
//! concludes Hebbian data is "completely lost in random matrix spectra" in the
//! quantized model. The operative mechanism is the classical flow, so only that
//! is distilled here. The paper's RGB→qutrit image encoder (§IV) is likewise
//! application-specific; the recall mechanism is encoder-agnostic.
//!
//! See `.research/466_CPd_minus_1_Hopfield_Top_Eigenvector_Recall.md` and
//! `.plans/567_cp_hopfield_top_eigenvector_recall.md`.
//!
//! CP^(d-1) Symmetric-Space Hopfield — Top-Eigenvector Recall (Plan 567,
//! Research 466, Galitski "High-Capacity Generalized Hopfield Networks",
//! alphaXiv 2607.hopfield-networks, JQI/UMD 2026-07-31).
//!
//! Associative-memory recall on the symmetric space CP^(d-1) = SU(d)/U(d-1)
//! instead of the sphere. The memory kernel K_i = Σ_μ O_μ^(i) |ξ_i^μ⟩⟨ξ_i^μ| is a
//! d×d Hermitian SPIKED random matrix and recall aligns the neuron with its TOP
//! EIGENVECTOR, which is BBP-protected against GUE crosstalk. That gap is why
//! CP^(d-1) capacity grows with d (α_c ≈ 0.62 at d=3, 2.41 at d=4) where gapless
//! vector alignment on S^(n-1) decays as 4/(27n) — and it is the mechanism the
//! Plan 276 AttractorKernel lacked when it failed G2.1 at random init.
//!
//! Pure modelless: closed-form SU(d) basis construction + Hebbian outer-product
//! kernel + Rayleigh-quotient ascent. No gradient descent, so memories load from a
//! frozen snapshot (freeze/thaw Path 1) rather than being trained.
//!
//! OPT-IN pending the Plan 567 GOAT gate. Load-bearing gates are G5 (the Plan 276
//! modelless unblock) and G7 (whether the BBP gap survives at N=8/N=64 rather than
//! only asymptotically in N). Zero runtime cost unless a caller constructs a
//! CpHopfieldRecaller.

mod basis;
pub mod capacity;
mod complex;
pub mod llg;
mod recaller;

#[cfg(test)]
mod tests;

pub use basis::{GellMannBasis, StructureConstants};
pub use capacity::{CapacityCurve, CapacityPoint, MemoryDistribution, measure_capacity};
pub use complex::C32;
pub use llg::{LlgConfig, RecallResult, lie_bracket_into};
pub use recaller::{CpHopfieldRecaller, KernelSpectrum};

/// `CP^1 = S^2` — SU(2)/qubit. Top-eigenvector recall degenerates to ordinary
/// vector (Heisenberg) alignment here; `α_c ≈ 0.05`. Useful as the control arm
/// that isolates what the `d ≥ 3` spiked-matrix mechanism actually buys.
pub type CpHopfield2 = CpHopfieldRecaller<2, 3>;

/// `CP²` — SU(3)/qutrit, `α_c ≈ 0.62` asymptotically. The first dimension where
/// `CP^(d-1)` is not a sphere and the BBP-protected mechanism activates. Its
/// Bloch dimension `d²−1 = 8` coincides with the 8-dim `katgpt-sense` belief
/// state, which is why this is the default arm for the Plan 276 unblock PoC.
pub type CpHopfield3 = CpHopfieldRecaller<3, 8>;

/// `CP³` — SU(4), `α_c ≈ 2.41` asymptotically. First dimension with `α_c > 1`
/// (more memories than neurons).
pub type CpHopfield4 = CpHopfieldRecaller<4, 15>;

/// `CP⁷` — SU(8), `α_c ≈ 40` asymptotically. The paper's replica analysis is
/// replica-symmetric and it explicitly declines to explore RSB corrections, which
/// are most likely to bite at large `d` — treat this arm's capacity as an upper
/// bound, not a measurement.
pub type CpHopfield8 = CpHopfieldRecaller<8, 63>;

/// Squared Bloch-vector norm of any pure state on `CP^(d-1)`: `|s|² = 2(1 − 1/d)`.
///
/// Follows from the SU(d) completeness relation; see the module docs.
#[inline]
pub fn bloch_norm_sq(d: usize) -> f32 {
    2.0 * (1.0 - 1.0 / d as f32)
}

/// Right-hand side of the non-linear `CP^(d-1)` constraint
/// `d_abc s_a s_b = (2 − 4/d) s_c`.
///
/// Derived from purity `ρ² = ρ` for `ρ = (1/d)I + (1/2) s_a λ_a`. At `d = 3` this
/// is `2/3`, matching the source paper's §VIII.C form; at `d = 2` it vanishes
/// alongside `d_abc` itself, leaving only the norm constraint (`CP^1 = S^2` has
/// no constraint beyond the sphere).
#[inline]
pub fn constraint_rhs(d: usize) -> f32 {
    2.0 - 4.0 / d as f32
}

/// Normalized Mattis overlap between two Bloch vectors on `CP^(d-1)`.
///
/// Returns `1.0` for identical states and `0.0` in expectation for independent
/// Haar-random states, so it reads directly as a recall-quality score:
///
/// ```text
/// m = (|⟨u|v⟩|² − 1/d) / (1 − 1/d) = (s_u · s_v) / (2(1 − 1/d))
/// ```
///
/// The identity `|⟨u|v⟩|² = 1/d + (1/2) s_u · s_v` (from `Tr(ρ_u ρ_v)`) is what
/// lets the whole overlap computation stay in real Bloch arithmetic instead of
/// round-tripping through complex amplitudes.
///
/// Range is `[−1/(d−1), 1]`, not `[−1, 1]`: two pure states can never be more
/// than orthogonal.
#[inline]
pub fn bloch_overlap(u: &[f32], v: &[f32], d: usize) -> f32 {
    debug_assert_eq!(u.len(), v.len());
    let dot: f32 = u.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
    dot / bloch_norm_sq(d)
}
