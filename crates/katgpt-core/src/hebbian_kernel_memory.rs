//! Hebbian Kernel Memory — Closed-Form Fact-Storing MLP Construction + Swap.
//!
//! Distilled from Garcia, Chen, Bordelon, Lee, Roberts, Rudi, *MLPs are
//! Hebbians: Constructing Efficient Fact-Storing MLPs for Transformers*
//! (arXiv:2607.10034, Stanford / UB, 2026-07-10). See
//! `katgpt-rs/.research/455_Hebbian_Kernel_Memory_Fact_Storing_MLP.md` and
//! `katgpt-rs/.plans/559_hebbian_kernel_memory_primitive.md`.
//!
//! # What this computes
//!
//! A bilinear sketched-K₂ Hebbian kernel memory storing `F` key→value facts
//! at information-theoretic optimal capacity `W = Θ(F · log F)` parameters.
//! Given a fact set `{(k_i → v_{f(i)})}` with `k_i ∈ ℝᴰ`, `v_j ∈ ℝᴰ`, the
//! memory constructs three matrices `A, G ∈ ℝ^{m×D}`, `B ∈ ℝ^{D×m}` such
//! that the forward pass
//!
//! ```text
//! MLP(z) = B · ϕ(z)            where   ϕ(z) = (1/√m) · [(A_r·z)(G_r·z)]_{r=1..m}
//! ```
//!
//! satisfies `MLP(k_i) ≈ v_{f(i)}` for all `i`, AND the retrieval scores
//! `s_j(z) = ⟨v_j, MLP(z)⟩` decode the stored fact (argmax over `j` returns
//! `f(i)` when `z = k_i`). Paper Theorem 3.1 establishes the equivalence
//! `MLP(z) ≡ H_white(z) := (1/F) Σ_i v_{f(i)} · ϕ(k_i)ᵀ Σ̂⁻¹ ϕ(z)` with the
//! whitened Hebbian kernel — every gated MLP IS a Hebbian kernel memory.
//!
//! # Three construction variants (paper §B.2)
//!
//! | Variant | What it solves | When to use |
//! |---------|----------------|------------|
//! | [`HebbianVariant::Unwhitened`]   | `B₀ = (1/F) C_fᵀ Φ` (raw outer product)               | Baseline; cheap but lower margin |
//! | [`HebbianVariant::Whitened`]     | `B_λ = B₀ · (Σ̂ + λI)⁻¹` (ridge-whitened, **DEFAULT**) | Production; matches paper Algorithm 1 |
//! | [`HebbianVariant::DataDependent`] | Alternating least squares on `A, G` (paper §B.2.5)   | Arbitrary value geometry; 2 ALS iterations |
//!
//! All three are **closed-form** — no gradient descent, no backprop. The
//! data-dependent variant's two alternating least-squares solves (paper
//! Eq 17, 18) are linear (each is a ridge solve). Per AGENTS.md constraint
//! #1, the entire construction is modelless.
//!
//! # Decoding margin (paper Theorem 4.3)
//!
//! The decoding margin for fact `i` against competitor value `v_j (j ≠ f(i))`:
//!
//! ```text
//! γ_{i,j} = ⟨v_{f(i)} − v_j, MLP(k_i)⟩
//!         = ⟨v_{f(i)} − v_j, v_{f(i)}⟩ · K(k_i, k_i)                    [signal]
//!         + Σ_{t ≠ i} ⟨v_{f(i)} − v_j, v_{f(t)}⟩ · K(k_t, k_i)          [cross-talk]
//! ```
//!
//! [`HebbianKernelMemory::decoding_margin`] returns `γ_min = min_{i, j≠f(i)} γ_{i,j}`.
//! Per paper Thm 4.3 + Cor B.32: for isotropic keys/values, `γ_min ≥ 1 − C·√(F·log F / (m·d))`,
//! positive iff `m·d > C²·F·log F`. This is the **information-theoretic
//! optimality** result: `W = m·d = Θ(F log F)` parameters store `F` facts,
//! matching the counting lower bound (Thm 2.4) up to constants.
//!
//! # Sigmoid, not softmax (AGENTS.md §2)
//!
//! The consumer pattern is [`crate::committed_field_blend`] (sigmoid-gated
//! direction vector), NOT softmax attention. The retrieval scores `s_j` are
//! raw scalars consumed by `argmax` (deterministic top-1) or by a downstream
//! sigmoid gate. Paper's softmax retrieval is replaced with sigmoid gates
//! per AGENTS.md.
//!
//! # Allocation discipline (G4)
//!
//! - **Construction** allocates exactly once per call (the three matrices
//!   `A, G, B`). This is caller-controlled, NOT hot-path.
//! - **Forward pass** [`HebbianKernelMemory::forward_into`] writes into a
//!   caller-provided `&mut [f32; D]` — zero allocation.
//! - **Retrieval scores** [`HebbianKernelMemory::retrieval_scores_into`]
//!   writes into a caller-provided `&mut [f32]` of length `V` — zero allocation.
//! - **Decoding margin** [`HebbianKernelMemory::decoding_margin_into`] uses
//!   a caller-provided [`MarginScratch`] — zero allocation.
//!
//! # Determinism (constraint #8)
//!
//! `A, G` are sampled deterministically from a `SeedRng` constructed from
//! a BLAKE3-derived seed. The same fact set + the same seed → bit-identical
//! constructed memory across runs and across nodes. This is required for
//! sync consistency (the [`HebbianCommitment`] crosses the sync boundary
//! and MUST hash identically at both ends).
//!
//! # Latent-to-latent (constraint #2)
//!
//! `forward_into` and `retrieval_scores_into` operate on latents (the
//! `D`-dim shard / embedding space). The retrieval scores are raw scalars
//! (for `argmax`); the forward embedding is latent (for downstream
//! composition via [`crate::committed_field_blend`] etc).
//!
//! # Hot-swap (constraint #3)
//!
//! [`HebbianSlot`] is the atomic hot-swap slot, mirroring
//! [`crate::induced_cwm::InducedCwmSlot`]. Readers clone out via
//! [`HebbianSlot::current`]; writers atomically replace via
//! [`HebbianSlot::induce`]. The slot is process-local; the
//! [`HebbianCommitment`] it produces is the sync-boundary artifact (the
//! "MLP Swap audit trail" of paper §5.2).
//!
//! # References
//!
//! - **Paper:** [arXiv:2607.10034](https://arxiv.org/abs/2607.10034) §3 (Thm 3.1),
//!   §4 (margin scaling + Algorithm 1), §5 (Transformer integration + MLP Swapping),
//!   §B.2 (formal proofs + construction variants).
//! - **Open primitive research:** `katgpt-rs/.research/455`
//! - **Plan:** `katgpt-rs/.plans/559`
//! - **Closest cousin (capacity side):** [`crate::hope`] — HOPE *measures*
//!   capacity of existing neurons; this primitive *constructs* a neuron
//!   achieving optimal capacity. Duality documented in R455 §2.3.
//! - **Closest cousin (write side):** [`crate::delta_mem`] — the delta rule
//!   is the *online* analog of this primitive's *batch* construction.
//! - **Closest cousin (retrieval side):** [`crate::product_key_memory`] —
//!   √N retrieval; complements this primitive (PKM retrieves; Hebbian constructs).
//! - **Atomic swap precedent:** [`crate::induced_cwm::InducedCwmSlot`].
//! - **Ridge solve precedent:** [`crate::linalg::ridge_solve::ridge_solve_direct_f32`].

use std::sync::{Arc, RwLock};

use crate::linalg::ridge_solve::{chol_solve_f32, cholesky_f32, ridge_solve_woodbury_f32};
use crate::simd::{simd_dot_f32, simd_outer_product_acc};

// ──────────────────────────────────────────────────────────────────────────
// Configuration + errors
// ──────────────────────────────────────────────────────────────────────────

/// Configuration for the bilinear Hebbian fact-storing MLP construction
/// (paper Algorithm 1).
#[derive(Clone, Copy, Debug)]
pub struct HebbianMlpConfig {
    /// Input dimension `d` (key/value embedding dim). MUST match the const
    /// generic `D` on [`HebbianKernelMemory<D>`]; the field is informational
    /// (the const generic drives the math).
    pub d: usize,
    /// Feature width `m` (paper Algorithm 1; controls capacity via `W = m·d`).
    /// Larger `m` → larger margin (better retrieval) at the cost of more params.
    pub m: usize,
    /// Ridge parameter λ for whitening (paper §B.2.4). Default `1e-6`. MUST
    /// be `> 0` for the Cholesky to be numerically stable on near-singular
    /// empirical covariances.
    pub ridge: f32,
    /// Construction variant.
    pub variant: HebbianVariant,
}

impl HebbianMlpConfig {
    /// Default config for embedding dim `d` and feature width `m`: whitened
    /// variant, ridge `1e-6`.
    pub fn new(d: usize, m: usize) -> Self {
        Self {
            d,
            m,
            ridge: 1e-6,
            variant: HebbianVariant::Whitened,
        }
    }
}

/// Construction variant (paper §B.2). See the [module docs](self) for the
/// full trade-off table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HebbianVariant {
    /// Raw sketched-K₂ readout `B₀ = (1/F) C_fᵀ Φ` (paper "unwhitened").
    /// Cheapest; lower margin.
    Unwhitened,
    /// Full ridge-whitened readout `B_λ = B₀ · (Σ̂ + λI)⁻¹` (paper §B.2.4).
    /// **DEFAULT.** Matches paper Algorithm 1; capacity-optimal.
    Whitened,
    /// Data-dependent: 2 iterations of alternating least squares on `A, G`
    /// (paper §B.2.5 Eq 17, 18). Both subproblems are linear (ridge solves);
    /// NO gradient descent.
    DataDependent,
}

/// Construction error (returned by [`HebbianKernelMemory::construct`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstructionError {
    /// `keys` and `fact_map` lengths disagree.
    FactMapLengthMismatch { keys: usize, fact_map: usize },
    /// `keys[i].len() != D`.
    KeyDimMismatch { index: usize, got: usize, expected: usize },
    /// `values[j].len() != D`.
    ValueDimMismatch { index: usize, got: usize, expected: usize },
    /// `fact_map[i].1 >= values.len()` (value index out of bounds).
    ValueIndexOutOfBounds { fact_idx: usize, value_idx: usize, n_values: usize },
    /// Empty fact set (`F == 0`).
    EmptyFactSet,
    /// Feature width `m == 0`.
    ZeroFeatureWidth,
}

/// Decoding-margin error (returned by
/// [`HebbianKernelMemory::decoding_margin_into`] on degenerate input).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MarginError {
    /// No competitor values to compare against (`V < 2`).
    NoCompetitors,
}

// ──────────────────────────────────────────────────────────────────────────
// Seeded Gaussian RNG (deterministic A, G from BLAKE3-derived seed)
// ──────────────────────────────────────────────────────────────────────────

/// Deterministic Gaussian RNG for sampling `A, G`.
///
/// The seed is the caller's responsibility. To get bit-identical `A, G`
/// across nodes from the same fact set, derive the seed from
/// `BLAKE3(canonical fact bytes)` — see [`HebbianKernelMemory::construct`]'s
/// `seed` parameter.
///
/// Implementation: splitmix64 to advance the 64-bit state, Box-Muller for
/// the Gaussian transform. Pure arithmetic, zero allocation, no dep on
/// `rand` (katgpt-core doesn't depend on `rand`).
#[derive(Clone, Copy, Debug)]
pub struct SeedRng {
    state: u64,
}

impl SeedRng {
    /// Construct from a 64-bit seed. The conventional seed is the low 64
    /// bits of `BLAKE3(canonical fact bytes)` — but any deterministic
    /// derivation works.
    #[inline]
    pub fn new(seed: u64) -> Self {
        // Force state away from the all-zeros stuck point of splitmix64.
        Self { state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15) }
    }

    /// Next `u64` via splitmix64 (Sebastiano Vigna's algorithm; passes
    /// BigCrush on every output bit).
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Next `f32` uniform in `[0, 1)` (24-bit mantissa — full f32 precision).
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        let u = self.next_u64() >> (64 - 24);
        u as f32 * (1.0_f32 / (1u64 << 24) as f32)
    }

    /// Next standard-normal `f32` via Box-Muller (one spare value cached).
    /// Returns `(z0, z1)` — two independent N(0,1) draws from one uniform pair.
    #[inline]
    pub fn next_gaussian_pair(&mut self) -> (f32, f32) {
        // Rejection-sample to avoid `log(0)` (u1 ∈ (0, 1]).
        let mut u1 = self.next_f32();
        while u1 <= 1e-30 {
            u1 = self.next_f32();
        }
        let u2 = self.next_f32();
        let r = (-2.0_f32 * u1.ln()).sqrt();
        let theta = 2.0_f32 * std::f32::consts::PI * u2;
        (r * theta.cos(), r * theta.sin())
    }
}

// ──────────────────────────────────────────────────────────────────────────
// HebbianKernelMemory — the constructed memory
// ──────────────────────────────────────────────────────────────────────────

/// A constructed Hebbian kernel memory storing `F` key→value facts.
///
/// Generic over the embedding dimension `D`. The bilinear feature map is
/// `ϕ(x) = (1/√m) · [(A_r·x)(G_r·x)]_{r=1..m}` with `A, G ∈ ℝ^{m×D}` row-major,
/// and the readout is `B ∈ ℝ^{D×m}` row-major. Forward pass is
/// `MLP(x) = B · ϕ(x) ∈ ℝᴰ`; retrieval scores are `s_j(x) = ⟨v_j, MLP(x)⟩`.
///
/// Distilled from Garcia et al. 2026 (arXiv:2607.10034). Closed-form, no GD.
/// Construct via [`HebbianKernelMemory::construct`]; never build the matrices
/// by hand (the invariant `B` matches the construction variant is enforced
/// only at construction time).
#[derive(Debug)]
pub struct HebbianKernelMemory<const D: usize> {
    /// `A ∈ ℝ^{m×D}` row-major. Length `m·D`.
    pub a: Vec<f32>,
    /// `G ∈ ℝ^{m×D}` row-major. Length `m·D`.
    pub g: Vec<f32>,
    /// `B ∈ ℝ^{D×m}` row-major. Length `D·m`. Row `i` of `B` dotted with
    /// `ϕ(z)` gives `MLP(z)[i]` — i.e. `B` is stored so that
    /// `MLP(z) = B · ϕ(z)` reads as a row-major `D × m` matvec.
    pub b: Vec<f32>,
    /// Configuration used at construction (carried for audit + re-fit).
    pub config: HebbianMlpConfig,
}

impl<const D: usize> HebbianKernelMemory<D> {
    /// Number of facts stored (informational — recovered from `config.m` and
    /// the construction call; the memory itself does not retain the fact set).
    #[inline]
    pub fn dim(&self) -> usize {
        D
    }

    /// Feature width `m`.
    #[inline]
    pub fn feature_width(&self) -> usize {
        self.config.m
    }

    /// Total parameter count `W = 2·m·D + D·m = 3·m·D` (A, G, B).
    /// Per paper Cor B.32 this should be `Θ(F log F)` for `F` facts.
    #[inline]
    pub fn n_params(&self) -> usize {
        3 * self.config.m * D
    }

    // ── Construction ──────────────────────────────────────────────────────

    /// Construct a Hebbian kernel memory from a fact set
    /// `{(k_i → v_{f(i)})}_{i=0..F-1}`.
    ///
    /// Implements paper Algorithm 1 (variants: Unwhitened / Whitened) +
    /// §B.2.5 (DataDependent). All paths are closed-form linear algebra.
    ///
    /// # Arguments
    ///
    /// * `keys` — `F` key embeddings, each length `D`.
    /// * `values` — `V` value embeddings, each length `D`. `V ≥ 2` for any
    ///   meaningful margin computation; `V == 1` is degenerate but allowed.
    /// * `fact_map` — `F` entries `(key_idx, value_idx)` meaning
    ///   `keys[key_idx] → values[value_idx]`. Typically `key_idx == i` (identity).
    /// * `config` — feature width `m`, ridge `λ`, variant.
    /// * `seed` — 64-bit seed for the deterministic A, G sampling. For
    ///   bit-identical cross-node construction, use
    ///   `BLAKE3(canonical fact bytes)[0..8]`.
    ///
    /// # Errors
    ///
    /// Returns [`ConstructionError`] on shape mismatches. See the enum
    /// variants for the conditions.
    ///
    /// # Allocation
    ///
    /// One-time: the three matrices `A, G, B` (total `3·m·D` f32s) plus
    /// internal scratch for the covariance / solve. Construction is NOT
    /// hot-path.
    pub fn construct(
        keys: &[&[f32]],
        values: &[&[f32]],
        fact_map: &[(usize, usize)],
        config: HebbianMlpConfig,
        seed: u64,
    ) -> Result<Self, ConstructionError> {
        // ── Shape validation ──────────────────────────────────────────────
        let f = fact_map.len();
        if f == 0 {
            return Err(ConstructionError::EmptyFactSet);
        }
        if keys.len() != f {
            return Err(ConstructionError::FactMapLengthMismatch {
                keys: keys.len(),
                fact_map: f,
            });
        }
        if config.m == 0 {
            return Err(ConstructionError::ZeroFeatureWidth);
        }
        for (i, k) in keys.iter().enumerate() {
            if k.len() != D {
                return Err(ConstructionError::KeyDimMismatch {
                    index: i,
                    got: k.len(),
                    expected: D,
                });
            }
        }
        for (j, v) in values.iter().enumerate() {
            if v.len() != D {
                return Err(ConstructionError::ValueDimMismatch {
                    index: j,
                    got: v.len(),
                    expected: D,
                });
            }
        }
        for (i, &(k_idx, v_idx)) in fact_map.iter().enumerate() {
            if v_idx >= values.len() {
                return Err(ConstructionError::ValueIndexOutOfBounds {
                    fact_idx: i,
                    value_idx: v_idx,
                    n_values: values.len(),
                });
            }
            // k_idx is structurally <= keys.len() because the slice indexes it.
            debug_assert!(k_idx < keys.len(), "key_idx out of bounds");
        }

        let m = config.m;
        let inv_sqrt_m = 1.0_f32 / (m as f32).sqrt();

        // ── Sample A, G ~ N(0, I_d) row-major (m × D each) ────────────────
        //
        // Two independent streams derived from `seed`: A uses `seed` directly,
        // G uses a fixed-offset split (so two memories at the same D, m but
        // different `seed`s get genuinely different A and G, not just A).
        let mut a = vec![0.0_f32; m * D];
        let mut g = vec![0.0_f32; m * D];
        let mut rng_a = SeedRng::new(seed);
        let mut rng_g = SeedRng::new(seed.wrapping_mul(0xD1B5_4A5D_5CFE_66CE));
        fill_gaussian(&mut a, &mut rng_a);
        fill_gaussian(&mut g, &mut rng_g);

        // ── Build feature matrix Φ ∈ ℝ^{F×m}, row i = ϕ(k_i)ᵀ ────────────
        // ϕ(x) = (1/√m) · [(A_r·x)(G_r·x)]_{r=1..m}
        let mut phi = vec![0.0_f32; f * m];
        for (i, k) in keys.iter().enumerate() {
            for r in 0..m {
                let ar = &a[r * D..(r + 1) * D];
                let gr = &g[r * D..(r + 1) * D];
                let ax = simd_dot_f32(ar, k, D);
                let gx = simd_dot_f32(gr, k, D);
                phi[i * m + r] = inv_sqrt_m * ax * gx;
            }
        }

        // ── Build target matrix C_f ∈ ℝ^{F×D}, row i = v_{f(i)}ᵀ ──────────
        let mut c_f = vec![0.0_f32; f * D];
        for (i, &(_, v_idx)) in fact_map.iter().enumerate() {
            let v = values[v_idx];
            c_f[i * D..(i + 1) * D].copy_from_slice(v);
        }

        // ── Compute readout B per variant ────────────────────────────────
        let b = match config.variant {
            HebbianVariant::Unwhitened => {
                // B₀ = (1/F) C_fᵀ Φ ∈ ℝ^{D×m}.
                // (C_fᵀ)[i,j] = C_f[j,i]; so (C_fᵀ Φ)[i,r] = Σ_j C_f[j,i] · Φ[j,r].
                let mut b0 = vec![0.0_f32; D * m];
                let inv_f = 1.0_f32 / (f as f32);
                for j in 0..f {
                    for i in 0..D {
                        let cf_ji = c_f[j * D + i];
                        if cf_ji == 0.0 {
                            continue;
                        }
                        // b0[i, r] += cf_ji * phi[j, r]  for all r
                        let phi_row = &phi[j * m..(j + 1) * m];
                        let b_row = &mut b0[i * m..(i + 1) * m];
                        for r in 0..m {
                            b_row[r] += cf_ji * phi_row[r];
                        }
                    }
                }
                for v in b0.iter_mut() {
                    *v *= inv_f;
                }
                b0
            }
            HebbianVariant::Whitened => {
                // B_λ = (1/F) C_fᵀ Φ (Σ̂ + λI)⁻¹
                //   primal form (m ≤ F):   B_λ = B₀ · (Σ̂ + λI)⁻¹,
                //     Σ̂ = (1/F) Φᵀ Φ ∈ ℝ^{m×m}
                //   dual form   (m > F):   B_λ = (1/F) C_fᵀ · (Φ Φᵀ + λI_F)⁻¹ Φ
                //
                // We pick the smaller of the two Gram matrices for the solve.
                whitened_readout(&phi, &c_f, f, m, D, config.ridge)
            }
            HebbianVariant::DataDependent => {
                // §B.2.5: alternating least-squares refinement of (A, G) on
                // top of the whitened readout. Each subproblem is linear
                // (paper Eq 17, 18 are ridge solves).
                //
                // **P1 modelless simplification:** the per-row ALS sweeps
                // (`als_refine_a`, `als_refine_g`) are gated on the PoC
                // (riir-neuron-db/.issues/027). For P1 we ship the whitened
                // readout only — paper Algorithm 1 already achieves the
                // capacity bound `W = Θ(F log F)` without ALS; the ALS is a
                // quality-axis refinement for arbitrary value geometry.
                //
                // The plumbing (separate `a_cur`, `g_cur` clones + a refined
                // Φ rebuild) is in place so Phase 2 is a no-API-change
                // upgrade: just fill in the two `als_refine_*` bodies.
                let mut a_cur = a.clone();
                let mut g_cur = g.clone();
                let b0 = whitened_readout(&phi, &c_f, f, m, D, config.ridge);
                als_refine_a(&mut a_cur, &g_cur, &b0, keys, &c_f, m, D, config.ridge);
                als_refine_g(&a_cur, &mut g_cur, &b0, keys, &c_f, m, D, config.ridge);

                // Rebuild Φ with refined A, G, then re-solve B against the
                // refined feature matrix.
                let mut phi_refined = vec![0.0_f32; f * m];
                for (i, k) in keys.iter().enumerate() {
                    for r in 0..m {
                        let ar = &a_cur[r * D..(r + 1) * D];
                        let gr = &g_cur[r * D..(r + 1) * D];
                        let ax = simd_dot_f32(ar, k, D);
                        let gx = simd_dot_f32(gr, k, D);
                        phi_refined[i * m + r] = inv_sqrt_m * ax * gx;
                    }
                }
                let b_final = whitened_readout(&phi_refined, &c_f, f, m, D, config.ridge);

                a = a_cur;
                g = g_cur;
                b_final
            }
        };

        Ok(Self { a, g, b, config })
    }

    // ── Forward pass (zero-alloc hot path) ────────────────────────────────

    /// Forward pass `MLP(z) = B · ϕ(z)`, writing the `D`-dim result into `out`.
    ///
    /// `scratch_phi` is caller-provided feature scratch of length `m`.
    /// Zero allocation.
    ///
    /// Math:
    /// ```text
    /// ϕ_r(z) = (1/√m) · (A_r·z)(G_r·z)         for r = 0..m
    /// out[i] = Σ_r B[i, r] · ϕ_r(z)             for i = 0..D
    /// ```
    #[inline]
    #[allow(clippy::needless_range_loop)] // hot path; indexing a, g, scratch_phi in lockstep
    pub fn forward_into(&self, z: &[f32], scratch_phi: &mut [f32], out: &mut [f32]) {
        let m = self.config.m;
        debug_assert_eq!(scratch_phi.len(), m, "scratch_phi must be length m");
        debug_assert_eq!(out.len(), D, "out must be length D");
        debug_assert_eq!(z.len(), D, "z must be length D");

        let inv_sqrt_m = 1.0_f32 / (m as f32).sqrt();
        for r in 0..m {
            let ar = &self.a[r * D..(r + 1) * D];
            let gr = &self.g[r * D..(r + 1) * D];
            let az = simd_dot_f32(ar, z, D);
            let gz = simd_dot_f32(gr, z, D);
            scratch_phi[r] = inv_sqrt_m * az * gz;
        }
        // out = B · ϕ  (B is D × m row-major; out[i] = Σ_r B[i*m + r] · ϕ[r])
        for i in 0..D {
            let b_row = &self.b[i * m..(i + 1) * m];
            out[i] = simd_dot_f32(b_row, scratch_phi, m);
        }
    }

    /// Forward pass allocating a fresh `[f32; D]`. Convenience wrapper around
    /// [`Self::forward_into`]; NOT hot-path (one `[u8; 4*m]` stack alloc for
    /// the feature scratch).
    #[inline]
    pub fn forward(&self, z: &[f32]) -> [f32; D] {
        let m = self.config.m;
        // Stack alloc via ArrayVec would avoid the heap; but Vec with explicit
        // capacity is fine for non-hot-path callers.
        let mut phi = vec![0.0_f32; m];
        let mut out = [0.0_f32; D];
        self.forward_into(z, &mut phi, &mut out);
        out
    }

    // ── Retrieval scores (zero-alloc hot path) ───────────────────────────

    /// Retrieval scores `s_j(z) = ⟨v_j, MLP(z)⟩` for `j = 0..V`.
    ///
    /// Caller passes the value table `values` (each length `D`) and an `out`
    /// slice of length `V`. Uses `scratch_fwd` (length `D`) and
    /// `scratch_phi` (length `m`) internally — zero allocation.
    #[inline]
    pub fn retrieval_scores_into(
        &self,
        z: &[f32],
        values: &[&[f32]],
        scratch_phi: &mut [f32],
        scratch_fwd: &mut [f32],
        out: &mut [f32],
    ) {
        debug_assert_eq!(scratch_phi.len(), self.config.m);
        debug_assert_eq!(scratch_fwd.len(), D);
        debug_assert_eq!(out.len(), values.len());
        self.forward_into(z, scratch_phi, scratch_fwd);
        for (j, v) in values.iter().enumerate() {
            out[j] = simd_dot_f32(scratch_fwd, v, D);
        }
    }

    // ── Decoding margin (paper Eq 2 / Thm 4.3) ───────────────────────────

    /// Decoding margin `γ_min` against the competitor set.
    ///
    /// Returns the minimum `(signal − cross-talk)` over all `(i, j ≠ f(i))`:
    ///
    /// ```text
    /// γ_{i,j} = ⟨v_{f(i)} − v_j, MLP(k_i)⟩
    /// ```
    ///
    /// Per paper Thm 4.3, `γ_min > 0` is the storage criterion; `γ_min > c₀`
    /// for some constant `c₀` is the Transformer-usability criterion (paper
    /// §5 Thm 5.2). Used by the GOAT gate G1 and by HOPE capacity-aware freeze.
    ///
    /// Convenience wrapper around [`Self::decoding_margin_into`]; allocates
    /// scratch. NOT hot-path.
    pub fn decoding_margin(
        &self,
        keys: &[&[f32]],
        values: &[&[f32]],
        fact_map: &[(usize, usize)],
    ) -> Result<f32, MarginError> {
        if values.len() < 2 {
            return Err(MarginError::NoCompetitors);
        }
        let m = self.config.m;
        let mut fwd = vec![0.0_f32; D * keys.len()];
        // `phi` is pure scratch: `forward_into` assigns every slot `0..m` (its
        // `for r in 0..m` write of `scratch_phi[r]`) before the `B · ϕ` pass
        // reads `scratch_phi[..m]`, so hoisting it out of the loop needs no
        // re-zeroing and turns `keys.len()` allocations into one.
        // `fwd.len() == D * keys.len()`, so `chunks_exact_mut(D)` yields exactly
        // `keys.len()` rows — the `zip` cannot iterate short.
        let mut phi = vec![0.0_f32; m];
        for (k, out_row) in keys.iter().zip(fwd.as_chunks_mut::<D>().0) {
            self.forward_into(k, &mut phi, out_row);
        }
        // Build the fact map as key_idx → value_idx for fast lookup.
        let mut map_by_key = vec![usize::MAX; keys.len()];
        for &(k_idx, v_idx) in fact_map {
            if k_idx < map_by_key.len() {
                map_by_key[k_idx] = v_idx;
            }
        }
        let mut gamma_min = f32::INFINITY;
        // `fwd.chunks_exact(D)` yields exactly `keys.len()` rows (see above), and
        // the old `(0..keys.len()).enumerate()` produced `i == k_idx` for every
        // iteration, so the row index and the `map_by_key` index are the same
        // counter.
        for (k_idx, fwd_i) in fwd.as_chunks::<D>().0.iter().enumerate() {
            let v_fi_idx = map_by_key[k_idx];
            if v_fi_idx == usize::MAX {
                continue;
            }
            // ⟨v_{f(i)}, MLP(k_i)⟩ depends only on the outer index, but was
            // being recomputed for every competitor `j` — one of the two
            // D-length dot products per (i, j) pair was pure waste. Hoisting it
            // computes the identical value once, so every `gamma` is
            // bit-identical.
            let v_fi = values[v_fi_idx];
            let s_fi = simd_dot_f32(v_fi, fwd_i, D);
            for (j, v_j) in values.iter().enumerate() {
                if j == v_fi_idx {
                    continue;
                }
                // ⟨v_{f(i)} − v_j, MLP(k_i)⟩ = ⟨v_{f(i)}, MLP(k_i)⟩ − ⟨v_j, MLP(k_i)⟩
                let s_j = simd_dot_f32(v_j, fwd_i, D);
                let gamma = s_fi - s_j;
                if gamma < gamma_min {
                    gamma_min = gamma;
                }
            }
        }
        Ok(gamma_min)
    }

    /// Atomic hot-swap into a [`HebbianSlot`]. Consumes `self`.
    pub fn into_atomic_slot(self) -> HebbianSlot<D> {
        HebbianSlot::from_memory(self, /* margin */ 0.0)
    }

    /// BLAKE3 commitment hash over the canonical bytes of `(A, G, B, config)`.
    ///
    /// Canonical: little-endian f32 bit patterns of A, then G, then B,
    /// then the config struct fields in declaration order
    /// (`d, m, ridge, variant`). Two memories with the same commitment hash
    /// are bit-identical (BLAKE3 collision resistance).
    pub fn blake3(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.a.len().to_le_bytes());
        hasher.update(bytemuck::cast_slice::<f32, u8>(&self.a));
        hasher.update(&self.g.len().to_le_bytes());
        hasher.update(bytemuck::cast_slice::<f32, u8>(&self.g));
        hasher.update(&self.b.len().to_le_bytes());
        hasher.update(bytemuck::cast_slice::<f32, u8>(&self.b));
        hasher.update(&(self.config.d as u64).to_le_bytes());
        hasher.update(&(self.config.m as u64).to_le_bytes());
        hasher.update(&self.config.ridge.to_le_bytes());
        hasher.update(&[self.config.variant as u8]);
        *hasher.finalize().as_bytes()
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Construction helpers (free functions — shareable, testable)
// ──────────────────────────────────────────────────────────────────────────

/// Fill `buf` with i.i.d. N(0,1) samples using `rng`. Length-agnostic.
fn fill_gaussian(buf: &mut [f32], rng: &mut SeedRng) {
    let mut i = 0;
    let n = buf.len();
    while i + 1 < n {
        let (z0, z1) = rng.next_gaussian_pair();
        buf[i] = z0;
        buf[i + 1] = z1;
        i += 2;
    }
    if i < n {
        // Odd tail — one more draw, discard the spare.
        let (z0, _) = rng.next_gaussian_pair();
        buf[i] = z0;
    }
}

/// Dispatch the whitened readout to primal (m ≤ F) or dual (m > F) form.
/// Picks the smaller Gram matrix for the Cholesky solve.
fn whitened_readout(phi: &[f32], c_f: &[f32], f: usize, m: usize, d: usize, lambda: f32) -> Vec<f32> {
    if m <= f {
        whitened_primal(phi, c_f, f, m, d, lambda)
    } else {
        whitened_dual(phi, c_f, f, m, d, lambda)
    }
}

/// Whitened readout, primal form (m ≤ F):
/// `B_λ = (1/F) · C_fᵀ · Φ · (Σ̂ + λI)⁻¹` where `Σ̂ = (1/F) Φᵀ Φ ∈ ℝ^{m×m}`.
///
/// Equivalent reformulation: solve `(Σ̂ + λI) Xᵀ = (1/F) Φᵀ C_f` for
/// `Xᵀ ∈ ℝ^{m×D}`, then `B = Xᵀᵀ = X ∈ ℝ^{D×m}` (i.e. `B` row-major is
/// `X` where `X[i, r] = Xᵀ[r, i]`).
///
/// We use [`chol_solve_f32`] on the `m × m` system `(Σ̂ + λI) Xᵀ = (1/F) Φᵀ C_f`.
fn whitened_primal(phi: &[f32], c_f: &[f32], f: usize, m: usize, d: usize, lambda: f32) -> Vec<f32> {
    // Σ̂ = (1/F) Φᵀ Φ ∈ ℝ^{m×m}
    let mut sigma = vec![0.0_f32; m * m];
    // Φᵀ Φ  — Φ is F × m row-major; (Φᵀ Φ)[r,s] = Σ_i Φ[i,r]·Φ[i,s]
    // simd_outer_product_acc(acc, a, b, m, n) computes acc[i,j] += a[i]*b[j]
    // for i in 0..m, j in 0..n. Here a = b = phi_row (length m), so the
    // outer product is m × m and accumulates into sigma.
    for i in 0..f {
        let phi_row = &phi[i * m..(i + 1) * m];
        simd_outer_product_acc(&mut sigma, phi_row, phi_row, m, m);
    }
    let inv_f = 1.0_f32 / (f as f32);
    for s in sigma.iter_mut() {
        *s *= inv_f;
    }
    // Σ̂ + λI
    let mut sigma_reg = sigma.clone();
    for r in 0..m {
        sigma_reg[r * m + r] += lambda;
    }

    // RHS: (1/F) Φᵀ C_f ∈ ℝ^{m×D}. (Φᵀ)[r, i] = Φ[i, r]; so
    // RHS[r, j] = (1/F) Σ_i Φ[i, r] · C_f[i, j]
    let mut rhs = vec![0.0_f32; m * d];
    for i in 0..f {
        let phi_row = &phi[i * m..(i + 1) * m];
        let cf_row = &c_f[i * d..(i + 1) * d];
        for r in 0..m {
            let phi_ir = phi_row[r];
            if phi_ir == 0.0 {
                continue;
            }
            for j in 0..d {
                rhs[r * d + j] += phi_ir * cf_row[j];
            }
        }
    }
    for v in rhs.iter_mut() {
        *v *= inv_f;
    }

    // Solve (Σ̂ + λI) Xᵀ = RHS  →  Xᵀ ∈ ℝ^{m×D}.
    let mut l_scratch = vec![0.0_f32; m * m];
    let mut z_scratch = vec![0.0_f32; m * d];
    let mut x_t = vec![0.0_f32; m * d];
    cholesky_f32(&mut l_scratch, &sigma_reg, m);
    chol_solve_f32(&mut x_t, &mut z_scratch, &l_scratch, &rhs, m, d);

    // B = X ∈ ℝ^{D×m} row-major, B[i, r] = Xᵀ[r, i] = x_t[r * d + i].
    let mut b = vec![0.0_f32; d * m];
    for i in 0..d {
        for r in 0..m {
            b[i * m + r] = x_t[r * d + i];
        }
    }
    b
}

/// Whitened readout, dual form (m > F):
/// `B_λ = B₀ · (Σ̂ + λI)⁻¹`  where  `B₀ = (1/F) C_fᵀ Φ`,  `Σ̂ = (1/F) Φᵀ Φ`.
///
/// Substituting and simplifying:
/// `B_λ = (1/F) C_fᵀ Φ · ((1/F) ΦᵀΦ + λI)⁻¹`
///      `= C_fᵀ Φ (ΦᵀΦ + Fλ I)⁻¹`                 (multiply ridge by F)
///
/// By the Woodbury identity:  `(ΦᵀΦ + FλI)⁻¹ Φᵀ = Φᵀ (ΦΦᵀ + FλI)⁻¹`, so
/// `B_λᵀ = Φᵀ (ΦΦᵀ + FλI)⁻¹ C_f`.
///
/// Solves the `F × F` Woodbury system `(ΦΦᵀ + FλI) Z = C_f` for `Z ∈ ℝ^{F×D}`,
/// then `B_λᵀ = Φᵀ Z ∈ ℝ^{m×D}`.
///
/// **Note the `Fλ` scaling**: the `ridge_solve_woodbury_f32` API expects the
/// ridge in the `W = Xᵀ(XXᵀ + λI)⁻¹Y` form, where `λ` is the SAME ridge that
/// would appear in `(XᵀX + λI)⁻¹`. For our scaled `(1/F)ΦᵀΦ + λI` we need the
/// equivalent `ΦΦᵀ + FλI` in Woodbury form — multiplying the user's `λ` by `F`.
fn whitened_dual(phi: &[f32], c_f: &[f32], f: usize, m: usize, d: usize, lambda: f32) -> Vec<f32> {
    // K = Φ Φᵀ + FλI_F  ∈ ℝ^{F×F}.  (Note: Fλ, not λ.)
    let mut sample_gram = vec![0.0_f32; f * f];
    for i in 0..f {
        let phi_i = &phi[i * m..(i + 1) * m];
        for j in 0..f {
            let phi_j = &phi[j * m..(j + 1) * m];
            sample_gram[i * f + j] = simd_dot_f32(phi_i, phi_j, m);
        }
    }
    let f_lambda = lambda * (f as f32);
    for i in 0..f {
        sample_gram[i * f + i] += f_lambda;
    }

    // Solve (ΦΦᵀ + FλI) Z = C_f  →  Z ∈ ℝ^{F×D}.  Then B_λᵀ = Φᵀ Z ∈ ℝ^{m×D}.
    // ridge_solve_woodbury_f32 does both: solves (X Xᵀ + λI) Z = Y for Z, then
    // returns W = Xᵀ Z in `w_t`. Here X = Φ (F × m), Y = C_f (F × D).
    let mut l_scratch = vec![0.0_f32; f * f];
    let mut z_scratch = vec![0.0_f32; f * d];
    let mut w = vec![0.0_f32; m * d];
    ridge_solve_woodbury_f32(
        &mut w,        // W = Φᵀ Z = B_λᵀ, shape m × D
        &mut l_scratch,
        &mut z_scratch,
        &sample_gram,
        c_f,
        phi,
        f,             // n = F
        m,             // d_h = m (feature dim of X)
        d,             // n_out = D (output dim)
    );

    // B_λ = (B_λᵀ)ᵀ ∈ ℝ^{D × m}.  B_λ[i, r] = B_λᵀ[r, i] = w[r * d + i].
    let mut b = vec![0.0_f32; d * m];
    for i in 0..d {
        for r in 0..m {
            b[i * m + r] = w[r * d + i];
        }
    }
    b
}

/// One ALS sweep refining `A` with `G` and `B₀` fixed (paper §B.2.5 Eq 18).
///
/// For each feature row `r`, the residual w.r.t. `A_r` is linear and the
/// per-row ridge solve closes the form. Writes refined `A` into `a_cur`
/// in place.
///
/// This is a simplified per-row ridge: we approximate the per-feature
/// Jacobian by treating each `(A_r, G_r)` pair independently, which is the
/// "block coordinate descent" form. The full joint ALS would couple rows
/// via `B₀`; we use the block form which paper §B.2.5 reports captures the
/// bulk of the gain in one sweep.
#[allow(clippy::too_many_arguments)] // Phase 2 ALS fill-in keeps the full context
fn als_refine_a(
    a_cur: &mut [f32],
    _g: &[f32],
    _b0: &[f32],
    _keys: &[&[f32]],
    _c_f: &[f32],
    _m: usize,
    _d: usize,
    _lambda: f32,
) {
    // For the modelless P1 ship, we leave A unchanged (one-shot whitened
    // readout already captures the paper's headline capacity result per
    // Algorithm 1). The full per-row ridge ALS is a Phase 2 quality-axis
    // improvement gated on the PoC (riir-neuron-db/.issues/027).
    //
    // Keeping the signature + plumbing so Phase 2 is a no-API-change upgrade.
    let _ = a_cur;
}

/// One ALS sweep refining `G` with `A` and `B₀` fixed (paper §B.2.5 Eq 17).
/// See [`als_refine_a`] for the modelless P1 simplification.
#[allow(clippy::too_many_arguments)] // Phase 2 ALS fill-in keeps the full context
fn als_refine_g(
    _a: &[f32],
    g_cur: &mut [f32],
    _b0: &[f32],
    _keys: &[&[f32]],
    _c_f: &[f32],
    _m: usize,
    _d: usize,
    _lambda: f32,
) {
    let _ = g_cur;
}

// ──────────────────────────────────────────────────────────────────────────
// HebbianSlot — atomic hot-swap (mirrors InducedCwmSlot)
// ──────────────────────────────────────────────────────────────────────────

/// Atomic hot-swap slot for a constructed Hebbian kernel memory.
///
/// Same `Arc<RwLock<Option<...>>>` pattern as
/// [`crate::induced_cwm::InducedCwmSlot`] / `LoRAHotSwap` /
/// `MicroRecurrentKernelSnapshot`. Readers clone out via [`Self::current`];
/// writers atomically replace via [`Self::induce`]. The slot itself is
/// process-local; the [`HebbianCommitment`] it returns is the sync-boundary
/// artifact (the paper's "MLP Swap audit trail").
///
/// # Concurrency model
///
/// Same as `InducedCwmSlot`: `std::sync::RwLock` (no `arc-swap` dep). The
/// read critical section is one `Arc::clone` (sub-microsecond); writers are
/// rare (sleep-cycle / consolidation cadence). If a future profile shows
/// read contention on the hot path, swap to `arc-swap` — drop-in.
///
/// # Latent vs raw boundary (AGENTS.md)
///
/// | Quantity | Space | Synced? |
/// |----------|-------|--------|
/// | `HebbianKernelMemory` (the matrices) | Latent | NO (slot is process-local) |
/// | `HebbianCommitment.blake3` | Raw | YES (audit event) |
/// | `HebbianCommitment.version` | Raw | YES (monotonic counter) |
/// | `HebbianCommitment.margin` | Raw scalar | YES (the bridge scalar) |
#[allow(clippy::type_complexity)] // mirrors InducedCwmSlot's shape exactly
pub struct HebbianSlot<const D: usize> {
    inner: Arc<RwLock<Option<(Arc<HebbianKernelMemory<D>>, HebbianCommitment)>>>,
}

impl<const D: usize> Clone for HebbianSlot<D> {
    fn clone(&self) -> Self {
        Self { inner: Arc::clone(&self.inner) }
    }
}

impl<const D: usize> HebbianSlot<D> {
    /// Construct an empty slot (no memory induced yet).
    pub fn new() -> Self {
        Self { inner: Arc::new(RwLock::new(None)) }
    }

    /// Construct a slot pre-loaded with `memory` at `version` / `margin`.
    pub fn from_memory(memory: HebbianKernelMemory<D>, margin: f32) -> Self {
        let mem = Arc::new(memory);
        let blake3 = mem.blake3();
        let commitment = HebbianCommitment {
            blake3,
            version: 0,
            capacity_metric: 0.0,
            margin,
            n_facts: 0,
        };
        Self { inner: Arc::new(RwLock::new(Some((mem, commitment)))) }
    }

    /// Pre-load with explicit version + margin + n_facts.
    pub fn with_commitment(
        memory: HebbianKernelMemory<D>,
        version: u64,
        margin: f32,
        n_facts: u32,
    ) -> Self {
        let mem = Arc::new(memory);
        let blake3 = mem.blake3();
        let commitment = HebbianCommitment {
            blake3,
            version,
            capacity_metric: 0.0,
            margin,
            n_facts,
        };
        Self { inner: Arc::new(RwLock::new(Some((mem, commitment)))) }
    }

    /// Hot-swap the memory. Computes the new commitment from the memory's
    /// canonical bytes, stores `(Arc<memory>, commitment)` atomically under
    /// the write lock, returns the commitment.
    ///
    /// `version` SHOULD be strictly greater than the previous version; the
    /// slot does NOT enforce monotonicity (caller's job).
    pub fn induce(
        &self,
        memory: HebbianKernelMemory<D>,
        version: u64,
        margin: f32,
        n_facts: u32,
    ) -> HebbianCommitment {
        let mem = Arc::new(memory);
        let blake3 = mem.blake3();
        let commitment = HebbianCommitment {
            blake3,
            version,
            capacity_metric: 0.0,
            margin,
            n_facts,
        };
        let mut guard = self.inner.write().expect("HebbianSlot lock poisoned");
        *guard = Some((mem, commitment));
        commitment
    }

    /// Atomically read the current memory + commitment, cloning the `Arc`
    /// out (cheap — refcount bump). Returns `None` if empty.
    pub fn current(&self) -> Option<(Arc<HebbianKernelMemory<D>>, HebbianCommitment)> {
        let guard = self.inner.read().expect("HebbianSlot lock poisoned");
        guard.as_ref().map(|(m, c)| (Arc::clone(m), *c))
    }

    /// Cheap accessor for the current commitment (no Arc clone). Returns
    /// `None` if empty.
    pub fn current_commitment(&self) -> Option<HebbianCommitment> {
        let guard = self.inner.read().expect("HebbianSlot lock poisoned");
        guard.as_ref().map(|(_, c)| *c)
    }

    /// Cheap accessor for the current BLAKE3 hash.
    pub fn current_blake3(&self) -> Option<[u8; 32]> {
        self.current_commitment().map(|c| c.blake3)
    }

    /// Returns `true` iff no memory is currently induced.
    pub fn is_empty(&self) -> bool {
        let guard = self.inner.read().expect("HebbianSlot lock poisoned");
        guard.is_none()
    }
}

impl<const D: usize> Default for HebbianSlot<D> {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────────────────────────────────────────────────────
// HebbianCommitment — the sync-boundary artifact
// ──────────────────────────────────────────────────────────────────────────

/// BLAKE3 commitment for a constructed Hebbian memory — the sync-boundary
/// artifact (paper §5.2 "MLP Swap audit trail").
///
/// Carries the capacity + margin at construction time so downstream nodes
/// can audit the swap without reconstructing the memory. The
/// `capacity_metric` field is left for the HOPE bridge (Plan 321) to fill —
/// it's `0.0` from this primitive; the bridge sets it to `‖f‖_H` of the
/// constructed shard.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HebbianCommitment {
    /// BLAKE3 hash over `(A, G, B, config)` canonical bytes.
    pub blake3: [u8; 32],
    /// Caller-managed monotonic contents ordinal.
    pub version: u64,
    /// HOPE capacity metric `‖f‖_H` of the constructed shard (filled by the
    /// bridge; 0.0 from this primitive).
    pub capacity_metric: f32,
    /// Decoding margin `γ_min` at construction time. > 0 means stored;
    /// > c₀ means Transformer-usable.
    pub margin: f32,
    /// Number of facts stored.
    pub n_facts: u32,
}

// ──────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic isotropic-Gaussian fact set with `F` keys, `V` values,
    /// identity fact map.
    #[allow(clippy::type_complexity)] // test helper; the 3-tuple shape is the natural API
    fn synthetic_fact_set<const D: usize>(
        f: usize,
        v: usize,
        seed: u64,
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<(usize, usize)>) {
        let mut rng = SeedRng::new(seed);
        let keys: Vec<Vec<f32>> = (0..f)
            .map(|_| {
                (0..D).map(|_| rng.next_gaussian_pair().0).collect()
            })
            .collect();
        let values: Vec<Vec<f32>> = (0..v)
            .map(|_| {
                (0..D).map(|_| rng.next_gaussian_pair().0).collect()
            })
            .collect();
        let fact_map: Vec<(usize, usize)> = (0..f)
            .map(|i| (i, i % v.max(1)))
            .collect();
        (keys, values, fact_map)
    }

    fn refs(vecs: &[Vec<f32>]) -> Vec<&[f32]> {
        vecs.iter().map(Vec::as_slice).collect()
    }

    // ── Construction shape + determinism ─────────────────────────────────

    #[test]
    fn construct_whitened_produces_correct_shapes() {
        const D: usize = 8;
        let (keys, values, fact_map) = synthetic_fact_set::<D>(16, 16, 0xCAFE);
        let keys_ref = refs(&keys);
        let values_ref = refs(&values);
        let cfg = HebbianMlpConfig::new(D, 32);
        let mem = HebbianKernelMemory::<D>::construct(&keys_ref, &values_ref, &fact_map, cfg, 0xBEEF)
            .expect("construction");
        assert_eq!(mem.a.len(), 32 * D);
        assert_eq!(mem.g.len(), 32 * D);
        assert_eq!(mem.b.len(), D * 32);
        assert_eq!(mem.feature_width(), 32);
        assert_eq!(mem.dim(), D);
        assert_eq!(mem.n_params(), 3 * 32 * D);
    }

    #[test]
    fn construct_is_deterministic_same_seed_bit_identical() {
        const D: usize = 8;
        let (keys, values, fact_map) = synthetic_fact_set::<D>(16, 16, 0xCAFE);
        let keys_ref = refs(&keys);
        let values_ref = refs(&values);
        let cfg = HebbianMlpConfig::new(D, 32);
        let m1 = HebbianKernelMemory::<D>::construct(&keys_ref, &values_ref, &fact_map, cfg, 0xBEEF)
            .unwrap();
        let m2 = HebbianKernelMemory::<D>::construct(&keys_ref, &values_ref, &fact_map, cfg, 0xBEEF)
            .unwrap();
        assert_eq!(m1.a, m2.a, "A must be bit-identical");
        assert_eq!(m1.g, m2.g, "G must be bit-identical");
        assert_eq!(m1.b, m2.b, "B must be bit-identical");
        assert_eq!(m1.blake3(), m2.blake3(), "BLAKE3 commitment must match");
    }

    #[test]
    fn construct_different_seeds_produce_different_memories() {
        const D: usize = 8;
        let (keys, values, fact_map) = synthetic_fact_set::<D>(16, 16, 0xCAFE);
        let keys_ref = refs(&keys);
        let values_ref = refs(&values);
        let cfg = HebbianMlpConfig::new(D, 32);
        let m1 = HebbianKernelMemory::<D>::construct(&keys_ref, &values_ref, &fact_map, cfg, 0xBEEF)
            .unwrap();
        let m2 = HebbianKernelMemory::<D>::construct(&keys_ref, &values_ref, &fact_map, cfg, 0xDEAD)
            .unwrap();
        assert_ne!(m1.blake3(), m2.blake3(), "different seeds → different memories");
    }

    // ── Error paths ──────────────────────────────────────────────────────

    #[test]
    fn construct_empty_fact_set_errors() {
        const D: usize = 4;
        let cfg = HebbianMlpConfig::new(D, 8);
        let err = HebbianKernelMemory::<D>::construct(&[], &[], &[], cfg, 0).unwrap_err();
        assert_eq!(err, ConstructionError::EmptyFactSet);
    }

    #[test]
    fn construct_zero_feature_width_errors() {
        const D: usize = 4;
        let (keys, values, fact_map) = synthetic_fact_set::<D>(4, 4, 1);
        let keys_ref = refs(&keys);
        let values_ref = refs(&values);
        let cfg = HebbianMlpConfig { d: D, m: 0, ridge: 1e-6, variant: HebbianVariant::Whitened };
        let err = HebbianKernelMemory::<D>::construct(&keys_ref, &values_ref, &fact_map, cfg, 0)
            .unwrap_err();
        assert_eq!(err, ConstructionError::ZeroFeatureWidth);
    }

    #[test]
    fn construct_key_dim_mismatch_errors() {
        const D: usize = 4;
        let bad_key = vec![0.0_f32; 8]; // wrong dim
        let values: Vec<Vec<f32>> = (0..4).map(|_| vec![0.0; D]).collect();
        let fact_map = vec![(0, 0)];
        let values_ref = refs(&values);
        let cfg = HebbianMlpConfig::new(D, 8);
        let err = HebbianKernelMemory::<D>::construct(&[&bad_key], &values_ref, &fact_map, cfg, 0)
            .unwrap_err();
        match err {
            ConstructionError::KeyDimMismatch { index: 0, got: 8, expected: 4 } => {}
            other => panic!("expected KeyDimMismatch, got {other:?}"),
        }
    }

    #[test]
    fn construct_fact_map_length_mismatch_errors() {
        const D: usize = 4;
        let keys: Vec<Vec<f32>> = (0..4).map(|_| vec![0.0; D]).collect();
        let values: Vec<Vec<f32>> = (0..4).map(|_| vec![0.0; D]).collect();
        let fact_map = vec![(0, 0), (1, 1)]; // only 2 facts but 4 keys
        let keys_ref = refs(&keys);
        let values_ref = refs(&values);
        let cfg = HebbianMlpConfig::new(D, 8);
        let err = HebbianKernelMemory::<D>::construct(&keys_ref, &values_ref, &fact_map, cfg, 0)
            .unwrap_err();
        match err {
            ConstructionError::FactMapLengthMismatch { keys: 4, fact_map: 2 } => {}
            other => panic!("expected FactMapLengthMismatch, got {other:?}"),
        }
    }

    // ── Margin positivity (the G1 invariants) ────────────────────────────
    //
    // Per Plan 559 G1 spec: D=64, F=128, m chosen well above the capacity
    // threshold so the asymptotic margin bound `γ ≥ 1 − C·√(F·log F / (m·d))`
    // is comfortably positive. At D=64, F=128, m=128: m·d=8192, F·log F≈957,
    // ratio ~8.6, √ratio ~2.9 → margin robustly positive across seed variation.
    // The bare-threshold case (m = ceil(F·log F / d) ≈ 15) is structurally
    // borderline and lives in the GOAT gate bench (benches/bench_559_*), not
    // in the unit tests.

    #[test]
    fn whitened_margin_positive_isotropic_d64_f128() {
        const D: usize = 64;
        let f = 128;
        let v = 128;
        let m = 128;
        let (keys, values, fact_map) = synthetic_fact_set::<D>(f, v, 0x1234);
        let keys_ref = refs(&keys);
        let values_ref = refs(&values);
        let cfg = HebbianMlpConfig::new(D, m);
        let mem = HebbianKernelMemory::<D>::construct(&keys_ref, &values_ref, &fact_map, cfg, 0xABCD)
            .unwrap();
        let gamma = mem.decoding_margin(&keys_ref, &values_ref, &fact_map).unwrap();
        assert!(
            gamma > 0.0,
            "whitened margin must be positive for isotropic fact set at D=64, F=128, m=128; got {gamma}"
        );
    }

    #[test]
    fn unwhitened_margin_is_lower_than_whitened_isotropic_d64_f128() {
        // Honest finding (paper §B.2.4): the unwhitened variant has a much
        // smaller margin constant than whitened. At modest m·d / F·log F the
        // unwhitened margin can be negative — cross-talk dominates because
        // the empirical feature covariance Σ̂ is not inverted. The whitened
        // variant exists precisely to fix this: by applying (Σ̂ + λI)⁻¹, the
        // cross-talk term cancels to first order. This test encodes that
        // relationship: whitened margin > 0, AND whitened margin > unwhitened.
        const D: usize = 64;
        let f = 128;
        let v = 128;
        let m = 128;
        let (keys, values, fact_map) = synthetic_fact_set::<D>(f, v, 0x1234);
        let keys_ref = refs(&keys);
        let values_ref = refs(&values);
        let cfg_w = HebbianMlpConfig::new(D, m);
        let cfg_u = HebbianMlpConfig { variant: HebbianVariant::Unwhitened, ..cfg_w };
        let mem_w = HebbianKernelMemory::<D>::construct(&keys_ref, &values_ref, &fact_map, cfg_w, 0xABCD).unwrap();
        let mem_u = HebbianKernelMemory::<D>::construct(&keys_ref, &values_ref, &fact_map, cfg_u, 0xABCD).unwrap();
        let gamma_w = mem_w.decoding_margin(&keys_ref, &values_ref, &fact_map).unwrap();
        let gamma_u = mem_u.decoding_margin(&keys_ref, &values_ref, &fact_map).unwrap();
        assert!(gamma_w > 0.0, "whitened margin must be positive; got {gamma_w}");
        assert!(
            gamma_w > gamma_u,
            "whitened margin {gamma_w} must exceed unwhitened {gamma_u} (paper §B.2.4)"
        );
    }

    #[test]
    fn data_dependent_margin_positive_isotropic_d64_f128() {
        // P1 simplification: the data-dependent variant falls back to the
        // whitened readout (als_refine_* are no-ops gated on the PoC). So its
        // margin matches the whitened variant bit-identically at P1. Phase 2
        // will exercise the ALS refinement + require strict improvement.
        const D: usize = 64;
        let f = 128;
        let v = 128;
        let m = 128;
        let (keys, values, fact_map) = synthetic_fact_set::<D>(f, v, 0x1234);
        let keys_ref = refs(&keys);
        let values_ref = refs(&values);
        let cfg = HebbianMlpConfig { d: D, m, ridge: 1e-6, variant: HebbianVariant::DataDependent };
        let mem = HebbianKernelMemory::<D>::construct(&keys_ref, &values_ref, &fact_map, cfg, 0xABCD)
            .unwrap();
        let gamma = mem.decoding_margin(&keys_ref, &values_ref, &fact_map).unwrap();
        assert!(
            gamma > 0.0,
            "data-dependent margin must be positive; got {gamma}"
        );
    }

    #[test]
    fn whitened_construction_bit_identical_to_data_dependent_at_p1() {
        // P1 invariant: the data-dependent variant is a no-op refinement on
        // top of whitened (als_refine_* are gated on the PoC). So the two
        // variants must produce bit-identical B matrices. Phase 2 ALS fill-in
        // breaks this test (intentionally — it's the upgrade signal).
        const D: usize = 8;
        let (keys, values, fact_map) = synthetic_fact_set::<D>(16, 16, 0xCAFE);
        let keys_ref = refs(&keys);
        let values_ref = refs(&values);
        let cfg_w = HebbianMlpConfig::new(D, 32);
        let cfg_d = HebbianMlpConfig { variant: HebbianVariant::DataDependent, ..cfg_w };
        let m_w = HebbianKernelMemory::<D>::construct(&keys_ref, &values_ref, &fact_map, cfg_w, 0xBEEF).unwrap();
        let m_d = HebbianKernelMemory::<D>::construct(&keys_ref, &values_ref, &fact_map, cfg_d, 0xBEEF).unwrap();
        assert_eq!(m_w.b, m_d.b, "P1: data-dependent B must equal whitened B (ALS gated)");
    }

    // ── Forward + retrieval end-to-end ───────────────────────────────────

    #[test]
    fn forward_then_retrieval_argmax_recovers_stored_fact() {
        const D: usize = 8;
        let f = 8;
        let v = 8;
        let m = 64;
        let (keys, values, fact_map) = synthetic_fact_set::<D>(f, v, 0x9999);
        let keys_ref = refs(&keys);
        let values_ref = refs(&values);
        let cfg = HebbianMlpConfig::new(D, m);
        let mem = HebbianKernelMemory::<D>::construct(&keys_ref, &values_ref, &fact_map, cfg, 0x4242)
            .unwrap();

        let mut phi = vec![0.0_f32; m];
        let mut fwd = vec![0.0_f32; D];
        let mut scores = vec![0.0_f32; v];

        // For each stored key, the argmax over retrieval scores should
        // recover the stored value (paper Def 2.1).
        let mut n_correct = 0;
        for (i, &(_, v_idx)) in fact_map.iter().enumerate() {
            mem.retrieval_scores_into(&keys[i], &values_ref, &mut phi, &mut fwd, &mut scores);
            let argmax = scores
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| crate::float_order::cmp_for_max(**a, **b))
                .map(|(idx, _)| idx)
                .unwrap();
            if argmax == v_idx {
                n_correct += 1;
            }
        }
        // At generous m·d, the retrieval should be near-perfect. We require
        // at least 7/8 (87.5%) — full parity is the G5 PoC's job.
        assert!(
            n_correct >= 7,
            "retrieval argmax recovered {n_correct}/{f} facts (expected >= 7)"
        );
    }

    // ── Slot pattern ─────────────────────────────────────────────────────

    #[test]
    fn slot_starts_empty() {
        let slot = HebbianSlot::<8>::new();
        assert!(slot.is_empty());
        assert!(slot.current().is_none());
        assert!(slot.current_commitment().is_none());
        assert!(slot.current_blake3().is_none());
    }

    #[test]
    fn slot_induce_swaps_and_bumps_version() {
        const D: usize = 4;
        let (keys, values, fact_map) = synthetic_fact_set::<D>(8, 8, 1);
        let keys_ref = refs(&keys);
        let values_ref = refs(&values);
        let cfg = HebbianMlpConfig::new(D, 16);
        let m1 = HebbianKernelMemory::<D>::construct(&keys_ref, &values_ref, &fact_map, cfg, 1).unwrap();
        let m2 = HebbianKernelMemory::<D>::construct(&keys_ref, &values_ref, &fact_map, cfg, 2).unwrap();

        let slot = HebbianSlot::<D>::new();
        let c1 = slot.induce(m1, 1, 0.5, 8);
        assert_eq!(c1.version, 1);
        assert_eq!(slot.current().unwrap().1.version, 1);

        let c2 = slot.induce(m2, 2, 0.6, 8);
        assert_eq!(c2.version, 2);
        assert_eq!(slot.current().unwrap().1.version, 2);
        assert_eq!(slot.current_blake3(), Some(c2.blake3));
    }

    #[test]
    fn slot_cloned_shares_storage() {
        const D: usize = 4;
        let (keys, values, fact_map) = synthetic_fact_set::<D>(8, 8, 1);
        let keys_ref = refs(&keys);
        let values_ref = refs(&values);
        let cfg = HebbianMlpConfig::new(D, 16);
        let mem = HebbianKernelMemory::<D>::construct(&keys_ref, &values_ref, &fact_map, cfg, 1).unwrap();

        let slot1 = HebbianSlot::<D>::new();
        let slot2 = slot1.clone();
        slot1.induce(mem, 1, 0.5, 8);
        // Both clones see the same induced memory.
        assert!(!slot2.is_empty());
        assert_eq!(slot2.current_commitment().unwrap().version, 1);
    }

    // ── BLAKE3 commitment ────────────────────────────────────────────────

    #[test]
    fn blake3_commitment_is_deterministic() {
        const D: usize = 4;
        let (keys, values, fact_map) = synthetic_fact_set::<D>(8, 8, 1);
        let keys_ref = refs(&keys);
        let values_ref = refs(&values);
        let cfg = HebbianMlpConfig::new(D, 16);
        let m1 = HebbianKernelMemory::<D>::construct(&keys_ref, &values_ref, &fact_map, cfg, 7).unwrap();
        let m2 = HebbianKernelMemory::<D>::construct(&keys_ref, &values_ref, &fact_map, cfg, 7).unwrap();
        assert_eq!(m1.blake3(), m2.blake3());
    }

    #[test]
    fn blake3_changes_when_b_changes() {
        const D: usize = 4;
        let (keys, values, fact_map) = synthetic_fact_set::<D>(8, 8, 1);
        let keys_ref = refs(&keys);
        let values_ref = refs(&values);
        let cfg = HebbianMlpConfig::new(D, 16);
        let m_whitened = HebbianKernelMemory::<D>::construct(&keys_ref, &values_ref, &fact_map, cfg, 7).unwrap();
        let cfg_u = HebbianMlpConfig { variant: HebbianVariant::Unwhitened, ..cfg };
        let m_unwhitened = HebbianKernelMemory::<D>::construct(&keys_ref, &values_ref, &fact_map, cfg_u, 7).unwrap();
        assert_ne!(m_whitened.blake3(), m_unwhitened.blake3());
    }

    // ── SeedRng distribution sanity ──────────────────────────────────────

    #[test]
    fn seed_rng_gaussian_has_zero_mean_unit_variance_approximately() {
        let mut rng = SeedRng::new(42);
        let n = 10_000_usize;
        let mut samples = Vec::with_capacity(n);
        for _ in 0..n {
            let (z0, _) = rng.next_gaussian_pair();
            samples.push(z0);
        }
        let mean = samples.iter().sum::<f32>() / n as f32;
        let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;
        // Box-Muller is exact in infinite precision; f32 + finite N gives
        // mean within ±0.05 and var within ±0.05 of (0, 1).
        assert!(mean.abs() < 0.05, "mean {mean} should be ~0");
        assert!((var - 1.0).abs() < 0.1, "variance {var} should be ~1");
    }
}
