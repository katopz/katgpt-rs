//! slice_tca — modelless slice-rank decomposition for 3rd-order tensors
//! `X[n, t, k]` (entity × time × episode). Plan 596 / Research 309.
//!
//! # What this is
//!
//! A deterministic, zero-hyperparameter tensor decomposition that separates a
//! tensor's variability into three *covariability classes* — one per axis:
//! a class-σ component is `loading ⊗ slice_matrix`, with the loading vector
//! on axis σ and a unit-Frobenius-norm slice matrix spanning the other two
//! axes. The full model is the per-class sum
//!
//! ```text
//! X ≈ Σ_σ Σ_{r=1}^{R_σ} a^{(σ)}_r ⊗ M^{(σ)}_r
//! ```
//!
//! with a closed-form covariability classifier deciding which classes are
//! present, truncated-SVD single-class fits (Eckart–Young global optima via
//! the paper's own reduction: a single slice type ⇔ unfolding matrix
//! factorization), and a joint deterministic-ALS demixer that re-projects
//! each component's loading against the residual of all others.
//!
//! # Class taxonomy
//!
//! | Class | Loading axis | Slice matrix | Interpretation |
//! |---|---|---|---|
//! | [`SliceClass::Entity`] | 0 (n) | `t × k` | population covariability: entities share time×episode profiles |
//! | [`SliceClass::Time`] | 1 (t) | `n × k` | temporal covariability: timebins share entity×episode profiles |
//! | [`SliceClass::Episode`] | 2 (k) | `n × t` | episode covariability: episodes share entity×time profiles |
//!
//! The three classes are **not mutually exclusive** — a tensor can be
//! simultaneously entity- and time-covariable, which is the mixed case the
//! joint demixer exists for. That is why routing is a per-class
//! **sigmoid** (`sigmoid(α·(EVR−θ))`), never a softmax.
//!
//! # Slice rank
//!
//! The exact-algebra notion is *slice rank* — the minimum number of slice
//! hyperplanes (tensors supported on a single index of one axis) summing to
//! a tensor — introduced by Tao (2016) and developed by Tao & Sawin
//! ("The slice rank of a tensor", arXiv:1605.06702, 2016) for capset-type
//! bounds. sliceTCA (below) is the continuous least-squares relaxation used
//! in practice; this module implements that continuous form.
//!
//! # Sources & lineage
//!
//! - **sliceTCA**: C. Pellegrino, G. Stein & N. Çayko-Gajic,
//!   "sliceTCA: a method for decomposing tensor data", *Nat Neurosci* 27,
//!   1199–1210 (2024). doi:10.1038/s41593-024-01626-2. The class taxonomy,
//!   the single-class ≡ unfolding-SVD reduction, and the covariability
//!   motivation are theirs.
//! - **Slice rank**: T. Tao (2016); T. Tao & W. Sawin, arXiv:1605.06702.
//! - **ALS lineage**: J. D. Carroll & J.-J. Chang (1970), *Psychometrika* 25,
//!   283–319; R. A. Harshman (1970), *Psychometrika* 25, 219–219 (PARAFAC);
//!   orthogonalized ALS: arXiv:1703.01804. **Fitting novelty is NOT
//!   claimed** — the delta vs the paper's SGD fitter is the deterministic
//!   pure-function contract and the zero-hyperparameter block updates (see
//!   `als.rs` for the block-minimizer derivation). A sibling closed-form-ALS
//!   primitive in this crate is `tpr::als` (ridge-ALS over binding blocks —
//!   different model, same no-gradient doctrine).
//!
//! # Determinism contract
//!
//! **Same input bytes → bit-identical factors on the same target triple and
//! codegen.** BLAKE3-replayable via [`SliceDecomposition::canonical_hash`].
//! The contract is enforced by construction, not convention:
//!
//! - no `HashMap`/`HashSet` iteration anywhere in the fit path (the only
//!   collections are `Vec`s walked in construction order);
//! - no `rayon` in the fit path (sequential fixed-order sweeps; the opt-in
//!   CV grid is sequential too);
//! - no runtime-adaptive ε: every threshold is a `const` or a config field
//!   set before the fit (`NORM_EPS`, `ENERGY_FLOOR_TAU`, `ROUTE_THETA`,
//!   `ROUTE_ALPHA`, sweep counts);
//! - fixed accumulation order everywhere (chunk-8 lanes, folded
//!   left-to-right; SVD sweeps in substrate order);
//! - canonical output form: unit-norm slices, amplitude in the loadings,
//!   largest-|·|-entry-positive sign rule (first-index tie-break),
//!   variance-desc sort with a lexicographic (bit-pattern) tie-break — so
//!   SVD sign ambiguity and component permutation cannot leak into the
//!   committed bytes.
//!
//! Scope: the *committed surface* is the covariability shares/routing and
//! the canonicalized factor bytes — degeneracy-immune (order/sign/scale
//! gauge freedoms are fixed by canonicalization). Cross-platform
//! bit-identity is NOT claimed (f32 codegen differs); same-triple same-codegen
//! identity is pinned by an in-module test across 16 calls plus a
//! drop-and-rebuild.
//!
//! # Boundary
//!
//! Pure linear algebra — no game/chain/shard semantics, no platform APIs, no
//! `std::time` in `src/` (cross-compiles to wasm32). Private consumers wire
//! it separately (riir-neuron-db Plan 329, riir-train Plan 397).

mod als;
mod rank;
mod svd;
mod types;

#[cfg(test)]
mod tests;

pub use als::{fit_slice_into, fit_with_ranks_into, reallocate_class, relative_loss};
pub use rank::{select_ranks_blocked_cv, select_ranks_knee};
pub use svd::{covariability_shares, covariability_shares_into, fit_single_class_into, route, route_default};
pub use types::{
    CHUNK, DEFAULT_ALS_SWEEPS, ENERGY_FLOOR_TAU, InitMode, MAX_COMPONENTS, MAX_RANK_PER_CLASS,
    NORM_EPS, ROUTE_ALPHA, ROUTE_THETA, SliceClass, SliceComponent, SliceDecomposition,
    SliceTcaConfig, SliceTcaError, SliceTcaScratch, Tensor3,
};
