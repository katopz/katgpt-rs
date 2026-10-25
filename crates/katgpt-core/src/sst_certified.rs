//! sst_certified — tail-subspace substrate for weight-delta anatomy and
//! factored-tail shipping (Issue 958; Research 631, arXiv:2610.22293
//! "Subspace Tuning").
//!
//! # The law this module serves
//!
//! The stack composes weight mutations at runtime — freeze/thaw deltas,
//! `LoraPair { reader, writer }` overlays, TIES-style merges, consolidation
//! deltas, transcoded deltas — with no spectral accounting of WHERE a delta
//! writes relative to the base weights' capability substrate. The paper
//! supplies two load-bearing facts:
//!
//! 1. **~98% of an unconstrained delta's energy lands in the top-99%
//!    protected span** (their Fig. 4 controlled study). Existing stack
//!    deltas were produced WITHOUT tail confinement, so the modelless tool
//!    for them is an **anatomy report** (protected/free energy split — an
//!    interference predictor for merges, a drift quantifier for freeze
//!    deltas), never a compliance gate.
//! 2. **The composition-order law:** any elementwise transform between
//!    accumulation and projection leaks; quantizer ROUNDING is isotropic
//!    noise, ~`1−(1−r)²` of which lands in the protected block at
//!    `r = 0.99`, at magnitude comparable to the delta. Dense-project-then-
//!    quantize is unsalvageable at near-zero tolerance; the only certifiable
//!    shipping form for a quantized tail delta is **factored-tail**:
//!    `ΔW = U_⊥ C V_⊥ᵀ` with `C` quantized — the tail span is closed under
//!    the product, so confinement is preserved EXACTLY, certified by
//!    construction.
//!
//! # What this module owns
//!
//! - [`fit_anchor`] — the two-sided protected anchor over a reference weight
//!   matrix: Gram eigendecomposition on the SMALLER side
//!   ([`crate::karc::jacobi_eigen`], f64), singular directions propagated to
//!   the wider side through `Σ_k⁻¹` (precomputed reciprocals, zero-guarded).
//!   Storage economy per the issue: the NARROW side stores the FREE
//!   complement (`dim − k` columns, truly skinny); the WIDE side stores the
//!   PROTECTED `k` columns.
//! - [`SstAnchor::anatomy_into`] / [`SstAnchor::anatomy_factored_into`] — the
//!   per-delta energy split. Computed as `‖P_lΔ‖² + ‖P_⊥l·Δ·P_r‖²` (two
//!   POSITIVE terms — algebraically the issue's three-term form, but without
//!   the catastrophic cancellation of `E1 + E2 − E3` in f32).
//! - [`SstAnchor::factor_tail_into`] — the factored-tail shipping form.
//!   Payload `c = U_⊥ᵀ Δ V_⊥`; quantizing `c` keeps the decoded delta
//!   tail-confined by construction (the error stays in the span).
//! - [`SstAnchor::certify_decoded_tail_into`] — the RELATIVE-tolerance
//!   certifier over a decoded (e.g. quantized) delta: refuses when the
//!   protected-touching fraction exceeds `rel_tol`, reporting the measured
//!   leak. Never a near-zero absolute bar.
//! - [`compose_tail`] — chained factored-tail overlays stay tail-confined
//!   under addition (same anchor, bit-identical bases); the composite's leak
//!   bound is additive (triangle inequality).
//! - [`SstCertificate::verify_against`] — the re-anchor churn gate (Lane 6):
//!   a certificate is bound to the anchor's rotation-invariant digest;
//!   re-anchoring invalidates outstanding certifications — stale certs
//!   REFUSE loud, never silently pass.
//!
//! # Anchor commitment is rotation-invariant (Lane 4)
//!
//! The anchor's id is a BLAKE3 digest over the PROJECTOR action on 64
//! deterministic probe vectors (`U_⊥U_⊥ᵀ` / `V_kV_kᵀ` applied without
//! materialization, ordered folds), never over raw basis bytes: at r ≈ 0.99
//! the cut sits inside the Marchenko–Pastur bulk where near-degenerate gaps
//! make the tail basis defined only up to rotation. The digest is EXACTLY
//! invariant under sign flips / within-span coordinate permutations of the
//! basis (the products `b_j·(b_jᵀx)` are unchanged, in the same order), and
//! bit-stable across same-device recomputes; CROSS-device it is
//! digest-level only — near-equality is compared through the anatomy
//! quantities within tolerance instead (the digest pins fp-exactness, not
//! mathematical equality).
//!
//! # Honest cost envelope (documented per the issue's anchor-cost honesty)
//!
//! `fit_anchor` forms the smaller-side Gram (`O(m²n)`-class) and
//! eigendecomposes it (`O(min³)`-class per sweep) — at MLP shapes the anchor
//! is a CALIBRATION-TIME cost, roughly base-matrix-sized in its wide-side
//! sidecar. [`SstAnchor::factor_tail_into`]'s WIDER-side tail basis is
//! propagated from the narrow complement through the singular values, so it
//! covers the **σ-visible tail** (`≤ min − k` directions — the directions
//! the anchor's own spectrum can see); the null-space remainder of a wide
//! matrix needs the wide Gram (infeasible at these shapes) and is out of
//! scope — the certifier still covers any decoded delta regardless, since
//! the projector applies never need the missing basis.
//!
//! # Envelope
//!
//! f64 Gram accumulation with sequential folds; f32 bases with
//! sign-canonicalized columns; every armed hot path (`anatomy`, `factor`,
//! `certify`, `apply`, `decode`, `compose`) is zero-alloc over a
//! caller-owned [`SstScratch`] after a single warm call. Deterministic
//! under fixed input (sorted eigenpairs, stable sort, canonical signs).
//! Modelless — no training, no gradient descent. Opt-in feature
//! `sst_project` (implies `subspace_phase_gate` + `karc_forecaster` — the
//! thin-SVD test oracle and the f64 eigensolver this module consumes; the
//! implied-by-default pair mirrors the `basis_split` precedent so a
//! consumer forwarding `sst_project` without katgpt-core defaults never
//! hits E0433).

use blake3::Hasher;

use crate::karc::jacobi_eigen;

/// Relative singular-value floor below which a direction is treated as
/// numerically null (relative to the largest singular value).
const SIGMA_REL_FLOOR: f64 = 1e-6;

/// Probe count for the rotation-invariant projector digest (Lane 4).
const DIGEST_PROBES: usize = 64;

/// LCG seed for the digest probe vectors (fixed — the digest must be
/// reproducible across runs and processes).
const DIGEST_LCG_SEED: u64 = 0x9585_8958;

/// Jacobi sweeps upper bound for the Gram eigendecomposition (the solver
/// typically converges in 5–10; the bound is a safety ceiling).
const JACOBI_MAX_SWEEPS: usize = 60;

// ---------------------------------------------------------------------------
// Errors and refusals
// ---------------------------------------------------------------------------

/// Refuse-loud errors from anchor fitting and factor construction.
#[derive(Debug, Clone, PartialEq)]
pub enum SstError {
    /// A zero dimension was supplied.
    EmptyMatrix,
    /// `free_min` leaves no protected span: `min − free_min == 0`. Nothing
    /// protected means plain-SFT behavior wearing the flag — an SST
    /// non-case, refused by the width guard.
    NoProtectedSpan { min_dim: usize, free_min: usize },
    /// Fewer than the requested `k` singular directions sit above the
    /// relative floor — the reference matrix's spectrum cannot support the
    /// requested protected span.
    RankDeficientAnchor {
        requested: usize,
        available: usize,
        floor: f32,
    },
    /// A factored-tail side asked for more tail directions than the
    /// σ-visible tail carries.
    TailCapacityExceeded {
        requested: usize,
        available: usize,
        side: &'static str,
    },
    /// A slice length did not match the anchor's declared shape.
    DimensionMismatch { expected: usize, got: usize },
    /// A quantization step must be finite and strictly positive.
    BadQuantStep(f32),
    /// Composed factors disagree (anchor, shape, or exact bases).
    ComposeMismatch(&'static str),
}

impl core::fmt::Display for SstError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyMatrix => write!(f, "sst: zero matrix dimension"),
            Self::NoProtectedSpan { min_dim, free_min } => write!(
                f,
                "sst: free_min {free_min} leaves no protected span of min dim {min_dim} — refused (SST non-case)"
            ),
            Self::RankDeficientAnchor {
                requested,
                available,
                floor,
            } => write!(
                f,
                "sst: anchor spectrum supports k={available} above floor {floor:.3e}, requested {requested}"
            ),
            Self::TailCapacityExceeded {
                requested,
                available,
                side,
            } => write!(
                f,
                "sst: {side} tail capacity {available} < requested {requested} (σ-visible tail bound)"
            ),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "sst: expected {expected} elements, got {got}")
            }
            Self::BadQuantStep(s) => write!(f, "sst: quantization step {s} not finite/positive"),
            Self::ComposeMismatch(why) => write!(f, "sst: compose mismatch — {why}"),
        }
    }
}

impl std::error::Error for SstError {}

/// Refuse-loud certification verdicts (Lane 3/6).
#[derive(Debug, Clone, PartialEq)]
pub enum SstRefusal {
    /// The decoded delta's protected-touching fraction exceeds `rel_tol`.
    /// The measured leak rides the variant — the certifier refuses WITH
    /// the leak measured, never a bare no.
    LeakAboveTolerance {
        protected_fraction: f32,
        rel_tol: f32,
    },
    /// `rel_tol` outside the sane band `[1e-5, 1]` — below the module's f32
    /// resolution floor or not a tolerance at all.
    BadTolerance(f32),
    /// The certificate was issued against a different anchor (a re-anchor
    /// happened between issue and verify). Both digests are reported.
    StaleAnchor {
        cert_anchor: [u8; 32],
        live_anchor: [u8; 32],
    },
    /// A slice length did not match the anchor's declared shape.
    ShapeMismatch { expected: usize, got: usize },
}

impl core::fmt::Display for SstRefusal {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::LeakAboveTolerance {
                protected_fraction,
                rel_tol,
            } => write!(
                f,
                "sst: leak {protected_fraction:.3e} > rel_tol {rel_tol:.3e} — decoded delta touches the protected span"
            ),
            Self::BadTolerance(t) => write!(f, "sst: rel_tol {t} outside [1e-5, 1]"),
            Self::StaleAnchor {
                cert_anchor,
                live_anchor,
            } => write!(
                f,
                "sst: stale certificate — cert anchor {} vs live anchor {} (re-anchor invalidates outstanding certifications)",
                hex32(cert_anchor),
                hex32(live_anchor)
            ),
            Self::ShapeMismatch { expected, got } => {
                write!(f, "sst: expected {expected} elements, got {got}")
            }
        }
    }
}

impl std::error::Error for SstRefusal {}

fn hex32(bytes: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

// ---------------------------------------------------------------------------
// Per-side basis storage
// ---------------------------------------------------------------------------

/// One side's basis, stored under the issue's economy rule: the NARROW side
/// stores the FREE complement (truly skinny); the WIDE side stores the
/// PROTECTED `k` columns. Either representation derives both of the side's
/// projectors without ever materializing a `dim × dim` matrix.
#[derive(Debug, Clone, PartialEq)]
pub enum SideBasis {
    /// The FREE (tail) basis: `dim × (dim − k)`, row-major. The protected
    /// projector is `I − B Bᵀ`; the tail projector is `B Bᵀ`.
    Complement { basis: Vec<f32> },
    /// The PROTECTED basis: `dim × k`, row-major. The protected projector is
    /// `B Bᵀ`; the tail projector is `I − B Bᵀ`.
    Protected { basis: Vec<f32> },
}

impl SideBasis {
    /// Column count of this side's stored basis for a side of `dim` rows.
    pub fn cols_for(&self, dim: usize) -> usize {
        match self {
            Self::Complement { basis } | Self::Protected { basis } => basis.len() / dim,
        }
    }
}

// ---------------------------------------------------------------------------
// Scratch
// ---------------------------------------------------------------------------

/// Caller-owned scratch for every armed path. Grow-on-demand: after one warm
/// call at the working shape, all later calls allocate ZERO (G4).
#[derive(Default)]
pub struct SstScratch {
    // f64 eigensolver work
    gram: Vec<f64>,
    eigvals: Vec<f64>,
    eigvecs: Vec<f64>,
    jac: Vec<f64>,
    col64: Vec<f64>,
    order: Vec<usize>,
    // f32 work buffers (grow to the largest working shape once)
    t1: Vec<f32>,
    y1: Vec<f32>,
    y2: Vec<f32>,
    gr1: Vec<f32>,
    gr2: Vec<f32>,
    gr3: Vec<f32>,
    // digest probes + projection vectors
    probe: Vec<f32>,
    proj: Vec<f32>,
    zvec: Vec<f32>,
    wvec: Vec<f32>,
}

fn ensure(v: &mut Vec<f32>, n: usize) {
    if v.len() < n {
        v.resize(n, 0.0);
    }
}

fn ensure64(v: &mut Vec<f64>, n: usize) {
    if v.len() < n {
        v.resize(n, 0.0);
    }
}

// ---------------------------------------------------------------------------
// Deterministic probe RNG (module-private; the unseeded-global-RNG gate)
// ---------------------------------------------------------------------------

struct Lcg(u64);

impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn next_gauss(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos()
    }
}

// ---------------------------------------------------------------------------
// Ordered folds (Issue 940 determinism law — local instrument)
// ---------------------------------------------------------------------------

/// Sequential sum of squares in fixed (slice) order. Local to the anatomy —
/// nothing here crosses a sync/quorum boundary; the fold order is pinned
/// anyway so the certifier's measured fractions are reproducible.
fn sum_sq_ordered(slice: &[f32]) -> f32 {
    slice.iter().map(|v| v * v).sum()
}

// ---------------------------------------------------------------------------
// Anchor
// ---------------------------------------------------------------------------

/// The fitted two-sided protected anchor over a reference weight matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct SstAnchor {
    /// Output dimension (rows of the anchored matrix).
    pub m: usize,
    /// Input dimension (columns of the anchored matrix).
    pub n: usize,
    /// Protected rank (both sides carry the same `k`).
    pub k: usize,
    /// Free rank on the narrow side (`min(m, n) − k`).
    pub f_narrow: usize,
    /// The row (output) side's basis.
    pub left: SideBasis,
    /// The column (input) side's basis.
    pub right: SideBasis,
    /// Full singular spectrum of the anchor (descending, length
    /// `min(m, n)`; the tail entries feed `factor_tail`'s σ-propagation).
    pub sigma: Vec<f32>,
    /// Rotation-invariant BLAKE3 digest of the two-sided protected projector
    /// (Lane 4) — the anchor's identity.
    pub id: [u8; 32],
}

impl SstAnchor {
    /// The narrow side's dimension.
    pub fn narrow_dim(&self) -> usize {
        self.m.min(self.n)
    }

    /// The wide side's dimension.
    pub fn wide_dim(&self) -> usize {
        self.m.max(self.n)
    }

    /// Whether the LEFT (row) side is the narrow one.
    pub fn left_is_narrow(&self) -> bool {
        self.m <= self.n
    }

    // -- anatomy ------------------------------------------------------------

    /// The energy split of a dense delta `Δ` (`m × n`, row-major): the
    /// protected-touching fraction `E_prot / ‖Δ‖²_F` where
    /// `E_prot = ‖P_lΔ‖² + ‖P_⊥l·Δ·P_r‖²` — algebraically the issue's
    /// three-term form, computed as two POSITIVE sums so f32 never
    /// cancels. Zero-alloc over `scratch` after a warm call.
    pub fn anatomy_into(
        &self,
        delta: &[f32],
        scratch: &mut SstScratch,
    ) -> Result<SstAnatomy, SstError> {
        if delta.len() != self.m * self.n {
            return Err(SstError::DimensionMismatch {
                expected: self.m * self.n,
                got: delta.len(),
            });
        }
        let (m, n) = (self.m, self.n);
        ensure(&mut scratch.y1, m * n);
        ensure(&mut scratch.y2, m * n);
        // t1 hosts BOTH applies' T blocks: the left apply's BᵀX is f_l×n,
        // the right apply's X·B is m×f_r — size for the larger.
        let t_need = (self.left.cols_for(m) * n).max(m * self.right.cols_for(n));
        ensure(&mut scratch.t1, t_need);

        // Term 1: ‖P_l Δ‖² (protected projector on the rows).
        protected_apply_left_into(&self.left, delta, m, n, &mut scratch.t1, &mut scratch.y1);
        let term1 = sum_sq_ordered(&scratch.y1[..m * n]);

        // Term 2: ‖P_⊥l Δ P_r‖² — tail rows first (Δ − P_lΔ, elementwise),
        // then the protected projector on the columns.
        for (y2, (d, y1)) in scratch.y2[..m * n]
            .iter_mut()
            .zip(delta[..m * n].iter().zip(scratch.y1[..m * n].iter()))
        {
            *y2 = d - y1;
        }
        protected_apply_right_into(
            &self.right,
            &scratch.y2[..m * n],
            m,
            n,
            &mut scratch.t1,
            &mut scratch.y1,
        );
        let term2 = sum_sq_ordered(&scratch.y1[..m * n]);

        let total = sum_sq_ordered(delta);
        let protected = term1 + term2;
        Ok(SstAnatomy {
            total_energy: total,
            protected_energy: protected,
            free_energy: total - protected,
            protected_fraction: if total > 0.0 { protected / total } else { 0.0 },
        })
    }

    /// Convenience (allocating scratch) form of [`SstAnchor::anatomy_into`].
    pub fn anatomy(&self, delta: &[f32]) -> Result<SstAnatomy, SstError> {
        let mut s = SstScratch::default();
        self.anatomy_into(delta, &mut s)
    }

    /// The energy split of a FACTORED delta `Δ = A·Bᵀ` (`A`: `m × r`,
    /// `B`: `n × r`, both row-major) — the rank-`r` path the stack's actual
    /// deltas (LoRA overlays, low-rank freeze deltas) take. Never forms the
    /// dense `m × n`; all terms are `r × r` fold pairs.
    pub fn anatomy_factored_into(
        &self,
        a_mat: &[f32],
        b_mat: &[f32],
        r: usize,
        scratch: &mut SstScratch,
    ) -> Result<SstAnatomy, SstError> {
        if a_mat.len() != self.m * r || b_mat.len() != self.n * r {
            return Err(SstError::DimensionMismatch {
                expected: (self.m + self.n) * r,
                got: a_mat.len() + b_mat.len(),
            });
        }
        let (m, n) = (self.m, self.n);
        // y1 hosts both the left-projected A (m×r) and the right-projected B
        // (n×r) — size for the larger.
        ensure(&mut scratch.y1, (m * r).max(n * r));
        ensure(&mut scratch.y2, (m * r).max(n * r));
        // t1 hosts both applies' T blocks — both are left-applies here
        // (BᵀA and Bᵀ(P_r B) over the r columns): f_max×r.
        let t_need = self.left.cols_for(m).max(self.right.cols_for(n)) * r;
        ensure(&mut scratch.t1, t_need);
        ensure(&mut scratch.gr1, r * r);
        ensure(&mut scratch.gr2, r * r);
        ensure(&mut scratch.gr3, r * r);

        // S_B = BᵀB (r×r), sequential row-fold per entry.
        gram_cols_into(b_mat, n, r, &mut scratch.gr3);

        // P_l A → gr1 = (P_l A)ᵀ(P_l A); term1 = ⟨gr1, S_B⟩.
        protected_apply_left_into(&self.left, a_mat, m, r, &mut scratch.t1, &mut scratch.y1);
        gram_cols_into(&scratch.y1[..m * r], m, r, &mut scratch.gr1);
        let term1 = frob_pair(&scratch.gr1[..r * r], &scratch.gr3[..r * r]);

        // total = ⟨AᵀA, S_B⟩.
        gram_cols_into(a_mat, m, r, &mut scratch.gr1);
        let total_pair = frob_pair(&scratch.gr1[..r * r], &scratch.gr3[..r * r]);

        // P_⊥l A = A − P_l A → gr2; P_r B → S2 overwrites gr1; term2 = ⟨gr2, S2⟩.
        for (y2, (a, y1)) in scratch.y2[..m * r]
            .iter_mut()
            .zip(a_mat[..m * r].iter().zip(scratch.y1[..m * r].iter()))
        {
            *y2 = a - y1;
        }
        gram_cols_into(&scratch.y2[..m * r], m, r, &mut scratch.gr2);
        // P_r·B is a LEFT-apply: the projector acts on the ROWS of the n×r
        // matrix (Δ·P_r = A·(P_r B)ᵀ — B's row index IS the projector's
        // side dim). The column-side helper would derive its width from the
        // r columns and read the basis as r-wide garbage.
        protected_apply_left_into(&self.right, b_mat, n, r, &mut scratch.t1, &mut scratch.y1);
        gram_cols_into(&scratch.y1[..n * r], n, r, &mut scratch.gr1);
        let term2 = frob_pair(&scratch.gr2[..r * r], &scratch.gr1[..r * r]);

        // Note: term2 pairs (P_⊥l A)ᵀ(P_⊥l A) with (P_r B)ᵀ(P_r B) — the
        // Frobenius identity ‖XYᵀ‖²_F = ⟨XᵀX, YᵀY⟩ applied to the tail-left ×
        // protected-right block.
        let protected = term1 + term2;
        Ok(SstAnatomy {
            total_energy: total_pair,
            protected_energy: protected,
            free_energy: total_pair - protected,
            protected_fraction: if total_pair > 0.0 {
                protected / total_pair
            } else {
                0.0
            },
        })
    }

    // -- factored tail ------------------------------------------------------

    /// Build the factored-tail shipping form of a dense delta
    /// `Δ = U_⊥ C V_⊥ᵀ` (zero-alloc over `out`/`scratch` after a warm call).
    ///
    /// `tail_left_req` / `tail_right_req` choose each side's tail-basis
    /// width, bounded by the σ-visible tail (the narrow side's stored
    /// complement; the wider side's σ-propagated directions — see the
    /// module docs). The residual (dropped tail + protected) energy is
    /// reported in [`SstFactorReport`].
    ///
    /// `w_ref` must be the SAME reference the anchor was fitted on (the
    /// σ-propagation reads it); a drifted reference silently mis-propagates
    /// — the caller's contract, asserted by the digest on the way OUT via
    /// [`SstCertificate::verify_against`].
    #[allow(clippy::too_many_arguments)]
    pub fn factor_tail_into(
        &self,
        w_ref: &[f32],
        delta: &[f32],
        tail_left_req: usize,
        tail_right_req: usize,
        out: &mut TailFactor,
        report: &mut SstFactorReport,
        scratch: &mut SstScratch,
    ) -> Result<(), SstError> {
        if w_ref.len() != self.m * self.n || delta.len() != self.m * self.n {
            return Err(SstError::DimensionMismatch {
                expected: self.m * self.n,
                got: if w_ref.len() != self.m * self.n {
                    w_ref.len()
                } else {
                    delta.len()
                },
            });
        }
        // Availability: the narrow side carries f_narrow tail columns
        // (stored complement); the wide side is σ-propagated from the same
        // columns, so its availability is the count of tail sigmas above
        // the floor.
        let floor = self.sigma[0] as f64 * SIGMA_REL_FLOOR;
        let mut visible = 0usize;
        for j in 0..self.f_narrow {
            if (self.sigma[self.k + j] as f64) > floor {
                visible += 1;
            } else {
                break;
            }
        }
        if tail_left_req > visible {
            return Err(SstError::TailCapacityExceeded {
                requested: tail_left_req,
                available: visible,
                side: if self.left_is_narrow() {
                    "left"
                } else {
                    "right"
                },
            });
        }
        if tail_right_req > visible {
            return Err(SstError::TailCapacityExceeded {
                requested: tail_right_req,
                available: visible,
                side: if self.left_is_narrow() {
                    "right"
                } else {
                    "left"
                },
            });
        }
        let (f_l, f_r) = (tail_left_req, tail_right_req);

        // Materialize the two tail bases (literal, row-major):
        //   narrow side: the stored complement's first f_req columns;
        //   wide side:   propagated W-mapped columns (σ⁻¹-scaled).
        let narrow_f = if self.left_is_narrow() { f_l } else { f_r };
        let wide_f = if self.left_is_narrow() { f_r } else { f_l };
        let (nar, wide) = (self.narrow_dim(), self.wide_dim());
        out.ensure_shapes(self.m, self.n, f_l, f_r);
        out.anchor_id = self.id;

        // Narrow-side basis columns come straight from the stored complement.
        let comp = match if self.left_is_narrow() {
            &self.left
        } else {
            &self.right
        } {
            SideBasis::Complement { basis } => basis,
            SideBasis::Protected { .. } => unreachable!("narrow side always stores Complement"),
        };
        {
            let narrow_out: &mut Vec<f32> = if self.left_is_narrow() {
                &mut out.left_basis
            } else {
                &mut out.right_basis
            };
            let nar_cols = self.f_narrow;
            for col in 0..narrow_f {
                for row in 0..nar {
                    narrow_out[row * narrow_f + col] = comp[row * nar_cols + col];
                }
            }
        }

        // Wide-side basis: propagate the narrow complement's first `wide_f`
        // columns through W (σ⁻¹ from the precomputed, zero-guarded tail
        // sigmas).
        ensure(&mut scratch.zvec, nar);
        ensure(&mut scratch.wvec, wide);
        for col in 0..wide_f {
            let sigma_tail = self.sigma[self.k + col];
            // Zero-guard per the algebraic-op discipline: the reciprocal is
            // taken once, only above the floor (visible counted exactly
            // those).
            let recip = 1.0f32 / sigma_tail;
            // z = narrow complement column (f64-free: cast per element).
            for row in 0..nar {
                scratch.zvec[row] = comp[row * self.f_narrow + col];
            }
            // wide = (left narrow ? Wᵀ z : W z) · σ⁻¹, sequential folds.
            if self.left_is_narrow() {
                // Wᵀ z: v[j] = Σ_i W[i,j]·z[i] — W is m×n.
                for j in 0..wide {
                    let mut acc = 0.0f32;
                    for i in 0..nar {
                        acc += w_ref[i * self.n + j] * scratch.zvec[i];
                    }
                    scratch.wvec[j] = acc * recip;
                }
            } else {
                // W z: u[i] = Σ_j W[i,j]·z[j] — W is m×n, nar = n.
                for i in 0..wide {
                    let mut acc = 0.0f32;
                    for j in 0..nar {
                        acc += w_ref[i * self.n + j] * scratch.zvec[j];
                    }
                    scratch.wvec[i] = acc * recip;
                }
            }
            // Sign-canonicalize (largest-|entry| positive, first wins ties).
            let mut best = 0usize;
            let mut best_abs = 0.0f32;
            for i in 0..wide {
                let a = scratch.wvec[i].abs();
                if a > best_abs {
                    best_abs = a;
                    best = i;
                }
            }
            let flip = scratch.wvec[best] < 0.0;
            let wide_out: &mut Vec<f32> = if self.left_is_narrow() {
                &mut out.right_basis
            } else {
                &mut out.left_basis
            };
            for i in 0..wide {
                let v = if flip {
                    -scratch.wvec[i]
                } else {
                    scratch.wvec[i]
                };
                wide_out[i * wide_f + col] = v;
            }
        }

        // Payload c = U_⊥ᵀ · Δ · V_tail: T = Δ·V_tail (m×f_r), then
        // c = U_tailᵀ·T (f_l×f_r).
        ensure(&mut scratch.y1, self.m * f_r);
        xb_into(
            &out.right_basis,
            f_r,
            delta,
            self.m,
            self.n,
            &mut scratch.y1[..self.m * f_r],
        );
        {
            let (left_basis, c) = (&out.left_basis, &mut out.c[..f_l * f_r]);
            btx_into(left_basis, f_l, &scratch.y1[..self.m * f_r], self.m, f_r, c);
        }

        // Report: shipped = ‖c‖² (Parseval), tail = total − protected,
        // dropped = tail − shipped.
        let anatomy = self.anatomy_into(delta, scratch)?;
        let shipped = sum_sq_ordered(&out.c[..f_l * f_r]);
        *report = SstFactorReport {
            delta_energy: anatomy.total_energy,
            tail_energy: anatomy.free_energy,
            shipped_energy: shipped,
            dropped_tail_energy: anatomy.free_energy - shipped,
            f_left: f_l,
            f_right: f_r,
        };
        Ok(())
    }

    /// Convenience (allocating) form of [`SstAnchor::factor_tail_into`].
    pub fn factor_tail(
        &self,
        w_ref: &[f32],
        delta: &[f32],
        tail_left_req: usize,
        tail_right_req: usize,
    ) -> Result<(TailFactor, SstFactorReport), SstError> {
        let mut out = TailFactor::default();
        let mut report = SstFactorReport::default();
        let mut s = SstScratch::default();
        self.factor_tail_into(
            w_ref,
            delta,
            tail_left_req,
            tail_right_req,
            &mut out,
            &mut report,
            &mut s,
        )?;
        Ok((out, report))
    }

    // -- certification ------------------------------------------------------

    /// Certify a decoded (e.g. quantize/dequantize round-tripped) delta:
    /// refuse when the protected-touching fraction exceeds `rel_tol`,
    /// reporting the measured leak. Relative, never absolute — `rel_tol` is
    /// calibrated against the EXPECTED quantization-noise floor by the
    /// caller (the module enforces only the sane band `[1e-5, 1]`, above
    /// its f32 resolution floor). Zero-alloc over `scratch`.
    pub fn certify_decoded_tail_into(
        &self,
        decoded: &[f32],
        rel_tol: f32,
        scratch: &mut SstScratch,
    ) -> Result<SstCertificate, SstRefusal> {
        if !(rel_tol.is_finite() && (1e-5..=1.0).contains(&rel_tol)) {
            return Err(SstRefusal::BadTolerance(rel_tol));
        }
        if decoded.len() != self.m * self.n {
            return Err(SstRefusal::ShapeMismatch {
                expected: self.m * self.n,
                got: decoded.len(),
            });
        }
        let anatomy =
            self.anatomy_into(decoded, scratch)
                .map_err(|_| SstRefusal::ShapeMismatch {
                    expected: self.m * self.n,
                    got: decoded.len(),
                })?;
        let certificate = SstCertificate {
            anchor_id: self.id,
            total_energy: anatomy.total_energy,
            protected_energy: anatomy.protected_energy,
            protected_fraction: anatomy.protected_fraction,
            rel_tol,
        };
        if anatomy.protected_fraction > rel_tol {
            return Err(SstRefusal::LeakAboveTolerance {
                protected_fraction: anatomy.protected_fraction,
                rel_tol,
            });
        }
        Ok(certificate)
    }

    /// Convenience (allocating scratch) form of
    /// [`SstAnchor::certify_decoded_tail_into`].
    pub fn certify_decoded_tail(
        &self,
        decoded: &[f32],
        rel_tol: f32,
    ) -> Result<SstCertificate, SstRefusal> {
        let mut s = SstScratch::default();
        self.certify_decoded_tail_into(decoded, rel_tol, &mut s)
    }
}

/// Per-side projector applies (module-private; both `SideBasis` forms).
///
/// `P_prot` on the LEFT of an `m×c` row-major matrix `x`:
/// - `Complement{B}` (B: m×f): `y = x − B(Bᵀx)`
/// - `Protected{B}`  (B: m×k): `y = B(Bᵀx)`
fn protected_apply_left_into(
    basis: &SideBasis,
    x: &[f32],
    rows: usize,
    cols: usize,
    t: &mut [f32],
    y: &mut [f32],
) {
    match basis {
        SideBasis::Complement { basis: b } => {
            let f = b.len() / rows;
            left_factor_apply_into(b, f, x, rows, cols, t, y, false);
        }
        SideBasis::Protected { basis: b } => {
            let k = b.len() / rows;
            left_factor_apply_into(b, k, x, rows, cols, t, y, true);
        }
    }
}

/// `P_prot` on the RIGHT: `y = x · P` with `P` acting on the columns.
fn protected_apply_right_into(
    basis: &SideBasis,
    x: &[f32],
    rows: usize,
    cols: usize,
    t: &mut [f32],
    y: &mut [f32],
) {
    match basis {
        SideBasis::Complement { basis: b } => {
            let f = b.len() / cols;
            right_factor_apply_into(b, f, x, rows, cols, t, y, false);
        }
        SideBasis::Protected { basis: b } => {
            let k = b.len() / cols;
            right_factor_apply_into(b, k, x, rows, cols, t, y, true);
        }
    }
}

/// `y = B·(Bᵀ·x)` (subtract=true: `y = x − B·(Bᵀ·x)`), B row-major `rows×f`.
/// Sequential folds in fixed order — deterministic.
/// (8 args by contract: the hot kernel's explicit scratch discipline — the
/// caller-owned buffers ARE the zero-alloc API.)
#[allow(clippy::too_many_arguments)]
fn left_factor_apply_into(
    b: &[f32],
    f: usize,
    x: &[f32],
    rows: usize,
    cols: usize,
    t: &mut [f32],
    y: &mut [f32],
    keep: bool,
) {
    // T = Bᵀ x  (f×cols)
    for c in 0..cols {
        for j in 0..f {
            let mut acc = 0.0f32;
            for r in 0..rows {
                acc += b[r * f + j] * x[r * cols + c];
            }
            t[j * cols + c] = acc;
        }
    }
    // Y = B·T (keep) or X − B·T (complement)
    for r in 0..rows {
        for c in 0..cols {
            let mut acc = 0.0f32;
            for j in 0..f {
                acc += b[r * f + j] * t[j * cols + c];
            }
            y[r * cols + c] = if keep { acc } else { x[r * cols + c] - acc };
        }
    }
}

/// `y = x·(B·Bᵀ)` (keep) / `x·(I − B·Bᵀ)` (complement), B row-major `cols×f`.
/// (8 args by contract — see [`left_factor_apply_into`].)
#[allow(clippy::too_many_arguments)]
fn right_factor_apply_into(
    b: &[f32],
    f: usize,
    x: &[f32],
    rows: usize,
    cols: usize,
    t: &mut [f32],
    y: &mut [f32],
    keep: bool,
) {
    // T = x·B  (rows×f)
    for r in 0..rows {
        for j in 0..f {
            let mut acc = 0.0f32;
            for c in 0..cols {
                acc += x[r * cols + c] * b[c * f + j];
            }
            t[r * f + j] = acc;
        }
    }
    // Y = T·Bᵀ (keep) or X − T·Bᵀ
    for r in 0..rows {
        for c in 0..cols {
            let mut acc = 0.0f32;
            for j in 0..f {
                acc += t[r * f + j] * b[c * f + j];
            }
            y[r * cols + c] = if keep { acc } else { x[r * cols + c] - acc };
        }
    }
}

/// Column Gram of a `rows×f` row-major matrix (`f×f` out):
/// `out[j][s] = Σ_r x[r·f + j]·x[r·f + s]` — `XᵀX`, sequential in the row
/// index (deterministic). Serves both the factored-anatomy Grams and the
/// right-side second moments — the two formulas are the same operator
/// (the DRY law; there is no separate "row" variant).
fn gram_cols_into(x: &[f32], rows: usize, f: usize, out: &mut [f32]) {
    for j in 0..f {
        for s in 0..f {
            let mut acc = 0.0f32;
            for r in 0..rows {
                acc += x[r * f + j] * x[r * f + s];
            }
            out[j * f + s] = acc;
        }
    }
}

/// Frobenius inner product of two `n`-element slices, sequential.
fn frob_pair(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ---------------------------------------------------------------------------
// Fit
// ---------------------------------------------------------------------------

/// Fit the two-sided protected anchor over `w_ref` (`m×n`, row-major).
///
/// `free_min` is the FREE-span floor: the protected rank is
/// `k = min(m, n) − free_min`; `k == 0` is an SST non-case and is REFUSED
/// (the width guard). Allocation is calibration-time (the returned anchor
/// owns its bases); the f64 eigensolver work rides `scratch`.
pub fn fit_anchor_into(
    w_ref: &[f32],
    m: usize,
    n: usize,
    free_min: usize,
    scratch: &mut SstScratch,
) -> Result<SstAnchor, SstError> {
    if m == 0 || n == 0 {
        return Err(SstError::EmptyMatrix);
    }
    let nar = m.min(n);
    let k0 = nar.saturating_sub(free_min);
    if k0 == 0 {
        return Err(SstError::NoProtectedSpan {
            min_dim: nar,
            free_min,
        });
    }
    if w_ref.len() != m * n {
        return Err(SstError::DimensionMismatch {
            expected: m * n,
            got: w_ref.len(),
        });
    }
    let left_narrow = m <= n;

    // Gram on the SMALLER side (f64, sequential accumulation): G = W·Wᵀ
    // (left narrow) or Wᵀ·W (right narrow). The Gram's eigenvectors are
    // the narrow side's singular directions.
    ensure64(&mut scratch.gram, nar * nar);
    ensure64(&mut scratch.eigvals, nar);
    ensure64(&mut scratch.eigvecs, nar * nar);
    ensure64(&mut scratch.jac, nar * nar);
    {
        let g = &mut scratch.gram[..nar * nar];
        g.fill(0.0);
        // W viewed as rows (m×n): row i at w[i*n .. i*n+n].
        if left_narrow {
            for i in 0..nar {
                for j in i..nar {
                    let mut acc = 0.0f64;
                    for t in 0..n {
                        acc += (w_ref[i * n + t] as f64) * (w_ref[j * n + t] as f64);
                    }
                    g[i * nar + j] = acc;
                    g[j * nar + i] = acc;
                }
            }
        } else {
            // Columns of W: column j at w[t*n + j] over rows t.
            for i in 0..nar {
                for j in i..nar {
                    let mut acc = 0.0f64;
                    for t in 0..m {
                        acc += (w_ref[t * n + i] as f64) * (w_ref[t * n + j] as f64);
                    }
                    g[i * nar + j] = acc;
                    g[j * nar + i] = acc;
                }
            }
        }
        // Absolute off-diagonal-mass tolerance scaled to the trace.
        let trace: f64 = (0..nar).map(|i| g[i * nar + i]).sum();
        let tol = 1e-14 * (trace.abs() + 1.0);
        jacobi_eigen(
            &mut scratch.eigvals[..nar],
            &mut scratch.eigvecs[..nar * nar],
            &scratch.gram[..nar * nar],
            &mut scratch.jac[..nar * nar],
            nar,
            tol,
            JACOBI_MAX_SWEEPS,
        );
    }

    // Sort eigenpairs descending (stable, deterministic), gather columns,
    // canonicalize signs (largest-|entry| positive, first wins ties).
    scratch.order.clear();
    scratch.order.extend(0..nar);
    scratch
        .order
        .sort_by(|&x, &y| scratch.eigvals[y].total_cmp(&scratch.eigvals[x]));
    ensure64(&mut scratch.col64, nar * nar);
    {
        let src = &scratch.eigvecs[..nar * nar];
        let dst = &mut scratch.col64[..nar * nar];
        for row in 0..nar {
            for t in 0..nar {
                dst[row * nar + t] = src[row * nar + scratch.order[t]];
            }
        }
        for t in 0..nar {
            let mut best = 0usize;
            let mut best_abs = 0.0f64;
            for i in 0..nar {
                let a = dst[i * nar + t].abs();
                if a > best_abs {
                    best_abs = a;
                    best = i;
                }
            }
            if dst[best * nar + t] < 0.0 {
                for i in 0..nar {
                    dst[i * nar + t] = -dst[i * nar + t];
                }
            }
        }
    }

    // Spectrum + guards. The sigma order must follow the SAME sort that
    // permuted the eigenvector columns (paired by `order`, never the
    // solver's raw diagonal order — the propagation and the floor guard
    // both read (eigenpair i, sigma i) as one unit).
    let sigma64: Vec<f64> = (0..nar)
        .map(|i| scratch.eigvals[scratch.order[i]].max(0.0).sqrt())
        .collect();
    let sigma: Vec<f32> = sigma64.iter().map(|s| *s as f32).collect();
    let sigma_max = sigma64[0];
    if sigma_max <= 0.0 {
        return Err(SstError::RankDeficientAnchor {
            requested: k0,
            available: 0,
            floor: 0.0,
        });
    }
    let floor = sigma_max * SIGMA_REL_FLOOR;
    let mut k_eff = 0usize;
    for &s in sigma64.iter().take(k0) {
        if s > floor {
            k_eff += 1;
        } else {
            break;
        }
    }
    if k_eff < k0 {
        return Err(SstError::RankDeficientAnchor {
            requested: k0,
            available: k_eff,
            floor: floor as f32,
        });
    }
    let f_narrow = nar - k_eff;

    // Narrow side: the FREE complement (columns k..nar), f32.
    let mut narrow_basis = vec![0.0f32; nar * f_narrow];
    for row in 0..nar {
        for col in 0..f_narrow {
            narrow_basis[row * f_narrow + col] = scratch.col64[row * nar + (k_eff + col)] as f32;
        }
    }

    // Wide side: the PROTECTED k columns, σ-propagated from the narrow
    // eigenvectors (precomputed reciprocals, zero-guarded — every i < k_eff
    // is above the floor by the guard above).
    let wide = m.max(n);
    let mut wide_basis = vec![0.0f32; wide * k_eff];
    ensure(&mut scratch.zvec, nar);
    ensure(&mut scratch.wvec, wide);
    for i in 0..k_eff {
        for row in 0..nar {
            scratch.zvec[row] = scratch.col64[row * nar + i] as f32;
        }
        let recip = 1.0f32 / sigma[i];
        if left_narrow {
            // wide (right) vector v = Wᵀ u · σ⁻¹, sequential i-fold.
            for j in 0..n {
                let mut acc = 0.0f32;
                for t in 0..m {
                    acc += w_ref[t * n + j] * scratch.zvec[t];
                }
                scratch.wvec[j] = acc * recip;
            }
        } else {
            // wide (left) vector u = W v · σ⁻¹, sequential j-fold.
            for t in 0..m {
                let mut acc = 0.0f32;
                for j in 0..nar {
                    acc += w_ref[t * n + j] * scratch.zvec[j];
                }
                scratch.wvec[t] = acc * recip;
            }
        }
        // Sign-canonicalize the propagated column.
        let mut best = 0usize;
        let mut best_abs = 0.0f32;
        for idx in 0..wide {
            let a = scratch.wvec[idx].abs();
            if a > best_abs {
                best_abs = a;
                best = idx;
            }
        }
        let flip = scratch.wvec[best] < 0.0;
        for idx in 0..wide {
            let v = if flip {
                -scratch.wvec[idx]
            } else {
                scratch.wvec[idx]
            };
            wide_basis[idx * k_eff + i] = v;
        }
    }

    let (left, right) = if left_narrow {
        (
            SideBasis::Complement {
                basis: narrow_basis,
            },
            SideBasis::Protected { basis: wide_basis },
        )
    } else {
        (
            SideBasis::Protected { basis: wide_basis },
            SideBasis::Complement {
                basis: narrow_basis,
            },
        )
    };

    let mut anchor = SstAnchor {
        m,
        n,
        k: k_eff,
        f_narrow,
        left,
        right,
        sigma,
        id: [0u8; 32],
    };
    anchor.id = anchor_digest(&anchor, scratch);
    Ok(anchor)
}

/// Convenience (allocating scratch) form of [`fit_anchor_into`].
pub fn fit_anchor(
    w_ref: &[f32],
    m: usize,
    n: usize,
    free_min: usize,
) -> Result<SstAnchor, SstError> {
    let mut s = SstScratch::default();
    fit_anchor_into(w_ref, m, n, free_min, &mut s)
}

/// Rotation-invariant anchor digest (Lane 4): BLAKE3 over the protected
/// projector's action on 64 deterministic probe vectors per side, ordered
/// folds, plus the shape. Sign flips / within-span coordinate permutations
/// of the basis leave the projector — and therefore every probe action —
/// unchanged, so the digest is EXACTLY invariant under them.
fn anchor_digest(anchor: &SstAnchor, scratch: &mut SstScratch) -> [u8; 32] {
    let mut lcg = Lcg(DIGEST_LCG_SEED);
    let left_digest = side_digest(&anchor.left, anchor.m, &mut lcg, scratch);
    let right_digest = side_digest(&anchor.right, anchor.n, &mut lcg, scratch);
    let mut h = Hasher::new();
    h.update(&anchor.m.to_le_bytes());
    h.update(&anchor.n.to_le_bytes());
    h.update(&anchor.k.to_le_bytes());
    h.update(&left_digest);
    h.update(&right_digest);
    *h.finalize().as_bytes()
}

fn side_digest(basis: &SideBasis, dim: usize, lcg: &mut Lcg, scratch: &mut SstScratch) -> [u8; 32] {
    ensure(&mut scratch.probe, dim);
    ensure(&mut scratch.proj, dim);
    let mut h = Hasher::new();
    for _ in 0..DIGEST_PROBES {
        for v in &mut scratch.probe[..dim] {
            *v = lcg.next_gauss() as f32;
        }
        // y = P_prot · x (vector form); the t scratch rides scratch.t1
        // (prefix — grow-on-demand, no per-probe allocation).
        match basis {
            SideBasis::Complement { basis: b } => {
                let f = b.len() / dim;
                ensure(&mut scratch.t1, f);
                for j in 0..f {
                    let mut acc = 0.0f32;
                    for r in 0..dim {
                        acc += b[r * f + j] * scratch.probe[r];
                    }
                    scratch.t1[j] = acc;
                }
                for r in 0..dim {
                    let mut acc = 0.0f32;
                    for j in 0..f {
                        acc += b[r * f + j] * scratch.t1[j];
                    }
                    scratch.proj[r] = scratch.probe[r] - acc;
                }
            }
            SideBasis::Protected { basis: b } => {
                let k = b.len() / dim;
                ensure(&mut scratch.t1, k);
                for j in 0..k {
                    let mut acc = 0.0f32;
                    for r in 0..dim {
                        acc += b[r * k + j] * scratch.probe[r];
                    }
                    scratch.t1[j] = acc;
                }
                for r in 0..dim {
                    let mut acc = 0.0f32;
                    for j in 0..k {
                        acc += b[r * k + j] * scratch.t1[j];
                    }
                    scratch.proj[r] = acc;
                }
            }
        }
        for v in &scratch.proj[..dim] {
            h.update(&v.to_le_bytes());
        }
    }
    *h.finalize().as_bytes()
}

// ---------------------------------------------------------------------------
// Basis multiplies (the payload path — NOT projector applies)
// ---------------------------------------------------------------------------

/// `out = x·B` for B row-major `cols×f`: `out[r][j] = Σ_c x[r][c]·B[c][j]`.
fn xb_into(b: &[f32], f: usize, x: &[f32], rows: usize, cols: usize, out: &mut [f32]) {
    for r in 0..rows {
        for j in 0..f {
            let mut acc = 0.0f32;
            for c in 0..cols {
                acc += x[r * cols + c] * b[c * f + j];
            }
            out[r * f + j] = acc;
        }
    }
}

/// `out = Bᵀ·x` for B row-major `rows×f`, x `rows×cols`:
/// `out[j][s] = Σ_r B[r][j]·x[r][s]`.
fn btx_into(b: &[f32], f: usize, x: &[f32], rows: usize, cols: usize, out: &mut [f32]) {
    for j in 0..f {
        for s in 0..cols {
            let mut acc = 0.0f32;
            for r in 0..rows {
                acc += b[r * f + j] * x[r * cols + s];
            }
            out[j * cols + s] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Anatomy / report / factor / certificate
// ---------------------------------------------------------------------------

/// The energy split of a delta against an anchor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SstAnatomy {
    /// `‖Δ‖²_F`.
    pub total_energy: f32,
    /// Energy in every block that touches a protected span (everything but
    /// the free×free block).
    pub protected_energy: f32,
    /// `total − protected` — the tail component's energy.
    pub free_energy: f32,
    /// `protected / total` — the interference fraction (the anatomy
    /// report's headline number; ~0.98 for unconstrained stack deltas per
    /// the paper's study).
    pub protected_fraction: f32,
}

/// What [`SstAnchor::factor_tail_into`] shipped and what it dropped.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SstFactorReport {
    pub delta_energy: f32,
    /// The delta's tail energy (what a FULL tail factor would ship).
    pub tail_energy: f32,
    /// `‖C‖²_F` — what the (possibly truncated) factor ships (Parseval:
    /// the bases are orthonormal, so `‖U_⊥CV_⊥ᵀ‖² = ‖C‖²`).
    pub shipped_energy: f32,
    /// Tail energy beyond the requested truncation — reported, never hidden.
    pub dropped_tail_energy: f32,
    pub f_left: usize,
    pub f_right: usize,
}

/// The factored-tail shipping form `ΔW = U_⊥ C V_⊥ᵀ`.
///
/// Both tail bases are stored LITERAL (row-major, `m×f_left` / `n×f_right`):
/// the narrow side's is the anchor's free complement; the wider side's is
/// the σ-propagated, σ-visible sub-basis (module docs — full wide-tail
/// coverage needs the wide Gram, out of scope by cost). Any sub-basis of
/// the tail is still exactly tail-confined, so confinement survives the
/// truncation; the dropped energy rides [`SstFactorReport`], never hidden.
///
/// **The quantization-safety property:** decode is `U_⊥·C·V_⊥ᵀ` — ANY `C`
/// (including a quantized one) decodes into the tail span, so the decoded
/// delta's protected fraction stays at the f32 floor. Dense-project-then-
/// quantize has no such closure (the isotropic-rounding negative in the
/// G1 battery proves the certifier catches it).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TailFactor {
    /// The issuing anchor's rotation-invariant digest.
    pub anchor_id: [u8; 32],
    pub m: usize,
    pub n: usize,
    pub f_left: usize,
    pub f_right: usize,
    /// Left tail basis, row-major `m × f_left`.
    pub left_basis: Vec<f32>,
    /// Right tail basis, row-major `n × f_right`.
    pub right_basis: Vec<f32>,
    /// Payload coordinates, row-major `f_left × f_right`.
    pub c: Vec<f32>,
    /// Additive relative-L2 leak bound carried through composition (the
    /// triangle inequality — norms add, energies do not). A fresh factor is
    /// exact (0.0); a quantizing consumer sets its measured payload error;
    /// [`compose_tail`] sums the bounds.
    pub leak_bound: f32,
}

impl TailFactor {
    /// Grow the factor's buffers to the requested shape (grow-only: after a
    /// warm call at the working shape, [`SstAnchor::factor_tail_into`] and
    /// [`compose_tail_into`] allocate nothing).
    pub fn ensure_shapes(&mut self, m: usize, n: usize, f_left: usize, f_right: usize) {
        ensure(&mut self.left_basis, m * f_left);
        ensure(&mut self.right_basis, n * f_right);
        ensure(&mut self.c, f_left * f_right);
        self.m = m;
        self.n = n;
        self.f_left = f_left;
        self.f_right = f_right;
    }

    /// Decode to the dense `m × n` delta (zero-alloc over `scratch`).
    /// `out = U_⊥·C·V_⊥ᵀ` in two passes through the scratch.
    pub fn decode_into(&self, out: &mut [f32], scratch: &mut SstScratch) {
        assert_eq!(out.len(), self.m * self.n, "decode_into: out shape");
        let (m, n, fl, fr) = (self.m, self.n, self.f_left, self.f_right);
        // T = C·Vᵀ (f_l×n) into t1; out = U·T (m×n).
        ensure(&mut scratch.t1, fl * n);
        {
            let (t, rb, c) = (
                &mut scratch.t1[..fl * n],
                &self.right_basis,
                &self.c[..fl * fr],
            );
            for t_row in 0..fl {
                for j in 0..n {
                    let mut acc = 0.0f32;
                    for s in 0..fr {
                        acc += c[t_row * fr + s] * rb[j * fr + s];
                    }
                    t[t_row * n + j] = acc;
                }
            }
        }
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for t in 0..fl {
                    acc += self.left_basis[i * fl + t] * scratch.t1[t * n + j];
                }
                out[i * n + j] = acc;
            }
        }
    }

    /// Apply the decoded delta to a column block: `out = Δ̂·X` (`x`:
    /// `n × cols`, `out`: `m × cols`) without materializing `Δ̂` —
    /// `U·(C·(Vᵀ·X))`, zero-alloc over `scratch`.
    pub fn apply_into(&self, x: &[f32], cols: usize, out: &mut [f32], scratch: &mut SstScratch) {
        assert_eq!(x.len(), self.n * cols, "apply_into: x shape");
        assert_eq!(out.len(), self.m * cols, "apply_into: out shape");
        let (m, n, fl, fr) = (self.m, self.n, self.f_left, self.f_right);
        ensure(&mut scratch.t1, fr * cols);
        ensure(&mut scratch.y2, fl * cols);
        // S = Vᵀ·X (f_r×cols)
        btx_into(
            &self.right_basis,
            fr,
            x,
            n,
            cols,
            &mut scratch.t1[..fr * cols],
        );
        // P = C·S (f_l×cols)
        {
            let (p, c, s) = (
                &mut scratch.y2[..fl * cols],
                &self.c[..fl * fr],
                &scratch.t1[..fr * cols],
            );
            for t in 0..fl {
                for j in 0..cols {
                    let mut acc = 0.0f32;
                    for s_i in 0..fr {
                        acc += c[t * fr + s_i] * s[s_i * cols + j];
                    }
                    p[t * cols + j] = acc;
                }
            }
        }
        // out = U·P (m×cols)
        for i in 0..m {
            for j in 0..cols {
                let mut acc = 0.0f32;
                for t in 0..fl {
                    acc += self.left_basis[i * fl + t] * scratch.y2[t * cols + j];
                }
                out[i * cols + j] = acc;
            }
        }
    }

    /// The quantized shipping copy: `C` rounded to the grid `step`
    /// (deterministic mid-rounding), bases shared. The copy's `leak_bound`
    /// is the MEASURED relative payload error `‖C − Q‖/‖C‖` — by Parseval
    /// that is exactly the decoded delta's tail-relative noise norm. The
    /// caller passes this bound through [`compose_tail`] and checks it
    /// against its tolerance budget.
    ///
    /// Shipping-time (allocating) by contract — it clones the factor.
    pub fn quantized(&self, step: f32) -> Result<TailFactor, SstError> {
        if !step.is_finite() || step <= 0.0 {
            return Err(SstError::BadQuantStep(step));
        }
        let mut q = self.clone();
        let inv = 1.0f32 / step;
        let mut err_sq = 0.0f32;
        let mut norm_sq = 0.0f32;
        for (v, out) in self.c.iter().zip(q.c.iter_mut()) {
            let rounded = (*v * inv).round() * step;
            *out = rounded;
            let d = *v - rounded;
            err_sq += d * d;
            norm_sq += *v * *v;
        }
        q.leak_bound = if norm_sq > 0.0 {
            err_sq.sqrt() / norm_sq.sqrt()
        } else {
            0.0
        };
        Ok(q)
    }
}

/// A certification verdict issued by
/// [`SstAnchor::certify_decoded_tail_into`] — bound to the issuing anchor's
/// rotation-invariant digest (Lane 6).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SstCertificate {
    pub anchor_id: [u8; 32],
    pub total_energy: f32,
    pub protected_energy: f32,
    pub protected_fraction: f32,
    pub rel_tol: f32,
}

impl SstCertificate {
    /// The re-anchor churn gate: a certificate verifies only against the
    /// anchor that issued it. Re-fitting on a drifted `W_ref` (or a
    /// different `free_min`) produces a different digest and EVERY
    /// outstanding certificate refuses loud — never a silent stale pass.
    ///
    /// Cost model (Lane 6): the re-anchor churn cost follows the paper's
    /// K-schedule curve family (re-anchor every R deltas ≈ +5.1% at R=100,
    /// +37.6% at R=10) — the churn DIAL is the consumer's; the substrate's
    /// contract is only that staleness refuses.
    pub fn verify_against(&self, anchor: &SstAnchor) -> Result<(), SstRefusal> {
        if self.anchor_id != anchor.id {
            return Err(SstRefusal::StaleAnchor {
                cert_anchor: self.anchor_id,
                live_anchor: anchor.id,
            });
        }
        Ok(())
    }
}

/// Compose chained factored-tail overlays (Lane 5): same-anchor factors
/// with BIT-IDENTICAL tail bases stay tail-confined under addition —
/// `U_⊥C₁V_⊥ᵀ + U_⊥C₂V_⊥ᵀ = U_⊥(C₁+C₂)V_⊥ᵀ`. The composite's `leak_bound`
/// is the sum of the links' bounds (triangle inequality). A link whose
/// bases differ (a re-anchor or a different truncation between links)
/// REFUSES loud naming the cause — re-factor every link against one anchor.
/// Zero-alloc over `out` after a warm call.
pub fn compose_tail_into(links: &[&TailFactor], out: &mut TailFactor) -> Result<(), SstError> {
    let Some(first) = links.first() else {
        return Err(SstError::ComposeMismatch("no links"));
    };
    for l in links {
        if l.anchor_id != first.anchor_id {
            return Err(SstError::ComposeMismatch(
                "anchor digest differs across links",
            ));
        }
        if (l.m, l.n, l.f_left, l.f_right) != (first.m, first.n, first.f_left, first.f_right) {
            return Err(SstError::ComposeMismatch(
                "factor shapes differ across links",
            ));
        }
        if l.left_basis[..l.m * l.f_left] != first.left_basis[..first.m * first.f_left]
            || l.right_basis[..l.n * l.f_right] != first.right_basis[..first.n * first.f_right]
        {
            return Err(SstError::ComposeMismatch(
                "tail bases differ (re-anchor or truncation) — re-factor every link against one anchor",
            ));
        }
        if !l.leak_bound.is_finite() {
            return Err(SstError::ComposeMismatch("non-finite link leak bound"));
        }
    }
    out.ensure_shapes(first.m, first.n, first.f_left, first.f_right);
    out.anchor_id = first.anchor_id;
    out.left_basis[..first.m * first.f_left]
        .copy_from_slice(&first.left_basis[..first.m * first.f_left]);
    out.right_basis[..first.n * first.f_right]
        .copy_from_slice(&first.right_basis[..first.n * first.f_right]);
    let fl_fr = first.f_left * first.f_right;
    out.c[..fl_fr].fill(0.0);
    for l in links {
        for (o, v) in out.c[..fl_fr].iter_mut().zip(l.c[..fl_fr].iter()) {
            *o += *v;
        }
    }
    out.leak_bound = links.iter().map(|l| l.leak_bound).sum();
    Ok(())
}

/// Convenience (allocating) form of [`compose_tail_into`].
pub fn compose_tail(links: &[&TailFactor]) -> Result<TailFactor, SstError> {
    let mut out = TailFactor::default();
    compose_tail_into(links, &mut out)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Deterministic fixture (the unseeded-global-RNG gate: a local LCG,
    // never a global draw).
    fn gauss_w(seed: u64, m: usize, n: usize) -> Vec<f32> {
        let mut rng = Lcg(seed);
        (0..m * n).map(|_| rng.next_gauss() as f32).collect()
    }

    fn eye(dim: usize) -> Vec<f32> {
        let mut e = vec![0.0f32; dim * dim];
        for i in 0..dim {
            e[i * dim + i] = 1.0;
        }
        e
    }

    fn mat_mul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for t in 0..k {
                    acc += a[i * k + t] * b[t * n + j];
                }
                out[i * n + j] = acc;
            }
        }
        out
    }

    /// Materialized test-side projector helpers — the brute-force oracle.
    fn bb_t(b: &[f32], dim: usize, f: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                let mut acc = 0.0f32;
                for t in 0..f {
                    acc += b[i * f + t] * b[j * f + t];
                }
                out[i * dim + j] = acc;
            }
        }
        out
    }

    /// The protected projectors (left m×m, right n×n) materialized.
    fn materialized_protected(anchor: &SstAnchor) -> (Vec<f32>, Vec<f32>) {
        let pl = match &anchor.left {
            SideBasis::Complement { basis } => {
                let f = basis.len() / anchor.m;
                let bbt = bb_t(basis, anchor.m, f);
                let eye = eye(anchor.m);
                eye.iter().zip(bbt.iter()).map(|(e, p)| e - p).collect()
            }
            SideBasis::Protected { basis } => bb_t(basis, anchor.m, anchor.k),
        };
        let pr = match &anchor.right {
            SideBasis::Complement { basis } => {
                let f = basis.len() / anchor.n;
                let bbt = bb_t(basis, anchor.n, f);
                let eye = eye(anchor.n);
                eye.iter().zip(bbt.iter()).map(|(e, p)| e - p).collect()
            }
            SideBasis::Protected { basis } => bb_t(basis, anchor.n, anchor.k),
        };
        (pl, pr)
    }

    fn two_sided(x: &[f32], pl: &[f32], pr: &[f32], m: usize, n: usize) -> Vec<f32> {
        let t = mat_mul(pl, x, m, m, n);
        mat_mul(&t, pr, m, n, n)
    }

    fn rel_err(a: &[f32], b: &[f32]) -> f32 {
        let num: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum();
        let den: f32 = b.iter().map(|v| v * v).sum();
        if den > 0.0 {
            num.sqrt() / den.sqrt()
        } else {
            num
        }
    }

    const W_M: usize = 8;
    const W_N: usize = 10;
    const FREE: usize = 3;

    fn fixture() -> (Vec<f32>, SstAnchor) {
        let w = gauss_w(0x5958_0001, W_M, W_N);
        let anchor = fit_anchor(&w, W_M, W_N, FREE).expect("fixture anchor fits");
        (w, anchor)
    }

    #[test]
    fn rust_spec_match_of_the_lean_two_by_two_counterexample() {
        // The Lean arm `KatgptProof.Sst.two_by_two_diagonal_leaks`: the
        // protected span span{![1, 1]} under diag(1, 0) — the protected
        // vector ![1, 1] transforms to ![1, 0], whose second coordinate (0)
        // is incompatible with any multiple r·![1, 1] (r = 0 and r = 1
        // simultaneously). The Rust side reads the same fact off a diagonal
        // "anchor": with the protected basis = ![1, 1]/√2 and the transform
        // diag(1, 0), the transformed protected vector's complement
        // component (the leak the certifier measures) is nonzero.
        // (The exact 2×2 statement is decided in Lean; this mirror pins the
        // Rust projector machinery against the same instance.)
        let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
        let protected = inv_sqrt2; // the single protected direction's norm-1 vector (1-D span)
        // The anchor this instance describes (kept as documentation; the
        // protected-apply is module-private, so the leak is computed by
        // hand below rather than through it). P_prot = B Bᵀ on ![1, 1]: the
        // projection is the WHOLE vector (it IS the protected direction).
        let _side = SideBasis::Protected {
            basis: vec![protected, protected],
        };
        // The transform diag(1, 0) applied to ![1, 1] is ![1, 0]; its
        // protected projection is ![0.5, 0.5], so the leak (complement
        // component) has norm² = ‖![1,0] − [0.5,0.5]‖² = 0.5 — nonzero,
        // matching the Lean arm's unresolvable-coordinate contradiction.
        let transformed = [1.0f32, 0.0];
        let proj = [protected * protected, protected * protected]; // = [0.5, 0.5]
        let leak: f32 = transformed
            .iter()
            .zip(proj.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        assert!((leak - 0.5).abs() < 1e-6, "leak² {leak} != 0.5");
    }

    #[test]
    fn width_guard_refuses_empty_protected_span() {
        let w = gauss_w(1, 6, 6);
        // free_min == min → k == 0 → the SST non-case.
        assert_eq!(
            fit_anchor(&w, 6, 6, 6),
            Err(SstError::NoProtectedSpan {
                min_dim: 6,
                free_min: 6
            })
        );
        // free_min > min → the checked_sub underflow arm, same verdict.
        assert_eq!(
            fit_anchor(&w, 6, 6, 9),
            Err(SstError::NoProtectedSpan {
                min_dim: 6,
                free_min: 9
            })
        );
    }

    #[test]
    fn disarmed_k_equals_min_factor_decodes_to_zeros() {
        // f_min = 0 protects EVERYTHING: the tail span is empty, the factor's
        // payload is 0×0, and the decoded rebuild is the zero matrix — the
        // disarmed posture pinned byte-identically.
        let w = gauss_w(2, 6, 6);
        let anchor = fit_anchor(&w, 6, 6, 0).expect("k=min anchor");
        assert_eq!(anchor.k, 6);
        assert_eq!(anchor.f_narrow, 0);
        let delta = gauss_w(3, 6, 6);
        let (factor, report) = anchor.factor_tail(&w, &delta, 0, 0).expect("factor");
        assert_eq!((factor.f_left, factor.f_right), (0, 0));
        let mut decoded = vec![0.0f32; 36];
        let mut s = SstScratch::default();
        factor.decode_into(&mut decoded, &mut s);
        assert!(
            decoded.iter().all(|v| *v == 0.0),
            "disarmed decode must be zeros"
        );
        assert_eq!(report.shipped_energy, 0.0);
    }

    #[test]
    fn anchor_digest_invariant_under_basis_sign_flip_and_refit() {
        // (a) The projector digest is EXACTLY invariant under basis sign
        //     flips (the products b_j·(b_jᵀx) are unchanged, same order).
        let basis: Vec<f32> = gauss_w(4, 6, 2);
        let mut flipped = basis.clone();
        for i in 0..6 {
            flipped[i * 2 + 1] = -flipped[i * 2 + 1];
        }
        let mut s = SstScratch::default();
        let d1 = side_digest(
            &SideBasis::Complement { basis },
            6,
            &mut Lcg(DIGEST_LCG_SEED),
            &mut s,
        );
        let d2 = side_digest(
            &SideBasis::Complement { basis: flipped },
            6,
            &mut Lcg(DIGEST_LCG_SEED),
            &mut s,
        );
        assert_eq!(d1, d2, "sign flip must not move the projector digest");

        // (b) Same-device refit determinism: the same W fits to the same id.
        let (w, anchor) = fixture();
        let refit = fit_anchor(&w, W_M, W_N, FREE).expect("refit");
        assert_eq!(anchor.id, refit.id);
        assert_eq!(anchor.left, refit.left);

        // (c) A different W produces a different digest.
        let w2 = gauss_w(5, W_M, W_N);
        let other = fit_anchor(&w2, W_M, W_N, FREE).expect("other");
        assert_ne!(anchor.id, other.id);
    }

    #[test]
    fn anatomy_splits_protected_and_tail_energy() {
        let (_w, anchor) = fixture();
        let w = gauss_w(0x5958_0001, W_M, W_N);
        let (pl, pr) = materialized_protected(&anchor);
        let s = SstScratch::default();
        let _ = s;

        // The fully-protected component: fraction ≈ 1.
        let prot = two_sided(&w, &pl, &pr, W_M, W_N);
        let a_prot = anchor.anatomy(&prot).expect("anatomy");
        assert!(
            a_prot.protected_fraction > 0.9999,
            "protected component fraction {}, want ~1",
            a_prot.protected_fraction
        );

        // The fully-tail component: fraction ≈ 0 (the two-positive-terms
        // form must not cancel).
        let ilt = eye(W_M)
            .iter()
            .zip(pl.iter())
            .map(|(e, p)| e - p)
            .collect::<Vec<f32>>();
        let irt = eye(W_N)
            .iter()
            .zip(pr.iter())
            .map(|(e, p)| e - p)
            .collect::<Vec<f32>>();
        let tail = two_sided(&w, &ilt, &irt, W_M, W_N);
        let a_tail = anchor.anatomy(&tail).expect("anatomy");
        assert!(
            a_tail.protected_fraction < 1e-4,
            "tail component fraction {}, want ~0",
            a_tail.protected_fraction
        );
        assert!(a_tail.free_energy > 0.0);
    }

    #[test]
    fn factor_decode_roundtrip_matches_the_shipped_tail_projection() {
        let (w, anchor) = fixture();
        let (pl, pr) = materialized_protected(&anchor);
        let ilt = eye(W_M)
            .iter()
            .zip(pl.iter())
            .map(|(e, p)| e - p)
            .collect::<Vec<f32>>();
        let irt = eye(W_N)
            .iter()
            .zip(pr.iter())
            .map(|(e, p)| e - p)
            .collect::<Vec<f32>>();
        // A tail-confined delta (planted through the FULL projectors).
        let delta = two_sided(&gauss_w(6, W_M, W_N), &ilt, &irt, W_M, W_N);

        let (factor, report) = anchor.factor_tail(&w, &delta, FREE, FREE).expect("factor");
        assert_eq!(factor.anchor_id, anchor.id);

        // The shipped right tail is the σ-VISIBLE truncation (module docs:
        // ≤ min−k directions at a wide side), so decode reproduces
        // `U Uᵀ Δ V Vᵀ` — the projection onto the SHIPPED tail — exactly.
        let vvt = bb_t(&factor.right_basis, W_N, factor.f_right);
        let brute = two_sided(&delta, &ilt, &vvt, W_M, W_N);
        let mut decoded = vec![0.0f32; W_M * W_N];
        let mut s = SstScratch::default();
        factor.decode_into(&mut decoded, &mut s);
        assert!(
            rel_err(&decoded, &brute) < 1e-4,
            "decode vs shipped-tail projection err {}",
            rel_err(&decoded, &brute)
        );
        // Decode is tail-confined (the certifier's own read).
        let cert = anchor
            .certify_decoded_tail(&decoded, 1e-3)
            .expect("decode must certify");
        assert!(cert.protected_fraction < 1e-4);
        // The report accounts for every unit of energy: shipped + dropped =
        // tail (within fp), and shipped matches ‖decode‖² (Parseval).
        let decode_sq = decoded.iter().map(|v| v * v).sum::<f32>();
        assert!((report.shipped_energy - decode_sq).abs() < 1e-3 * report.tail_energy.max(1e-9));
        assert!(
            (report.shipped_energy + report.dropped_tail_energy - report.tail_energy).abs()
                < 1e-3 * report.tail_energy.max(1e-9),
            "shipped + dropped must account for the tail"
        );
    }

    #[test]
    fn quantized_factor_stays_tail_confined_and_certifies() {
        let (w, anchor) = fixture();
        let (pl, pr) = materialized_protected(&anchor);
        let ilt = eye(W_M)
            .iter()
            .zip(pl.iter())
            .map(|(e, p)| e - p)
            .collect::<Vec<f32>>();
        let irt = eye(W_N)
            .iter()
            .zip(pr.iter())
            .map(|(e, p)| e - p)
            .collect::<Vec<f32>>();
        let delta = two_sided(&gauss_w(7, W_M, W_N), &ilt, &irt, W_M, W_N);
        let (factor, _) = anchor.factor_tail(&w, &delta, FREE, FREE).expect("factor");

        // Quantize the payload at a coarse grid; decode; the decoded delta
        // is STILL tail-confined (the span is closed under the product).
        let step = 0.05f32;
        let q = factor.quantized(step).expect("quantize");
        assert!(q.leak_bound > 0.0, "coarse grid must measure a bound");
        let mut decoded = vec![0.0f32; W_M * W_N];
        let mut s = SstScratch::default();
        q.decode_into(&mut decoded, &mut s);
        let cert = anchor
            .certify_decoded_tail(&decoded, 1e-3)
            .expect("quantized factor must certify");
        assert!(cert.protected_fraction < 1e-5);
        cert.verify_against(&anchor).expect("live cert");
    }

    #[test]
    fn certifier_refuses_planted_leak_with_measured_magnitude() {
        let (_w, anchor) = fixture();
        let (pl, pr) = materialized_protected(&anchor);
        let ilt = eye(W_M)
            .iter()
            .zip(pl.iter())
            .map(|(e, p)| e - p)
            .collect::<Vec<f32>>();
        let irt = eye(W_N)
            .iter()
            .zip(pr.iter())
            .map(|(e, p)| e - p)
            .collect::<Vec<f32>>();
        let tail = two_sided(&gauss_w(8, W_M, W_N), &ilt, &irt, W_M, W_N);
        let prot = two_sided(&gauss_w(9, W_M, W_N), &pl, &pr, W_M, W_N);
        // Plant a leak with ‖prot‖ / ‖tail‖ = 1/3 → fraction ≈ 0.1.
        let alpha = 1.0f32 / 3.0;
        let mut decoded = vec![0.0f32; W_M * W_N];
        for i in 0..decoded.len() {
            decoded[i] = tail[i] + alpha * prot[i];
        }
        match anchor.certify_decoded_tail(&decoded, 1e-3) {
            Err(SstRefusal::LeakAboveTolerance {
                protected_fraction,
                rel_tol,
            }) => {
                assert_eq!(rel_tol, 1e-3);
                let expected = alpha * alpha / (1.0 + alpha * alpha);
                assert!(
                    (protected_fraction - expected).abs() < 0.05,
                    "measured leak {protected_fraction}, planted ~{expected}"
                );
            }
            other => panic!("planted leak must refuse, got {other:?}"),
        }
    }

    #[test]
    fn compose_sums_payloads_bounds_and_refuses_mismatch() {
        let (w, anchor) = fixture();
        let d1 = gauss_w(10, W_M, W_N);
        let d2 = gauss_w(11, W_M, W_N);
        let (mut f1, _) = anchor.factor_tail(&w, &d1, FREE, FREE).expect("f1");
        let (f2, _) = anchor.factor_tail(&w, &d2, FREE, FREE).expect("f2");
        f1.leak_bound = 0.25;
        let composed = compose_tail(&[&f1, &f2]).expect("compose");
        assert!((composed.leak_bound - 0.25).abs() < 1e-6);
        let mut cd = vec![0.0f32; W_M * W_N];
        let mut s = SstScratch::default();
        composed.decode_into(&mut cd, &mut s);
        // Composition is linear in the payload: the composite decodes to the
        // SUM OF THE LINK DECODES (each link ships its own tail truncation).
        let mut d1 = vec![0.0f32; W_M * W_N];
        let mut d2 = vec![0.0f32; W_M * W_N];
        f1.decode_into(&mut d1, &mut s);
        f2.decode_into(&mut d2, &mut s);
        let mut sum = vec![0.0f32; W_M * W_N];
        for i in 0..sum.len() {
            sum[i] = d1[i] + d2[i];
        }
        assert!(
            rel_err(&cd, &sum) < 1e-4,
            "compose decode = sum of link decodes"
        );

        // An anchor mismatch refuses loud.
        let w2 = gauss_w(12, W_M, W_N);
        let other = fit_anchor(&w2, W_M, W_N, FREE).expect("other");
        let (f3, _) = other.factor_tail(&w2, &d1, FREE, FREE).expect("f3");
        assert!(compose_tail(&[&f1, &f3]).is_err());

        // A basis-byte difference (same anchor, rotated re-fit) refuses loud.
        let mut f4 = f2.clone();
        f4.left_basis[0] += 1e-6;
        assert!(compose_tail(&[&f1, &f4]).is_err());
    }

    #[test]
    fn stale_certificate_refuses_after_re_anchor() {
        let (_w, anchor) = fixture();
        // Certify a tail-confined delta — a random dense delta is mostly
        // protected-touching BY THE LAW the module serves (~85% here), and
        // the certifier correctly refuses it at 1e-2.
        let (pl, pr) = materialized_protected(&anchor);
        let ilt = eye(W_M)
            .iter()
            .zip(pl.iter())
            .map(|(e, p)| e - p)
            .collect::<Vec<f32>>();
        let irt = eye(W_N)
            .iter()
            .zip(pr.iter())
            .map(|(e, p)| e - p)
            .collect::<Vec<f32>>();
        let tail = two_sided(&gauss_w(13, W_M, W_N), &ilt, &irt, W_M, W_N);
        let cert = anchor
            .certify_decoded_tail(&tail, 1e-2)
            .expect("tail delta certifies");
        cert.verify_against(&anchor).expect("live");
        // A re-anchor (different W) invalidates the outstanding cert.
        let w2 = gauss_w(14, W_M, W_N);
        let re_anchored = fit_anchor(&w2, W_M, W_N, FREE).expect("re-anchor");
        match cert.verify_against(&re_anchored) {
            Err(SstRefusal::StaleAnchor {
                cert_anchor,
                live_anchor,
            }) => {
                assert_eq!(cert_anchor, anchor.id);
                assert_eq!(live_anchor, re_anchored.id);
            }
            other => panic!("stale cert must refuse, got {other:?}"),
        }
    }
}
