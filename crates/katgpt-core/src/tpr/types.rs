//! Decoupled types for the TPR binding-algebra primitive (Issue 707 /
//! Research 527): error vocabulary, role schemes, the frozen BLAKE3-committed
//! artifact, and the ALS config/report records. Runtime algebra lives in
//! [`super`]; the fit lives in [`super::als`].

use std::fmt;

/// Artifact schema version — bump on any wire/layout change.
pub const SCHEMA_VERSION: u32 = 1;

/// Upper bound on the cached projection Gram (`K×K`, `K = m·d`): below it the
/// Cholesky ships inside the artifact; above it `project_into` refuses rather
/// than silently allocating a ~4·K² f32 blob per fit (K = 1024 → 4 MiB; K =
/// 6144 → 151 MiB). Runtime ops are unaffected — only the denoise readout is
/// unavailable.
pub const TPR_MAX_PROJECTION_K: usize = 1024;

/// Error vocabulary for fit + runtime ops. Hand-written `Display` (std-only —
/// the `tpr` feature pulls zero deps beyond what katgpt-core already carries).
#[derive(Debug, Clone, PartialEq)]
pub enum TprError {
    /// A slice length does not match the artifact/input geometry it was
    /// handed to.
    DimMismatch {
        what: &'static str,
        expected: usize,
        got: usize,
    },
    /// `project_into` on an artifact whose cached inverse was not built
    /// (K > [`TPR_MAX_PROJECTION_K`], `build_projection = false`, or the Gram
    /// was not positive definite at fit time).
    ProjectionUnavailable,
    /// Input referenced a role id / filler id outside the artifact.
    BadId {
        what: &'static str,
        max: usize,
        got: usize,
    },
    /// Malformed artifact bytes (bad version, truncated body, length
    /// mismatch).
    BadEncoding(&'static str),
    /// The stored commitment does not match the recomputed one — the artifact
    /// was tampered with or written by a different schema version.
    CommitmentMismatch,
    /// The primitive is disabled at runtime via `RIIR_TPR=0` (the G3
    /// kill-switch).
    Disabled,
    /// A fit produced non-finite values (degenerate input, e.g. an
    /// all-constant state column or a zero filler table).
    NonFinite(&'static str),
}

impl fmt::Display for TprError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TprError::DimMismatch {
                what,
                expected,
                got,
            } => write!(
                f,
                "tpr: {what} length mismatch: expected {expected}, got {got}"
            ),
            TprError::ProjectionUnavailable => write!(
                f,
                "tpr: projection Cholesky unavailable on this artifact \
                 (K > TPR_MAX_PROJECTION_K, build_projection = false, or \
                 non-PD Gram at fit time)"
            ),
            TprError::BadId { what, max, got } => {
                write!(f, "tpr: {what} id {got} out of range (max {max})")
            }
            TprError::BadEncoding(why) => write!(f, "tpr: bad artifact encoding: {why}"),
            TprError::CommitmentMismatch => {
                write!(f, "tpr: BLAKE3 commitment mismatch — artifact tampered or foreign schema")
            }
            TprError::Disabled => write!(f, "tpr: disabled via RIIR_TPR=0 kill-switch"),
            TprError::NonFinite(what) => write!(f, "tpr: non-finite value in {what}"),
        }
    }
}

impl std::error::Error for TprError {}

/// Role scheme. `arity` (`m`) is BOTH the core block count and (for
/// [`TprScheme::RoleVectors`]) the role-vector dimension: the core vec
/// `c ∈ R^{m·d}` is role-major (block `i` = coords `[i·d, (i+1)·d)`), and a
/// binding with role id `p` contributes `r_p[i]·f` to block `i`.
#[derive(Debug, Clone, PartialEq)]
pub enum TprScheme {
    /// One-hot role blocks: binding role id `p` occupies block `p` exactly.
    /// Unbind basis = the canonical basis; crosstalk μ = 0; recovery exact.
    Orthogonal {
        /// Number of role blocks (`m`).
        arity: usize,
    },
    /// General role vectors in `R^m` (the DISCOVER pair-role shape, e.g.
    /// `subject_adj-object_noun`). `n_bind_slots` distinct role ids, each an
    /// `arity`-length row of `roles` (row-major `n_bind_slots × arity`).
    /// At most `arity` distinct slots are supported for unbind (the square
    /// case) — the fit validates this.
    RoleVectors {
        /// Role-vector dimension and core block count (`m`).
        arity: usize,
        /// `n_bind_slots × arity` row-major role vectors. Fitted by the ALS
        /// roles block; the fit clones and owns its working copy.
        roles: Vec<f32>,
    },
}

impl TprScheme {
    /// Core block count / role-vector dimension.
    pub fn arity(&self) -> usize {
        match self {
            Self::Orthogonal { arity } => *arity,
            Self::RoleVectors { arity, .. } => *arity,
        }
    }

    /// Number of distinct bindable role ids.
    pub fn n_bind_slots(&self) -> usize {
        match self {
            Self::Orthogonal { arity } => *arity,
            Self::RoleVectors { arity, roles } => {
                debug_assert_eq!(roles.len() % (*arity).max(1), 0);
                roles.len() / (*arity).max(1)
            }
        }
    }

    /// Role vector for bind slot `p` (length `arity`); `None` for
    /// [`TprScheme::Orthogonal`] (the one-hot is implicit).
    pub fn role_vec(&self, p: u16) -> Option<&[f32]> {
        match self {
            Self::Orthogonal { .. } => None,
            Self::RoleVectors { arity, roles } => {
                let a = *arity;
                let start = p as usize * a;
                roles.get(start..start + a)
            }
        }
    }

    pub(crate) fn set_role_vec(&mut self, p: usize, src: &[f32]) {
        if let Self::RoleVectors { arity, roles } = self {
            let a = *arity;
            roles[p * a..(p + 1) * a].copy_from_slice(src);
        }
    }
}

/// Per-state bindings: parallel vectors of role ids and filler ids. A binding
/// `(roles[i], fillers[i])` contributes `Σ_i r_p[i]·(W^{(i)}·f_v)` to the
/// state (one-hot roles: exactly `W^{(p)}·f_v`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TprBindings {
    pub roles: Vec<u16>,
    pub fillers: Vec<u16>,
}

impl TprBindings {
    pub fn len(&self) -> usize {
        debug_assert_eq!(self.roles.len(), self.fillers.len());
        self.roles.len()
    }

    pub fn is_empty(&self) -> bool {
        self.roles.is_empty()
    }

    /// Zip-constructor (the common shape: role/filler id pairs).
    pub fn from_pairs(pairs: &[(u16, u16)]) -> Self {
        Self {
            roles: pairs.iter().map(|&(r, _)| r).collect(),
            fillers: pairs.iter().map(|&(_, f)| f).collect(),
        }
    }
}

/// The frozen, BLAKE3-committed TPR artifact (Issue 707 T5): the fitted
/// encoder `W ∈ R^{D×(m·d)}` (role-major column blocks), bias, the shared
/// filler table, the scheme (+ fitted roles + signed unbind basis), the
/// offline certificate scalars, and the optional cached projection Cholesky.
///
/// Freeze/thaw contract: [`TprArtifact::freeze`] computes the commitment over
/// the canonical byte encoding; [`TprArtifact::verify`] recomputes it.
/// [`TprArtifact::to_bytes`] / [`TprArtifact::from_bytes`] round-trip the
/// whole artifact with the commitment checked on thaw.
#[derive(Debug, Clone, PartialEq)]
pub struct TprArtifact {
    pub version: u32,
    /// Entity/state dimension `D`.
    pub dim: usize,
    /// Filler dimension `d`.
    pub d: usize,
    /// Core block count / role arity `m`.
    pub m: usize,
    /// Filler-table row count.
    pub n_fillers: usize,
    /// Encoder, stored **role-block-major, column-slice**: role block `i`
    /// occupies `w[i·D·d .. (i+1)·D·d]`, and INSIDE that block the `j`-th
    /// filler coordinate's column `W_i[:, j] ∈ R^D` is the contiguous run
    /// `[j·D, (j+1)·D)`.
    ///
    /// This is T1's "per-role pre-sliced `W_r ∈ R^{D×d}`", transposed, and
    /// every part of it is a MEASURED choice (D=768, d=8, 1 µs surgery bar):
    ///
    /// | layout | bind p50 | why |
    /// |---|---:|---|
    /// | `D × (m·d)` row-major | 2708 ns | strides `d` floats per row; touches `m×` the cache lines it reads |
    /// | block-major `D × d` | 2625 ns | contiguous, but each row is a length-`d` REDUCTION — dependency-chain bound |
    /// | block-major, `d × D` columns | ~250 ns | `d` contiguous axpys, no reduction, `out` stays in L1 |
    ///
    /// The column-slice form also makes the two cold ops contiguous: decode
    /// (`W·c`) is `K` axpys, and `Wᵀ(e−b)` is `K` contiguous length-`D` dot
    /// products — so nothing pays for the hot path's layout.
    pub w: Vec<f32>,
    /// State bias, length `D`.
    pub bias: Vec<f32>,
    /// Shared filler table, row-major `n_fillers × d` — ONE vector per filler
    /// across ALL roles (the systematicity carrier: a filler observed only
    /// under other roles still has a learned row, which is what the
    /// withheld-pair OOD gate measures).
    pub fillers: Vec<f32>,
    /// Role scheme (fitted roles inside for [`TprScheme::RoleVectors`]).
    pub scheme: TprScheme,
    /// Signed + permuted unbind basis, `m × m` **column-major**
    /// (`basis[p*m + col]`): column `p` is the (signed) unbind direction for
    /// bind slot `p`. `Some` only for [`TprScheme::RoleVectors`] — the
    /// orthogonal scheme unbinds by exact slice copy.
    pub unbind_basis: Option<Vec<f32>>,
    /// Crosstalk coherence `μ = max_{p≠q} |⟨r̂_q, B_p⟩|` — the UNIT-NORMALIZED
    /// role `q` against unbind basis column `p`, so `μ ∈ [0, 1]` is a true
    /// coherence and the `μ(m−1)max‖f‖` envelope carries the filler scale
    /// exactly once. 0 for the orthogonal scheme (exact recovery).
    pub crosstalk_mu: f32,
    /// `min_p ⟨r_p, B_p⟩` — the recovery scale (≈ ±1 for near-orthogonal
    /// schemes).
    pub unbind_diag_min: f32,
    /// `max_v ‖f_v‖` over the fitted filler table — the scale term of the
    /// unbind crosstalk envelope.
    pub max_filler_norm: f32,
    /// Offline fit-residual certificate over the fit corpus (state-space L2
    /// norms): median / 99th percentile / max.
    pub residual_p50: f32,
    pub residual_p99: f32,
    pub residual_max: f32,
    /// `Σ‖residual‖² / Σ‖e − ē‖²` — the fraction of centered state energy the
    /// fitted manifold does NOT capture (the a-priori denoise certificate).
    pub residual_energy_fraction: f32,
    pub n_fit_states: usize,
    /// Frozen structure label from the BIC scheme selection (T7), or the fit
    /// scheme's label when BIC was not run.
    pub bic_label: String,
    /// Final objective (sum of squared residuals) and sweep count.
    pub fit_objective: f64,
    pub als_sweeps: u32,
    /// Sweep proposals REJECTED by the descent guard (0 = every sweep
    /// descended). A rejection rolls the parameters back and stops the fit,
    /// so the artifact is always the minimum of the recorded trajectory —
    /// check `fit_objective == min(ssr_per_sweep)` rather than trusting this
    /// counter alone.
    pub monotone_violations: u32,
    /// Filler-table dimensions killed by the L2,1 prune step (0 unless
    /// [`L21::PruneRefit`]).
    pub pruned_dims: usize,
    /// Cached `(WᵀW + λI)⁻¹` (`K×K` row-major, `K = m·d`) — present iff
    /// `K ≤ TPR_MAX_PROJECTION_K` and the Gram was PD at fit time.
    ///
    /// The fit still FACTORS the Gram (that is where positive-definiteness is
    /// checked, via `linalg::spd_inverse_f32`'s Cholesky); it ships the
    /// explicit inverse because the runtime readout is `C⁻¹t` and a
    /// triangular solve at K=32 is a 32-step serial dependency chain —
    /// measured, it put the whole projection at 2.14× its two-GEMV floor
    /// against a 2× bar. One `K×K` matvec has no such chain, and the ridge
    /// keeps `C` well-conditioned enough that the explicit inverse is a
    /// denoise readout, not an ill-posed solve.
    pub projection_inv: Option<Vec<f32>>,
    pub projection_lambda: f32,
    /// BLAKE3 commitment over the canonical bytes; zeroed until
    /// [`TprArtifact::freeze`].
    pub commitment: [u8; 32],
}

impl TprArtifact {
    /// `m·d` — the core-vector length.
    pub fn core_len(&self) -> usize {
        self.m * self.d
    }

    /// Start offset of role block `i` inside [`TprArtifact::w`]. Column `j`
    /// of that block is then `[block_offset(i) + j·D, … + D)`.
    pub(crate) fn block_offset(&self, i: usize) -> usize {
        i * self.dim * self.d
    }


    /// Recompute the BLAKE3 commitment over the canonical bytes (excluding
    /// the stored commitment itself).
    pub fn commitment(&self) -> [u8; 32] {
        let mut buf = Vec::with_capacity(64 * 1024);
        self.canonical_bytes(&mut buf);
        let mut h = blake3::Hasher::new();
        h.update(&(buf.len() as u64).to_le_bytes());
        h.update(&buf);
        *h.finalize().as_bytes()
    }

    /// Populate [`TprArtifact::commitment`].
    pub fn freeze(&mut self) {
        self.commitment = self.commitment();
    }

    /// Recompute-and-compare the commitment. A tampered artifact fails here.
    pub fn verify(&self) -> bool {
        self.commitment == self.commitment()
    }

    /// Canonical byte encoding (LE, fixed field order). `include_commitment`
    /// = false yields the commitment preimage; true the full wire form.
    fn canonical_bytes(&self, out: &mut Vec<u8>) {
        let w8 = |out: &mut Vec<u8>, v: u32| out.extend_from_slice(&v.to_le_bytes());
        let wf = |out: &mut Vec<u8>, v: f32| out.extend_from_slice(&v.to_le_bytes());
        let wd = |out: &mut Vec<u8>, v: f64| out.extend_from_slice(&v.to_le_bytes());
        let wsl = |out: &mut Vec<u8>, s: &[f32]| {
            w8(out, s.len() as u32);
            for &v in s {
                wf(out, v);
            }
        };

        w8(out, self.version);
        w8(out, self.dim as u32);
        w8(out, self.d as u32);
        w8(out, self.m as u32);
        w8(out, self.n_fillers as u32);

        match &self.scheme {
            TprScheme::Orthogonal { arity } => {
                w8(out, 0);
                w8(out, *arity as u32);
                w8(out, 0);
            }
            TprScheme::RoleVectors { arity, roles } => {
                w8(out, 1);
                w8(out, *arity as u32);
                w8(out, (roles.len() / (*arity).max(1)) as u32);
                wsl(out, roles);
            }
        }

        wsl(out, &self.w);
        wsl(out, &self.bias);
        wsl(out, &self.fillers);
        match &self.unbind_basis {
            None => w8(out, 0),
            Some(b) => {
                w8(out, 1);
                wsl(out, b);
            }
        }

        wf(out, self.crosstalk_mu);
        wf(out, self.unbind_diag_min);
        wf(out, self.max_filler_norm);
        wf(out, self.residual_p50);
        wf(out, self.residual_p99);
        wf(out, self.residual_max);
        wf(out, self.residual_energy_fraction);

        w8(out, self.n_fit_states as u32);
        w8(out, self.bic_label.len() as u32);
        out.extend_from_slice(self.bic_label.as_bytes());
        wd(out, self.fit_objective);
        w8(out, self.als_sweeps);
        w8(out, self.monotone_violations);
        w8(out, self.pruned_dims as u32);

        match &self.projection_inv {
            None => {
                w8(out, 0);
                wf(out, self.projection_lambda);
            }
            Some(l) => {
                w8(out, 1);
                wf(out, self.projection_lambda);
                wsl(out, l);
            }
        }
    }

    /// Full wire form (commitment included). Round-trips via
    /// [`TprArtifact::from_bytes`].
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(64 * 1024);
        self.canonical_bytes(&mut out);
        out.extend_from_slice(&self.commitment);
        out
    }

    /// Decode + verify. Any mismatch ([`TprError::BadEncoding`] /
    /// [`TprError::CommitmentMismatch`]) refuses the artifact.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TprError> {
        #[inline]
        fn need(pos: usize, n: usize, len: usize) -> Result<(), TprError> {
            if pos + n > len {
                Err(TprError::BadEncoding("truncated"))
            } else {
                Ok(())
            }
        }
        fn ru32(bytes: &[u8], pos: &mut usize) -> Result<u32, TprError> {
            need(*pos, 4, bytes.len())?;
            let v = u32::from_le_bytes(bytes[*pos..*pos + 4].try_into().unwrap());
            *pos += 4;
            Ok(v)
        }
        fn rf32(bytes: &[u8], pos: &mut usize) -> Result<f32, TprError> {
            need(*pos, 4, bytes.len())?;
            let v = f32::from_le_bytes(bytes[*pos..*pos + 4].try_into().unwrap());
            *pos += 4;
            Ok(v)
        }
        fn rf64(bytes: &[u8], pos: &mut usize) -> Result<f64, TprError> {
            need(*pos, 8, bytes.len())?;
            let v = f64::from_le_bytes(bytes[*pos..*pos + 8].try_into().unwrap());
            *pos += 8;
            Ok(v)
        }
        fn rsl(bytes: &[u8], pos: &mut usize) -> Result<Vec<f32>, TprError> {
            let n = ru32(bytes, pos)? as usize;
            need(*pos, n * 4, bytes.len())?;
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(rf32(bytes, pos)?);
            }
            Ok(v)
        }

        let mut pos = 0usize;
        let version = ru32(bytes, &mut pos)?;
        if version != SCHEMA_VERSION {
            return Err(TprError::BadEncoding("unsupported schema version"));
        }
        let dim = ru32(bytes, &mut pos)? as usize;
        let d = ru32(bytes, &mut pos)? as usize;
        let m = ru32(bytes, &mut pos)? as usize;
        let n_fillers = ru32(bytes, &mut pos)? as usize;
        if dim == 0 || d == 0 || m == 0 {
            return Err(TprError::BadEncoding("zero dimension"));
        }

        let tag = ru32(bytes, &mut pos)?;
        let scheme = match tag {
            0 => {
                let arity = ru32(bytes, &mut pos)? as usize;
                let _ = ru32(bytes, &mut pos)?;
                TprScheme::Orthogonal { arity }
            }
            1 => {
                let arity = ru32(bytes, &mut pos)? as usize;
                let n_roles = ru32(bytes, &mut pos)? as usize;
                let roles = rsl(bytes, &mut pos)?;
                if roles.len() != n_roles * arity {
                    return Err(TprError::BadEncoding("role table length"));
                }
                TprScheme::RoleVectors { arity, roles }
            }
            _ => return Err(TprError::BadEncoding("scheme tag")),
        };

        let w = rsl(bytes, &mut pos)?;
        let bias = rsl(bytes, &mut pos)?;
        let fillers = rsl(bytes, &mut pos)?;
        if w.len() != dim * m * d || bias.len() != dim || fillers.len() != n_fillers * d {
            return Err(TprError::BadEncoding("w/bias/fillers length"));
        }
        let unbind_basis = if ru32(bytes, &mut pos)? == 1 {
            let b = rsl(bytes, &mut pos)?;
            if b.len() != m * m {
                return Err(TprError::BadEncoding("unbind basis length"));
            }
            Some(b)
        } else {
            None
        };

        let crosstalk_mu = rf32(bytes, &mut pos)?;
        let unbind_diag_min = rf32(bytes, &mut pos)?;
        let max_filler_norm = rf32(bytes, &mut pos)?;
        let residual_p50 = rf32(bytes, &mut pos)?;
        let residual_p99 = rf32(bytes, &mut pos)?;
        let residual_max = rf32(bytes, &mut pos)?;
        let residual_energy_fraction = rf32(bytes, &mut pos)?;

        let n_fit_states = ru32(bytes, &mut pos)? as usize;
        let label_len = ru32(bytes, &mut pos)? as usize;
        need(pos, label_len, bytes.len())?;
        let bic_label = String::from_utf8(bytes[pos..pos + label_len].to_vec())
            .map_err(|_| TprError::BadEncoding("bic label utf8"))?;
        pos += label_len;
        let fit_objective = rf64(bytes, &mut pos)?;
        let als_sweeps = ru32(bytes, &mut pos)?;
        let monotone_violations = ru32(bytes, &mut pos)?;
        let pruned_dims = ru32(bytes, &mut pos)? as usize;

        let has_chol = ru32(bytes, &mut pos)?;
        let projection_lambda = rf32(bytes, &mut pos)?;
        let projection_inv = if has_chol == 1 {
            let l = rsl(bytes, &mut pos)?;
            if l.len() != m * d * m * d {
                return Err(TprError::BadEncoding("projection inverse length"));
            }
            Some(l)
        } else {
            None
        };

        need(pos, 32, bytes.len())?;
        let mut commitment = [0u8; 32];
        commitment.copy_from_slice(&bytes[pos..pos + 32]);

        let art = Self {
            version,
            dim,
            d,
            m,
            n_fillers,
            w,
            bias,
            fillers,
            scheme,
            unbind_basis,
            crosstalk_mu,
            unbind_diag_min,
            max_filler_norm,
            residual_p50,
            residual_p99,
            residual_max,
            residual_energy_fraction,
            n_fit_states,
            bic_label,
            fit_objective,
            als_sweeps,
            monotone_violations,
            pruned_dims,
            projection_inv,
            projection_lambda,
            commitment,
        };
        if !art.verify() {
            return Err(TprError::CommitmentMismatch);
        }
        Ok(art)
    }
}

/// L2,1 mode for the filler table (Issue 707 T5). The paper's finding:
/// L2,1 on the filler/role embedding matrices is load-bearing for OOD
/// generalization — it kills whole embedding dimensions, preventing
/// degenerate per-pair memorization. Both modes here are gradient-free.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum L21 {
    /// No regularization.
    Off,
    /// Post-fit prune + exact refit (the recommended route): kill every
    /// filler-table column whose cross-filler norm is below
    /// `tau_frac · max_norm`, then re-solve the `W, b` block exactly against
    /// the pruned cores. Exact zeros → sparser runtime GEMVs; the same
    /// dead-dim identification the penalty achieves.
    PruneRefit {
        /// Fraction of the max column norm below which a dimension dies
        /// (0.05 = kill anything under 5% of the strongest column).
        tau_frac: f32,
    },
    /// Reweighted-ridge MM (fallback): `iters` outer passes, each solving the
    /// filler block with per-coordinate diagonal weights
    /// `λ·1/(‖F[:,j]‖ + ε)` — the MM surrogate of the L2,1 penalty. Monotone
    /// in the *penalized* objective (tracked in
    /// [`AlsReport::monotone_violations`]).
    ReweightedMm {
        iters: u32,
        /// L2,1 penalty strength λ.
        lambda: f32,
    },
}

/// ALS fit configuration (Issue 707 T5). Every field is deterministic: the
/// same config + data produce bit-identical artifacts (G1 double-run gate).
#[derive(Debug, Clone)]
pub struct AlsConfig {
    /// Filler dimension `d`.
    pub d: usize,
    /// Initial role scheme. For [`TprScheme::RoleVectors`] the roles are the
    /// ALS initialization (fitted in place by the roles block); the canonical
    /// init is identity rows (each bind slot starts at its own block).
    pub scheme: TprScheme,
    /// Ridge λ for the `W, b` block Gram (`XᵀX + λI`).
    pub ridge_lambda: f32,
    /// Ridge λ for each per-filler / per-role `d×d` / `m×m` solve.
    pub filler_ridge: f32,
    /// Max ALS sweeps.
    pub max_sweeps: u32,
    /// Stop early when the relative objective improvement drops below this.
    pub tol: f64,
    pub l21: L21,
    /// Seed for the deterministic filler initialization (SplitMix64; the
    /// default is a fixed constant so double fits are bit-identical).
    pub lcg_seed: u64,
    /// Build the cached projection Cholesky (K ≤ [`TPR_MAX_PROJECTION_K`]).
    pub build_projection: bool,
}

impl AlsConfig {
    /// Deterministic defaults: λ = 1e-3 both blocks, 32 sweeps, tol 1e-6,
    /// L2,1 off, fixed seed, projection on.
    pub fn new(d: usize, scheme: TprScheme) -> Self {
        Self {
            d,
            scheme,
            ridge_lambda: 1e-3,
            filler_ridge: 1e-3,
            max_sweeps: 32,
            tol: 1e-6,
            l21: L21::Off,
            lcg_seed: 0x5350_4C17_7070_7071,
            build_projection: true,
        }
    }
}

/// Fit input: states (`N × D` row-major) + per-state bindings + the filler
/// vocabulary size. Borrows the data — the fit allocates only its own working
/// buffers (offline path; steady-state zero-alloc applies to the runtime ops,
/// not the fit).
#[derive(Debug, Clone, Copy)]
pub struct AlsInput<'a> {
    /// State dimension `D`.
    pub dim: usize,
    /// Number of distinct fillers (ids must be `< n_fillers`).
    pub n_fillers: usize,
    /// `N × D` row-major states.
    pub states: &'a [f32],
    /// One entry per state.
    pub bindings: &'a [TprBindings],
}

impl<'a> AlsInput<'a> {
    /// Number of states.
    pub fn n_states(&self) -> usize {
        self.states.len() / self.dim.max(1)
    }
}

/// Fit report (the certificate bundle the GOAT gates read).
#[derive(Debug, Clone, Default)]
pub struct AlsReport {
    /// Objective (Σ‖e − ŵ‖²) after each sweep.
    pub ssr_per_sweep: Vec<f64>,
    /// Sweep proposals rejected by the descent guard — an increase beyond
    /// `1e-9·ssr_0`, scaled by the INITIAL objective so f32 noise at the
    /// convergence floor is not mistaken for divergence. A rejection rolls
    /// the parameters back and ends the fit.
    pub monotone_violations: u32,
    pub final_ssr: f64,
    pub residual_p50: f32,
    pub residual_p99: f32,
    pub residual_max: f32,
    /// Residual energy / centered state energy.
    pub residual_energy_fraction: f32,
    /// Filler dims killed by [`L21::PruneRefit`].
    pub pruned_dims: usize,
    /// SSR increase caused by the prune (bounded by the prune contract,
    /// typically ≪1%).
    pub prune_ssr_increase: f64,
    pub sweeps: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_artifact() -> TprArtifact {
        let d = 4usize;
        let m = 3usize;
        let dim = 8usize;
        let n_fillers = 5usize;
        let k = m * d;
        TprArtifact {
            version: SCHEMA_VERSION,
            dim,
            d,
            m,
            n_fillers,
            w: (0..dim * k).map(|i| (i as f32) * 0.25 - 3.0).collect(),
            bias: (0..dim).map(|i| i as f32 * 0.5).collect(),
            fillers: (0..n_fillers * d).map(|i| (i as f32) * 0.1 - 1.0).collect(),
            scheme: TprScheme::Orthogonal { arity: m },
            unbind_basis: None,
            crosstalk_mu: 0.0,
            unbind_diag_min: 1.0,
            max_filler_norm: 1.5,
            residual_p50: 0.01,
            residual_p99: 0.03,
            residual_max: 0.05,
            residual_energy_fraction: 0.02,
            n_fit_states: 64,
            bic_label: "orthogonal:3".to_string(),
            fit_objective: 1.25,
            als_sweeps: 7,
            monotone_violations: 0,
            pruned_dims: 0,
            projection_inv: None,
            projection_lambda: 1e-3,
            commitment: [0u8; 32],
        }
    }

    #[test]
    fn scheme_slot_counts() {
        let o = TprScheme::Orthogonal { arity: 5 };
        assert_eq!(o.arity(), 5);
        assert_eq!(o.n_bind_slots(), 5);
        assert!(o.role_vec(0).is_none());

        let roles = vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let rv = TprScheme::RoleVectors {
            arity: 3,
            roles: roles.clone(),
        };
        assert_eq!(rv.arity(), 3);
        assert_eq!(rv.n_bind_slots(), 3);
        assert_eq!(rv.role_vec(2), Some(&roles[6..9]));
        assert_eq!(rv.role_vec(3), None);
    }

    #[test]
    fn freeze_verify_roundtrip() {
        let mut art = sample_artifact();
        assert!(!art.verify(), "zeroed commitment must not verify");
        art.freeze();
        assert!(art.verify());

        let bytes = art.to_bytes();
        let thawed = TprArtifact::from_bytes(&bytes).expect("round trip");
        assert!(thawed.verify());
        assert_eq!(thawed, art);

        // Tamper: flip one payload byte → commitment mismatch.
        let mut tampered = bytes.clone();
        let mid = (tampered.len() - 32) / 2;
        tampered[mid] ^= 0x01;
        assert!(matches!(
            TprArtifact::from_bytes(&tampered),
            Err(TprError::CommitmentMismatch) | Err(TprError::BadEncoding(_))
        ));
    }

    #[test]
    fn rolevector_artifact_roundtrip() {
        let mut art = sample_artifact();
        art.scheme = TprScheme::RoleVectors {
            arity: 3,
            roles: vec![1.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.5, 0.0, 1.0],
        };
        art.unbind_basis = Some(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        art.crosstalk_mu = 0.47;
        art.freeze();
        let thawed = TprArtifact::from_bytes(&art.to_bytes()).expect("round trip");
        assert_eq!(thawed, art);
    }

    #[test]
    fn display_is_informative() {
        let e = TprError::DimMismatch {
            what: "state",
            expected: 8,
            got: 3,
        };
        let s = e.to_string();
        assert!(s.contains("state") && s.contains("8") && s.contains("3"));
    }
}
