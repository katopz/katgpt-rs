//! Counter-anchored set admission — the modelless fan-out gate (Plan 599,
//! Research 564, arXiv:2603.06397 "R4T" reward distillation).
//!
//! Greedy set admission over a candidate pool scoring
//!
//! ```text
//! score(x) = g(x) + α·cos(x, q₀) + κ·log(1 + x̂ᵀM⁻¹x̂)
//! ```
//!
//! where `g` is the caller's modular quality prior, `α·cos` anchors the set
//! to the query, and `κ·log(1 + x̂ᵀM⁻¹x̂)` is the exact marginal log-det gain
//! `log|I + G + xx̂ᵀ| − log|I + G|` of the admitted dual Gram `G = X̂ᵀX̂`
//! (`M ≡ I + G`, `M⁻¹` maintained by Sherman–Morrison rank-1 updates —
//! O(d²) per admission, never a matrix inverse). Candidates colinear with
//! an admitted member (`cos > θ_coll`, the `ColinearityBatchGate` 0.95
//! precedent) are rejected outright.
//!
//! Post-hoc, an **exact** cosine-kernel Vendi certificate runs on the dual
//! Gram's eigenvalues — the K×K and d×d Grams share nonzero spectra
//! (eigenduality), so the pinned
//! [`crate::spectral_pencil::jacobi_eigen`] kernel diagonalizes the
//! maintained d×d `G` and [`crate::certified_frontier::vendi_diversity`]
//! reads the score. The incremental participation ratio `(tr G)²/tr(G²)` is
//! the eigensolve-free fast path maintained during admission.
//!
//! # The three anchors and their failure families (the T0.2 spec)
//!
//! Each anchor holds one degenerate family shut; zeroing it re-opens that
//! family (the L2 property), and a modular-only objective over a
//! duplicate-tolerant pool admits an effective-rank-1 set (L1):
//!
//! | Weight zeroed | Family reachable | Mechanism |
//! |---|---|---|
//! | `kappa_div = 0` | **paraphrase-collapse** (selection) | no log-det gain → near-duplicates win on `g` alone |
//! | `alpha_align = 0` | **semantic-drift** (selection) | no query pull → the highest-`g` cluster wins wherever it lives |
//! | `rho_vendi = 0` | **coordinate-gaming** (detection) | the certificate floor drops to 0 → a spectrally rank-1 set that passes every pairwise colinearity check is never flagged `collapsed` |
//!
//! The first two families are SELECTION failures (who gets admitted); the
//! third is a DETECTION failure (the report misses a degenerate set) —
//! `rho_vendi` weights the certificate floor, never the greedy, so
//! admission is ρ-invariant by construction. The certificate is the honesty
//! instrument, not a per-candidate reward (R4T's RL-trained certificate
//! head, replaced by an exact closed form; the optional Phase-2 exp-tilt
//! arm is the RL-fixed-point replacement — measured there, not assumed
//! here). With all three weights at defaults the full triple excludes all
//! three interiors at the spec worlds (the `l1_*` / `l2_*` tests).
//!
//! # Phase 2 — the fan-out construction side (query expansion)
//!
//! [`DIRECTION_BANK`] is the frozen 1 KB table (32 maximally-separated
//! unit directions in R⁸) every fan-out shares: tangent-projected per
//! query for the cap ([`fan_cap_ladder_into`] — the plan's `U tᵢ`; the
//! basis U is never materialized), used directly as the PCA fan's fixed
//! low-discrepancy `{zᵢ}` ([`fan_pca_ladder_into`]). The θ-ladder walks
//! cone angles until every fanned candidate GROUNDS against the corpus —
//! `minᵢ cos(cᵢ, snap(cᵢ)) ≥ τ` — and a collapsed certificate fires the
//! bounded re-fan-out recovery loop ([`admit_with_recovery_into`], the
//! CGSP fire-and-inject shape). [`exp_tilt_resample_into`] is the
//! deterministic exp-tilt arm over the [`systematic_resample_into`
//! substrate. Every post-snap candidate is a corpus point by identity —
//! virtual queries are never admitted, only measured against.
//!
//! # Boundaries
//!
//! - **Kernel fixed = cosine** (T0.1): eigenduality + exact Vendi require a
//!   linear/cosine kernel; an RBF kernel would need Nyström approximation
//!   and loses exactness — out of scope.
//! - **d = 8 ceiling** ([`DIM`]): Vendi ≤ min(K, d) — the depth ceiling is
//!   reported as [`CertificateReport::saturated`], never hidden.
//! - Zero-alloc on the hot path: all linear algebra runs in caller-owned
//!   fixed arrays; the two reusable lists inside [`AdmissionScratch`]
//!   reserve once (warmup) and are reused across cycles.
//! - Deterministic: pinned single-pass argmax, lowest-index tie-break;
//!   same pool + config → identical indices, bytes included.
//! - Opt-in: feature `set_admission = ["certified_frontier",
//!   "spectral_pencil"]` — consumes both substrates, forks neither.
//!
//! set_admission — counter-anchored set admission (Plan 599, Research 564,
//! arXiv:2603.06397 "R4T"): greedy set admission scoring g + α·cos(x,q₀) +
//! κ·log(1 + x̂ᵀM⁻¹x̂) under the 0.95 colinearity cap (Sherman–Morrison M⁻¹,
//! incremental participation-ratio tripwire), exact cosine-kernel Vendi
//! certificate via d×d eigenduality through the pinned spectral_pencil
//! Jacobi + certified_frontier's vendi_diversity. Opt-in; both substrates
//! consumed, neither forked.

use crate::certified_frontier::vendi_diversity;
use crate::distributional_steering::systematic_resample_into;
use crate::spectral_pencil::dense::{DenseScratch, jacobi_eigen};

/// Latent dimension of the admission space (the plan's d = 8 working
/// ceiling; Vendi and PR saturate at `min(K, DIM)`).
pub const DIM: usize = 8;

/// Saturation bar for [`CertificateReport::saturated`] (the honesty field):
/// the certificate says "at the depth ceiling" this close to `min(K, DIM)`.
pub const SATURATION_FRACTION: f32 = 0.95;

/// Configuration for the counter-anchored admission gate.
///
/// Defaults are R4T's reward proportions (0.6 / 0.2 / 0.2) as the starting
/// sweep point for α/κ — a prior, not a claim (the plan's wording); the
/// colinearity cap takes the `ColinearityBatchGate` 0.95 precedent.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SetAdmissionConfig {
    /// Query-alignment anchor weight (α). Zero re-opens semantic-drift (L2).
    pub alpha_align: f32,
    /// Diversity (log-det) anchor weight (κ). Zero re-opens
    /// paraphrase-collapse (L2).
    pub kappa_div: f32,
    /// Colinearity cap (θ_coll): candidates with `cos(x̂, admitted) > θ` are
    /// rejected outright.
    pub theta_coll: f32,
    /// Certificate floor fraction (ρ): the report flags
    /// `collapsed = vendi < ρ·min(K, DIM)`. Zero disables the flag entirely —
    /// coordinate-gaming becomes undetectable (L2). Admission is ρ-invariant.
    pub rho_vendi: f32,
}

impl Default for SetAdmissionConfig {
    fn default() -> Self {
        Self {
            alpha_align: 0.6,
            kappa_div: 0.2,
            theta_coll: 0.95,
            rho_vendi: 0.2,
        }
    }
}

/// Post-admission certificate report (the honesty instrument).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CertificateReport {
    /// Exact cosine-kernel Vendi score of the admitted set:
    /// `exp(−Σ pᵢ ln pᵢ)` over the dual Gram's positive normalized
    /// eigenvalues. Upper bound `min(K, DIM)`.
    pub vendi: f32,
    /// Incremental participation ratio `(tr G)²/tr(G²)` maintained during
    /// admission (the between-certifications fast path — no eigensolve).
    pub participation_ratio: f32,
    /// `vendi ≥ SATURATION_FRACTION·min(K, DIM)` — the depth ceiling
    /// reached; the d = 8 saturation is a first-class output, never silent.
    pub saturated: bool,
    /// `vendi < ρ·min(K, DIM)` — the set is rank-collapsed against the
    /// certificate floor. Always `false` when ρ = 0 (the L2-ρ hole).
    pub collapsed: bool,
}

/// Caller-owned scratch for [`admit_into`] / [`certify_scratch`]: fixed
/// linear algebra, reusable lists that keep their allocations across
/// cycles.
#[derive(Clone)]
pub struct AdmissionScratch {
    /// Dual Gram `G = X̂ᵀX̂` of the admitted set (row-major).
    pub gram: [[f32; DIM]; DIM],
    /// `M⁻¹ = (I + G)⁻¹`, maintained by Sherman–Morrison rank-1 updates
    /// (the admitted Gram alone can be singular; `I + G` never is).
    pub minv: [[f32; DIM]; DIM],
    /// Normalized admitted latents (the colinearity cap's members).
    pub admitted: Vec<[f32; DIM]>,
    /// Normalized pool (cleared and refilled per cycle — the allocation
    /// persists).
    pub pool_hats: Vec<Option<[f32; DIM]>>,
    /// Running trace of `G`.
    pub tr_g: f32,
    /// Running trace of `G²`.
    pub tr_g2: f32,
}

impl Default for AdmissionScratch {
    fn default() -> Self {
        Self::new()
    }
}

impl AdmissionScratch {
    /// Fresh scratch: `G = 0`, `M⁻¹ = I`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            gram: [[0.0; DIM]; DIM],
            minv: identity(),
            admitted: Vec::new(),
            pool_hats: Vec::new(),
            tr_g: 0.0,
            tr_g2: 0.0,
        }
    }

    /// Reset for a fresh cycle (keeps every allocation).
    pub fn clear(&mut self) {
        self.gram = [[0.0; DIM]; DIM];
        self.minv = identity();
        self.admitted.clear();
        self.pool_hats.clear();
        self.tr_g = 0.0;
        self.tr_g2 = 0.0;
    }
}

#[inline]
fn identity() -> [[f32; DIM]; DIM] {
    let mut m = [[0.0_f32; DIM]; DIM];
    for (i, row) in m.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    m
}

/// Normalize a latent onto the cosine kernel's unit shell.
/// `None` for zero/non-finite vectors (rejected, never NaN).
/// `pub(crate)`: the re-freeze lane (set_admission_freeze) reuses the exact
/// shell normalization — a second copy would be a second convention.
#[inline]
pub(crate) fn normalize(x: &[f32; DIM]) -> Option<[f32; DIM]> {
    if x.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let n = x.iter().map(|v| v * v).sum::<f32>().sqrt();
    if !n.is_finite() || n < 1e-12 {
        return None;
    }
    let mut out = [0.0_f32; DIM];
    for (o, v) in out.iter_mut().zip(x) {
        *o = v / n;
    }
    Some(out)
}

#[inline]
fn dot(a: &[f32; DIM], b: &[f32; DIM]) -> f32 {
    a.iter().zip(b).map(|(u, v)| u * v).sum()
}

/// One Sherman–Morrison rank-1 update of `M⁻¹` in place:
/// `M⁻¹ ← M⁻¹ − (M⁻¹x)(M⁻¹x)ᵀ / (1 + xᵀM⁻¹x)`.
fn sherman_morrison_rank1(minv: &mut [[f32; DIM]; DIM], x: &[f32; DIM]) {
    let mut mx = [0.0_f32; DIM];
    for (row, mrow) in mx.iter_mut().zip(minv.iter()) {
        *row = dot(mrow, x);
    }
    let denom = 1.0 + dot(x, &mx);
    if !denom.is_finite() || denom.abs() < 1e-12 {
        return; // numerically degenerate update — keep the prior inverse
    }
    for (mrow, &mxi) in minv.iter_mut().zip(&mx) {
        for (mij, &mxj) in mrow.iter_mut().zip(&mx) {
            *mij -= mxi * mxj / denom;
        }
    }
}

/// Admit one unit latent into the scratch: rank-1 Gram update, PR traces,
/// Sherman–Morrison, member list.
fn admit_latent(scratch: &mut AdmissionScratch, x: &[f32; DIM]) {
    // xᵀGx on the PRIOR Gram — the tr(G²) update needs the old product.
    let mut gx = [0.0_f32; DIM];
    for (gxi, row) in gx.iter_mut().zip(scratch.gram.iter()) {
        *gxi = dot(row, x);
    }
    let xgx = dot(&gx, x);

    // G ← G + xxᵀ.
    for i in 0..DIM {
        for j in 0..DIM {
            scratch.gram[i][j] += x[i] * x[j];
        }
    }
    // tr(G) += ‖x̂‖² = 1 on the unit shell;
    // tr(G²) += 2·xᵀGx + (xᵀx)² = 2·xᵀGx + 1.
    scratch.tr_g += 1.0;
    scratch.tr_g2 += 2.0 * xgx + 1.0;

    sherman_morrison_rank1(&mut scratch.minv, x);
    scratch.admitted.push(*x);
}

/// Greedy admission core — `quality = None` scores every candidate with a
/// neutral modular prior of 0 (the recovery path admits straight off the
/// snapped corpus, where no caller prior exists).
fn admit_core(
    cfg: &SetAdmissionConfig,
    pool: &[[f32; DIM]],
    quality: Option<&[f32]>,
    query: &[f32; DIM],
    out_indices: &mut [u16],
    scratch: &mut AdmissionScratch,
) -> usize {
    if let Some(q) = quality {
        assert_eq!(pool.len(), q.len(), "pool/quality length mismatch");
    }
    assert!(
        pool.len() <= u16::MAX as usize,
        "pool exceeds the u16 index encoding"
    );
    scratch.clear();
    let k_target = out_indices.len();
    for slot in out_indices.iter_mut() {
        *slot = u16::MAX;
    }
    scratch.admitted.reserve(k_target);
    scratch.pool_hats.reserve(pool.len());
    if k_target == 0 || pool.is_empty() {
        return 0;
    }
    let q_hat = normalize(query);
    for x in pool {
        scratch.pool_hats.push(normalize(x));
    }

    for _ in 0..k_target {
        let mut best: Option<(f32, usize)> = None;
        for (idx, hat) in scratch.pool_hats.iter().enumerate() {
            let Some(x) = hat else { continue };
            // Colinearity cap against every admitted member.
            let mut colinear = false;
            for a in &scratch.admitted {
                if dot(x, a) > cfg.theta_coll {
                    colinear = true;
                    break;
                }
            }
            if colinear {
                continue;
            }
            // Exact marginal log-det gain: log(1 + x̂ᵀM⁻¹x̂).
            let mut mx = [0.0_f32; DIM];
            for (row, mrow) in mx.iter_mut().zip(scratch.minv.iter()) {
                *row = dot(mrow, x);
            }
            let logdet_gain = (1.0 + dot(x, &mx)).ln();
            let align = q_hat.map_or(0.0, |q| dot(x, &q));
            let score = quality.map_or(0.0, |q| q[idx])
                + cfg.alpha_align * align
                + cfg.kappa_div * logdet_gain;
            if score.is_finite() && best.is_none_or(|(bs, _)| score > bs) {
                best = Some((score, idx));
            }
        }
        let Some((_, idx)) = best else { break };
        let x = scratch.pool_hats[idx].expect("best index came from a normalized row");
        admit_latent(scratch, &x);
        out_indices[scratch.admitted.len() - 1] = idx as u16;
    }
    scratch.admitted.len()
}

/// Greedy counter-anchored admission over one pool.
///
/// Writes admitted pool indices into `out_indices` (the caller sizes it =
/// the target set size K; slots beyond the returned count carry the
/// `u16::MAX` sentinel) and returns the admitted count. `quality[g]` is the
/// modular prior for `pool[g]`; `query` is the anchor `q₀` (a non-normalizable
/// query zeroes the alignment term for every candidate). `scratch` is
/// cleared internally — one call = one fresh cycle.
///
/// # Panics
/// Panics if `pool.len() != quality.len()` or `pool.len() > u16::MAX`
/// (the index encoding).
pub fn admit_into(
    cfg: &SetAdmissionConfig,
    pool: &[[f32; DIM]],
    quality: &[f32],
    query: &[f32; DIM],
    out_indices: &mut [u16],
    scratch: &mut AdmissionScratch,
) -> usize {
    admit_core(cfg, pool, Some(quality), query, out_indices, scratch)
}

/// The exact certificate over the scratch's current admitted set:
/// dual-Gram eigenvalues through the pinned Jacobi, Vendi over them, the
/// maintained participation ratio, and the collapse / saturation flags
/// against `min(K, DIM)` (K = the admitted count).
pub fn certify_scratch(cfg: &SetAdmissionConfig, scratch: &AdmissionScratch) -> CertificateReport {
    let k = scratch.admitted.len();
    if k == 0 {
        return CertificateReport {
            vendi: 0.0,
            participation_ratio: 0.0,
            saturated: false,
            collapsed: false,
        };
    }
    let cap = k.min(DIM) as f32;
    let participation_ratio = if scratch.tr_g2 > 0.0 {
        scratch.tr_g * scratch.tr_g / scratch.tr_g2
    } else {
        0.0
    };
    let mut dense = crate::spectral_pencil::dense::DenseScratch::<DIM>::new();
    let _ = jacobi_eigen(&scratch.gram, false, &mut dense);
    let vendi = vendi_diversity(&dense.values);
    CertificateReport {
        vendi,
        participation_ratio,
        saturated: vendi >= SATURATION_FRACTION * cap,
        collapsed: vendi < cfg.rho_vendi * cap,
    }
}

/// Certificate for a caller-held set (no greedy pass) — the detection
/// surface consumers and audits use on already-delivered sets.
pub fn certify_set(cfg: &SetAdmissionConfig, set: &[[f32; DIM]]) -> CertificateReport {
    let mut scratch = AdmissionScratch::new();
    certify_set_into(cfg, set, &mut scratch)
}

/// The scratch-driven [`certify_set`]: same certificate, caller-owned
/// scratch — zero steady-state allocation for set-certify hot paths (the
/// `admit_into`/`certify_scratch` split, applied to the owned convenience;
/// riir-ai Issue 966's zone-attention triad is the first consumer).
pub fn certify_set_into(
    cfg: &SetAdmissionConfig,
    set: &[[f32; DIM]],
    scratch: &mut AdmissionScratch,
) -> CertificateReport {
    scratch.clear();
    scratch.admitted.reserve(set.len());
    for x in set {
        if let Some(hat) = normalize(x) {
            admit_latent(scratch, &hat);
        }
    }
    certify_scratch(cfg, scratch)
}

// ── Phase 2: latent fan-out construction (query expansion, modelless) ──
//
// The fan-out paths build a candidate POOL around an anchor for the
// Phase-1 admission gate to consume: the tangent cap spreads a cone at
// angle θ around the query (T2.1), the θ-ladder climbs until every fanned
// candidate grounds against the corpus (T2.2), the local-PCA variant
// replaces the cone with the anchor's kNN second moment (T2.3), and the
// exp-tilt arm is the deterministic RL-fixed-point replacement (T2.4 —
// measured, not assumed). Every fanned candidate is snapped to a corpus
// point before it can be admitted — the grounding invariant is "post-snap
// candidates are corpus points by identity", never virtual queries.

/// Direction-bank size (B) — the frozen-table working point. At d = 8 the
/// frozen table is 32 × 8 × 4 B = 1 KB, the plan's stated size.
pub const BANK_SIZE: usize = 32;

/// Deterministic generator seed for the frozen bank. The generator is
/// integer-arithmetic only (hash → scale → normalize: mul/add/div/sqrt,
/// every op correctly rounded per IEEE-754 — no transcendentals), so the
/// frozen table is bit-exact on every platform and BLAKE3-freezable.
pub const BANK_SEED: u64 = 0x0059_9599_0000_0001;

/// Quasi-random oversample the exclusion selection runs over.
pub const BANK_OVERSAMPLE: usize = 256;

/// Sphere-exclusion threshold the bank centers are selected at: accepted
/// centers sit more than `BANK_THRESHOLD` apart in chord distance, i.e.
/// pairwise cosine < `1 − t²/2` = 0.595 at t = 0.9.
pub const BANK_THRESHOLD: f32 = 0.9;

/// The frozen world-space direction bank: B maximally-separated unit
/// directions in R⁸, integer-hash quasi-random oversample reduced by
/// sphere-exclusion at [`BANK_THRESHOLD`]. Serves both fan-out variants —
/// tangent-projected per query for the cap (the plan's `U tᵢ`; the explicit
/// basis U is never materialized), used directly as the PCA fan's fixed
/// low-discrepancy `{zᵢ}`.
pub const DIRECTION_BANK: [[f32; DIM]; BANK_SIZE] = [
    [
        -0.18498105,
        0.020929877,
        -0.20243602,
        -0.2796477,
        0.57323194,
        -0.49377272,
        0.24620427,
        -0.46166855,
    ],
    [
        -0.43481842,
        0.23811823,
        0.06538444,
        -0.38332427,
        0.28319138,
        0.13945791,
        -0.40225014,
        -0.584439,
    ],
    [
        -0.29147413,
        0.26997712,
        -0.5327417,
        0.38454625,
        0.34810287,
        -0.26517278,
        -0.29506156,
        0.3631971,
    ],
    [
        0.08543542,
        0.5662278,
        0.30353147,
        0.16277681,
        -0.23108833,
        0.6008127,
        0.33273757,
        -0.1684262,
    ],
    [
        -0.4531369,
        -0.45138165,
        -0.34307182,
        0.5851804,
        0.088933505,
        0.057739187,
        0.042536825,
        -0.3431252,
    ],
    [
        0.51941466,
        -0.056054946,
        -0.53733873,
        -0.17486742,
        0.27577418,
        -0.42821532,
        0.35619068,
        0.14650258,
    ],
    [
        -0.38406414,
        -0.51782054,
        -0.02953691,
        -0.36955562,
        0.50124735,
        -0.2999952,
        -0.27026135,
        0.1806261,
    ],
    [
        0.12760931,
        0.39356092,
        -0.34980616,
        -0.3653336,
        -0.083282605,
        0.035723038,
        0.69633454,
        0.28266388,
    ],
    [
        0.22571465,
        0.015220811,
        0.55736417,
        0.21928859,
        0.29703853,
        -0.33631754,
        -0.22455902,
        0.58164495,
    ],
    [
        0.07162182,
        -0.48736265,
        -0.23222117,
        0.120254554,
        -0.69115275,
        0.45123705,
        0.0861978,
        0.014932767,
    ],
    [
        -0.2564659,
        -0.13183844,
        0.08518178,
        0.44267172,
        -0.5157971,
        -0.061919656,
        0.5065748,
        0.4325863,
    ],
    [
        0.6032349,
        -0.64853287,
        0.00858205,
        -0.17020862,
        0.29598683,
        0.22101097,
        0.20471513,
        0.09003174,
    ],
    [
        -0.28152084,
        -0.16148901,
        -0.38746944,
        -0.2927755,
        -0.47127002,
        -0.3431035,
        -0.3483124,
        0.44461247,
    ],
    [
        0.55339557,
        -0.35349014,
        0.015106536,
        0.3601104,
        -0.34968978,
        -0.12059883,
        0.027068939,
        0.5489359,
    ],
    [
        -0.31562123,
        0.3177148,
        0.337085,
        -0.29239184,
        0.36297047,
        0.34171897,
        -0.017498653,
        0.59287065,
    ],
    [
        0.21621752,
        -0.32902792,
        0.36584345,
        0.2368232,
        0.041284192,
        0.051969435,
        0.56356114,
        -0.5771114,
    ],
    [
        -0.15887523,
        -0.51818275,
        0.4711864,
        -0.55361724,
        -0.36258784,
        -0.13555664,
        0.1355401,
        -0.09757102,
    ],
    [
        -0.18583812,
        -0.38743088,
        -0.38200188,
        0.4350387,
        0.40856767,
        0.29154825,
        0.30087516,
        0.37111142,
    ],
    [
        0.45215154,
        -0.117611706,
        -0.12618542,
        -0.29712036,
        -0.44332707,
        -0.5061992,
        -0.33184832,
        -0.33856064,
    ],
    [
        -0.33059263,
        0.13639778,
        0.503391,
        0.38182038,
        0.4209128,
        0.104773246,
        0.52159244,
        -0.11274423,
    ],
    [
        0.18685153,
        0.2150147,
        0.5695327,
        0.5097679,
        -0.49913207,
        0.06685798,
        -0.20318386,
        0.1993434,
    ],
    [
        -0.46434447,
        0.18301056,
        0.2493306,
        0.47083277,
        -0.39769417,
        0.10480154,
        -0.35200435,
        -0.41712222,
    ],
    [
        0.08300887,
        0.5019934,
        0.09118554,
        0.28732097,
        -0.52761084,
        -0.50154173,
        0.122398116,
        0.32456958,
    ],
    [
        -0.08693149,
        0.15056732,
        -0.55899316,
        -0.019649254,
        0.57182705,
        0.28783584,
        0.48206037,
        -0.121223256,
    ],
    [
        -0.0872744,
        -0.63406914,
        0.4911607,
        0.24928771,
        0.2735848,
        0.24601203,
        -0.3881843,
        0.029980022,
    ],
    [
        0.14324042,
        0.22511871,
        -0.056318633,
        -0.52987736,
        -0.42235276,
        -0.6264014,
        0.1704145,
        0.21227396,
    ],
    [
        -0.5517878,
        -0.14955638,
        -0.1350746,
        -0.51715803,
        -0.11305615,
        0.029244943,
        0.4669457,
        -0.39470273,
    ],
    [
        -0.03862766,
        0.026887204,
        -0.4021502,
        -0.49727547,
        0.028859505,
        0.526956,
        0.13425775,
        0.5405891,
    ],
    [
        0.47668752,
        0.31468758,
        -0.12690976,
        -0.56823677,
        -0.18942563,
        0.28710988,
        -0.377447,
        -0.27195808,
    ],
    [
        0.035989426,
        -0.5580819,
        -0.497602,
        -0.03754518,
        -0.25788298,
        -0.41208643,
        0.30560324,
        0.32942343,
    ],
    [
        0.35492066,
        -0.16292827,
        -0.3326687,
        0.57787573,
        -0.16839132,
        -0.6004978,
        0.116290495,
        -0.020002482,
    ],
    [
        0.22912525,
        -0.38801235,
        0.04508648,
        -0.14985463,
        0.043225665,
        0.48760945,
        -0.5833344,
        -0.4388035,
    ],
];

/// SplitMix64 — the house deterministic RNG (bench_576 convention; this
/// module carries its own copy per the standalone-binary convention).
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// The deterministic oversample pool the bank is selected from:
/// integer-hash quasi-random directions on the unit sphere
/// (uniform-in-cube, normalized — the mild cube-shell bias is irrelevant
/// after exclusion selection, and integer arithmetic keeps the pool
/// bit-exact everywhere). Public for the re-freeze lane and the
/// regen/cross-check tests.
#[must_use]
pub fn bank_sample_pool<const D: usize>() -> [[f32; D]; BANK_OVERSAMPLE] {
    let mut rng = SplitMix64(BANK_SEED);
    let mut pool = [[0.0_f32; D]; BANK_OVERSAMPLE];
    for row in &mut pool {
        loop {
            let mut acc = 0.0_f32;
            for v in row.iter_mut() {
                // u32 → [0, 1): the division is correctly rounded; the
                // rescale keeps every op in the exact-arithmetic set.
                let u = (rng.next_u64() as u32) as f32 / u32::MAX as f32;
                *v = u * 2.0 - 1.0;
                acc += *v * *v;
            }
            if acc > 1e-6 && acc.is_finite() {
                let n = acc.sqrt();
                for v in row.iter_mut() {
                    *v /= n;
                }
                break;
            }
        }
    }
    pool
}

/// Sphere-exclusion center pick returning the accepted indices (first
/// `out.len()` centers, scan order). The rule mirrors
/// [`crate::certified_frontier::sphere_exclusion_coverage`] exactly —
/// that scoreboard deliberately reports only a count; the bank generator
/// needs the centers. Count agreement is pinned by test.
pub fn exclusion_centers_into<const D: usize>(
    samples: &[[f32; D]],
    threshold: f32,
    out: &mut [usize],
) -> usize {
    let t2 = threshold * threshold;
    // In-place accepted list — the scan needs the prior centers' rows.
    let mut accepted = [0_usize; crate::certified_frontier::SPHERE_EXCLUSION_MAX_CENTERS];
    let mut centers = 0_usize;
    let mut written = 0_usize;
    for (i, s) in samples.iter().enumerate() {
        let covered = accepted[..centers].iter().any(|&c| {
            let t = &samples[c];
            let mut d = 0.0_f32;
            for (a, b) in s.iter().zip(t.iter()) {
                let e = a - b;
                d += e * e;
            }
            d <= t2
        });
        if covered {
            continue;
        }
        if centers == accepted.len() {
            break;
        }
        accepted[centers] = i;
        centers += 1;
        if written < out.len() {
            out[written] = i;
            written += 1;
        }
    }
    written
}

/// Generate the direction bank: the oversample pool reduced by
/// sphere-exclusion at the frozen threshold. Returns the first
/// [`BANK_SIZE`] centers and the total center count the scan found
/// (`≥ BANK_SIZE` is the caller's sanity bar — the frozen table's regen
/// test pins it).
#[must_use]
pub fn generate_direction_bank<const D: usize>() -> ([[f32; D]; BANK_SIZE], usize) {
    let samples = bank_sample_pool::<D>();
    let mut picked = [0_usize; crate::certified_frontier::SPHERE_EXCLUSION_MAX_CENTERS];
    let written = exclusion_centers_into(&samples, BANK_THRESHOLD, &mut picked);
    let mut bank = [[0.0_f32; D]; BANK_SIZE];
    for (row, &idx) in bank.iter_mut().zip(picked.iter()) {
        *row = samples[idx];
    }
    (bank, written)
}

/// Project one bank direction into the tangent space of unit `q_hat`
/// (subtract the q̂-component, renormalize). `None` for a bank row that is
/// (anti)parallel to the query — the slot is dropped, never NaN.
#[inline]
fn tangent_project(q_hat: &[f32; DIM], t: &[f32; DIM]) -> Option<[f32; DIM]> {
    let d = dot(q_hat, t);
    let mut out = [0.0_f32; DIM];
    let mut acc = 0.0_f32;
    for (o, (tv, qv)) in out.iter_mut().zip(t.iter().zip(q_hat.iter())) {
        *o = tv - d * qv;
        acc += *o * *o;
    }
    if !acc.is_finite() || acc < 1e-12 {
        return None;
    }
    let n = acc.sqrt();
    for v in &mut out {
        *v /= n;
    }
    Some(out)
}

/// Cap candidate at cone angle θ: `cosθ·q̂ + sinθ·t̂` — unit by
/// construction (q̂ ⊥ t̂, cos² + sin² = 1).
#[inline]
fn cap_candidate(q_hat: &[f32; DIM], t_hat: &[f32; DIM], theta: f32) -> [f32; DIM] {
    let (c, s) = (theta.cos(), theta.sin());
    let mut out = [0.0_f32; DIM];
    for (o, (qv, tv)) in out.iter_mut().zip(q_hat.iter().zip(t_hat.iter())) {
        *o = c * qv + s * tv;
    }
    out
}

/// One ladder outcome (the fan-out report).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FanOutReport {
    /// Ladder index that produced the returned fan (the last attempt when
    /// no rung grounded).
    pub rung: usize,
    /// The rung's parameter (cone angle θ for the cap, scale for PCA).
    pub theta: f32,
    /// Every fanned slot snapped with grounding cosine ≥ τ onto ≥ 2
    /// distinct corpus points.
    pub grounded: bool,
    /// Min grounding cosine at the returned rung (0.0 when nothing
    /// snapped).
    pub min_ground: f32,
    /// Distinct corpus points the returned fan snapped to — a fan that
    /// lands everything on one point is no fan (collapse honesty).
    pub distinct_snaps: usize,
}

fn failed_report(rungs: &[f32]) -> FanOutReport {
    FanOutReport {
        rung: 0,
        theta: rungs.first().copied().unwrap_or(0.0),
        grounded: false,
        min_ground: 0.0,
        distinct_snaps: 0,
    }
}

/// Default ladder for the tangent cap (cone angles, radians, ascending —
/// pass ascending for the plan's smallest-grounded-θ semantics).
pub const CAP_RUNGS: [f32; 4] = [0.35, 0.55, 0.75, 0.95];

/// Default ladder for the local-PCA fan (scales on the √Λ step).
pub const PCA_SCALES: [f32; 4] = [0.5, 1.0, 1.5, 2.0];

/// Default grounding bar τ: a snapped candidate must sit within
/// `arccos(τ)` of its virtual query to count as grounded. A prior, not a
/// claim (the α/κ wording).
pub const DEFAULT_GROUND_TAU: f32 = 0.5;

/// Caller-owned scratch for the fan-out paths — mirrors
/// [`AdmissionScratch`]'s reusable-list pattern (the corpus list reserves
/// once and is refilled per call; everything else is fixed arrays).
#[derive(Clone)]
pub struct FanScratch {
    /// Normalized corpus (cleared and refilled per call).
    pub corpus_hats: Vec<Option<[f32; DIM]>>,
    /// Per-rung candidate buffer (fixed — bounded by [`BANK_SIZE`]).
    pub cands: [[f32; DIM]; BANK_SIZE],
    /// Per-slot grounding cosines (fixed).
    pub ground: [f32; BANK_SIZE],
}

impl Default for FanScratch {
    fn default() -> Self {
        Self::new()
    }
}

impl FanScratch {
    /// Fresh scratch.
    #[must_use]
    pub fn new() -> Self {
        Self {
            corpus_hats: Vec::new(),
            cands: [[0.0; DIM]; BANK_SIZE],
            ground: [0.0; BANK_SIZE],
        }
    }

    /// Reset for a fresh cycle (keeps every allocation).
    pub fn clear(&mut self) {
        self.corpus_hats.clear();
        self.cands = [[0.0; DIM]; BANK_SIZE];
        self.ground = [0.0; BANK_SIZE];
    }
}

/// Refill the scratch's normalized corpus (reserve-once, refill per call).
fn fill_corpus_hats(corpus: &[[f32; DIM]], fan: &mut FanScratch) {
    fan.corpus_hats.clear();
    fan.corpus_hats.reserve(corpus.len());
    for x in corpus {
        fan.corpus_hats.push(normalize(x));
    }
}

/// Snap the first `n` candidates to their nearest corpus point (argmax
/// cosine over normalized rows; both unit). Per slot: corpus identity
/// (`u16::MAX` when nothing snapped), grounding cosine, and the SNAPPED
/// CORPUS UNIT VECTOR written to `out_pool` (zeros when unsnapped — the
/// zero vector is non-normalizable, so downstream admission skips it
/// cleanly). Returns (snapped count, distinct count, min grounding).
fn snap_candidates(
    cands: &[[f32; DIM]; BANK_SIZE],
    n: usize,
    corpus_hats: &[Option<[f32; DIM]>],
    out_idx: &mut [u16],
    out_ground: &mut [f32; BANK_SIZE],
    out_pool: &mut [[f32; DIM]],
) -> (usize, usize, f32) {
    let mut snapped = 0_usize;
    let mut min_ground = f32::INFINITY;
    for slot in 0..n {
        let mut best: Option<(f32, usize)> = None;
        for (j, hat) in corpus_hats.iter().enumerate() {
            let Some(c) = hat else { continue };
            let d = dot(&cands[slot], c);
            if d.is_finite() && best.is_none_or(|(bd, _)| d > bd) {
                best = Some((d, j));
            }
        }
        match best {
            Some((d, j)) => {
                out_idx[slot] = j as u16;
                out_ground[slot] = d;
                out_pool[slot] = corpus_hats[j].expect("best index came from a normalized row");
                snapped += 1;
                if d < min_ground {
                    min_ground = d;
                }
            }
            None => {
                out_idx[slot] = u16::MAX;
                out_ground[slot] = -1.0;
                out_pool[slot] = [0.0; DIM];
            }
        }
    }
    // Distinct snapped corpus points (n ≤ 32 — the pairwise scan is free).
    let mut distinct = 0_usize;
    for i in 0..n {
        if out_idx[i] != u16::MAX && !(0..i).any(|k| out_idx[k] == out_idx[i]) {
            distinct += 1;
        }
    }
    if snapped == 0 {
        min_ground = 0.0;
    }
    (snapped, distinct, min_ground)
}

/// Shared ladder walk: for each rung (in the order given — ascending for
/// the plan's smallest-grounded-θ semantics), build candidates via `build`
/// and snap; return the first grounded rung. When no rung grounds, the
/// LAST rung's fan stays in the out buffers (best effort,
/// `grounded = false`).
#[allow(clippy::too_many_arguments)] // the scratch-taking ladder seam — every param is a distinct buffer
fn fan_ladder_walk<F>(
    rungs: &[f32],
    tau: f32,
    n: usize,
    corpus_hats: &[Option<[f32; DIM]>],
    out_idx: &mut [u16],
    out_pool: &mut [[f32; DIM]],
    cands: &mut [[f32; DIM]; BANK_SIZE],
    ground: &mut [f32; BANK_SIZE],
    mut build: F,
) -> FanOutReport
where
    F: FnMut(f32, &mut [[f32; DIM]; BANK_SIZE]),
{
    let mut report = failed_report(rungs);
    for (rung, &theta) in rungs.iter().enumerate() {
        build(theta, cands);
        let (snapped, distinct, min_ground) =
            snap_candidates(cands, n, corpus_hats, out_idx, ground, out_pool);
        report = FanOutReport {
            rung,
            theta,
            grounded: snapped == n && distinct >= 2 && min_ground >= tau,
            min_ground,
            distinct_snaps: distinct,
        };
        if report.grounded {
            return report;
        }
    }
    report
}

/// T2.1+T2.2 — the tangent-cap fan with the grounding θ-ladder: at each
/// rung θ, candidates `cᵢ(θ) = cosθ·q̂ + sinθ·t̂ᵢ` over the bank's
/// tangent-projected directions (the first `n` projections that survive —
/// a bank row (anti)parallel to the query is dropped, never NaN), snapped
/// to their nearest corpus point (argmax cosine). Returns the first rung
/// where every slot snapped with grounding cosine ≥ τ onto ≥ 2 distinct
/// corpus points; on exhaustion the last rung's fan stays in the out
/// buffers with `grounded = false`.
///
/// `out_idx`/`out_pool` carry the snapped corpus identity + unit vectors —
/// the admission pool (every post-snap candidate is a corpus point by
/// identity). Unsnapped slots carry `u16::MAX` + the zero vector.
///
/// # Panics
/// Panics if `out_idx.len() != out_pool.len()`, the length exceeds
/// [`BANK_SIZE`], or the corpus exceeds `u16::MAX` (the index encoding).
pub fn fan_cap_ladder_into(
    query: &[f32; DIM],
    corpus: &[[f32; DIM]],
    rungs: &[f32],
    tau: f32,
    out_idx: &mut [u16],
    out_pool: &mut [[f32; DIM]],
    fan: &mut FanScratch,
) -> FanOutReport {
    fan_cap_ladder_into_bank(
        query,
        corpus,
        rungs,
        tau,
        out_idx,
        out_pool,
        fan,
        &DIRECTION_BANK,
    )
}

/// The bank-parameterized twin of [`fan_cap_ladder_into`] — the re-frozen
/// bank's consumption path (Plan 599 T4.4): identical walk, candidates built
/// from the caller's `bank` rows (tangent-projected per query, first `n`
/// projections that survive). Bit-identical to [`fan_cap_ladder_into`] when
/// `bank == &DIRECTION_BANK` (pinned by test in `set_admission_freeze`).
/// Rows are expected unit directions; a non-unit row degrades its own slot's
/// grounding (candidates stay finite — `tangent_project` handles the rest).
///
/// # Panics
/// Same as [`fan_cap_ladder_into`].
#[allow(clippy::too_many_arguments)] // the fan + bank seam — every param is a distinct role
pub fn fan_cap_ladder_into_bank(
    query: &[f32; DIM],
    corpus: &[[f32; DIM]],
    rungs: &[f32],
    tau: f32,
    out_idx: &mut [u16],
    out_pool: &mut [[f32; DIM]],
    fan: &mut FanScratch,
    bank: &[[f32; DIM]],
) -> FanOutReport {
    assert_eq!(out_idx.len(), out_pool.len(), "out length mismatch");
    assert!(
        out_idx.len() <= BANK_SIZE,
        "fan size exceeds the direction bank"
    );
    assert!(
        corpus.len() <= u16::MAX as usize,
        "corpus exceeds the u16 index encoding"
    );
    for slot in out_idx.iter_mut() {
        *slot = u16::MAX;
    }
    for row in out_pool.iter_mut() {
        *row = [0.0; DIM];
    }
    let n = out_idx.len();
    if n == 0 {
        return failed_report(rungs);
    }
    let Some(q_hat) = normalize(query) else {
        return failed_report(rungs);
    };
    fill_corpus_hats(corpus, fan);
    // Tangent projections once (θ-independent); compacted in bank order.
    let mut t_hats = [[0.0_f32; DIM]; BANK_SIZE];
    let mut t_count = 0_usize;
    for t in bank.iter().take(BANK_SIZE) {
        if let Some(hat) = tangent_project(&q_hat, t) {
            t_hats[t_count] = hat;
            t_count += 1;
        }
    }
    if t_count < n {
        return failed_report(rungs);
    }
    let build = |theta: f32, cands: &mut [[f32; DIM]; BANK_SIZE]| {
        for (slot, cand) in cands.iter_mut().enumerate().take(n) {
            *cand = cap_candidate(&q_hat, &t_hats[slot], theta);
        }
    };
    fan_ladder_walk(
        rungs,
        tau,
        n,
        &fan.corpus_hats,
        out_idx,
        out_pool,
        &mut fan.cands,
        &mut fan.ground,
        build,
    )
}

/// kNN width cap for the local-PCA covariance (fixed scratch).
pub const PCA_MAX_KNN: usize = 16;

/// Caller-owned scratch for [`fan_pca_ladder_into`]: the pinned Jacobi
/// kernel's scratch plus the fixed kNN selection buffers.
pub struct PcaScratch {
    /// The pinned Jacobi kernel's scratch.
    pub dense: DenseScratch<DIM>,
    /// kNN corpus indices (selection buffers, fixed).
    pub knn_idx: [u16; PCA_MAX_KNN],
    /// kNN cosines (selection buffers, fixed).
    pub knn_cos: [f32; PCA_MAX_KNN],
}

impl Default for PcaScratch {
    fn default() -> Self {
        Self::new()
    }
}

impl PcaScratch {
    /// Fresh scratch.
    #[must_use]
    pub fn new() -> Self {
        Self {
            dense: DenseScratch::new(),
            knn_idx: [u16::MAX; PCA_MAX_KNN],
            knn_cos: [0.0; PCA_MAX_KNN],
        }
    }
}

/// T2.3 — the local-PCA fan: the anchor's kNN covariance (d×d, about the
/// local mean) eigendecomposed through the pinned Jacobi, then candidates
/// `N(x̂₀ + θ·Q√Λ·zᵢ)` over the frozen bank rows as the fixed
/// low-discrepancy `{zᵢ}` — the corpus-adaptive second moment standing in
/// for the diffusion student's learned one. Snapped + grounded through the
/// same ladder as the cap.
///
/// # Panics
/// Same as [`fan_cap_ladder_into`]. `k_nn` is clamped to
/// `2..=[PCA_MAX_KNN]`.
#[allow(clippy::too_many_arguments)] // scratch-taking fan seam — anchor/corpus/rungs/outs/two scratches are distinct roles
pub fn fan_pca_ladder_into(
    anchor: &[f32; DIM],
    corpus: &[[f32; DIM]],
    k_nn: usize,
    rungs: &[f32],
    tau: f32,
    out_idx: &mut [u16],
    out_pool: &mut [[f32; DIM]],
    fan: &mut FanScratch,
    pca: &mut PcaScratch,
) -> FanOutReport {
    assert_eq!(out_idx.len(), out_pool.len(), "out length mismatch");
    assert!(
        out_idx.len() <= BANK_SIZE,
        "fan size exceeds the direction bank"
    );
    assert!(
        corpus.len() <= u16::MAX as usize,
        "corpus exceeds the u16 index encoding"
    );
    for slot in out_idx.iter_mut() {
        *slot = u16::MAX;
    }
    for row in out_pool.iter_mut() {
        *row = [0.0; DIM];
    }
    let n = out_idx.len();
    if n == 0 {
        return failed_report(rungs);
    }
    let Some(a_hat) = normalize(anchor) else {
        return failed_report(rungs);
    };
    fill_corpus_hats(corpus, fan);
    let k = k_nn.clamp(2, PCA_MAX_KNN);

    // kNN of the anchor: single pass, insertion into the fixed top-k
    // (sorted descending).
    for i in 0..PCA_MAX_KNN {
        pca.knn_idx[i] = u16::MAX;
        pca.knn_cos[i] = f32::NEG_INFINITY;
    }
    for (j, hat) in fan.corpus_hats.iter().enumerate() {
        let Some(c) = hat else { continue };
        let d = dot(&a_hat, c);
        if !d.is_finite() || d <= pca.knn_cos[k - 1] {
            continue;
        }
        let mut pos = k - 1;
        while pos > 0 && pca.knn_cos[pos - 1] < d {
            pca.knn_cos[pos] = pca.knn_cos[pos - 1];
            pca.knn_idx[pos] = pca.knn_idx[pos - 1];
            pos -= 1;
        }
        pca.knn_cos[pos] = d;
        pca.knn_idx[pos] = j as u16;
    }
    if pca.knn_idx[k - 1] == u16::MAX {
        // Fewer than k normalizable rows — no covariance, no fan.
        return failed_report(rungs);
    }

    // Covariance of the kNN about the local mean (d×d symmetric).
    let mut mean = [0.0_f32; DIM];
    for &j in &pca.knn_idx[..k] {
        let c = fan.corpus_hats[j as usize].expect("knn index came from a normalized row");
        for (m, &v) in mean.iter_mut().zip(c.iter()) {
            *m += v;
        }
    }
    for m in &mut mean {
        *m /= k as f32;
    }
    let mut cov = [[0.0_f32; DIM]; DIM];
    for &j in &pca.knn_idx[..k] {
        let c = fan.corpus_hats[j as usize].expect("knn index came from a normalized row");
        for i in 0..DIM {
            let ci = c[i] - mean[i];
            for l in 0..DIM {
                cov[i][l] += ci * (c[l] - mean[l]);
            }
        }
    }
    let kf = k as f32;
    for row in &mut cov {
        for v in row.iter_mut() {
            *v /= kf;
        }
    }

    let _ = jacobi_eigen(&cov, true, &mut pca.dense);
    // Q√Λ as a single matrix: column j scaled by √(max(λⱼ, 0)) — the
    // Jacobi's eigenvectors are COLUMNS aligned with the ascending values.
    let mut ql = [[0.0_f32; DIM]; DIM];
    for (i, row) in ql.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = pca.dense.v[i][j] * pca.dense.values[j].max(0.0).sqrt();
        }
    }

    let build = |theta: f32, cands: &mut [[f32; DIM]; BANK_SIZE]| {
        for (slot, cand) in cands.iter_mut().enumerate().take(n) {
            // c = x̂₀ + θ·(Q√Λ)·zᵢ, renormalized — a step that cancels the
            // anchor degenerates to the anchor itself (distinct-count
            // honesty stays with the ladder).
            let z = &DIRECTION_BANK[slot];
            let mut c = a_hat;
            for i in 0..DIM {
                let mut s = 0.0_f32;
                for j in 0..DIM {
                    s += ql[i][j] * z[j];
                }
                c[i] += theta * s;
            }
            *cand = normalize(&c).unwrap_or(a_hat);
        }
    };
    fan_ladder_walk(
        rungs,
        tau,
        n,
        &fan.corpus_hats,
        out_idx,
        out_pool,
        &mut fan.cands,
        &mut fan.ground,
        build,
    )
}

/// Outcome of the bounded collapse→re-fan-out recovery loop.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RecoveryOutcome {
    /// The initial admission's certificate flagged collapse.
    pub fired: bool,
    /// The final certificate is not collapsed (trivially `true` when the
    /// loop never fired).
    pub recovered: bool,
    /// Ladder rungs the loop walked (0 when it never fired).
    pub rungs_tried: usize,
    /// Admitted count of the winning set.
    pub admitted: usize,
    /// Final certificate.
    pub certificate: CertificateReport,
}

/// Admission with the T2.2 recovery loop: admit over the caller's pool;
/// if the certificate flags `collapsed`, re-fan the cap θ-ladder around
/// the query over `corpus` and RE-ADMIT over the snapped pool alone — the
/// collapsed pool is the failure mode, never merged back (the injected
/// candidates REPLACE the disease, they do not compete with it). Each
/// rung fans exactly `k` slots (the re-admission needs k candidates) and
/// re-admits with a neutral modular prior ([`DEFAULT_GROUND_TAU`] grounds
/// the fan); the loop stops at the first rung whose certificate is clean.
/// Bounded by [`CAP_RUNGS`]' length — no unbounded loop, deterministic end
/// to end.
///
/// `out_pool_idx` (k slots) is valid when the ORIGINAL pool won
/// (`!fired`) — indices into `pool`. `out_corpus_idx` (k slots) is valid
/// when recovery replaced it — indices into `corpus` (the identity
/// contract: each is the snapped corpus point the fan contributed). The
/// invalid one is sentinel-filled. If a grounded fan re-admitted but
/// stayed collapsed, the last attempt's corpus indices stay (best effort,
/// `recovered = false`); if NO rung grounded, both outputs are
/// sentinel-filled.
///
/// # Panics
/// Same as [`admit_into`] and [`fan_cap_ladder_into`].
#[allow(clippy::too_many_arguments)] // admission + fan + two output contracts — each param is a distinct role
pub fn admit_with_recovery_into(
    cfg: &SetAdmissionConfig,
    pool: &[[f32; DIM]],
    quality: &[f32],
    query: &[f32; DIM],
    corpus: &[[f32; DIM]],
    k: usize,
    out_pool_idx: &mut [u16],
    out_corpus_idx: &mut [u16],
    scratch: &mut AdmissionScratch,
    fan: &mut FanScratch,
) -> RecoveryOutcome {
    assert_eq!(out_pool_idx.len(), k, "out_pool_idx must be k slots");
    assert_eq!(out_corpus_idx.len(), k, "out_corpus_idx must be k slots");
    for slot in out_corpus_idx.iter_mut() {
        *slot = u16::MAX;
    }

    let admitted = admit_core(cfg, pool, Some(quality), query, out_pool_idx, scratch);
    let certificate = certify_scratch(cfg, scratch);
    if !certificate.collapsed {
        return RecoveryOutcome {
            fired: false,
            recovered: true,
            rungs_tried: 0,
            admitted,
            certificate,
        };
    }

    // Recovery: fan → re-admit over the snapped corpus pool alone.
    let mut fan_out = [[0.0_f32; DIM]; BANK_SIZE];
    let mut fan_idx = [u16::MAX; BANK_SIZE];
    for slot in out_pool_idx.iter_mut() {
        *slot = u16::MAX;
    }
    let mut outcome = RecoveryOutcome {
        fired: true,
        recovered: false,
        rungs_tried: 0,
        admitted,
        certificate,
    };
    let mut readmitted = false;
    for (tried, &theta) in CAP_RUNGS.iter().enumerate() {
        let report = fan_cap_ladder_into(
            query,
            corpus,
            &[theta],
            DEFAULT_GROUND_TAU,
            &mut fan_idx[..k],
            &mut fan_out[..k],
            fan,
        );
        outcome.rungs_tried = tried + 1;
        if !report.grounded {
            continue;
        }
        let count = admit_core(cfg, &fan_out, None, query, out_corpus_idx, scratch);
        // Translate fan-pool positions to corpus indices — the out
        // contract is corpus identity, and every admitted row snapped
        // (zero rows cannot score), so the mapping is total over the
        // admitted prefix.
        for slot in out_corpus_idx[..count].iter_mut() {
            let p = *slot as usize;
            *slot = fan_idx[p];
        }
        let cert = certify_scratch(cfg, scratch);
        outcome.admitted = count;
        outcome.certificate = cert;
        readmitted = true;
        if !cert.collapsed {
            outcome.recovered = true;
            return outcome;
        }
    }
    if !readmitted {
        for slot in out_corpus_idx.iter_mut() {
            *slot = u16::MAX;
        }
    }
    outcome
}

/// T2.4 — the deterministic exp-tilt reweight + systematic resample (the
/// RL-fixed-point replacement, measured in the tests rather than assumed):
/// `wᵢ ∝ exp(λ·(Ψᵢ − max Ψ))` normalized into `out_weights`, then
/// [`systematic_resample_into`] at the caller's fixed `u`. The Ψ policy is
/// caller-owned — this is machinery, not a policy. λ = 0 is the uniform
/// arm; non-finite Ψ entries tilt to zero weight naturally (an
/// all-degenerate input falls back to uniform).
///
/// # Panics
/// Panics on length mismatches (`psi` / `out_weights` / `out_ancestors`).
pub fn exp_tilt_resample_into(
    psi: &[f32],
    lambda: f32,
    u: f32,
    out_weights: &mut [f32],
    out_ancestors: &mut [u32],
) {
    let n = psi.len();
    assert_eq!(out_weights.len(), n, "weights length mismatch");
    assert_eq!(out_ancestors.len(), n, "ancestors length mismatch");
    if n == 0 {
        return;
    }
    let mut max = f32::NEG_INFINITY;
    for &p in psi {
        if p.is_finite() && p > max {
            max = p;
        }
    }
    let mut total = 0.0_f32;
    for (w, &p) in out_weights.iter_mut().zip(psi) {
        let v = if p.is_finite() {
            (lambda * (p - max)).exp()
        } else {
            0.0
        };
        *w = v;
        total += v;
    }
    if !total.is_finite() || total <= 0.0 {
        let uniform = 1.0 / n as f32;
        out_weights.fill(uniform);
    } else {
        for w in out_weights.iter_mut() {
            *w /= total;
        }
    }
    systematic_resample_into(out_weights, n, u, out_ancestors);
}

#[cfg(test)]
mod tests {
    use super::*;
    use fastrand::Rng;

    fn unit(x: [f32; DIM]) -> [f32; DIM] {
        normalize(&x).expect("test vector is normalizable")
    }

    fn cfg_with(alpha: f32, kappa: f32, rho: f32) -> SetAdmissionConfig {
        SetAdmissionConfig {
            alpha_align: alpha,
            kappa_div: kappa,
            theta_coll: 0.95,
            rho_vendi: rho,
        }
    }

    /// certify_set_into must be EXACTLY certify_set over a caller-owned
    /// scratch — one body, two entries (the fan_cap_ladder_into_bank
    /// precedent). Bit-identity on the report, both flags, and the PR.
    #[test]
    fn certify_set_into_bit_identical_to_certify_set() {
        let worlds: [Vec<[f32; DIM]>; 4] = [
            wobble_set(0.25, 6),
            wobble_set(0.37, 4),
            (0..DIM)
                .map(|i| {
                    let mut x = [0.0_f32; DIM];
                    x[i] = 1.0;
                    x
                })
                .collect(),
            vec![[1.0_f32; DIM]; 5],
        ];
        for cfg in [
            SetAdmissionConfig::default(),
            cfg_with(0.0, 0.2, 0.2),
            cfg_with(0.6, 0.0, 0.0),
        ] {
            for set in &worlds {
                let owned = certify_set(&cfg, set);
                let mut scratch = AdmissionScratch::new();
                let into = certify_set_into(&cfg, set, &mut scratch);
                assert_eq!(owned, into, "certify_set_into diverged from certify_set");
                // And a REUSED scratch reproduces the same report (no
                // cross-call state leakage).
                let again = certify_set_into(&cfg, set, &mut scratch);
                assert_eq!(into, again, "scratch reuse leaked state across cycles");
            }
        }
    }

    /// The paraphrase/gaming family: `normalize(e0 + wobble·s_j·e_j)` —
    /// pairwise cosines strictly under the 0.95 cap (0.9412 at wobble 0.25,
    /// 0.8796 at 0.37) but spectrally one dominant direction. Yields exactly
    /// `k` distinct members, alternating coordinate and sign
    /// (k ≤ DIM is the caller's contract — indices stay in-bounds).
    fn wobble_set(wobble: f32, k: usize) -> Vec<[f32; DIM]> {
        assert!(k <= DIM, "the family has one member per (coord, sign) slot");
        let mut set = Vec::with_capacity(k);
        let mut j = 1_usize;
        while set.len() < k {
            let sign = if set.len() % 2 == 0 { 1.0 } else { -1.0 };
            let mut x = [0.0_f32; DIM];
            x[0] = 1.0;
            x[j] = wobble * sign;
            set.push(unit(x));
            j += 1;
            if j == DIM {
                j = 1;
            }
        }
        set.shrink_to_fit();
        set
    }

    // ── T1.5 known answers ──────────────────────────────────────

    #[test]
    fn identical_set_has_vendi_and_pr_one() {
        // K = 6 identical members: spectrum = one nonzero eigenvalue →
        // vendi = 1 exactly; PR = K²/K² = 1. Floor 0.2·6 = 1.2 > 1 → the
        // certificate flags the collapse.
        let set: Vec<[f32; DIM]> = vec![unit([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]); 6];
        let r = certify_set(&SetAdmissionConfig::default(), &set);
        assert!((r.vendi - 1.0).abs() < 1e-4, "vendi {}", r.vendi);
        assert!(
            (r.participation_ratio - 1.0).abs() < 1e-4,
            "pr {}",
            r.participation_ratio
        );
        assert!(r.collapsed, "identical set must flag collapsed");
        assert!(!r.saturated);
    }

    #[test]
    fn orthonormal_set_saturates_at_min_k_d() {
        // K = 5 orthonormal members in d = 8: vendi = PR = min(K, d) = 5.
        let mut set = Vec::with_capacity(5);
        for j in 0..5 {
            let mut x = [0.0_f32; DIM];
            x[j] = 1.0;
            set.push(x);
        }
        let r = certify_set(&SetAdmissionConfig::default(), &set);
        assert!((r.vendi - 5.0).abs() < 0.01, "vendi {}", r.vendi);
        assert!(
            (r.participation_ratio - 5.0).abs() < 0.01,
            "pr {}",
            r.participation_ratio
        );
        assert!(
            r.saturated,
            "5 orthogonal members at the d=8... min(5,8)=5: vendi ≥ 0.95·5"
        );
        assert!(!r.collapsed);

        // K = 8 = DIM: full depth ceiling reached.
        let mut set = Vec::new();
        for j in 0..DIM {
            let mut x = [0.0_f32; DIM];
            x[j] = 1.0;
            set.push(x);
        }
        let r = certify_set(&SetAdmissionConfig::default(), &set);
        assert!((r.vendi - 8.0).abs() < 0.01, "vendi {}", r.vendi);
        assert!(r.saturated, "K = DIM orthonormal must saturate");
    }

    #[test]
    fn pr_and_vendi_rank_correlate_over_random_sets() {
        // T1.5's empirical pin: two different functionals that agree on the
        // ORDERING of diversity (Spearman ≥ 0.95 over 10⁴ seeded sets).
        let mut rng = Rng::with_seed(0x5E7A_D001);
        let trials = 10_000_usize;
        let mut prs = Vec::with_capacity(trials);
        let mut vendis = Vec::with_capacity(trials);
        for _ in 0..trials {
            let k = 2 + rng.usize(..7); // 2..=8 members
            let mut set = Vec::with_capacity(k);
            for _ in 0..k {
                let mut x = [0.0_f32; DIM];
                for v in x.iter_mut() {
                    *v = rng.f32() * 2.0 - 1.0;
                }
                set.push(x); // certify_set normalizes internally
            }
            let r = certify_set(&SetAdmissionConfig::default(), &set);
            prs.push(r.participation_ratio);
            vendis.push(r.vendi);
        }
        let rho = spearman(&prs, &vendis);
        println!("[t15] spearman(pr, vendi) over {trials} random sets: {rho:.4}");
        assert!(rho >= 0.95, "rank correlation {rho} < 0.95");
    }

    /// Spearman rank correlation (distinct ranks — ties are measure-zero
    /// on the continuous random sets above).
    fn spearman(a: &[f32], b: &[f32]) -> f64 {
        let ra = ranks(a);
        let rb = ranks(b);
        let n = a.len() as f64;
        let (ma, mb) = (ra.iter().sum::<f64>() / n, rb.iter().sum::<f64>() / n);
        let (mut cov, mut va, mut vb) = (0.0, 0.0, 0.0);
        for i in 0..a.len() {
            let da = ra[i] - ma;
            let db = rb[i] - mb;
            cov += da * db;
            va += da * da;
            vb += db * db;
        }
        cov / (va.sqrt() * vb.sqrt())
    }

    fn ranks(v: &[f32]) -> Vec<f64> {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&i, &j| v[i].partial_cmp(&v[j]).expect("no NaN in test data"));
        let mut out = vec![0.0_f64; v.len()];
        for (rank, &i) in idx.iter().enumerate() {
            out[i] = rank as f64;
        }
        out
    }

    // ── L1: the modular-only disease ────────────────────────────────

    /// 14 duplicate-tolerant members (pairwise cos 0.8796 < 0.95) — the
    /// L1/L2-κ paraphrase family: `normalize(e0 ± 0.37·e_j)`, j = 1..8.
    fn dup_pool() -> Vec<[f32; DIM]> {
        let mut pool = Vec::new();
        for j in 1..DIM {
            for sign in [1.0_f32, -1.0] {
                let mut x = [0.0_f32; DIM];
                x[0] = 1.0;
                x[j] = 0.37 * sign;
                pool.push(unit(x));
            }
        }
        pool.shrink_to_fit();
        pool
    }

    /// 10 diverse fillers over the e1..e7 complement: `normalize(e_j + 0.5·e_k)`
    /// on distinct pairs — pairwise cosines ≤ 0.45, nothing colinear.
    fn filler_pool() -> Vec<[f32; DIM]> {
        let pairs = [
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (6, 7),
            (1, 3),
            (2, 4),
            (3, 5),
            (4, 6),
        ];
        pairs
            .iter()
            .map(|&(a, b)| {
                let mut x = [0.0_f32; DIM];
                x[a] = 1.0;
                x[b] = 0.5;
                unit(x)
            })
            .collect()
    }

    fn n_slots(out: &[u16]) -> usize {
        out.iter().take_while(|&&i| i != u16::MAX).count()
    }

    /// Mean pairwise cosine of the admitted members (the duplication
    /// metric — the paraphrase family's own instrument).
    fn mean_pairwise_cos(pool: &[[f32; DIM]], out: &[u16]) -> f32 {
        let slots = n_slots(out);
        let members: Vec<[f32; DIM]> = out[..slots]
            .iter()
            .map(|&i| normalize(&pool[i as usize]).expect("pool row"))
            .collect();
        let mut total = 0.0_f32;
        let mut pairs = 0_usize;
        for (i, a) in members.iter().enumerate() {
            for b in &members[i + 1..] {
                total += dot(a, b);
                pairs += 1;
            }
        }
        if pairs == 0 {
            0.0
        } else {
            total / pairs as f32
        }
    }

    #[test]
    fn l1_modular_only_admits_effective_rank_one_set() {
        let mut pool = dup_pool();
        pool.extend(filler_pool());
        let quality: Vec<f32> = pool
            .iter()
            .enumerate()
            .map(|(i, _)| if i < 14 { 1.0 } else { 0.97 })
            .collect();
        let query = unit([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        // Modular-only (α = κ = 0): the 0.03 g-gap admits duplicates in
        // every slot — an effective-rank-1 set.
        let mut scratch = AdmissionScratch::new();
        let mut out = [0_u16; 8];
        let n = admit_into(
            &cfg_with(0.0, 0.0, 0.2),
            &pool,
            &quality,
            &query,
            &mut out,
            &mut scratch,
        );
        assert_eq!(n, 8, "the cap never fires on pairwise-0.88 dups");
        let r = certify_scratch(&cfg_with(0.0, 0.0, 0.2), &scratch);
        let mp = mean_pairwise_cos(&pool, &out);
        println!(
            "[L1] modular-only: vendi={:.3} pr={:.3} mean-pairwise-cos={:.3}",
            r.vendi, r.participation_ratio, mp
        );
        // The L1 claim is EFFECTIVE RANK: the participation ratio (the
        // set's own rank functional) must read ≈ 1, and the admitted
        // members must be near-duplicates. The vendi floor flag belongs to
        // the ρ family (L2-ρ) and is deliberately not asserted here.
        assert!(
            r.participation_ratio < 1.5,
            "pr {} — the set must be effectively rank-1",
            r.participation_ratio
        );
        assert!(
            mp >= 0.85,
            "mean pairwise cos {mp:.3} — admitted members must be near-duplicates"
        );
    }

    // ── L2-κ: paraphrase-collapse is a selection failure ─────────────

    #[test]
    fn l2_paraphrase_collapse_reachable_only_when_kappa_zero() {
        // The world: the paraphrase family wobbles around e0 (g = 1.0);
        // the filler family spreads over the three OTHER even axes (e2,
        // e4, e6 — mutually orthogonal bases, g = 0.99). The query sits at
        // normalize(e0+e2+e4+e6) so EVERY member reads the exact same
        // cos-to-query (0.5/1.0668 = 0.4688) — the α anchor cancels out of
        // the comparison and κ alone decides. Fillers stay genuinely fresh
        // (orthogonal bases ⇒ near-zero mutual overlap), so κ's log-det
        // correction keeps winning for as long as fresh fillers exist.
        let q0 = unit([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
        let cap_iter = [1_usize, 3, 5, 7];
        let mut pool = Vec::with_capacity(cap_iter.len());
        // Paraphrase dups: base e0, wobble on the odd axes, g = 1.0.
        for j in cap_iter {
            for sign in [1.0_f32, -1.0] {
                let mut x = [0.0_f32; DIM];
                x[0] = 1.0;
                x[j] = 0.37 * sign;
                pool.push(unit(x));
            }
        }
        let dup_len = pool.len(); // 8
        // Diverse fillers: orthogonal even bases, same wobble shape, g = 0.99.
        for base in [2_usize, 4, 6] {
            for j in [1_usize, 3, 5, 7] {
                for sign in [1.0_f32, -1.0] {
                    let mut x = [0.0_f32; DIM];
                    x[base] = 1.0;
                    x[j] = 0.37 * sign;
                    pool.push(unit(x));
                }
            }
        }
        let quality: Vec<f32> = pool
            .iter()
            .enumerate()
            .map(|(i, _)| if i < dup_len { 1.0 } else { 0.99 })
            .collect();
        let k = 6_usize;
        let dup_count = |out: &[u16]| -> usize {
            out[..n_slots(out)]
                .iter()
                .filter(|&&i| (i as usize) < dup_len)
                .count()
        };

        // κ = 0: collapse reachable — the 0.01 g-gap admits the whole
        // duplicate family.
        let mut scratch = AdmissionScratch::new();
        let mut out = [0_u16; 6];
        admit_into(
            &cfg_with(0.6, 0.0, 0.2),
            &pool,
            &quality,
            &q0,
            &mut out,
            &mut scratch,
        );
        let d0 = dup_count(&out);
        let mp0 = mean_pairwise_cos(&pool, &out);
        println!("[L2k] kappa=0: dups={d0}/{k} mean-pairwise-cos={mp0:.3}");
        assert!(
            d0 >= k - 1,
            "kappa=0 must fill the set with near-duplicates"
        );
        assert!(
            mp0 >= 0.8,
            "kappa=0 admitted set must be duplicated (mean pairwise cos {mp0:.3})"
        );

        // Defaults: κ = 0.2's fresh-direction log-det gain (≈ 0.2·0.22 ≈
        // 0.045) dominates the 0.01 g-gap, and the orthogonal filler bases
        // keep the gain fresh — fillers win nearly every slot; the full
        // triple excludes the family.
        let mut scratch = AdmissionScratch::new();
        let mut out = [0_u16; 6];
        admit_into(
            &SetAdmissionConfig::default(),
            &pool,
            &quality,
            &q0,
            &mut out,
            &mut scratch,
        );
        let r = certify_scratch(&SetAdmissionConfig::default(), &scratch);
        let d = dup_count(&out);
        println!("[L2k] defaults: dups={d}/{k} vendi={:.3}", r.vendi);
        assert!(
            d <= 2,
            "defaults must not paraphrase-collapse (dup_count {d})"
        );
        assert!(
            r.vendi >= 2.5,
            "default set must be spectrally diverse (vendi {:.3})",
            r.vendi
        );
    }

    // ── L2-α: semantic-drift is a selection failure ──────────────────

    /// Far cluster (cos-to-query ≈ 0.18, high g) vs near cluster
    /// (cos-to-query ≈ 0.94, lower g), both internally cap-safe.
    fn drift_world() -> (Vec<[f32; DIM]>, Vec<f32>, [f32; DIM]) {
        let mut pool = Vec::new();
        let q0 = [1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        // Near cluster: normalize(e0 ± 0.375·e_j), j = 1..8 — cos to q0 ≈ 0.936.
        for j in 1..DIM {
            for sign in [1.0_f32, -1.0] {
                let mut x = [0.0_f32; DIM];
                x[0] = 1.0;
                x[j] = 0.375 * sign;
                pool.push(unit(x));
            }
        }
        let near_len = pool.len(); // 14
        // Far cluster: normalize(0.2·e0 + 1.0·e3 ± 0.5·e_j), j ∈ {4,5,6} —
        // cos to q0 ≈ 0.18.
        for j in [4_usize, 5, 6] {
            for sign in [1.0_f32, -1.0] {
                let mut x = [0.0_f32; DIM];
                x[0] = 0.2;
                x[3] = 1.0;
                x[j] = 0.5 * sign;
                pool.push(unit(x));
            }
        }
        // Near g = 0.9 (indices < 14), far g = 1.0.
        let quality: Vec<f32> = pool
            .iter()
            .enumerate()
            .map(|(i, _)| if i < near_len { 0.9 } else { 1.0 })
            .collect();
        (pool, quality, q0)
    }

    #[test]
    fn l2_semantic_drift_reachable_only_when_alpha_zero() {
        let (pool, quality, query) = drift_world();
        let near_len = 14_usize;
        let k = 6_usize;
        let q = normalize(&query).expect("q0");
        let mean_cos = |out: &[u16]| -> f32 {
            let slots = n_slots(out);
            out[..slots]
                .iter()
                .map(|&i| dot(&normalize(&pool[i as usize]).expect("pool"), &q))
                .sum::<f32>()
                / slots as f32
        };

        // α = 0: the far cluster wins on g — the set drifts off-query.
        let mut scratch = AdmissionScratch::new();
        let mut out = [0_u16; 6];
        admit_into(
            &cfg_with(0.0, 0.2, 0.2),
            &pool,
            &quality,
            &query,
            &mut out,
            &mut scratch,
        );
        let mc0 = mean_cos(&out);
        let far0 = out[..n_slots(&out)]
            .iter()
            .filter(|&&i| (i as usize) >= near_len)
            .count();
        println!("[L2a] alpha=0: mean-cos={mc0:.3} far-members={far0}/{k}");
        assert!(
            far0 >= k - 1 && mc0 < 0.5,
            "alpha=0 set must drift off-query"
        );

        // Defaults: the alignment anchor holds the set on-query.
        let mut scratch = AdmissionScratch::new();
        let mut out = [0_u16; 6];
        admit_into(
            &SetAdmissionConfig::default(),
            &pool,
            &quality,
            &query,
            &mut out,
            &mut scratch,
        );
        let mc = mean_cos(&out);
        println!("[L2a] defaults: mean-cos={mc:.3}");
        assert!(
            mc >= 0.8,
            "defaults must hold the set on-query (mean cos {mc:.3})"
        );
    }

    // ── L2-ρ: coordinate-gaming is a detection failure ───────────────

    #[test]
    fn l2_coordinate_gaming_reachable_only_when_rho_zero() {
        // The gaming set: 8 members at pairwise cos 0.9412 — it passes
        // EVERY pairwise colinearity screen (< 0.95) but its spectrum is
        // one dominant direction (vendi ≈ 1.36 < the default floor 1.6).
        let gaming = wobble_set(0.25, DIM); // K = 8 members
        let cfg = SetAdmissionConfig::default();

        // Detection distinctness: the set really does pass the cap.
        let mut max_pair = 0.0_f32;
        for (i, a) in gaming.iter().enumerate() {
            for b in &gaming[i + 1..] {
                max_pair = max_pair.max(dot(a, b));
            }
        }
        println!("[L2r] gaming set: max pairwise cos={max_pair:.4} (< 0.95 cap)");
        assert!(
            max_pair < cfg.theta_coll,
            "the world must pass pairwise screens"
        );

        let flagged = certify_set(&cfg, &gaming);
        println!(
            "[L2r] defaults: vendi={:.3} collapsed={}",
            flagged.vendi, flagged.collapsed
        );
        assert!(
            flagged.collapsed,
            "defaults must flag the spectrally-degenerate set"
        );

        // ρ = 0: the hole — same set, same spectrum, never flagged.
        let blind = certify_set(&cfg_with(0.6, 0.2, 0.0), &gaming);
        assert!(
            !blind.collapsed,
            "rho=0 must leave the gaming set unflagged"
        );
        assert!(
            (blind.vendi - flagged.vendi).abs() < 1e-6,
            "the certificate is rho-invariant"
        );
    }

    // ── Cap, determinism, robustness ────────────────────────────────

    #[test]
    fn colinearity_cap_rejects_exact_duplicates() {
        let mut pool = vec![unit([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]); 6];
        pool.push(unit([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
        let quality = vec![1.0_f32; pool.len()];
        let query = unit([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let mut scratch = AdmissionScratch::new();
        let mut out = [0_u16; 4];
        let n = admit_into(
            &SetAdmissionConfig::default(),
            &pool,
            &quality,
            &query,
            &mut out,
            &mut scratch,
        );
        assert_eq!(
            n, 2,
            "cap admits one duplicate + the orthogonal axis, got {n}"
        );
    }

    #[test]
    fn admission_is_deterministic_and_rho_invariant() {
        let mut rng = Rng::with_seed(0xA0_AD_00_01_u64);
        let mut pool = Vec::with_capacity(64);
        for _ in 0..64 {
            let mut x = [0.0_f32; DIM];
            for v in x.iter_mut() {
                *v = rng.f32() * 2.0 - 1.0;
            }
            pool.push(x);
        }
        let quality: Vec<f32> = (0..pool.len())
            .map(|i| 1.0 - 0.3 * (i % 5) as f32 / 4.0)
            .collect();
        let query = unit([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        let run = |rho: f32| {
            let mut scratch = AdmissionScratch::new();
            let mut out = [0_u16; 8];
            admit_into(
                &cfg_with(0.6, 0.2, rho),
                &pool,
                &quality,
                &query,
                &mut out,
                &mut scratch,
            );
            out
        };
        let a = run(0.2);
        let b = run(0.2);
        let c = run(0.0); // rho only weights the report — the greedy is invariant
        assert_eq!(a, b, "same pool + config must admit identically");
        assert_eq!(a, c, "rho must not change the admission (rho-invariance)");
    }

    #[test]
    fn non_normalizable_rows_are_skipped() {
        let pool = vec![
            [0.0_f32; DIM],                                 // zero vector
            unit([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), // good
        ];
        let quality = vec![1.0_f32, 0.5];
        let query = unit([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let mut scratch = AdmissionScratch::new();
        let mut out = [0_u16; 2];
        let n = admit_into(
            &SetAdmissionConfig::default(),
            &pool,
            &quality,
            &query,
            &mut out,
            &mut scratch,
        );
        assert_eq!(n, 1, "only the normalizable row admits");
        assert_eq!(out[0], 1, "the admitted index is the good row");

        // A non-finite query zeroes alignment without panicking.
        let bad_query = [f32::NAN; DIM];
        let good = [pool[1]; 1];
        let q2 = vec![1.0_f32];
        let mut out2 = [0_u16; 1];
        let n2 = admit_into(
            &SetAdmissionConfig::default(),
            &good,
            &q2,
            &bad_query,
            &mut out2,
            &mut scratch,
        );
        assert_eq!(n2, 1, "a NaN query must not panic the admission");
    }
}
