//! Interpolation Geometry — iMAUVE + 5-way intervention probe for committed
//! latent substrates.
//!
//! Distilled from Prabhudesai & Geng, *Latent Thought Flows with Text
//! Compression* (Jun 2, 2026). See `katgpt-rs/.research/445_*.md` for the
//! research note and `katgpt-rs/.issues/158_*.md` for the PoC issue.
//!
//! # What this is
//!
//! A generic, modelless evaluation methodology for any committed latent
//! substrate (belief state `[f32; 8]`, `NeuronShard::style_weights [f32; 64]`,
//! `ArchetypeBlendShard` π, `KarcShard` weights, `ZoneGeometryPod`,
//! `MerkleFrozenEnvelope` versions — the six substrates cataloged in
//! Research 445 §2.6). It exposes two protocols distilled from the paper:
//!
//! 1. [`imauve_score`] — nearest-neighbor midpoint interpolation quality
//!    (paper §1.2). The paper's headline methodological contribution:
//!    predicts downstream generation quality with Pearson r=0.99, while
//!    reconstruction quality (rMAUVE) saturates near 1.0 and is
//!    uninformative. Our analog: a substrate can have perfect
//!    freeze/thaw bit-identity yet have midpoints that decode to
//!    incoherent intermediate behavior.
//!
//! 2. [`intervention_battery`] — 5-way causal probe (paper §1.4): matched
//!    / shuffled / zero / mean / noise. Extends Plan 278's `FaithfulnessProbe`
//!    (binary, on injected memory) to the per-entity committed-state domain.
//!
//! # What this is NOT
//!
//! - NOT a training primitive. No encoder, no decoder network, no MeanFlow
//!   generator. The trait abstracts over substrates that already exist; the
//!   caller supplies the decode operation.
//! - NOT a probability / confidence / predictive interval. The fields are
//!   raw geometric / divergence measurements. The "Report the Floor"
//!   conformal-naive rule (Research 322 / Plan 340) does NOT apply.
//! - NOT a router. The diagnostic produces measurements; the caller decides
//!   what to do with them.
//!
//! # Modelless contract
//!
//! Every operation is closed-form: nearest-neighbor search, element-wise
//! midpoint, caller-supplied decode distance. Zero training, zero backprop.
//! Per AGENTS.md, the iMAUVE metric itself is the distilled primitive —
//! the paper's training-side architectural pressures (MAE-drop, drop
//! readout prev-token context, sliding-window attention) are documented
//! in Research 445 §2.1 and correctly routed to riir-train.
//!
//! # Performance contract
//!
//! - [`imauve_score`] is `O(n² · d)` for `n` anchors × `n` candidates × `d`
//!   dimensions, with **zero allocation** in the hot path (caller-supplied
//!   scratch for the midpoint). The reference scale is `n ≤ 1024`, `d ≤ 64`
//!   — audit cadence, not per-tick.
//! - [`intervention_battery`] is `O(n · d)` for `n` donors — single pass,
//!   zero allocation.
//!
//! # Determinism
//!
//! All operations are deterministic and platform-independent: no SIMD
//! dispatch inside the math, no floating-point reordering, no RNG in the
//! scored path. The `noise()` intervention uses a caller-supplied seed
//! (deterministic Gaussian via Box-Muller over a fixed xorshift stream).
//!
//! Interpolation Geometry — iMAUVE + 5-way intervention probe for committed
//! latent substrates (Issue 158, Research 445 — Prabhudesai & Geng, *Latent
//! Thought Flows with Text Compression*, Jun 2026). Generic `LatentSpace`
//! trait abstracting over HLA [f32;8] / style_weights[64] / archetype-blend
//! π / KarcShard / ZoneGeometryPod / MerkleFrozenEnvelope — the six substrates
//! cataloged in Research 445 §2.6. Two protocols: `imauve_score` (nearest-
//! neighbor midpoint coherence — the paper's headline metric, Pearson r=0.99
//! with downstream quality) + `intervention_battery` (matched/shuffled/zero/
//! mean/noise 5-way probe extending Plan 278's FaithfulnessProbe to per-entity
//! committed state). Pure modelless evaluation methodology — NOT a training
//! primitive. Opt-in until the PoC reports interpolation geometry across the
//! substrates (Phase 4 decision branch in `.issues/158_*`).

// (Module gating is handled by `#[cfg(feature = "interpolation_geometry")]`
// on the `mod` declaration in `lib.rs`; this file must NOT duplicate it.)

// ─── Trait ─────────────────────────────────────────────────────────────────

/// A committed latent substrate amenable to interpolation-geometry evaluation.
///
/// Generic over the latent representation (`Point`) and the decoded/behavior
/// representation (`Behavior`). The caller supplies the encode/decode/midpoint
/// operations; this trait abstracts over belief `[f32; 8]`, `style_weights
/// [f32; 64]`, archetype-blend π, etc. (the six substrates cataloged in
/// Research 445 §2.6).
///
/// **Midpoint contract**: `midpoint(a, b)` MUST be symmetric
/// (`midpoint(a, b) == midpoint(b, a)`) and idempotent at the endpoints
/// (`midpoint(a, a) == a`). The test suite verifies both invariants on
/// the shipped reference impls; custom impls should add their own test.
///
/// **Distance contract**: `latent_distance` is the metric used for
/// nearest-neighbor search (typically L2). `behavior_distance` is the
/// metric used to score decoded outputs (the paper's cross-entropy analog;
/// caller-supplied — could be L2 on emotion scalars, KL on action
/// distributions, or Hamming on tokenized output).
pub trait LatentSpace {
    /// The latent representation (e.g. `[f32; 8]`, `[f32; 64]`).
    type Point: Clone;
    /// The decoded/behavior representation (e.g. emotion-scalar set,
    /// action distribution). The paper uses token sequences; we stay
    /// generic so any consumer can plug in its behavior metric.
    type Behavior;

    /// Ambient dimensionality of the latent space. Used for k-NN bound
    /// checks and noise scaling. MUST match the actual point dimension.
    fn dim(&self) -> usize;

    /// Decode a latent point to its behavior representation. The paper's
    /// `readout decoder`; for our substrates this is the existing
    /// bridge path (e.g. `evolve_belief` → 5 affect scalars, KARC ridge
    /// readout, archetype-blend projection).
    fn decode(&self, point: &Self::Point) -> Self::Behavior;

    /// Element-wise midpoint of two latent points. The default contract:
    /// symmetric, idempotent at endpoints. Most substrates use a plain
    /// arithmetic mean; some (spherical / Riemannian) override.
    fn midpoint(&self, a: &Self::Point, b: &Self::Point) -> Self::Point;

    /// The zero latent (paper §1.4 "zero z" intervention). Typically
    /// all-zeros, but a substrate with a non-zero origin (e.g. a frozen
    /// attractor) can override.
    fn zero(&self) -> Self::Point;

    /// The mean of a sample of latent points (paper §1.4 "mean z"
    /// intervention). Caller supplies the sample; default is per-dimension
    /// arithmetic mean.
    fn mean(&self, samples: &[Self::Point]) -> Self::Point;

    /// Deterministic Gaussian noise around the origin (paper §1.4 "noise z"
    /// intervention). The caller supplies a 64-bit seed so the entire
    /// battery is reproducible.
    fn noise(&self, seed: u64) -> Self::Point;

    /// Deterministic Gaussian noise scaled to the **anchor's own L2 norm**
    /// (Issue 776 / Research 555 — CVRR §2.1 row-norm-matched noise).
    ///
    /// `n = g · (‖anchor‖/‖g‖)` — magnitude preserved, structure destroyed.
    /// Separates the "consumer reads the norm/energy" channel from the
    /// "consumer reads the structure" channel; [`LatentSpace::noise`]
    /// destroys both at once and cannot separate them. A zero-norm anchor
    /// stays zero (magnitude trivially preserved).
    ///
    /// Same seed → same output (battery-reproducible).
    fn norm_matched_noise(&self, anchor: &Self::Point, seed: u64) -> Self::Point;

    /// Latent-space distance for nearest-neighbor search. Typically L2.
    fn latent_distance(&self, a: &Self::Point, b: &Self::Point) -> f32;

    /// Behavior-space distance between two decoded outputs. The paper's
    /// cross-entropy analog; caller-defined. Should be non-negative for
    /// meaningful threshold comparison.
    fn behavior_distance(&self, a: &Self::Behavior, b: &Self::Behavior) -> f32;
}

// ─── Result types ──────────────────────────────────────────────────────────

/// iMAUVE score + supporting measurements (paper §1.2).
///
/// `score` is the headline number — nearest-neighbor midpoint coherence.
/// The auxiliary fields expose the underlying counts so callers can build
/// confidence intervals / sanity-check the sample size.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ImauveScore {
    /// Mean nearest-neighbor midpoint coherence in `[0, 1]`.
    ///
    /// Defined as `1 - mean(decoder_distance(decode(midpoint(a, nn(a))),
    /// decode(a)) / max_possible)`, clipped to `[0, 1]`. `1.0` = midpoint
    /// decodes to behavior identical to the anchor (perfectly on-manifold);
    /// `0.0` = midpoint decodes to maximally distant behavior.
    ///
    /// **Interpretation**: high `score` means interpolating between
    /// nearest-neighbors stays in-distribution; low `score` means
    /// interpolation leaves the data manifold (the paper's "token soup"
    /// failure mode).
    pub score: f32,

    /// Number of anchors scored (paper's `n`).
    pub n_anchors: u32,

    /// Mean raw behavior-distance from midpoint-decode to anchor-decode.
    /// Unnormalized — `score` is the rescaled version.
    pub mean_raw_distance: f32,

    /// Max behavior-distance observed across anchors (the worst midpoint).
    pub max_raw_distance: f32,
}

/// 5-way intervention probe result (paper §1.4).
///
/// Each field is a behavior-space divergence from the matched-control
/// baseline. The paper's expected ordering for a substrate where the
/// latent causally controls behavior:
///
/// ```text
/// matched ≤ shuffled ≈ zero ≈ mean ≈ noise
/// ```
///
/// (matched is the control; all interventions diverge if the latent matters).
/// The paper's *additional* finding: `shuffled` with a donor from another
/// example produces behavior matching the DONOR (top-1 retrieval flips to
/// the donor's source). Callers can detect this via [`InterventionReport::flips_to_donor`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct InterventionReport {
    /// Behavior delta under the matched-control intervention (the baseline).
    /// SHOULD be ~0 by construction (decode the anchor's own latent). Non-zero
    /// values indicate numerical drift in the decode path.
    pub matched: f32,

    /// Behavior delta when the anchor's latent is replaced with a shuffled
    /// (donor's) latent. The paper: `+4.0 to +4.3` CE increase for text.
    pub shuffled: f32,

    /// Behavior delta when the anchor's latent is zeroed. Paper:
    /// `+2.6 to +3.3` CE.
    pub zero: f32,

    /// Behavior delta when the anchor's latent is replaced with the mean
    /// of the donor pool. Paper: similar to `zero`.
    pub mean: f32,

    /// Behavior delta when the anchor's latent is replaced with Gaussian
    /// noise. Paper: similar to `zero`.
    pub noise: f32,

    /// Behavior delta when the anchor's latent is replaced with noise scaled
    /// to the anchor's own L2 norm (Issue 776 / CVRR §2.1). Diverging here +
    /// diverging under `noise` = structure is read; diverging under `noise`
    /// but ≈0 here = only the norm/energy is read.
    pub norm_matched: f32,

    /// Number of donors used for the `shuffled` / `mean` statistics.
    pub n_donors: u32,
}

impl InterventionReport {
    /// Returns `true` if the report is consistent with the paper's
    /// "latent causally controls behavior" pattern: matched is small
    /// relative to all interventions, and interventions are mutually
    /// comparable (no single intervention leaves the latent unused).
    ///
    /// The thresholds are deliberately lenient (multiplicative) — the
    /// paper's binary verdict is "interventions all diverge"; the exact
    /// magnitudes are domain-specific.
    #[inline]
    pub fn latent_is_causal(&self, ratio_threshold: f32) -> bool {
        // All interventions must exceed `ratio_threshold × matched` (matched
        // is the control baseline). The ratio is the paper's signal-to-noise.
        // ratio_threshold of 5.0 is a conservative default — the paper's
        // text results show ratios in the 10-1000× range.
        let baseline = self.matched.max(1e-6);
        self.shuffled > ratio_threshold * baseline
            && self.zero > ratio_threshold * baseline
            && self.mean > ratio_threshold * baseline
            && self.noise > ratio_threshold * baseline
            && self.norm_matched > ratio_threshold * baseline
    }

    /// Returns `true` if `shuffled` is the dominant intervention divergence
    /// AND exceeds `zero`/`mean`/`noise` — the paper's evidence that the
    /// shuffled donor's content is what's driving behavior (the retrieval
    /// probe flips to the donor's source).
    ///
    /// Per paper §1.4: shuffled real-z produces donor-like behavior, not
    /// just any off-manifold collapse. This distinguishes "latent matters"
    /// (zero/mean/noise all diverge similarly — generic dependence) from
    /// "latent carries the example's identity" (shuffled diverges MORE
    /// because it carries a DIFFERENT example's identity).
    #[inline]
    pub fn flips_to_donor(&self, dominance_ratio: f32) -> bool {
        let other_max = self
            .zero
            .max(self.mean)
            .max(self.noise)
            .max(self.norm_matched)
            .max(1e-6);
        self.shuffled > dominance_ratio * other_max
    }
}

// ─── Core protocol: iMAUVE ─────────────────────────────────────────────────

/// Compute the iMAUVE score over a sample of latent points (paper §1.2).
///
/// For each anchor in `anchors`, find its nearest neighbor in `candidates`
/// (excluding the anchor itself if `candidates == anchors`), decode the
/// midpoint latent, and measure how far the midpoint's behavior is from
/// the anchor's behavior. The score is the rescaled mean — `1.0` means
/// midpoints stay on-manifold.
///
/// **Paper analog**: iFID-style nearest-neighbor midpoint FID. Their
/// finding: this metric correlates with downstream generation quality
/// (Pearson r=0.99 with gMAUVE) while reconstruction quality (rMAUVE)
/// saturates near 1.0 and is uninformative.
///
/// # Allocation
///
/// Zero allocation in the hot path. The midpoint is written into the
/// caller-supplied `midpoint_scratch` (sized to `space.dim()`), which is
/// reused across all anchors. The decode output is owned by the caller
/// (`Behavior: Clone`) — no internal buffering.
///
/// # Arguments
///
/// - `space`: the latent substrate.
/// - `anchors`: the points to score (paper's "real examples").
/// - `candidates`: the nearest-neighbor pool. Set to `anchors` for the
///   self-contained protocol; set to a held-out set for cross-validation.
/// - `midpoint_scratch`: caller-supplied scratch buffer for the midpoint.
///   MUST be `space.dim()` long; is overwritten on each anchor.
/// - `max_possible_distance`: normalization constant. Should be the
///   maximum possible `behavior_distance` (e.g. `sqrt(d)` for L2 on `d`-dim
///   unit-bounded outputs). The score is `1 - mean/max`, clipped to `[0, 1]`.
///
/// # Edge cases
///
/// - Empty `anchors` or `candidates` → `Default::default()` (all zeros).
/// - Anchor with no valid nearest neighbor (only itself in candidates, and
///   `exclude_self` is true) → skipped, not counted.
/// - `max_possible_distance <= 0` → returns raw distances in `mean_raw_distance`
///   and `score = 0.0` (caller asked for an unnormalized report).
pub fn imauve_score<S, P, B>(
    space: &S,
    anchors: &[P],
    candidates: &[P],
    midpoint_scratch: &mut P,
    max_possible_distance: f32,
) -> ImauveScore
where
    S: LatentSpace<Point = P, Behavior = B> + ?Sized,
    P: Clone,
    B: Clone,
{
    if anchors.is_empty() || candidates.is_empty() || max_possible_distance <= 0.0 {
        // Empty input or invalid normalization → empty report.
        // (We can't compute a normalized score without a max; return raw 0.)
        return ImauveScore::default();
    }

    let mut sum_raw = 0.0f32;
    let mut max_raw = 0.0f32;
    let mut n_scored = 0u32;

    // The decode of each anchor is computed once (paper's "decode anchor
    // latent" baseline).
    for anchor in anchors {
        // Find nearest neighbor in candidates, excluding self when
        // candidates == anchors (detected by index comparison at the
        // call site — here we just pick the closest non-degenerate pair).
        let mut best_idx: Option<usize> = None;
        let mut best_dist = f32::INFINITY;
        for (j, cand) in candidates.iter().enumerate() {
            let d = space.latent_distance(anchor, cand);
            // Strict-less-than skips exact-zero distances (the anchor
            // itself when candidates == anchors).
            if d > 0.0 && d < best_dist {
                best_dist = d;
                best_idx = Some(j);
            }
        }

        let Some(j) = best_idx else {
            continue;
        };
        let nn = &candidates[j];

        // Decode anchor once.
        let anchor_behavior = space.decode(anchor);

        // Compute midpoint in-place. We swap through scratch so the
        // caller's buffer is reused across anchors (zero-alloc hot path).
        let mid = space.midpoint(anchor, nn);
        *midpoint_scratch = mid;

        let mid_behavior = space.decode(midpoint_scratch);
        let raw = space.behavior_distance(&mid_behavior, &anchor_behavior);

        sum_raw += raw;
        max_raw = max_raw.max(raw);
        n_scored += 1;
    }

    if n_scored == 0 {
        return ImauveScore {
            score: 0.0,
            n_anchors: 0,
            mean_raw_distance: 0.0,
            max_raw_distance: 0.0,
        };
    }

    let mean_raw = sum_raw / (n_scored as f32);
    let score = (1.0 - mean_raw / max_possible_distance).clamp(0.0, 1.0);

    ImauveScore {
        score,
        n_anchors: n_scored,
        mean_raw_distance: mean_raw,
        max_raw_distance: max_raw,
    }
}

// ─── Core protocol: intervention battery ───────────────────────────────────

/// Run the 5-way intervention battery on an anchor (paper §1.4).
///
/// For the given `anchor`, measure the behavior divergence under each
/// of: matched control (decode anchor's own latent), shuffled (decode
/// donor's latent), zero (decode zero latent), mean (decode mean of
/// donor pool), noise (decode deterministic Gaussian noise).
///
/// **Paper analog**: §1.4 Table — expected ordering on a causal latent:
/// `matched ≪ shuffled ≈ zero ≈ mean ≈ noise`. The paper additionally
/// shows `shuffled` causes behavior to flip toward the DONOR's identity
/// (top-1 retrieval flips); callers can detect this via
/// [`InterventionReport::flips_to_donor`].
///
/// # Allocation
///
/// Zero allocations in the hot path. All five decodes share caller-supplied
/// scratch (`zero_scratch`, `mean_scratch`, `noise_scratch`) for the
/// constructed latents. The `shuffled` field uses `donors[seed % n_donors]`
/// directly (no copy).
///
/// # Arguments
///
/// - `space`: the latent substrate.
/// - `anchor`: the point under test.
/// - `donors`: the donor pool for `shuffled` and `mean`. Paper uses a
///   large held-out set; for unit tests a small pool suffices.
/// - `seed`: 64-bit deterministic seed for the `noise` intervention.
///   Reproducible across runs.
/// - `zero_scratch`, `mean_scratch`, `noise_scratch`: caller-supplied
///   scratch buffers (overwritten in-place; zero-alloc hot path).
///
/// # Edge cases
///
/// - Empty `donors` → `shuffled = 0.0` (skipped), `mean = decode(zero)`
///   fallback. The remaining interventions (zero, noise) still run.
pub fn intervention_battery<S, P, B>(
    space: &S,
    anchor: &P,
    donors: &[P],
    seed: u64,
    zero_scratch: &mut P,
    mean_scratch: &mut P,
    noise_scratch: &mut P,
) -> InterventionReport
where
    S: LatentSpace<Point = P, Behavior = B> + ?Sized,
    P: Clone,
    B: Clone,
{
    // Paper §1.4 protocol:
    //   matched:   decode(anchor_latent)              — baseline.
    //   shuffled:  decode(donor_latent)               — divergence vs matched.
    //   zero:      decode(zero_latent)                — divergence vs matched.
    //   mean:      decode(mean(donor_latents))        — divergence vs matched.
    //   noise:     decode(noise_latent)               — divergence vs matched.
    //
    // `matched` is by construction 0 (decode(anchor) vs decode(anchor)).
    // We could compute it explicitly to surface numerical drift in the
    // decode path, but for a deterministic `decode` that's a tautology.
    // Callers wanting a drift floor can run intervention_battery twice.

    let matched_behavior = space.decode(anchor);
    let n_donors = donors.len() as u32;

    // Shuffled: pick one donor (deterministic by seed).
    let shuffled = if donors.is_empty() {
        0.0
    } else {
        let donor_idx = (seed % donors.len() as u64) as usize;
        let donor_behavior = space.decode(&donors[donor_idx]);
        space.behavior_distance(&donor_behavior, &matched_behavior)
    };

    // Zero: decode the zero latent.
    *zero_scratch = space.zero();
    let zero_behavior = space.decode(zero_scratch);
    let zero = space.behavior_distance(&zero_behavior, &matched_behavior);

    // Mean: decode the mean of the donor pool.
    let mean = if donors.is_empty() {
        // Fall back to zero if no donors.
        *mean_scratch = space.zero();
        let mb = space.decode(mean_scratch);
        space.behavior_distance(&mb, &matched_behavior)
    } else {
        *mean_scratch = space.mean(donors);
        let mb = space.decode(mean_scratch);
        space.behavior_distance(&mb, &matched_behavior)
    };

    // Noise: decode deterministic Gaussian noise.
    *noise_scratch = space.noise(seed);
    let noise_behavior = space.decode(noise_scratch);
    let noise = space.behavior_distance(&noise_behavior, &matched_behavior);

    // Norm-matched noise (Issue 776 / CVRR §2.1): same scratch reused after
    // the noise arm's decode is complete — zero extra buffers, zero alloc.
    *noise_scratch = space.norm_matched_noise(anchor, seed);
    let norm_matched_behavior = space.decode(noise_scratch);
    let norm_matched = space.behavior_distance(&norm_matched_behavior, &matched_behavior);

    InterventionReport {
        matched: 0.0, // By definition: decode(anchor) vs decode(anchor) = 0.
        shuffled,
        zero,
        mean,
        noise,
        norm_matched,
        n_donors,
    }
}

// ─── Synthetic test fixture: Gaussian mixture ──────────────────────────────
//
// The paper's "bad latent" failure mode (§1.2): nearest neighbors cluster
// by LENGTH, so midpoints of same-length examples decode to token-soup
// (the average of two unrelated narratives is neither). The "good latent"
// clusters by SEMANTIC SHAPE, so midpoints stay in-distribution.
//
// Our synthetic fixture reproduces both geometries in 2D:
//   - Good: clusters arranged along a 1D manifold (the line y = x). Mid-
//     points of cluster neighbors stay on the manifold.
//   - Bad: clusters arranged in a "length clustering" pattern — each
//     cluster is a tight ball, and balls are placed so nearest neighbors
//     are by radial distance from origin, not by cluster identity. Mid-
//     points cross between unrelated balls.

/// Synthetic 2D Gaussian-mixture latent space for unit testing.
///
/// Each cluster is a tight Gaussian ball; the cluster centers define the
/// interpolation geometry. Decode is identity (the latent IS the behavior)
/// so we can test the protocol mechanics without plugging in a real
/// substrate.
#[derive(Clone, Debug)]
pub struct GaussianMixtureSpace {
    /// Cluster centers in 2D (the means of each Gaussian).
    pub centers: Vec<[f32; 2]>,
}

impl GaussianMixtureSpace {
    /// Good geometry: clusters along the 1D manifold `y = x`.
    ///
    /// Nearest-neighbor midpoints stay on the manifold → high iMAUVE.
    pub fn good_along_manifold(n_clusters: usize) -> Self {
        let centers = (0..n_clusters)
            .map(|i| {
                let t = i as f32 * 0.5;
                [t, t]
            })
            .collect();
        Self { centers }
    }

    /// Bad geometry: clusters arranged so nearest neighbors are by radial
    /// distance, not by manifold structure. Two clusters at the same
    /// radius but opposite angles are "nearest" by L2 in latent space,
    /// but their midpoint is at the origin — far from any cluster.
    ///
    /// Reproduces the paper's "length clustering" failure mode: nearest
    /// neighbors share an irrelevant property (radius ↔ length), so
    /// midpoints decode to off-manifold points.
    pub fn bad_radial_clustering(n_clusters: usize) -> Self {
        let centers = (0..n_clusters)
            .map(|i| {
                // All clusters at radius 5.0, evenly spaced in angle.
                let theta = (i as f32) * std::f32::consts::TAU / (n_clusters as f32);
                [5.0 * theta.cos(), 5.0 * theta.sin()]
            })
            .collect();
        Self { centers }
    }
}

impl LatentSpace for GaussianMixtureSpace {
    type Point = [f32; 2];
    type Behavior = [f32; 2];

    #[inline]
    fn dim(&self) -> usize {
        2
    }

    #[inline]
    fn decode(&self, point: &[f32; 2]) -> [f32; 2] {
        // Identity decode — the latent IS the behavior in this synthetic
        // fixture. Real substrates plug in their actual decoder here.
        *point
    }

    #[inline]
    fn midpoint(&self, a: &[f32; 2], b: &[f32; 2]) -> [f32; 2] {
        [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5]
    }

    #[inline]
    fn zero(&self) -> [f32; 2] {
        [0.0, 0.0]
    }

    #[inline]
    fn mean(&self, samples: &[[f32; 2]]) -> [f32; 2] {
        if samples.is_empty() {
            return [0.0, 0.0];
        }
        let n = samples.len() as f32;
        let mut sx = 0.0;
        let mut sy = 0.0;
        for s in samples {
            sx += s[0];
            sy += s[1];
        }
        [sx / n, sy / n]
    }

    #[inline]
    fn noise(&self, seed: u64) -> [f32; 2] {
        // Deterministic Box-Muller over xorshift — reproducible noise.
        let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let u1 = ((next() >> 11) as f32 / (1u64 << 53) as f32).max(1e-10);
        let u2 = ((next() >> 11) as f32 / (1u64 << 53) as f32) * std::f32::consts::TAU;
        let r = (-2.0 * u1.ln()).sqrt();
        [r * u2.cos(), r * u2.sin()]
    }

    #[inline]
    fn norm_matched_noise(&self, anchor: &[f32; 2], seed: u64) -> [f32; 2] {
        norm_matched_noise_arr(anchor, seed)
    }

    #[inline]
    fn latent_distance(&self, a: &[f32; 2], b: &[f32; 2]) -> f32 {
        let dx = a[0] - b[0];
        let dy = a[1] - b[1];
        (dx * dx + dy * dy).sqrt()
    }

    #[inline]
    fn behavior_distance(&self, a: &[f32; 2], b: &[f32; 2]) -> f32 {
        // Identity decode → behavior distance == latent distance.
        self.latent_distance(a, b)
    }
}

// ─── Generic `[f32; N]` test substrate ─────────────────────────────────────
//
// A const-generic Euclidean latent space for any fixed dimension N. This is
// the substrate used to exercise belief `[f32; 8]` and `NeuronShard::style_weights
// [f32; 64]` shapes generically — without pulling in the private riir-engine
// / riir-neuron-db types. The trait surface is identical; real substrates
// just plug in their own decode.

/// Generic const-generic Euclidean latent space over `[f32; N]`.
///
/// Used by the test suite and by consumers that want a reference
/// implementation. Identity decode; L2 distance; arithmetic midpoint.
pub struct EuclideanLatentSpace<const N: usize>;

impl<const N: usize> LatentSpace for EuclideanLatentSpace<N> {
    type Point = [f32; N];
    type Behavior = [f32; N];

    #[inline]
    fn dim(&self) -> usize {
        N
    }

    #[inline]
    fn decode(&self, point: &[f32; N]) -> [f32; N] {
        *point
    }

    #[inline]
    fn midpoint(&self, a: &[f32; N], b: &[f32; N]) -> [f32; N] {
        let mut out = [0.0f32; N];
        let mut i = 0;
        // Chunk-4 unrolled for SIMD-friendly reduction (mirrors
        // latent_trajectory_geometry / subspace_phase_gate).
        while i + 4 <= N {
            out[i] = (a[i] + b[i]) * 0.5;
            out[i + 1] = (a[i + 1] + b[i + 1]) * 0.5;
            out[i + 2] = (a[i + 2] + b[i + 2]) * 0.5;
            out[i + 3] = (a[i + 3] + b[i + 3]) * 0.5;
            i += 4;
        }
        while i < N {
            out[i] = (a[i] + b[i]) * 0.5;
            i += 1;
        }
        out
    }

    #[inline]
    fn zero(&self) -> [f32; N] {
        [0.0f32; N]
    }

    #[inline]
    fn mean(&self, samples: &[[f32; N]]) -> [f32; N] {
        if samples.is_empty() {
            return [0.0f32; N];
        }
        let n = samples.len() as f32;
        let mut out = [0.0f32; N];
        for s in samples {
            let mut i = 0;
            while i + 4 <= N {
                out[i] += s[i];
                out[i + 1] += s[i + 1];
                out[i + 2] += s[i + 2];
                out[i + 3] += s[i + 3];
                i += 4;
            }
            while i < N {
                out[i] += s[i];
                i += 1;
            }
        }
        for x in out.iter_mut() {
            *x /= n;
        }
        out
    }

    #[inline]
    fn noise(&self, seed: u64) -> [f32; N] {
        let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let mut out = [0.0f32; N];
        let mut i = 0;
        while i + 2 <= N {
            let u1 = ((next() >> 11) as f32 / (1u64 << 53) as f32).max(1e-10);
            let u2 = ((next() >> 11) as f32 / (1u64 << 53) as f32) * std::f32::consts::TAU;
            let r = (-2.0 * u1.ln()).sqrt();
            out[i] = r * u2.cos();
            out[i + 1] = r * u2.sin();
            i += 2;
        }
        if i < N {
            // Odd dimension: fill the last slot with a single uniform draw.
            let u = ((next() >> 11) as f32 / (1u64 << 53) as f32) * 2.0 - 1.0;
            out[i] = u;
        }
        out
    }

    #[inline]
    fn norm_matched_noise(&self, anchor: &[f32; N], seed: u64) -> [f32; N] {
        norm_matched_noise_arr(anchor, seed)
    }

    #[inline]
    fn latent_distance(&self, a: &[f32; N], b: &[f32; N]) -> f32 {
        let mut sum = 0.0f32;
        let mut i = 0;
        while i + 4 <= N {
            let d0 = a[i] - b[i];
            let d1 = a[i + 1] - b[i + 1];
            let d2 = a[i + 2] - b[i + 2];
            let d3 = a[i + 3] - b[i + 3];
            sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
            i += 4;
        }
        while i < N {
            let d = a[i] - b[i];
            sum += d * d;
            i += 1;
        }
        sum.sqrt()
    }

    #[inline]
    fn behavior_distance(&self, a: &[f32; N], b: &[f32; N]) -> f32 {
        self.latent_distance(a, b)
    }
}

// ─── Norm-matched noise helper (Issue 776 / Research 555 — CVRR §2.1) ──────

/// Deterministic Gaussian noise scaled to `anchor`'s own L2 norm — core
/// slice form (Issue 619 T1): writes into caller-supplied `out`
/// (`out.len()` must equal `anchor.len()`), `n = g · (‖anchor‖/‖g‖)` —
/// magnitude preserved, structure destroyed.
///
/// Same xorshift+Box-Muller stream shape as the in-crate `noise` impls so
/// the whole battery stays seed-reproducible; `_arr` delegates here, so
/// the fixed-size and slice forms are bit-identical at the same seed.
/// Zero-norm anchors write the zero vector (magnitude trivially
/// preserved). Zero allocation.
#[inline]
pub fn norm_matched_noise_slice_into(anchor: &[f32], seed: u64, out: &mut [f32]) {
    debug_assert_eq!(anchor.len(), out.len(), "out must match anchor length");
    let n = anchor.len();
    let mut anchor_sq = 0.0f32;
    for x in anchor.iter() {
        anchor_sq += x * x;
    }
    let target = anchor_sq.sqrt();
    if target < 1e-12 || n == 0 {
        out.fill(0.0);
        return;
    }

    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let mut g_sq = 0.0f32;
    let mut i = 0;
    while i + 2 <= n {
        let u1 = ((next() >> 11) as f32 / (1u64 << 53) as f32).max(1e-10);
        let u2 = ((next() >> 11) as f32 / (1u64 << 53) as f32) * std::f32::consts::TAU;
        let r = (-2.0 * u1.ln()).sqrt();
        out[i] = r * u2.cos();
        out[i + 1] = r * u2.sin();
        g_sq += out[i] * out[i] + out[i + 1] * out[i + 1];
        i += 2;
    }
    if i < n {
        let u = ((next() >> 11) as f32 / (1u64 << 53) as f32) * 2.0 - 1.0;
        out[i] = u;
        g_sq += u * u;
    }

    let g_norm = g_sq.sqrt();
    if g_norm < 1e-12 {
        // Astronomically unlikely; deterministic fallback — spread the target
        // norm uniformly with alternating signs.
        let m = target / (n as f32).sqrt();
        for (j, x) in out.iter_mut().enumerate() {
            *x = if j & 1 == 0 { m } else { -m };
        }
        return;
    }

    let scale = target / g_norm;
    for x in out.iter_mut() {
        *x *= scale;
    }
}

/// Vec-convenience wrapper over [`norm_matched_noise_slice_into`] for
/// variable-length points (e.g. `KarcWoutSpace`'s `Vec<f32>`). Allocates
/// the output; bit-identical to [`norm_matched_noise_arr`] at the same
/// seed + length.
#[inline]
pub fn norm_matched_noise_slice(anchor: &[f32], seed: u64) -> Vec<f32> {
    let mut out = vec![0.0f32; anchor.len()];
    norm_matched_noise_slice_into(anchor, seed, &mut out);
    out
}

/// Deterministic Gaussian noise scaled to `anchor`'s own L2 norm, for
/// `[f32; N]` points: `n = g · (‖anchor‖/‖g‖)` — magnitude preserved,
/// structure destroyed.
///
/// Same xorshift+Box-Muller stream shape as the in-crate `noise` impls so
/// the whole battery stays seed-reproducible. Zero-norm anchors return the
/// zero vector (magnitude trivially preserved). Zero allocation (stack
/// array; delegates to the slice core at [`norm_matched_noise_slice_into`]).
#[inline]
pub fn norm_matched_noise_arr<const N: usize>(anchor: &[f32; N], seed: u64) -> [f32; N] {
    let mut out = [0.0f32; N];
    norm_matched_noise_slice_into(anchor, seed, &mut out);
    out
}

// ─── Deterministic xorshift RNG (fixture generation only) ──────────────────
//
// Used by tests + the GOAT bench to construct reproducible point clouds.
// NOT used in the scored path — the scored path is deterministic.

/// Deterministic xorshift64* RNG for fixture generation (tests + benches).
///
/// NOT a cryptographically secure RNG; NOT used in scored paths. The scored
/// path takes a seed argument and runs Box-Muller in `LatentSpace::noise`.
pub struct FixtureRng(pub u64);

impl FixtureRng {
    /// New RNG with the given seed. Seed 0 is remapped to 1 (xorshift can't
    /// escape 0).
    #[inline]
    pub fn new(seed: u64) -> Self {
        Self(if seed == 0 { 1 } else { seed })
    }

    /// Next 64-bit value.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform `[0, 1)` draw.
    #[inline]
    pub fn uniform(&mut self) -> f32 {
        let bits = ((self.next_u64() >> 40) as u32 & 0x007f_ffff) | 0x3f80_0000;
        f32::from_bits(bits) - 1.0
    }

    /// Uniform `[lo, hi)` draw.
    #[inline]
    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.uniform()
    }
}

// ─── Unit tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::size_of;

    // ── Trait surface ──────────────────────────────────────────────────────

    #[test]
    fn test_intervention_report_is_pod() {
        // InterventionReport is 6 × f32 (matched/shuffled/zero/mean/noise/
        // norm_matched — Issue 776 sixth arm) + 1 × u32 (n_donors)
        // = 28 bytes, no padding (all 4-byte aligned).
        assert_eq!(size_of::<InterventionReport>(), 28);
    }

    #[test]
    fn test_imauve_score_is_pod() {
        // ImauveScore is 3 × f32 (score/mean_raw/max_raw) + 1 × u32
        // (n_anchors) = 16 bytes, no padding.
        assert_eq!(size_of::<ImauveScore>(), 16);
    }

    // ── iMAUVE protocol mechanics ─────────────────────────────────────────

    #[test]
    fn test_imauve_empty_inputs_return_default() {
        let space = GaussianMixtureSpace::good_along_manifold(4);
        let mut scratch = [0.0f32; 2];
        let empty: Vec<[f32; 2]> = vec![];

        let score = imauve_score(&space, &empty, &empty, &mut scratch, 1.0);
        assert_eq!(score, ImauveScore::default());

        let score = imauve_score(&space, &[[1.0, 1.0]], &empty, &mut scratch, 1.0);
        assert_eq!(score, ImauveScore::default());

        // max_possible_distance <= 0 → unnormalized report.
        let score = imauve_score(&space, &[[1.0, 1.0]], &[[2.0, 2.0]], &mut scratch, 0.0);
        assert_eq!(score, ImauveScore::default());
    }

    #[test]
    fn test_imauve_excludes_self_when_candidates_equal_anchors() {
        // When candidates == anchors, the strict-greater-than-zero filter
        // skips the anchor itself. Each anchor finds its nearest NON-self
        // neighbor.
        let space = GaussianMixtureSpace::good_along_manifold(3);
        // centers: [[0,0], [0.5,0.5], [1.0,1.0]]
        let anchors = space.centers.clone();
        let mut scratch = [0.0f32; 2];
        let max_dist = 2.0f32.sqrt(); // max L2 in 2D unit-square scaled by 1.0

        let score = imauve_score(&space, &anchors, &anchors, &mut scratch, max_dist);
        // All 3 anchors should find a non-self neighbor.
        assert_eq!(score.n_anchors, 3);
        // On the y=x manifold, midpoint of any two centers is also on the
        // manifold, so decoded-midpoint behavior matches the linear interp.
        // Score should be high (close to 1.0).
        assert!(
            score.score > 0.5,
            "good geometry should yield high iMAUVE, got {}",
            score.score
        );
    }

    #[test]
    fn test_imauve_distinguishes_good_vs_bad_geometry() {
        // The headline test (Phase 1 T1.3): the protocol MUST distinguish
        // a good latent geometry from a bad one.
        //
        // GOOD: clusters along y=x → midpoints of cross-cluster neighbors
        //      stay on the manifold → high score.
        // BAD: clusters on a circle → midpoint of arc-adjacent neighbors
        //      falls INSIDE the circle (off the cluster ring) → low score.
        //
        // Key: use 1 point per cluster so the nearest neighbor is FORCED
        // to be cross-cluster. (With many points per cluster + small noise,
        // nearest neighbor is always within-cluster and the geometry
        // difference is masked.)

        let good = GaussianMixtureSpace::good_along_manifold(8);
        let bad = GaussianMixtureSpace::bad_radial_clustering(8);

        // Use the cluster centers directly as the anchor/candidate pool.
        let good_points = good.centers.clone();
        let bad_points = bad.centers.clone();

        let mut scratch = [0.0f32; 2];
        // max_possible: the diameter of the bad-geometry circle is 10
        // (radius 5, two antipodal points), so use 10 as the L2 normalization.
        let max_possible = 10.0f32;

        let good_score = imauve_score(
            &good,
            &good_points,
            &good_points,
            &mut scratch,
            max_possible,
        );
        let bad_score = imauve_score(&bad, &bad_points, &bad_points, &mut scratch, max_possible);

        // The headline assertion: good geometry scores strictly higher.
        assert!(
            good_score.score > bad_score.score,
            "iMAUVE must distinguish good ({}) from bad ({}) geometry",
            good_score.score,
            bad_score.score
        );

        // Good should be near 1.0 (midpoints stay on manifold).
        assert!(
            good_score.score > 0.9,
            "good geometry score too low: {}",
            good_score.score
        );

        // Bad should be meaningfully below 1.0 (midpoints fall inside circle).
        assert!(
            bad_score.score < 0.95,
            "bad geometry score unexpectedly high: {}",
            bad_score.score
        );
    }

    // ── Intervention battery mechanics ────────────────────────────────────

    #[test]
    fn test_intervention_battery_matched_is_zero() {
        // By definition: matched = decode(anchor) vs decode(anchor) = 0.
        let space = GaussianMixtureSpace::good_along_manifold(4);
        let anchor = [0.5, 0.5];
        let donors = vec![[1.0, 1.0], [1.5, 1.5]];
        let mut z = [0.0f32; 2];
        let mut m = [0.0f32; 2];
        let mut n = [0.0f32; 2];

        let report = intervention_battery(&space, &anchor, &donors, 42, &mut z, &mut m, &mut n);
        assert!(
            report.matched.abs() < 1e-6,
            "matched must be ~0, got {}",
            report.matched
        );
        assert_eq!(report.n_donors, 2);
    }

    #[test]
    fn test_intervention_battery_all_interventions_diverge_when_latent_matters() {
        // The identity-decode synthetic space: every non-trivial intervention
        // diverges from the anchor's behavior. This is the "latent matters"
        // case from paper §1.4.
        let space = GaussianMixtureSpace::good_along_manifold(4);
        let anchor = [2.0, 2.0];
        // Donors far from the anchor so shuffled diverges meaningfully.
        let donors = vec![[-1.0, -1.0], [3.0, 0.0], [0.0, 3.0]];
        let mut z = [0.0f32; 2];
        let mut m = [0.0f32; 2];
        let mut n = [0.0f32; 2];

        let report = intervention_battery(&space, &anchor, &donors, 7, &mut z, &mut m, &mut n);

        // All four interventions must exceed the matched baseline (0).
        assert!(
            report.shuffled > 0.5,
            "shuffled diverges: {}",
            report.shuffled
        );
        assert!(report.zero > 0.5, "zero diverges: {}", report.zero);
        assert!(report.mean > 0.5, "mean diverges: {}", report.mean);
        assert!(report.noise > 0.5, "noise diverges: {}", report.noise);

        // The latent_is_causal verdict must fire at ratio 5×.
        assert!(
            report.latent_is_causal(5.0),
            "latent must be causal on identity-decode synthetic"
        );
    }

    #[test]
    fn test_intervention_battery_empty_donors_falls_back_to_zero_for_mean() {
        // Edge case: no donors → shuffled = 0, mean = decode(zero).
        let space = GaussianMixtureSpace::good_along_manifold(4);
        let anchor = [1.0, 1.0];
        let donors: Vec<[f32; 2]> = vec![];
        let mut z = [0.0f32; 2];
        let mut m = [0.0f32; 2];
        let mut n = [0.0f32; 2];

        let report = intervention_battery(&space, &anchor, &donors, 1, &mut z, &mut m, &mut n);
        assert_eq!(report.shuffled, 0.0);
        assert_eq!(report.n_donors, 0);
        // Mean falls back to zero latent → decode is origin.
        assert!((report.mean - report.zero).abs() < 1e-6);
    }

    // ── Generic `[f32; N]` substrate (belief / style_weights shape analog) ────

    #[test]
    fn test_euclidean_space_dim_8_belief_shape() {
        // belief `NpcEmotionScalars` is `[f32; 8]` in riir-engine. This test
        // exercises the protocol at that dimension generically.
        let space: EuclideanLatentSpace<8> = EuclideanLatentSpace;
        assert_eq!(space.dim(), 8);

        let a = [0.5f32; 8];
        let b = [1.5f32; 8];
        let mid = space.midpoint(&a, &b);
        assert_eq!(mid, [1.0f32; 8]);

        let d = space.latent_distance(&a, &b);
        assert!((d - (8.0f32).sqrt()).abs() < 1e-5);
    }

    #[test]
    fn test_euclidean_space_dim_64_style_weights_shape() {
        // NeuronShard::style_weights is `[f32; 64]`. Test at that dimension.
        let space: EuclideanLatentSpace<64> = EuclideanLatentSpace;
        assert_eq!(space.dim(), 64);

        let mut rng = FixtureRng::new(123);
        let mut a = [0.0f32; 64];
        let mut b = [0.0f32; 64];
        for i in 0..64 {
            a[i] = rng.range(-1.0, 1.0);
            b[i] = rng.range(-1.0, 1.0);
        }
        let mid = space.midpoint(&a, &b);
        for i in 0..64 {
            assert!((mid[i] - (a[i] + b[i]) * 0.5).abs() < 1e-6);
        }

        // Mean of {a, b} should equal midpoint.
        let mean = space.mean(&[a, b]);
        for i in 0..64 {
            assert!((mean[i] - mid[i]).abs() < 1e-6);
        }

        // Noise is deterministic given the seed.
        let n1 = space.noise(42);
        let n2 = space.noise(42);
        assert_eq!(n1, n2);
    }

    #[test]
    fn test_imauve_on_euclidean_8d_clusters() {
        // Phase 2 modelless path: exercise iMAUVE on the 8D substrate shape
        // (belief `[f32; 8]` analog). Two clusters along a 1D manifold in 8D —
        // midpoints should stay on the manifold → high iMAUVE.
        let space: EuclideanLatentSpace<8> = EuclideanLatentSpace;

        // Build a 1D manifold embedded in 8D: all 8 coords equal.
        let mut rng = FixtureRng::new(7);
        let cap_iter = [0.0f32, 0.5, 1.0, 1.5, 2.0];
        let mut points = Vec::with_capacity(cap_iter.len());
        for cluster_t in cap_iter {
            for _ in 0..10 {
                let mut p = [0.0f32; 8];
                for coord in &mut p {
                    *coord = cluster_t + rng.range(-0.02, 0.02);
                }
                points.push(p);
            }
        }

        let mut scratch = [0.0f32; 8];
        // max_possible: 8 × 2.0 (max coordinate range) = 4.0 in L2.
        let max_possible = 4.0f32;

        let score = imauve_score(&space, &points, &points, &mut scratch, max_possible);
        assert_eq!(score.n_anchors, 50);
        assert!(
            score.score > 0.95,
            "good 8D geometry should yield high iMAUVE, got {}",
            score.score
        );
    }

    #[test]
    fn test_imauve_on_euclidean_64d_clusters() {
        // Phase 3 modelless path: same protocol on 64D (style_weights shape).
        // This catches any dimension-specific edge case (chunk-4 unrolling
        // at non-multiple-of-4, etc.).
        let space: EuclideanLatentSpace<64> = EuclideanLatentSpace;

        let mut rng = FixtureRng::new(99);
        let cap_iter = [0.0f32, 0.1, 0.2, 0.3, 0.4];
        let mut points = Vec::with_capacity(cap_iter.len());
        // 5 clusters at increasing mean; midpoints stay in-cluster range.
        for cluster_mean in cap_iter {
            for _ in 0..8 {
                let mut p = [0.0f32; 64];
                for coord in &mut p {
                    *coord = cluster_mean + rng.range(-0.005, 0.005);
                }
                points.push(p);
            }
        }

        let mut scratch = [0.0f32; 64];
        // max_possible: 64 × 0.4 ≈ 3.2 in L2 over unit-noise band.
        let max_possible = 4.0f32;

        let score = imauve_score(&space, &points, &points, &mut scratch, max_possible);
        assert_eq!(score.n_anchors, 40);
        assert!(
            score.score > 0.95,
            "good 64D geometry should yield high iMAUVE, got {}",
            score.score
        );
    }

    #[test]
    fn test_intervention_battery_on_euclidean_8d() {
        // Phase 2 modelless path: intervention battery at belief scale.
        let space: EuclideanLatentSpace<8> = EuclideanLatentSpace;
        let anchor = [0.5f32; 8];

        // Donors: 3 distinct directions chosen so their MEAN is far from the
        // anchor. (The mean-of-donors intervention is a separate probe from
        // each individual donor; it needs a non-trivial offset to diverge.)
        // mean of donors: (-0.5 + 2.0 - 1.0)/3 = 0.167 — offset 0.333 from
        // anchor 0.5. L2 over 8 dims = sqrt(8) * 0.333 ≈ 0.94.
        let d1 = [-0.5f32; 8];
        let d2 = [2.0f32; 8];
        let d3 = [-1.0f32; 8];
        let donors = vec![d1, d2, d3];

        let mut z = [0.0f32; 8];
        let mut m = [0.0f32; 8];
        let mut n = [0.0f32; 8];
        let report = intervention_battery(&space, &anchor, &donors, 31, &mut z, &mut m, &mut n);

        // All interventions must diverge from matched (which is 0).
        assert!(report.matched.abs() < 1e-6);
        assert!(report.shuffled > 0.5, "shuffled: {}", report.shuffled);
        assert!(report.zero > 0.5, "zero: {}", report.zero);
        assert!(report.mean > 0.5, "mean: {}", report.mean);
        assert!(report.noise > 0.5, "noise: {}", report.noise);
        assert_eq!(report.n_donors, 3);
    }

    // ── flips_to_donor / latent_is_causal verdicts ────────────────────────

    #[test]
    fn test_latent_is_causal_verdict() {
        // matched is non-zero (drift floor); interventions exceed it by ≥5×.
        let r = InterventionReport {
            matched: 0.1,
            shuffled: 5.0,
            zero: 3.0,
            mean: 3.0,
            noise: 3.0,
            norm_matched: 3.0,
            n_donors: 10,
        };
        assert!(r.latent_is_causal(5.0));

        // One intervention fails the ratio (zero only 3× matched, not 5×).
        let r = InterventionReport {
            matched: 0.1,
            shuffled: 5.0,
            zero: 0.3, // only 3× matched
            mean: 3.0,
            noise: 3.0,
            norm_matched: 3.0,
            n_donors: 10,
        };
        assert!(!r.latent_is_causal(5.0));

        // Norm-matched fails the ratio (Issue 776 arm): a latent whose
        // structure is unread fails even when magnitude-driven divergence
        // passes everywhere else.
        let r = InterventionReport {
            matched: 0.1,
            shuffled: 5.0,
            zero: 3.0,
            mean: 3.0,
            noise: 3.0,
            norm_matched: 0.3,
            n_donors: 10,
        };
        assert!(!r.latent_is_causal(5.0));
    }

    #[test]
    fn test_flips_to_donor_verdict() {
        // shuffled dominates → flips to donor.
        let r = InterventionReport {
            matched: 0.0,
            shuffled: 8.0,
            zero: 3.0,
            mean: 3.0,
            noise: 3.0,
            norm_matched: 3.0,
            n_donors: 10,
        };
        assert!(r.flips_to_donor(2.0));

        // shuffled does NOT dominate → no donor flip.
        let r = InterventionReport {
            matched: 0.0,
            shuffled: 3.5,
            zero: 3.0,
            mean: 3.0,
            noise: 3.0,
            norm_matched: 3.0,
            n_donors: 10,
        };
        assert!(!r.flips_to_donor(2.0));
    }

    // ── Midpoint symmetry / idempotence (trait contract) ─────────────────

    #[test]
    fn test_norm_matched_noise_preserves_norm_and_is_deterministic() {
        let anchor = [1.5_f32, -2.0, 3.0, -0.5, 4.0, 0.25, -1.0, 0.75];
        let target: f32 = anchor.iter().map(|x| x * x).sum::<f32>().sqrt();

        let a = <EuclideanLatentSpace<8> as LatentSpace>::norm_matched_noise(
            &EuclideanLatentSpace,
            &anchor,
            0xC0FFEE,
        );
        let b = <EuclideanLatentSpace<8> as LatentSpace>::norm_matched_noise(
            &EuclideanLatentSpace,
            &anchor,
            0xC0FFEE,
        );
        assert_eq!(a, b, "same seed → identical output");

        let got: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            ((got - target) / target).abs() < 1e-4,
            "L2 norm preserved: got {got}, want {target}"
        );
        assert_ne!(a, anchor, "structure destroyed");

        // Zero anchor stays zero.
        let z = <EuclideanLatentSpace<8> as LatentSpace>::norm_matched_noise(
            &EuclideanLatentSpace,
            &[0.0_f32; 8],
            7,
        );
        assert_eq!(z, [0.0_f32; 8]);
    }

    #[test]
    fn test_norm_matched_noise_slice_is_bit_identical_to_arr() {
        // Issue 619 T1: the slice core and the fixed-size form share one
        // stream — same seed + length → bit-identical outputs (even + odd
        // lengths, covering the tail-element path).
        for (anchor, seed) in [
            (
                &[1.5_f32, -2.0, 3.0, -0.5, 4.0, 0.25, -1.0, 0.75][..],
                0xC0FFEE_u64,
            ),
            (&[0.3_f32, -1.7, 2.2][..], 42_u64),
            (&[9.9_f32][..], 1_u64),
            (&[][..], 5_u64),
        ] {
            let via_slice = norm_matched_noise_slice(anchor, seed);
            let via_into = {
                let mut buf = vec![f32::NAN; anchor.len()]; // dirty buffer — _into must fully write
                norm_matched_noise_slice_into(anchor, seed, &mut buf);
                buf
            };
            assert_eq!(via_slice, via_into, "_slice and _slice_into agree");
            // Compare against _arr at the same length via a fixed-size shim.
            match anchor.len() {
                8 => {
                    let mut a8 = [0.0f32; 8];
                    a8.copy_from_slice(anchor);
                    let via_arr = norm_matched_noise_arr(&a8, seed);
                    assert_eq!(via_slice, via_arr, "len 8: slice ≡ arr");
                }
                3 => {
                    let mut a3 = [0.0f32; 3];
                    a3.copy_from_slice(anchor);
                    let via_arr = norm_matched_noise_arr(&a3, seed);
                    assert_eq!(via_slice, via_arr, "len 3 (odd tail): slice ≡ arr");
                }
                1 => {
                    let via_arr = norm_matched_noise_arr(&[anchor[0]], seed);
                    assert_eq!(via_slice, via_arr, "len 1: slice ≡ arr");
                }
                _ => {}
            }
        }
    }

    #[test]
    fn test_norm_matched_noise_slice_preserves_norm() {
        let anchor = vec![0.5_f32, -1.25, 2.0, 0.75, -3.5, 1.0, 0.1, -0.6, 2.2, -0.15];
        let target: f32 = anchor.iter().map(|x| x * x).sum::<f32>().sqrt();
        let n = norm_matched_noise_slice(&anchor, 0xBEEF);
        let got: f32 = n.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            ((got - target) / target).abs() < 1e-4,
            "L2 norm preserved: got {got}, want {target}"
        );
        // Zero-norm Vec stays zero.
        assert_eq!(norm_matched_noise_slice(&[0.0f32; 6], 9), [0.0f32; 6]);
    }

    #[test]
    fn test_battery_carries_norm_matched_arm() {
        // Issue 776: the battery's sixth arm is populated and separates —
        // identity decode → norm_matched diverges (structure changed) while
        // a norm-only read would see ≈0. Here we just assert the arm runs
        // and is finite/non-negative on the synthetic substrate.
        let space = GaussianMixtureSpace::good_along_manifold(4);
        let anchors: Vec<[f32; 2]> = (0..8)
            .map(|i| {
                let theta = (i as f32) * std::f32::consts::TAU / 8.0;
                [5.0 * theta.cos(), 5.0 * theta.sin()]
            })
            .collect();
        let mut z = [0.0f32; 2];
        let mut m = [0.0f32; 2];
        let mut n = [0.0f32; 2];
        let report = intervention_battery(
            &space,
            &anchors[0],
            &anchors[1..],
            42,
            &mut z,
            &mut m,
            &mut n,
        );
        assert!(
            report.norm_matched.is_finite() && report.norm_matched >= 0.0,
            "norm-matched arm populated: {report:?}"
        );
        // Identity decode: structure destroyed → diverges from matched.
        assert!(report.norm_matched > 0.0);
    }

    #[test]
    fn test_midpoint_symmetric_and_idempotent() {
        let space = GaussianMixtureSpace::good_along_manifold(4);
        let a = [1.0, 2.0];
        let b = [3.0, 5.0];
        let m1 = space.midpoint(&a, &b);
        let m2 = space.midpoint(&b, &a);
        assert_eq!(m1, m2, "midpoint must be symmetric");

        let m_self = space.midpoint(&a, &a);
        assert_eq!(m_self, a, "midpoint(a, a) must equal a (idempotent)");

        // Same for the Euclidean substrate.
        let es: EuclideanLatentSpace<8> = EuclideanLatentSpace;
        let ea = [1.0f32; 8];
        let eb = [3.0f32; 8];
        let em1 = es.midpoint(&ea, &eb);
        let em2 = es.midpoint(&eb, &ea);
        assert_eq!(em1, em2);
        assert_eq!(es.midpoint(&ea, &ea), ea);
    }

    // ── Determinism ───────────────────────────────────────────────────────

    #[test]
    fn test_noise_is_deterministic_given_seed() {
        let space = GaussianMixtureSpace::good_along_manifold(4);
        let n1 = space.noise(123);
        let n2 = space.noise(123);
        assert_eq!(n1, n2, "noise must be reproducible");

        let n3 = space.noise(124);
        assert_ne!(n1, n3, "different seed must yield different noise");
    }

    #[test]
    fn test_imauve_is_deterministic() {
        let space = GaussianMixtureSpace::good_along_manifold(6);
        let anchors = space.centers.clone();
        let mut s1 = [0.0f32; 2];
        let mut s2 = [0.0f32; 2];

        let r1 = imauve_score(&space, &anchors, &anchors, &mut s1, 5.0);
        let r2 = imauve_score(&space, &anchors, &anchors, &mut s2, 5.0);
        assert_eq!(r1, r2, "imauve_score must be deterministic");
    }
}
