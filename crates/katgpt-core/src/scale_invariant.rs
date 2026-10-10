//! Scale-Invariant Attention — the position-dependent affine logit schedule.
//!
//! Distillation of Anson, Wang & Aitchison, *Scale-invariant Attention*
//! (arXiv:2505.17083 v2, NeurIPS 2025) — Plan 622, Research 610. The paper
//! derives the closed-form Gaussian-logit schedule under which per-decade
//! unnormalized attention mass stays Θ(1) as context grows:
//!
//! ```text
//! L_t → a_t·L_t + m_t
//! a_t² = 2·[log(t/τ + 1) − log α + β/α],   m_t = −a_t² + β/α
//! ```
//!
//! pinned at the boundary `(a₀², m₀) = (1, 0)` — distance-0 logits are
//! untouched, with a ~τ-token near-identity ramp — which forces
//! **α = β = e^0.5**. Both constants are DERIVED here from the boundary pin
//! (never hard-coded); the single knob is `τ` (the paper's default 10).
//!
//! This is the position-DEPENDENT generalization of [`crate::ssmax`]'s
//! position-INDEPENDENT scalar `s_L·log N` — the signal-diff that justifies a
//! separate primitive (Research 610 §3.6): SSMax consumes aggregate length N,
//! this schedule consumes per-position distance `t = i − j`. Zero learned
//! parameters.
//!
//! # The ordering law (mutual exclusion with SSMax)
//!
//! Both this schedule and SSMax are length temperatures. Arming both stacks
//! two sharpeners and double-sharpens attention. When `scale_invariant_attn`
//! is armed, the SSMax length multiplier is bypassed — and arming both in one
//! [`crate::parallax_attn::ParallaxConfig`] is a loud config error (pinned by
//! test in `parallax_attn::tests`). Never stack them.
//!
//! # The sink carve-out
//!
//! The schedule's Gaussianity premise excludes the BOS attention sink (paper
//! App. J — the QQ-plot check excludes it on every checkpoint measured).
//! Key 0 is therefore EXCLUDED from the transform in
//! [`ScaleInvariantLut::apply_inplace`]: its logit passes through untouched.
//!
//! # Sigmoid-arm support (the novel fusion half)
//!
//! The paper is softmax-only. The tilt-transfer theorem (Research 610,
//! derived for this stack, reviewer-re-derived independently) shows the same
//! LUT serves the normalized-sigmoid arm: under the schedule,
//! `E[σ(L_t)] = (α/2 + o(1))/(t/τ + 1)` — Θ(1) per-decade mass with a slow
//! monotone drift from ≈0.29α toward α/2 (expected behavior, not a defect;
//! the drift is `O(1/√log t)` because `a_t²` grows only logarithmically).
//! Derive via the exponential tilt, never via the probit approximation —
//! `E[σ(L)] ≈ Φ(m/√(1+πa²/8))` mispredicts the mass decay as `t^−2.26` in
//! the growing-variance regime (true: `t^−1`). The G1 tests assert the tilt
//! identity numerically: Gauss–Hermite `E[σ(L_t)]·(t/τ+1)/α` equals
//! `E[σ(−L′_t)]` (`L′ ~ N(1, a_t²)`) within 1e-6, and that equals the closed
//! form `Φ(−1/a_t)` within `C·a_t⁻³` with C fixed once.
//!
//! # Distance semantics (non-causal callers)
//!
//! The schedule is causal (language modeling): distance `t = i − j` for
//! `i ≥ j`. Keys AT or AHEAD of the query (`j ≥ i`) have `t = 0`, whose
//! schedule entry is the exact identity `(a, m) = (1, 0)`. Rather than
//! applying that identity (which would flip a `-0.0` score's sign), those
//! keys are skipped outright — bit-exact pass-through. The shipped SDPA and
//! parallax kernels are full (non-causal) attention; the upper triangle is
//! therefore untouched by design.
//!
//! # Allocation discipline (G4)
//!
//! The interleaved `(a_t, m_t)` LUT is built once at session start (~512 KB
//! @ 64k context; O(N) build, sub-millisecond) and is read-only on the hot
//! path. [`ScaleInvariantLut::apply_inplace`] is allocation-free: one fused
//! affine per causal key, unit-stride descending LUT reads (the contiguous
//! distance slice per query).
//!
//! # Per-head logit calibration (Phase 3 — the precondition law)
//!
//! The schedule's constants assume ~UNIT-VARIANCE base logits (the paper's
//! models carry QK-norm). A raw-logit checkpoint needs a training-free
//! per-head calibration first: measure each head's logit spread once at load
//! ([`SpreadAccumulator`]), freeze the per-head scales into a BLAKE3-checked
//! sidecar ([`HeadCalibration`]; loud-fail on digest mismatch), and divide
//! each head's scores by its scale BEFORE the schedule
//! ([`ScaleInvariantLut::apply_inplace_calibrated`] — the fused divide+affine;
//! `apply_inplace` is the `head_scale = 1.0` pass-through, bit-identical).
//! The calibrated posture is RUNTIME DATA under the one feature — no second
//! feature flag: the absent-table path is the pinned byte-identical
//! posture, and the sidecar's digest is the safety (Plan 622 Phase 3,
//! decision recorded in the plan).
//!
//! References:
//! - Plan 622 — open-primitive spec (Phases 1–7)
//! - Research 610 — distillation + the sigmoid tilt transfer
//! - arXiv:2505.17083 — the paper

// ──────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────

/// The scale-invariant schedule LUT: interleaved `(a_t, m_t)` pairs indexed
/// by causal distance `t`.
///
/// Build once per session/layer with [`ScaleInvariantLut::build`] (the
/// `RopeFreqTable` precedent: preallocate at session start, read-only hot
/// path), then apply per score row with
/// [`ScaleInvariantLut::apply_inplace`].
#[derive(Debug, Clone)]
pub struct ScaleInvariantLut {
    /// Interleaved schedule pairs `[a_0, m_0, a_1, m_1, …]` — length
    /// `2·(max_distance + 1)`.
    am: Vec<f32>,
    /// The schedule's single knob: the ramp length τ. Paper default 10.
    tau: f32,
}

impl ScaleInvariantLut {
    /// Highest distance the LUT can serve. Applying a query whose distance
    /// exceeds this fails loud (slice index panic in both profiles).
    #[inline]
    pub fn max_distance(&self) -> usize {
        self.am.len() / 2 - 1
    }

    /// The schedule's single knob `τ`.
    #[inline]
    pub fn tau(&self) -> f32 {
        self.tau
    }

    /// The schedule pair `(a_t, m_t)` at causal distance `t`.
    ///
    /// Diagnostics (Phase 5 per-decade probes) and test references read the
    /// schedule through this instead of the private interleaved layout.
    #[inline]
    pub fn pair(&self, t: usize) -> (f32, f32) {
        (self.am[2 * t], self.am[2 * t + 1])
    }

    /// Build the LUT for distances `0..=max_ctx`.
    ///
    /// The constants α and β are derived from the boundary pin
    /// `(a₀², m₀) = (1, 0)` in f64, never hard-coded:
    ///
    /// ```text
    /// m₀ = −a₀² + β/α = 0        ⟹ β/α = 1
    /// a₀² = 2·[−log α + β/α] = 1 ⟹ log α = β/α − ½
    /// ```
    ///
    /// giving α = β = e^0.5. The paper's validity requirement
    /// `β ≥ α·log α` is asserted (guaranteed by the pin; kept as the
    /// documented gate so a future schedule variant fails loud).
    ///
    /// `max_ctx` must cover the largest query distance any caller will apply
    /// (the sequence length for the shipped full-attention kernels); a
    /// distance beyond `max_ctx` fails loud at apply time.
    pub fn build(tau: f32, max_ctx: usize) -> Self {
        assert!(
            tau.is_finite() && tau > 0.0,
            "scale-invariant schedule: tau must be finite positive, got {tau}"
        );
        const A0_SQUARED: f64 = 1.0;
        const M0: f64 = 0.0;
        let beta_over_alpha = M0 + A0_SQUARED;
        let log_alpha = beta_over_alpha - A0_SQUARED * 0.5;
        assert!(
            beta_over_alpha >= log_alpha,
            "scale-invariant schedule invalid: β ≥ α·log α violated \
             (β/α = {beta_over_alpha}, log α = {log_alpha})"
        );
        let mut am = Vec::with_capacity((max_ctx + 1) * 2);
        for t in 0..=max_ctx {
            // f64 schedule arithmetic; the f32 cast error (~6e-8 relative)
            // is two orders under the G1 1e-6 relative gate.
            let x = t as f64 / tau as f64 + 1.0;
            let a_sq = 2.0 * (x.ln() - log_alpha + beta_over_alpha);
            am.push(a_sq.sqrt() as f32);
            am.push((-a_sq + beta_over_alpha) as f32);
        }
        Self { am, tau }
    }

    /// Apply the schedule to one score row in place: `s → a_t·s + m_t` per
    /// key, `t = query_i − key_j` the causal distance. The uncalibrated
    /// posture — exactly [`Self::apply_inplace_calibrated`] at
    /// `head_scale = 1.0` (bit-identical: `x · 1.0 == x` for every f32).
    ///
    /// # Arguments
    ///
    /// - `scores` — the row's pre-normalization scores, modified in place.
    /// - `query_i` — the query's global index.
    /// - `key_start` — the global index of `scores[0]` (the slice's first
    ///   key). Row-wide calls from the parallax kernels pass `0`.
    /// - `logit_scale` — the softmax scale NOT yet folded into `scores`:
    ///   the affine is applied in pre-scale space (`s → a_t·s +
    ///   m_t/logit_scale`) so a kernel that multiplies by `logit_scale` at
    ///   its exp step realizes exactly `a_t·(scale·dot) + m_t` — folding the
    ///   scale AFTER the affine would wrongly rescale `m_t` (the additive
    ///   schedule term is not a content score; the same law that keeps the
    ///   prior lane unscaled). Pass `1.0` when `scores` are already scaled
    ///   logits (the parallax kernels multiply the scale into the dot
    ///   products directly) — `m·1.0 == m` is IEEE-exact, so `1.0` is
    ///   bit-identical to applying the schedule in logit space.
    ///
    /// # Contract
    ///
    /// - **Sink carve-out:** key 0 (BOS) is never transformed. It is in this
    ///   slice only when `key_start == 0`.
    /// - **Future keys** (`key_j > query_i`, i.e. `t = 0`) are left BIT-EXACT
    ///   untouched — their schedule entry is the identity, and applying
    ///   `1.0·s + 0.0` would still flip a `-0.0` score's sign.
    /// - **Loud bounds:** `query_i − key_start` must be ≤
    ///   [`Self::max_distance`] — a debug assert plus the slice-index panic
    ///   in release; rebuild with a larger `max_ctx` rather than clamping
    ///   (a clamped schedule is silently wrong at the decades the clamp
    ///   flattens).
    ///
    /// # Allocation discipline (G4)
    ///
    /// Zero allocation: one fused affine per causal key over a contiguous
    /// descending distance slice; the interleaved LUT makes each element two
    /// adjacent loads (`a = am[2t]`, `m = am[2t+1]`).
    #[inline]
    pub fn apply_inplace(
        &self,
        scores: &mut [f32],
        query_i: usize,
        key_start: usize,
        logit_scale: f32,
    ) {
        self.apply_inplace_calibrated(scores, query_i, key_start, logit_scale, 1.0);
    }

    /// The calibrated apply (Plan 622 Phase 3): the head's scores are
    /// divided by `head_scale` (its measured logit σ from the frozen
    /// [`HeadCalibration`] table) BEFORE the schedule — the normalization
    /// the schedule's unit-variance premise requires on a raw-logit
    /// checkpoint. One fused loop (no extra pass): `s → a_t·(s/σ_h) +
    /// m_t/logit_scale`.
    ///
    /// `head_scale = 1.0` is BIT-IDENTICAL to [`Self::apply_inplace`] (the
    /// pass-through pin, G3) — `x · 1.0 == x` for every f32 including
    /// `-0.0`; a non-finite or non-positive scale fails loud (a divided-by-
    /// zero or NaN score row would silently poison the softmax).
    #[inline]
    pub fn apply_inplace_calibrated(
        &self,
        scores: &mut [f32],
        query_i: usize,
        key_start: usize,
        logit_scale: f32,
        head_scale: f32,
    ) {
        assert!(
            logit_scale.is_finite() && logit_scale > 0.0,
            "scale-invariant schedule: logit_scale must be finite positive, got {logit_scale}"
        );
        assert!(
            head_scale.is_finite() && head_scale > 0.0,
            "scale-invariant calibration: head_scale must be finite positive, got {head_scale}"
        );
        let Some(t_hi) = query_i.checked_sub(key_start) else {
            return; // every key in the slice is at/ahead of the query: untouched
        };
        debug_assert!(
            t_hi <= self.max_distance(),
            "scale-invariant LUT too small: distance {} > max_distance {} — rebuild with a larger max_ctx",
            t_hi,
            self.max_distance(),
        );
        let causal_len = scores.len().min(t_hi + 1);
        // Sink carve-out: key 0 is in this slice only when the slice starts
        // at key 0; j == 0 is then the sink and is skipped. The calibration
        // divide respects the same carve-out — the sink's logit is outside
        // the schedule's Gaussianity premise on BOTH axes (paper App. J).
        let j_start = usize::from(key_start == 0);
        // Pre-scale space: the caller's exp step multiplies by logit_scale,
        // so m_t enters divided by it (a_t multiplies — commutes exactly);
        // the calibration divide runs on the score BEFORE the affine.
        let inv_scale = 1.0 / logit_scale;
        let inv_head = 1.0 / head_scale;
        let am = &self.am;
        for (j, s) in scores[..causal_len].iter_mut().enumerate().skip(j_start) {
            let t = t_hi - j;
            let a = am[2 * t];
            let m = am[2 * t + 1];
            *s = a * (*s * inv_head) + m * inv_scale;
        }
    }
}

/// The armed-schedule handle threaded through the SDPA kernel chain: the
/// LUT + the head's calibration scale (Plan 622 Phase 3). One parameter
/// instead of two — the head divide is meaningful only with the schedule
/// armed, and `head_scale = 1.0` is bit-identical to the uncalibrated
/// posture (`x · 1.0 == x` for every f32), which keeps every existing call
/// site byte-exact (G3).
#[derive(Debug, Clone, Copy)]
pub struct SiArm<'a> {
    /// The schedule LUT (session-built, read-only hot path).
    pub lut: &'a ScaleInvariantLut,
    /// The head's measured logit σ — the divide the calibrated apply
    /// runs before the affine. `1.0` = uncalibrated.
    pub head_scale: f32,
}

impl<'a> SiArm<'a> {
    /// The uncalibrated arm (the Plan-622-Phase-2 posture, byte-exact).
    #[inline]
    #[must_use]
    pub fn uncalibrated(lut: &'a ScaleInvariantLut) -> Self {
        Self { lut, head_scale: 1.0 }
    }

    /// The calibrated arm — `head_scale` is the head's measured σ from the
    /// frozen [`HeadCalibration`] table; non-finite or non-positive fails
    /// loud here, at the arm's construction, never deep in the kernel.
    #[inline]
    pub fn calibrated(lut: &'a ScaleInvariantLut, head_scale: f32) -> Self {
        assert!(
            head_scale.is_finite() && head_scale > 0.0,
            "scale-invariant calibration: head_scale must be finite positive, got {head_scale}"
        );
        Self { lut, head_scale }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Per-head logit calibration (Phase 3): the load-pass probe + the frozen
// table. Runtime data under the ONE feature — no second flag.
// ──────────────────────────────────────────────────────────────────────────

/// The load-pass logit-spread probe (Plan 622 Phase 3 row 1): per-head
/// Welford accumulators over the scores ONE calibration forward feeds —
/// count / mean / M2 per head, `std = sqrt(M2 / (count − 1))`.
///
/// Load-time instrument, never a hot-path type. What to feed: the same
/// population the schedule will serve — the head's pre-normalization logits
/// over a representative calibration batch, with the SINK column dropped
/// (the schedule never transforms key 0, so the sink's outsized logit must
/// not enter the head's σ; the paper's Gaussianity premise excludes it).
/// The digest-checked lifecycle: measure → [`HeadCalibration::build`] →
/// freeze the sidecar beside the checkpoint → load-time
/// [`HeadCalibration::from_sidecar`] with the loud-fail.
#[derive(Debug, Clone, Default)]
pub struct SpreadAccumulator {
    stats: Vec<(u64, f64, f64)>, // (count, mean, m2) per head
}

impl SpreadAccumulator {
    /// An accumulator for `heads` heads (grow-on-first-observe is fine too;
    /// the constructor just pre-sizes).
    #[must_use]
    pub fn new(heads: usize) -> Self {
        Self {
            stats: vec![(0, 0.0, 0.0); heads],
        }
    }

    /// Feed one head's scores (any batching: call per row, per tile, or per
    /// whole buffer — the Welford merge is order-exact per head, and heads
    /// are independent slots).
    pub fn observe_head(&mut self, head: usize, scores: &[f32]) {
        if self.stats.len() <= head {
            self.stats.resize(head + 1, (0, 0.0, 0.0));
        }
        let (count, mean, m2) = &mut self.stats[head];
        for &x in scores {
            let x = f64::from(x);
            *count += 1;
            let delta = x - *mean;
            // u32 cast: the count reaching u32::MAX means a single head was
            // fed > 4×10⁹ elements — not a measurement posture; saturate
            // (the mean's precision is long gone before the cast matters).
            *mean += delta / f64::from(u32::try_from(*count).unwrap_or(u32::MAX));
            *m2 += delta * (x - *mean);
        }
    }

    /// The head's sample std estimate. `None` below 2 observations (a σ from
    /// one sample is not an estimate — and m2/(count−1) would divide by
    /// zero into NaN); the caller decides whether that refuses the
    /// calibration or keeps sampling.
    #[must_use]
    pub fn std_estimate(&self, head: usize) -> Option<f64> {
        let (count, _mean, m2) = *self.stats.get(head)?;
        if count < 2 {
            return None;
        }
        // u32 cast: a calibration forward feeding > 4×10⁹ score elements to
        // ONE head is not a measurement posture (and the f64 m2 accumulation
        // has long since lost meaning at that count); the try guards it loud.
        let denom = u32::try_from(count - 1).ok()?;
        Some((m2 / f64::from(denom)).sqrt())
    }

    /// Heads observed so far.
    #[must_use]
    pub fn heads(&self) -> usize {
        self.stats.len()
    }
}

/// The frozen per-head scale table (Plan 622 Phase 3 row 2): one positive
/// σ per head, BLAKE3-digested, with the sidecar bytes the checkpoint ships
/// beside. Loud-fail everywhere: a digest mismatch, a wrong head count, a
/// non-finite or non-positive scale all refuse — a silently wrong
/// calibration poisons the schedule's premise, so nothing here
/// best-efforts.
#[derive(Debug, Clone, PartialEq)]
pub struct HeadCalibration {
    scales: Vec<f32>,
    digest: [u8; 32],
}

/// The sidecar magic ("SICAL" — scale-invariant calibration).
const SIDECAR_MAGIC: [u8; 5] = *b"SICAL";
/// The sidecar format version (bumped only on a wire break).
const SIDECAR_VERSION: u32 = 1;

impl HeadCalibration {
    /// Build + digest from a measured scale table. Every scale must be
    /// finite positive (a zero-σ head is a degenerate checkpoint reading —
    /// refuse at build, never at serve).
    pub fn build(scales: Vec<f32>) -> Result<Self, String> {
        if scales.is_empty() {
            return Err("head calibration: empty scale table".to_string());
        }
        for (h, s) in scales.iter().enumerate() {
            if !s.is_finite() || *s <= 0.0 {
                return Err(format!(
                    "head calibration: head {h} scale must be finite positive, got {s}"
                ));
            }
        }
        let digest = Self::digest_of(SIDECAR_VERSION, &scales);
        Ok(Self { scales, digest })
    }

    fn digest_of(version: u32, scales: &[f32]) -> [u8; 32] {
        let mut h = blake3::Hasher::new();
        h.update(&SIDECAR_MAGIC);
        h.update(&version.to_le_bytes());
        h.update(&u32::try_from(scales.len()).expect("head count fits u32").to_le_bytes());
        for s in scales {
            h.update(&s.to_le_bytes());
        }
        *h.finalize().as_bytes()
    }

    /// The head's σ (the divide the calibrated apply consumes).
    /// Out of bounds fails loud — a head index beyond the table is a
    /// checkpoint/sidecar mismatch, never a default.
    #[inline]
    #[must_use]
    pub fn scale_for(&self, head: usize) -> f32 {
        self.scales[head]
    }

    /// Heads covered.
    #[must_use]
    pub fn heads(&self) -> usize {
        self.scales.len()
    }

    /// The BLAKE3 digest over the table (the checkpoint-sidecar binding the
    /// loader can pin).
    #[must_use]
    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }

    /// The table must cover exactly the checkpoint's head count — a
    /// mismatched sidecar is the wrong sidecar.
    pub fn validate_heads(&self, expected: usize) -> Result<(), String> {
        if self.scales.len() != expected {
            return Err(format!(
                "head calibration: sidecar covers {} head(s), checkpoint has {expected}",
                self.scales.len()
            ));
        }
        Ok(())
    }

    /// The frozen sidecar bytes: magic + version + head count + LE f32
    /// scales + the BLAKE3 digest over everything before it.
    #[must_use]
    pub fn to_sidecar(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(5 + 4 + 4 + self.scales.len() * 4 + 32);
        out.extend_from_slice(&SIDECAR_MAGIC);
        out.extend_from_slice(&SIDECAR_VERSION.to_le_bytes());
        out.extend_from_slice(&u32::try_from(self.scales.len()).expect("head count fits u32").to_le_bytes());
        for s in &self.scales {
            out.extend_from_slice(&s.to_le_bytes());
        }
        out.extend_from_slice(&self.digest);
        out
    }

    /// Load + verify sidecar bytes. ANY deviation — magic, version, length,
    /// digest, scale validity — is a loud `Err` naming the cause.
    pub fn from_sidecar(bytes: &[u8]) -> Result<Self, String> {
        let header = 5 + 4 + 4;
        if bytes.len() < header + 32 {
            return Err(format!(
                "head calibration sidecar: {} byte(s) — too short for the header + digest",
                bytes.len()
            ));
        }
        if bytes[..5] != SIDECAR_MAGIC {
            return Err("head calibration sidecar: bad magic (expected SICAL)".to_string());
        }
        let version = u32::from_le_bytes(bytes[5..9].try_into().expect("4 bytes"));
        if version != SIDECAR_VERSION {
            return Err(format!(
                "head calibration sidecar: unsupported version {version} (expected {SIDECAR_VERSION})"
            ));
        }
        let heads = u32::from_le_bytes(bytes[9..13].try_into().expect("4 bytes")) as usize;
        let scales_len = heads * 4;
        if bytes.len() != header + scales_len + 32 {
            return Err(format!(
                "head calibration sidecar: {} byte(s) — expected {} for {heads} head(s)",
                bytes.len(),
                header + scales_len + 32
            ));
        }
        let mut scales = Vec::with_capacity(heads);
        for h in 0..heads {
            let b = &bytes[header + h * 4..header + h * 4 + 4];
            scales.push(f32::from_le_bytes(b.try_into().expect("4 bytes")));
        }
        let claimed: [u8; 32] = bytes[header + scales_len..].try_into().expect("32 bytes");
        let computed = Self::digest_of(version, &scales);
        if claimed != computed {
            return Err(
                "head calibration sidecar: BLAKE3 digest MISMATCH — the table does not match \
                 its digest (wrong sidecar for this checkpoint, or corrupted in transit)"
                    .to_string(),
            );
        }
        // Validity re-checked on the decode path too (a digest-valid table
        // with a zero scale could only come from a foreign builder — refuse
        // anyway; the build-side check must not be the only gate).
        for (h, s) in scales.iter().enumerate() {
            if !s.is_finite() || *s <= 0.0 {
                return Err(format!(
                    "head calibration sidecar: head {h} scale must be finite positive, got {s}"
                ));
            }
        }
        Ok(Self { scales, digest: claimed })
    }
}

/// `E_{N(mu, sigma²)}[f]` by adaptive Simpson on the standardized variable
/// `z = (x − mu)/sigma` over ±40 sigma.
///
/// The diagnostic quadrature behind the Phase-2 G1 tilt assertions and the
/// Phase-5 sigmoid probes (Plan 622). σ-shaped integrands carry a transition
/// of width ~1/a in z — Gauss–Hermite nodes (sparse near zero) MISS it once
/// a ≳ 5, reading quadrature noise as the answer; adaptive subdivision
/// resolves it. (GH stays the instrument for the `e^x` Lemma — entire
/// integrand, spectrally exact there. And MC has no support on the σ tail:
/// at t = 10⁵ the sigmoid mass is ~7e-5, so 100k draws see ~7 events.)
pub fn gaussian_expectation(mu: f64, sigma: f64, f: impl Fn(f64) -> f64) -> f64 {
    let g = |z: f64| {
        let x = mu + sigma * z;
        f(x) * (-0.5 * z * z).exp() / (2.0 * core::f64::consts::PI).sqrt()
    };
    // Adaptive Simpson: returns the integral of g over [a, b] with
    // whole/halves refinement, tolerance absolute + relative.
    #[allow(clippy::too_many_arguments)]
    fn rec(
        g: &impl Fn(f64) -> f64,
        a: f64,
        b: f64,
        fa: f64,
        fm: f64,
        fb: f64,
        tol: f64,
        depth: usize,
    ) -> f64 {
        let m = 0.5 * (a + b);
        let h = 0.5 * (b - a);
        let lm = 0.5 * (a + m);
        let rm = 0.5 * (m + b);
        let flm = g(lm);
        let frm = g(rm);
        let whole = h / 3.0 * (fa + 4.0 * fm + fb);
        let left = h / 6.0 * (fa + 4.0 * flm + fm);
        let right = h / 6.0 * (fm + 4.0 * frm + fb);
        let delta = left + right - whole;
        if depth == 0 || delta.abs() <= 15.0 * tol {
            return left + right + delta / 15.0;
        }
        rec(g, a, m, fa, flm, fm, 0.5 * tol, depth - 1)
            + rec(g, m, b, fm, frm, fb, 0.5 * tol, depth - 1)
    }
    let (a, b) = (-40.0, 40.0);
    let m = 0.5 * (a + b);
    rec(&g, a, b, g(a), g(m), g(b), 1e-12, 52)
}

// ──────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-score generator (no RNG dep): smooth, varied
    /// signs, includes negative zeros via the explicit entry below.
    fn gen_scores(n: usize) -> Vec<f32> {
        let mut v: Vec<f32> = (0..n)
            .map(|i| ((i as f32) * 0.37).sin() * 3.0 - 0.5)
            .collect();
        if n > 2 {
            v[2] = -0.0; // the identity-transform sign-flip probe
        }
        v
    }

    /// Independent f64 re-derivation of the schedule from the boundary pin —
    /// the closed form the LUT must reproduce (G1: LUT vs closed form).
    fn closed_form(t: f64, tau: f64) -> (f64, f64) {
        // Same pin arithmetic as `build`, spelled independently.
        let beta_over_alpha = 0.0_f64 + 1.0; // m₀ = −a₀² + β/α = 0
        let log_alpha = beta_over_alpha - 1.0 * 0.5; // a₀² = 2[−log α + β/α] = 1
        let x = t / tau + 1.0;
        let a_sq = 2.0 * (x.ln() - log_alpha + beta_over_alpha);
        (a_sq.sqrt(), -a_sq + beta_over_alpha)
    }

    #[test]
    fn boundary_pin_is_exact() {
        let lut = ScaleInvariantLut::build(10.0, 64);
        // a₀² = 1 and m₀ = 0 EXACTLY in f32 — the derivation's whole point.
        assert_eq!(lut.am[0], 1.0, "a_0 must be exactly 1.0");
        assert_eq!(lut.am[1], 0.0, "m_0 must be exactly 0.0");
        // Applying at distance 0 is therefore the exact identity…
        let mut s = vec![1.5f32, -2.0, 0.25];
        let before = s.clone();
        lut.apply_inplace(&mut s, 3, 3, 1.0);
        assert_eq!(s, before, "distance-0 application must be bit-exact identity");
    }

    #[test]
    fn lut_matches_closed_form_to_1e6_relative() {
        let tau = 10.0;
        let lut = ScaleInvariantLut::build(tau, 1_000_000);
        for &t in &[0usize, 1, 2, 3, 7, 10, 25, 99, 1000, 12_345, 999_999] {
            let (a_cf, m_cf) = closed_form(t as f64, tau as f64);
            let (a_lut, m_lut) = (lut.am[2 * t] as f64, lut.am[2 * t + 1] as f64);
            let tol = |v: f64| 1e-6 * v.abs().max(1.0);
            assert!(
                (a_lut - a_cf).abs() <= tol(a_cf),
                "a_{t}: lut {a_lut} vs closed form {a_cf}"
            );
            assert!(
                (m_lut - m_cf).abs() <= tol(m_cf),
                "m_{t}: lut {m_lut} vs closed form {m_cf}"
            );
        }
    }

    #[test]
    fn ramp_is_monotone_and_near_identity_for_t_le_tau() {
        let tau = 10.0f64;
        let lut = ScaleInvariantLut::build(10.0, 64);
        let mut prev_a_sq = 1.0f64;
        let mut prev_m = 0.0f64;
        for t in 1..=10usize {
            let (a, m) = (lut.am[2 * t] as f64, lut.am[2 * t + 1] as f64);
            let a_sq = a * a;
            assert!(
                a_sq > prev_a_sq,
                "a_t² must grow monotonically on the ramp (t={t})"
            );
            assert!(
                m < prev_m,
                "m_t must fall monotonically on the ramp (t={t})"
            );
            prev_a_sq = a_sq;
            prev_m = m;
        }
        // The ramp stays near identity: a² ∈ (1, 2.5), m ∈ (−1.5, 0) for the
        // whole t ≤ τ window (closed form at t=τ: a² = 2·ln2 + 1 ≈ 2.386,
        // m = −a² + 1 ≈ −1.386 — derived, not pinned).
        let (a_tau, m_tau) = closed_form(10.0, tau);
        assert!(a_tau * a_tau < 2.5 && m_tau > -1.5);
        for t in 1..=10usize {
            let a_sq = (lut.am[2 * t] as f64).powi(2);
            let m = lut.am[2 * t + 1] as f64;
            assert!(a_sq > 1.0 && a_sq < 2.5, "a² out of the ramp band at t={t}");
            assert!(m < 0.0 && m > -1.5, "m out of the ramp band at t={t}");
        }
    }

    #[test]
    fn no_nan_to_one_million() {
        let lut = ScaleInvariantLut::build(10.0, 1_000_000);
        assert_eq!(lut.max_distance(), 1_000_000);
        assert!(lut.am.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn apply_matches_naive_reference_bit_exact() {
        let lut = ScaleInvariantLut::build(10.0, 128);
        for &(query_i, key_start, len) in &[
            (5usize, 0usize, 6usize),
            (17, 4, 9),
            (64, 60, 4),
            (0, 0, 1),
            (9, 12, 5), // all-future slice
        ] {
            let mut s = gen_scores(len);
            lut.apply_inplace(&mut s, query_i, key_start, 1.0);
            // Naive reference with the SAME per-element rules.
            let mut r = gen_scores(len);
            for (j, s) in r.iter_mut().enumerate() {
                let key_j = key_start + j;
                if key_j == 0 || key_j > query_i {
                    continue; // sink carve-out + future pass-through
                }
                let t = query_i - key_j;
                let (a, m) = closed_form(t as f64, 10.0);
                *s = (a as f32) * *s + (m as f32);
            }
            assert_eq!(s, r, "apply at query={query_i} key_start={key_start}");
        }
    }

    #[test]
    fn sink_and_future_keys_are_bit_untouched() {
        let lut = ScaleInvariantLut::build(10.0, 128);
        let mut s = gen_scores(8);
        s[0] = 7.5; // the sink slot (key 0)
        s[6] = -0.0; // a future key carrying negative zero (query 5)
        let before = s.clone();
        lut.apply_inplace(&mut s, 5, 0, 1.0);
        assert_eq!(s[0], before[0], "sink key must be untouched");
        for j in 6..8 {
            assert_eq!(s[j], before[j], "future key {j} must be untouched (incl. −0.0)");
        }
        // Sanity: causal keys DID move (the test is not vacuous).
        assert_ne!(s[4], before[4], "causal key 4 must transform");
    }

    #[test]
    fn all_future_slice_is_noop() {
        let lut = ScaleInvariantLut::build(10.0, 128);
        let mut s = gen_scores(5);
        let before = s.clone();
        lut.apply_inplace(&mut s, 2, 7, 1.0); // keys 7..12, query 2: all future
        assert_eq!(s, before, "an all-future slice must be a bit-exact no-op");
    }

    #[test]
    fn lut_too_small_fails_loud() {
        let lut = ScaleInvariantLut::build(10.0, 8);
        let mut s = vec![0.0f32; 4];
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            lut.apply_inplace(&mut s, 9, 0, 1.0) // distance 9 > max 8
        }));
        assert!(result.is_err(), "distance beyond max_ctx must fail loud");
    }

    #[test]
    fn build_rejects_bad_tau() {
        assert!(std::panic::catch_unwind(|| ScaleInvariantLut::build(0.0, 4)).is_err());
        assert!(std::panic::catch_unwind(|| ScaleInvariantLut::build(f32::NAN, 4)).is_err());
    }

    // ── Gauss–Hermite quadrature (the G1 instrument) ─────────────────────

    /// Gauss–Hermite nodes/weights (physicist weight e^{−x²}) via
    /// Golub–Welsch: eigenpairs of the symmetric tridiagonal Jacobi matrix
    /// (zero diagonal, off-diagonal √(k/2)); weights `w_i = √π·(v_{0,i})²`
    /// from the accumulated rotation matrix. Deterministic, dependency-free.
    /// Convergence is asserted by the moment checks in
    /// [`gauss_hermite_moments_are_exact`].
    fn gauss_hermite(n: usize) -> (Vec<f64>, Vec<f64>) {
        let mut a = vec![0.0f64; n * n];
        for k in 1..n {
            let v = (k as f64 / 2.0).sqrt();
            a[k * n + (k - 1)] = v;
            a[(k - 1) * n + k] = v;
        }
        let mut v = vec![0.0f64; n * n];
        for i in 0..n {
            v[i * n + i] = 1.0;
        }
        // Cyclic Jacobi rotations on A, accumulated into V (V ← V·J).
        for _ in 0..60 {
            let mut off = 0.0f64;
            for i in 0..n {
                for j in (i + 1)..n {
                    off += a[i * n + j] * a[i * n + j];
                }
            }
            if off < 1e-28 {
                break;
            }
            for p in 0..n {
                for q in (p + 1)..n {
                    let apq = a[p * n + q];
                    if apq.abs() < 1e-18 {
                        continue;
                    }
                    let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                    let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                    let c = 1.0 / (t * t + 1.0).sqrt();
                    let s = t * c;
                    for k in 0..n {
                        let akp = a[k * n + p];
                        let akq = a[k * n + q];
                        a[k * n + p] = c * akp - s * akq;
                        a[k * n + q] = s * akp + c * akq;
                        let vkp = v[k * n + p];
                        let vkq = v[k * n + q];
                        v[k * n + p] = c * vkp - s * vkq;
                        v[k * n + q] = s * vkp + c * vkq;
                    }
                    for k in 0..n {
                        let apk = a[p * n + k];
                        let aqk = a[q * n + k];
                        a[p * n + k] = c * apk - s * aqk;
                        a[q * n + k] = s * apk + c * aqk;
                    }
                }
            }
        }
        let sqrt_pi = std::f64::consts::PI.sqrt();
        let nodes: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
        let weights: Vec<f64> = (0..n).map(|i| sqrt_pi * v[i] * v[i]).collect();
        (nodes, weights)
    }

    /// `E_{N(mu, sigma²)}[f]` by n-point Gauss–Hermite:
    /// `(1/√π)·Σ w_i·f(mu + sigma·√2·x_i)`.
    fn gh_expect(
        nodes: &[f64],
        weights: &[f64],
        mu: f64,
        sigma: f64,
        f: impl Fn(f64) -> f64,
    ) -> f64 {
        let sqrt2 = std::f64::consts::SQRT_2;
        let mut acc = 0.0;
        for (&x, &w) in nodes.iter().zip(weights.iter()) {
            acc += w * f(mu + sigma * sqrt2 * x);
        }
        acc / std::f64::consts::PI.sqrt()
    }

    /// Normal CDF via Abramowitz & Stegun 7.1.26 (|ε| ≤ 1.5e-7).
    fn normal_cdf(z: f64) -> f64 {
        let az = z.abs() / std::f64::consts::SQRT_2;
        let t = 1.0 / (1.0 + 0.327_591_1 * az);
        let poly = t
            * (0.254_829_592
                + t * (-0.284_496_736
                    + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
        let erf = 1.0 - poly * (-az * az).exp();
        if z >= 0.0 {
            0.5 * (1.0 + erf)
        } else {
            0.5 * (1.0 - erf)
        }
    }

    // The Phase-2 quadrature moved to module level as `gaussian_expectation`
    // (pub — the Phase-5 probes in katgpt-attn consume it); aliased back to
    // the historical name at these call sites.
    use super::gaussian_expectation as simpson_expect;

    #[test]
    fn gauss_hermite_moments_are_exact() {
        // A broken eigensolver cannot fake polynomial exactness: E[x] and
        // E[x²] under N(m, a²) must equal m and m² + a² to machine precision
        // (GH is exact for polynomials of degree ≤ 2n−1; n = 40).
        let gh = gauss_hermite(40);
        for &(m, a) in &[(1.0f64, 1.0f64), (-13.8, 3.85), (0.0, 5.0)] {
            let e1 = gh_expect(&gh.0, &gh.1, m, a, |x| x);
            let e2 = gh_expect(&gh.0, &gh.1, m, a, |x| x * x);
            assert!((e1 - m).abs() < 1e-10, "E[x]: {e1} vs {m}");
            assert!((e2 - (m * m + a * a)).abs() < 1e-8, "E[x²]: {e2}");
        }
    }

    /// The schedule's defining Lemma (paper Thm 1's Gaussian construction):
    /// `E[e^{L_t}] = α/(t/τ+1)` and `E[L_t·e^{L_t}] = β/(t/τ+1)` with
    /// α = β = e^0.5 — the two equations the (a, m) schedule SOLVES.
    #[test]
    fn schedule_satisfies_the_lemma() {
        let gh = gauss_hermite(40);
        let tau = 10.0f64;
        // α and β from the pin, derived independently of the LUT:
        // log α = β/α − a₀²/2 = 1 − ½  and  β/α = 1 (from m₀ = 0).
        let alpha = (1.0f64 - 0.5).exp();
        let beta = alpha;
        for &t in &[0.0f64, 1.0, 10.0, 100.0, 1000.0, 10_000.0] {
            let (a, m) = closed_form(t, tau);
            let mass = gh_expect(&gh.0, &gh.1, m, a, |x| x.exp());
            let target = alpha / (t / tau + 1.0);
            assert!(
                (mass - target).abs() <= 1e-8 * target.abs().max(1.0),
                "E[e^L] at t={t}: {mass} vs α/(t/τ+1) = {target}"
            );
            let lmass = gh_expect(&gh.0, &gh.1, m, a, |x| x * x.exp());
            let ltarget = beta / (t / tau + 1.0);
            assert!(
                (lmass - ltarget).abs() <= 1e-7 * ltarget.abs().max(1.0),
                "E[L·e^L] at t={t}: {lmass} vs β/(t/τ+1) = {ltarget}"
            );
        }
    }

    /// Phase 2's sigmoid-arm mass assertion (Plan 622): measured
    /// `E[σ(L_t)]·(t/τ+1)/α` equals `E[σ(−L′_t)]`
    /// (`L′ ~ N(1, a_t²)`) within 1e-6 — the tilt-transfer identity, both
    /// sides by accurate quadrature (adaptive Simpson; an MC sample could
    /// never reach 1e-6, and GH's near-zero node sparsity misses the σ
    /// transition at large a). Boundary bonus: at t = 0, `E[σ(L_0)] = ½`
    /// exactly by the symmetry of σ about 0.
    #[test]
    fn sigmoid_tilt_transfer_identity() {
        let tau = 10.0f64;
        let alpha = (1.0f64 - 0.5).exp();
        for &t in &[0.0f64, 1.0, 10.0, 100.0, 1000.0, 10_000.0] {
            let (a, m) = closed_form(t, tau);
            let lhs = simpson_expect(m, a, |x| 1.0 / (1.0 + (-x).exp()));
            let inner = simpson_expect(1.0, a, |x| 1.0 / (1.0 + x.exp()));
            let rhs = (alpha / (t / tau + 1.0)) * inner;
            assert!(
                (lhs - rhs).abs() <= 1e-6,
                "tilt transfer at t={t}: lhs {lhs} vs rhs {rhs}"
            );
        }
        let (a0, m0) = closed_form(0.0, tau);
        let at_boundary = simpson_expect(m0, a0, |x| 1.0 / (1.0 + (-x).exp()));
        assert!(
            (at_boundary - 0.5).abs() < 1e-12,
            "E[σ(L_0)] must be exactly ½ by symmetry, got {at_boundary}"
        );
    }

    /// The G1 tilt law: `E[σ(−L′_t)]` equals the closed form `Φ(−1/a_t)`
    /// within `C·a_t⁻³`, **C fixed once** here from the measured sup (the
    /// correction is the logistic-vs-step smoothing — odd about the
    /// transition, so its leading term is the density slope there, order
    /// a⁻³). Never the one-sided `1/(a√2π)` estimate, whose leading term
    /// CANCELS inside the exact quantity (Research 610). Φ is A&S 7.1.26
    /// (1.5e-7), far under the C·a⁻³ budget at every grid point; quadrature
    /// is adaptive Simpson (GH misreads the a ≳ 5 transition entirely).
    #[test]
    fn tilt_matches_phi_minus_one_over_a_within_c_a_cubed() {
        // Fixed once from the measured sup: diff·a³ rises monotonically
        // 0.145 (a=1) → 0.83 (a=6) → 0.63 (a=10) → 0.68 (a=100) → 0.756
        // (a=200), extrapolated asymptote ≈ 0.78 (residual a⁻⁵ decay) —
        // C = 1.0 pins the law with ≈28% headroom at the asymptote and
        // ≥30% at every grid point. An implementation of the correction via
        // the one-sided 1/(a√2π) estimate (0.399 at a=1 vs the true 0.145)
        // fails this at multiple points — that is the discrimination the
        // law is for.
        const C: f64 = 1.0;
        for &a in &[1.0f64, 1.5, 2.0, 3.0, 4.44, 6.0, 10.0, 30.0, 100.0, 200.0] {
            let inner = simpson_expect(1.0, a, |x| 1.0 / (1.0 + x.exp()));
            let phi = normal_cdf(-1.0 / a);
            let diff = (inner - phi).abs();
            assert!(
                diff <= C * a.powi(-3),
                "|E[σ(−L′)] − Φ(−1/a)| = {diff} at a={a}, budget {}",
                C * a.powi(-3)
            );
        }
    }

    // ── Phase 3: per-head logit calibration ─────────────────────────────

    /// G3 pass-through: the calibrated apply at `head_scale = 1.0` is
    /// BYTE-IDENTICAL to the uncalibrated apply (the fusion multiplies by
    /// an exact 1.0; every schedule constant and carve-out unchanged).
    #[test]
    fn calibrated_apply_at_one_is_bit_identical_to_uncalibrated() {
        let lut = ScaleInvariantLut::build(10.0, 256);
        let scores = gen_scores(256);
        let mut a = scores.clone();
        let mut b = scores.clone();
        lut.apply_inplace(&mut a, 255, 0, 1.0);
        lut.apply_inplace_calibrated(&mut b, 255, 0, 1.0, 1.0);
        assert_eq!(a, b, "head_scale = 1.0 must be byte-identical (G3)");
        // And through the mid-row entry too.
        let mut c = scores.clone();
        lut.apply_inplace(&mut c[16..], 255, 16, 1.0);
        let mut d = scores.clone();
        lut.apply_inplace_calibrated(&mut d[16..], 255, 16, 1.0, 1.0);
        assert_eq!(c, d);
    }

    /// G1: the calibrated apply equals the same-shape f32 reference —
    /// per-element `a·(s·inv_head) + m·inv_scale` with the LUT's own pair()
    /// constants. Bit-exact is the pin's point: the reference replicates
    /// the expression shape (LLVM contracts mul+add into FMA inside one
    /// expression; a two-step divide-then-affine reference rounds `s·inv`
    /// through memory and would differ by 1 ulp — that is a property of
    /// the contraction, not of the loop, and the shape-matched reference
    /// is what pins the loop's INDEXING: causal segment, sink carve-out,
    /// future-key skip).
    #[test]
    fn calibrated_apply_matches_the_same_shape_reference_bit_exact() {
        let lut = ScaleInvariantLut::build(10.0, 256);
        let scores = gen_scores(256);
        let sigma = 1.7;
        let mut fused = scores.clone();
        lut.apply_inplace_calibrated(&mut fused, 200, 0, 1.0, sigma);
        let inv_head = 1.0 / sigma;
        for (j, orig) in scores.iter().enumerate() {
            if j == 0 || j > 200 {
                continue; // the sink and the future keys: untouched
            }
            let t = 200 - j;
            let (a, m) = lut.pair(t);
            let expected = a * (orig * inv_head) + m;
            assert_eq!(fused[j], expected, "j={j} t={t}");
        }
    }

    /// The calibration composes the law it exists for: after the divide,
    /// the head's logit population is ~unit-variance, so the schedule sees
    /// what the paper's QK-norm'd models see. Measured, not asserted
    /// blindly: a σ=4 population divided by its own σ reads ≈1 within
    /// sampling noise (and the fused apply reproduces the composed form).
    #[test]
    fn calibration_normalizes_the_population_sigma() {
        let lut = ScaleInvariantLut::build(10.0, 512);
        // σ ≈ 4·√2 ≈ 5.66 content logits (the std of A·sin is A/√2), sink
        // at +16 (the sink must NOT inflate σ_h).
        let mut row: Vec<f32> = (0..256).map(|i| 8.0 * ((i as f32) * 0.113).sin()).collect();
        row[0] = 16.0;
        let content = &row[1..];
        let mean = content.iter().sum::<f32>() / content.len() as f32;
        let var = content
            .iter()
            .map(|x| {
                let d = x - mean;
                d * d
            })
            .sum::<f32>()
            / (content.len() - 1) as f32;
        let sigma = var.sqrt();
        let mut calibrated = row.clone();
        lut.apply_inplace_calibrated(&mut calibrated, 255, 0, 1.0, sigma);
        // The schedule is affine per distance: the calibrated value at
        // DISTANCE t (key j = 255 − t) equals a_t·(s_j/σ) + m_t. Spot-verify
        // through the LUT's own constants.
        let t = 100;
        let j = 255 - t;
        let (a, m) = lut.pair(t);
        let s = row[j];
        let expected = a * (s / sigma) + m;
        assert!((calibrated[j] - expected).abs() <= 1.0e-6, "fused form at t={t}");
    }

    #[test]
    fn corrupt_head_scale_fails_loud() {
        let lut = ScaleInvariantLut::build(10.0, 64);
        let mut s = gen_scores(32);
        for bad in [0.0f32, -1.0, f32::NAN, f32::INFINITY] {
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                lut.apply_inplace_calibrated(&mut s, 31, 0, 1.0, bad);
            }));
            assert!(r.is_err(), "head_scale {bad} must refuse loud");
        }
    }

    /// The Welford accumulator: exact vs the direct two-pass std on a fixed
    /// population, per-head isolation, and the sink-drop contract.
    #[test]
    fn spread_accumulator_matches_direct_std_and_isolates_heads() {
        let mut acc = SpreadAccumulator::new(2);
        let h0: Vec<f32> = (0..64).map(|i| 3.0 * ((i as f32) * 0.21).cos()).collect();
        let h1: Vec<f32> = (0..64).map(|i| (i as f32) * 0.05 - 2.0).collect();
        // Feed h0 in two chunks (the batching contract), h1 once.
        acc.observe_head(0, &h0[..32]);
        acc.observe_head(0, &h0[32..]);
        acc.observe_head(1, &h1);
        // Direct two-pass std per head.
        for (h, data) in [(0usize, &h0), (1, &h1)] {
            let n = data.len() as f64;
            let mean = data.iter().map(|&x| f64::from(x)).sum::<f64>() / n;
            let var = data
                .iter()
                .map(|&x| {
                    let d = f64::from(x) - mean;
                    d * d
                })
                .sum::<f64>()
                / (n - 1.0);
            let got = acc.std_estimate(h).expect("n ≥ 2");
            assert!((got - var.sqrt()).abs() < 1e-9, "head {h}: {got} vs {}", var.sqrt());
        }
        assert_eq!(acc.heads(), 2);
        // Below 2 observations: None, never a fake σ (and never NaN).
        let mut one = SpreadAccumulator::new(1);
        one.observe_head(0, &[1.0]);
        assert!(one.std_estimate(0).is_none());
        assert!(one.std_estimate(7).is_none(), "unobserved head: None");
        // A DEGENERATE population (all-equal) reads σ = 0 — Some(0.0), not
        // None: the measurement succeeded; HeadCalibration::build refuses it.
        let mut flat = SpreadAccumulator::new(1);
        flat.observe_head(0, &[2.5; 8]);
        assert_eq!(flat.std_estimate(0), Some(0.0));
    }

    /// The sidecar lifecycle: build → freeze → load round-trips the exact
    /// scales; and every corruption direction fails loud with a named cause.
    #[test]
    fn head_calibration_sidecar_roundtrips_and_refuses_corruption() {
        let cal = HeadCalibration::build(vec![1.0, 2.5, 0.317]).expect("valid scales");
        assert_eq!(cal.heads(), 3);
        assert_eq!(cal.scale_for(1), 2.5);
        let bytes = cal.to_sidecar();
        let loaded = HeadCalibration::from_sidecar(&bytes).expect("round trip");
        assert_eq!(loaded, cal);
        assert_eq!(loaded.digest(), cal.digest());
        // validate_heads: the checkpoint-binding check.
        assert!(loaded.validate_heads(3).is_ok());
        assert!(loaded.validate_heads(4).is_err());

        // Truncation.
        assert!(HeadCalibration::from_sidecar(&bytes[..bytes.len() - 8]).is_err());
        // Digest corruption (flip one scale byte — the digest no longer
        // matches).
        let mut corrupted = bytes.clone();
        let scale_region = 13;
        corrupted[scale_region] ^= 0x80;
        let err = HeadCalibration::from_sidecar(&corrupted).unwrap_err();
        assert!(err.contains("digest"), "the failure names the digest: {err}");
        // Bad magic.
        let mut bad_magic = bytes.clone();
        bad_magic[0] = b'X';
        assert!(HeadCalibration::from_sidecar(&bad_magic)
            .unwrap_err()
            .contains("magic"));
        // Bad version.
        let mut bad_version = bytes.clone();
        bad_version[5] = 9;
        assert!(HeadCalibration::from_sidecar(&bad_version)
            .unwrap_err()
            .contains("version"));
        // Build-side validity: zero / negative / NaN / empty all refuse.
        assert!(HeadCalibration::build(vec![]).is_err());
        assert!(HeadCalibration::build(vec![0.0]).is_err());
        assert!(HeadCalibration::build(vec![-1.0]).is_err());
        assert!(HeadCalibration::build(vec![f32::NAN]).is_err());
        assert!(HeadCalibration::build(vec![f32::INFINITY]).is_err());
        // A scale flipped to zero IN the bytes passes nothing: the digest
        // re-computation over the decoded scales catches the mutation first
        // (the digest covers the scale bytes), so the decode-side validity
        // re-check is belt-and-braces for a foreign builder.
        let mut zeroed = bytes.clone();
        let last_scale = 13 + 2 * 4;
        zeroed[last_scale..last_scale + 4].copy_from_slice(&0.0f32.to_le_bytes());
        let err = HeadCalibration::from_sidecar(&zeroed).unwrap_err();
        assert!(err.contains("digest"), "digest covers the scale bytes: {err}");
    }

    /// The full pipeline (the Phase-3 contract end to end): synthetic
    /// per-head logits with known shape → SpreadAccumulator →
    /// HeadCalibration → the calibrated apply; the served head's inverted
    /// pre-schedule population reads ~unit-variance (measured through the
    /// schedule's own a_t·s + m_t inversion). The σ̂ assertion compares
    /// against the DIRECT two-pass std of the measured population — the
    /// accumulator must be an unbiased estimator of what it was fed, and
    /// the fed population's own std is the reference (A·sin has std
    /// A/√2, not A — the planted amplitude is not the σ).
    #[test]
    fn pipeline_measure_freeze_apply_reads_unit_variance() {
        // Two heads, different amplitudes; the sink rides head 0 (excluded
        // from the measurement per the contract).
        let amp = [2.0f32, 0.5];
        let mut acc = SpreadAccumulator::new(2);
        let rows: Vec<Vec<f32>> = (0..2)
            .map(|h| {
                let mut r: Vec<f32> = (0..512)
                    .map(|i| amp[h] * ((i as f32) * 0.091 + h as f32).sin())
                    .collect();
                r[0] = 20.0 * (h as f32 + 1.0); // the sink, big on both
                r
            })
            .collect();
        // Direct content std per head (the sink excluded) — the reference.
        let direct: Vec<f64> = rows
            .iter()
            .map(|r| {
                let c = &r[1..];
                let n = c.len() as f64;
                let mean = c.iter().map(|&x| f64::from(x)).sum::<f64>() / n;
                (c
                    .iter()
                    .map(|&x| {
                        let d = f64::from(x) - mean;
                        d * d
                    })
                    .sum::<f64>()
                    / (n - 1.0))
                    .sqrt()
            })
            .collect();
        for (h, r) in rows.iter().enumerate() {
            acc.observe_head(h, &r[1..]); // the sink-drop contract
        }
        let scales: Vec<f32> = (0..2)
            .map(|h| acc.std_estimate(h).expect("n ≥ 2") as f32)
            .collect();
        // The Welford estimate matches the direct std (same population;
        // the band is one f32-ulp class — the estimate round-trips f64 →
        // f32 and back).
        for (h, s) in scales.iter().enumerate() {
            assert!(
                (f64::from(*s) - direct[h]).abs() < 1.0e-6,
                "head {h}: σ̂ {s} vs direct {}",
                direct[h]
            );
        }
        let cal = HeadCalibration::build(scales).expect("estimates valid");
        let lut = ScaleInvariantLut::build(10.0, 512);
        // Post-calibration, the schedule's input population is ~unit-σ:
        // invert the affine on a mid-distance key and measure the residual
        // spread against unity.
        for (h, r) in rows.iter().enumerate() {
            let mut served = r.clone();
            lut.apply_inplace_calibrated(&mut served, 511, 0, 1.0, cal.scale_for(h));
            // Invert PER KEY: s_pre[j] = (served[j] − m_t)/a_t with t = 511 − j
            // (the affine constants are per-distance).
            let pre: Vec<f64> = served[1..]
                .iter()
                .enumerate()
                .map(|(jj, &x)| {
                    let j = jj + 1;
                    let t = 511 - j;
                    let (a, m) = lut.pair(t);
                    f64::from((x - m) / a)
                })
                .collect();
            let n = pre.len() as f64;
            let mean = pre.iter().sum::<f64>() / n;
            let sd = (pre
                .iter()
                .map(|x| {
                    let d = x - mean;
                    d * d
                })
                .sum::<f64>()
                / (n - 1.0))
                .sqrt();
            let rel = (sd - 1.0).abs();
            assert!(rel < 0.05, "head {h}: post-calibration σ {sd} not ~unit");
        }
    }
}
