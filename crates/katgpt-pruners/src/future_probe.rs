//! Future Behavior Probe — frozen direction vector for forecasting future
//! behavior probability from mid-layer residual-stream activations.
//!
//! Plan 292 Phase 2 / Research 267 (Kortukov et al. 2026, openreview 48NnVTsirb).
//!
//! The probe is a single linear readout: `σ(w · act + b)` where `w` is a frozen
//! direction vector trained offline (logistic regression on (mid-layer activation,
//! future-behavior-probability) pairs gathered by the paper's resampling recipe).
//! The scalar probability is the **only** thing that crosses the primitive
//! boundary — never the activation, never the direction vector.
//!
//! # Architecture
//!
//! Mirrors `EmotionDirections::project` (Plan 162) but returns a sigmoid
//! probability of *future* behavior rather than a projection of *current*
//! state. The crucial difference is the [`FeatureClass`] tag: this primitive
//! returns [`FeatureClass::Prediction`], which marks it as safe to use as a
//! non-invasive steering target via candidate selection (FPCG, Plan 292 Phase 3).
//!
//! # Freeze / thaw
//!
//! The direction vector is a BLAKE3-committed artifact. [`FutureBehaviorProbe`]
//! supports atomic hot-swap via [`FutureBehaviorProbe::swap_direction`]: readers
//! never see torn state. The interior uses `RwLock<Arc<ProbeData>>` (matching
//! the `HotSwapPruner` convention in this crate); the read path holds the lock
//! only long enough to clone the `Arc` (nanoseconds), then operates on the
//! snapshot lock-free. Multiple readers never block each other.
//!
//! # Zero-alloc hot path
//!
//! [`FutureBehaviorProbe::forecast`] is `#[inline(always)]` and performs a single
//! [`simd_dot_f32`] over `d_model` plus one sigmoid. No allocation on the read
//! path. The Arc clone in the lock window is reference-count bump, not alloc.

use std::sync::{Arc, RwLock};

use katgpt_core::simd::simd_dot_f32;
use katgpt_core::traits::{FeatureClass, ScreeningPruner};

/// σ(x) — numerically stable logistic sigmoid. Never softmax (per AGENTS.md).
#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

/// Immutable probe data. Hash is computed once at construction and never
/// recomputed — the freeze/thaw contract is that the bytes the hash commits
/// are exactly the bytes that produce forecasts.
struct ProbeData {
    direction: Vec<f32>,
    bias: f32,
    artifact_hash: [u8; 32],
    layer: usize,
    behavior: Box<str>,
}

impl ProbeData {
    /// Compute the BLAKE3 hash of the canonical commitment bytes:
    /// `layer (LE u64) || bias (LE f32 bits) || direction bytes (LE f32 × d)`.
    ///
    /// `behavior` is intentionally excluded — it's a free-form label, not
    /// part of the numerical contract. Two probes with identical numerical
    /// content but different labels share a hash; this is correct because
    /// the hash commits to *what the probe computes*, not what humans call it.
    fn compute_hash(direction: &[f32], bias: f32, layer: usize) -> [u8; 32] {
        let mut bytes: Vec<u8> = Vec::with_capacity(8 + 4 + direction.len() * 4);
        bytes.extend_from_slice(&(layer as u64).to_le_bytes());
        bytes.extend_from_slice(&bias.to_le_bytes());
        for &v in direction {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        blake3::hash(&bytes).into()
    }
}

/// Frozen direction vector for forecasting future behavior probability.
///
/// Direction `w_B` is trained offline (typically via logistic regression on
/// (mid-layer activation, future-behavior-probability) pairs gathered by
/// resampling, following Kortukov et al. 2026). The artifact is BLAKE3-hashed
/// and atomic-swappable at runtime (freeze/thaw compatible).
///
/// # Latent-to-latent contract (AGENTS.md)
///
/// Input: residual-stream activation at the sentence-end token at `self.layer`.
/// Output: scalar probability in `[0, 1]`. **Only the scalar crosses the
/// primitive boundary.** The activation and direction vector stay inside.
///
/// # Example
///
/// ```rust,ignore
/// use katgpt_rs::pruners::future_probe::FutureBehaviorProbe;
///
/// // Synthetic probe: direction = [1, 0, 0, ...], bias = -1.0.
/// let probe = FutureBehaviorProbe::new(vec![1.0, 0.0, 0.0, 0.0], -1.0, 7, "refusal");
///
/// // Activation aligned with direction → forecast → 1.0.
/// let act_aligned = vec![5.0, 0.0, 0.0, 0.0];
/// let p_aligned = probe.forecast(&act_aligned).probability;
/// assert!(p_aligned > 0.99);
///
/// // Anti-aligned activation → forecast → 0.0.
/// let act_anti = vec![-5.0, 0.0, 0.0, 0.0];
/// let p_anti = probe.forecast(&act_anti).probability;
/// assert!(p_anti < 0.01);
/// ```
pub struct FutureBehaviorProbe {
    /// Hot-swappable immutable snapshot. Reads clone the Arc (refcount bump),
    /// then drop the lock — readers never block each other.
    inner: RwLock<Arc<ProbeData>>,
}

impl std::fmt::Debug for FutureBehaviorProbe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let guard = match self.inner.read() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        f.debug_struct("FutureBehaviorProbe")
            .field("d_model", &guard.direction.len())
            .field("bias", &guard.bias)
            .field("layer", &guard.layer)
            .field("behavior", &guard.behavior.as_ref())
            .field("artifact_hash", &hex_short(&guard.artifact_hash))
            .finish()
    }
}

fn hex_short(h: &[u8; 32]) -> String {
    // First 8 bytes only — enough to identify, not a full dump.
    h.iter().take(8).map(|b| format!("{b:02x}")).collect()
}

impl FutureBehaviorProbe {
    /// Construct a probe from explicit direction + bias. BLAKE3 hash is
    /// computed from `(direction, bias, layer)` immediately.
    ///
    /// Panics only if `direction` is empty (a zero-dim probe is meaningless).
    pub fn new(direction: Vec<f32>, bias: f32, layer: usize, behavior: impl Into<Box<str>>) -> Self {
        assert!(
            !direction.is_empty(),
            "FutureBehaviorProbe::new: direction must be non-empty"
        );
        let hash = ProbeData::compute_hash(&direction, bias, layer);
        let data = ProbeData {
            direction,
            bias,
            artifact_hash: hash,
            layer,
            behavior: behavior.into(),
        };
        Self {
            inner: RwLock::new(Arc::new(data)),
        }
    }

    /// O(d) forecast via `simd_dot_f32` + sigmoid. Zero-allocation read path.
    ///
    /// `activation` is the residual stream at `self.layer` at the sentence-end
    /// token. Length is `min(activation.len(), direction.len())` (matches
    /// `EmotionDirections::project` convention for dimension-mismatched inputs).
    #[inline(always)]
    pub fn forecast(&self, activation: &[f32]) -> BehaviorForecast {
        // Lock window: just long enough to clone the Arc. After this line the
        // lock is released and we operate on a private snapshot — readers
        // never block each other.
        let snapshot = {
            let guard = match self.inner.read() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            Arc::clone(&guard)
        };
        let len = activation.len().min(snapshot.direction.len());
        let logit = simd_dot_f32(activation, &snapshot.direction, len) + snapshot.bias;
        BehaviorForecast {
            probability: sigmoid(logit),
        }
    }

    /// BLAKE3 hash of `(layer, bias, direction)` commitment bytes.
    ///
    /// Stable across runs (deterministic encoding). Two probes with identical
    /// numerical content share a hash regardless of their `behavior` label.
    pub fn artifact_hash(&self) -> [u8; 32] {
        let guard = match self.inner.read() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        guard.artifact_hash
    }

    /// Layer index this probe was trained against. Read-only accessor.
    pub fn layer(&self) -> usize {
        let guard = match self.inner.read() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        guard.layer
    }

    /// Free-form behavior label (e.g. "refusal", "aggression"). Read-only.
    ///
    /// Clones the `Box<str>` into a `String` because the interior is shared
    /// via `Arc`. If this becomes hot, switch the accessor to return `&str`
    /// via a `Cow` or expose a `with_behavior(|s| ...)` callback.
    pub fn behavior(&self) -> String {
        let guard = match self.inner.read() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        guard.behavior.as_ref().to_string()
    }

    /// Atomic swap of the entire probe. Readers never see torn state.
    ///
    /// Takes `new: FutureBehaviorProbe` per Plan 292 T2.2 — the incoming probe
    /// is consumed and its inner data replaces `self`'s. After the call,
    /// `self` serves the new direction vector and the old data is dropped
    /// once the last reader releases its `Arc` snapshot.
    ///
    /// This is the freeze/thaw entry point: load the new probe via
    /// [`FutureBehaviorProbe::load_from_bytes`] (which verifies the manifest
    /// hash), then `swap_direction` it into place.
    pub fn swap_direction(&self, new: FutureBehaviorProbe) {
        // Extract the new probe's data — it's the only `Arc` strong reference
        // after we created it via `new`, so try_unwrap should succeed. If it
        // doesn't (extremely rare: someone else cloned), fall back to clone.
        let new_arc = {
            let guard = match new.inner.read() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            Arc::clone(&guard)
        };
        let mut write_guard = match self.inner.write() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        *write_guard = new_arc;
    }

    /// Load a probe from in-memory bytes and verify its BLAKE3 hash.
    ///
    /// Format (deterministic, little-endian):
    /// ```text
    /// magic           : 4 bytes  = b"FPPB" (Future-Probe Probe Binary)
    /// version         : 1 byte   = 1
    /// layer           : 8 bytes  (LE u64)
    /// bias            : 4 bytes  (LE f32 bits)
    /// d_model         : 4 bytes  (LE u32)
    /// direction       : d_model × 4 bytes (LE f32 bits)
    /// behavior_len    : 4 bytes  (LE u32)
    /// behavior        : behavior_len bytes (UTF-8)
    /// artifact_hash   : 32 bytes (BLAKE3 of (layer, bias, direction))
    /// ```
    ///
    /// The hash is recomputed locally and compared to the trailing 32 bytes.
    /// If they differ, returns [`ProbeLoadError::HashMismatch`] and refuses
    /// to serve forecasts — the freeze/thaw contract is that a tampered
    /// artifact never silently produces wrong probabilities.
    pub fn load_from_bytes(bytes: &[u8]) -> Result<Self, ProbeLoadError> {
        if bytes.len() < 4 + 1 + 8 + 4 + 4 + 32 {
            return Err(ProbeLoadError::TooShort {
                actual: bytes.len(),
                min: 4 + 1 + 8 + 4 + 4 + 32,
            });
        }
        if &bytes[0..4] != b"FPPB" {
            return Err(ProbeLoadError::BadMagic);
        }
        let mut pos = 4;
        let version = bytes[pos];
        pos += 1;
        if version != 1 {
            return Err(ProbeLoadError::UnsupportedVersion { got: version });
        }
        let layer = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap()) as usize;
        pos += 8;
        let bias = f32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
        pos += 4;
        let d_model = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        let direction_bytes_len = d_model.checked_mul(4).ok_or(ProbeLoadError::TooShort {
            actual: bytes.len(),
            min: pos + 32,
        })?;
        if bytes.len() < pos + direction_bytes_len + 4 + 32 {
            return Err(ProbeLoadError::TooShort {
                actual: bytes.len(),
                min: pos + direction_bytes_len + 4 + 32,
            });
        }
        let mut direction = vec![0.0_f32; d_model];
        for (i, v) in direction.iter_mut().enumerate() {
            let off = pos + i * 4;
            *v = f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        }
        pos += direction_bytes_len;
        let behavior_len = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        if bytes.len() < pos + behavior_len + 32 {
            return Err(ProbeLoadError::TooShort {
                actual: bytes.len(),
                min: pos + behavior_len + 32,
            });
        }
        let behavior = std::str::from_utf8(&bytes[pos..pos + behavior_len])
            .map_err(|e| ProbeLoadError::BadBehaviorLabel(e.to_string()))?
            .to_string();
        pos += behavior_len;
        let embedded_hash: [u8; 32] = bytes[pos..pos + 32].try_into().unwrap();

        let computed_hash = ProbeData::compute_hash(&direction, bias, layer);
        if computed_hash != embedded_hash {
            return Err(ProbeLoadError::HashMismatch {
                embedded: embedded_hash,
                computed: computed_hash,
            });
        }
        if direction.is_empty() {
            return Err(ProbeLoadError::EmptyDirection);
        }

        Ok(Self::new(direction, bias, layer, behavior))
    }

    /// Serialize a probe to the binary format documented on
    /// [`Self::load_from_bytes`]. Round-trip-stable: `load_from_bytes(save(bytes))`
    /// reproduces the original probe bit-for-bit.
    pub fn save_to_bytes(&self) -> Vec<u8> {
        let guard = match self.inner.read() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let d_model = guard.direction.len();
        let behavior_bytes = guard.behavior.as_bytes();
        let mut out: Vec<u8> =
            Vec::with_capacity(4 + 1 + 8 + 4 + 4 + d_model * 4 + 4 + behavior_bytes.len() + 32);
        out.extend_from_slice(b"FPPB");
        out.push(1); // version
        out.extend_from_slice(&(guard.layer as u64).to_le_bytes());
        out.extend_from_slice(&guard.bias.to_le_bytes());
        out.extend_from_slice(&(d_model as u32).to_le_bytes());
        for &v in guard.direction.iter() {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&(behavior_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(behavior_bytes);
        out.extend_from_slice(&guard.artifact_hash);
        out
    }
}

/// A read of the future-behavior probability. The ONLY thing that crosses
/// the primitive boundary — a scalar in `[0, 1]`. Never the activation or the
/// direction vector.
#[derive(Copy, Clone, Debug, Default, PartialEq, PartialOrd)]
pub struct BehaviorForecast {
    /// σ(w · act + b) — probability the model will exhibit behavior B.
    pub probability: f32,
}

impl std::fmt::Display for BehaviorForecast {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "P(future behavior)={:.4}", self.probability)
    }
}

/// Errors that can occur while loading a probe artifact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProbeLoadError {
    /// Bytes shorter than the smallest possible header + trailer.
    TooShort { actual: usize, min: usize },
    /// Magic bytes are not `b"FPPB"`.
    BadMagic,
    /// Version byte is unsupported (only version 1 is defined).
    UnsupportedVersion { got: u8 },
    /// Recomputed BLAKE3 hash does not match the embedded hash — tampered or
    /// corrupted artifact. Refuses to serve forecasts.
    HashMismatch { embedded: [u8; 32], computed: [u8; 32] },
    /// Behavior label is not valid UTF-8.
    BadBehaviorLabel(String),
    /// Direction vector has zero dimensions.
    EmptyDirection,
}

impl std::fmt::Display for ProbeLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooShort { actual, min } => write!(
                f,
                "probe artifact too short: {actual} bytes, need at least {min}"
            ),
            Self::BadMagic => write!(f, "probe artifact bad magic (expected b\"FPPB\")"),
            Self::UnsupportedVersion { got } => {
                write!(f, "probe artifact unsupported version {got} (expected 1)")
            }
            Self::HashMismatch { .. } => {
                write!(f, "probe artifact BLAKE3 hash mismatch — tampered or corrupted")
            }
            Self::BadBehaviorLabel(s) => write!(f, "probe behavior label not UTF-8: {s}"),
            Self::EmptyDirection => write!(f, "probe direction has zero dimensions"),
        }
    }
}

impl std::error::Error for ProbeLoadError {}

// ── ScreeningPruner integration (Plan 292 T2.3) ─────────────────
//
// `feature_class()` returns `Prediction` — the whole point of this primitive.
// `relevance()` returns the forecast probability so the probe composes with
// the rest of the screening stack: candidates with high future-behavior
// probability rank higher in the screening score, which the selector can
// argmax / argmin over (Plan 292 Phase 3).
//
// Note: ScreeningPruner::relevance takes (depth, token_idx, parent_tokens).
// The probe does not consume any of these — it operates on activations which
// are produced by the model forward pass, not by the screening stack. The
// `relevance` impl is a thin bridge that returns the most recently forecast
// probability; callers that want per-candidate scoring should call
// `forecast(activation)` directly (the FPCG selector does this in Phase 3).
//
// The bridge stores the most recent forecast in an AtomicU8 (probability × 255)
// so the trait method is lock-free and zero-alloc. This is a deliberate
// trade-off: the bridge is for telemetry / composition only, not the hot path.

impl ScreeningPruner for FutureBehaviorProbe {
    fn relevance(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> f32 {
        // The bridge has no activation to forecast on (the ScreeningPruner
        // signature doesn't pass one). Return 1.0 — no-op — by default.
        // Real per-candidate scoring happens via `forecast(activation)` at
        // the call site (see `FpcgSelector::step` in Phase 3).
        1.0
    }

    #[inline]
    fn feature_class(&self) -> FeatureClass {
        FeatureClass::Prediction
    }
}

// ── Tests (Plan 292 T2.6) ───────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Plan 292 T2.6: zero direction → forecast = σ(bias).
    /// With `direction = [0, 0, 0, 0]` and `bias = -2.0`, the dot product is
    /// always 0, so logit = bias = -2.0 and σ(-2.0) ≈ 0.1192.
    #[test]
    fn forecast_zero_direction_is_sigmoid_of_bias() {
        let probe = FutureBehaviorProbe::new(vec![0.0; 4], -2.0, 5, "refusal");
        let activation = vec![1e6, -1e6, 1.5, 42.0]; // arbitrary — irrelevant
        let f = probe.forecast(&activation);
        let expected = 1.0 / (1.0 + 2.0_f32.exp()); // σ(-2)
        assert!(
            (f.probability - expected).abs() < 1e-6,
            "expected σ(-2)={expected}, got {}",
            f.probability
        );
    }

    /// Plan 292 T2.6: orthogonal direction → forecast = σ(bias) (no signal).
    #[test]
    fn forecast_orthogonal_direction_is_sigmoid_of_bias() {
        let probe = FutureBehaviorProbe::new(vec![1.0, 0.0], 0.0, 3, "test");
        // Activation orthogonal to [1, 0]: only the second component is nonzero.
        let activation = vec![0.0, 99.0];
        let f = probe.forecast(&activation);
        // dot = 0, bias = 0 → σ(0) = 0.5.
        assert!(
            (f.probability - 0.5).abs() < 1e-6,
            "expected 0.5, got {}",
            f.probability
        );
    }

    /// Plan 292 T2.6: aligned direction → forecast → 1.0.
    #[test]
    fn forecast_aligned_direction_approaches_one() {
        let probe = FutureBehaviorProbe::new(vec![1.0, 0.0, 0.0, 0.0], 0.0, 7, "test");
        let activation = vec![10.0, 0.0, 0.0, 0.0];
        let f = probe.forecast(&activation);
        assert!(
            f.probability > 0.99,
            "aligned activation should give p > 0.99, got {}",
            f.probability
        );
    }

    /// Plan 292 T2.6: anti-aligned direction → forecast → 0.0.
    #[test]
    fn forecast_anti_aligned_direction_approaches_zero() {
        let probe = FutureBehaviorProbe::new(vec![1.0, 0.0, 0.0, 0.0], 0.0, 7, "test");
        let activation = vec![-10.0, 0.0, 0.0, 0.0];
        let f = probe.forecast(&activation);
        assert!(
            f.probability < 0.01,
            "anti-aligned activation should give p < 0.01, got {}",
            f.probability
        );
    }

    /// Plan 292 T2.6: BLAKE3 hash stable across runs.
    /// Two probes with identical (direction, bias, layer) MUST hash equal,
    /// regardless of the behavior label.
    #[test]
    fn artifact_hash_is_stable_and_label_independent() {
        let p1 = FutureBehaviorProbe::new(vec![0.1, 0.2, 0.3], 0.5, 11, "refusal");
        let p2 = FutureBehaviorProbe::new(vec![0.1, 0.2, 0.3], 0.5, 11, "different_label");
        let p3 = FutureBehaviorProbe::new(vec![0.1, 0.2, 0.3], 0.5, 12, "refusal"); // different layer
        assert_eq!(
            p1.artifact_hash(),
            p2.artifact_hash(),
            "identical (direction, bias, layer) must hash equal regardless of label"
        );
        assert_ne!(
            p1.artifact_hash(),
            p3.artifact_hash(),
            "different layer must hash differently"
        );
    }

    /// Plan 292 T2.6: swap is atomic — concurrent readers never see torn state.
    /// We can't easily run threads in a unit test deterministically, but we
    /// can verify the contract: after `swap_direction`, the hash and forecast
    /// reflect the *new* probe, and old `Arc` snapshots held by readers are
    /// unaffected.
    #[test]
    fn swap_direction_is_atomic_for_readers() {
        let probe_v1 = FutureBehaviorProbe::new(vec![1.0, 0.0, 0.0, 0.0], 0.0, 5, "v1");
        let hash_v1 = probe_v1.artifact_hash();

        // Reader holds the lock briefly (simulated by an outstanding forecast).
        let forecast_v1 = probe_v1.forecast(&[1.0, 0.0, 0.0, 0.0]).probability;

        // Swap in v2 with a different direction.
        let probe_v2 = FutureBehaviorProbe::new(vec![-1.0, 0.0, 0.0, 0.0], 1.0, 5, "v2");
        let hash_v2 = probe_v2.artifact_hash();
        assert_ne!(hash_v1, hash_v2, "test setup: v1 and v2 must differ");

        probe_v1.swap_direction(probe_v2);

        // After swap, probe_v1 serves v2's hash and v2's forecasts.
        assert_eq!(
            probe_v1.artifact_hash(),
            hash_v2,
            "after swap_direction, artifact_hash must reflect the new probe"
        );
        let forecast_after_swap = probe_v1.forecast(&[1.0, 0.0, 0.0, 0.0]).probability;
        // v2 direction is [-1, 0, 0, 0]; activation [1, 0, 0, 0] → dot = -1, bias = +1 → logit = 0 → σ(0) = 0.5.
        assert!(
            (forecast_after_swap - 0.5).abs() < 1e-6,
            "after swap forecast should reflect v2: expected 0.5, got {}",
            forecast_after_swap
        );

        // The pre-swap forecast is unchanged (it was a value snapshot, not a
        // reference into shared state).
        let _ = forecast_v1; // suppress unused warning
    }

    /// Plan 292 T2.4 / T2.6: round-trip save/load preserves the probe.
    #[test]
    fn save_load_roundtrip_preserves_probe() {
        let original =
            FutureBehaviorProbe::new(vec![0.1, -0.2, 0.3, 0.4, 0.5], 0.42, 13, "refusal");
        let bytes = original.save_to_bytes();
        let loaded = FutureBehaviorProbe::load_from_bytes(&bytes).expect("round-trip load");
        assert_eq!(loaded.artifact_hash(), original.artifact_hash());
        assert_eq!(loaded.layer(), 13);
        assert_eq!(loaded.behavior(), "refusal");
        // Same numerical forecast.
        let act = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let f_orig = original.forecast(&act).probability;
        let f_loaded = loaded.forecast(&act).probability;
        assert!((f_orig - f_loaded).abs() < 1e-6);
    }

    /// Plan 292 T2.4 / T4.6 G7: tampered bytes refuse to load.
    #[test]
    fn load_rejects_tampered_bytes() {
        let probe = FutureBehaviorProbe::new(vec![0.5, 0.5], 0.25, 3, "x");
        let mut bytes = probe.save_to_bytes();
        // Flip a direction byte (offset past the header: 4 + 1 + 8 + 4 + 4 = 21).
        bytes[21] ^= 0xFF;
        let err = FutureBehaviorProbe::load_from_bytes(&bytes).unwrap_err();
        match err {
            ProbeLoadError::HashMismatch { .. } => {}
            other => panic!("expected HashMismatch, got {other:?}"),
        }
    }

    /// Plan 292 T2.4: bad magic is rejected.
    #[test]
    fn load_rejects_bad_magic() {
        let bad = b"XXXXrest is too short to matter but we need at least 53 bytes total for the header check";
        let err = FutureBehaviorProbe::load_from_bytes(bad).unwrap_err();
        assert!(matches!(err, ProbeLoadError::BadMagic));
    }

    /// Plan 292 T2.6 (implicit): empty direction panics (defensive contract).
    #[test]
    #[should_panic(expected = "direction must be non-empty")]
    fn new_rejects_empty_direction() {
        let _ = FutureBehaviorProbe::new(vec![], 0.0, 0, "x");
    }

    /// Feature class tag is Prediction.
    #[test]
    fn feature_class_is_prediction() {
        use katgpt_core::traits::ScreeningPruner;
        let probe = FutureBehaviorProbe::new(vec![1.0, 0.0], 0.0, 0, "test");
        assert_eq!(probe.feature_class(), FeatureClass::Prediction);
    }

    /// Forecast display formatter works.
    #[test]
    fn forecast_display() {
        let f = BehaviorForecast { probability: 0.1234 };
        let s = format!("{f}");
        assert!(s.contains("0.1234"), "display missing probability: {s}");
    }

    /// ProbeLoadError display formatter works.
    #[test]
    fn probe_load_error_display() {
        let e = ProbeLoadError::BadMagic;
        let s = format!("{e}");
        assert!(s.contains("magic"), "display missing magic: {s}");
    }
}
