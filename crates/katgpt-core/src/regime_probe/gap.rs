//! T2/M2 — two-sample entropy-gap detector (Issue 740, Research 541 M2).
//!
//! Compares the per-position conditional-entropy distribution over a
//! **reference corpus** (training data) against the distribution over
//! **generated sequences** (paper Figs 3–4: memorization ⇒ near-zero entropy
//! on training data while generated/unseen data carries finite entropy; the
//! gap **closes exactly at the memorization→generalization transition**).
//!
//! Statistic: the mean gap first (KS optional — skipped; no new dep, and the
//! mean gap carries the sign the detector needs). Also reports a
//! standardized (Cohen's-d-style) gap so callers can threshold scale-free.
//!
//! Bit-determinism: all statistics accumulate in f64 and are stored as f32;
//! identical inputs give bit-identical outputs on the same binary. The
//! report carries a BLAKE3 artifact over a canonical little-endian encoding
//! (magic + counts + stats + every sample's f32 bits), so an audit can
//! re-verify the exact measurement that produced a verdict.

/// Canonical-encoding magic: "katgpt regime probe — entropy gap".
pub const GAP_MAGIC: &[u8; 4] = b"KRPG";

/// Encoding version (bumped only on a wire-incompatible encoding change).
pub const GAP_ENCODING_VERSION: u32 = 1;

/// Two-sample entropy-gap report with a BLAKE3-committed artifact.
///
/// Sample vectors are retained so the artifact is self-contained (an auditor
/// can re-hash without the original measurement session).
#[derive(Clone, Debug, Default)]
pub struct EntropyGapReport {
    /// Per-position entropies (nats) over the reference corpus.
    pub ref_entropies: Vec<f32>,
    /// Per-position entropies (nats) over the generated sequences.
    pub gen_entropies: Vec<f32>,
    /// Mean reference-corpus entropy.
    pub mean_ref: f32,
    /// Mean generated-sequence entropy.
    pub mean_gen: f32,
    /// `mean_gen − mean_ref`. **Positive ⇒ generated entropy exceeds the
    /// reference ⇒ memorization signature**; ≈ 0 ⇒ the distributions have
    /// converged (the transition / generalized regime, paper Fig 4).
    pub mean_gap: f32,
    /// `mean_gap / pooled_std` (Cohen's d over the two samples). `0.0` when
    /// the pooled std is 0 or either sample has fewer than 2 entries —
    /// never a fabricated effect size.
    pub standardized_gap: f32,
    /// BLAKE3 over the canonical encoding (see [`Self::write_bytes_into`]).
    pub artifact: [u8; 32],
}

impl PartialEq for EntropyGapReport {
    /// Content equality over the measured fields only.
    fn eq(&self, other: &Self) -> bool {
        self.ref_entropies == other.ref_entropies
            && self.gen_entropies == other.gen_entropies
            && self.mean_ref == other.mean_ref
            && self.mean_gen == other.mean_gen
            && self.mean_gap == other.mean_gap
            && self.standardized_gap == other.standardized_gap
            && self.artifact == other.artifact
    }
}

impl EntropyGapReport {
    /// Canonical byte encoding: `GAP_MAGIC | u32 version | u64 n_ref |
    /// u64 n_gen | f32-LE stats(4) | f32-LE bits of ref samples | f32-LE
    /// bits of gen samples`. Little-endian is pinned so the artifact is
    /// cross-platform.
    ///
    /// Writes into `out` (clearing it first) — reuses capacity across calls,
    /// so a warmed buffer makes this zero-allocation (G4).
    pub fn write_bytes_into(&self, out: &mut Vec<u8>) {
        out.clear();
        out.extend_from_slice(GAP_MAGIC);
        out.extend_from_slice(&GAP_ENCODING_VERSION.to_le_bytes());
        out.extend_from_slice(&(self.ref_entropies.len() as u64).to_le_bytes());
        out.extend_from_slice(&(self.gen_entropies.len() as u64).to_le_bytes());
        out.extend_from_slice(&self.mean_ref.to_le_bytes());
        out.extend_from_slice(&self.mean_gen.to_le_bytes());
        out.extend_from_slice(&self.mean_gap.to_le_bytes());
        out.extend_from_slice(&self.standardized_gap.to_le_bytes());
        for s in &self.ref_entropies {
            out.extend_from_slice(&s.to_le_bytes());
        }
        for s in &self.gen_entropies {
            out.extend_from_slice(&s.to_le_bytes());
        }
    }

    /// Convenience allocating wrapper around [`Self::write_bytes_into`]
    /// (offline artifact path, NOT hot — prefer the `_into` form in loops).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        self.write_bytes_into(&mut buf);
        buf
    }
}

/// Compute the two-sample entropy-gap statistic into a caller-owned report
/// (buffers reused across calls — zero allocation once warmed, G4).
///
/// Bit-deterministic: f64 accumulation, fixed summation order, f32 storage.
pub fn entropy_gap_into(ref_entropies: &[f32], gen_entropies: &[f32], report: &mut EntropyGapReport) {
    report.ref_entropies.clear();
    report.ref_entropies.extend_from_slice(ref_entropies);
    report.gen_entropies.clear();
    report.gen_entropies.extend_from_slice(gen_entropies);

    let mean_ref = mean_f64(ref_entropies);
    let mean_gen = mean_f64(gen_entropies);
    report.mean_ref = mean_ref as f32;
    report.mean_gen = mean_gen as f32;
    report.mean_gap = (mean_gen - mean_ref) as f32;

    report.standardized_gap = match pooled_std(ref_entropies, mean_ref, gen_entropies, mean_gen) {
        Some(s) if s > 0.0 => ((mean_gen - mean_ref) / s) as f32,
        _ => 0.0,
    };

    // Hash the canonical encoding. The digest is computed by feeding the
    // SAME pieces, in the SAME order, that `write_bytes_into` concatenates —
    // identical to hashing the flat encoding, with no byte buffer at all
    // (zero allocation, G4).
    let mut h = blake3::Hasher::new();
    h.update(GAP_MAGIC);
    h.update(&GAP_ENCODING_VERSION.to_le_bytes());
    h.update(&(report.ref_entropies.len() as u64).to_le_bytes());
    h.update(&(report.gen_entropies.len() as u64).to_le_bytes());
    h.update(&report.mean_ref.to_le_bytes());
    h.update(&report.mean_gen.to_le_bytes());
    h.update(&report.mean_gap.to_le_bytes());
    h.update(&report.standardized_gap.to_le_bytes());
    for s in &report.ref_entropies {
        h.update(&s.to_le_bytes());
    }
    for s in &report.gen_entropies {
        h.update(&s.to_le_bytes());
    }
    report.artifact = *h.finalize().as_bytes();
}

/// Mean of a slice in f64 (0.0 for empty).
fn mean_f64(xs: &[f32]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().map(|&x| x as f64).sum::<f64>() / xs.len() as f64
}

/// Pooled sample std (f64 internal, f64 out). `None` when either sample has
/// fewer than 2 entries (undefined).
fn pooled_std(a: &[f32], mean_a: f64, b: &[f32], mean_b: f64) -> Option<f64> {
    if a.len() < 2 || b.len() < 2 {
        return None;
    }
    let ss_a: f64 = a.iter().map(|&x| { let d = x as f64 - mean_a; d * d }).sum();
    let ss_b: f64 = b.iter().map(|&x| { let d = x as f64 - mean_b; d * d }).sum();
    let n = (a.len() + b.len() - 2) as f64;
    Some(((ss_a + ss_b) / n).sqrt())
}

/// Convenience allocating wrapper around [`entropy_gap_into`].
pub fn entropy_gap(ref_entropies: &[f32], gen_entropies: &[f32]) -> EntropyGapReport {
    let mut report = EntropyGapReport {
        ref_entropies: Vec::with_capacity(ref_entropies.len()),
        gen_entropies: Vec::with_capacity(gen_entropies.len()),
        ..EntropyGapReport::default()
    };
    entropy_gap_into(ref_entropies, gen_entropies, &mut report);
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regime_probe::entropy::conditional_entropy_nats;

    #[test]
    fn memorization_signature_positive_gap() {
        // Reference: near-one-hot logits (memorized) → ~0 entropy.
        // Generated: uniform logits → ln V entropy.
        let mut ref_logits = vec![0.0f32; 8];
        ref_logits[2] = 40.0;
        let gen_logits = vec![0.0f32; 8];
        let ref_ents: Vec<f32> = (0..16).map(|_| conditional_entropy_nats(&ref_logits)).collect();
        let gen_ents: Vec<f32> = (0..16).map(|_| conditional_entropy_nats(&gen_logits)).collect();
        let r = entropy_gap(&ref_ents, &gen_ents);
        assert!(r.mean_gap > 1.0, "memorization gap must be large: {}", r.mean_gap);
        assert!(r.mean_ref < 0.1);
        assert!((r.mean_gen - 8.0f32.ln()).abs() < 1e-4);
    }

    #[test]
    fn converged_signature_zero_gap() {
        let ents: Vec<f32> = (0..32).map(|i| (i % 5) as f32 * 0.1).collect();
        let r = entropy_gap(&ents, &ents);
        assert_eq!(r.mean_gap, 0.0);
        assert_eq!(r.standardized_gap, 0.0);
    }

    #[test]
    fn determinism_bit_identical_twice() {
        let a: Vec<f32> = (0..64).map(|i| (i % 7) as f32).collect();
        let b: Vec<f32> = (0..64).map(|i| ((i * 3) % 11) as f32 * 0.25).collect();
        let r1 = entropy_gap(&a, &b);
        let r2 = entropy_gap(&a, &b);
        assert_eq!(r1.artifact, r2.artifact, "same inputs must hash identically");
        assert_eq!(r1, r2);
        // Different inputs → different artifact (sanity, not avalanche).
        let c: Vec<f32> = b.iter().map(|x| x + 1.0).collect();
        let r3 = entropy_gap(&a, &c);
        assert_ne!(r1.artifact, r3.artifact);
    }

    #[test]
    fn canonical_encoding_is_stable() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![0.5f32; 4];
        let r = entropy_gap(&a, &b);
        let bytes = r.to_bytes();
        assert_eq!(&bytes[..4], GAP_MAGIC);
        assert_eq!(bytes.len(), 4 + 4 + 8 + 8 + 16 + 4 * (3 + 4));
    }

    #[test]
    fn tiny_samples_standardized_gap_is_zero() {
        let r = entropy_gap(&[1.0], &[2.0]);
        assert_eq!(r.standardized_gap, 0.0, "no pooled std under 2 samples — never fabricate");
        assert_eq!(r.mean_gap, 1.0);
    }
}
