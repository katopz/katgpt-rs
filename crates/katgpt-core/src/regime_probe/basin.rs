//! T3/M3 — corrupt → renovate → recover basin probe (Issue 740, Research 541
//! M3; paper eq 12 protocol).
//!
//! Corrupt a fraction `ρ` of a sequence's positions (seeded, deterministic
//! choice + deterministic replacement tokens), then renovate **only the
//! corrupted positions** for `sweeps` rounds through a caller-supplied
//! frozen renovator (the clean remainder is held as evidence — that is what
//! "recovery on corrupted positions" conditions on), and report the recovery
//! rate on the corrupted positions plus the whole-sequence overlap.
//!
//! The renovator seam mirrors `ugc_schedule::UgcDenoiser` (the issue's
//! shaping requirement): `posterior_into(i, x, out)` writes
//! `P(x_i | rest)` for masked site `i` given the current state `x`. A
//! diffusion-style renovator, an autoregressive LUT, or a Hebbian memory can
//! all implement it — the probe is predictor-agnostic.
//!
//! Bit-determinism (G3): position selection is a partial Fisher–Yates over a
//! scratch order array driven by a seeded `fastrand::Rng` (the
//! `data_probe::markov` harness RNG — a katgpt-core dependency, seedable,
//! platform-stable); renovation is sequential in ascending corrupted-index
//! order with ties broken toward the lowest symbol. Same seed + same
//! renovator ⇒ bit-identical report and artifact.
//!
//! Zero allocation in steady state (G4): [`BasinScratch`] and
//! [`BasinReport`] are caller-owned and capacity-reused via
//! [`basin_probe_into`].

/// Frozen renovator seam — shaped like `ugc_schedule::UgcDenoiser` (Issue
/// 740 T3). The probe holds the renovator for the whole measurement; it is
/// the caller's frozen predictor and is never mutated.
pub trait FrozenRenovator {
    /// Sequence length `L` this renovator operates on.
    fn len(&self) -> usize;
    /// Always `false` — a renovator has a fixed positive length (clippy's
    /// `len_without_is_empty` contract; a zero-length renovator is not a
    /// constructible state).
    fn is_empty(&self) -> bool {
        false
    }
    /// Alphabet size `|A|` (symbols `0..alphabet`).
    fn alphabet(&self) -> usize;
    /// Write `P(x_i = a | revealed rest)` for site `i` given the current
    /// state `x` (length `len()`), into `out` (length `alphabet()`).
    /// Deterministic: same `x` ⇒ same `out`.
    fn posterior_into(&self, i: usize, x: &[usize], out: &mut [f32]);
}

/// Caller-owned scratch for the basin probe (capacity reused — G4).
#[derive(Clone, Debug, Default)]
pub struct BasinScratch {
    corrupted: Vec<bool>,
    order: Vec<u32>,
    post: Vec<f32>,
    state: Vec<usize>,
    artifact_buf: Vec<u8>,
}

impl BasinScratch {
    /// Pre-size the scratch for sequences of length `len` over an alphabet
    /// of size `alphabet`.
    pub fn new(len: usize, alphabet: usize) -> Self {
        Self {
            corrupted: vec![false; len],
            order: (0..len as u32).collect(),
            post: vec![0.0; alphabet.max(1)],
            state: vec![0; len],
            artifact_buf: Vec::with_capacity(64 + 8 * len),
        }
    }
}

/// Corrupt → renovate → recover report with a BLAKE3 artifact.
#[derive(Clone, Debug, Default)]
pub struct BasinReport {
    /// Corrupted fraction requested (as passed in).
    pub rho: f32,
    /// Renovation sweeps performed.
    pub sweeps: usize,
    /// Corruption RNG seed (part of the artifact identity).
    pub seed: u64,
    /// Sequence length `L`.
    pub n_positions: usize,
    /// Number of corrupted positions (`round(ρ·L)`, clamped to `[0, L]`).
    pub n_corrupted: usize,
    /// Corrupted positions restored to their original symbol.
    pub recovered: usize,
    /// `recovered / n_corrupted` (`1.0` when nothing was corrupted — the
    /// vacuous read, never a fabricated recovery).
    pub recovery_rate: f32,
    /// Fraction of ALL positions matching the original after renovation.
    pub overlap: f32,
    /// BLAKE3 over the canonical artifact encoding.
    pub artifact: [u8; 32],
    /// The original sequence (artifact is self-contained).
    pub original: Vec<usize>,
    /// The post-renovation sequence.
    pub final_state: Vec<usize>,
}

impl BasinReport {
    /// Canonical artifact bytes: `KRPB | u32 version | u64 seed | u32 L |
    /// u32 alphabet-agnostic counts + stats | original | final` — see the
    /// body for the exact layout. Little-endian throughout. `out` is cleared
    /// and capacity-reused (zero-alloc when warmed, G4).
    pub fn write_bytes_into(&self, out: &mut Vec<u8>) {
        out.clear();
        out.extend_from_slice(b"KRPB");
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&self.seed.to_le_bytes());
        out.extend_from_slice(&(self.n_positions as u32).to_le_bytes());
        out.extend_from_slice(&(self.sweeps as u32).to_le_bytes());
        out.extend_from_slice(&self.rho.to_le_bytes());
        out.extend_from_slice(&(self.n_corrupted as u32).to_le_bytes());
        out.extend_from_slice(&(self.recovered as u32).to_le_bytes());
        for s in &self.original {
            out.extend_from_slice(&((*s).min(u32::MAX as usize) as u32).to_le_bytes());
        }
        for s in &self.final_state {
            out.extend_from_slice(&((*s).min(u32::MAX as usize) as u32).to_le_bytes());
        }
    }

    /// Convenience allocating wrapper around [`Self::write_bytes_into`].
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        self.write_bytes_into(&mut buf);
        buf
    }
}

/// Run the eq-12 corrupt → renovate → recover protocol into a caller-owned
/// report + scratch (zero allocation once both are warmed, G4).
///
/// `rho` is clamped to `[0, 1]`. `sweeps == 0` measures pure corruption
/// (renovation never runs) — useful as the negative control.
pub fn basin_probe_into(
    renovator: &dyn FrozenRenovator,
    original: &[usize],
    rho: f32,
    sweeps: usize,
    seed: u64,
    scratch: &mut BasinScratch,
    report: &mut BasinReport,
) {
    let len = renovator.len();
    let alphabet = renovator.alphabet();
    debug_assert_eq!(original.len(), len, "original must match renovator length");
    debug_assert!(alphabet >= 2, "need at least 2 symbols to corrupt");

    // ── Reset caller-owned buffers (capacity reused) ────────────────────
    scratch.corrupted.clear();
    scratch.corrupted.resize(len, false);
    scratch.order.clear();
    scratch.order.extend(0..len as u32);
    scratch.post.clear();
    scratch.post.resize(alphabet, 0.0);
    scratch.state.clear();
    scratch.state.extend_from_slice(original);

    report.rho = rho;
    report.sweeps = sweeps;
    report.seed = seed;
    report.n_positions = len;
    report.original.clear();
    report.original.extend_from_slice(original);

    // ── Corrupt: round(ρ·L) distinct positions, partial Fisher–Yates ────
    let rho = rho.clamp(0.0, 1.0);
    let n_corrupted = ((rho * len as f32).round() as usize).min(len);
    report.n_corrupted = n_corrupted;
    let mut rng = fastrand::Rng::with_seed(seed);
    for i in 0..n_corrupted {
        let j = i + (rng.f32() * (len - i) as f32) as usize;
        let j = j.min(len - 1);
        scratch.order.swap(i, j);
        let site = scratch.order[i] as usize;
        // Replace with a DIFFERENT random symbol (a corruption to the same
        // symbol is not a corruption).
        let step = 1 + (rng.f32() * (alphabet - 1) as f32) as usize;
        scratch.state[site] = (original[site] + step.min(alphabet - 1)) % alphabet;
        scratch.corrupted[site] = true;
    }

    // ── Renovate: sweeps passes over corrupted sites in ascending order ──
    for _ in 0..sweeps {
        for k in 0..n_corrupted {
            let site = scratch.order[k] as usize;
            renovator.posterior_into(site, &scratch.state, &mut scratch.post);
            // Argmax renovation — deterministic, ties toward the lowest symbol.
            let mut best = 0usize;
            let mut best_p = scratch.post[0];
            for (a, &p) in scratch.post.iter().enumerate().skip(1) {
                if p > best_p {
                    best = a;
                    best_p = p;
                }
            }
            scratch.state[site] = best;
        }
    }

    // ── Score ────────────────────────────────────────────────────────────
    let mut recovered = 0usize;
    let mut matches = 0usize;
    for (site, &s) in scratch.state.iter().enumerate() {
        if s == original[site] {
            matches += 1;
            if scratch.corrupted[site] {
                recovered += 1;
            }
        }
    }
    report.recovered = recovered;
    report.recovery_rate = if n_corrupted == 0 {
        1.0
    } else {
        recovered as f32 / n_corrupted as f32
    };
    report.overlap = matches as f32 / len as f32;
    report.final_state.clear();
    report.final_state.extend_from_slice(&scratch.state);

    // ── Artifact ─────────────────────────────────────────────────────────
    // Canonical bytes go through the scratch's reusable buffer (G4 — no
    // per-call allocation); the digest equals hashing the flat encoding.
    report.write_bytes_into(&mut scratch.artifact_buf);
    let mut h = blake3::Hasher::new();
    h.update(&scratch.artifact_buf);
    report.artifact = *h.finalize().as_bytes();
}

/// Convenience wrapper: allocates its own scratch + report.
pub fn basin_probe(
    renovator: &dyn FrozenRenovator,
    original: &[usize],
    rho: f32,
    sweeps: usize,
    seed: u64,
) -> BasinReport {
    let mut scratch = BasinScratch::new(renovator.len(), renovator.alphabet());
    let mut report = BasinReport::default();
    basin_probe_into(renovator, original, rho, sweeps, seed, &mut scratch, &mut report);
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Oracle renovator: posterior one-hot at a FIXED reference answer —
    /// makes recovery fully predictable regardless of corruption.
    struct Oracle {
        answer: Vec<usize>,
    }
    impl FrozenRenovator for Oracle {
        fn len(&self) -> usize {
            self.answer.len()
        }
        fn alphabet(&self) -> usize {
            4
        }
        fn posterior_into(&self, i: usize, _x: &[usize], out: &mut [f32]) {
            out.fill(0.0);
            out[self.answer[i]] = 1.0;
        }
    }

    /// Contrary renovator: always renovates to `(original + 1) % alphabet`
    /// — recovery exactly 0 by construction (the negative control).
    struct Contrary {
        answer: Vec<usize>,
    }
    impl FrozenRenovator for Contrary {
        fn len(&self) -> usize {
            self.answer.len()
        }
        fn alphabet(&self) -> usize {
            4
        }
        fn posterior_into(&self, i: usize, _x: &[usize], out: &mut [f32]) {
            out.fill(0.0);
            out[(self.answer[i] + 1) % 4] = 1.0;
        }
    }

    fn seq(len: usize) -> Vec<usize> {
        (0..len).map(|i| (i * 3 + 1) % 4).collect()
    }

    #[test]
    fn oracle_renovator_recovers_everything() {
        let original = seq(40);
        let r = basin_probe(&Oracle { answer: original.clone() }, &original, 0.4, 2, 7);
        assert_eq!(r.n_corrupted, 16, "round(0.4·40)");
        assert_eq!(r.recovered, 16);
        assert_eq!(r.recovery_rate, 1.0);
        assert_eq!(r.overlap, 1.0);
    }

    #[test]
    fn contrary_renovator_recovers_nothing() {
        let original = seq(40);
        let r = basin_probe(&Contrary { answer: original.clone() }, &original, 0.5, 3, 7);
        assert_eq!(r.n_corrupted, 20);
        assert_eq!(r.recovered, 0);
        assert_eq!(r.recovery_rate, 0.0);
    }

    #[test]
    fn zero_rho_is_vacuous_not_fabricated() {
        let original = seq(16);
        let r = basin_probe(&Oracle { answer: original.clone() }, &original, 0.0, 2, 7);
        assert_eq!(r.n_corrupted, 0);
        assert_eq!(r.recovery_rate, 1.0);
        assert_eq!(r.overlap, 1.0);
    }

    #[test]
    fn determinism_bit_identical_twice() {
        let original = seq(64);
        let ren = Oracle { answer: original.clone() };
        let a = basin_probe(&ren, &original, 0.35, 2, 0xDEADBEEF);
        let b = basin_probe(&ren, &original, 0.35, 2, 0xDEADBEEF);
        assert_eq!(a.artifact, b.artifact);
        assert_eq!(a.final_state, b.final_state);
        // Different seed → (almost surely) different corrupted set →
        // different artifact bytes.
        let c = basin_probe(&ren, &original, 0.35, 2, 0xDEADBEEF + 1);
        assert_ne!(a.artifact, c.artifact);
    }

    #[test]
    fn zero_sweeps_is_the_pure_corruption_control() {
        let original = seq(32);
        let r = basin_probe(&Oracle { answer: original.clone() }, &original, 0.5, 0, 7);
        assert_eq!(r.recovered, 0, "no sweeps → no renovation → no recovery");
        assert!(r.overlap < 1.0);
    }
}
