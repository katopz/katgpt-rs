//! Integration tests for the Alien Sampler — exercises the full
//! sampler + MedianTopM stack end-to-end.
//!
//! Per-module unit tests live inline in each submodule; this file holds the
//! cross-module integration tests that verify the pieces compose correctly
//! (e.g. AlienSampler with MedianTopMAvailability as the availability scorer).
//!
//! Reference: Plan 311 (T1.7).

use super::median_top_m::MedianTopMAvailability;
use super::sampler::AlienSampler;
use super::traits::{AvailabilityScorer, CoherenceScorer};
use super::types::{AlienConfig, AlienSamplerError, ScoredCandidate};

// ── Reference scorers ────────────────────────────────────────────────────

/// Coherence = dot product of `atoms` against a fixed "personality direction".
/// Mirrors the paper's per-author coherence Guide: candidates that align with
/// the personality are scored higher.
struct DotCoherence {
    direction: Vec<f32>,
}

impl CoherenceScorer<f32> for DotCoherence {
    #[inline]
    fn coherence(&self, atoms: &[f32]) -> f32 {
        let mut s = 0.0_f32;
        for (a, b) in atoms.iter().zip(self.direction.iter()) {
            s += a * b;
        }
        s
    }
}

// ── T1.7 required tests ──────────────────────────────────────────────────

/// T1.7: `rank_empty_returns_empty`
#[test]
fn rank_empty_returns_empty() {
    let coh = DotCoherence { direction: vec![1.0, 0.0] };
    let avail = MedianTopMAvailability::new(vec![], 10);
    let sampler = AlienSampler::new(coh, avail, AlienConfig::paper_default());
    let sc: &mut [f32] = &mut [];
    let sa: &mut [f32] = &mut [];
    let out = sampler.rank(&[], sc, sa).unwrap();
    assert!(out.is_empty());
}

/// T1.7: `rank_single_returns_one`
#[test]
fn rank_single_returns_one() {
    let coh = DotCoherence { direction: vec![1.0, 0.0] };
    let avail = MedianTopMAvailability::new(vec![vec![1.0, 0.0]], 10);
    let sampler = AlienSampler::new(coh, avail, AlienConfig::paper_default());
    let candidates = vec![vec![0.5, 0.5]];
    let mut sc = [0.0];
    let mut sa = [0.0];
    let out = sampler.rank(&candidates, &mut sc, &mut sa).unwrap();
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].idx, 0);
}

/// T1.7: `beta_zero_is_coherence_only` — Fβ=0 ranking equals coherence-only ranking.
#[test]
fn beta_zero_is_coherence_only() {
    let direction = vec![1.0, 0.0, 0.0];
    // Bank designed so availability ordering differs from coherence ordering.
    // Bank items:
    //   [0, 1, 0]  — orthogonal to direction; cosine with [0.5, 0.5, 0] is 0.707
    //   [1, 0, 0]  — aligned with direction; cosine with [0.5, 0.5, 0] is 0.707
    //   [0, 0, 1]  — orthogonal to both; cosine with [0.5, 0.5, 0] is 0
    let bank = vec![
        vec![0.0, 1.0, 0.0],
        vec![1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];
    let coh = DotCoherence { direction };
    let avail = MedianTopMAvailability::new(bank, 1);
    let sampler = AlienSampler::new(coh, avail, AlienConfig::coherence_only());

    // Candidates: A is high-coherence, B is low-coherence, C is mid.
    let candidates = vec![
        vec![2.0, 0.0, 0.0],   // coh=2.0 (highest)
        vec![0.0, 2.0, 0.0],   // coh=0.0 (lowest)
        vec![1.0, 0.0, 0.0],   // coh=1.0 (middle)
    ];
    let mut sc = [0.0; 3];
    let mut sa = [0.0; 3];
    let out = sampler.rank(&candidates, &mut sc, &mut sa).unwrap();

    // Coherence-only ranking: idx 0 (2.0) > idx 2 (1.0) > idx 1 (0.0).
    assert_eq!(out[0].idx, 0, "highest coherence first");
    assert_eq!(out[1].idx, 2, "mid coherence second");
    assert_eq!(out[2].idx, 1, "lowest coherence last");
}

/// T1.7: `beta_one_is_unavailability_only` — Fβ=1 ranking equals negated availability.
#[test]
fn beta_one_is_unavailability_only() {
    // β=1 → score = zU = -zA. Candidates with LOWER availability (more alien)
    // rank first.
    let direction = vec![1.0, 0.0];
    // Bank: one item aligned with x-axis. Candidates near x-axis have high
    // availability (cosine ~1); candidates near y-axis have low availability
    // (cosine ~0).
    let bank = vec![vec![1.0, 0.0]];
    let coh = DotCoherence { direction };
    let avail = MedianTopMAvailability::new(bank, 1);
    let sampler = AlienSampler::new(coh, avail, AlienConfig::availability_only());

    let candidates = vec![
        vec![1.0, 0.0],   // cosine with bank = 1.0 (high availability → low alien)
        vec![0.0, 1.0],   // cosine with bank = 0.0 (low availability → high alien)
        vec![0.6, 0.8],   // cosine with bank = 0.6 (mid)
    ];
    let mut sc = [0.0; 3];
    let mut sa = [0.0; 3];
    let out = sampler.rank(&candidates, &mut sc, &mut sa).unwrap();

    // Most alien (lowest availability) first: idx 1 (cos 0), then idx 2 (cos 0.6),
    // then idx 0 (cos 1.0).
    assert_eq!(out[0].idx, 1, "most alien (cos=0) should rank first");
    assert_eq!(out[1].idx, 2, "mid alien (cos=0.6) second");
    assert_eq!(out[2].idx, 0, "least alien (cos=1.0) last");
}

/// T1.7: `median_top_m_top1_fallback` — bank size 1 → returns that one cosine.
#[test]
fn median_top_m_top1_fallback() {
    let bank = vec![vec![1.0, 0.0, 0.0]];
    let avail = MedianTopMAvailability::new(bank, 10); // m=10 but bank has 1
    // Effective m = 1, so availability = the single cosine.
    // Candidate [1,0,0] vs bank [1,0,0]: cosine = 1.0.
    let v = avail.availability(&[1.0, 0.0, 0.0]);
    assert!((v - 1.0).abs() < 1e-6);
}

/// T1.7: `median_top_m_paper_default_m10` — m=10, bank of 50, verify median of top-10.
#[test]
fn median_top_m_paper_default_m10() {
    let candidate = vec![1.0, 0.0];
    // 50 bank items at angles θ_i = (i/50) * (π/2). cosine(candidate, item) = cos θ.
    let mut bank: Vec<Vec<f32>> = Vec::with_capacity(50);
    let mut cosines: Vec<f32> = Vec::with_capacity(50);
    for i in 0..50 {
        let theta = (i as f32 / 50.0) * (core::f32::consts::PI / 2.0);
        bank.push(vec![theta.cos(), theta.sin()]);
        cosines.push(theta.cos());
    }
    let avail = MedianTopMAvailability::new(bank, 10);
    let got = avail.availability(&candidate);
    // Expected: median of top-10 cosines.
    cosines.sort_by(|a, b| a.partial_cmp(&b).unwrap());
    let top10 = &cosines[40..];
    let expected = (top10[4] + top10[5]) * 0.5; // even-count median
    assert!((got - expected).abs() < 1e-5, "got {got}, expected {expected}");
}

/// T1.7: `z_score_handles_zero_variance` — all-equal scores → z=0, no NaN.
#[test]
fn z_score_handles_zero_variance() {
    // All candidates identical → all coherence identical, all availability
    // identical → all z = 0 → all scores = 0.
    let direction = vec![1.0, 0.0];
    let bank = vec![vec![1.0, 0.0]];
    let coh = DotCoherence { direction };
    let avail = MedianTopMAvailability::new(bank, 1);
    let sampler = AlienSampler::new(coh, avail, AlienConfig::paper_default());

    let candidates = vec![
        vec![0.5, 0.5],
        vec![0.5, 0.5],
        vec![0.5, 0.5],
    ];
    let mut sc = [0.0; 3];
    let mut sa = [0.0; 3];
    let out = sampler.rank(&candidates, &mut sc, &mut sa).unwrap();
    for s in &out {
        assert!(s.score.abs() < 1e-6, "zero-variance pool: score should be 0, got {}", s.score);
        assert!(s.score.is_finite(), "score must not be NaN/inf");
    }
}

/// T1.7: `determinism_same_seed_same_order` — run twice, identical output.
#[test]
fn determinism_same_seed_same_order() {
    let direction = vec![1.0, 0.5, 0.3];
    let bank = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.5, 0.5, 0.5],
        vec![0.3, 0.3, 0.3],
    ];
    let candidates = vec![
        vec![0.9, 0.1, 0.0],
        vec![0.1, 0.9, 0.0],
        vec![0.3, 0.3, 0.3],
        vec![0.5, 0.5, 0.5],
        vec![0.0, 0.0, 1.0],
    ];
    let mk = || {
        let coh = DotCoherence { direction: direction.clone() };
        let avail = MedianTopMAvailability::new(bank.clone(), 2);
        AlienSampler::new(coh, avail, AlienConfig::paper_default())
    };
    let s1 = mk();
    let s2 = mk();
    let mut sc1 = [0.0; 5];
    let mut sa1 = [0.0; 5];
    let mut sc2 = [0.0; 5];
    let mut sa2 = [0.0; 5];
    let out1 = s1.rank(&candidates, &mut sc1, &mut sa1).unwrap();
    let out2 = s2.rank(&candidates, &mut sc2, &mut sa2).unwrap();
    assert_eq!(out1, out2, "deterministic output");
}

// ── Phase 2 T2.1 property-style tests (fuzz, no proptest dep) ────────────

/// T2.1: `rank_is_permutation_of_indices` — output indices are a permutation
/// of `0..candidates.len()`.
#[test]
fn rank_is_permutation_of_indices() {
    let direction = vec![0.7, 0.3, 0.5, 0.1];
    let bank = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0],
        vec![0.0, 0.0, 0.0, 1.0],
        vec![0.5, 0.5, 0.5, 0.5],
    ];
    let coh = DotCoherence { direction };
    let avail = MedianTopMAvailability::new(bank, 2);
    let sampler = AlienSampler::new(coh, avail, AlienConfig::paper_default());

    // Deterministic LCG to generate candidates (mirrors salience_tri_gate bench).
    let mut rng = Lcg::new(0xCAFE_BABE);
    let n = 50;
    let dim = 4;
    let candidates: Vec<Vec<f32>> = (0..n)
        .map(|_| {
            (0..dim).map(|_| rng.next_f32() * 2.0 - 1.0).collect()
        })
        .collect();
    let mut sc = vec![0.0; n];
    let mut sa = vec![0.0; n];
    let out = sampler.rank(&candidates, &mut sc, &mut sa).unwrap();

    assert_eq!(out.len(), n);
    let mut idxs: Vec<usize> = out.iter().map(|s| s.idx).collect();
    idxs.sort_unstable();
    assert_eq!(idxs, (0..n).collect::<Vec<_>>(), "indices must be a permutation");
}

/// T2.1: `beta_monotone_in_coherence_when_avail_const` — when availability is
/// constant across the pool, decreasing β toward 0 produces more
/// coherence-driven ranking (top item is the highest-coherence one).
#[test]
fn beta_monotone_in_coherence_when_avail_const() {
    // When availability is constant, z_a = 0, so Fβ = (1−β)·zC — the ranking
    // is the same for ALL β in [0,1) (zC dominates, scaled by (1−β) which is
    // a positive monotone transform). The top item is always the
    // highest-coherence one.
    struct ConstAvail;
    impl AvailabilityScorer<f32> for ConstAvail {
        fn availability(&self, _atoms: &[f32]) -> f32 { 0.5 }
    }
    let _coh = DotCoherence { direction: vec![1.0, 0.0] };
    let candidates = vec![
        vec![0.1, 0.9],
        vec![0.9, 0.1],
        vec![0.5, 0.5],
        vec![0.3, 0.7],
    ];

    for beta in [0.0_f32, 0.1, 0.3, 0.5, 0.7, 0.9] {
        let sampler = AlienSampler::new(
            DotCoherence { direction: vec![1.0, 0.0] },
            ConstAvail,
            AlienConfig { beta, top_m: 10 },
        );
        let mut sc = [0.0; 4];
        let mut sa = [0.0; 4];
        let out = sampler.rank(&candidates, &mut sc, &mut sa).unwrap();
        // Highest coherence = idx 1 (coh = 0.9). With constant availability,
        // it must rank first for all β < 1.0.
        assert_eq!(
            out[0].idx, 1,
            "β={beta}: highest-coherence candidate should rank first"
        );
    }
}

/// T2.1: `median_top_m_invariant_to_bank_permutation` — shuffle bank → same result.
#[test]
fn median_top_m_invariant_to_bank_permutation() {
    let candidate = vec![0.7, 0.3, 0.5];
    let bank = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
        vec![0.5, 0.5, 0.5],
        vec![0.3, 0.3, 0.3],
        vec![0.9, 0.1, 0.0],
    ];
    // Permute: rotate by 2.
    let mut bank_rotated = bank.clone();
    bank_rotated.rotate_left(2);

    let sa = MedianTopMAvailability::new(bank, 3);
    let sb = MedianTopMAvailability::new(bank_rotated, 3);
    let va = sa.availability(&candidate);
    let vb = sb.availability(&candidate);
    assert!(
        (va - vb).abs() < 1e-6,
        "bank permutation changed availability: {va} vs {vb}"
    );
}

// ── T2.1 G4 latent boundary ─────────────────────────────────────────────

/// G4 static check: `rank()` returns `Vec<ScoredCandidate>` — no `Vec<f32>` in
/// the public output. The bank and per-candidate embeddings stay inside the
/// call. This is enforced by the type signature: `ScoredCandidate` is
/// `{ score: f32, idx: usize }` — no Vec, no Box, no embedding escapes.
#[test]
fn rank_output_carries_no_embedding() {
    let direction = vec![1.0, 0.0];
    let bank = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let coh = DotCoherence { direction };
    let avail = MedianTopMAvailability::new(bank, 1);
    let sampler = AlienSampler::new(coh, avail, AlienConfig::paper_default());
    let candidates = vec![vec![0.5, 0.5], vec![0.3, 0.7]];
    let mut sc = [0.0; 2];
    let mut sa = [0.0; 2];
    let out: Vec<ScoredCandidate> = sampler.rank(&candidates, &mut sc, &mut sa).unwrap();
    // ScoredCandidate is a plain { f32, usize } POD — no embedding data.
    // This is a static guarantee; the runtime check here just confirms the
    // output compiles and runs.
    for s in &out {
        let _score: f32 = s.score; // f32, not Vec<f32>
        let _idx: usize = s.idx;   // usize, not Vec<f32>
    }
    // ScoredCandidate is repr(C) + Copy — no heap indirection possible.
    fn _assert_copy<T: Copy>() {}
    _assert_copy::<ScoredCandidate>();
}

// ── Helpers ──────────────────────────────────────────────────────────────

/// Deterministic LCG mirroring the salience_tri_gate bench convention.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }
    fn next_f32(&mut self) -> f32 {
        // Divide by 2^31 (not u32::MAX) so we get [0, 1).
        (self.next() as f32) / ((1u64 << 31) as f32)
    }
}

// Suppress unused-import warning for AlienSamplerError (referenced in docs
// but not in test bodies — keeps the re-export chain auditable).
#[allow(dead_code)]
fn _type_check(_: AlienSamplerError) {}
