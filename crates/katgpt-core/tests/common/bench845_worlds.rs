//! Shared Bench 845 W1/W2 world builders (extracted at Issue 985 T6 so the
//! soft-read second-signal cell re-runs the SAME worlds instead of a fork;
//! bench_845 consumes this module too — moved functions, byte-identical
//! behavior).
//!
//! One geometry, two error worlds (the bench_845 discipline):
//! - **W1 (OOD-blind errors)** — the model's reported score carries no OOD
//!   information; distance-style signals carry error information the score
//!   lacks.
//! - **W2 (score-only errors)** — identical geometry/scores/OOD share, but
//!   accuracy is score-driven everywhere: distance-style signals carry
//!   NOTHING, and any advantage they show is a rigged fixture.
//!
//! Also carries the shared risk–coverage METRICS (AURC + selective
//! accuracy) both benches gate on — moved here from the store-falsification
//! harness at Issue 985 so each bench includes ONE metrics home. (The
//! harness keeps private twins: `#[path]` double-includes are
//! duplicate-symbol errors.)

/// The latent width of the shared worlds (both benches).
pub const D: usize = 64;
pub const N_CLUSTERS: usize = 4;
pub const PER_CLUSTER: usize = 32;
pub const K_EXEMPLARS: usize = N_CLUSTERS * PER_CLUSTER; // 128
pub const N_TEST: usize = 4096;
/// The OOD query share — the fraction of arrivals from outside the corpus.
pub const OOD_SHARE: f32 = 0.30;
/// Cluster spread (angular jitter around the cluster center).
pub const CLUSTER_SIGMA: f32 = 0.12;
/// The sigmoid operating point on the max-similarity axis (bench 845's
/// fixed operating point, the same standing as bench 808's τ=0.75).
pub const MID: f32 = 0.45;
pub const SCALE: f32 = 12.0;

/// One decision arrival: the latent query, the model's reported confidence,
/// the OOD flag, and the drawn outcome.
pub struct Arrival {
    pub q: [f32; D],
    pub p: f32,
    pub ood: bool,
    pub correct: bool,
}

/// Deterministic `SplitMix64` (the workspace-bench idiom — no rand dep).
pub struct SplitMix64(u64);
impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    pub fn next_unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    pub fn next_sym(&mut self) -> f32 {
        self.next_unit() * 2.0 - 1.0
    }
}

pub fn unit(v: [f32; D]) -> [f32; D] {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let inv = 1.0 / n;
    v.map(|x| x * inv)
}

/// The corpus: `N_CLUSTERS` tight clusters in R^D (built once; replicates
/// vary arrivals + outcomes, not geometry). Cluster centers are biased into
/// one quadrant so clusters separate (quadrant bias gives centers ~0.2-0.4
/// mutual cosine).
pub fn build_corpus() -> Vec<[f32; D]> {
    let mut rng = SplitMix64::new(0x845_0001);
    let mut rows = Vec::with_capacity(K_EXEMPLARS);
    for c in 0..N_CLUSTERS {
        let sign = |i: usize| if (c >> (i % 3)) & 1 == 1 { 1.0 } else { -1.0 };
        let center = unit(std::array::from_fn(|i| sign(i) * (0.5 + rng.next_unit())));
        for _ in 0..PER_CLUSTER {
            let row = unit(center.map(|x| x + CLUSTER_SIGMA * rng.next_sym()));
            rows.push(row);
        }
    }
    rows
}

/// In-corpus query: a cluster member with jitter.
fn in_corpus_query(rng: &mut SplitMix64, corpus: &[[f32; D]]) -> [f32; D] {
    let idx = (rng.next_u64() as usize) % corpus.len();
    unit(corpus[idx].map(|x| x + 2.0 * CLUSTER_SIGMA * rng.next_sym()))
}

/// OOD query: a fresh random direction — near-orthogonal to everything
/// registered (E[cos] ≈ 0 at D=64).
fn ood_query(rng: &mut SplitMix64) -> [f32; D] {
    unit(std::array::from_fn(|_| rng.next_sym()))
}

/// The model's reported confidence — deliberately OOD-blind: the same
/// confident distribution everywhere (this is the failure mode under test).
fn reported_confidence(rng: &mut SplitMix64) -> f32 {
    0.5 + 0.49 * rng.next_unit()
}

/// W1 true accuracy: in-corpus follows a mildly overconfident calibration;
/// OOD collapses toward coin-flip regardless of the reported score.
pub fn w1_true_p(p: f32, ood: bool) -> f32 {
    if ood {
        0.5 + 0.05 * (p - 0.5)
    } else {
        sigmoid(1.2 * (p / (1.0 - p)).ln() - 0.2)
    }
}

/// W2 true accuracy: the score drives correctness EVERYWHERE (distance is
/// error-irrelevant by construction).
pub fn w2_true_p(p: f32, _ood: bool) -> f32 {
    sigmoid(1.2 * (p / (1.0 - p)).ln() - 0.2)
}

/// The bench's original plain sigmoid — NOT `crate::exact_sigmoid` (a
/// ≤3-ULP difference here would flip `correct` draws and change the
/// worlds; byte-identical behavior of the moved bench_845 is the law).
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Build one world replicate (W1 with `w1_true_p`, W2 with `w2_true_p`).
pub fn build_world(
    seed: u64,
    corpus: &[[f32; D]],
    true_p_fn: fn(f32, bool) -> f32,
) -> Vec<Arrival> {
    let mut rng = SplitMix64::new(seed);
    (0..N_TEST)
        .map(|_| {
            let ood = rng.next_unit() < OOD_SHARE;
            let q = if ood {
                ood_query(&mut rng)
            } else {
                in_corpus_query(&mut rng, corpus)
            };
            let p = reported_confidence(&mut rng);
            let correct = rng.next_unit() < true_p_fn(p, ood);
            Arrival { q, p, ood, correct }
        })
        .collect()
}

/// The deep-abstention rate the readouts quote (ρ=30% — the regime where
/// the OOD mass is reachable). `#[allow(dead_code)]`: each bench include
/// consumes a different subset of this shared module, and a binary crate
/// dead-codes pub items its own main never touches.
#[allow(dead_code)]
pub const SEL30_RATE: f32 = 0.30;

/// Risk–coverage AURC: order by signal DESCENDING (most confident retained
/// first), average the prefix error rate over every coverage level. Lower
/// is better. O(n log n), one alloc — measurement-side only. (Moved from
/// bench_845 at Issue 984, re-homed here at Issue 985.)
pub fn aurc(signals: &[f32], errors: &[bool]) -> f32 {
    let mut idx: Vec<usize> = (0..signals.len()).collect();
    idx.sort_by(|a, b| signals[*b].total_cmp(&signals[*a]));
    let mut err_sum = 0usize;
    let mut total = 0.0f32;
    for (k, i) in idx.iter().enumerate() {
        err_sum += usize::from(errors[*i]);
        total += err_sum as f32 / (k + 1) as f32;
    }
    total / signals.len() as f32
}

/// Selective accuracy at matched abstain rate ρ: abstain the ρ-fraction of
/// LOWEST signal, return accuracy on the retained set. (Moved from
/// bench_845 at Issue 984, re-homed here at Issue 985.)
pub fn selective_accuracy_at(signals: &[f32], errors: &[bool], rate: f32) -> f32 {
    let mut sorted: Vec<f32> = signals.to_vec();
    sorted.sort_by(f32::total_cmp);
    let cut = ((rate * signals.len() as f32) as usize).min(signals.len() - 1);
    let threshold = sorted[cut];
    let mut ok = 0usize;
    let mut kept = 0usize;
    for (s, e) in signals.iter().zip(errors) {
        if *s >= threshold {
            ok += usize::from(!*e);
            kept += 1;
        }
    }
    ok as f32 / kept as f32
}
