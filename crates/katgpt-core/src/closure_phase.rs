//! Issue 1244 T3 — the phase-clustering arm of the [`crate::closure_audit`]
//! family (`closure_phase`, same opt-in feature, separate module —
//! `closure_audit.rs` sits at its line budget and the FSM-agreement
//! surface is a distinct consumer lane).
//!
//! # The question this answers
//!
//! The macrostate gate (Issue 1244 T0–T2) measures whether a latent
//! summary is CLOSED — nothing about its internal phase structure. An
//! FSM-agreement consumer needs discrete phases: cluster the recorded
//! states, then disclose whether the clustering is a property of the DATA
//! or a coin flip of the initialization. Lloyd's k-means is
//! seed-dependent, so the disclosure is NMI across every PAIR of ≥5
//! pinned seeds, typed:
//!
//! - [`PhaseStability::Stable`] iff the MINIMUM pairwise NMI is at the
//!   floor — every seed reproduces the phases;
//! - [`PhaseStability::StochasticBelow`] otherwise — named exactly per the
//!   filed task; a single lucky seed never promotes anything (the min
//!   rule is the "never promoted on one seed" law).
//!
//! The agglomerative arm (average linkage over the capped subsample) is
//! deterministic — no seeds — and its agreement with each k-means seed is
//! the cross-family read.
//!
//! # Arms
//!
//! - **Lloyd's k-means** — k-means++ D² initialization over the strided
//!   subsample (mirroring [`crate::factorized_action::codebook`]'s
//!   conventions: the cumulative-sum walk takes the FIRST index at/above
//!   the draw; assignment argmin keeps the FIRST minimum, strict `<`;
//!   dead-cluster revival to the farthest point with an Issue-997-style
//!   refresh against each just-placed center). Runtime `dim`/`k` — the
//!   const-generic codebook lane cannot serve arbitrary audit recordings.
//!   Tie divergence, pinned HERE: every tie in this module is FIRST-index
//!   (the filed task's "first-index ties"), including the revival
//!   farthest pick — codebook's revival deliberately pins LAST-index; the
//!   two lanes are independent pins, never merged silently.
//! - **Average-linkage agglomerative** (Lance–Williams weighted average)
//!   over a LOWER strided cap — the O(n²) distance matrix is the memory
//!   wall — merged until `k` clusters remain, canonical relabel by
//!   smallest member (first-index order).
//!
//! # Determinism + allocation posture (house style)
//!
//! - Seeded SplitMix64 (the house zero-dep RNG); every reduction is a
//!   sequential f64 fold in ascending order (the Issue-940 law).
//! - [`phase_cluster`] / [`phase_agreement`] are OFFLINE instruments with
//!   disclosed walls ([`PhaseReport::budget`]) and disclosed subsample
//!   posture (`stride`, `states_clustered`). Nothing here runs per tick.
//! - [`normalized_mutual_information`] pins the arithmetic-mean
//!   normalization, clamped to [0, 1]; a zero-entropy input conventionally
//!   shares no information (returns 0.0, documented) — EXCEPT the
//!   agreement entry point refuses a zero-entropy REFERENCE loud
//!   ([`PhaseClusterError::DegenerateReference`]): an FSM labeling with
//!   one state is a consumer bug, never a silent 0.0.
//!
//! # What this is NOT
//!
//! - NOT a router or a fix: the arm measures; the caller decides.
//! - NOT a per-state predictor: clustering is a sample statistic over the
//!   strided subsample; interstitial states are not labeled (map back by
//!   [`PhaseReport::stride`] where a consumer needs the phase of a
//!   sampled state).

use std::collections::HashMap;
use std::fmt;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// The minimum pinned-seed count — the stability disclosure is a PAIRWISE
/// read and a single seed promotes nothing.
pub const MIN_PHASE_SEEDS: usize = 5;

/// The default pinned seeds (five literals, ascending).
pub const DEFAULT_PHASE_SEEDS: [u64; MIN_PHASE_SEEDS] = [
    0x1244_7001,
    0x1244_7002,
    0x1244_7003,
    0x1244_7004,
    0x1244_7005,
];

/// The default stability/agreement floor on NMI. OUR bar, pinned at T3
/// landing — a planted-phase corpus must clear it on the MINIMUM pairwise
/// seed NMI, unstructured noise must not.
pub const DEFAULT_NMI_FLOOR: f64 = 0.8;

/// The default k-means corpus cap (strided subsample over the flattened
/// states).
pub const DEFAULT_STATE_CAP: usize = 4096;

/// The default agglomerative corpus cap — lower than the k-means cap: the
/// distance matrix is O(n²) f64 and the merge scan is O(n³) worst case.
pub const DEFAULT_AGGLOMERATIVE_CAP: usize = 1024;

/// The default Lloyd iteration cap.
pub const DEFAULT_MAX_ITERS: usize = 64;

/// The default cluster count. A consumer should set this to its FSM's
/// state count; the default is only a valid-config placeholder.
pub const DEFAULT_PHASE_K: usize = 4;

/// Phase-arm configuration. Validated by [`validate_config`] before any
/// clustering runs.
#[derive(Clone, Debug)]
pub struct PhaseClusterConfig {
    /// Cluster count — the consumer's expected FSM state count.
    pub k: usize,
    /// Pinned k-means seeds (≥ [`MIN_PHASE_SEEDS`]; the pairwise NMI
    /// disclosure runs over every pair).
    pub seeds: Vec<u64>,
    /// The floor both [`PhaseStability`] and [`AgreementVerdict`] gate on.
    pub nmi_floor: f64,
    /// k-means corpus cap (strided subsample; `>= k`).
    pub state_cap: usize,
    /// Agglomerative corpus cap (`>= k`; the O(n²) matrix is the wall).
    pub agglomerative_cap: usize,
    /// Lloyd iteration cap (`>= 1`).
    pub max_iters: usize,
}

impl Default for PhaseClusterConfig {
    fn default() -> Self {
        Self {
            k: DEFAULT_PHASE_K,
            seeds: DEFAULT_PHASE_SEEDS.to_vec(),
            nmi_floor: DEFAULT_NMI_FLOOR,
            state_cap: DEFAULT_STATE_CAP,
            agglomerative_cap: DEFAULT_AGGLOMERATIVE_CAP,
            max_iters: DEFAULT_MAX_ITERS,
        }
    }
}

/// Typed configuration/input refusal — never a silent default.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PhaseClusterError {
    /// The state slice was empty.
    NoStates,
    /// A zero-width state dimension.
    ZeroDimension,
    /// A non-finite value in an input state (typed, naming the state).
    NonFiniteInput { state: usize },
    /// A state row disagreed with the established dim.
    DimMismatch {
        context: &'static str,
        expected: usize,
        got: usize,
    },
    /// Fewer pinned seeds than the pairwise disclosure requires.
    TooFewSeeds { got: usize, min: usize },
    /// The agreement reference length disagreed with the state count.
    ReferenceLengthMismatch { expected: usize, got: usize },
    /// The agreement reference had zero entropy — an FSM labeling with one
    /// state is a consumer bug, refused loud (never a silent 0.0 NMI).
    DegenerateReference,
    /// The two labelings handed to [`normalized_mutual_information`]
    /// disagreed in length.
    LabelLengthMismatch { len_a: usize, len_b: usize },
    /// A configuration field refused.
    BadConfig { reason: &'static str },
}

impl fmt::Display for PhaseClusterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoStates => write!(f, "phase clustering: no states"),
            Self::ZeroDimension => write!(f, "phase clustering: zero-width dim"),
            Self::NonFiniteInput { state } => {
                write!(f, "phase clustering: non-finite value in state {state}")
            }
            Self::DimMismatch {
                context,
                expected,
                got,
            } => write!(
                f,
                "phase clustering: {context} dim {got} disagrees with established {expected}"
            ),
            Self::TooFewSeeds { got, min } => write!(
                f,
                "phase clustering: {got} pinned seeds is below the pairwise disclosure minimum {min}"
            ),
            Self::ReferenceLengthMismatch { expected, got } => write!(
                f,
                "phase clustering: reference length {got} disagrees with state count {expected}"
            ),
            Self::DegenerateReference => write!(
                f,
                "phase clustering: reference labeling has zero entropy (one state) — consumer bug"
            ),
            Self::LabelLengthMismatch { len_a, len_b } => write!(
                f,
                "phase clustering: labelings disagree in length ({len_a} vs {len_b})"
            ),
            Self::BadConfig { reason } => write!(f, "phase clustering: bad config: {reason}"),
        }
    }
}

impl std::error::Error for PhaseClusterError {}

fn validate_config(config: &PhaseClusterConfig) -> Result<(), PhaseClusterError> {
    if config.k == 0 {
        return Err(PhaseClusterError::BadConfig {
            reason: "k must be >= 1",
        });
    }
    if config.seeds.len() < MIN_PHASE_SEEDS {
        return Err(PhaseClusterError::TooFewSeeds {
            got: config.seeds.len(),
            min: MIN_PHASE_SEEDS,
        });
    }
    if !config.nmi_floor.is_finite() || !(0.0..=1.0).contains(&config.nmi_floor) {
        return Err(PhaseClusterError::BadConfig {
            reason: "nmi_floor must be finite within [0, 1]",
        });
    }
    if config.state_cap < config.k {
        return Err(PhaseClusterError::BadConfig {
            reason: "state_cap must be >= k",
        });
    }
    if config.agglomerative_cap < config.k {
        return Err(PhaseClusterError::BadConfig {
            reason: "agglomerative_cap must be >= k",
        });
    }
    if config.max_iters == 0 {
        return Err(PhaseClusterError::BadConfig {
            reason: "max_iters must be >= 1",
        });
    }
    Ok(())
}

fn validate_states(states: &[&[f32]], dim: usize) -> Result<(), PhaseClusterError> {
    if states.is_empty() {
        return Err(PhaseClusterError::NoStates);
    }
    if dim == 0 {
        return Err(PhaseClusterError::ZeroDimension);
    }
    for (i, row) in states.iter().enumerate() {
        if row.len() != dim {
            return Err(PhaseClusterError::DimMismatch {
                context: "state",
                expected: dim,
                got: row.len(),
            });
        }
        if row.iter().any(|v| !v.is_finite()) {
            return Err(PhaseClusterError::NonFiniteInput { state: i });
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Verdicts + reports
// ---------------------------------------------------------------------------

/// Seed-stability of the k-means phase clustering — the pairwise-NMI
/// disclosure. `Stable` requires the MINIMUM pairwise NMI at the floor:
/// every seed reproduces the phases, never one lucky draw.
#[derive(Clone, Debug, PartialEq)]
pub enum PhaseStability {
    /// Every pairwise seed NMI is at the floor.
    Stable { floor: f64, min_nmi: f64 },
    /// The phases did not reproduce across seeds — named per the filed
    /// task; never folded into a pass.
    StochasticBelow {
        floor: f64,
        mean_nmi: f64,
        min_nmi: f64,
        seeds_run: usize,
    },
}

/// Agreement of the clustering with a reference labeling (the consumer's
/// FSM states) — same min-rule law: agreement on one seed promotes
/// nothing.
#[derive(Clone, Debug, PartialEq)]
pub enum AgreementVerdict {
    /// Every seed's NMI vs the reference is at the floor.
    Agrees { floor: f64, min_nmi: f64 },
    /// No seed agreed — the phases and the FSM disagree.
    Disagrees { floor: f64, best_nmi: f64 },
    /// Some seeds agreed but not all — the initialization, not the data,
    /// decides the answer on this corpus.
    StochasticBelow {
        floor: f64,
        mean_nmi: f64,
        min_nmi: f64,
    },
}

/// Disclosed per-phase walls (an offline instrument — walls are
/// disclosed, never pinned).
#[derive(Clone, Debug, PartialEq)]
pub struct PhaseBudget {
    pub kmeans_ms: f64,
    pub agglomerative_ms: f64,
    pub nmi_ms: f64,
}

/// The clustering core shared by [`phase_cluster`] and [`phase_agreement`].
#[derive(Clone, Debug)]
struct ClusterCore {
    indices: Vec<usize>,
    stride: usize,
    seed_labels: Vec<Vec<u32>>,
    agglomerative_labels: Vec<u32>,
    kmeans_ms: f64,
    agglomerative_ms: f64,
}

/// The phase-clustering report: the typed stability verdict, the pairwise
/// disclosure numbers, and every labeling (the consumer maps a sampled
/// state back through `stride`).
#[derive(Clone, Debug)]
pub struct PhaseReport {
    /// Flattened state count handed in.
    pub states_total: usize,
    /// States actually clustered (after the strided subsample).
    pub states_clustered: usize,
    /// The subsample stride — sampled state `s` of the subsample is
    /// flattened state `s * stride`.
    pub stride: usize,
    /// The configured cluster count.
    pub k: usize,
    /// Populated clusters in the DETERMINISTIC (agglomerative) labeling.
    pub k_effective: usize,
    pub seeds_run: usize,
    /// Mean pairwise NMI across the k-means seeds.
    pub pairwise_nmi_mean: f64,
    /// Minimum pairwise NMI across the k-means seeds — the number
    /// [`PhaseStability`] gates on.
    pub pairwise_nmi_min: f64,
    /// Mean NMI of each k-means seed against the agglomerative arm.
    pub cross_family_nmi_mean: f64,
    pub stability: PhaseStability,
    /// One labeling per pinned seed (subsample order).
    pub seed_labels: Vec<Vec<u32>>,
    /// The deterministic agglomerative labeling (subsample order).
    pub agglomerative_labels: Vec<u32>,
    pub budget: PhaseBudget,
}

/// The FSM-agreement report: per-seed NMI against the reference labeling
/// plus the typed verdict.
#[derive(Clone, Debug)]
pub struct AgreementReport {
    pub seeds_run: usize,
    /// Per-seed NMI vs the reference (ascending seed order).
    pub per_seed_nmi: Vec<f64>,
    pub mean_nmi: f64,
    /// The number [`AgreementVerdict`] gates on.
    pub min_nmi: f64,
    pub verdict: AgreementVerdict,
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Cluster the recorded states into phases and disclose seed stability.
///
/// `states` is the trajectory-major recording (each row one flattened
/// state); the corpus is strided-subsampled to
/// [`PhaseClusterConfig::state_cap`]. Offline instrument with disclosed
/// walls — never a per-tick path.
pub fn phase_cluster(
    states: &[&[f32]],
    dim: usize,
    config: &PhaseClusterConfig,
) -> Result<PhaseReport, PhaseClusterError> {
    validate_config(config)?;
    validate_states(states, dim)?;
    let core = cluster_core(states, dim, config)?;

    let nmi_t = Instant::now();
    let s = core.seed_labels.len();
    let mut pairwise_sum = 0.0f64;
    let mut pairwise_min = f64::INFINITY;
    let mut pairs = 0usize;
    for i in 0..s {
        for j in (i + 1)..s {
            let nmi = normalized_mutual_information(&core.seed_labels[i], &core.seed_labels[j])?;
            pairwise_sum += nmi;
            if nmi < pairwise_min {
                pairwise_min = nmi;
            }
            pairs += 1;
        }
    }
    let mut cross_sum = 0.0f64;
    for labels in &core.seed_labels {
        cross_sum += normalized_mutual_information(labels, &core.agglomerative_labels)?;
    }
    let pairwise_nmi_mean = pairwise_sum / pairs as f64;
    let pairwise_nmi_min = pairwise_min;
    let cross_family_nmi_mean = cross_sum / s as f64;
    let nmi_ms = nmi_t.elapsed().as_secs_f64() * 1e3;

    let stability = if pairwise_min >= config.nmi_floor {
        PhaseStability::Stable {
            floor: config.nmi_floor,
            min_nmi: pairwise_min,
        }
    } else {
        PhaseStability::StochasticBelow {
            floor: config.nmi_floor,
            mean_nmi: pairwise_nmi_mean,
            min_nmi: pairwise_min,
            seeds_run: s,
        }
    };

    Ok(PhaseReport {
        states_total: states.len(),
        states_clustered: core.indices.len(),
        stride: core.stride,
        k: config.k,
        k_effective: count_populated(&core.agglomerative_labels),
        seeds_run: s,
        pairwise_nmi_mean,
        pairwise_nmi_min,
        cross_family_nmi_mean,
        stability,
        seed_labels: core.seed_labels,
        agglomerative_labels: core.agglomerative_labels,
        budget: PhaseBudget {
            kmeans_ms: core.kmeans_ms,
            agglomerative_ms: core.agglomerative_ms,
            nmi_ms,
        },
    })
}

/// Cluster the states and score every seed's NMI against a reference
/// labeling (the consumer's FSM states, flattened in the same
/// trajectory-major order).
///
/// The verdict gates on the MINIMUM per-seed NMI — agreement on one seed
/// promotes nothing. A zero-entropy reference (one FSM state everywhere)
/// refuses [`PhaseClusterError::DegenerateReference`] loud.
pub fn phase_agreement(
    states: &[&[f32]],
    dim: usize,
    reference: &[u32],
    config: &PhaseClusterConfig,
) -> Result<AgreementReport, PhaseClusterError> {
    validate_config(config)?;
    validate_states(states, dim)?;
    if reference.len() != states.len() {
        return Err(PhaseClusterError::ReferenceLengthMismatch {
            expected: states.len(),
            got: reference.len(),
        });
    }
    if reference.iter().all(|&x| x == reference[0]) {
        // The entropy check lives only on the reference: a collapsed
        // CLUSTERING is a finding (the verdicts carry it), a collapsed
        // REFERENCE is a consumer bug.
        return Err(PhaseClusterError::DegenerateReference);
    }
    let core = cluster_core(states, dim, config)?;
    let ref_sub: Vec<u32> = core.indices.iter().map(|&i| reference[i]).collect();

    let mut per_seed_nmi = Vec::with_capacity(core.seed_labels.len());
    let mut sum = 0.0f64;
    let mut min = f64::INFINITY;
    for labels in &core.seed_labels {
        let nmi = normalized_mutual_information(labels, &ref_sub)?;
        per_seed_nmi.push(nmi);
        sum += nmi;
        if nmi < min {
            min = nmi;
        }
    }
    let best = per_seed_nmi
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let mean = sum / per_seed_nmi.len() as f64;
    let verdict = if min >= config.nmi_floor {
        AgreementVerdict::Agrees {
            floor: config.nmi_floor,
            min_nmi: min,
        }
    } else if best < config.nmi_floor {
        AgreementVerdict::Disagrees {
            floor: config.nmi_floor,
            best_nmi: best,
        }
    } else {
        AgreementVerdict::StochasticBelow {
            floor: config.nmi_floor,
            mean_nmi: mean,
            min_nmi: min,
        }
    };

    Ok(AgreementReport {
        seeds_run: per_seed_nmi.len(),
        per_seed_nmi,
        mean_nmi: mean,
        min_nmi: min,
        verdict,
    })
}

/// Normalized mutual information between two labelings — the FSM-
/// agreement consumer's own comparison primitive.
///
/// Normalization is the arithmetic mean (`2·MI / (H_a + H_b)`), clamped to
/// [0, 1]. Either input with zero entropy shares no information by
/// definition — the convention returns 0.0 (the AGREEMENT entry point
/// refuses a zero-entropy reference; this raw fn is the honest measure).
pub fn normalized_mutual_information(a: &[u32], b: &[u32]) -> Result<f64, PhaseClusterError> {
    if a.len() != b.len() {
        return Err(PhaseClusterError::LabelLengthMismatch {
            len_a: a.len(),
            len_b: b.len(),
        });
    }
    let n = a.len();
    if n == 0 {
        return Ok(0.0);
    }
    // Dense renumber in first-appearance order — deterministic under any
    // relabeling of either side (the permutation-invariance fixture).
    let mut ka_map: HashMap<u32, usize> = HashMap::new();
    let mut kb_map: HashMap<u32, usize> = HashMap::new();
    let mut ka = 0usize;
    let mut kb = 0usize;
    let mut ja: Vec<usize> = Vec::with_capacity(n);
    let mut jb: Vec<usize> = Vec::with_capacity(n);
    for i in 0..n {
        let na = *ka_map.entry(a[i]).or_insert_with(|| {
            ka += 1;
            ka - 1
        });
        let nb = *kb_map.entry(b[i]).or_insert_with(|| {
            kb += 1;
            kb - 1
        });
        ja.push(na);
        jb.push(nb);
    }
    let mut joint = vec![0usize; ka * kb];
    let mut ma = vec![0usize; ka];
    let mut mb = vec![0usize; kb];
    for i in 0..n {
        joint[ja[i] * kb + jb[i]] += 1;
        ma[ja[i]] += 1;
        mb[jb[i]] += 1;
    }
    // Sequential f64 folds, ascending (i outer, j inner) — Issue-940.
    let nf = n as f64;
    let mut ha = 0.0f64;
    for &m in &ma {
        if m > 0 {
            let p = m as f64 / nf;
            ha -= p * p.ln();
        }
    }
    let mut hb = 0.0f64;
    for &m in &mb {
        if m > 0 {
            let p = m as f64 / nf;
            hb -= p * p.ln();
        }
    }
    let mut mi = 0.0f64;
    for i in 0..ka {
        for j in 0..kb {
            let nij = joint[i * kb + j];
            if nij > 0 {
                let p = nij as f64 / nf;
                mi += p * ((nij as f64 * nf) / (ma[i] as f64 * mb[j] as f64)).ln();
            }
        }
    }
    if ha + hb <= 0.0 {
        return Ok(0.0);
    }
    Ok(((2.0 * mi) / (ha + hb)).clamp(0.0, 1.0))
}

// ---------------------------------------------------------------------------
// Clustering core
// ---------------------------------------------------------------------------

/// The shared half: strided subsample, per-seed Lloyd's, the
/// deterministic agglomerative arm, disclosed walls.
fn cluster_core(
    states: &[&[f32]],
    dim: usize,
    config: &PhaseClusterConfig,
) -> Result<ClusterCore, PhaseClusterError> {
    let total = states.len();
    let stride = if total <= config.state_cap {
        1
    } else {
        total.div_ceil(config.state_cap)
    };
    let indices: Vec<usize> = (0..total).step_by(stride).collect();
    let n = indices.len();
    if config.k > n {
        return Err(PhaseClusterError::BadConfig {
            reason: "k must be <= the clustered state count",
        });
    }

    let kmeans_t = Instant::now();
    let mut seed_labels = Vec::with_capacity(config.seeds.len());
    for &seed in &config.seeds {
        seed_labels.push(kmeans_once(
            states,
            dim,
            &indices,
            config.k,
            seed,
            config.max_iters,
        ));
    }
    let kmeans_ms = kmeans_t.elapsed().as_secs_f64() * 1e3;

    let agglom_t = Instant::now();
    // The agglomerative arm takes its OWN (lower) strided subsample — the
    // O(n²) matrix is the wall, disclosed by the cap difference.
    let stride2 = if total <= config.agglomerative_cap {
        1
    } else {
        total.div_ceil(config.agglomerative_cap)
    };
    let indices2: Vec<usize> = (0..total).step_by(stride2).collect();
    let agglomerative_labels = agglomerative_once(states, &indices2, config.k);
    let agglomerative_ms = agglom_t.elapsed().as_secs_f64() * 1e3;

    Ok(ClusterCore {
        indices,
        stride,
        seed_labels,
        agglomerative_labels,
        kmeans_ms,
        agglomerative_ms,
    })
}

fn sq_dist(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        acc += d * d;
    }
    acc
}

/// Seeded k-means++ center selection over the subsampled points. The
/// FIRST center is the first-index anchor (divergence from codebook's
/// `random % n` — pinned here: the stability read wants its seed
/// diversity in the D² draws, not the anchor); every subsequent draw is
/// the FIRST cumulative index at/above `r` (strict `<` fold, first-index
/// tie); the all-coincide fallback mirrors codebook's `random % n`.
fn kmeans_pp_init(
    states: &[&[f32]],
    indices: &[usize],
    k: usize,
    rng: &mut SplitMix64,
) -> Vec<Vec<f32>> {
    let n = indices.len();
    let mut centers: Vec<Vec<f32>> = Vec::with_capacity(k);
    centers.push(states[indices[0]].to_vec());

    let mut d2_nearest: Vec<f64> = (0..n)
        .map(|i| sq_dist(states[indices[i]], &centers[0]) as f64)
        .collect();

    for _ in 1..k {
        let total: f64 = d2_nearest.iter().sum();
        let chosen = if total <= 0.0 {
            (rng.next_u64() as usize) % n
        } else {
            let r = rng.next_f64() * total;
            let mut acc = 0.0f64;
            let mut chosen = n - 1;
            for (i, &d2) in d2_nearest.iter().enumerate() {
                acc += d2;
                if acc >= r {
                    chosen = i;
                    break;
                }
            }
            chosen
        };
        centers.push(states[indices[chosen]].to_vec());
        let new_c = &centers[centers.len() - 1];
        for i in 0..n {
            let d2 = sq_dist(states[indices[i]], new_c) as f64;
            if d2 < d2_nearest[i] {
                d2_nearest[i] = d2;
            }
        }
    }
    centers
}

fn refresh_d2_nearest(
    d2_nearest: &mut [f64],
    states: &[&[f32]],
    indices: &[usize],
    center: &[f32],
) {
    for (i, d) in d2_nearest.iter_mut().enumerate() {
        let d2 = sq_dist(states[indices[i]], center) as f64;
        if d2 < *d {
            *d = d2;
        }
    }
}

/// One seeded Lloyd's run. Assignment argmin keeps the FIRST minimum
/// (strict `<` — the module's first-index tie law); a dead cluster
/// revives to the FARTHEST point (strict `>` fold — ALSO first-index,
/// the deliberate divergence from codebook's pinned last-max) with an
/// Issue-997-style refresh against each just-placed center.
fn kmeans_once(
    states: &[&[f32]],
    dim: usize,
    indices: &[usize],
    k: usize,
    seed: u64,
    max_iters: usize,
) -> Vec<u32> {
    let n = indices.len();
    let mut rng = SplitMix64::new(seed);
    let mut centers = kmeans_pp_init(states, indices, k, &mut rng);

    let mut d2_nearest: Vec<f64> = (0..n)
        .map(|i| sq_dist(states[indices[i]], &centers[0]) as f64)
        .collect();
    for c in centers.iter().skip(1) {
        refresh_d2_nearest(&mut d2_nearest, states, indices, c);
    }

    let mut assignments: Vec<usize> = vec![0; n];
    let mut sums: Vec<Vec<f64>> = vec![vec![0.0f64; dim]; k];
    let mut counts: Vec<usize> = vec![0; k];

    for _ in 0..max_iters {
        let mut changed = false;

        for i in 0..n {
            let mut best_k = 0usize;
            let mut best_d2 = f32::INFINITY;
            for (kk, c) in centers.iter().enumerate() {
                let d2 = sq_dist(states[indices[i]], c);
                if d2 < best_d2 {
                    best_d2 = d2;
                    best_k = kk;
                }
            }
            if assignments[i] != best_k {
                assignments[i] = best_k;
                changed = true;
            }
        }

        for s in sums.iter_mut() {
            s.fill(0.0);
        }
        counts.fill(0);
        for i in 0..n {
            let kk = assignments[i];
            counts[kk] += 1;
            let row = states[indices[i]];
            for d in 0..dim {
                sums[kk][d] += row[d] as f64;
            }
        }

        if counts.contains(&0) {
            for i in 0..n {
                let mut best_d2 = f64::INFINITY;
                for (kk, c) in centers.iter().enumerate() {
                    if counts[kk] == 0 {
                        continue;
                    }
                    let d2 = sq_dist(states[indices[i]], c) as f64;
                    if d2 < best_d2 {
                        best_d2 = d2;
                    }
                }
                d2_nearest[i] = best_d2;
            }
        }
        for kk in 0..k {
            if counts[kk] == 0 {
                // Farthest-point revival, FIRST-index tie (strict `>`).
                let mut far_idx = 0usize;
                let mut far_d2 = f64::NEG_INFINITY;
                for (i, &d) in d2_nearest.iter().enumerate() {
                    if d > far_d2 {
                        far_d2 = d;
                        far_idx = i;
                    }
                }
                centers[kk].copy_from_slice(states[indices[far_idx]]);
                d2_nearest[far_idx] = 0.0;
                refresh_d2_nearest(&mut d2_nearest, states, indices, &centers[kk]);
                changed = true;
            } else {
                let inv = 1.0 / counts[kk] as f64;
                for d in 0..dim {
                    centers[kk][d] = (sums[kk][d] * inv) as f32;
                }
            }
        }

        if !changed {
            break;
        }
    }

    assignments.iter().map(|&a| a as u32).collect()
}

/// Average-linkage agglomerative clustering (Lance–Williams weighted
/// average) over the subsampled points, merged until `k` clusters remain,
/// canonically relabeled by smallest member (first-index order). The
/// merge-min scan is ascending `(i, j)` with a strict `<` fold — the
/// first-index tie law again.
fn agglomerative_once(states: &[&[f32]], indices: &[usize], k: usize) -> Vec<u32> {
    let n = indices.len();
    // n initial rows + at most n − k merged rows, room to spare.
    let rows = 2 * n;
    let mut dist = vec![f64::INFINITY; rows * rows];
    for a in 0..n {
        dist[a * rows + a] = 0.0;
        for b in (a + 1)..n {
            let d = sq_dist(states[indices[a]], states[indices[b]]) as f64;
            dist[a * rows + b] = d;
            dist[b * rows + a] = d;
        }
    }
    let mut members: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    let mut alive: Vec<bool> = vec![true; rows];
    let mut alive_count = n;
    let mut next_id = n;

    while alive_count > k {
        let (mut best_a, mut best_b) = (usize::MAX, usize::MAX);
        let mut best_d = f64::INFINITY;
        for i in 0..next_id {
            if !alive[i] {
                continue;
            }
            for j in (i + 1)..next_id {
                if !alive[j] {
                    continue;
                }
                let d = dist[i * rows + j];
                if d < best_d {
                    best_d = d;
                    best_a = i;
                    best_b = j;
                }
            }
        }
        debug_assert!(best_a != usize::MAX, "agglomerative: no alive pair");
        let sa = members[best_a].len();
        let sb = members[best_b].len();
        let c = next_id;
        next_id += 1;
        for m in 0..next_id {
            if !alive[m] || m == best_a || m == best_b {
                continue;
            }
            let d = (sa as f64 * dist[best_a * rows + m] + sb as f64 * dist[best_b * rows + m])
                / (sa + sb) as f64;
            dist[c * rows + m] = d;
            dist[m * rows + c] = d;
        }
        dist[c * rows + c] = 0.0;
        let mut merged = members[best_a].clone();
        merged.extend_from_slice(&members[best_b]);
        members.push(merged);
        alive[best_a] = false;
        alive[best_b] = false;
        alive[c] = true;
        alive_count += 1; // two die, one is born
        alive_count -= 2;
    }

    // Canonical relabel: clusters ordered by their smallest member, ranks
    // ascending.
    let mut order: Vec<usize> = (0..next_id).filter(|&i| alive[i]).collect();
    order.sort_by_key(|&i| members[i][0]);
    let mut labels = vec![0u32; n];
    for (rank, &cid) in order.iter().enumerate() {
        for &m in &members[cid] {
            labels[m] = rank as u32;
        }
    }
    labels
}

fn count_populated(labels: &[u32]) -> usize {
    let mut seen = vec![false; labels.iter().copied().max().map_or(1, |m| m as usize + 1)];
    let mut count = 0usize;
    for &l in labels {
        let idx = l as usize;
        if !seen[idx] {
            seen[idx] = true;
            count += 1;
        }
    }
    count
}

// ---------------------------------------------------------------------------
// RNG — the house deterministic test/fixture RNG (zero-dep)
// ---------------------------------------------------------------------------

struct SplitMix64 {
    state: u64,
    /// Box–Muller second output (test/fixture draws only).
    #[cfg(test)]
    pending: Option<f64>,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self {
            state: seed,
            #[cfg(test)]
            pending: None,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9007199254740992.0)
    }

    /// Standard normal via Box–Muller (tests/fixtures only — the arm
    /// itself never draws normals).
    #[cfg(test)]
    fn next_normal(&mut self) -> f64 {
        if let Some(p) = self.pending.take() {
            return p;
        }
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        self.pending = Some(r * theta.sin());
        r * theta.cos()
    }
}

// ---------------------------------------------------------------------------
// Tests — the G1 fixture set (feature-gated with the module)
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "closure_audit"))]
mod tests {
    use super::*;

    /// `k` well-separated Gaussian blobs, one per axis prefix — planted
    /// labels ride along.
    fn blob_corpus(
        seed: u64,
        k: usize,
        per_blob: usize,
        dim: usize,
        separation: f32,
        noise: f32,
    ) -> (Vec<Vec<f32>>, Vec<u32>) {
        let mut rng = SplitMix64::new(seed);
        let mut states = Vec::with_capacity(k * per_blob);
        let mut labels = Vec::with_capacity(k * per_blob);
        for cluster in 0..k {
            for _ in 0..per_blob {
                let mut row = vec![0.0f32; dim];
                row[cluster] = separation;
                for v in row.iter_mut() {
                    *v += (rng.next_normal() as f32) * noise;
                }
                states.push(row);
                labels.push(cluster as u32);
            }
        }
        (states, labels)
    }

    fn slices(states: &[Vec<f32>]) -> Vec<&[f32]> {
        states.iter().map(|s| s.as_slice()).collect()
    }

    fn config_for(k: usize) -> PhaseClusterConfig {
        PhaseClusterConfig {
            k,
            ..PhaseClusterConfig::default()
        }
    }

    #[test]
    fn planted_blobs_read_stable_and_agree() {
        let (states, labels) = blob_corpus(0x1244_0001, 3, 100, 8, 6.0, 0.3);
        let slices = slices(&states);
        let config = config_for(3);
        let report = phase_cluster(&slices, 8, &config).expect("planted corpus clusters");
        assert_eq!(report.seeds_run, 5);
        assert_eq!(report.states_clustered, 300);
        assert_eq!(report.stride, 1);
        assert!(
            matches!(report.stability, PhaseStability::Stable { .. }),
            "planted phases must read Stable, got {:?} (min {})",
            report.stability,
            report.pairwise_nmi_min
        );
        assert!(report.pairwise_nmi_min >= config.nmi_floor);
        assert!(report.pairwise_nmi_mean >= report.pairwise_nmi_min);
        assert!(report.cross_family_nmi_mean >= config.nmi_floor);

        let agreement = phase_agreement(&slices, 8, &labels, &config).expect("planted agreement");
        assert_eq!(agreement.seeds_run, 5);
        assert!(
            matches!(agreement.verdict, AgreementVerdict::Agrees { .. }),
            "planted phases must agree with their labels, got {:?}",
            agreement.verdict
        );
        assert!(agreement.min_nmi >= config.nmi_floor);
    }

    #[test]
    fn unstructured_noise_reads_stochastic_below() {
        // One cloud — no phase structure; the seeds must NOT reproduce a
        // partition the data does not carry.
        let (states, _) = blob_corpus(0x1244_0002, 1, 300, 8, 0.0, 1.0);
        let slices = slices(&states);
        let report = phase_cluster(&slices, 8, &config_for(3)).expect("noise corpus clusters");
        assert!(
            matches!(report.stability, PhaseStability::StochasticBelow { .. }),
            "unstructured noise must read StochasticBelow, got {:?} (min {}, mean {})",
            report.stability,
            report.pairwise_nmi_min,
            report.pairwise_nmi_mean
        );
    }

    #[test]
    fn degenerate_identical_states_all_tie_to_first_cluster() {
        // Every state identical: every assignment distance is an exact
        // tie → the first-index rule pins ALL states to cluster 0; the
        // collapsed labeling has zero entropy → the pairwise disclosure
        // reads 0 and refuses to call the collapse Stable.
        let states: Vec<Vec<f32>> = (0..60).map(|_| vec![1.0f32, 2.0, 3.0, 4.0]).collect();
        let slices = slices(&states);
        let report = phase_cluster(&slices, 4, &config_for(3)).expect("degenerate corpus runs");
        for labels in &report.seed_labels {
            assert!(
                labels.iter().all(|&l| l == 0),
                "identical states must all tie to cluster 0, got {labels:?}"
            );
        }
        assert_eq!(report.pairwise_nmi_min, 0.0);
        assert!(matches!(
            report.stability,
            PhaseStability::StochasticBelow { .. }
        ));
    }

    #[test]
    fn agglomerative_arm_matches_planted_phases() {
        let (states, labels) = blob_corpus(0x1244_0003, 3, 100, 8, 6.0, 0.3);
        let slices = slices(&states);
        let report = phase_cluster(&slices, 8, &config_for(3)).expect("planted corpus clusters");
        let nmi = normalized_mutual_information(&report.agglomerative_labels, &labels)
            .expect("same length");
        assert!(
            nmi >= 0.8,
            "agglomerative arm must match planted phases, got {nmi}"
        );
        assert_eq!(report.k_effective, 3);
    }

    #[test]
    fn deterministic_bit_identical_reruns() {
        let (states, _) = blob_corpus(0x1244_0004, 3, 60, 6, 4.0, 0.4);
        let slices = slices(&states);
        let config = config_for(3);
        let a = phase_cluster(&slices, 6, &config).expect("run a");
        let b = phase_cluster(&slices, 6, &config).expect("run b");
        assert_eq!(a.seed_labels, b.seed_labels);
        assert_eq!(a.agglomerative_labels, b.agglomerative_labels);
        assert_eq!(a.pairwise_nmi_min, b.pairwise_nmi_min);
    }

    #[test]
    fn strided_subsample_disclosed() {
        let (states, _) = blob_corpus(0x1244_0005, 2, 200, 4, 5.0, 0.3);
        let slices = slices(&states);
        let config = PhaseClusterConfig {
            k: 2,
            state_cap: 100,
            agglomerative_cap: 100,
            ..PhaseClusterConfig::default()
        };
        let report = phase_cluster(&slices, 4, &config).expect("strided corpus clusters");
        assert_eq!(report.states_total, 400);
        assert_eq!(report.states_clustered, 100);
        assert_eq!(report.stride, 4);
        assert_eq!(report.seed_labels.len(), 5);
        assert!(report.seed_labels.iter().all(|l| l.len() == 100));
    }

    #[test]
    fn nmi_units() {
        // Identical partitions.
        let a = [0u32, 0, 1, 1, 2, 2];
        assert!((normalized_mutual_information(&a, &a).unwrap() - 1.0).abs() < 1e-12);
        // Relabeled-equal partition — permutation invariance.
        let b = [2u32, 2, 0, 0, 1, 1];
        assert!((normalized_mutual_information(&a, &b).unwrap() - 1.0).abs() < 1e-12);
        // The independent 2×2: every joint cell equiprobable → MI = 0.
        let x = [0u32, 0, 1, 1];
        let y = [0u32, 1, 0, 1];
        assert!(normalized_mutual_information(&x, &y).unwrap().abs() < 1e-12);
        // Zero-entropy input conventionally shares nothing.
        let c = [1u32, 1, 1, 1];
        assert_eq!(normalized_mutual_information(&x, &c).unwrap(), 0.0);
        // Length mismatch is typed.
        assert!(matches!(
            normalized_mutual_information(&x, &[0u32, 1]),
            Err(PhaseClusterError::LabelLengthMismatch { .. })
        ));
    }

    #[test]
    fn config_and_input_refusals() {
        let (states, _) = blob_corpus(0x1244_0006, 2, 50, 4, 5.0, 0.3);
        let slices = slices(&states);

        // Too few pinned seeds.
        let few = PhaseClusterConfig {
            k: 2,
            seeds: vec![1, 2],
            ..PhaseClusterConfig::default()
        };
        assert!(matches!(
            phase_cluster(&slices, 4, &few),
            Err(PhaseClusterError::TooFewSeeds { got: 2, min: 5 })
        ));

        // k = 0 / k > clustered count / bad floor / caps under k / iters 0.
        let k0 = PhaseClusterConfig {
            k: 0,
            ..Default::default()
        };
        assert!(matches!(
            phase_cluster(&slices, 4, &k0),
            Err(PhaseClusterError::BadConfig { .. })
        ));
        let kbig = PhaseClusterConfig {
            k: 10,
            state_cap: 5,
            agglomerative_cap: 5,
            ..Default::default()
        };
        assert!(matches!(
            phase_cluster(&slices, 4, &kbig),
            Err(PhaseClusterError::BadConfig { .. })
        ));
        let floor_bad = PhaseClusterConfig {
            k: 2,
            nmi_floor: 1.5,
            ..Default::default()
        };
        assert!(matches!(
            phase_cluster(&slices, 4, &floor_bad),
            Err(PhaseClusterError::BadConfig { .. })
        ));
        let cap_small = PhaseClusterConfig {
            k: 2,
            state_cap: 1,
            ..Default::default()
        };
        assert!(matches!(
            phase_cluster(&slices, 4, &cap_small),
            Err(PhaseClusterError::BadConfig { .. })
        ));
        let iters0 = PhaseClusterConfig {
            k: 2,
            max_iters: 0,
            ..Default::default()
        };
        assert!(matches!(
            phase_cluster(&slices, 4, &iters0),
            Err(PhaseClusterError::BadConfig { .. })
        ));

        // Input refusals: empty, zero dim, dim mismatch, non-finite.
        assert!(matches!(
            phase_cluster(&[], 4, &config_for(2)),
            Err(PhaseClusterError::NoStates)
        ));
        assert!(matches!(
            phase_cluster(&slices, 0, &config_for(2)),
            Err(PhaseClusterError::ZeroDimension)
        ));
        let ragged: Vec<&[f32]> = vec![&[1.0, 2.0], &[1.0, 2.0, 3.0]];
        assert!(matches!(
            phase_cluster(&ragged, 2, &config_for(2)),
            Err(PhaseClusterError::DimMismatch { got: 3, .. })
        ));
        let mut nan_row = vec![1.0f32, 2.0];
        nan_row[1] = f32::NAN;
        let nan_slices: Vec<&[f32]> = vec![&[1.0, 2.0], &nan_row];
        assert!(matches!(
            phase_cluster(&nan_slices, 2, &config_for(2)),
            Err(PhaseClusterError::NonFiniteInput { state: 1 })
        ));

        // Agreement refusals: length mismatch + degenerate reference.
        assert!(matches!(
            phase_agreement(&slices, 4, &[0u32; 10], &config_for(2)),
            Err(PhaseClusterError::ReferenceLengthMismatch { .. })
        ));
        let degenerate_ref = vec![0u32; states.len()];
        assert!(matches!(
            phase_agreement(&slices, 4, &degenerate_ref, &config_for(2)),
            Err(PhaseClusterError::DegenerateReference)
        ));
    }
}
