//! Leakage Probe — modelless attribute-transfer leak score for stored
//! vector stores (Issue 736, Research 540; threat model from
//! arXiv:2505.12540 *Harnessing the Universal Geometry of Embeddings*,
//! NeurIPS 2025).
//!
//! # The threat (why a defender wants this number)
//!
//! vec2vec translates embeddings between model spaces with NO paired data
//! and NO encoder access (cosine 0.85–0.96 to ground truth; modelless OT
//! baselines fail to random cross-backbone). Consequence: an attacker
//! holding only a stolen vector-store file can translate it into a space
//! where they hold labeled attribute anchors, then run zero-shot attribute
//! inference over the store.
//!
//! # What the probe measures
//!
//! The defender-side simulation of that attack, GD-free end to end:
//!
//! 1. OUR stored vectors (a sample of the store, with the attributes the
//!    defender knows) + a FOREIGN sample (what an adversary's own embedder
//!    produces over their corpus, with the public attribute tags).
//! 2. [`transport`] aligns the two samples UNPAIRED — no labels touch the
//!    alignment (labels only evaluate, exactly as in the attack).
//! 3. kNN attribute transfer scores how often the transported ours-vectors
//!    land in foreign neighborhoods carrying the CORRECT attribute —
//!    compared against the majority-class chance baseline.
//!
//! The report is a conservative LOWER bound on transferable attribute
//! structure: the transport here is the honest OT tier (the paper's own
//! baselines), so anything the probe measures, the trained-translator
//! attack measures at least as well. `InsufficientAlignment` means the
//! geometry did not align under the modelless chain — interpret nothing.
//!
//! # Labels and ontology
//!
//! `labels_ours` and `labels_foreign` must share one ontology (public
//! attribute tags in both spaces) — that shared tag space IS the attack
//! model. Label ids are arbitrary `u32`s.
//!
//! # Distinguisher (signal-diff vs shipped audits)
//!
//! `latent_confounder_audit` audits OUR direction vectors via
//! counterfactual slices in ONE space. This probe operates ACROSS two
//! spaces with no encoder access — no shipped primitive covered it
//! (Research 540 §3: repo-wide grep zero hits at distill time).
//!
//! # Gates (in-module; run with `--features leakage_probe`)
//!
//! - G1 planted-leak recovery: a two-space linear fixture with a shared
//!   latent + binary attribute recovers `High`.
//! - G1b monotone-in-noise: transfer accuracy is non-increasing in observed
//!   noise and beats chance at zero noise.
//! - G1c independent-spaces: two independent latents stay at chance
//!   (`Low`) — the honest-negative floor.
//! - G2 smoke: audit-cadence latency on a 128×32 problem (release-mode
//!   measurement belongs to the Issue 614 E2 adoption bench).
//!
//! Opt-in until the consumer (riir-neuron-db Issue 614 E2) lands; no GOAT
//! bench claim until then (the `cond_audit` precedent).

mod transport;

use std::fmt;

pub use transport::{MAX_DIM, MAX_LATENT_DIM, MIN_SAMPLES};

/// Probe configuration. All knobs are deterministic — no RNG anywhere.
#[derive(Debug, Clone)]
pub struct LeakProbeConfig {
    /// Common transport dimensionality (top-`k` principal subspace of each
    /// space). Effectively clamped to `min(latent_dim, d1, d2)`.
    pub latent_dim: usize,
    /// Entropic regularization for Sinkhorn (cost is cosine distance
    /// ∈ [0, 2]).
    pub sinkhorn_eps: f32,
    /// Sinkhorn scaling iterations.
    pub sinkhorn_iters: usize,
    /// Sinkhorn ↔ Procrustes alternating rounds.
    pub transport_rounds: usize,
    /// Subspace-iteration sweeps per space (top-k principal subspace).
    /// The within-subspace column convergence runs at (λ₂/λ₁)^sweeps — 30
    /// keeps the angle error ≪ 1° for eigenvalue ratios down to ~0.8.
    pub subspace_iters: usize,
    /// k for the kNN attribute vote.
    pub knn_k: usize,
    /// `lift ≥ high_lift` → [`LeakVerdict::High`].
    pub high_lift: f32,
    /// `lift ≥ elevated_lift` → [`LeakVerdict::Elevated`] (below `high_lift`).
    pub elevated_lift: f32,
    /// Mean pseudo-pair cosine below this → [`LeakVerdict::InsufficientAlignment`].
    pub min_alignment_cos: f32,
    /// Deterministic multi-start: run the Sinkhorn↔Procrustes loop from
    /// `k + 2` fixed orientations (identity, each single-axis sign flip,
    /// all-flip) and keep the best mean pair cosine. Breaks the wrong-basin
    /// ICP lock-in (cluster-permutation derangements) at (k+2)× cost.
    pub multistart: bool,
}

impl Default for LeakProbeConfig {
    fn default() -> Self {
        Self {
            latent_dim: 8,
            sinkhorn_eps: 0.1,
            sinkhorn_iters: 50,
            transport_rounds: 4,
            subspace_iters: 30,
            knn_k: 5,
            high_lift: 2.0,
            elevated_lift: 1.25,
            min_alignment_cos: 0.2,
            multistart: true,
        }
    }
}

/// Verdict tiers on the chance-normalized transfer lift.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeakVerdict {
    /// The modelless transport failed to align the geometries — the score
    /// is uninterpretable; treat as "not measurable here", NOT as safe.
    InsufficientAlignment,
    /// Transfer lift below the elevated bar — at chance-level attribute
    /// structure under this transport.
    Low,
    /// Above chance but below the high bar.
    Elevated,
    /// Strong attribute transfer — the store leaks this attribute under
    /// the vec2vec attack model.
    High,
}

impl fmt::Display for LeakVerdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            LeakVerdict::InsufficientAlignment => "insufficient-alignment",
            LeakVerdict::Low => "low",
            LeakVerdict::Elevated => "elevated",
            LeakVerdict::High => "high",
        };
        f.write_str(name)
    }
}

/// Measured leak report.
#[derive(Debug, Clone, PartialEq)]
pub struct LeakReport {
    /// Fraction of ours-vectors whose transported kNN majority vote
    /// recovers their own attribute label.
    pub attribute_transfer_top1: f32,
    /// Majority-class frequency among `labels_ours` — the no-information
    /// baseline for [`Self::attribute_transfer_top1`].
    pub chance_baseline: f32,
    /// `attribute_transfer_top1 / chance_baseline` — the headline score.
    pub lift: f32,
    /// Mean pseudo-pair cosine after the final transport round — alignment
    /// quality. Interpret the lift only above
    /// `LeakProbeConfig::min_alignment_cos`.
    pub alignment_mean_cos: f32,
    /// Fraction of pseudo-pairs whose partner lands in the transported
    /// vector's kNN set — a label-free transport-quality diagnostic.
    pub neighborhood_hit_rate: f32,
    /// `knn_k / n_foreign` — the chance floor for the hit rate.
    pub neighborhood_hit_chance: f32,
    pub verdict: LeakVerdict,
}

/// Probe input errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeakProbeError {
    /// Buffer length does not match `n * d` or a label length does not
    /// match its sample count.
    DimMismatch,
    /// A side has fewer than [`MIN_SAMPLES`] rows.
    TooFewSamples,
    /// A side exceeds [`MAX_DIM`] columns.
    DimTooLarge,
    /// `latent_dim` exceeds [`MAX_LATENT_DIM`].
    LatentDimTooLarge,
    /// Effective `k` needs ≥ 4·k rows per side for a stable covariance.
    LatentDimExceedsSamples,
    /// `labels_ours` has fewer than 2 distinct classes — the chance
    /// baseline is degenerate and the lift is undefined.
    InsufficientLabelClasses,
}

impl fmt::Display for LeakProbeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            LeakProbeError::DimMismatch => "dim mismatch (buffer vs n·d / labels vs n)",
            LeakProbeError::TooFewSamples => "fewer than MIN_SAMPLES rows on a side",
            LeakProbeError::DimTooLarge => "a side exceeds MAX_DIM columns",
            LeakProbeError::LatentDimTooLarge => "latent_dim exceeds MAX_LATENT_DIM",
            LeakProbeError::LatentDimExceedsSamples => "samples below 4·k for the effective k",
            LeakProbeError::InsufficientLabelClasses => "labels_ours needs ≥ 2 distinct classes",
        };
        f.write_str(name)
    }
}

impl std::error::Error for LeakProbeError {}

/// Runs the leak probe. See the module docs for the threat model and the
/// label-ontology requirement.
pub fn probe(
    x_ours: &[f32],
    labels_ours: &[u32],
    d1: usize,
    x_foreign: &[f32],
    labels_foreign: &[u32],
    d2: usize,
    cfg: &LeakProbeConfig,
) -> Result<LeakReport, LeakProbeError> {
    if d1 == 0 || d2 == 0 || !x_ours.len().is_multiple_of(d1) || !x_foreign.len().is_multiple_of(d2) {
        return Err(LeakProbeError::DimMismatch);
    }
    let n1 = x_ours.len() / d1;
    let n2 = x_foreign.len() / d2;
    if labels_ours.len() != n1 || labels_foreign.len() != n2 {
        return Err(LeakProbeError::DimMismatch);
    }
    if labels_ours.iter().collect::<std::collections::BTreeSet<_>>().len() < 2 {
        return Err(LeakProbeError::InsufficientLabelClasses);
    }
    let k = transport::validate_dims(n1, d1, n2, d2, cfg.latent_dim).map_err(probe_err)?;

    // Project both spaces onto their top-k principal subspace, then whiten
    // onto the unit sphere.
    let mut sc = transport::TransportScratch::new();
    let mut w1 = Vec::with_capacity(n1 * k);
    let mut w2 = Vec::with_capacity(n2 * k);
    transport::project_topk(
        x_ours, n1, d1, k, cfg.subspace_iters, &mut w1, &mut sc,
    );
    transport::project_topk(
        x_foreign, n2, d2, k, cfg.subspace_iters, &mut w2, &mut sc,
    );
    transport::whiten_inplace(&mut w1, n1, k, &mut sc);
    transport::whiten_inplace(&mut w2, n2, k, &mut sc);

    // Alternating Sinkhorn ↔ Procrustes. `mapped` = w1·R drives the cost
    // for pairing; the Procrustes solve always maps the ORIGINAL w1 so `r`
    // replaces (never composes).
    let mut mapped = w1.clone();
    let mut r = vec![0.0f32; k * k];
    let mut plan = Vec::new();
    let mut cost = vec![0.0f32; n1 * n2];
    let mut pairs: Vec<usize> = Vec::new();
    let mut sims: Vec<f32> = Vec::with_capacity(n2.max(n1));
    let mut top: Vec<u32> = Vec::with_capacity(cfg.knn_k);
    let mut alignment = 0.0f32;
    // Deterministic multi-start: identity, each single-axis sign flip,
    // then all-flip. Wrong-basin ICP lock-in (cluster-permutation
    // derangements) shows up as a LOWER final pair cosine than the true
    // basin once the fixture/spaces are non-congruent, so best-of-starts
    // by mean pair cosine escapes it.
    let n_starts = if cfg.multistart { k + 2 } else { 1 };
    let mut best_align = f32::NEG_INFINITY;
    let mut best_mapped: Vec<f32> = Vec::new();
    let mut best_pairs: Vec<usize> = Vec::new();
    for s in 0..n_starts {
        mapped.copy_from_slice(&w1);
        if s >= 1 {
            let flip_all = s == k + 1;
            let flip_axis = s - 1;
            for i in 0..n1 {
                if flip_all {
                    for t in 0..k {
                        mapped[i * k + t] = -mapped[i * k + t];
                    }
                } else {
                    mapped[i * k + flip_axis] = -mapped[i * k + flip_axis];
                }
            }
        }
        for _ in 0..cfg.transport_rounds.max(1) {
        // CSLS-corrected cosine cost (Conneau et al. 2018 — the canonical
        // hubness fix for unsupervised nearest-neighbor matching; without
        // it ICP locks onto high-degree hub points in isotropic clouds):
        // sim' = 2·cos − r_x − r_y, cost = −sim' = r_x + r_y − 2·cos.
        let r_m = mean_topk_sims_rowwise(&mapped, &w2, k, cfg.knn_k, &mut sims, &mut top);
        let r_f = mean_topk_sims_rowwise(&w2, &mapped, k, cfg.knn_k, &mut sims, &mut top);
        for i in 0..n1 {
            let mi = &mapped[i * k..(i + 1) * k];
            for j in 0..n2 {
                let fj = &w2[j * k..(j + 1) * k];
                let mut dot = 0.0f32;
                for t in 0..k {
                    dot += mi[t] * fj[t];
                }
                cost[i * n2 + j] = r_m[i] + r_f[j] - 2.0 * dot;
            }
        }
        transport::sinkhorn_plan(&cost, n1, n2, cfg.sinkhorn_eps, cfg.sinkhorn_iters, &mut plan);
        pairs = transport::greedy_pairs(&plan, n1, n2);
        transport::procrustes_polar_into(&w1, &w2, &pairs, n1, k, &mut sc, &mut r);
        // mapped = w1 · r
        for i in 0..n1 {
            let src_start = i * k;
            let src: Vec<f32> = w1[src_start..src_start + k].to_vec();
            for b in 0..k {
                let mut acc = 0.0f32;
                for t in 0..k {
                    acc += src[t] * r[t * k + b];
                }
                mapped[src_start + b] = acc;
            }
        }
        // Alignment quality on this round's pseudo-pairs.
        let mut cos_sum = 0.0f32;
        for (i, &j) in pairs.iter().enumerate() {
            let a = &mapped[i * k..(i + 1) * k];
            let b = &w2[j * k..(j + 1) * k];
            let mut dot = 0.0f32;
            for t in 0..k {
                dot += a[t] * b[t];
            }
            cos_sum += dot;
        }
        alignment = cos_sum / n1 as f32;
        }
        if alignment > best_align {
            best_align = alignment;
            best_mapped.clear();
            best_mapped.extend_from_slice(&mapped);
            best_pairs.clear();
            best_pairs.extend_from_slice(&pairs);
        }
    }
    let (mapped, pairs, alignment) = (best_mapped, best_pairs, best_align);

    // Final scoring: kNN attribute transfer + neighborhood hit.
    let knn_k = cfg.knn_k.clamp(1, n2);
    let mut correct = 0usize;
    let mut hits = 0usize;
    let mut vote_buf: Vec<(u32, usize)> = Vec::with_capacity(knn_k);
    let mut top: Vec<u32> = Vec::with_capacity(knn_k);
    let mut sims: Vec<f32> = Vec::with_capacity(n2);
    for i in 0..n1 {
        let mi = &mapped[i * k..(i + 1) * k];
        sims.clear();
        for j in 0..n2 {
            let fj = &w2[j * k..(j + 1) * k];
            let mut dot = 0.0f32;
            for t in 0..k {
                dot += mi[t] * fj[t];
            }
            sims.push(dot);
        }
        top_k_indices(&sims, knn_k, &mut top);
        // Majority vote (ties → lower label id, deterministic).
        vote_buf.clear();
        for &j in top.iter() {
            let lbl = labels_foreign[j as usize];
            if let Some(e) = vote_buf.iter_mut().find(|(l, _)| *l == lbl) {
                e.1 += 1;
            } else {
                vote_buf.push((lbl, 1));
            }
            // Neighborhood-hit diagnostic: is the OT partner in the kNN set?
            if j as usize == pairs[i] {
                hits += 1;
            }
        }
        vote_buf.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        if vote_buf[0].0 == labels_ours[i] {
            correct += 1;
        }
    }
    let top1 = correct as f32 / n1 as f32;
    let chance = majority_frequency(labels_ours);
    let lift = top1 / chance.max(1e-6);
    let hit_rate = hits as f32 / n1 as f32;
    let hit_chance = knn_k as f32 / n2 as f32;

    let verdict = if alignment < cfg.min_alignment_cos {
        LeakVerdict::InsufficientAlignment
    } else if lift >= cfg.high_lift {
        LeakVerdict::High
    } else if lift >= cfg.elevated_lift {
        LeakVerdict::Elevated
    } else {
        LeakVerdict::Low
    };

    Ok(LeakReport {
        attribute_transfer_top1: top1,
        chance_baseline: chance,
        lift,
        alignment_mean_cos: alignment,
        neighborhood_hit_rate: hit_rate,
        neighborhood_hit_chance: hit_chance,
        verdict,
    })
}

/// Frequency of the most common label — the no-information baseline.
fn majority_frequency(labels: &[u32]) -> f32 {
    let mut counts: Vec<(u32, usize)> = Vec::new();
    for &l in labels {
        if let Some(e) = counts.iter_mut().find(|(c, _)| *c == l) {
            e.1 += 1;
        } else {
            counts.push((l, 1));
        }
    }
    counts
        .iter()
        .map(|&(_, c)| c)
        .max()
        .unwrap_or(0) as f32
        / labels.len().max(1) as f32
}

/// Indices of the `k` largest values, best-first (ties → lower index,
/// deterministic). Selection-insertion into `out` — no sort of the full
/// `n`.
fn top_k_indices(vals: &[f32], k: usize, out: &mut Vec<u32>) {
    out.clear();
    for (idx, &v) in vals.iter().enumerate() {
        let idx = idx as u32;
        if out.len() < k {
            // Insertion position (first slot whose value < v keeps ties at
            // earlier indices ranked first).
            let pos = out
                .iter()
                .position(|&o| vals[o as usize] < v)
                .unwrap_or(out.len());
            out.insert(pos, idx);
        } else if vals[out[out.len() - 1] as usize] < v {
            let pos = out
                .iter()
                .position(|&o| vals[o as usize] < v)
                .unwrap_or(out.len());
            out.insert(pos, idx);
            out.truncate(k);
        }
    }
}

/// Per-row mean similarity to its own top-`knn` nearest rows on the other
/// side — the CSLS hubness correction term `r_x` / `r_y`.
fn mean_topk_sims_rowwise(
    a: &[f32],
    b: &[f32],
    k: usize,
    knn: usize,
    sims: &mut Vec<f32>,
    top: &mut Vec<u32>,
) -> Vec<f32> {
    let na = a.len() / k;
    let nb = b.len() / k;
    let knn = knn.clamp(1, nb);
    let mut r = vec![0.0f32; na];
    for i in 0..na {
        let ai = &a[i * k..(i + 1) * k];
        sims.clear();
        for j in 0..nb {
            let bj = &b[j * k..(j + 1) * k];
            let mut dot = 0.0f32;
            for t in 0..k {
                dot += ai[t] * bj[t];
            }
            sims.push(dot);
        }
        top_k_indices(sims, knn, top);
        let s: f32 = top.iter().map(|&o| sims[o as usize]).sum();
        r[i] = s / knn as f32;
    }
    r
}

fn probe_err(e: transport::TransportError) -> LeakProbeError {
    match e {
        transport::TransportError::DimMismatch => LeakProbeError::DimMismatch,
        transport::TransportError::TooFewSamples => LeakProbeError::TooFewSamples,
        transport::TransportError::DimTooLarge => LeakProbeError::DimTooLarge,
        transport::TransportError::LatentDimTooLarge => LeakProbeError::LatentDimTooLarge,
        transport::TransportError::LatentDimExceedsSamples => LeakProbeError::LatentDimExceedsSamples,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test LCG mirroring the transport-internal stream (deterministic
    /// fixtures, no RNG dependency).
    struct Lcg(u64);
    impl Lcg {
        fn next_unit(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 40) as f32 / 8_388_608.0_f32) - 1.0
        }
    }

    /// Shared latent `z` (n × d_latent): 4 well-separated clusters
    /// (±u, ±v) with DISTINCT per-cluster spreads (0.15 / 0.3 / 0.45 /
    /// 0.6). Non-congruent clusters matter: congruent blobs are
    /// information-theoretically ambiguous under unpaired alignment (any
    /// cluster bijection is geometrically self-consistent — the same
    /// degeneracy that makes the paper's OT baselines fail), while distinct
    /// shapes anchor the correct correspondence. The attribute = cluster
    /// id — the honest fixture for LOCAL attribute structure (topic /
    /// item-category inference). Deterministic.
    fn make_latent(n: usize, d_latent: usize, seed: u64) -> (Vec<f32>, Vec<u32>) {
        const SCALES: [f32; 4] = [0.15, 0.3, 0.45, 0.6];

        let mut lcg = Lcg(seed);
        let u: Vec<f32> = (0..d_latent).map(|_| lcg.next_unit()).collect();
        let v: Vec<f32> = (0..d_latent).map(|_| lcg.next_unit()).collect();
        let mut z = vec![0.0f32; n * d_latent];
        let mut labels = vec![0u32; n];
        for i in 0..n {
            let c = (i % 4) as u32;
            labels[i] = c;
            let spread = SCALES[c as usize];
            for t in 0..d_latent {
                let center = match c {
                    0 => u[t],
                    1 => -u[t],
                    2 => v[t],
                    _ => -v[t],
                };
                z[i * d_latent + t] = center + spread * lcg.next_unit();
            }
        }
        (z, labels)
    }

    /// Observed space: x = z·A + σ·noise (n × d_out). Deterministic per
    /// seed.
    fn make_space(z: &[f32], n: usize, d_latent: usize, d_out: usize, seed: u64, sigma: f32) -> Vec<f32> {
        let mut lcg = Lcg(seed);
        let a: Vec<f32> = (0..d_latent * d_out).map(|_| lcg.next_unit()).collect();
        let mut x = vec![0.0f32; n * d_out];
        for i in 0..n {
            for c in 0..d_out {
                let mut acc = 0.0f32;
                for t in 0..d_latent {
                    acc += z[i * d_latent + t] * a[t * d_out + c];
                }
                x[i * d_out + c] = acc + sigma * lcg.next_unit();
            }
        }
        x
    }

    const N: usize = 256;
    const D_LATENT: usize = 8;

    #[test]
    fn g1_planted_leak_recovers_high() {
        let (z, labels) = make_latent(N, D_LATENT, 42);
        let x1 = make_space(&z, N, D_LATENT, 12, 101, 0.05);
        let x2 = make_space(&z, N, D_LATENT, 16, 202, 0.05);
        let cfg = LeakProbeConfig::default();
        let report = probe(&x1, &labels, 12, &x2, &labels, 16, &cfg).expect("probe");
        // Chance = 0.25 (4 balanced clusters): recovering ≥ 0.75 is a 3×
        // lift over no-information guessing.
        assert!(
            report.attribute_transfer_top1 >= 0.75,
            "top1 {} too low (lift {}, align {})",
            report.attribute_transfer_top1,
            report.lift,
            report.alignment_mean_cos
        );
        assert!(report.lift >= 2.0, "lift {} below planted bar", report.lift);
        assert_eq!(report.verdict, LeakVerdict::High, "{:?}", report.verdict);
    }

    #[test]
    fn g1b_transfer_is_monotone_in_observed_noise() {
        let (z, labels) = make_latent(N, D_LATENT, 42);
        let x1 = make_space(&z, N, D_LATENT, 12, 101, 0.0);
        let cfg = LeakProbeConfig::default();
        let mut prev = 1.0f32;
        let mut first = 0.0f32;
        for (round, &sigma) in [0.0f32, 0.5, 1.0, 2.0].iter().enumerate() {
            let x2 = make_space(&z, N, D_LATENT, 16, 202, sigma);
            let report = probe(&x1, &labels, 12, &x2, &labels, 16, &cfg).expect("probe");
            if round == 0 {
                first = report.attribute_transfer_top1;
            }
            assert!(
                report.attribute_transfer_top1 <= prev + 0.05,
                "σ={sigma}: top1 {} rose above prev {prev}",
                report.attribute_transfer_top1
            );
            prev = report.attribute_transfer_top1;
        }
        assert!(
            first - prev >= 0.1,
            "noise must cost accuracy: clean {first} vs noisy {prev}"
        );
        assert!(first >= 0.7, "clean-signal top1 {first} below planted bar");
    }

    /// The honest negative control: the foreign corpus carries NO instance
    /// of the attribute in its neighborhoods (iid random foreign labels —
    /// transport quality is irrelevant when the label semantics do not
    /// transfer). The probe must stay at chance.
    #[test]
    fn g1c_independent_spaces_stay_low() {
        let (z1, labels1) = make_latent(N, D_LATENT, 42);
        let (z2, _labels2_unused) = make_latent(N, D_LATENT, 777);
        let x1 = make_space(&z1, N, D_LATENT, 12, 101, 0.05);
        let x2 = make_space(&z2, N, D_LATENT, 16, 202, 0.05);
        // Foreign attribute labels: iid uniform over the 4 classes — no
        // shared ontology semantics with ours.
        let mut lcg = Lcg(313);
        let labels2: Vec<u32> = (0..N)
            .map(|_| {
                let q = ((lcg.next_unit() + 1.0) / 2.0) * 4.0;
                (q as u32).min(3)
            })
            .collect();
        let cfg = LeakProbeConfig::default();
        let report = probe(&x1, &labels1, 12, &x2, &labels2, 16, &cfg).expect("probe");
        assert!(
            report.lift <= 1.3,
            "iid foreign labels must stay near chance: lift {} (top1 {} vs chance {})",
            report.lift,
            report.attribute_transfer_top1,
            report.chance_baseline
        );
        assert_eq!(report.verdict, LeakVerdict::Low, "{:?}", report.verdict);
    }

    #[test]
    fn single_class_ours_labels_is_an_error() {
        let (z, _labels) = make_latent(64, D_LATENT, 42);
        let x1 = make_space(&z, 64, D_LATENT, 12, 101, 0.05);
        let x2 = make_space(&z, 64, D_LATENT, 16, 202, 0.05);
        let labels = vec![7u32; 64];
        let cfg = LeakProbeConfig::default();
        assert_eq!(
            probe(&x1, &labels, 12, &x2, &labels, 16, &cfg),
            Err(LeakProbeError::InsufficientLabelClasses)
        );
    }

    #[test]
    fn length_mismatches_are_errors() {
        let (z, labels) = make_latent(64, D_LATENT, 42);
        let x1 = make_space(&z, 64, D_LATENT, 12, 101, 0.05);
        let x2 = make_space(&z, 64, D_LATENT, 16, 202, 0.05);
        let cfg = LeakProbeConfig::default();
        // Truncated label vector.
        assert_eq!(
            probe(&x1, &labels[..63], 12, &x2, &labels, 16, &cfg),
            Err(LeakProbeError::DimMismatch)
        );
        // Buffer length not divisible by d.
        assert_eq!(
            probe(&x1[..x1.len() - 1], &labels, 12, &x2, &labels, 16, &cfg),
            Err(LeakProbeError::DimMismatch)
        );
    }

    #[test]
    fn g2_audit_cadence_smoke() {
        // 128 × 32 smoke — must complete comfortably inside the test
        // budget. The release-mode measurement belongs to the Issue 614 E2
        // adoption bench.
        let n = 128;
        let d = 32;
        let (z, labels) = make_latent(n, 8, 42);
        let x1 = make_space(&z, n, 8, d, 101, 0.05);
        let x2 = make_space(&z, n, 8, d, 202, 0.05);
        let cfg = LeakProbeConfig::default();
        let started = std::time::Instant::now();
        let report = probe(&x1, &labels, d, &x2, &labels, d, &cfg).expect("probe");
        let elapsed = started.elapsed();
        assert!(
            elapsed < std::time::Duration::from_secs(30),
            "probe took {elapsed:?} — pathological regression"
        );
        assert!(report.lift.is_finite());
    }
}
