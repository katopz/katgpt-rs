//! Plan 574 T4 — GOAT gate for the modelless clustered LM head.
//!
//! Run with:
//! ```text
//! cargo test --release --test bench_574_clustered_lm_head_goat -- --nocapture
//! ```
//!
//! # What is being gated
//!
//! `clustered_lm_head` prunes the vocabulary: stage 1 scores each cluster via a
//! centroid dot-product, stage 2 computes exact logits **only** for tokens in
//! the top-`k` clusters and fills the rest with `-inf`. So the load-bearing
//! question is not speed, it is **argmax recall** — does the token that the
//! full LM head would have picked survive the pruning?
//!
//! A speedup on a wrong argmax is not a modelless gain (AGENTS.md), so G2 is
//! the gate that decides promotion, not G3.
//!
//! # Why two data regimes
//!
//! Real LM-head rows carry geometric structure — semantically related tokens
//! have similar output embeddings, which is exactly what k-means exploits.
//! Testing only on structured data would flatter the primitive, so this bench
//! also runs a **pure-random control** where no cluster structure exists. There
//! k-means *should* do no better than round-robin, and reporting that honestly
//! is the point: it bounds the claim to "wins where structure exists".

use katgpt_rs::transformer::{
    cluster_classifier_from_map, cluster_map_from_embeddings, cluster_map_round_robin,
    clustered_lm_head, standard_lm_head,
};
use std::time::Instant;

/// BPE-ish vocabulary. Large enough for the pruning ratio to be meaningful,
/// small enough that the full sweep stays under a minute in release.
const VOCAB: usize = 32768;
/// Embedding width.
const N_EMBD: usize = 512;
/// Tokens per cluster ⇒ `VOCAB / 128 = 256` clusters. Mirrors Gemma 4's shipping
/// ratio (2048 centroids over a 262144 vocabulary, per Research 078).
const CLUSTER_SIZE: usize = 128;
/// Probe vectors per recall measurement.
const PROBES: usize = 200;

/// Deterministic LCG — reproducible across runs and platforms.
struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }
}

/// LM head with **planted cluster structure** — `n_groups` centres, each token a
/// small perturbation of its group's centre. Models the geometry real output
/// embeddings actually have.
fn structured_lm_head(n_groups: usize, seed: u64) -> Vec<f32> {
    let mut rng = Lcg(seed);
    let mut centres = vec![0.0f32; n_groups * N_EMBD];
    for slot in centres.iter_mut() {
        *slot = rng.next_f32();
    }
    let mut w = vec![0.0f32; VOCAB * N_EMBD];
    for t in 0..VOCAB {
        // Interleaved group assignment: consecutive token IDs land in DIFFERENT
        // groups, so round-robin (which partitions by ID) cannot accidentally
        // recover the structure. Without this the baseline would look strong
        // for the wrong reason.
        let g = t % n_groups;
        for j in 0..N_EMBD {
            w[t * N_EMBD + j] = centres[g * N_EMBD + j] + 0.05 * rng.next_f32();
        }
    }
    w
}

/// Control: no structure at all. K-means has nothing to find here.
fn random_lm_head(seed: u64) -> Vec<f32> {
    let mut rng = Lcg(seed);
    let mut w = vec![0.0f32; VOCAB * N_EMBD];
    for slot in w.iter_mut() {
        *slot = rng.next_f32();
    }
    w
}

fn probe_vectors(count: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Lcg(seed);
    (0..count)
        .map(|_| (0..N_EMBD).map(|_| rng.next_f32()).collect())
        .collect()
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| katgpt_core::float_order::cmp_for_max(**a, **b))
        .map_or(0, |(i, _)| i)
}

/// Fraction of probes whose true argmax survives cluster pruning, plus the
/// mean fraction of the vocabulary left active.
fn argmax_recall(
    lm_head: &[f32],
    map: &[Vec<usize>],
    classifier: &[f32],
    probes: &[Vec<f32>],
    truth: &[usize],
    topk: usize,
) -> (f64, f64) {
    let mut hits = 0usize;
    let mut active_total = 0usize;

    let mut got = vec![0.0f32; VOCAB];
    let mut scores = vec![0.0f32; map.len()];
    let (mut idx_buf, mut out_buf) = (Vec::new(), Vec::new());

    for (probe_idx, hidden) in probes.iter().enumerate() {
        clustered_lm_head(
            &mut got,
            hidden,
            lm_head,
            classifier,
            map,
            VOCAB,
            N_EMBD,
            topk,
            &mut scores,
            &mut idx_buf,
            &mut out_buf,
        );
        if argmax(&got) == truth[probe_idx] {
            hits += 1;
        }
        active_total += got.iter().filter(|v| v.is_finite()).count();
    }

    (
        hits as f64 / probes.len() as f64,
        active_total as f64 / (probes.len() * VOCAB) as f64,
    )
}

fn median_ms(samples: &mut [f64]) -> f64 {
    samples.sort_by(|a, b| a.total_cmp(b));
    samples[samples.len() / 2]
}

/// True argmax per probe, computed once.
///
/// This does not depend on `topk`, so computing it inside the sweep (as an
/// earlier revision did) repeated a full-vocabulary matmul for every one of the
/// swept `topk` values — the single largest cost in the bench, and pure waste.
fn true_argmaxes(lm_head: &[f32], probes: &[Vec<f32>]) -> Vec<usize> {
    let mut logits = vec![0.0f32; VOCAB];
    probes
        .iter()
        .map(|hidden| {
            standard_lm_head(&mut logits, hidden, lm_head, VOCAB, N_EMBD);
            argmax(&logits)
        })
        .collect()
}

/// One `(topk, recall, active_fraction)` row per swept `topk`.
///
/// Geometric rather than exhaustive: 13 points instead of 256 capture the shape
/// at ~20× less work.
fn sweep(
    lm_head: &[f32],
    map: &[Vec<usize>],
    classifier: &[f32],
    probes: &[Vec<f32>],
    truth: &[usize],
) -> Vec<(usize, f64, f64)> {
    [1usize, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256]
        .iter()
        .filter(|&&k| k <= map.len())
        .map(|&k| {
            let (recall, active) = argmax_recall(lm_head, map, classifier, probes, truth, k);
            (k, recall, active)
        })
        .collect()
}

/// Recall at the largest `topk` whose active fraction still fits `budget`.
///
/// Comparing at equal *active fraction* rather than equal `topk` is the only
/// fair comparison: k-means produces **uneven** clusters, so its top-`k` covers
/// far fewer tokens than round-robin's top-`k` (uniform 128-token clusters), and
/// comparing at equal `topk` silently hands round-robin several times the
/// compute.
///
/// Found by binary search, not a fixed grid. `active(topk)` is monotonically
/// non-decreasing, so bisection is exact in ~8 evaluations. Two earlier
/// revisions of this bench got the comparison wrong here — first by comparing at
/// equal `topk` (unequal compute), then by using a coarse geometric grid that
/// stepped straight over k-means' optimum near `topk=102`. Both errors moved the
/// verdict, so the resolution of this search is load-bearing.
fn recall_at_budget(
    lm_head: &[f32],
    map: &[Vec<usize>],
    classifier: &[f32],
    probes: &[Vec<f32>],
    truth: &[usize],
    budget: f64,
) -> (f64, f64, usize) {
    let (mut lo, mut hi) = (1usize, map.len());
    let mut best = (0.0f64, 0.0f64, 0usize);
    while lo <= hi {
        let mid = lo + (hi - lo) / 2;
        let (recall, active) = argmax_recall(lm_head, map, classifier, probes, truth, mid);
        if active <= budget {
            best = (recall, active, mid);
            lo = mid + 1;
        } else {
            match mid {
                0 => break,
                _ => hi = mid - 1,
            }
        }
    }
    best
}

fn run_regime(label: &str, lm_head: &[f32], probes: &[Vec<f32>]) -> (f64, f64) {
    println!("\n══ {label} ══");

    let t0 = Instant::now();
    let km_map = cluster_map_from_embeddings(lm_head, VOCAB, N_EMBD, CLUSTER_SIZE);
    let build_s = t0.elapsed().as_secs_f64();
    let km_cls = cluster_classifier_from_map(lm_head, &km_map, N_EMBD);

    let rr_map = cluster_map_round_robin(VOCAB, CLUSTER_SIZE);
    let rr_cls = cluster_classifier_from_map(lm_head, &rr_map, N_EMBD);

    println!(
        "build: {build_s:.2}s   k-means clusters {}   round-robin clusters {}",
        km_map.len(),
        rr_map.len()
    );
    let truth = true_argmaxes(lm_head, probes);
    let km_rows = sweep(lm_head, &km_map, &km_cls, probes, &truth);
    let rr_rows = sweep(lm_head, &rr_map, &rr_cls, probes, &truth);

    // Raw topk sweep — diagnosis only. NOT the comparison: the two methods
    // spend different compute at the same topk.
    println!("  -- raw topk sweep (unequal compute — diagnostic only) --");
    println!("  topk   km_recall  km_active%   rr_recall  rr_active%");
    for (&(k, km_r, km_a), &(_, rr_r, rr_a)) in km_rows.iter().zip(&rr_rows) {
        println!(
            "{k:6}   {km_r:9.4}  {:9.2}%   {rr_r:9.4}  {:9.2}%",
            km_a * 100.0,
            rr_a * 100.0
        );
    }

    // The actual gate: recall at matched compute budget.
    println!("  -- recall at MATCHED active budget (the fair comparison) --");
    println!(" budget   km_recall (topk)   rr_recall (topk)   winner");
    let mut best_km = 0.0f64;
    let mut best_rr = 0.0f64;
    for &budget in &[0.02f64, 0.05, 0.10, 0.25] {
        let (km_r, _, km_k) = recall_at_budget(lm_head, &km_map, &km_cls, probes, &truth, budget);
        let (rr_r, _, rr_k) = recall_at_budget(lm_head, &rr_map, &rr_cls, probes, &truth, budget);
        best_km = best_km.max(km_r);
        best_rr = best_rr.max(rr_r);
        let winner = match (km_r - rr_r).abs() < 1e-9 {
            true => "tie",
            false if km_r > rr_r => "kmeans",
            false => "round-robin",
        };
        println!(
            "{:5.0}%   {km_r:9.4} ({km_k:3})   {rr_r:9.4} ({rr_k:3})   {winner}",
            budget * 100.0
        );
    }
    (best_km, best_rr)
}

#[test]
fn goat_574_clustered_lm_head() {
    println!("Plan 574 T4 — Clustered LM Head GOAT");
    println!("vocab={VOCAB} n_embd={N_EMBD} cluster_size={CLUSTER_SIZE} probes={PROBES}");

    let probes = probe_vectors(PROBES, 0xC0FFEE);

    // ── G2 quality, structured regime (the claim) ──
    //
    // Two group counts, because the ratio matters for fairness. With 64 groups
    // against 256 clusters each group is split ~4 ways, capping topk=1 recall
    // near 25% by construction — a fixture artifact, not a property of the
    // primitive. `n_groups == num_clusters` is the favourable-but-fair case and
    // is what the verdict is drawn from.
    let n_clusters = VOCAB / CLUSTER_SIZE;
    let structured = structured_lm_head(n_clusters, 0xA11CE);
    let (km_s, rr_s) = run_regime(
        "STRUCTURED (groups == clusters — favourable case)",
        &structured,
        &probes,
    );
    let split = structured_lm_head(64, 0xA11CE);
    let _ = run_regime(
        "STRUCTURED (64 groups vs 256 clusters — split penalty)",
        &split,
        &probes,
    );

    // ── G2 control, unstructured regime (the honest bound) ──
    let random = random_lm_head(0xB0B);
    let (km_r, rr_r) = run_regime("RANDOM (no structure — control)", &random, &probes);

    // ── G3 perf: clustered vs standard at a recall-viable topk ──
    println!("\n══ G3 latency (structured) ══");
    let map = cluster_map_from_embeddings(&structured, VOCAB, N_EMBD, CLUSTER_SIZE);
    let cls = cluster_classifier_from_map(&structured, &map, N_EMBD);
    let hidden = &probes[0];
    let mut logits = vec![0.0f32; VOCAB];
    let mut scores = vec![0.0f32; map.len()];
    let (mut idx_buf, mut out_buf) = (Vec::new(), Vec::new());

    let mut std_samples = Vec::with_capacity(50);
    let mut clu_samples = Vec::with_capacity(50);
    for _ in 0..50 {
        let t = Instant::now();
        standard_lm_head(&mut logits, hidden, &structured, VOCAB, N_EMBD);
        std_samples.push(t.elapsed().as_secs_f64() * 1e3);

        let t = Instant::now();
        clustered_lm_head(
            &mut logits,
            hidden,
            &structured,
            &cls,
            &map,
            VOCAB,
            N_EMBD,
            32,
            &mut scores,
            &mut idx_buf,
            &mut out_buf,
        );
        clu_samples.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let std_ms = median_ms(&mut std_samples);
    let clu_ms = median_ms(&mut clu_samples);
    println!("standard  {std_ms:.4} ms\nclustered {clu_ms:.4} ms  (topk=32)");
    println!("speedup   {:.2}x", std_ms / clu_ms);

    // ── Verdict ──
    println!("\n══ VERDICT ══");
    println!("G2 structured: kmeans {km_s:.4} vs round-robin {rr_s:.4}");
    println!("G2 random ctl: kmeans {km_r:.4} vs round-robin {rr_r:.4}");

    // Plan 574 states TWO things under G2. They are reported separately
    // because they can — and do — disagree.
    const RECALL_TARGET: f64 = 0.99;
    let g2_relative = km_s > rr_s;
    let g2_absolute = km_s >= RECALL_TARGET;
    let g3 = std_ms > clu_ms;

    let verdict = |ok: bool| if ok { "PASS" } else { "FAIL" };
    println!(
        "G2a relative (kmeans > round-robin):          {}",
        verdict(g2_relative)
    );
    println!(
        "G2b absolute (recall >= {RECALL_TARGET}, best {km_s:.4}):    {}",
        verdict(g2_absolute)
    );
    println!(
        "G3  perf (clustered < standard):              {}",
        verdict(g3)
    );
    println!(
        "\nPROMOTION: {} — AGENTS.md requires the quality gate to pass modellessly;\n\
         a speedup on a wrong argmax is not a modelless gain.",
        if g2_absolute && g3 {
            "ELIGIBLE"
        } else {
            "BLOCKED (G2b)"
        }
    );

    // Recorded, not asserted: the control regime is expected to show no k-means
    // advantage. Asserting it would encode "must not win on noise", which is a
    // statement about the fixture, not the primitive.
    // Only the relative claim is asserted. G2b (absolute recall) is *reported*
    // rather than asserted so the bench stays green as a recorded negative
    // result — a permanently red test would be swept as noise instead of read
    // as the promotion blocker it is. Plan 574 T6 carries the FAIL.
    assert!(
        g2_relative,
        "G2a: k-means must beat round-robin on structured data \
         (got kmeans={km_s:.4}, round-robin={rr_s:.4})"
    );
}
