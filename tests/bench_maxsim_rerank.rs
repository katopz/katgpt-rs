//! MaxSim Reranking Benchmark (Plan 080 T12).
//!
//! Proves: MaxSim reranking ≥2% better NDCG@10 than cosine similarity.
//!
//! Run: cargo test --features maxsim --test bench_maxsim_rerank -- --nocapture

#![cfg(feature = "maxsim")]

use katgpt_rs::rerank::{RerankMethod, ndcg_at, rerank};
use katgpt_rs::types::Rng;

// ── Constants ─────────────────────────────────────────────────

const LQ: usize = 8;
const LD: usize = 16;
const DIM: usize = 64;
const N_DOCS: usize = 50;
const N_SIGNAL: usize = 10;
const N_QUERY_SIGNAL: usize = 4;
const N_HIGH: usize = 5;
const N_PARTIAL: usize = 15;
const K: usize = 10;
const N_TRIALS: usize = 100;

// ── Helpers ───────────────────────────────────────────────────

/// Generate orthogonal signal vectors using non-overlapping dimension blocks.
///
/// Each vector has a "hot" block of `dim/n` dimensions set to ~3.0, rest near
/// zero. This creates near-orthogonal vectors that MaxSim can match token-level.
fn make_signal_vectors(n: usize, dim: usize, rng: &mut Rng) -> Vec<Vec<f32>> {
    let block = dim / n;
    (0..n)
        .map(|i| {
            (0..dim)
                .map(|d| match d >= i * block && d < (i + 1) * block {
                    true => 3.0 + rng.normal() * 0.3,
                    false => rng.normal() * 0.05,
                })
                .collect()
        })
        .collect()
}

/// Build a signal token: base vector + additive Gaussian noise.
fn make_signal_token(signal: &[f32], noise_scale: f32, rng: &mut Rng) -> Vec<f32> {
    signal
        .iter()
        .map(|&s| s + rng.normal() * noise_scale)
        .collect()
}

/// Build a pure noise token.
fn make_noise_token(dim: usize, scale: f32, rng: &mut Rng) -> Vec<f32> {
    (0..dim).map(|_| rng.normal() * scale).collect()
}

/// Apply quantization noise: multiply each dimension by a random factor in [0.8, 1.2].
fn add_quant_noise(flat: &mut [f32], rng: &mut Rng) {
    for v in flat.iter_mut() {
        *v *= 0.8 + rng.uniform() * 0.4;
    }
}

/// Fisher-Yates shuffle in-place.
fn shuffle<T>(slice: &mut [T], rng: &mut Rng) {
    for i in (1..slice.len()).rev() {
        let j = (rng.next() as usize) % (i + 1);
        slice.swap(i, j);
    }
}

// ── Trial ─────────────────────────────────────────────────────

/// Run a single trial with deterministic seed. Returns `(cosine_ndcg, maxsim_ndcg)`.
fn run_trial(seed: u64) -> (f32, f32) {
    let mut rng = Rng::new(seed);

    // 1. Signal vectors: near-orthogonal via non-overlapping dimension blocks.
    let signals = make_signal_vectors(N_SIGNAL, DIM, &mut rng);

    // 2. Pick query signal IDs (4 of 10, shuffled).
    let mut ids: Vec<usize> = (0..N_SIGNAL).collect();
    shuffle(&mut ids, &mut rng);
    let q_sig: Vec<usize> = ids[..N_QUERY_SIGNAL].to_vec();

    // 3. Build query: 4 signal tokens + 4 noise tokens.
    let mut query = Vec::with_capacity(LQ * DIM);
    for i in 0..N_QUERY_SIGNAL {
        query.extend(make_signal_token(&signals[q_sig[i]], 0.3, &mut rng));
    }
    for _ in N_QUERY_SIGNAL..LQ {
        query.extend(make_noise_token(DIM, 0.5, &mut rng));
    }
    add_quant_noise(&mut query, &mut rng);

    // 4. Document plan: (n_matching_signals, relevance_score).
    let n_irrel = N_DOCS - N_HIGH - N_PARTIAL;
    let mut plan: Vec<(usize, f32)> = Vec::with_capacity(N_DOCS);
    for _ in 0..N_HIGH {
        plan.push((4, 3.0));
    }
    for _ in 0..N_PARTIAL {
        let n_match = 2 + (rng.next() as usize) % 2; // 2 or 3
        plan.push((n_match, 1.5));
    }
    for _ in 0..n_irrel {
        plan.push((0, 0.0));
    }
    shuffle(&mut plan, &mut rng);

    // 5. Build documents from plan.
    let mut docs: Vec<Vec<f32>> = Vec::with_capacity(N_DOCS);
    let doc_lengths = vec![LD; N_DOCS];
    let mut ground_truth: Vec<f32> = Vec::with_capacity(N_DOCS);

    for &(n_match, rel) in &plan {
        let mut doc = Vec::with_capacity(LD * DIM);

        // Matching signal tokens.
        for t in 0..n_match {
            let sig_id = q_sig[t % N_QUERY_SIGNAL];
            doc.extend(make_signal_token(&signals[sig_id], 0.3, &mut rng));
        }

        // Remaining: noise + occasional distractor signal (non-query).
        for _ in n_match..LD {
            match rng.next() % 4 {
                0 => {
                    let mut others: Vec<usize> =
                        (0..N_SIGNAL).filter(|id| !q_sig.contains(id)).collect();
                    shuffle(&mut others, &mut rng);
                    doc.extend(make_signal_token(&signals[others[0]], 0.3, &mut rng));
                }
                _ => {
                    doc.extend(make_noise_token(DIM, 0.5, &mut rng));
                }
            }
        }

        add_quant_noise(&mut doc, &mut rng);
        docs.push(doc);
        ground_truth.push(rel);
    }

    // 6. Rerank with both methods.
    let cosine_ranked = rerank(&query, &docs, &doc_lengths, DIM, RerankMethod::Cosine);
    let maxsim_ranked = rerank(&query, &docs, &doc_lengths, DIM, RerankMethod::MaxSim);

    let cosine_ndcg = ndcg_at(&cosine_ranked, &ground_truth, K);
    let maxsim_ndcg = ndcg_at(&maxsim_ranked, &ground_truth, K);

    (cosine_ndcg, maxsim_ndcg)
}

// ── GOAT Gate Test ────────────────────────────────────────────

#[test]
fn bench_maxsim_rerank_ndcg() {
    let mut cosine_sum = 0.0f64;
    let mut maxsim_sum = 0.0f64;

    for trial in 0..N_TRIALS {
        let seed = 1000 + trial as u64;
        let (cos, ms) = run_trial(seed);
        cosine_sum += cos as f64;
        maxsim_sum += ms as f64;
    }

    let mean_cosine = cosine_sum / N_TRIALS as f64;
    let mean_maxsim = maxsim_sum / N_TRIALS as f64;
    let pct = (mean_maxsim - mean_cosine) / mean_cosine * 100.0;
    let n_irrel = N_DOCS - N_HIGH - N_PARTIAL;

    println!();
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  MaxSim Reranking Benchmark — Plan 080 T12               ║");
    println!("╠════════════════════════════════════════════════════════════╣");
    println!("║  Config: {N_TRIALS} trials × {N_DOCS} docs, Lq={LQ} Ld={LD} dim={DIM}");
    println!("║  Tiers: {N_HIGH} high + {N_PARTIAL} partial + {n_irrel} irrelevant");
    println!("║───────────────────────────────────────────────────────────║");
    println!("║  Method       │  Mean NDCG@{K}           ║");
    println!("║  Cosine       │  {mean_cosine:.4}                  ║");
    println!("║  MaxSim       │  {mean_maxsim:.4}                  ║");
    println!("║  Improvement  │  {pct:+.2}%                  ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();

    // GOAT gate: MaxSim ≥ 2% better than cosine.
    let threshold = mean_cosine * 1.02;
    assert!(
        mean_maxsim >= threshold,
        "GOAT gate failed: MaxSim NDCG@10 = {mean_maxsim:.4} \
         < Cosine {mean_cosine:.4} × 1.02 = {threshold:.4}"
    );
}
