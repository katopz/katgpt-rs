//! DatrieVocab Benchmark — HashMap vs Double-Array Trie (Research 137, T3).
//!
//! Compares `HashMap<Vec<u8>, usize>` vs `DatrieVocab` on realistic ToaST
//! workloads across multiple vocab sizes and input lengths.
//!
//! ```sh
//! cargo run --example datrie_01_bench --features datrie_vocab --release
//! ```

use std::collections::HashMap;
use std::hint::black_box;
use std::time::Instant;

use katgpt_rs::tokenizer::{DatrieTreeIndex, DatrieVocab, SplitNode, SplitTree};

// ── Config ─────────────────────────────────────────────────────

/// Vocab sizes to benchmark.
const VOCAB_SIZES: &[usize] = &[512, 4_096, 32_768, 131_072];

/// Input lengths (in bytes of text to encode).
const INPUT_LENGTHS: &[usize] = &[128, 512, 1024, 4096];

const LOOKUP_BATCH: usize = 1000;

/// Number of lookup iterations for steady-state measurement.
const LOOKUP_ITERS: usize = 10_000;

/// Number of encode iterations (fewer because each is many lookups).
const ENCODE_ITERS: usize = 1_000;

// ── Helpers ────────────────────────────────────────────────────

/// Simple deterministic PRNG for reproducible data generation.
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Generate a synthetic vocabulary of `n` tokens.
///
/// Mix of token patterns:
/// - Single bytes (0..256)
/// - Short ASCII words ("token_XXXX")
/// - UTF-8-like multi-byte sequences
fn make_vocab(n: usize, _seed: u64) -> HashMap<Vec<u8>, usize> {
    let mut vocab = HashMap::with_capacity(n);

    // Always include all single bytes.
    for b in 0u8..=255u8 {
        vocab.insert(vec![b], vocab.len());
    }

    // Add short tokens: "tok_XXXX"
    let mut i = 0u32;
    while vocab.len() < n {
        let key = format!("tok_{i:08x}").into_bytes();
        if !vocab.contains_key(&key) {
            vocab.insert(key, vocab.len());
        }
        i += 1;
    }

    vocab
}

/// Generate a random input text of `len` bytes using vocab keys.
fn make_input(vocab: &HashMap<Vec<u8>, usize>, len: usize, seed: u64) -> Vec<u8> {
    let mut rng = seed;
    let keys: Vec<&Vec<u8>> = vocab.keys().collect();
    let mut buf = Vec::with_capacity(len);

    while buf.len() < len {
        let idx = (xorshift64(&mut rng) as usize) % keys.len();
        buf.extend_from_slice(keys[idx]);
    }
    buf.truncate(len);
    buf
}

/// Generate a pretoken→tree map for tree index benchmarks.
fn make_trees(n: usize, seed: u64) -> HashMap<Vec<u8>, SplitTree> {
    let mut rng = seed;
    let mut trees = HashMap::with_capacity(n);

    for i in 0..n {
        let pretoken = format!("pretoken_{i:06x}").into_bytes();
        let tree = SplitTree {
            pretoken: pretoken.clone(),
            nodes: vec![SplitNode {
                start: 0,
                end: pretoken.len().min(u16::MAX as usize) as u16,
                left: None,
                right: None,
            }],
        };
        trees.insert(pretoken, tree);
        rng = rng.wrapping_add(1);
    }
    trees
}

/// Statistical summary of timing samples.
struct Stats {
    p50: f64,
    p99: f64,
    #[allow(dead_code)]
    mean: f64,
    #[allow(dead_code)]
    min: f64,
}

fn compute_stats(samples: &[f64]) -> Stats {
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    Stats {
        p50: sorted[n / 2],
        p99: sorted[(n as f64 * 0.99) as usize].min(sorted[n - 1]),
        mean: sorted.iter().sum::<f64>() / n as f64,
        min: sorted[0],
    }
}

// ════════════════════════════════════════════════════════════════
//  B1: Single-Lookup Latency
// ════════════════════════════════════════════════════════════════

fn bench_lookup_latency() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  B1: Single-Lookup Latency (HashMap vs DatrieVocab)");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    println!(
        "  {:>8} {:>14} {:>14} {:>14} {:>14} {:>10} {:>8}",
        "Vocab", "HM p50 ns/1K", "HM p99 ns/1K", "DA p50 ns/1K", "DA p99 ns/1K", "Speedup", "Ratio"
    );
    println!(
        "  ────────  ──────────────  ──────────────  ──────────────  ──────────────  ──────────  ────────"
    );

    for &vocab_size in VOCAB_SIZES {
        let vocab = make_vocab(vocab_size, 42);
        let datrie = DatrieVocab::build(&vocab);

        // Pick a mix of keys to look up (some hits, some misses).
        let keys: Vec<Vec<u8>> = vocab.keys().take(100).cloned().collect();
        let miss_keys: Vec<Vec<u8>> = (0..50)
            .map(|i| format!("missing_key_{i}").into_bytes())
            .collect();
        let all_keys: Vec<&[u8]> = keys
            .iter()
            .chain(miss_keys.iter())
            .map(|k| k.as_slice())
            .collect();

        // Warmup
        for _ in 0..1000 {
            for key in &all_keys {
                black_box(vocab.get(*key));
                black_box(datrie.lookup(*key));
            }
        }

        // HashMap benchmark — batch of LOOKUP_BATCH lookups per sample
        let mut hm_samples = Vec::with_capacity(LOOKUP_ITERS);
        for _ in 0..LOOKUP_ITERS {
            let start = Instant::now();
            for i in 0..LOOKUP_BATCH {
                let key = all_keys[i % all_keys.len()];
                black_box(vocab.get(key));
            }
            hm_samples.push(start.elapsed().as_nanos() as f64);
        }

        // DatrieVocab benchmark — same batch size
        let mut da_samples = Vec::with_capacity(LOOKUP_ITERS);
        for _ in 0..LOOKUP_ITERS {
            let start = Instant::now();
            for i in 0..LOOKUP_BATCH {
                let key = all_keys[i % all_keys.len()];
                black_box(datrie.lookup(key));
            }
            da_samples.push(start.elapsed().as_nanos() as f64);
        }

        let hm = compute_stats(&hm_samples);
        let da = compute_stats(&da_samples);
        let speedup = hm.p50 / da.p50;
        let ratio = if da.p50 < hm.p50 {
            "✓ DA win"
        } else {
            "✗ HM win"
        };

        println!(
            "  {:>8} {:>14.1} {:>14.1} {:>14.1} {:>14.1} {:>9.2}×  {:>8}",
            vocab_size, hm.p50, hm.p99, da.p50, da.p99, speedup, ratio
        );
    }
    println!();
}

// ════════════════════════════════════════════════════════════════
//  B2: Encode Throughput (full text tokenization)
// ════════════════════════════════════════════════════════════════

fn bench_encode_throughput() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  B2: Encode Throughput — vocab=32K, varying input length");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let vocab = make_vocab(32_768, 42);
    let datrie = DatrieVocab::build(&vocab);

    println!(
        "  {:>6} {:>12} {:>12} {:>10} {:>12} {:>8}",
        "Bytes", "HM µs/enc", "DA µs/enc", "Speedup", "HM MB/s", "DA MB/s"
    );
    println!("  ──────  ────────────  ────────────  ──────────  ────────────  ────────");

    for &input_len in INPUT_LENGTHS {
        let input = make_input(&vocab, input_len, 123);

        // Simulate ToaST-style encode: walk input, longest-prefix match at each position.
        // HashMap version
        let mut hm_samples = Vec::with_capacity(ENCODE_ITERS);
        for _ in 0..ENCODE_ITERS {
            let start = Instant::now();
            let mut pos = 0;
            while pos < input.len() {
                // Longest prefix match (brute force — how HashMap is typically used)
                let mut best_len = 1;
                for end in (pos + 1..=input.len()).rev().take(20) {
                    if vocab.get(&input[pos..end]).is_some() {
                        best_len = end - pos;
                        break;
                    }
                }
                black_box(best_len);
                pos += best_len;
            }
            hm_samples.push(start.elapsed().as_micros() as f64);
        }

        // DatrieVocab version — uses built-in longest_prefix
        let mut da_samples = Vec::with_capacity(ENCODE_ITERS);
        for _ in 0..ENCODE_ITERS {
            let start = Instant::now();
            let mut pos = 0;
            while pos < input.len() {
                if let Some((_id, end)) = datrie.longest_prefix(&input, pos) {
                    pos = end;
                } else {
                    pos += 1; // fallback byte
                }
            }
            da_samples.push(start.elapsed().as_micros() as f64);
        }

        let hm = compute_stats(&hm_samples);
        let da = compute_stats(&da_samples);
        let speedup = hm.p50 / da.p50;
        let hm_mbps = (input_len as f64 / 1_048_576.0) / (hm.p50 / 1_000_000.0);
        let da_mbps = (input_len as f64 / 1_048_576.0) / (da.p50 / 1_000_000.0);

        println!(
            "  {:>6} {:>12.1} {:>12.1} {:>9.2}× {:>12.1} {:>8.1}",
            input_len, hm.p50, da.p50, speedup, hm_mbps, da_mbps
        );
    }
    println!();
}

// ════════════════════════════════════════════════════════════════
//  B3: Build Time & Memory
// ════════════════════════════════════════════════════════════════

fn bench_build_time() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  B3: Build Time & Memory Footprint");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    println!(
        "  {:>8} {:>12} {:>12} {:>12} {:>8}",
        "Vocab", "HM build µs", "DA build µs", "DA arrays KB", "Build×"
    );
    println!("  ────────  ────────────  ────────────  ────────────  ────────");

    for &vocab_size in VOCAB_SIZES {
        let vocab = make_vocab(vocab_size, 42);

        // HashMap "build" — just clone (represents loading from serialized)
        let hm_start = Instant::now();
        let _hm: HashMap<Vec<u8>, usize> = vocab.clone();
        let hm_us = hm_start.elapsed().as_micros();

        // Datrie build
        let da_start = Instant::now();
        let datrie = DatrieVocab::build(&vocab);
        let da_us = da_start.elapsed().as_micros();

        // Estimate memory: HashMap ≈ (key.len() + 16) * n + overhead
        //                  Datrie ≈ (4 + 4 + 4) * array_len (base + check + value)
        let da_kb = datrie.inner_bytes() as f64 / 1024.0;
        let build_ratio = da_us as f64 / hm_us.max(1) as f64;

        println!(
            "  {:>8} {:>12} {:>12} {:>12.1} {:>7.1}×",
            vocab_size, hm_us, da_us, da_kb, build_ratio
        );
    }
    println!();
}

// ════════════════════════════════════════════════════════════════
//  B4: DatrieTreeIndex vs HashMap<Vec<u8>, SplitTree>
// ════════════════════════════════════════════════════════════════

fn bench_tree_index() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  B4: Pretoken→Tree Lookup (HashMap vs DatrieTreeIndex)");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let tree_counts: &[usize] = &[512, 4_096, 32_768];

    println!(
        "  {:>8} {:>16} {:>16} {:>10} {:>8}",
        "Trees", "HM p50 ns/1K", "DA p50 ns/1K", "Speedup", "Ratio"
    );
    println!("  ────────  ────────────────  ────────────────  ──────────  ────────");

    for &n in tree_counts {
        let trees = make_trees(n, 77);
        let lookup_keys: Vec<Vec<u8>> = trees.keys().take(50).cloned().collect();

        // Warmup
        for _ in 0..1000 {
            for key in &lookup_keys {
                black_box(trees.get(key.as_slice()));
            }
        }

        // HashMap lookup — batch
        let mut hm_samples = Vec::with_capacity(LOOKUP_ITERS);
        for _ in 0..LOOKUP_ITERS {
            let start = Instant::now();
            for i in 0..LOOKUP_BATCH {
                let key = &lookup_keys[i % lookup_keys.len()];
                black_box(trees.get(key.as_slice()));
            }
            hm_samples.push(start.elapsed().as_nanos() as f64);
        }

        // DatrieTreeIndex lookup — batch
        let datrie = DatrieTreeIndex::build(trees);
        let mut da_samples = Vec::with_capacity(LOOKUP_ITERS);
        for _ in 0..LOOKUP_ITERS {
            let start = Instant::now();
            for i in 0..LOOKUP_BATCH {
                let key = &lookup_keys[i % lookup_keys.len()];
                black_box(datrie.lookup(key));
            }
            da_samples.push(start.elapsed().as_nanos() as f64);
        }

        let hm = compute_stats(&hm_samples);
        let da = compute_stats(&da_samples);
        let speedup = hm.p50 / da.p50;
        let ratio = if da.p50 < hm.p50 { "✓ DA" } else { "✗ HM" };

        println!(
            "  {:>8} {:>16.1} {:>16.1} {:>9.2}×  {:>8}",
            n, hm.p50, da.p50, speedup, ratio
        );
    }
    println!();
}

// ════════════════════════════════════════════════════════════════
//  B5: Correctness Verification
// ════════════════════════════════════════════════════════════════

fn verify_correctness() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  B5: Correctness — Datrie matches HashMap on all keys");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    for &vocab_size in VOCAB_SIZES {
        let vocab = make_vocab(vocab_size, 42);
        let datrie = DatrieVocab::build(&vocab);

        let mut mismatches = 0u64;
        let mut misses = 0u64;

        for (key, &expected_id) in &vocab {
            match datrie.lookup(key) {
                Some(id) if id == expected_id => {}
                Some(id) => {
                    mismatches += 1;
                    if mismatches <= 3 {
                        eprintln!(
                            "  MISMATCH: key={:?} expected={expected_id} got={id}",
                            &key[..key.len().min(8)]
                        );
                    }
                }
                None => {
                    misses += 1;
                    if misses <= 3 {
                        eprintln!(
                            "  MISS: key={:?} expected={expected_id}",
                            &key[..key.len().min(8)]
                        );
                    }
                }
            }
        }

        // Also test some non-keys
        let mut false_positives = 0u64;
        for i in 0..1000u32 {
            let key = format!("nonexistent_{i}").into_bytes();
            if datrie.lookup(&key).is_some() {
                false_positives += 1;
            }
        }

        let status = if mismatches == 0 && misses == 0 && false_positives == 0 {
            "✓"
        } else {
            "✗"
        };
        println!(
            "  vocab={vocab_size:>8}: {} keys checked, {mismatches} mismatches, {misses} misses, {false_positives} false positives {status}",
            vocab.len()
        );
    }
    println!();
}

// ════════════════════════════════════════════════════════════════
//  Main
// ════════════════════════════════════════════════════════════════

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  DatrieVocab Benchmark — Research 137, Task T3");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    verify_correctness();
    bench_lookup_latency();
    bench_encode_throughput();
    bench_build_time();
    bench_tree_index();

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Benchmark Complete");
    println!("═══════════════════════════════════════════════════════════════");
}
