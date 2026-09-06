//! GOAT gate for `FastBpeEncoder::encode_into_pretok` (Issue 191 Phase 2.6 —
//! whitespace pretokenization + per-pretoken cache).
//!
//! # Gates
//!
//! - **G1 (correctness)**: `encode_into_pretok` produces bit-identical token
//!   IDs to `encode` / `encode_into`. The invariant is structural: the
//!   trainer's `split_whitespace()` means no merge rule crosses whitespace,
//!   so per-pretoken encode == whole-text encode. See
//!   `tests/fast_bpe_pretok_hypothesis.rs` for the standalone regression
//!   guard on that invariant.
//! - **G2 (perf)**: `encode_into_pretok` is faster than `encode_into` on
//!   natural-language inputs. Two wins compound:
//!   1. Structural: sum of O(k log k) per pretoken vs O(n log n) whole-text.
//!   2. Cache-hit: repeated words skip the merge loop entirely.
//!
//! # Honest scope note
//!
//! This is the FIRST pretokenized path in katgpt-tokenizer. It uses a plain
//! `HashMap<Vec<u8>, Vec<TokenId>>` cache — correct but not the vendored
//! `ShortPretokenCache` substrate (open-addressed + prefetched + 2 MiB-
//! aligned). The HashMap captures the structural + cache-hit win; the
//! cache-hierarchy optimization is a follow-up. The honest gain target
//! here is "faster than `encode_into` on natural language", NOT the
//! upstream gigatoken 1000× (which needs SIMD pretokenization + the full
//! cache hierarchy + ~99% hit rate at corpus scale).

#![cfg(feature = "fast_bpe")]

use katgpt_tokenizer::{BpeTokenizerImpl, BpeTrainer, FastBpeEncoder};

// ---------------------------------------------------------------------------
// G1 — bit-identical to encode / encode_into
// ---------------------------------------------------------------------------

#[test]
fn g1_pretok_bit_identical_to_encode_short_texts() {
    let corpus = "the cat sat on the mat the cat the mat the test hello world the test split \
                  the cat the mat the test hello world the cat the mat the test";
    let tokenizer = BpeTrainer::train(corpus, 64);
    let texts = [
        "hello",
        "the cat",
        "the cat sat on the mat",
        "world hello test",
        "xyzzy",
        "",
        "the the the the the the the the the the",
        "ca",
        "  leading spaces",
        "trailing spaces  ",
        "multiple   internal   spaces",
        "\ttab\tseparated\twords",
        "newline\nseparated\nlines",
        "mixed\t whitespace\n with various  separators",
    ];
    let mut encoder = FastBpeEncoder::from_tokenizer(&tokenizer);
    let mut out_pretok = Vec::new();
    let mut out_plain = Vec::new();
    for text in &texts {
        let reference = BpeTokenizerImpl::encode(&tokenizer, text);
        encoder.encode_into(text, &mut out_plain);
        encoder.encode_into_pretok(text, &mut out_pretok);
        assert_eq!(
            out_pretok, reference,
            "G1 pretok divergence vs encode on text={text:?}\n  pretok={out_pretok:?}\n  encode={reference:?}"
        );
        assert_eq!(
            out_pretok, out_plain,
            "G1 pretok divergence vs encode_into on text={text:?}"
        );
    }
}

#[test]
fn g1_pretok_bit_identical_on_code_like_text() {
    // Code-like text with punctuation — exercises the pretokenizer on
    // non-letter chars (which are NOT whitespace, so they accumulate into
    // pretokens along with adjacent letters).
    let corpus = "fn foo bar baz qux quux corge grault garply waldo fred plugh xyzzy \
                  fn foo bar baz qux quux corge grault garply waldo fred plugh xyzzy";
    let tokenizer = BpeTrainer::train(corpus, 256);
    let texts = [
        "fn foo() -> usize { 42 }",
        "use std::collections::HashMap;",
        "struct PairRankTable { dense: Box<[u32]> }",
        "let x = foo.bar.baz(qux, quux);",
        "/* comment */ fn body() { return corge; }",
    ];
    let mut encoder = FastBpeEncoder::from_tokenizer(&tokenizer);
    let mut out_pretok = Vec::new();
    for text in &texts {
        let reference = BpeTokenizerImpl::encode(&tokenizer, text);
        encoder.encode_into_pretok(text, &mut out_pretok);
        assert_eq!(
            out_pretok, reference,
            "G1 pretok divergence on code-like text (len={}): first diff at {}",
            text.len(),
            out_pretok
                .iter()
                .zip(reference.iter())
                .position(|(a, b)| a != b)
                .unwrap_or(out_pretok.len())
        );
    }
}

#[test]
fn g1_pretok_bit_identical_on_repeated_corpus() {
    // The structural correctness test: encode the training corpus itself.
    // The pretokenized path sees every word the trainer learned merges for,
    // and must produce the same merged output as whole-text encode.
    let corpus = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu \
                  alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu \
                  alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu \
                  nu xi omicron pi rho sigma tau upsilon phi chi psi omega \
                  nu xi omicron pi rho sigma tau upsilon phi chi psi omega";
    let tokenizer = BpeTrainer::train(corpus, 256);
    let mut encoder = FastBpeEncoder::from_tokenizer(&tokenizer);
    let mut out_pretok = Vec::new();
    let reference = BpeTokenizerImpl::encode(&tokenizer, corpus);
    encoder.encode_into_pretok(corpus, &mut out_pretok);
    assert_eq!(
        out_pretok, reference,
        "G1 pretok divergence on repeated corpus ({} chars)",
        corpus.len()
    );
    // The cache should have an entry per unique word.
    assert!(
        encoder.pretoken_cache_len() > 0,
        "Pretoken cache empty after encode — cache wiring broken"
    );
}

// ---------------------------------------------------------------------------
// G2 — perf: pretokenized path vs whole-text path
// ---------------------------------------------------------------------------

#[test]
fn g2_pretok_faster_than_whole_text_on_natural_language() {
    // The headline perf test. The pretokenized path should beat the whole-
    // text path on natural language because:
    //   1. Each pretoken is encoded with a small heap (structural win).
    //   2. Repeated words hit the cache (cache win).
    //
    // The corpus has high word repetition (each greek letter appears 3×),
    // so the cache hit rate climbs fast. After the first iteration the
    // cache is warm and subsequent iterations are mostly hits.
    let corpus = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu \
                  alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu \
                  alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu \
                  nu xi omicron pi rho sigma tau upsilon phi chi psi omega \
                  nu xi omicron pi rho sigma tau upsilon phi chi psi omega \
                  alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu";
    let tokenizer = BpeTrainer::train(corpus, 256);

    let n = 200;

    // Whole-text path (no pretokenization, no cache).
    let mut encoder_plain = FastBpeEncoder::from_tokenizer(&tokenizer);
    let mut out = Vec::new();
    let t_plain = std::time::Instant::now();
    for _ in 0..n {
        encoder_plain.encode_into(corpus, &mut out);
    }
    let plain_ns = t_plain.elapsed().as_nanos();

    // Pretokenized path (with cache). The first iteration is cold; the rest
    // are warm. We measure the total — that's the honest comparison for a
    // realistic use case (encode the same corpus many times, e.g. training
    // data loading).
    let mut encoder_pretok = FastBpeEncoder::from_tokenizer(&tokenizer);
    let t_pretok = std::time::Instant::now();
    for _ in 0..n {
        encoder_pretok.encode_into_pretok(corpus, &mut out);
    }
    let pretok_ns = t_pretok.elapsed().as_nanos();

    let speedup = plain_ns as f64 / pretok_ns as f64;
    eprintln!(
        "g2 pretok vs whole-text (corpus {} chars, {} iters): plain={plain_ns}ns pretok={pretok_ns}ns speedup={speedup:.2}x (cache entries={})",
        corpus.len(),
        n,
        encoder_pretok.pretoken_cache_len()
    );
    // Honest floor: ≥1.2× speedup. The pretokenized path's win depends on
    // cache hit rate + structural win; both compound but neither is huge
    // on this small corpus. A higher floor would be brittle on slow CI.
    assert!(
        speedup > 1.2,
        "G2 pretok speedup not observed: {speedup:.2}x (gate >1.2x). Cache entries: {}",
        encoder_pretok.pretoken_cache_len()
    );
}

#[test]
fn g2_pretok_cache_warm_vs_cold() {
    // Verify the cache actually pays off: the warm iteration should be
    // substantially faster than the cold one.
    let corpus = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu \
                  alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu \
                  alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu";
    let tokenizer = BpeTrainer::train(corpus, 256);

    let mut encoder = FastBpeEncoder::from_tokenizer(&tokenizer);
    let mut out = Vec::new();

    // Cold: first encode, populates the cache.
    let t_cold = std::time::Instant::now();
    encoder.encode_into_pretok(corpus, &mut out);
    let cold_ns = t_cold.elapsed().as_nanos();
    let cold_cache_len = encoder.pretoken_cache_len();

    // Warm: subsequent encodes, mostly cache hits.
    let n_warm = 100;
    let t_warm = std::time::Instant::now();
    for _ in 0..n_warm {
        encoder.encode_into_pretok(corpus, &mut out);
    }
    let warm_ns = t_warm.elapsed().as_nanos() / n_warm as u128;

    eprintln!(
        "g2 pretok cold vs warm: cold={cold_ns}ns (populated {cold_cache_len} cache entries), warm={warm_ns}ns/iter"
    );
    // Warm should be faster than cold (the cache populates on cold, hits on warm).
    assert!(
        warm_ns < cold_ns,
        "Warm pretok ({warm_ns}ns) not faster than cold ({cold_ns}ns) — cache not paying off"
    );
}

// ---------------------------------------------------------------------------
// G2 (characterization) — corpus-scale scaling curve
// ---------------------------------------------------------------------------
//
// Ignored by default (run with `-- --ignored`). Characterizes how the pretok
// speedup scales with corpus size, to inform Issue 191 Phase 3 trigger #3
// ("ShortPretokenCache wired AND measured gain on corpus-scale benchmark
// exceeds 10×"). This test answers the question: does the structural +
// HashMap-cache win ALONE (no ShortPretokenCache, no SIMD pretokenization)
// reach 10× at corpus scale? If yes, Phase 3 trigger #3's gain half is met
// and only the ShortPretokenCache wiring half remains. If no, the 10×
// trigger is unreachable without SIMD regex pretokenization (out of scope
// for Issue 191) and Phase 3 deferral is honest with evidence.
//
// Synthetic Zipfian corpus: ~1000 unique words, frequency ∝ 1/rank, mixed
// word lengths (3–10 chars). Gives a realistic cache hit rate (~50% of
// tokens come from the top ~100 words) without a real-text dependency.
// Trained at vocab=1024 so the trainer actually learns merges (the
// structural win only materializes when merges exist).

/// Build a synthetic corpus of approximately `target_chars` characters by
/// sampling words from a Zipfian vocabulary until the target length is hit.
/// Words are `word_NN` with N varying, so lengths span 6–9 chars and the
/// vocabulary is bounded by `vocab_size`.
fn build_zipfian_corpus(target_chars: usize, vocab_size: usize) -> String {
    use std::cell::RefCell;
    // Per-test-invocation LCG state — deterministic per (vocab_size, target_chars)
    // pair so re-runs reproduce. Seed mixes both params.
    let seed = (vocab_size as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ ((target_chars as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(13));
    let state = RefCell::new(seed);
    let next_rand = || {
        let mut s = state.borrow_mut();
        // xorshift64*
        *s ^= *s >> 12;
        *s ^= *s << 25;
        *s ^= *s >> 27;
        (*s).wrapping_mul(0x2545_F491_4F6C_DD1D)
    };
    // Build the vocabulary.
    let words: Vec<String> = (0..vocab_size).map(|i| format!("word{i}")).collect();
    // Precompute the cumulative Zipfian CDF for fast sampling. Frequency
    // ∝ 1/rank; normalize so the total is 1.0.
    let harmonic: f64 = (1..=vocab_size).map(|r| 1.0 / r as f64).sum();
    let mut cdf = Vec::with_capacity(vocab_size);
    let mut acc = 0.0_f64;
    for r in 1..=vocab_size {
        acc += (1.0 / r as f64) / harmonic;
        cdf.push(acc);
    }
    let mut out = String::with_capacity(target_chars + 16);
    while out.len() < target_chars {
        let u = (next_rand() >> 11) as f64 / (1u64 << 53) as f64;
        // Binary search for the smallest index whose CDF ≥ u.
        let idx = match cdf.binary_search_by(|p| p.total_cmp(&u)) {
            Ok(i) => i,
            Err(i) => i.min(vocab_size - 1),
        };
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(&words[idx]);
    }
    out
}

#[test]
#[ignore]
fn g2_pretok_corpus_scale_scaling_curve() {
    let vocab_size = 1000;
    // Train on the largest corpus we'll benchmark against, so the trainer
    // sees the full vocabulary and learns realistic merges.
    let training_corpus = build_zipfian_corpus(50_000, vocab_size);
    let tokenizer = BpeTrainer::train(&training_corpus, 1024);

    let scales: &[(usize, &str)] = &[
        (1_000, "1K"),
        (10_000, "10K"),
        (100_000, "100K"),
        (1_000_000, "1M"),
    ];

    eprintln!("scale   chars    plain_ns   pretok_ns  speedup  cache_entries  unique_pretokens");
    eprintln!("----- --------- ---------- ---------- -------- -------------- ----------------");

    for &(target_chars, label) in scales {
        let corpus = build_zipfian_corpus(target_chars, vocab_size);
        let actual_chars = corpus.len();
        // Count unique whitespace-delimited tokens for context on cache coverage.
        let unique_pretokens = corpus.split_whitespace().collect::<std::collections::HashSet<_>>().len();

        // Whole-text path (no pretokenization, no cache). Single encode — the
        // corpus is large enough that one pass dominates measurement noise.
        let mut encoder_plain = FastBpeEncoder::from_tokenizer(&tokenizer);
        let mut out = Vec::new();
        // Warmup the symbols/scratch buffers to the corpus size once.
        encoder_plain.encode_into(&corpus, &mut out);
        let t_plain = std::time::Instant::now();
        encoder_plain.encode_into(&corpus, &mut out);
        let plain_ns = t_plain.elapsed().as_nanos();

        // Pretokenized path (with cache). First call populates; second call
        // is the steady-state warm measurement.
        let mut encoder_pretok = FastBpeEncoder::from_tokenizer(&tokenizer);
        encoder_pretok.encode_into_pretok(&corpus, &mut out); // cold
        let t_pretok = std::time::Instant::now();
        encoder_pretok.encode_into_pretok(&corpus, &mut out); // warm
        let pretok_ns = t_pretok.elapsed().as_nanos();
        let cache_entries = encoder_pretok.pretoken_cache_len();

        let speedup = plain_ns as f64 / pretok_ns as f64;
        eprintln!(
            "{label:<5} {actual_chars:>8}  {plain_ns:>8.0}  {pretok_ns:>8.0}  {speedup:>6.2}x  {cache_entries:>12}  {unique_pretokens:>15}"
        );

        // G1 spot-check at each scale: pretok must still be bit-identical to
        // encode_into on the corpus-scale input.
        let mut a = Vec::new();
        let mut b = Vec::new();
        encoder_plain.encode_into(&corpus, &mut a);
        encoder_pretok.encode_into_pretok(&corpus, &mut b);
        assert_eq!(a, b, "G1 pretok divergence at {label} chars ({actual_chars})");
    }
}
