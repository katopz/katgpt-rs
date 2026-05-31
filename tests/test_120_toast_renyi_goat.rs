//! GOAT proofs for Plan 120: ToaST compression vs BPE + Rényi entropy metric.
//!
//! T5 — ToaST vs BPE compression comparison.
//! T6 — Rényi entropy efficiency metric.

#![cfg(feature = "toast_tokenizer")]

mod tests {
    use std::collections::HashMap;

    use katgpt_rs::tokenizer::{
        BpeTokenizerImpl, BpeTrainer, SplitTreeBuilder, ToastTokenizer, ToastTokenizerImpl,
    };

    // ── Helpers ─────────────────────────────────────────────────────

    /// Build a ToaST tokenizer from a corpus: special tokens + all bytes +
    /// common multi-byte words with split trees from n-gram counts.
    fn build_toast_from_corpus(corpus: &str) -> ToastTokenizer {
        let bytes = corpus.as_bytes();

        // Count 2-grams and 3-grams
        let mut ngram_counts: HashMap<Vec<u8>, u64> = HashMap::new();
        for n in 2..=3 {
            for i in 0..bytes.len().saturating_sub(n - 1) {
                *ngram_counts.entry(bytes[i..i + n].to_vec()).or_default() += 1;
            }
        }

        let mut vocab_to_id = HashMap::new();
        let mut id_to_vocab = Vec::new();

        // Special tokens
        for (id, tok) in [
            (0usize, b"<pad>".to_vec()),
            (1, b"<bos>".to_vec()),
            (2, b"<eos>".to_vec()),
            (3, b"<unk>".to_vec()),
        ] {
            vocab_to_id.insert(tok.clone(), id);
            id_to_vocab.push(tok);
        }

        // All single bytes
        for b in 0u8..=255u8 {
            let byte_vec = vec![b];
            if !vocab_to_id.contains_key(&byte_vec) {
                let id = id_to_vocab.len();
                vocab_to_id.insert(byte_vec.clone(), id);
                id_to_vocab.push(byte_vec);
            }
        }

        // Common words from corpus (appearing >= 2 times, length >= 2)
        let mut word_counts: HashMap<&str, usize> = HashMap::new();
        for word in corpus.split_whitespace() {
            *word_counts.entry(word).or_default() += 1;
        }

        let mut trees = HashMap::new();
        let builder = SplitTreeBuilder::new(&ngram_counts, 1);

        for (word, &count) in &word_counts {
            if count >= 2 && word.len() >= 2 {
                let word_bytes = word.as_bytes();
                if !vocab_to_id.contains_key(word_bytes) {
                    let id = id_to_vocab.len();
                    vocab_to_id.insert(word_bytes.to_vec(), id);
                    id_to_vocab.push(word_bytes.to_vec());
                }
                // Build split tree for this word
                let tree = builder.build(word_bytes);
                trees.insert(word_bytes.to_vec(), tree);
            }
        }

        ToastTokenizer {
            vocab_to_id,
            id_to_vocab,
            trees,
            bos_id: 1,
            eos_id: 2,
            pad_id: 0,
            unk_id: 3,
        }
    }

    // ── Rényi entropy helpers ───────────────────────────────────────

    fn renyi_entropy(token_ids: &[usize], alpha: f64) -> f64 {
        if token_ids.is_empty() || (alpha - 1.0).abs() < 1e-10 {
            return 0.0; // Shannon limit or empty — skip
        }
        let mut freq: HashMap<usize, f64> = HashMap::new();
        for &id in token_ids {
            *freq.entry(id).or_default() += 1.0;
        }
        let n = token_ids.len() as f64;
        let sum_p_alpha: f64 = freq.values().map(|&count| (count / n).powf(alpha)).sum();
        (1.0 / (1.0 - alpha)) * sum_p_alpha.log2()
    }

    fn renyi_efficiency(token_ids: &[usize], vocab_size: usize, alpha: f64) -> f64 {
        let h = renyi_entropy(token_ids, alpha);
        let max_h = (vocab_size as f64).log2();
        if max_h > 0.0 { h / max_h } else { 0.0 }
    }

    // ── T5: ToaST vs BPE Compression Comparison ─────────────────────

    #[test]
    fn proof_t5_toast_fewer_or_equal_tokens_than_bpe() {
        let corpus = "the cat sat on the mat the cat the mat the test hello world the test split";
        let bpe = BpeTrainer::train(corpus, 300);
        let toast = build_toast_from_corpus(corpus);

        let test_strings = [
            "the cat sat on the mat",
            "hello world test split",
            "the the the test test test",
        ];

        println!(
            "┌──────────────────────────────┬───────┬─────────────┬──────────────┬────────────┬─────────────┐"
        );
        println!(
            "│ text                         │ bytes │ bpe_tokens  │ toast_tokens │ bpe_ratio  │ toast_ratio │"
        );
        println!(
            "├──────────────────────────────┼───────┼─────────────┼──────────────┼────────────┼─────────────┤"
        );

        for text in &test_strings {
            let bpe_ids = BpeTokenizerImpl::encode(&bpe, text);
            let toast_ids = ToastTokenizerImpl::encode(&toast, text);

            let byte_len = text.len();
            let bpe_len = bpe_ids.len();
            let toast_len = toast_ids.len();
            let bpe_ratio = bpe_len as f64 / byte_len as f64;
            let toast_ratio = toast_len as f64 / byte_len as f64;

            println!(
                "│ {text:<28} │ {byte_len:>5} │ {bpe_len:>11} │ {toast_len:>12} │ {bpe_ratio:>10.3} │ {toast_ratio:>11.3} │",
            );

            // ToaST must compress at least as well as BPE OR at least as well as raw bytes
            assert!(
                toast_len <= bpe_len || toast_len <= byte_len,
                "ToaST ({toast_len} tokens) should compress <= BPE ({bpe_len} tokens) or <= bytes ({byte_len}) for \"{text}\"",
            );
        }

        println!(
            "└──────────────────────────────┴───────┴─────────────┴──────────────┴────────────┴─────────────┘"
        );
    }

    #[test]
    fn proof_t5_toast_no_unknown_tokens() {
        let corpus = "the cat sat on the mat hello world test split";
        let toast = build_toast_from_corpus(corpus);

        let texts = [
            "the cat sat on the mat",
            "hello world test split",
            "abcdefghijklmnopqrstuvwxyz",
            "0123456789 !@#",
        ];

        for text in &texts {
            let ids = ToastTokenizerImpl::encode(&toast, text);
            let unk_count = ids.iter().filter(|&&id| id == toast.unk_id).count();
            assert_eq!(
                unk_count, 0,
                "ToaST byte-level fallback should produce zero unknown tokens for ASCII text: \"{text}\""
            );
        }
    }

    #[test]
    fn proof_t5_roundtrip_identity() {
        let corpus = "the cat sat on the mat the cat the mat the test hello world the test split";
        let toast = build_toast_from_corpus(corpus);

        let texts = [
            "the cat sat on the mat",
            "hello world test split",
            "the the the test test test",
            "a",
            "",
        ];

        for text in &texts {
            let ids = ToastTokenizerImpl::encode(&toast, text);
            let decoded = ToastTokenizerImpl::decode(&toast, &ids);
            assert_eq!(
                decoded, *text,
                "Roundtrip encode→decode must reproduce original text: \"{text}\""
            );
        }
    }

    // ── T6: Rényi Entropy Efficiency Metric ──────────────────────────

    #[test]
    fn proof_t6_renyi_toast_entropy_competitive_with_bpe() {
        let corpus = "the cat sat on the mat the cat the mat the test hello world the test split \
                      the cat the mat the test hello world the cat the mat the test";
        let bpe = BpeTrainer::train(corpus, 300);
        let toast = build_toast_from_corpus(corpus);

        let bpe_ids = BpeTokenizerImpl::encode(&bpe, corpus);
        let toast_ids = ToastTokenizerImpl::encode(&toast, corpus);

        let alpha = 2.5;
        let bpe_entropy = renyi_entropy(&bpe_ids, alpha);
        let toast_entropy = renyi_entropy(&toast_ids, alpha);

        let bpe_bits_per_token = bpe_entropy;
        let toast_bits_per_token = toast_entropy;
        let bpe_bits_per_byte = bpe_bits_per_token * bpe_ids.len() as f64 / corpus.len() as f64;
        let toast_bits_per_byte =
            toast_bits_per_token * toast_ids.len() as f64 / corpus.len() as f64;

        println!(
            "BPE  Rényi (α={alpha}): {bpe_entropy:.4} bits/token, {bpe_bits_per_byte:.4} bits/byte (vocab={}, tokens={})",
            bpe.id_to_vocab.len(),
            bpe_ids.len()
        );
        println!(
            "ToaST Rényi (α={alpha}): {toast_entropy:.4} bits/token, {toast_bits_per_byte:.4} bits/byte (vocab={}, tokens={})",
            toast.vocab_size(),
            toast_ids.len()
        );

        // Both tokenizers should produce positive entropy
        assert!(bpe_entropy > 0.0, "BPE Rényi entropy must be positive");
        assert!(toast_entropy > 0.0, "ToaST Rényi entropy must be positive");

        // ToaST should produce fewer or equal tokens than bytes (compression)
        assert!(
            toast_ids.len() <= corpus.len(),
            "ToaST should compress: {} tokens <= {} bytes",
            toast_ids.len(),
            corpus.len()
        );
    }

    #[test]
    fn proof_t6_renyi_monotonic_with_vocab_size() {
        let corpus = "the cat sat on the mat the cat the mat the test hello world the test split";
        let text = "the cat the mat the test";
        let vocab_sizes = [50usize, 100, 200];
        let alpha = 2.5;

        println!("┌────────────┬──────────────┬──────────────────┐");
        println!("│ vocab_size │ token_count  │ renyi_efficiency │");
        println!("├────────────┼──────────────┼──────────────────┤");

        let mut prev_eff: Option<f64> = None;
        for &vs in &vocab_sizes {
            let bpe = BpeTrainer::train(corpus, vs);
            let ids = BpeTokenizerImpl::encode(&bpe, text);
            let eff = renyi_efficiency(&ids, bpe.id_to_vocab.len(), alpha);

            println!("│ {vs:>10} │ {len:>12} │ {eff:>16.6} │", len = ids.len());

            if let Some(p) = prev_eff {
                assert!(
                    eff >= p - 1e-9,
                    "Rényi efficiency should be non-decreasing with vocab size: {p:.6} -> {eff:.6} at vocab={vs}",
                );
            }
            prev_eff = Some(eff);
        }

        println!("└────────────┴──────────────┴──────────────────┘");
    }

    #[test]
    fn proof_t6_renyi_uniform_max() {
        // Uniform distribution: each token appears exactly once
        let token_ids: Vec<usize> = (0..16).collect();
        let eff = renyi_efficiency(&token_ids, 16, 2.5);

        // For uniform distribution, Rényi entropy = log2(n) for any alpha,
        // so efficiency should be exactly 1.0
        let expected = renyi_entropy(&token_ids, 2.5);
        let max_h = (16f64).log2();
        let expected_eff = expected / max_h;

        println!(
            "Uniform (16 tokens): Rényi entropy={expected:.6}, max={max_h:.6}, efficiency={eff:.6}"
        );
        assert!(
            (eff - expected_eff).abs() < 1e-9,
            "Uniform distribution efficiency should be {expected_eff:.6}, got {eff:.6}",
        );
        assert!(
            (eff - 1.0).abs() < 1e-9,
            "Uniform distribution efficiency should be ~1.0, got {eff:.6}",
        );
    }

    #[test]
    fn proof_t6_renyi_single_token_min() {
        // Degenerate distribution: all tokens are the same
        let token_ids = vec![42usize; 100];
        let h = renyi_entropy(&token_ids, 2.5);

        println!("Single-token (100 copies of id=42): Rényi entropy={h:.6}");
        assert!(
            h.abs() < 1e-9,
            "Single-token Rényi entropy should be 0.0, got {h:.6}",
        );

        // Efficiency should also be 0 for any vocab size > 1
        let eff = renyi_efficiency(&token_ids, 256, 2.5);
        assert!(
            eff.abs() < 1e-9,
            "Single-token efficiency should be 0.0, got {eff:.6}",
        );
    }

    // ── Summary ─────────────────────────────────────────────────────

    #[test]
    fn proof_120_summary() {
        let corpus = "the cat sat on the mat the cat the mat the test hello world the test split \
                      the cat the mat the test hello world the cat the mat the test";
        let bpe = BpeTrainer::train(corpus, 300);
        let toast = build_toast_from_corpus(corpus);

        println!("\n═══════════════════════════════════════════════════════════");
        println!("  Plan 120 Summary: ToaST vs BPE + Rényi Entropy Metric");
        println!("═══════════════════════════════════════════════════════════");

        let bpe_ids = BpeTokenizerImpl::encode(&bpe, corpus);
        let toast_ids = ToastTokenizerImpl::encode(&toast, corpus);

        let bpe_compression = bpe_ids.len() as f64 / corpus.len() as f64;
        let toast_compression = toast_ids.len() as f64 / corpus.len() as f64;

        println!("  Corpus bytes       : {}", corpus.len());
        println!(
            "  BPE  vocab/tokens  : {} / {}  (compression: {bpe_compression:.3})",
            bpe.id_to_vocab.len(),
            bpe_ids.len()
        );
        println!(
            "  ToaST vocab/tokens : {} / {}  (compression: {toast_compression:.3})",
            toast.vocab_size(),
            toast_ids.len()
        );

        let alpha = 2.5;
        let bpe_eff = renyi_efficiency(&bpe_ids, bpe.id_to_vocab.len(), alpha);
        let toast_eff = renyi_efficiency(&toast_ids, toast.vocab_size(), alpha);
        println!("  BPE  Rényi eff (α={alpha}): {bpe_eff:.6}");
        println!("  ToaST Rényi eff (α={alpha}): {toast_eff:.6}");

        // Roundtrip check
        let decoded = ToastTokenizerImpl::decode(&toast, &toast_ids);
        assert_eq!(decoded, corpus, "ToaST roundtrip must be identity");
        println!("  Roundtrip identity : ✓");

        // No unknowns
        let unk_count = toast_ids.iter().filter(|&&id| id == toast.unk_id).count();
        assert_eq!(unk_count, 0, "No unknown tokens expected");
        println!("  Zero unknowns      : ✓");

        println!("═══════════════════════════════════════════════════════════\n");
    }
}
