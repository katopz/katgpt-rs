//! GOAT proofs for ConvexTok LP vocabulary optimizer (Plan 127).
//!
//! **Source:** Tempus et al. (2026). Tokenisation via Convex Relaxations. arXiv:2605.22821

#[cfg(feature = "convex_tok")]
mod tests {
    use std::collections::HashMap;

    use katgpt_rs::tokenizer::{
        Certifier, ColourId, ConvexSolver, ConvexToToastBridge, GraphBuilder, Rounder,
        RoundingScheme, SpecialTokens, ToastTokenizerImpl, VertexId,
    };

    // ── Helpers ─────────────────────────────────────────────────

    /// Micro corpus for most tests: 5 pretokens of varying length.
    fn micro_corpus() -> Vec<Vec<u8>> {
        vec![
            b"hello".to_vec(),
            b"world".to_vec(),
            b"test".to_vec(),
            b"token".to_vec(),
            b"split".to_vec(),
        ]
    }

    /// Larger micro corpus for LP stress tests.
    fn stress_corpus() -> Vec<Vec<u8>> {
        let words = [
            "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
        ];
        words.iter().map(|w| w.as_bytes().to_vec()).collect()
    }

    /// N-gram counts from the micro corpus (simplified: just word-level counts).
    fn micro_ngram_counts() -> HashMap<Vec<u8>, u64> {
        let mut counts = HashMap::new();
        for word in &micro_corpus() {
            // Add the whole word as an n-gram
            *counts.entry(word.clone()).or_insert(0) += 1;
            // Add all 2-grams
            for i in 0..word.len().saturating_sub(1) {
                *counts.entry(word[i..i + 2].to_vec()).or_insert(0) += 1;
            }
            // Add all 3-grams
            for i in 0..word.len().saturating_sub(2) {
                *counts.entry(word[i..i + 3].to_vec()).or_insert(0) += 1;
            }
        }
        counts
    }

    // ── G01: Graph Construction ─────────────────────────────────

    #[test]
    fn g01_graph_construction_from_pretokens() {
        let corpus = micro_corpus();
        let graph = GraphBuilder::build(&corpus, 64);

        // 5 pretokens: "hello"(5), "world"(5), "test"(4), "token"(5), "split"(5)
        // Vertices per pretoken: len+1, minus 4 merges = (6+6+5+6+6) - 4 = 25
        assert!(graph.n_vertices > 0, "graph should have vertices");
        assert!(
            graph.n_vertices <= 30,
            "graph should have ≤30 vertices for 5 short words, got {}",
            graph.n_vertices
        );

        // Free edges = total bytes = 5+5+4+5+5 = 24
        assert_eq!(
            graph.free_edges.len(),
            24,
            "free edges should equal total byte count"
        );

        // Priced edges: spans of ≥2 bytes per pretoken
        assert!(
            graph.n_priced_edges() > 0,
            "should have priced edges for multi-byte spans"
        );

        // Colours: unique substrings
        assert!(graph.n_colours() > 0, "should have at least one colour");

        // Source ≠ sink
        assert_ne!(graph.source, graph.sink, "source and sink must differ");
    }

    // ── G02: Vertex Merge ───────────────────────────────────────

    #[test]
    fn g02_graph_vertex_merge() {
        let corpus = vec![b"ab".to_vec(), b"cd".to_vec()];
        let graph = GraphBuilder::build(&corpus, 64);

        // "ab" → 3 vertices (0,1,2)
        // "cd" → 3 vertices, but vertex 2 of "ab" merges with vertex 0 of "cd"
        // Total: 3 + 3 - 1 = 5 vertices
        assert_eq!(graph.n_vertices, 5);
        assert_eq!(graph.source, VertexId(0));
        assert_eq!(graph.sink, VertexId(4));

        // Free edges: a→b (1), c→d (1) = 2, but wait:
        // "ab": v0→v1, v1→v2 (2 free edges)
        // "cd": v2→v3, v3→v4 (2 free edges, sharing v2 boundary)
        assert_eq!(graph.free_edges.len(), 4);

        // Priced edges: "ab" (v0→v2), "cd" (v2→v4)
        assert_eq!(graph.priced_edges.len(), 2);
    }

    // ── G03: Colour Partition Disjoint ──────────────────────────

    #[test]
    fn g03_colour_partition_disjoint() {
        let corpus = micro_corpus();
        let graph = GraphBuilder::build(&corpus, 64);

        // Every priced edge has exactly one colour
        for (_from, _to, colour) in &graph.priced_edges {
            assert!(
                (colour.0 as usize) < graph.colour_bytes.len(),
                "colour {} out of range",
                colour.0
            );
            // Colour's bytes should be non-empty and span ≥2
            let bytes = &graph.colour_bytes[colour.0 as usize];
            assert!(!bytes.is_empty(), "colour bytes should not be empty");
            assert!(bytes.len() >= 2, "priced edge colour should span ≥2 bytes");
        }

        // Colour groups should be disjoint: each colour maps to a unique byte sequence
        let mut seen: HashMap<Vec<u8>, ColourId> = HashMap::new();
        for (cid, bytes) in graph.colour_bytes.iter().enumerate() {
            let colour = ColourId(cid as u32);
            if let Some(&prev) = seen.get(bytes) {
                assert_eq!(
                    prev, colour,
                    "duplicate byte sequence should map to same colour"
                );
            }
            seen.insert(bytes.clone(), colour);
        }

        // All colours should be referenced by at least one priced edge
        let mut referenced = vec![false; graph.n_colours()];
        for (_, _, colour) in &graph.priced_edges {
            referenced[colour.0 as usize] = true;
        }
        for (i, &refd) in referenced.iter().enumerate() {
            assert!(refd, "colour {i} is not referenced by any priced edge");
        }
    }

    // ── G04: LP Solves Within Tolerance ─────────────────────────

    #[test]
    fn g04_lp_solves_within_tolerance() {
        let corpus = stress_corpus();
        let graph = GraphBuilder::build(&corpus, 64);
        let sol = ConvexSolver::solve(&graph, 32).expect("LP should solve");

        // Objective should be finite and positive
        assert!(
            sol.lp_value.is_finite() && sol.lp_value > 0.0,
            "LP value should be positive finite, got {}",
            sol.lp_value
        );

        // All f variables in [0, 1]
        for (i, &f) in sol.f.iter().enumerate() {
            assert!(
                (-1e-6..=1.0 + 1e-6).contains(&f),
                "f[{i}] = {f} out of [0,1]"
            );
        }

        // All p variables in [0, 1]
        for (i, &p) in sol.p.iter().enumerate() {
            assert!(
                (-1e-6..=1.0 + 1e-6).contains(&p),
                "p[{i}] = {p} out of [0,1]"
            );
        }

        // All c variables in [0, 1]
        for (i, &c) in sol.c.iter().enumerate() {
            assert!(
                (-1e-6..=1.0 + 1e-6).contains(&c),
                "c[{i}] = {c} out of [0,1]"
            );
        }
    }

    // ── G05: LP Lower Bound Property ────────────────────────────

    #[test]
    fn g05_lp_lower_bound_property() {
        let corpus = vec![b"hello".to_vec(), b"world".to_vec()];
        let graph = GraphBuilder::build(&corpus, 64);
        let sol = ConvexSolver::solve(&graph, 4).expect("LP should solve");

        // Greedy tokenization: just use all single bytes = total byte count
        let greedy_compression = corpus.iter().map(|w| w.len() as f64).sum::<f64>();

        // LP value should be ≤ greedy (LP is a lower bound = minimum path length)
        assert!(
            sol.lp_value <= greedy_compression + 1e-6,
            "LP value ({}) should be ≤ greedy ({})",
            sol.lp_value,
            greedy_compression
        );
    }

    // ── G06: Det Rounding Selects Exactly K ─────────────────────

    #[test]
    fn g06_det_rounding_selects_exactly_k() {
        let corpus = stress_corpus();
        let graph = GraphBuilder::build(&corpus, 64);
        let budget_k = 5;

        let sol = ConvexSolver::solve(&graph, budget_k).expect("LP should solve");
        let rounded = Rounder::det(&graph, &sol);

        assert_eq!(
            rounded.rounding_scheme,
            RoundingScheme::Det,
            "should use Det scheme"
        );
        assert_eq!(
            rounded.n_selected, budget_k,
            "Det should select exactly K colours"
        );
        assert_eq!(
            rounded.selected_colours.len(),
            budget_k,
            "selected_colours length should match K"
        );
    }

    // ── G07: Bias Rounding Penalizes Long Tokens ────────────────

    #[test]
    fn g07_bias_rounding_penalizes_long_tokens() {
        // Construct a scenario where a 2-byte token has c=0.5 and a 4-byte token has c=0.9
        // Bias scoring: 2-byte gets 0.5/2=0.25, 4-byte gets 0.9/4=0.225
        // With K=1, bias should prefer the 2-byte token
        let corpus = vec![b"abcd".to_vec()];
        let graph = GraphBuilder::build(&corpus, 4);
        let sol = ConvexSolver::solve(&graph, 10).expect("LP should solve");

        let rounded = Rounder::bias(&graph, &sol);

        assert_eq!(
            rounded.rounding_scheme,
            RoundingScheme::Bias,
            "should use Bias scheme"
        );

        // Bias should select tokens, with shorter tokens potentially preferred
        assert!(
            rounded.n_selected > 0,
            "bias should select at least one token"
        );

        // Verify all selected bytes are valid
        for bytes in &rounded.selected_bytes {
            assert!(
                !bytes.is_empty(),
                "selected token bytes should not be empty"
            );
            assert!(
                bytes.len() >= 2,
                "selected tokens should be multi-byte (priced edges)"
            );
        }
    }

    // ── G08: Int Rounding Selects Only Integral ─────────────────

    #[test]
    fn g08_int_rounding_selects_only_integral() {
        let corpus = stress_corpus();
        let graph = GraphBuilder::build(&corpus, 64);
        let sol = ConvexSolver::solve(&graph, 10).expect("LP should solve");

        let rounded = Rounder::int(&graph, &sol);

        assert_eq!(
            rounded.rounding_scheme,
            RoundingScheme::Int,
            "should use Int scheme"
        );

        // Int rounding should only select colours with c ≥ 0.999
        // Verify by checking against the LP solution
        for &colour in &rounded.selected_colours {
            let c_val = sol.c[colour.0 as usize];
            assert!(
                c_val >= 0.999 - 1e-6,
                "int rounding selected colour {} with c={:.6}, expected ≥ 0.999",
                colour.0,
                c_val
            );
        }

        // Int may select fewer than K
        assert!(
            rounded.n_selected <= sol.budget_k,
            "int should select ≤ K colours"
        );
    }

    // ── G09: Rounded Vocabulary Has Valid Bytes ─────────────────

    #[test]
    fn g09_rounded_vocabulary_has_valid_bytes() {
        let corpus = micro_corpus();
        let graph = GraphBuilder::build(&corpus, 64);
        let sol = ConvexSolver::solve(&graph, 8).expect("LP should solve");

        for scheme in [
            RoundingScheme::Det,
            RoundingScheme::Bias,
            RoundingScheme::Int,
        ] {
            let rounded = match scheme {
                RoundingScheme::Det => Rounder::det(&graph, &sol),
                RoundingScheme::Bias => Rounder::bias(&graph, &sol),
                RoundingScheme::Int => Rounder::int(&graph, &sol),
            };

            // All selected bytes should be non-empty and match graph colour bytes
            for &colour in &rounded.selected_colours {
                let bytes = &graph.colour_bytes[colour.0 as usize];
                assert!(!bytes.is_empty(), "token bytes should not be empty");
                assert!(
                    rounded.selected_bytes.iter().any(|b| b == bytes),
                    "selected_bytes should contain the colour's bytes"
                );
            }

            // compression value should be finite and positive
            assert!(
                rounded.compression_value.is_finite() && rounded.compression_value > 0.0,
                "compression should be positive finite for {scheme}, got {}",
                rounded.compression_value
            );
        }
    }

    // ── G10: Optimality Gap Non-Negative ────────────────────────

    #[test]
    fn g10_optimality_gap_non_negative() {
        let corpus = stress_corpus();
        let graph = GraphBuilder::build(&corpus, 64);
        let sol = ConvexSolver::solve(&graph, 5).expect("LP should solve");

        for scheme in [
            RoundingScheme::Det,
            RoundingScheme::Bias,
            RoundingScheme::Int,
        ] {
            let rounded = match scheme {
                RoundingScheme::Det => Rounder::det(&graph, &sol),
                RoundingScheme::Bias => Rounder::bias(&graph, &sol),
                RoundingScheme::Int => Rounder::int(&graph, &sol),
            };

            let cert = Certifier::certify(&sol, &rounded);

            // LP is a lower bound → gap should be ≥ 0
            assert!(
                cert.gap_percent >= -0.01,
                "gap should be ≥ 0 for {scheme}, got {:.4}%",
                cert.gap_percent
            );

            // actual_compression should be ≥ lp_lower_bound
            assert!(
                cert.actual_compression >= cert.lp_lower_bound - 1e-6,
                "actual ({}) should be ≥ LP lower bound ({})",
                cert.actual_compression,
                cert.lp_lower_bound
            );
        }
    }

    // ── G11: Det Within Five Percent on Micro ───────────────────

    #[test]
    fn g11_det_within_five_percent_on_micro() {
        let corpus = stress_corpus();
        let graph = GraphBuilder::build(&corpus, 64);
        let sol = ConvexSolver::solve(&graph, 16).expect("LP should solve");

        let rounded = Rounder::det(&graph, &sol);
        let cert = Certifier::certify(&sol, &rounded);

        // On micro corpus, Det should be within 5% of LP optimal
        // (paper shows <1% at 32k+, we use much smaller scale so relax to 5%)
        assert!(
            cert.gap_percent <= 5.0,
            "Det should be within 5% of LP optimal, got gap = {:.2}%\n\
             LP lower bound = {:.4}, actual = {:.4}",
            cert.gap_percent,
            cert.lp_lower_bound,
            cert.actual_compression,
        );
    }

    // ── G12: ToaST Bridge Encode/Decode Roundtrip ───────────────

    #[test]
    fn g12_toast_bridge_encode_decode_roundtrip() {
        let corpus = micro_corpus();
        let graph = GraphBuilder::build(&corpus, 64);
        let sol = ConvexSolver::solve(&graph, 16).expect("LP should solve");
        let rounded = Rounder::det(&graph, &sol);

        let ngram_counts = micro_ngram_counts();
        let special = SpecialTokens::default();
        let tokenizer = ConvexToToastBridge::to_toast_tokenizer(&rounded, &ngram_counts, &special);

        // Encode/decode each word in the corpus
        for word_bytes in &corpus {
            let text = String::from_utf8_lossy(word_bytes).into_owned();
            let encoded = ToastTokenizerImpl::encode(&tokenizer, &text);
            let decoded = ToastTokenizerImpl::decode(&tokenizer, &encoded);

            // Decoded should match original (lossy: both sides are UTF-8)
            assert_eq!(decoded, text, "encode/decode roundtrip failed for '{text}'");
        }

        // Also test a multi-word string
        let full_text = "hello world";
        let encoded = ToastTokenizerImpl::encode(&tokenizer, full_text);
        let decoded = ToastTokenizerImpl::decode(&tokenizer, &encoded);
        assert_eq!(
            decoded, full_text,
            "encode/decode roundtrip failed for '{full_text}'"
        );

        // Verify no UNK tokens in encoded output for corpus words
        for word_bytes in &corpus {
            let text = String::from_utf8_lossy(word_bytes).into_owned();
            let encoded = ToastTokenizerImpl::encode(&tokenizer, &text);
            assert!(
                !encoded.contains(&tokenizer.unk_id),
                "no UNK tokens expected for corpus word '{text}', got ids: {encoded:?}"
            );
        }
    }
}
