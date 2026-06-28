    use super::*;

    #[test]
    fn test_with_overrides_none_unchanged() {
        let config = Config::draft();
        let overrides = InferenceOverrides::default();
        let original_tb = config.tree_budget;
        let original_temp = config.temperature;
        let original_dl = config.draft_lookahead;
        let result = config.with_overrides(&overrides);
        assert_eq!(result.tree_budget, original_tb);
        assert_eq!(result.temperature, original_temp);
        assert_eq!(result.draft_lookahead, original_dl);
    }

    #[test]
    fn test_with_overrides_some_applied() {
        let config = Config::draft();
        let overrides = InferenceOverrides {
            tree_budget: Some(99),
            temperature: Some(0.123),
            ..Default::default()
        };
        let original_dl = config.draft_lookahead;
        let result = config.with_overrides(&overrides);
        assert_eq!(result.tree_budget, 99);
        assert!((result.temperature - 0.123).abs() < 1e-6);
        // Non-overridden fields stay the same
        assert_eq!(result.draft_lookahead, original_dl);
    }

    #[test]
    fn test_with_overrides_all_fields() {
        let config = Config::draft();
        let overrides = InferenceOverrides {
            tree_budget: Some(1),
            draft_lookahead: Some(2),
            parallel_threshold: Some(3),
            screening_threshold: Some(0.1),
            temperature: Some(0.2),
            sparse_threshold: Some(0.3),
            early_exit_patience: Some(4),
            early_exit_gap: Some(0.4),
            mtp_activation_threshold: Some(5),
            mtp_cluster_vocab_threshold: Some(6),
            mtp_shared_kv_prompt_threshold: Some(7),
            mtp_cluster_size: Some(8),
            mtp_min_output_tokens: Some(10),
            mtp_cluster_topk: Some(2),
            sp_kv_threshold: Some(0.5),
            width_rollouts: Some(9),
            early_stop_threshold: Some(0.6),
            convergence_selector: Some(ConvergenceSelector::Top1Converged),
            mls_layers: Some(3),
            // drafter_lora_path is consumed by the caller, not applied to Config
            drafter_lora_path: None,
            max_plan_horizon: Some(5),
            #[cfg(feature = "hydra_budget")]
            hydra_skip_threshold: None,
            #[cfg(feature = "hydra_budget")]
            hydra_skip_erasure_draft: None,
            depth_tier: None,
        };
        let result = config.with_overrides(&overrides);
        assert_eq!(result.tree_budget, 1);
        assert_eq!(result.draft_lookahead, 2);
        assert_eq!(result.parallel_threshold, 3);
        assert!((result.screening_threshold - 0.1).abs() < 1e-6);
        assert!((result.temperature - 0.2).abs() < 1e-6);
        assert!((result.sparse_threshold - 0.3).abs() < 1e-6);
        assert_eq!(result.early_exit_patience, 4);
        assert!((result.early_exit_gap - 0.4).abs() < 1e-6);
        assert_eq!(result.mtp_activation_threshold, 5);
        assert_eq!(result.mtp_cluster_vocab_threshold, 6);
        assert_eq!(result.mtp_shared_kv_prompt_threshold, 7);
        assert_eq!(result.mtp_cluster_size, 8);
        assert_eq!(result.mtp_min_output_tokens, 10);
        assert_eq!(result.mtp_cluster_topk, 2);
        assert!((result.sp_kv_threshold - 0.5).abs() < 1e-6);
        assert_eq!(result.width_rollouts, 9);
        assert!((result.early_stop_threshold - 0.6).abs() < 1e-6);
        assert_eq!(
            result.convergence_selector,
            ConvergenceSelector::Top1Converged
        );
        assert_eq!(result.mls_layers, 3);
        // max_plan_horizon caps draft_lookahead when lower (Plan 112 T11)
        assert_eq!(result.draft_lookahead, 2); // 2 (from override) < 5 (horizon cap), stays 2
    }

    #[test]
    fn test_early_exit_defaults_disabled() {
        let config = Config::micro();
        assert_eq!(config.early_exit_patience, 0);
        assert!((config.early_exit_gap).abs() < 1e-6);
    }

    #[test]
    #[cfg(feature = "domain_latent")]
    fn test_domain_latent_save_load_roundtrip() {
        let tmp = std::env::temp_dir().join("katgpt_core_test_domain_latent.bin");
        let original = DomainLatent::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        original.save(&tmp).unwrap();
        let loaded = DomainLatent::load(&tmp).unwrap();
        assert_eq!(original.embedding, loaded.embedding);
        drop(std::fs::remove_file(&tmp));
    }

    #[test]
    #[cfg(feature = "domain_latent")]
    fn test_domain_latent_zeros() {
        let dl = DomainLatent::zeros(8);
        assert_eq!(dl.embedding.len(), 8);
        assert!(dl.embedding.iter().all(|&v| v == 0.0));
    }

    #[test]
    #[cfg(feature = "domain_latent")]
    fn test_domain_latent_invalid_magic() {
        let tmp = std::env::temp_dir().join("katgpt_core_test_bad_magic.bin");
        let mut buf = b"XXXX".to_vec();
        buf.push(1); // version
        buf.extend_from_slice(&4u32.to_le_bytes()); // kv_dim
        buf.extend_from_slice(
            &[
                0.0f32.to_le_bytes(),
                0.0f32.to_le_bytes(),
                0.0f32.to_le_bytes(),
                0.0f32.to_le_bytes(),
            ]
            .concat(),
        );
        let hash = blake3::hash(&buf);
        buf.extend_from_slice(hash.as_bytes());
        std::fs::write(&tmp, &buf).unwrap();
        assert!(DomainLatent::load(&tmp).is_err());
        drop(std::fs::remove_file(&tmp));
    }

    #[test]
    #[cfg(feature = "domain_latent")]
    fn test_domain_latent_checksum_mismatch() {
        let tmp = std::env::temp_dir().join("katgpt_core_test_bad_checksum.bin");
        let mut buf = b"DLAT".to_vec();
        buf.push(1); // version
        buf.extend_from_slice(&4u32.to_le_bytes()); // kv_dim
        buf.extend_from_slice(&[0.0f32.to_le_bytes(); 4].concat());
        buf.extend_from_slice(&[0u8; 32]); // wrong checksum
        std::fs::write(&tmp, &buf).unwrap();
        assert!(DomainLatent::load(&tmp).is_err());
        drop(std::fs::remove_file(&tmp));
    }

    #[test]
    #[cfg(feature = "domain_latent")]
    fn test_domain_latent_file_too_small() {
        let tmp = std::env::temp_dir().join("katgpt_core_test_too_small.bin");
        std::fs::write(&tmp, b"DLAT").unwrap();
        assert!(DomainLatent::load(&tmp).is_err());
        drop(std::fs::remove_file(&tmp));
    }

    #[test]
    fn test_config_game() {
        let config = Config::game();
        assert_eq!(config.vocab_size, 10);
        assert_eq!(config.block_size, 170);
        assert_eq!(config.n_embd, 32);
        assert_eq!(config.n_head, 4);
        assert_eq!(config.head_dim, 8);
        assert_eq!(config.mlp_hidden, 128);
        assert!(config.validate().is_ok());
    }

    #[test]
    #[cfg(feature = "gpart_adapter")]
    fn test_gpart_isometry() {
        let adapter = GpartAdapter {
            d: 4,
            seed: 42,
            theta: vec![1.0, 2.0, 3.0, 4.0],
        };
        assert!(adapter.check_isometry(256));
    }

    #[test]
    #[cfg(feature = "gpart_adapter")]
    fn test_gpart_apply_correctness() {
        let adapter = GpartAdapter {
            d: 2,
            seed: 123,
            theta: vec![1.0, -1.0],
        };
        let mut weights = vec![0.0f32; 8];
        adapter.apply(&mut weights);
        // Every weight should have changed (each element assigned to group 0 or 1)
        assert!(weights.iter().all(|&w| w != 0.0));
        // Verify deterministic: apply again with same adapter should produce same result
        let mut weights2 = vec![0.0f32; 8];
        adapter.apply(&mut weights2);
        assert_eq!(weights, weights2);
    }

    #[test]
    #[cfg(feature = "gpart_adapter")]
    fn test_gpart_commitment_roundtrip() {
        let adapter = GpartAdapter {
            d: 4,
            seed: 99,
            theta: vec![0.5, -0.5, 1.0, -1.0],
        };
        let commit = adapter.commitment();
        assert!(adapter.verify(&commit));
    }

    #[test]
    #[cfg(feature = "gpart_adapter")]
    fn test_gpart_tamper_detection() {
        let adapter = GpartAdapter {
            d: 4,
            seed: 99,
            theta: vec![0.5, -0.5, 1.0, -1.0],
        };
        let commit = adapter.commitment();
        // Tampered commitment
        let mut tampered = commit;
        tampered[0] ^= 0xFF;
        assert!(!adapter.verify(&tampered));
        // Different theta produces different commitment
        let tampered_adapter = GpartAdapter {
            d: 4,
            seed: 99,
            theta: vec![0.5, -0.5, 1.0, 0.0], // last element changed
        };
        assert_ne!(adapter.commitment(), tampered_adapter.commitment());
        // Different seed produces different commitment
        let seed_adapter = GpartAdapter {
            d: 4,
            seed: 100,
            theta: vec![0.5, -0.5, 1.0, -1.0],
        };
        assert_ne!(adapter.commitment(), seed_adapter.commitment());
    }

    #[test]
    #[cfg(feature = "gpart_adapter")]
    fn test_gpart_save_load_roundtrip() {
        let tmp = std::env::temp_dir().join("katgpt_core_test_gpart.bin");
        let adapter = GpartAdapter {
            d: 8,
            seed: 42,
            theta: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        };
        adapter.save(&tmp).unwrap();
        let loaded = GpartAdapter::load(&tmp).unwrap();
        assert_eq!(loaded.d, adapter.d);
        assert_eq!(loaded.seed, adapter.seed);
        assert_eq!(loaded.theta, adapter.theta);
        drop(std::fs::remove_file(&tmp));
    }

    #[test]
    #[cfg(feature = "gpart_adapter")]
    fn test_gpart_determinism_same_seed() {
        let adapter = GpartAdapter {
            d: 4,
            seed: 42,
            theta: vec![1.0, 2.0, 3.0, 4.0],
        };
        let mut w1 = vec![0.0f32; 64];
        let mut w2 = vec![0.0f32; 64];
        adapter.apply(&mut w1);
        adapter.apply(&mut w2);
        assert_eq!(w1, w2);
    }

    #[test]
    #[cfg(feature = "gpart_pruning")]
    fn test_gpart_masked_all_true_matches_unmasked() {
        // When group_mask is all-true, masked apply must equal unmasked apply.
        let adapter = GpartAdapter {
            d: 4,
            seed: 42,
            theta: vec![1.0, 2.0, 3.0, 4.0],
        };
        let n = 256;
        let mut w_ref = vec![0.0f32; n];
        let mut assignments = vec![0usize; n];
        let mut group_sizes = vec![0usize; adapter.d];
        adapter.apply_with_scratch(&mut w_ref, &mut assignments, &mut group_sizes);

        let mut w_masked = vec![0.0f32; n];
        let mask = vec![true; adapter.d];
        adapter.apply_with_scratch_masked(
            &mut w_masked,
            &mut assignments,
            &mut group_sizes,
            &mask,
        );

        for (i, (a, b)) in w_ref.iter().zip(w_masked.iter()).enumerate() {
            assert_eq!(a, b, "mismatch at index {i}");
        }
    }

    #[test]
    #[cfg(feature = "gpart_pruning")]
    fn test_gpart_masked_zeroes_masked_groups() {
        // If group g is masked off, NO base_weights element assigned to g
        // should receive any delta from theta[g]. Concretely: the masked
        // delta for elements of masked groups must be exactly zero.
        let adapter = GpartAdapter {
            d: 4,
            seed: 7,
            theta: vec![1.0, 2.0, 3.0, 4.0],
        };
        let n = 512;
        let mut w_zero = vec![0.0f32; n]; // baseline: all groups masked off
        let mut assignments = vec![0usize; n];
        let mut group_sizes = vec![0usize; adapter.d];
        let all_false = vec![false; adapter.d];
        adapter.apply_with_scratch_masked(
            &mut w_zero,
            &mut assignments,
            &mut group_sizes,
            &all_false,
        );
        // With every group masked, weights should remain exactly zero.
        assert!(w_zero.iter().all(|&w| w == 0.0));
    }

    #[test]
    #[cfg(feature = "gpart_pruning")]
    fn test_gpart_masked_single_group_preserves_others() {
        // Masking group 0 only: every element in groups 1..d should still receive
        // its full delta, while elements in group 0 should stay at baseline.
        let adapter = GpartAdapter {
            d: 4,
            seed: 7,
            theta: vec![1.0, 2.0, 3.0, 4.0],
        };
        let n = 512;

        // Full (unmasked) reference.
        let mut w_full = vec![0.0f32; n];
        let mut assignments_full = vec![0usize; n];
        let mut group_sizes = vec![0usize; adapter.d];
        adapter.apply_with_scratch(&mut w_full, &mut assignments_full, &mut group_sizes);

        // Masked: drop group 0.
        let mut w_masked = vec![0.0f32; n];
        let mut assignments = vec![0usize; n];
        let mut mask = vec![true; adapter.d];
        mask[0] = false;
        adapter.apply_with_scratch_masked(
            &mut w_masked,
            &mut assignments,
            &mut group_sizes,
            &mask,
        );

        // assignments[] is regenerated deterministically from seed, so it matches
        // between the two calls — compare per-element using the masked call's
        // assignment vector.
        for i in 0..n {
            let g = assignments[i];
            if g == 0 {
                // Group 0 masked off → masked output should equal baseline (0.0).
                assert_eq!(w_masked[i], 0.0, "group-0 element {i} should be zero");
            } else {
                // Other groups untouched → should equal full reference.
                assert_eq!(
                    w_masked[i], w_full[i],
                    "group-{g} element {i} should match unmasked"
                );
            }
        }
    }

    #[test]
    #[cfg(feature = "gpart_pruning")]
    fn test_gpart_topk_mask_edge_cases() {
        let adapter = GpartAdapter {
            d: 4,
            seed: 0,
            theta: vec![0.1, 0.5, 0.3, 0.9],
        };
        // k >= d → all active
        assert!(adapter.topk_mask(adapter.d).iter().all(|&b| b));
        assert!(adapter.topk_mask(adapter.d + 10).iter().all(|&b| b));
        // k == 0 → all inactive
        assert!(adapter.topk_mask(0).iter().all(|&b| !b));
    }

    #[test]
    #[cfg(feature = "gpart_pruning")]
    fn test_gpart_topk_mask_selects_largest_magnitudes() {
        // θ = [0.1, 0.5, -0.3, 0.9] → |θ| = [0.1, 0.5, 0.3, 0.9]
        // top-2 by magnitude should keep groups {3 (0.9), 1 (0.5)}.
        let adapter = GpartAdapter {
            d: 4,
            seed: 0,
            theta: vec![0.1, 0.5, -0.3, 0.9],
        };
        let mask = adapter.topk_mask(2);
        assert_eq!(mask.len(), 4);
        assert!(mask[1] && mask[3], "groups 1 and 3 must be in top-2");
        assert!(!mask[0] && !mask[2], "groups 0 and 2 must be pruned");
        // Exactly 2 active.
        assert_eq!(mask.iter().filter(|&&b| b).count(), 2);
    }

    // ── Issue 299: multi-adapter LoRA loading ──────────────────────

    /// Build a Plan 008 LoRA binary file with `n` adapters for testing.
    /// Each adapter's A/B matrices are filled with a distinct sentinel pattern
    /// so we can verify the loader returns the right data per adapter.
    fn build_test_lora_file(path: &std::path::Path, n_adapters: usize, rank: usize) {
        let in_dim = 4usize;
        let out_dim = 4usize;
        let alpha = 8.0f32;

        // Payload: [n_adapters(4)][rank(4)][alpha(4)] + per-adapter [in_dim(4)][out_dim(4)][A][B]
        let header = 4 + 4 + 4;
        let per_adapter = 4 + 4 + (rank * in_dim + out_dim * rank) * std::mem::size_of::<f32>();
        let mut payload = Vec::with_capacity(header + n_adapters * per_adapter);

        payload.extend_from_slice(&(n_adapters as u32).to_le_bytes());
        payload.extend_from_slice(&(rank as u32).to_le_bytes());
        payload.extend_from_slice(&alpha.to_le_bytes());

        for i in 0..n_adapters {
            payload.extend_from_slice(&(in_dim as u32).to_le_bytes());
            payload.extend_from_slice(&(out_dim as u32).to_le_bytes());
            // A: [rank × in_dim], sentinel = (i + 1) * 0.1
            let a_sentinel = (i as f32 + 1.0) * 0.1;
            for _ in 0..(rank * in_dim) {
                payload.extend_from_slice(&a_sentinel.to_le_bytes());
            }
            // B: [out_dim × rank], sentinel = (i + 1) * -0.2
            let b_sentinel = (i as f32 + 1.0) * -0.2;
            for _ in 0..(out_dim * rank) {
                payload.extend_from_slice(&b_sentinel.to_le_bytes());
            }
        }

        let checksum = blake3::hash(&payload);
        let mut file_data = Vec::with_capacity(4 + 4 + 32 + payload.len());
        file_data.extend_from_slice(b"LORA");
        file_data.extend_from_slice(&1u32.to_le_bytes());
        file_data.extend_from_slice(checksum.as_bytes());
        file_data.extend_from_slice(&payload);
        std::fs::write(path, &file_data).unwrap();
    }

    #[test]
    fn test_lora_load_single_adapter_returns_one_element_vec() {
        let tmp = std::env::temp_dir().join("katgpt_core_test_lora_single.bin");
        build_test_lora_file(&tmp, 1, 4);
        let adapters = LoraAdapter::load(&tmp).expect("single-adapter load should succeed");
        assert_eq!(adapters.len(), 1, "single-adapter file must yield 1-element Vec");
        assert_eq!(adapters[0].rank, 4);
        assert_eq!(adapters[0].in_dim, 4);
        assert_eq!(adapters[0].out_dim, 4);
        assert!((adapters[0].alpha - 8.0).abs() < 1e-6);
        drop(std::fs::remove_file(&tmp));
    }

    /// Issue 299 regression: a 12-adapter file (L2 model, 6 adapters/layer × 2 layers)
    /// must load ALL 12 adapters. Previously `load()` returned only the first.
    #[test]
    fn test_lora_load_multi_adapter_returns_all() {
        let tmp = std::env::temp_dir().join("katgpt_core_test_lora_multi.bin");
        build_test_lora_file(&tmp, 12, 4);
        let adapters = LoraAdapter::load(&tmp).expect("12-adapter load should succeed");
        assert_eq!(
            adapters.len(),
            12,
            "12-adapter file must yield 12-element Vec (Issue 299 regression)"
        );

        // Verify each adapter has its own sentinel data — proves layers 1..11 aren't dropped
        for (i, a) in adapters.iter().enumerate() {
            let expected_a = (i as f32 + 1.0) * 0.1;
            let expected_b = (i as f32 + 1.0) * -0.2;
            assert_eq!(a.rank, 4, "adapter {i} rank");
            assert_eq!(a.in_dim, 4, "adapter {i} in_dim");
            assert_eq!(a.out_dim, 4, "adapter {i} out_dim");
            assert_eq!(a.a.len(), 4 * 4, "adapter {i} A matrix size");
            assert_eq!(a.b.len(), 4 * 4, "adapter {i} B matrix size");
            assert!(
                a.a.iter().all(|&v| (v - expected_a).abs() < 1e-6),
                "adapter {i} A data mismatch"
            );
            assert!(
                a.b.iter().all(|&v| (v - expected_b).abs() < 1e-6),
                "adapter {i} B data mismatch"
            );
        }
        drop(std::fs::remove_file(&tmp));
    }

    #[test]
    fn test_lora_load_first_returns_only_first_adapter() {
        let tmp = std::env::temp_dir().join("katgpt_core_test_lora_first.bin");
        build_test_lora_file(&tmp, 6, 4);
        let first = LoraAdapter::load_first(&tmp).expect("load_first should succeed");
        assert_eq!(first.rank, 4);
        // Adapter 0 sentinel is 1.0 * 0.1 = 0.1
        assert!(first.a.iter().all(|&v| (v - 0.1).abs() < 1e-6));
        assert!(first.b.iter().all(|&v| (v - (-0.2)).abs() < 1e-6));
        drop(std::fs::remove_file(&tmp));
    }

    #[test]
    fn test_lora_load_rejects_truncated_multi_adapter() {
        // Truncate a 12-adapter file so only 6 adapters' bytes fit — loader must error
        let tmp = std::env::temp_dir().join("katgpt_core_test_lora_truncated.bin");
        build_test_lora_file(&tmp, 12, 4);
        let full = std::fs::read(&tmp).unwrap();
        // Cut the payload in half (keep header + checksum + ~half the adapters)
        let cut = full.len() / 2 + 20; // keep enough to pass header check
        std::fs::write(&tmp, &full[..cut]).unwrap();
        // Re-generate checksum won't match — but if it did, the truncation check fires.
        // Either way, load must fail (not silently return wrong number of adapters).
        let result = LoraAdapter::load(&tmp);
        assert!(result.is_err(), "truncated multi-adapter file must error, not silently drop adapters");
        drop(std::fs::remove_file(&tmp));
    }

    #[test]
    fn test_lora_load_rejects_bad_magic() {
        let tmp = std::env::temp_dir().join("katgpt_core_test_lora_bad_magic.bin");
        std::fs::write(&tmp, b"NOPE\0\0\0\0").unwrap();
        assert!(LoraAdapter::load(&tmp).is_err());
        drop(std::fs::remove_file(&tmp));
    }

    #[test]
    fn test_lora_load_rejects_zero_adapters() {
        let tmp = std::env::temp_dir().join("katgpt_core_test_lora_zero.bin");
        // Build a valid-checksum file that declares 0 adapters
        let mut payload = Vec::new();
        payload.extend_from_slice(&0u32.to_le_bytes()); // n_adapters = 0
        payload.extend_from_slice(&4u32.to_le_bytes()); // rank
        payload.extend_from_slice(&8.0f32.to_le_bytes()); // alpha
        let checksum = blake3::hash(&payload);
        let mut buf = Vec::new();
        buf.extend_from_slice(b"LORA");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(checksum.as_bytes());
        buf.extend_from_slice(&payload);
        std::fs::write(&tmp, &buf).unwrap();
        assert!(LoraAdapter::load(&tmp).is_err(), "zero-adapter declaration must be rejected");
        drop(std::fs::remove_file(&tmp));
    }
