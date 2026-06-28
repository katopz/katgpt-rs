//! VocabChannel Pruner Demo — Before/After DDTree Stats (Plan 228).
//!
//! Demonstrates the ROTATE-derived ConstraintPruner:
//! 1. Creates synthetic weights with structured neuron→token mappings
//! 2. Runs ROTATE decomposition to discover vocabulary channels
//! 3. Builds reachability maps (per-neuron token sets)
//! 4. Shows pruning effectiveness: which tokens are reachable vs pruned
//!
//! Run: `cargo run --example vocab_channel_pruner_demo --features vocab_channel_pruner`

#[cfg(feature = "vocab_channel_pruner")]
fn main() {
    use katgpt_rs::speculative::{
        VocabChannelConfig, VocabChannelDecomposer, VocabChannelMap, VocabChannelPruner,
        decompose_layer_channels, householder_apply, skewness,
    };

    println!("🔬 VocabChannel Pruner Demo — ROTATE Weight Decomposition");
    println!("{}", "═".repeat(60));

    // ── 1. Synthetic Setup ──────────────────────────────────────
    let n_embd = 16;
    let mlp_hidden = 8;
    let vocab_size = 50;
    let n_layers = 2;

    println!("\n📐 Model dimensions:");
    println!("   n_embd = {n_embd}, mlp_hidden = {mlp_hidden}, vocab_size = {vocab_size}");
    println!("   n_layers = {n_layers}");

    // Create structured lm_head: each token has a distinct embedding direction
    let mut lm_head = vec![0.0f32; vocab_size * n_embd];
    for t in 0..vocab_size {
        let base_dim = t % n_embd;
        lm_head[t * n_embd + base_dim] = 2.0; // strong signal on one dimension
        // Add small noise on other dimensions
        for d in 0..n_embd {
            if d != base_dim {
                lm_head[t * n_embd + d] = 0.1 * ((t * 7 + d * 13) as f32).sin();
            }
        }
    }

    // ── 2. Per-Layer Decomposition ──────────────────────────────
    let config = VocabChannelConfig {
        max_channels: 3,
        top_k_tokens: 10,
        kurtosis_threshold: 0.5,
        max_iterations: 10,
        ..Default::default()
    };

    println!("\n🔧 VocabChannelConfig:");
    println!(
        "   max_channels={}, top_k_tokens={}",
        config.max_channels, config.top_k_tokens
    );
    println!(
        "   kurtosis_threshold={:.2}, lambda={:.2}, eta={:.3}",
        config.kurtosis_threshold, config.lambda, config.eta
    );
    println!(
        "   max_iterations={}, sigma_mask={:.1}",
        config.max_iterations, config.sigma_mask
    );

    let mut per_layer_channels: Vec<Vec<Vec<usize>>> = Vec::new();
    let _decomposer = VocabChannelDecomposer::new(config.clone());

    for layer in 0..n_layers {
        // Create synthetic mlp_w2: neurons respond to specific embedding dimensions
        let mut mlp_w2 = vec![0.0f32; n_embd * mlp_hidden];
        for j in 0..mlp_hidden {
            let target_dim = j % n_embd;
            for i in 0..n_embd {
                if i == target_dim {
                    mlp_w2[i * mlp_hidden + j] = 3.0; // strong weight on target dimension
                } else {
                    mlp_w2[i * mlp_hidden + j] = 0.05 * ((j * 11 + i * 3) as f32).cos();
                }
            }
        }

        let start = std::time::Instant::now();
        let neuron_tokens =
            decompose_layer_channels(&mlp_w2, &lm_head, n_embd, mlp_hidden, vocab_size, &config);
        let elapsed = start.elapsed();

        let total_tokens: usize = neuron_tokens.iter().map(|s: &Vec<usize>| s.len()).sum();
        let avg_tokens = if neuron_tokens.is_empty() {
            0.0
        } else {
            total_tokens as f64 / neuron_tokens.len() as f64
        };

        println!(
            "\n   Layer {layer}: {} neurons, {} total reachable tokens",
            neuron_tokens.len(),
            total_tokens
        );
        println!("   Avg tokens/neuron: {avg_tokens:.1}, Decomposition time: {elapsed:?}");

        per_layer_channels.push(neuron_tokens);
    }

    // ── 3. Build Reachability Map ───────────────────────────────
    let map = VocabChannelMap::from_channels(&per_layer_channels, vocab_size);

    println!("\n🗺️  Reachability Map:");
    println!(
        "   Layers: {}, Total neurons: {}",
        map.layer_count(),
        (0..map.layer_count())
            .map(|l| map.neuron_count(l))
            .sum::<usize>()
    );

    for layer in 0..map.layer_count() {
        let union = map.layer_union(layer);
        println!(
            "   Layer {layer}: {} tokens in union ({})",
            union.len(),
            if union.len() == vocab_size {
                "ALL"
            } else {
                "PARTIAL"
            }
        );
    }

    let global = map.global_union();
    println!(
        "   Global union: {}/{} tokens reachable",
        global.len(),
        vocab_size
    );

    // ── 4. Pruner Demo ──────────────────────────────────────────
    let pruner = VocabChannelPruner::new(map);

    println!("\n✂️  Pruner effectiveness:");

    // Show per-layer pruning via the map
    for layer in 0..n_layers {
        let reachable = pruner.map().layer_union(layer);
        let pruned = vocab_size - reachable.len();
        let pct = if vocab_size > 0 {
            pruned as f64 / vocab_size as f64 * 100.0
        } else {
            0.0
        };
        println!("   Layer {layer}: {pruned}/{vocab_size} tokens pruned ({pct:.1}%)");
    }

    // ── 5. Neuron-Specific Pruning ──────────────────────────────
    println!("\n🧠 Neuron-specific pruning (layer 0):");
    let active_neurons = [0, 1, 2, 3, 4];
    let mut combined_reachable = std::collections::HashSet::new();
    for &neuron in &active_neurons {
        let tokens = pruner.map().neuron_tokens(0, neuron);
        for &t in tokens {
            combined_reachable.insert(t);
        }
    }
    let pruned = vocab_size - combined_reachable.len();
    let pct = if vocab_size > 0 {
        pruned as f64 / vocab_size as f64 * 100.0
    } else {
        0.0
    };
    println!(
        "   Active neurons {:?}: {}/{} reachable, {pruned} pruned ({pct:.1}%)",
        active_neurons,
        combined_reachable.len(),
        vocab_size
    );

    // ── 6. ConstraintPruner trait demo ──────────────────────────
    println!("\n🎯 ConstraintPruner trait integration:");
    use katgpt_rs::speculative::ConstraintPruner;

    // Set active context: layer 0, neurons [0, 1, 2]
    pruner.set_active_context(0, &[0, 1, 2]);

    // Check some tokens
    let test_tokens = [0, 1, 64, 128, 200, 300, 499];
    for &t in &test_tokens {
        let valid = pruner.is_valid(0, t, &[]);
        println!(
            "   Token {t}: {}",
            if valid { "✅ valid" } else { "❌ pruned" }
        );
    }

    // Batch check
    let mut results = [false; 7];
    pruner.batch_is_valid(0, &test_tokens, &[], &mut results);
    let valid_count = results.iter().filter(|&&v| v).count();
    println!("   Batch: {valid_count}/{} tokens valid", test_tokens.len());

    // ── 7. Householder Reflection Demo ──────────────────────────
    println!("\n🔄 Householder reflection:");
    let h = vec![1.0f32, 0.0, 0.0, 0.0]; // reflect across x-axis
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let reflected = householder_apply(&h, &x);
    println!("   h = {h:?}");
    println!("   x = {x:?}");
    println!("   R(h)x = {reflected:?}");

    // ── 8. Serialization ────────────────────────────────────────
    let map2 = VocabChannelMap::from_channels(&per_layer_channels, vocab_size);
    let serialized = map2.serialize();
    let deserialized = VocabChannelMap::deserialize(&serialized).unwrap();
    assert_eq!(map2.layer_count(), deserialized.layer_count());
    println!(
        "\n💾 Serialization: {} bytes, round-trip OK",
        serialized.len()
    );

    // ── 9. Skewness Demo ────────────────────────────────────────
    println!("\n📊 Statistics demo:");
    let peaked = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0];
    let sym = [1.0, 2.0, 3.0, 2.0, 1.0];
    println!("   Skewness(peaked) = {:.3}", skewness(&peaked));
    println!("   Skewness(symmetric) = {:.3}", skewness(&sym));

    println!("\n{}", "═".repeat(60));
    println!("✅ VocabChannel Pruner demo complete");
}

#[cfg(not(feature = "vocab_channel_pruner"))]
fn main() {
    eprintln!("This example requires the `vocab_channel_pruner` feature.");
    eprintln!("Run: cargo run --example vocab_channel_pruner_demo --features vocab_channel_pruner");
}
