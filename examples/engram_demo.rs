//! Plan 299 T7.10 — Engram end-to-end demo.
//!
//! Populates a small pattern table from a few hardcoded "sentences" (really
//! just sequences of token ids — the full NFKC pipeline is heavy for an
//! example, so we use a trivial identity tokenizer shim), computes multi-head
//! hashes from a trigram suffix, looks up K patterns, and sigmoid-fuses them
//! into a hidden state. Prints the before/after hidden state L2 norm.
//!
//! # Run
//!
//! ```text
//! cargo run --features engram --example engram_demo
//! ```

#![cfg(feature = "engram")]

use katgpt_core::engram::{
    CanonicalId, EngramConfig, EngramTable, EngramTableBuilder, HashHead, K_MAX, multi_head_hash,
};

/// Trivial tokenizer shim: maps each "word" in our hardcoded corpus to a
/// raw token id. This is NOT a real tokenizer — it just demonstrates the
/// pipeline without pulling in Hugging Face `tokenizers` or sentencepiece.
struct TrivialTokenizer {
    words: Vec<&'static str>,
}

impl TrivialTokenizer {
    fn new() -> Self {
        Self {
            words: vec![
                "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "apple", "banana",
                "cherry", "date",
            ],
        }
    }

    fn encode(&self, word: &str) -> Option<u32> {
        self.words.iter().position(|w| *w == word).map(|i| i as u32)
    }
}

/// Build a small K_MAX head set for the demo. Uses small primes so the
/// example runs fast (no need for production-quality hash decorrelation).
fn make_demo_heads() -> [HashHead; K_MAX] {
    let mut heads = [HashHead {
        n: 0,
        k: 0,
        modulus: 1,
        seed: 0,
    }; K_MAX];
    for k in 0..K_MAX {
        // Distinct prime per head, all ≥ 256 (the table size).
        let prime = next_prime(256 + (k as u64) * 17);
        heads[k] = HashHead {
            n: 8,
            k: k as u8,
            modulus: prime,
            seed: 0xCAFE_BABE_0000_0000u64 + (k as u64).wrapping_mul(0xDEAD_BEEF),
        };
    }
    heads
}

fn next_prime(n: u64) -> u64 {
    let mut c = n.max(2);
    loop {
        if is_prime(c) {
            return c;
        }
        c += 1;
    }
}

fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n < 4 {
        return true;
    }
    if n % 2 == 0 {
        return false;
    }
    let mut i = 3u64;
    while i.saturating_mul(i) <= n {
        if n % i == 0 {
            return false;
        }
        i += 2;
    }
    true
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Plan 299 — Engram Demo (hash → lookup → sigmoid fuse)       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // ── Setup ────────────────────────────────────────────────────────────
    let d = 32; // hidden dim
    let n_slots = 256; // table size
    let tokenizer = TrivialTokenizer::new();
    let heads = make_demo_heads();

    // Build a small pattern table. For each word in our vocab, hash the
    // trigram (word-2, word-1, word) and populate the slot with a synthetic
    // pattern. In a real system these patterns would be learned embeddings;
    // here we use a deterministic synthetic pattern for demo purposes.
    let mut builder = EngramTableBuilder::new(n_slots, d).with_heads(heads);

    let corpus = [
        ["the", "quick", "brown"],
        ["quick", "brown", "fox"],
        ["brown", "fox", "jumps"],
        ["fox", "jumps", "over"],
        ["jumps", "over", "the"],
        ["over", "the", "lazy"],
        ["the", "lazy", "dog"],
        ["apple", "banana", "cherry"],
        ["banana", "cherry", "date"],
    ];

    println!("Populating table from {} trigrams...", corpus.len());
    for trigram in &corpus {
        // Encode the trigram → 3 canonical ids (identity projection here).
        let cids: Vec<CanonicalId> = trigram
            .iter()
            .filter_map(|w| tokenizer.encode(w))
            .map(|id| CanonicalId(id as u64))
            .collect();
        if cids.len() != 3 {
            continue; // skip trigrams with unknown words
        }

        // Multi-head hash → 16 slot keys.
        let keys = multi_head_hash(&cids, &heads);

        // Use the first key to pick a slot; populate with a synthetic pattern.
        let slot_hash = keys[0];
        let pattern: Vec<f32> = (0..d)
            .map(|j| {
                let seed = slot_hash.0.wrapping_add(j as u64);
                ((seed.wrapping_mul(0x9E37) as f32) / (1u64 << 32) as f32) * 2.0 - 1.0
            })
            .collect();
        builder.add_pattern(slot_hash, &pattern);
    }
    let table = builder.build();
    println!(
        "  Table built: {} slots × D={}",
        table.num_slots(),
        table.dim()
    );
    println!();

    // ── Lookup + fuse ───────────────────────────────────────────────────
    println!("Querying the table with suffix \"quick brown fox\"...");
    let suffix = [
        tokenizer
            .encode("quick")
            .map(|i| CanonicalId(i as u64))
            .unwrap(),
        tokenizer
            .encode("brown")
            .map(|i| CanonicalId(i as u64))
            .unwrap(),
        tokenizer
            .encode("fox")
            .map(|i| CanonicalId(i as u64))
            .unwrap(),
    ];
    let keys = multi_head_hash(&suffix, &heads);
    println!("  multi_head_hash → {} slot keys (first 4 shown):", K_MAX);
    for k in 0..4 {
        println!(
            "    head {k:2}: hash = {} → slot {}",
            keys[k].0,
            keys[k].0 % n_slots as u64
        );
    }
    println!();

    // Hidden state — starts as a sinusoidal pattern (synthetic "live state").
    let mut hidden: Vec<f32> = (0..d).map(|i| (i as f32 * 0.3).sin()).collect();
    let l2_before = hidden.iter().map(|v| v * v).sum::<f32>().sqrt();
    println!("Hidden state before fuse: L2 norm = {l2_before:.4}");

    // Configure the sigmoid fusion: tau = √D = √32 ≈ 5.66.
    let cfg = EngramConfig::for_dim(d);
    println!(
        "Sigmoid fusion config: tau = {:.4}, k_heads = {}",
        cfg.fusion.tau, cfg.k_heads
    );
    println!("  (CRITICAL: sigmoid, NOT softmax — per AGENTS.md)");
    println!();

    // Scratch buffers (caller-allocated for zero-alloc hot path).
    let mut scratch_lookup = vec![0.0f32; K_MAX * d];
    let mut scratch_norm = vec![0.0f32; d];
    let mut scratch_out = vec![0.0f32; d];

    // Query vector — use the hidden state itself as the query (autoregressive
    // pattern: "what should I retrieve given my current state?").
    let query = hidden.clone();

    // Fuse! This is the end-to-end primitive: lookup K patterns, sigmoid-gate
    // each, residual-add into the hidden state.
    katgpt_core::engram::fuse_into_hidden_state(
        &mut hidden,
        &query,
        &table,
        &keys,
        &cfg,
        &mut scratch_lookup,
        &mut scratch_norm,
        &mut scratch_out,
    );

    let l2_after = hidden.iter().map(|v| v * v).sum::<f32>().sqrt();
    println!("Hidden state after fuse:  L2 norm = {l2_after:.4}");
    println!("  Δ L2 = {:+.4}", l2_after - l2_before);
    println!();

    // ── Show a sample of the retrieved patterns ─────────────────────────
    println!("Sample of retrieved patterns (first 4 elements of first 4 heads):");
    for k in 0..4 {
        let slot = &scratch_lookup[k * d..(k + 1) * d];
        let l2 = slot.iter().map(|v| v * v).sum::<f32>().sqrt();
        let first4: Vec<String> = slot.iter().take(4).map(|v| format!("{:+.3}", v)).collect();
        let hit = if l2 > 0.0 { "HIT" } else { "miss" };
        println!("  head {k}: [{first4:?}] L2={l2:.3}  ({hit})");
    }
    println!();

    // ── Show the table identity ─────────────────────────────────────────
    use katgpt_core::engram::EngramTableId;
    let id = EngramTableId::from_table(&table);
    let id_hex: String = id.0.iter().take(8).map(|b| format!("{:02x}", b)).collect();
    println!("Table identity (EngramTableId, BLAKE3 Merkle root):");
    println!("  first 8 bytes: 0x{id_hex}");
    println!("  verify:        {}", id.verify(&table));
    println!();

    println!("Done. The open primitive is end-to-end functional:");
    println!("  corpus → trigrams → multi-head hash → table lookup → sigmoid fuse → hidden state.");
}
