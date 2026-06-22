#![allow(unexpected_cfgs)]
//! Algorithmic-Probability Sampler Demo (Plan 305, Research 284).
//!
//! Demonstrates the `CompressionPriorSampler` vs uniform sampling on a
//! synthetic low-K optimum: 16-bit action space where the optimum is
//! `0xFFFF` (all-ones) — K = O(1) under RLE (a single run of identical bits).
//!
//! Expected result: the K-prior sampler finds the optimum in O(K) samples,
//! while uniform sampling needs O(|X|/2) ≈ 32768 samples on average.
//!
//! Run: `cargo run --example algorithmic_probability_sampler_demo --features complexity_prior_sampler`

#[cfg(feature = "complexity_prior_sampler")]
fn main() {
    use katgpt_rs::screening::{
        CompressionPriorSampler, EntropyComplexity, L1Complexity, RleComplexity,
    };

    println!("🧠 Algorithmic-Probability Sampler Demo (Plan 305)");
    println!("{}", "═".repeat(60));
    println!("Source: Dingle & Hutter 2026, Entropy 28(2):226");
    println!();

    // 16-bit action space; optimum is 0xFFFF (K=O(1) under RLE).
    const N: usize = 1 << 16; // 65536 candidates
    const OPTIMUM: u16 = 0xFFFF;

    // Score: -popcount(x XOR 0xFFFF). Optimum (0xFFFF) scores 0; everything
    // else scores negative. This is a simple objective in the Dingle–Hutter
    // sense (low Kolmogorov complexity).
    fn score(x: u16) -> i32 {
        -((x ^ OPTIMUM).count_ones() as i32)
    }

    // Pre-build the candidate byte-slices (low byte + high byte of each u16).
    let candidates: Vec<[u8; 2]> = (0..N).map(|i| (i as u16).to_le_bytes()).collect();
    let candidates_ref: Vec<&[u8]> = candidates.iter().map(|c| c.as_slice()).collect();

    println!("Action space:    |X| = {} (16-bit)", N);
    println!("Optimum:         x* = 0x{:04X} (score = 0)", OPTIMUM);
    println!("Objective:       f(x) = -popcount(x XOR 0xFFFF)  [low-K]");
    println!();

    // ── Uniform baseline ─────────────────────────────────────────
    // Theoretical expected time-to-optimum for uniform sampling = |X|/2.
    let mut rng = fastrand::Rng::with_seed(0xCAFE);
    let mut uniform_hits = 0usize;
    for _ in 0..N {
        let idx = rng.usize(0..N);
        if idx as u16 == OPTIMUM {
            uniform_hits += 1;
            break;
        }
    }
    let uniform_expected = N / 2;
    println!("── Uniform baseline ─────────────────────────────────");
    println!(
        "Theoretical E[time-to-optimum] ≈ |X|/2 = {}",
        uniform_expected
    );
    println!(
        "Hit in first N draws? {}",
        if uniform_hits > 0 { "yes" } else { "no" }
    );
    println!();

    // ── K-prior sampler (RLE proxy) ─────────────────────────────
    // The optimum 0xFFFF encodes as bytes [0xFF, 0xFF] — a single RLE run
    // of length 2 → K̃ ≈ 2/2 = 1.0 (max RLE compressibility for 2 bytes).
    // A random candidate has 2 distinct bytes → K̃ ≈ 1.0 too (RLE doesn't
    // help at length 2). So we expect the RLE proxy to NOT significantly
    // beat uniform on this 2-byte encoding — a useful honest negative.
    //
    // Switch to L1 proxy: optimum [0xFF, 0xFF] has max L1 (= 510/510 = 1.0).
    // Random candidates average L1 ≈ 0.5. So L1 PRIORITIZES the optimum
    // here (counterintuitively — high L1 = simple "all max" pattern).
    // We invert by using α < 0 in the log-prob to favor high-L1.
    println!("── K-prior sampler (L1 proxy, α=-2.0) ──────────────");
    let sampler_l1 = CompressionPriorSampler::new(L1Complexity::new(), -2.0, 0.0);
    let mut rng = fastrand::Rng::with_seed(0xBEEF);
    let mut l1_hits_at: Option<usize> = None;
    let mut scratch = vec![0.0f32; N];
    for trial in 0..1_000 {
        let idx = sampler_l1.sample_ix(&candidates_ref, &mut scratch, &mut rng);
        if idx as u16 == OPTIMUM {
            l1_hits_at = Some(trial);
            break;
        }
    }
    match l1_hits_at {
        Some(t) => println!("✅ L1-prior sampler hit optimum at trial {}", t),
        None => println!("⚠️  L1-prior sampler did not hit optimum in 1000 trials"),
    }
    println!();

    // ── Show top-K by each proxy ────────────────────────────────
    println!("── Top-5 candidates by K̃ proxy ─────────────────────");
    // Each proxy is a distinct type, so we run them sequentially rather than
    // stuffing them into a single array (which would need trait objects).
    let mut top5 = [0usize; 5];

    let rle_sampler = CompressionPriorSampler::new(RleComplexity::new(), 1.0, 0.0);
    rle_sampler.top_k(&candidates_ref, 5, &mut top5);
    print_top5("RLE", &top5, score);

    let entropy_sampler = CompressionPriorSampler::new(EntropyComplexity::new(), 1.0, 0.0);
    entropy_sampler.top_k(&candidates_ref, 5, &mut top5);
    print_top5("Entropy", &top5, score);

    let l1_sampler = CompressionPriorSampler::new(L1Complexity::new(), 1.0, 0.0);
    l1_sampler.top_k(&candidates_ref, 5, &mut top5);
    print_top5("L1", &top5, score);
    println!();

    // ── Honest verdict ──────────────────────────────────────────
    println!("── Honest verdict ───────────────────────────────────");
    println!("On this 16-bit 'all-ones optimum' synthetic:");
    println!("  • Uniform E[time-to-optimum] = {}", uniform_expected);
    match l1_hits_at {
        Some(t) => println!(
            "  • L1-prior (α=-2.0): hit at trial {} ({:.1}× speedup)",
            t,
            uniform_expected as f64 / (t as f64 + 1.0)
        ),
        None => {
            println!("  • L1-prior (α=-2.0): did NOT hit in 1000 trials on this 2-byte encoding")
        }
    }
    println!();
    println!("Note: RLE proxy at 2-byte encoding has limited discriminative");
    println!("power (all 2-byte slices compress to ≤ 2 bytes regardless).");
    println!("The L1 proxy with inverted α exploits the 'all max bytes' pattern");
    println!("characteristic of this optimum, but 2 bytes is too short for the");
    println!("signal to dominate sampling noise. Phase 2 GOAT gate G2 (Plan 305)");
    println!("uses a longer-encoding synthetic to demonstrate exponential lift.");
    println!();
    println!("See .plans/305_algorithmic_probability_sampler.md Phase 2 for the");
    println!("formal G1 (sampler safety) + G2 (exponential speedup) benchmark.");
}

#[cfg(not(feature = "complexity_prior_sampler"))]
fn main() {
    println!("This example requires the `complexity_prior_sampler` feature.");
    println!(
        "Run: cargo run --example algorithmic_probability_sampler_demo --features complexity_prior_sampler"
    );
}

#[cfg(feature = "complexity_prior_sampler")]
fn print_top5(name: &str, top5: &[usize], score: impl Fn(u16) -> i32) {
    let scores: Vec<i32> = top5.iter().map(|&i| score(i as u16)).collect();
    let bytes: Vec<String> = top5
        .iter()
        .map(|&i| format!("0x{:04X}", i as u16))
        .collect();
    println!("  {:8} → {}  scores={:?}", name, bytes.join(", "), scores);
}
