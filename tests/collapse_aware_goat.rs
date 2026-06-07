//! GOAT proof: collapse_aware_thinking feature improves detection accuracy (Plan 212 T7).
//!
//! Run with:
//!   cargo test --features collapse_aware_thinking --test collapse_aware_goat -- --nocapture
//!
//! GOAT Criteria:
//! 1. Collapse detection accuracy ≥80% across 50+ synthetic traces
//! 2. Option stripping effectiveness — prevents MCQ shortcut matching
//! 3. Efficiency reward shaping correctness across all ThinkingMode variants

#![cfg(feature = "collapse_aware_thinking")]

use katgpt_core::traits::CollapseDetector;
use katgpt_core::types::ThinkingBudget;
use katgpt_rs::pruners::{OptionStripper, S2FCollapseDetector, efficiency_reward};
use katgpt_rs::speculative::thinking_controller::ThinkingMode;
use katgpt_rs::speculative::types::NoScreeningPruner;

/// Deterministic RNG for reproducible synthetic traces.
struct TestRng {
    state: u32,
}

impl TestRng {
    fn new(seed: u32) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    fn next_u32(&mut self) -> u32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 17;
        self.state ^= self.state << 5;
        self.state
    }
}

/// Helper: create a detector with specific hesitation tokens and threshold.
fn make_detector(hesitation_tokens: Vec<u32>, threshold: u32) -> S2FCollapseDetector {
    let budget = ThinkingBudget {
        max_tokens: 4096,
        collapse_threshold: threshold,
        efficiency_gamma: 0.5,
    };
    S2FCollapseDetector::new(hesitation_tokens, &budget)
}

// ── GOAT Test 8: Collapse Detection Accuracy ─────────────────────

/// Generate a synthetic token stream with a known collapse/no-collapse ground truth.
///
/// Returns `(tokens, should_collapse)` where `should_collapse` is true if the stream
/// contains enough hesitation tokens to trigger collapse at the given threshold.
fn generate_trace(
    rng: &mut TestRng,
    hesitation_token: u32,
    normal_tokens: &[u32],
    trace_len: usize,
    hesitation_prob: f32,
    threshold: u32,
) -> (Vec<u32>, bool) {
    let mut tokens = Vec::with_capacity(trace_len);
    let mut hesitation_count = 0u32;

    for _ in 0..trace_len {
        // Roll a deterministic "coin" using the RNG.
        let roll = (rng.next_u32() as f32) / (u32::MAX as f32);
        if roll < hesitation_prob {
            tokens.push(hesitation_token);
            hesitation_count += 1;
        } else {
            let idx = (rng.next_u32() as usize) % normal_tokens.len();
            tokens.push(normal_tokens[idx]);
        }
    }

    let should_collapse = hesitation_count >= threshold;
    (tokens, should_collapse)
}

#[test]
fn goat_collapse_detection_accuracy() {
    let hesitation_token = 42u32;
    let normal_tokens: &[u32] = &[10, 20, 30, 40, 50, 60, 70, 80];
    let threshold = 4u32;
    let trace_len = 20usize;
    let num_traces = 60usize; // > 50 as required

    let mut rng = TestRng::new(12345);
    let mut correct = 0u32;
    let mut total = 0u32;

    for trace_idx in 0..num_traces {
        // Vary hesitation probability to get a mix of collapse/non-collapse cases.
        let hesitation_prob = match trace_idx % 4 {
            0 => 0.05, // Very low hesitation — unlikely to collapse
            1 => 0.15, // Low hesitation
            2 => 0.35, // Moderate — may or may not collapse
            _ => 0.60, // High — likely to collapse
        };

        let (tokens, should_collapse) = generate_trace(
            &mut rng,
            hesitation_token,
            normal_tokens,
            trace_len,
            hesitation_prob,
            threshold,
        );

        let mut detector = make_detector(vec![hesitation_token], threshold);
        let mut detected_collapse = false;

        for (pos, &token) in tokens.iter().enumerate() {
            if detector.check_collapse(token, pos) {
                detected_collapse = true;
                break;
            }
        }

        let is_correct = detected_collapse == should_collapse;
        if is_correct {
            correct += 1;
        }
        total += 1;
    }

    let accuracy = correct as f32 / total as f32;
    assert!(
        accuracy >= 0.80,
        "Collapse detection accuracy {accuracy:.2} is below 80% threshold ({correct}/{total} correct)"
    );
}

// ── GOAT Test 9: Option Stripper Effectiveness ────────────────────

/// Synthetic MCQ prompts for testing option stripping.
fn make_mcq_prompts() -> Vec<&'static str> {
    vec![
        "What is 2+2?\nA) 3\nB) 4\nC) 5\nD) 6",
        "Capital of Japan?\nA) Seoul\nB) Tokyo\nC) Beijing\nD) Bangkok",
        "H2O is?\nA) Water\nB) Air\nC) Fire\nD) Earth",
        "Speed of light?\n1) Fast\n2) Slow\n3) Medium\n4) Zero",
        "Largest planet?\nA. Jupiter\nB. Saturn\nC. Mars\nD. Venus",
        "Which is prime?\nA) 4\nB) 6\nC) 7\nD) 8",
        "Fe chemical symbol?\nA) Iron\nB) Gold\nC) Silver\nD) Copper",
        "Boiling point?\nA) 50C\nB) 100C\nC) 150C\nD) 200C",
        "DNA stands for?\nA) Deoxyribonucleic\nB) Dinucleotide\nC) Dinitrogen\nD) None",
        "Square root of 9?\n1. 2\n2. 3\n3. 4\n4. 5",
        "Smallest country?\nA) Monaco\nB) Vatican\nC) Malta\nD) Luxembourg",
        "Author of Hamlet?\nA) Dickens\nB) Shakespeare\nC) Twain\nD) Austen",
        "Chemical for salt?\nA) NaCl\nB) KCl\nC) HCl\nD) CaCl",
        "Year of moon landing?\n1) 1965\n2) 1969\n3) 1972\n4) 1975",
        "Largest ocean?\nA. Pacific\nB. Atlantic\nC. Indian\nD. Arctic",
        "Sides of hexagon?\nA) 5\nB) 6\nC) 7\nD) 8",
        "Fastest animal?\nA) Lion\nB) Cheetah\nC) Horse\nD) Gazelle",
        "Photosynthesis uses?\nA) CO2\nB) O2\nC) N2\nD) H2",
        "Opposite of hot?\n1) Warm\n2) Cold\n3) Cool\n4) Freezing",
        "Pi approximately?\nA) 3.14\nB) 2.72\nC) 1.62\nD) 4.13",
    ]
}

#[test]
fn goat_option_stripper_effectiveness() {
    let prompts = make_mcq_prompts();
    assert!(
        prompts.len() >= 20,
        "Need at least 20 prompts for GOAT test"
    );

    let mut total = 0u32;
    let mut effective = 0u32;

    for prompt in &prompts {
        let mut stripper = OptionStripper::new(NoScreeningPruner);

        // Pre-strip score with options visible (simulates option-matching shortcut).
        // NoScreeningPruner always returns 1.0 — the model "gets it right" by matching.
        let score_with_options = stripper.two_pass_score(0, 0, &[], true);
        assert_eq!(
            score_with_options, 1.0,
            "With options and match: should be 1.0"
        );

        // Strip options from the prompt.
        let stripped = stripper.strip_options(prompt);

        // Verify stripping actually changed the prompt (options were present).
        let changed = stripped != *prompt;
        assert!(
            changed,
            "Stripping must modify prompts that contain options"
        );

        // After stripping, the prompt should not contain option patterns.
        assert!(!stripped.contains("A) "), "A) pattern should be stripped");
        assert!(!stripped.contains("B) "), "B) pattern should be stripped");

        // Key anti-shortcut check: unmatched answer gets score 0.0 even with NoScreeningPruner.
        let score_unmatched = stripper.two_pass_score(0, 0, &[], false);
        if score_unmatched < score_with_options {
            effective += 1;
        }
        total += 1;
    }

    // The min-bottleneck must be effective for all prompts.
    let effectiveness = effective as f32 / total as f32;
    assert!(
        effectiveness >= 1.0,
        "Option stripper effectiveness {effectiveness:.2} must be 100% — all unmatched answers must score lower"
    );
}

// ── GOAT Test 10: Efficiency Reward Shaping Correctness ───────────

#[test]
fn goat_efficiency_reward_shaping_correctness() {
    let max_budget = 4096u32;
    let gamma = 0.5f32;

    // ── Correct + Direct → always 1.0 regardless of tokens ──
    let reward_direct_0 = efficiency_reward(true, 0, max_budget, ThinkingMode::Direct, gamma);
    let reward_direct_100 = efficiency_reward(true, 100, max_budget, ThinkingMode::Direct, gamma);
    let reward_direct_full =
        efficiency_reward(true, max_budget, max_budget, ThinkingMode::Direct, gamma);

    assert!(
        (reward_direct_0 - 1.0).abs() < 1e-6,
        "Direct correct (0 tokens) should be 1.0, got {reward_direct_0}"
    );
    assert!(
        (reward_direct_100 - 1.0).abs() < 1e-6,
        "Direct correct (100 tokens) should be 1.0, got {reward_direct_100}"
    );
    assert!(
        (reward_direct_full - 1.0).abs() < 1e-6,
        "Direct correct (full budget) should be 1.0, got {reward_direct_full}"
    );

    // ── Correct + Latent → monotonically decreasing with tokens_used ──
    let mut prev_reward = 2.0f32; // Higher than any possible reward.
    for tokens in [0u32, 100, 500, 1000, 2000, 3000, 4000] {
        let reward = efficiency_reward(true, tokens, max_budget, ThinkingMode::Latent, gamma);
        let expected = 1.0 - gamma * (tokens as f32 / max_budget as f32);
        assert!(
            (reward - expected).abs() < 1e-4,
            "Latent correct ({tokens} tokens): expected {expected:.4}, got {reward:.4}"
        );
        assert!(
            reward < prev_reward,
            "Latent reward must decrease with more tokens: {reward:.4} vs prev {prev_reward:.4}"
        );
        prev_reward = reward;
    }

    // ── Correct + CpuResample → always 0.0 (not yet calibrated) ──
    let reward_cpu = efficiency_reward(true, 0, max_budget, ThinkingMode::CpuResample, gamma);
    assert!(
        (reward_cpu - 0.0).abs() < 1e-6,
        "CpuResample correct should be 0.0, got {reward_cpu}"
    );
    let reward_cpu_full = efficiency_reward(
        true,
        max_budget,
        max_budget,
        ThinkingMode::CpuResample,
        gamma,
    );
    assert!(
        (reward_cpu_full - 0.0).abs() < 1e-6,
        "CpuResample correct (full budget) should be 0.0, got {reward_cpu_full}"
    );

    // ── Incorrect → always -1.0 regardless of mode ──
    for mode in [
        ThinkingMode::Direct,
        ThinkingMode::Latent,
        ThinkingMode::CpuResample,
    ] {
        let reward = efficiency_reward(false, 500, max_budget, mode, gamma);
        assert!(
            (reward - (-1.0)).abs() < 1e-6,
            "Incorrect ({mode:?}) should be -1.0, got {reward}"
        );
    }

    // ── Reward bounds: all values in [-1.0, 1.0] ──
    let test_cases = [
        (true, 0u32, ThinkingMode::Direct),
        (true, 2048, ThinkingMode::Latent),
        (true, 4096, ThinkingMode::Latent),
        (true, 0, ThinkingMode::CpuResample),
        (false, 0, ThinkingMode::Direct),
        (false, 2048, ThinkingMode::Latent),
        (false, 4096, ThinkingMode::CpuResample),
    ];

    for (correct, tokens, mode) in test_cases {
        let reward = efficiency_reward(correct, tokens, max_budget, mode, gamma);
        assert!(
            (-1.0..=1.0).contains(&reward),
            "Reward {reward:.4} out of bounds [-1, 1] for correct={correct}, tokens={tokens}, mode={mode:?}"
        );
    }

    // ── Monotonicity: Direct >= Latent for correct answers ──
    for tokens in [0u32, 500, 1000, 2000, 4096] {
        let reward_direct =
            efficiency_reward(true, tokens, max_budget, ThinkingMode::Direct, gamma);
        let reward_latent =
            efficiency_reward(true, tokens, max_budget, ThinkingMode::Latent, gamma);
        assert!(
            reward_direct >= reward_latent,
            "Direct ({reward_direct:.4}) should >= Latent ({reward_latent:.4}) at {tokens} tokens"
        );
    }

    // ── Gamma sensitivity: higher γ penalizes Latent more ──
    let reward_low_gamma = efficiency_reward(true, 2000, max_budget, ThinkingMode::Latent, 0.1);
    let reward_high_gamma = efficiency_reward(true, 2000, max_budget, ThinkingMode::Latent, 0.9);
    assert!(
        reward_low_gamma > reward_high_gamma,
        "Low gamma reward ({reward_low_gamma:.4}) should exceed high gamma ({reward_high_gamma:.4})"
    );

    // ── Edge case: zero max_budget → utilization = 1.0 ──
    let reward_zero_budget = efficiency_reward(true, 100, 0, ThinkingMode::Latent, 0.5);
    let expected_zero = 1.0 - 0.5 * 1.0;
    assert!(
        (reward_zero_budget - expected_zero).abs() < 1e-4,
        "Zero budget: expected {expected_zero:.4}, got {reward_zero_budget:.4}"
    );
}
