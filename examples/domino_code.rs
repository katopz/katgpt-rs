//! Domino Code: DDTree + DominoPruner for syntax-aware speculative decoding.
//!
//! Demonstrates prefix-conditioned syntax correction for code generation:
//! - Shows how domino correction suppresses invalid syntax patterns
//! - Example: `if` followed by `{` gets boosted, `if` followed by `fn` gets suppressed
//!
//! Run: cargo run --example domino_code --features domino_correction,validator

#[cfg(not(all(feature = "domino_correction", feature = "validator")))]
fn main() {
    eprintln!("This example requires both `domino_correction` and `validator` features.");
    eprintln!("Run with: cargo run --example domino_code --features domino_correction,validator");
}

#[cfg(all(feature = "domino_correction", feature = "validator"))]
fn main() {
    use katgpt_rs::speculative::{PrefixCorrectionTable, domino_correct_marginals, domino_score};

    println!("🁣 Domino Code: Syntax-Aware Prefix Correction");
    println!("{}", "═".repeat(60));

    // Mini vocab for demonstration: special tokens + keywords
    // 0=PAD, 1=if, 2=fn, 3={, 4=(, 5=let, 6=mut, 7=}, 8=), 9=;
    let vocab_size = 10;

    // ── 1. Build correction table ───────────────────────────────────
    // Rule: after `if`, boost `{` and `(`, suppress `fn`, `let`, `mut`
    let mut if_correction = vec![0.0f32; vocab_size];
    if_correction[3] = 0.3; // { : boost
    if_correction[4] = 0.2; // ( : boost
    if_correction[2] = -0.4; // fn : suppress
    if_correction[5] = -0.3; // let : suppress
    if_correction[6] = -0.3; // mut : suppress

    // Rule: after `{`, boost `let`, `fn`, `if`, suppress `}`, `;`
    let mut brace_correction = vec![0.0f32; vocab_size];
    brace_correction[5] = 0.2; // let
    brace_correction[2] = 0.2; // fn
    brace_correction[1] = 0.2; // if
    brace_correction[7] = -0.3; // } : suppress (empty blocks unlikely)
    brace_correction[9] = -0.2; // ;

    let table = PrefixCorrectionTable::builder(vocab_size)
        .add_correction(&[1], &if_correction) // prefix "if"
        .add_correction(&[3], &brace_correction) // prefix "{"
        .build();

    println!("\n📋 Correction Table");
    println!("  Entries: {}", table.len());
    println!("  Vocab size: {}", table.vocab_size());

    // ── 2. Simulate uniform marginals ────────────────────────────────
    let mut marginals = vec![
        vec![0.0, 0.3, 0.1, 0.15, 0.15, 0.1, 0.1, 0.03, 0.02, 0.05], // depth 0
        vec![0.0, 0.1, 0.2, 0.15, 0.15, 0.15, 0.1, 0.05, 0.05, 0.05], // depth 1
        vec![0.0, 0.15, 0.1, 0.2, 0.1, 0.15, 0.1, 0.05, 0.05, 0.1],  // depth 2
    ];

    let token_names = ["PAD", "if", "fn", "{", "(", "let", "mut", "}", ")", ";"];

    println!("\n📊 Before Correction (depth 1 — after token 'if'):");
    print_marginal(&marginals[1], &token_names);

    // ── 3. Apply domino correction ──────────────────────────────────
    // sampled_tokens[0] = 1 (if), meaning depth 1 sees prefix [1]
    let sampled_tokens = [1usize, 3]; // if → {
    domino_correct_marginals(&mut marginals, &sampled_tokens, &table);

    println!("\n📊 After Correction (depth 1 — prefix 'if' applied):");
    print_marginal(&marginals[1], &token_names);

    println!("\n📊 After Correction (depth 2 — prefix '{{' applied):");
    print_marginal(&marginals[2], &token_names);

    // ── 4. Verify correction effects ────────────────────────────────
    // After correction, `{` should be higher and `fn` should be lower at depth 1
    let pre_if_brace = 0.15f32; // original
    let post_if_brace = marginals[1][3]; // corrected
    println!(
        "\n✓ '{{' after 'if': {:.3} → {:.3} ({:+.1}%)",
        pre_if_brace,
        post_if_brace,
        (post_if_brace - pre_if_brace) / pre_if_brace * 100.0
    );

    let pre_if_fn = 0.2f32; // original
    let post_if_fn = marginals[1][2]; // corrected
    println!(
        "✓ 'fn' after 'if': {:.3} → {:.3} ({:+.1}%)",
        pre_if_fn,
        post_if_fn,
        (post_if_fn - pre_if_fn) / pre_if_fn * 100.0
    );

    // ── 5. Domino score demo ────────────────────────────────────────
    println!("\n🧮 Domino Score (code generation scenario)");
    println!("  prefix_strength reflects confidence in the sampled path:");
    for strength in [1.0, 0.9, 0.7, 0.5] {
        let score = domino_score(-2.0, 3, strength);
        println!(
            "  depth=3, strength={:.1} → score={:.4} (base=-2.0)",
            strength, score
        );
    }

    println!("\n✅ Domino code correction is modelless — pure pattern extraction, no training.");
}

#[cfg(all(feature = "domino_correction", feature = "validator"))]
fn print_marginal(marginal: &[f32], names: &[&str]) {
    let sum: f32 = marginal.iter().sum();
    for (i, &prob) in marginal.iter().enumerate() {
        if prob > 0.001 {
            println!(
                "    {:>4} [{:>2}]: {:.4} (norm: {:.3})",
                names[i],
                i,
                prob,
                prob / sum
            );
        }
    }
}
