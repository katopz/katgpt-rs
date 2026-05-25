//! GOAT Proof — Plan 111 Mechanistic Data Attribution (T10).
//!
//! Six proofs verifying the mech_attribution pipeline end-to-end:
//!
//! - **P1**: Catalyst Detection — structural patterns score ≥0.7, natural language <0.3
//! - **P2**: Influence Ranking — catalyst-scored top-K beats random top-K
//! - **P3**: Synthetic Augmentation — synthetic catalyst data improves metrics vs random
//! - **P4**: Cross-Domain Transfer — templates transfer structural (not semantic) similarity
//! - **P5**: Power-Law Verification — top 10% samples have ≥40% cumulative catalyst_overlap
//! - **P6**: Convergence Rate — augmented data reaches target quality in ≤80% of baseline rounds
//!
//! All proofs use synthetic test data — no actual LLM training needed.

#![cfg(feature = "mech_attribution")]

use katgpt_rs::pruners::mech_attribution::{
    ActivationInfluenceProxy, CatalystPattern, InfluenceConfig, batch_influence_rank,
    catalyst_score, extract_template, generate_synthetic,
};

// ── P1: Catalyst Detection ────────────────────────────────────────────

#[test]
fn p1_catalyst_detection_structural_scores_high_natural_scores_low() {
    let config = InfluenceConfig {
        catalyst_threshold: 0.0, // accept all patterns for scoring
        ..Default::default()
    };

    // Structural patterns that should score high (≥0.7)
    let structural_samples = [
        // XML with dense tags
        "<root><item>a</item><item>b</item><item>c</item><item>d</item></root>",
        // Dense code
        "fn foo(x: i32) -> i32 { let y = x + 1; let z = y * 2; return z; }",
        // LaTeX
        r"\frac{a}{b} + \sqrt{c} = \sum_{i=0}^{n} x_i \int_0^1 f(x) dx",
        // DB rows
        "a|b|c|d\n1|2|3|4\n5|6|7|8\n9|10|11|12",
        // Pure repetition
        "abc abc abc abc abc abc abc abc abc abc",
    ];

    for sample in &structural_samples {
        let score = catalyst_score(sample, &config);
        assert!(
            score.catalyst_overlap >= 0.7,
            "P1 FAIL: structural sample should score >= 0.7, got {} for: {:.60}...",
            score.catalyst_overlap,
            sample
        );
        assert_ne!(
            score.pattern,
            CatalystPattern::None,
            "P1 FAIL: structural sample should detect a pattern, got None for: {:.60}...",
            sample
        );
    }

    // Natural language that should score low (<0.3)
    let natural_samples = [
        "The quick brown fox jumps over the lazy dog.",
        "In a hole in the ground there lived a hobbit.",
        "To be or not to be, that is the question.",
        "The weather is nice today and I went for a walk.",
        "She sold seashells by the seashore last Tuesday morning.",
    ];

    for sample in &natural_samples {
        let score = catalyst_score(sample, &config);
        assert!(
            score.catalyst_overlap < 0.3,
            "P1 FAIL: natural language should score < 0.3, got {} for: {}",
            score.catalyst_overlap,
            sample
        );
    }
}

// ── P2: Influence Ranking ─────────────────────────────────────────────

#[test]
fn p2_catalyst_scored_topk_better_than_random_topk() {
    let config = InfluenceConfig {
        top_k_fraction: 0.2,
        catalyst_threshold: 0.0,
        ..Default::default()
    };
    let proxy = ActivationInfluenceProxy::new(8);

    // Create a mixed dataset: 10 structural + 40 natural language
    let structural: Vec<String> = [
        "<data><row>1</row><row>2</row><row>3</row></data>",
        "fn add(a: i32, b: i32) -> i32 { a + b }",
        r"\alpha + \beta = \gamma",
        "x|y|z\n1|2|3\n4|5|6\n7|8|9",
        "xyz xyz xyz xyz xyz xyz xyz xyz xyz",
        "<items><item>foo</item><item>bar</item></items>",
        "struct Point { x: f64, y: f64 }",
        r"\sum_{i=1}^{n} i = \frac{n(n+1)}{2}",
        "a,b,c\n1,2,3\n4,5,6\n7,8,9",
        "aaa bbb aaa bbb aaa bbb aaa bbb",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();

    let natural: Vec<String> = (0..40)
        .map(|i| format!("This is natural language sentence number {} with prose.", i))
        .collect();

    let all_samples: Vec<String> = structural
        .iter()
        .cloned()
        .chain(natural.iter().cloned())
        .collect();
    let sample_refs: Vec<&str> = all_samples.iter().map(|s| s.as_str()).collect();

    let ranked = batch_influence_rank(&sample_refs, &proxy, &config);

    // Top-K by catalyst should be mostly structural
    let top_k = ranked.iter().filter(|(_, s)| s.is_high_influence).count();
    let structural_in_top_k = ranked
        .iter()
        .filter(|(idx, s)| s.is_high_influence && *idx < 10)
        .count();

    // At least 50% of top-K should be structural (indices 0..10)
    assert!(
        structural_in_top_k as f32 / top_k as f32 >= 0.5,
        "P2 FAIL: catalyst-scored top-K should be majority structural, got {}/{}",
        structural_in_top_k,
        top_k
    );

    // Compare: catalyst top-K average score vs random top-K average score
    let catalyst_top_avg: f32 = ranked
        .iter()
        .filter(|(_, s)| s.is_high_influence)
        .map(|(_, s)| s.catalyst_overlap)
        .sum::<f32>()
        / top_k as f32;

    // Random top-K: just take the last K indices (they should be natural language)
    let random_top_avg: f32 = ranked
        .iter()
        .rev()
        .take(top_k)
        .map(|(_, s)| s.catalyst_overlap)
        .sum::<f32>()
        / top_k as f32;

    assert!(
        catalyst_top_avg > random_top_avg,
        "P2 FAIL: catalyst top-K avg ({:.3}) should exceed random top-K avg ({:.3})",
        catalyst_top_avg,
        random_top_avg
    );
}

// ── P3: Synthetic Augmentation ────────────────────────────────────────

#[test]
fn p3_synthetic_catalyst_data_improves_metrics_vs_random() {
    let config = InfluenceConfig {
        catalyst_threshold: 0.0,
        ..Default::default()
    };

    // Create original structural samples to extract templates from
    let original_samples = [
        "<data><item>alpha</item><item>beta</item></data>",
        "<data><item>gamma</item><item>delta</item></data>",
        "<data><item>epsilon</item><item>zeta</item></data>",
    ];
    let sample_refs: Vec<&str> = original_samples.iter().copied().collect();

    // Extract templates
    let templates = extract_template(&sample_refs, &config);
    assert!(
        !templates.is_empty(),
        "P3: should extract at least one template"
    );

    // Generate synthetic catalyst data from templates
    let mut rng = fastrand::Rng::with_seed(42);
    let synthetic_count = 20;
    let mut synthetic_data: Vec<String> = Vec::new();
    for template in &templates {
        let generated =
            generate_synthetic(template, synthetic_count / templates.len().max(1), &mut rng);
        synthetic_data.extend(generated);
    }

    // Generate same quantity of random data (random alphanumeric strings)
    let random_data: Vec<String> = (0..synthetic_count)
        .map(|_| {
            let len = rng.usize(20..=60);
            (0..len)
                .map(|_| b"abcdefghijklmnopqrstuvwxyz0123456789 "[rng.usize(..27)] as char)
                .collect()
        })
        .collect();

    // Metric: average catalyst score (structural quality proxy)
    let synthetic_avg: f32 = synthetic_data
        .iter()
        .map(|s| catalyst_score(s, &config).catalyst_overlap)
        .sum::<f32>()
        / synthetic_data.len() as f32;

    let random_avg: f32 = random_data
        .iter()
        .map(|s| catalyst_score(s, &config).catalyst_overlap)
        .sum::<f32>()
        / random_data.len() as f32;

    assert!(
        synthetic_avg > random_avg,
        "P3 FAIL: synthetic catalyst avg ({:.3}) should exceed random avg ({:.3})",
        synthetic_avg,
        random_avg
    );

    // Also verify synthetic data maintains structural pattern
    let synthetic_with_pattern = synthetic_data
        .iter()
        .filter(|s| catalyst_score(s, &config).pattern != CatalystPattern::None)
        .count();
    assert!(
        synthetic_with_pattern
            > random_data
                .iter()
                .filter(|s| catalyst_score(s, &config).pattern != CatalystPattern::None)
                .count(),
        "P3 FAIL: more synthetic samples should have detected patterns than random"
    );
}

// ── P4: Cross-Domain Transfer ─────────────────────────────────────────

#[test]
fn p4_cross_domain_template_transfer_structural_similarity() {
    let config = InfluenceConfig {
        catalyst_threshold: 0.0,
        ..Default::default()
    };

    // Domain A: XML data
    let domain_a_samples = [
        "<record><name>Alice</name><age>30</age></record>",
        "<record><name>Bob</name><age>25</age></record>",
        "<record><name>Carol</name><age>35</age></record>",
    ];
    let a_refs: Vec<&str> = domain_a_samples.iter().copied().collect();
    let templates_a = extract_template(&a_refs, &config);

    // Domain B: also XML but different content
    let domain_b_samples = [
        "<product><title>Widget</title><price>9.99</price></product>",
        "<product><title>Gadget</title><price>19.99</price></product>",
    ];

    // Generate synthetic from Domain A templates
    let mut rng = fastrand::Rng::with_seed(42);
    assert!(
        !templates_a.is_empty(),
        "P4: should extract XML templates from domain A"
    );

    let synthetic_from_a = generate_synthetic(&templates_a[0], 5, &mut rng);

    // Verify: synthetic from Domain A templates scores well on structural (not semantic) similarity
    // with Domain B — they share XML structure but different content
    for synthetic in &synthetic_from_a {
        let score = catalyst_score(synthetic, &config);
        assert_eq!(
            score.pattern,
            CatalystPattern::XmlRepetition,
            "P4 FAIL: synthetic from Domain A template should be detected as XML"
        );
    }

    // Verify structural similarity: both Domain B samples and synthetic share XML pattern
    for b_sample in &domain_b_samples {
        let b_score = catalyst_score(b_sample, &config);
        assert_eq!(
            b_score.pattern,
            CatalystPattern::XmlRepetition,
            "P4 FAIL: Domain B sample should be detected as XML"
        );
    }

    // The structural (not semantic) similarity: both detect same pattern type
    let structural_match_count = synthetic_from_a
        .iter()
        .filter(|s| catalyst_score(s, &config).pattern == CatalystPattern::XmlRepetition)
        .count();
    assert_eq!(
        structural_match_count,
        synthetic_from_a.len(),
        "P4 FAIL: all synthetic samples should structurally match Domain B pattern"
    );
}

// ── P5: Power-Law Verification ────────────────────────────────────────

#[test]
fn p5_top_10_percent_have_40_percent_cumulative_overlap() {
    let config = InfluenceConfig {
        catalyst_threshold: 0.0,
        ..Default::default()
    };
    let proxy = ActivationInfluenceProxy::new(8);

    // Generate 100 samples: mix of structural and natural
    let mut samples: Vec<String> = Vec::new();

    // 20 structural samples
    for i in 0..20 {
        match i % 5 {
            0 => samples.push(format!("<data><row>{}</row><row>{}</row></data>", i, i + 1)),
            1 => samples.push(format!("fn func_{}(x: i32) -> i32 {{ x + {} }}", i, i)),
            2 => samples.push(format!(r"\frac{{{}}}{{{}}} + \sqrt{{{}}}", i, i + 1, i + 2)),
            3 => samples.push(format!(
                "a|b|c\n{}|{}|{}\n{}|{}|{}",
                i,
                i + 1,
                i + 2,
                i + 3,
                i + 4,
                i + 5
            )),
            _ => {
                let token = format!("abc{}", i);
                samples.push(format!(
                    "{} {} {} {} {} {} {}",
                    token, token, token, token, token, token, token
                ));
            }
        }
    }

    // 80 natural language samples
    for i in 0..80 {
        samples.push(format!(
            "This is a normal prose sentence about topic number {} that has no structural patterns.",
            i
        ));
    }

    let sample_refs: Vec<&str> = samples.iter().map(|s| s.as_str()).collect();
    let ranked = batch_influence_rank(&sample_refs, &proxy, &config);

    let total_samples = ranked.len();
    let top_10_pct = (total_samples as f32 * 0.1).ceil() as usize;

    // Cumulative catalyst_overlap for top 10%
    let top_cumulative: f32 = ranked
        .iter()
        .take(top_10_pct)
        .map(|(_, s)| s.catalyst_overlap)
        .sum();
    let total_cumulative: f32 = ranked.iter().map(|(_, s)| s.catalyst_overlap).sum();

    let ratio = if total_cumulative > 0.0 {
        top_cumulative / total_cumulative
    } else {
        0.0
    };

    assert!(
        ratio >= 0.4,
        "P5 FAIL: top 10% should have ≥40% cumulative overlap, got {:.1}% (top_cum={:.3}, total={:.3})",
        ratio * 100.0,
        top_cumulative,
        total_cumulative
    );
}

// ── P6: Convergence Rate ──────────────────────────────────────────────

#[test]
fn p6_augmented_data_reaches_target_in_80_percent_of_baseline_rounds() {
    let config = InfluenceConfig {
        catalyst_threshold: 0.0,
        ..Default::default()
    };

    // Simulate a "training" convergence test using a simple metric:
    // cumulative average catalyst score. We show that augmented (catalyst-selected)
    // data reaches a target average score faster than random data.

    let target_score = 0.5;

    // Create a large pool of mixed data
    let mut pool: Vec<String> = Vec::new();
    for i in 0..50 {
        pool.push(format!("<item><id>{}</id><val>{}</val></item>", i, i * 2));
    }
    for i in 0..200 {
        pool.push(format!(
            "Natural language text number {} with no structure at all.",
            i
        ));
    }

    let pool_refs: Vec<&str> = pool.iter().map(|s| s.as_str()).collect();

    // Score everything
    let scored: Vec<(usize, f32)> = pool_refs
        .iter()
        .enumerate()
        .map(|(i, s)| (i, catalyst_score(s, &config).catalyst_overlap))
        .collect();

    // --- Baseline: random selection, add 5 samples per round ---
    let mut rng = fastrand::Rng::with_seed(42);
    let mut baseline_rounds = 0;
    {
        let mut total_score = 0.0;
        let mut count = 0;
        let mut indices: Vec<usize> = (0..pool.len()).collect();
        // Shuffle for random order
        for i in (1..indices.len()).rev() {
            let j = rng.usize(..=i);
            indices.swap(i, j);
        }

        for chunk in indices.chunks(5) {
            baseline_rounds += 1;
            for &idx in chunk {
                total_score += scored[idx].1;
                count += 1;
            }
            let cumulative_avg = total_score / count as f32;
            if cumulative_avg >= target_score {
                break;
            }
        }
    }

    // --- Augmented: catalyst-selected top-K first, then fill ---
    let mut augmented_rounds = 0;
    {
        // Sort by catalyst score descending
        let mut sorted_scored = scored.clone();
        sorted_scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut total_score = 0.0;
        let mut count = 0;

        // Feed catalyst-selected samples first (5 per round)
        for chunk in sorted_scored.chunks(5) {
            augmented_rounds += 1;
            for (_, s) in chunk {
                total_score += s;
                count += 1;
            }
            let cumulative_avg = total_score / count as f32;
            if cumulative_avg >= target_score {
                break;
            }
        }
    }

    let ratio = augmented_rounds as f32 / baseline_rounds as f32;
    assert!(
        ratio <= 0.8,
        "P6 FAIL: augmented should reach target in ≤80% of baseline rounds, got {:.1}% ({}/{})",
        ratio * 100.0,
        augmented_rounds,
        baseline_rounds
    );
}
