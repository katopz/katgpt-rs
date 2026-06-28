//! G1 correctness tests for the paired loss gap diagnostic (Plan 335 T1.10).
//!
//! Synthetic fixtures with KNOWN per-token gaps, exact filtered aggregates.
//! These are the G1 gate (correctness on a controlled fixture). The G4 gate
//! (filter amplifies gap ≥ 1.5× on a micro-GPT A/B) lands in Phase 2 on a
//! real inference path — not here.

use crate::paired_loss::{
    ClassGapReport, ClassGapRow, ClassSizeBound, CopyNGramTagger, FilterKind, PairedLossGap,
    TokenClass, TokenTagger,
};

/// Helper: compare two f32 with tolerance (paper-scale gaps are ~0.01–0.1
/// nats; 1e-6 tolerance is strict enough to catch bugs without flaking on
/// f32 rounding).
#[inline]
fn approx(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-6
}

// ── T1.3 from_log_probs: exact per-token Δ_i ──────────────────────────────

#[test]
fn from_log_probs_exact_deltas() {
    let a = [2.0f32, 1.0, 3.0, 2.0, 1.0, 4.0, 3.0, 2.0];
    let b = [1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let gap = PairedLossGap::from_log_probs(&a, &b);
    let expected = [1.0f32, 0.0, 2.0, 1.0, 0.0, 3.0, 2.0, 1.0];
    assert_eq!(gap.deltas(), expected, "Δ_i = ℓ_A[i] − ℓ_B[i]");
    assert_eq!(gap.len(), 8);
    assert!(!gap.is_empty());
}

#[test]
fn from_log_probs_negative_deltas() {
    // Sign convention: Δ > 0 means A worse (B-favored). Verify negative Δ
    // when B is worse.
    let a = [1.0f32, 2.0, 3.0];
    let b = [2.0f32, 3.0, 4.0];
    let gap = PairedLossGap::from_log_probs(&a, &b);
    assert_eq!(gap.deltas(), &[-1.0f32, -1.0, -1.0]);
}

#[test]
#[should_panic(expected = "must have equal length")]
fn from_log_probs_unequal_lengths_panics() {
    let _ = PairedLossGap::from_log_probs(&[1.0, 2.0], &[1.0]);
}

#[test]
fn from_log_probs_empty_traces() {
    let gap = PairedLossGap::from_log_probs(&[] as &[f32], &[] as &[f32]);
    assert!(gap.is_empty());
    assert_eq!(gap.len(), 0);
    assert_eq!(gap.mean_gap(), 0.0, "empty → 0.0 (not NaN)");
}

// ── T1.4 mean_gap: aggregate Δ̄ ────────────────────────────────────────────

#[test]
fn mean_gap_all_tokens() {
    let a = [2.0f32, 1.0, 3.0, 2.0, 1.0, 4.0, 3.0, 2.0];
    let b = [1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let gap = PairedLossGap::from_log_probs(&a, &b);
    // (1+0+2+1+0+3+2+1)/8 = 10/8 = 1.25
    assert!(approx(gap.mean_gap(), 1.25), "got {}", gap.mean_gap());
}

#[test]
fn mean_gap_zero_when_traces_identical() {
    let a = [1.5f32, 2.5, 3.5];
    let gap = PairedLossGap::from_log_probs(&a, &a);
    assert!(approx(gap.mean_gap(), 0.0));
}

#[test]
fn mean_gap_uniform_shift() {
    // Constant Δ → mean = the constant.
    let a = [3.0f32, 3.0, 3.0, 3.0];
    let b = [1.0f32, 1.0, 1.0, 1.0];
    let gap = PairedLossGap::from_log_probs(&a, &b);
    assert!(approx(gap.mean_gap(), 2.0));
}

// ── T1.5 mean_gap_for_class: tag-stratified raw means ─────────────────────

/// The canonical G1 fixture: 8 positions with mixed classes and known deltas.
fn g1_fixture() -> (PairedLossGap, [TokenClass; 8]) {
    // a − b = [1, 0, 2, 1, 0, 3, 2, 1]
    let a = [2.0f32, 1.0, 3.0, 2.0, 1.0, 4.0, 3.0, 2.0];
    let b = [1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let gap = PairedLossGap::from_log_probs(&a, &b);
    let classes = [
        TokenClass::Content,    // pos 0, Δ=1
        TokenClass::Function,   // pos 1, Δ=0
        TokenClass::Content,    // pos 2, Δ=2
        TokenClass::CopyN(2),   // pos 3, Δ=1
        TokenClass::Other,      // pos 4, Δ=0
        TokenClass::Content,    // pos 5, Δ=3
        TokenClass::BracketOpen,  // pos 6, Δ=2
        TokenClass::BracketClose, // pos 7, Δ=1
    ];
    (gap, classes)
}

#[test]
fn mean_gap_for_class_content() {
    let (gap, classes) = g1_fixture();
    // Content at positions 0, 2, 5: (1+2+3)/3 = 2.0
    let m = gap.mean_gap_for_class(&classes, TokenClass::Content);
    assert!(approx(m, 2.0), "Content mean = {}, want 2.0", m);
}

#[test]
fn mean_gap_for_class_function() {
    let (gap, classes) = g1_fixture();
    // Function at position 1: 0/1 = 0.0
    let m = gap.mean_gap_for_class(&classes, TokenClass::Function);
    assert!(approx(m, 0.0), "Function mean = {}, want 0.0", m);
}

#[test]
fn mean_gap_for_class_copy_n_exact_match() {
    let (gap, classes) = g1_fixture();
    // CopyN(2) at position 3: 1/1 = 1.0
    let m = gap.mean_gap_for_class(&classes, TokenClass::CopyN(2));
    assert!(approx(m, 1.0), "CopyN(2) mean = {}, want 1.0", m);
}

#[test]
fn mean_gap_for_class_copy_n_wrong_n_is_empty() {
    let (gap, classes) = g1_fixture();
    // CopyN(5) matches no positions → 0.0
    let m = gap.mean_gap_for_class(&classes, TokenClass::CopyN(5));
    assert!(approx(m, 0.0), "CopyN(5) mean = {}, want 0.0 (empty)", m);
}

#[test]
fn mean_gap_for_class_brackets() {
    let (gap, classes) = g1_fixture();
    // BracketOpen at position 6: 2/1 = 2.0
    let m_open = gap.mean_gap_for_class(&classes, TokenClass::BracketOpen);
    assert!(approx(m_open, 2.0), "BracketOpen mean = {}", m_open);
    // BracketClose at position 7: 1/1 = 1.0
    let m_close = gap.mean_gap_for_class(&classes, TokenClass::BracketClose);
    assert!(approx(m_close, 1.0), "BracketClose mean = {}", m_close);
}

#[test]
fn mean_gap_for_class_other() {
    let (gap, classes) = g1_fixture();
    // Other at position 4: 0/1 = 0.0
    let m = gap.mean_gap_for_class(&classes, TokenClass::Other);
    assert!(approx(m, 0.0), "Other mean = {}", m);
}

#[test]
fn mean_gap_for_class_empty_returns_zero() {
    let (gap, _classes) = g1_fixture();
    // The fixture has Function at pos 1; to test the empty-bucket path we
    // need a classes array with NO Function positions (same deltas).
    let classes_no_fn = [
        TokenClass::Content,
        TokenClass::Content,
        TokenClass::Other,
        TokenClass::Other,
        TokenClass::Other,
        TokenClass::Content,
        TokenClass::BracketOpen,
        TokenClass::BracketClose,
    ];
    let m = gap.mean_gap_for_class(&classes_no_fn, TokenClass::Function);
    assert!(approx(m, 0.0), "empty bucket → 0.0, got {}", m);
}

// ── T1.6 filtered_mean: ALL / TOP-K∩NO-COPY / COPY-N-ONLY ─────────────────

#[test]
fn filtered_mean_all_tokens_equals_mean_gap() {
    let (gap, classes) = g1_fixture();
    let m_all = gap.filtered_mean(&classes, FilterKind::AllTokens);
    assert!(approx(m_all, gap.mean_gap()), "AllTokens == mean_gap");
    assert!(approx(m_all, 1.25));
}

#[test]
fn filtered_mean_copy_n_only() {
    let (gap, classes) = g1_fixture();
    // CopyN(2) at position 3: 1.0
    let m = gap.filtered_mean(&classes, FilterKind::CopyNOnly { n: 2 });
    assert!(approx(m, 1.0), "CopyNOnly(2) = {}, want 1.0", m);
}

#[test]
fn filtered_mean_copy_n_only_no_match() {
    let (gap, classes) = g1_fixture();
    // CopyN(5) matches nothing → 0.0
    let m = gap.filtered_mean(&classes, FilterKind::CopyNOnly { n: 5 });
    assert!(approx(m, 0.0), "CopyNOnly(5) = {}, want 0.0 (empty)", m);
}

#[test]
fn filtered_mean_topk_nocopy_k2() {
    let (gap, classes) = g1_fixture();
    // Candidates: Content (mean 2.0), Function (mean 0.0).
    // Top-2 = both. Selected = {Content, Function}.
    // Mask = positions 0, 1, 2, 5 (Content or Function).
    // CopyN(2) at pos 3 already excluded (not Content/Function).
    // Mean over {1.0, 0.0, 2.0, 3.0} = 6.0/4 = 1.5
    let m = gap.filtered_mean(
        &classes,
        FilterKind::TopKNoCopy {
            k: 2,
            max_ngram: 4,
        },
    );
    assert!(approx(m, 1.5), "TopKNoCopy(k=2) = {}, want 1.5", m);
}

#[test]
fn filtered_mean_topk_nocopy_k1_picks_higher_mean_class() {
    let (gap, classes) = g1_fixture();
    // k=1: only Content (mean 2.0) beats Function (mean 0.0).
    // Mask = positions 0, 2, 5 (Content only).
    // Mean over {1.0, 2.0, 3.0} = 6.0/3 = 2.0
    let m = gap.filtered_mean(
        &classes,
        FilterKind::TopKNoCopy {
            k: 1,
            max_ngram: 4,
        },
    );
    assert!(approx(m, 2.0), "TopKNoCopy(k=1) = {}, want 2.0", m);
}

#[test]
fn filtered_mean_topk_nocopy_k0_is_empty() {
    let (gap, classes) = g1_fixture();
    // k=0: select nothing → empty mask → 0.0
    let m = gap.filtered_mean(
        &classes,
        FilterKind::TopKNoCopy {
            k: 0,
            max_ngram: 4,
        },
    );
    assert!(approx(m, 0.0), "TopKNoCopy(k=0) = {}, want 0.0 (empty)", m);
}

#[test]
fn filtered_mean_topk_nocopy_excludes_copy_brackets_other() {
    // Verify TopKNoCopy does NOT pick up CopyN/Bracket/Other positions even
    // when their deltas are high. Build a fixture where a CopyN position has
    // a huge delta — it must still be excluded.
    let a = [10.0f32, 1.0]; // Δ = [9.0, 0.0]
    let b = [1.0f32, 1.0];
    let gap = PairedLossGap::from_log_probs(&a, &b);
    let classes = [TokenClass::CopyN(2), TokenClass::Content];
    // Content mean = 0.0 (only pos 1). TopK(k=2) selects {Content} (Function
    // absent → NEG_INFINITY, sorts last, k=2 still only picks Content since
    // Function has no positions). The CopyN(2) position with Δ=9.0 is
    // excluded. Mean over {0.0} = 0.0.
    let m = gap.filtered_mean(
        &classes,
        FilterKind::TopKNoCopy {
            k: 2,
            max_ngram: 4,
        },
    );
    assert!(approx(m, 0.0), "CopyN excluded: got {}, want 0.0", m);
}

#[test]
fn filtered_mean_amplifies_gap_vs_aggregate() {
    // The paper's headline finding: filtered gap > aggregate gap in |.|.
    // Our fixture: aggregate = 1.25, TopKNoCopy(k=2) = 1.5 → 1.2× amplification.
    // (The paper shows ~2× on Olmo; our synthetic fixture is arbitrary — the
    // G4 gate on a real micro-GPT A/B lands in Phase 2.)
    let (gap, classes) = g1_fixture();
    let m_all = gap.filtered_mean(&classes, FilterKind::AllTokens).abs();
    let m_topk = gap
        .filtered_mean(
            &classes,
            FilterKind::TopKNoCopy {
                k: 2,
                max_ngram: 4,
            },
        )
        .abs();
    assert!(
        m_topk >= m_all,
        "filter should amplify (or match): topk={} < all={}",
        m_topk,
        m_all
    );
}

// ── T1.7/T1.8 ClassSizeBound: Proposition 1 ────────────────────────────────

#[test]
fn class_size_bound_boolean() {
    let b = ClassSizeBound::for_vocab_size(2);
    assert!(approx(b.log_v_tau, 2.0f32.ln()));
    assert!(approx(b.log_v_tau, 0.693_147_18));
    assert!(approx(b.reducible_loss_ceiling(), b.log_v_tau));
}

#[test]
fn class_size_bound_u8() {
    let b = ClassSizeBound::for_vocab_size(256);
    assert!(approx(b.log_v_tau, 256.0f32.ln()));
    assert!(approx(b.log_v_tau, 5.545_177_4));
}

#[test]
fn class_size_bound_open_class_noun() {
    let b = ClassSizeBound::for_vocab_size(50_000);
    assert!(approx(b.log_v_tau, 50_000.0f32.ln()));
    // ln(50000) ≈ 10.8198 (verified: 10.819778284410283).
    assert!(approx(b.log_v_tau, 10.819_778));
}

#[test]
fn class_size_bound_deterministic_class_is_zero() {
    // V_τ = 1 → log 1 = 0 → no room for any richer feature (correct: a
    // deterministic class has nothing to predict).
    let b = ClassSizeBound::for_vocab_size(1);
    assert!(approx(b.log_v_tau, 0.0));
    assert!(approx(b.reducible_loss_ceiling(), 0.0));
}

#[test]
fn class_size_bound_zero_vocab_is_infinity() {
    // V_τ = 0 → undefined (log 0). Guard returns +inf (no overclaim either
    // direction). This is a degenerate input; the guard just avoids NaN.
    let b = ClassSizeBound::for_vocab_size(0);
    assert!(b.log_v_tau.is_infinite() && b.log_v_tau.is_sign_positive());
}

#[test]
fn class_size_bound_raw_vs_latent_justification() {
    // Research 319 §2.2: physical (small V_τ) → raw sufficient; semantic
    // (large V_τ) → latent earns its keep. The bound quantifies the gap.
    let physical = ClassSizeBound::for_vocab_size(2); // boolean
    let semantic = ClassSizeBound::for_vocab_size(50_000); // open-class noun
    assert!(
        physical.reducible_loss_ceiling() < semantic.reducible_loss_ceiling(),
        "physical domain should have a tighter bound than semantic"
    );
    // Physical boolean: ~0.69 nats ceiling — small.
    assert!(physical.reducible_loss_ceiling() < 1.0);
    // Semantic noun: ~10.82 nats ceiling — large, room for latent.
    assert!(semantic.reducible_loss_ceiling() > 10.0);
}

// ── T1.9 CopyNGramTagger ───────────────────────────────────────────────────

#[test]
fn copy_ngram_tagger_doc_example() {
    // The doc example: prefix = [10, 20, 10, 20, 10], n = 2.
    let tagger = CopyNGramTagger::new(2);
    let prefix = [10u32, 20, 10, 20, 10];
    // position 0: not enough context → Other
    assert_eq!(
        tagger.classify(prefix[0], 0, &prefix),
        TokenClass::Other
    );
    // position 1: n-gram [10,20], no earlier → Other
    assert_eq!(
        tagger.classify(prefix[1], 1, &prefix),
        TokenClass::Other
    );
    // position 2: n-gram [20,10], no earlier match → Other
    assert_eq!(
        tagger.classify(prefix[2], 2, &prefix),
        TokenClass::Other
    );
    // position 3: n-gram [10,20], earlier at j=0 → CopyN(2)
    assert_eq!(
        tagger.classify(prefix[3], 3, &prefix),
        TokenClass::CopyN(2)
    );
    // position 4: n-gram [20,10], earlier at j=1 → CopyN(2)
    assert_eq!(
        tagger.classify(prefix[4], 4, &prefix),
        TokenClass::CopyN(2)
    );
}

#[test]
fn copy_ngram_tagger_no_repeats() {
    let tagger = CopyNGramTagger::new(2);
    let prefix = [1u32, 2, 3, 4, 5];
    for i in 0..prefix.len() {
        assert_eq!(
            tagger.classify(prefix[i], i, &prefix),
            TokenClass::Other,
            "position {} should be Other (no repeats)",
            i
        );
    }
}

#[test]
fn copy_ngram_tagger_n3() {
    let tagger = CopyNGramTagger::new(3);
    // [1,2,3,4,1,2,3] — the 3-gram [1,2,3] appears at j=0 and again at j=4.
    let prefix = [1u32, 2, 3, 4, 1, 2, 3];
    // positions 0,1,2: not enough context or no earlier 3-gram
    for i in 0..3 {
        assert_eq!(
            tagger.classify(prefix[i], i, &prefix),
            TokenClass::Other,
            "position {} should be Other",
            i
        );
    }
    // position 3: n-gram [2,3,4], no earlier → Other
    assert_eq!(
        tagger.classify(prefix[3], 3, &prefix),
        TokenClass::Other
    );
    // position 4: n-gram [3,4,1], no earlier → Other
    assert_eq!(
        tagger.classify(prefix[4], 4, &prefix),
        TokenClass::Other
    );
    // position 5: n-gram [4,1,2], no earlier → Other
    assert_eq!(
        tagger.classify(prefix[5], 5, &prefix),
        TokenClass::Other
    );
    // position 6: n-gram [1,2,3], earlier at j=0 → CopyN(3)
    assert_eq!(
        tagger.classify(prefix[6], 6, &prefix),
        TokenClass::CopyN(3)
    );
}

#[test]
fn copy_ngram_tagger_n1_trivially_matches() {
    // n=1: a position is CopyN(1) if its token appeared earlier.
    let tagger = CopyNGramTagger::new(1);
    let prefix = [5u32, 5, 6];
    // position 0: nothing earlier → Other
    assert_eq!(tagger.classify(prefix[0], 0, &prefix), TokenClass::Other);
    // position 1: token 5 appeared at j=0 → CopyN(1)
    assert_eq!(
        tagger.classify(prefix[1], 1, &prefix),
        TokenClass::CopyN(1)
    );
    // position 2: token 6 not seen earlier → Other
    assert_eq!(tagger.classify(prefix[2], 2, &prefix), TokenClass::Other);
}

#[test]
fn copy_ngram_tagger_short_prefix() {
    let tagger = CopyNGramTagger::new(3);
    let prefix = [1u32, 2]; // len 2 < n=3
    for i in 0..prefix.len() {
        assert_eq!(
            tagger.classify(prefix[i], i, &prefix),
            TokenClass::Other,
            "position {}: prefix too short for n=3",
            i
        );
    }
}

#[test]
fn copy_ngram_tagger_n0_is_noop() {
    // n=0 is degenerate; tagger returns Other for all positions.
    let tagger = CopyNGramTagger::new(0);
    let prefix = [1u32, 2, 3];
    for i in 0..prefix.len() {
        assert_eq!(
            tagger.classify(prefix[i], i, &prefix),
            TokenClass::Other,
            "n=0 is a no-op"
        );
    }
}

#[test]
fn copy_ngram_tagger_does_not_self_match() {
    // The current occurrence must not match itself. With a unique n-gram that
    // appears exactly once, no position should be CopyN.
    let tagger = CopyNGramTagger::new(2);
    let prefix = [1u32, 2, 3];
    for i in 0..prefix.len() {
        assert_eq!(
            tagger.classify(prefix[i], i, &prefix),
            TokenClass::Other,
            "no self-match at position {}",
            i
        );
    }
}

// ── T3.1 annotate_with_class_bounds: Proposition 1 annotation ─────────────

#[test]
fn annotate_basic_fixture_per_class_means_and_ratios() {
    // Reuse the G1 fixture: classes = [Content, Function, Content, CopyN(2),
    // Other, Content, BracketOpen, BracketClose], deltas = [1,0,2,1,0,3,2,1].
    let (gap, classes) = g1_fixture();

    // Provide bounds: Content has a big V_τ (room), CopyN(2) has a small V_τ,
    // Function has a medium V_τ. Other/brackets get no bound → NaN.
    let mut bounds = std::collections::HashMap::new();
    bounds.insert(TokenClass::Content, ClassSizeBound::for_vocab_size(50_000));
    bounds.insert(TokenClass::Function, ClassSizeBound::for_vocab_size(500));
    bounds.insert(TokenClass::CopyN(2), ClassSizeBound::for_vocab_size(2));

    let report = gap.annotate_with_class_bounds(&classes, &bounds);

    // 5 distinct classes: Content, Function, CopyN(2), Other, BracketOpen,
    // BracketClose → that's 6.
    assert_eq!(report.len(), 6);
    assert!(!report.is_empty());

    // Content: positions 0,2,5 → deltas 1,2,3 → mean = 2.0. log_v_tau =
    // ln(50000) ≈ 10.8198. ratio = 2.0 / 10.8198 ≈ 0.1848.
    let content = report.row_for(TokenClass::Content).unwrap();
    assert_eq!(content.count, 3);
    assert!(approx(content.mean_gap, 2.0), "content mean = {}", content.mean_gap);
    assert!(approx(content.log_v_tau, 10.819_778));
    assert!(approx(content.gap_to_bound_ratio, 2.0 / 10.819_778));

    // Function: position 1 → delta 0 → mean = 0. ratio = 0/anything = 0.
    let function = report.row_for(TokenClass::Function).unwrap();
    assert_eq!(function.count, 1);
    assert!(approx(function.mean_gap, 0.0));
    assert!(approx(function.gap_to_bound_ratio, 0.0));

    // CopyN(2): position 3 → delta 1 → mean = 1.0. log_v_tau = ln(2) ≈ 0.693.
    // ratio = 1.0 / 0.693 ≈ 1.4427 → > 1 (exceeds bound; bound is generous, this
    // just means the A/B gap is larger than log V_τ for this tiny class).
    let copyn = report.row_for(TokenClass::CopyN(2)).unwrap();
    assert_eq!(copyn.count, 1);
    assert!(approx(copyn.mean_gap, 1.0));
    assert!(approx(copyn.log_v_tau, 0.693_147_18));
    assert!(approx(copyn.gap_to_bound_ratio, 1.0 / 0.693_147_18));

    // Other: position 4 → delta 0. No bound provided → log_v_tau = NaN,
    // ratio = NaN.
    let other = report.row_for(TokenClass::Other).unwrap();
    assert_eq!(other.count, 1);
    assert!(approx(other.mean_gap, 0.0));
    assert!(other.log_v_tau.is_nan(), "missing bound → NaN log_v_tau");
    assert!(other.gap_to_bound_ratio.is_nan(), "missing bound → NaN ratio");
}

#[test]
fn annotate_rows_sorted_by_ratio_descending_nan_last() {
    // Construct a fixture where CopyN(2) has the highest ratio (small V_τ),
    // Content has a medium ratio, Function has zero ratio, Other has no bound
    // (NaN). Sorted order: CopyN(2) > Content > Function=0 > [NaN: Other,
    // brackets].
    let (gap, classes) = g1_fixture();
    let mut bounds = std::collections::HashMap::new();
    bounds.insert(TokenClass::Content, ClassSizeBound::for_vocab_size(50_000));
    bounds.insert(TokenClass::Function, ClassSizeBound::for_vocab_size(500));
    bounds.insert(TokenClass::CopyN(2), ClassSizeBound::for_vocab_size(2));

    let report = gap.annotate_with_class_bounds(&classes, &bounds);

    let ratios: Vec<f32> = report.rows.iter().map(|r| r.gap_to_bound_ratio).collect();
    // Non-NaN ratios must be descending.
    let mut last = f32::INFINITY;
    for &r in &ratios {
        if !r.is_nan() {
            assert!(
                r <= last + 1e-6,
                "ratios not descending: {} after {}",
                r,
                last
            );
            last = r;
        }
    }
    // All NaN rows must be at the end.
    let first_nan = ratios.iter().position(|r| r.is_nan());
    let last_non_nan = ratios.iter().rposition(|r| !r.is_nan());
    if let (Some(fn_idx), Some(ln_idx)) = (first_nan, last_non_nan) {
        assert!(
            ln_idx < fn_idx,
            "NaN rows must sort last: last_non_nan={} first_nan={}",
            ln_idx,
            fn_idx
        );
    }
    // First row should be CopyN(2) (highest ratio ≈ 1.44).
    assert_eq!(report.rows[0].class, TokenClass::CopyN(2));
}

#[test]
fn annotate_empty_classes_returns_empty_report() {
    let gap = PairedLossGap::from_log_probs(&[] as &[f32], &[] as &[f32]);
    let bounds = std::collections::HashMap::new();
    let report = gap.annotate_with_class_bounds(&[] as &[TokenClass], &bounds);
    assert!(report.is_empty());
    assert_eq!(report.len(), 0);
}

#[test]
fn annotate_no_bounds_all_nan_ratios() {
    // All classes present, no bounds provided → every ratio is NaN.
    let (gap, classes) = g1_fixture();
    let bounds = std::collections::HashMap::new();
    let report = gap.annotate_with_class_bounds(&classes, &bounds);
    assert_eq!(report.len(), 6);
    for row in &report.rows {
        assert!(row.log_v_tau.is_nan());
        assert!(row.gap_to_bound_ratio.is_nan());
        // mean_gap and count are still valid.
        assert!(row.count > 0);
    }
}

#[test]
fn annotate_deterministic_class_v_tau_1_yields_nan_ratio() {
    // V_τ = 1 → log_v_tau = 0 → 0/0 = NaN (guard against division by zero for
    // deterministic classes). mean_gap should also be ~0 for a deterministic
    // class in a well-posed A/B, but we don't enforce that — the ratio is NaN
    // regardless.
    let a = [1.0f32, 2.0, 3.0];
    let b = [0.5f32, 1.5, 2.5];
    let gap = PairedLossGap::from_log_probs(&a, &b);
    let classes = [TokenClass::Content, TokenClass::Content, TokenClass::Content];
    let mut bounds = std::collections::HashMap::new();
    bounds.insert(TokenClass::Content, ClassSizeBound::for_vocab_size(1));
    let report = gap.annotate_with_class_bounds(&classes, &bounds);
    let row = report.row_for(TokenClass::Content).unwrap();
    assert!(approx(row.log_v_tau, 0.0));
    assert!(row.gap_to_bound_ratio.is_nan(), "0/0 = NaN for V_τ=1");
}

#[test]
fn annotate_zero_vocab_bound_yields_zero_ratio() {
    // V_τ = 0 → ClassSizeBound guard returns log_v_tau = +inf. finite/inf = 0.
    // (This is a degenerate input; the guard just avoids NaN propagation.)
    let a = [1.0f32, 2.0];
    let b = [0.5f32, 1.5];
    let gap = PairedLossGap::from_log_probs(&a, &b);
    let classes = [TokenClass::Content, TokenClass::Content];
    let mut bounds = std::collections::HashMap::new();
    bounds.insert(TokenClass::Content, ClassSizeBound::for_vocab_size(0));
    let report = gap.annotate_with_class_bounds(&classes, &bounds);
    let row = report.row_for(TokenClass::Content).unwrap();
    assert!(row.log_v_tau.is_infinite() && row.log_v_tau.is_sign_positive());
    assert!(approx(row.gap_to_bound_ratio, 0.0), "finite/inf = 0");
}

#[test]
fn annotate_negative_mean_gap_yields_negative_ratio() {
    // B worse than A → Δ < 0 → ratio < 0. Sign preserved ("backwards" A/B).
    let a = [1.0f32, 1.0];
    let b = [2.0f32, 2.0]; // b > a → Δ = -1
    let gap = PairedLossGap::from_log_probs(&a, &b);
    let classes = [TokenClass::Content, TokenClass::Content];
    let mut bounds = std::collections::HashMap::new();
    bounds.insert(TokenClass::Content, ClassSizeBound::for_vocab_size(10));
    let report = gap.annotate_with_class_bounds(&classes, &bounds);
    let row = report.row_for(TokenClass::Content).unwrap();
    assert!(approx(row.mean_gap, -1.0));
    assert!(approx(row.log_v_tau, 10.0f32.ln()));
    assert!(row.gap_to_bound_ratio < 0.0, "negative mean → negative ratio");
}

#[test]
fn annotate_row_for_missing_class_returns_none() {
    let (gap, classes) = g1_fixture();
    let bounds = std::collections::HashMap::new();
    let report = gap.annotate_with_class_bounds(&classes, &bounds);
    // CopyN(3) never appears in the G1 fixture (only CopyN(2)).
    assert!(report.row_for(TokenClass::CopyN(3)).is_none());
}

#[test]
fn annotate_distinct_copy_n_values_are_separate_rows() {
    // CopyN(2) and CopyN(3) are distinct HashMap keys → separate rows with
    // separate bounds. This mirrors the paper's COPY-N-ONLY filter semantics
    // (exact N).
    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let b = [0.5f32, 1.5, 2.5, 3.5, 4.5]; // all Δ = 0.5
    let gap = PairedLossGap::from_log_probs(&a, &b);
    let classes = [
        TokenClass::CopyN(2),
        TokenClass::CopyN(3),
        TokenClass::CopyN(2),
        TokenClass::CopyN(3),
        TokenClass::Content,
    ];
    let mut bounds = std::collections::HashMap::new();
    bounds.insert(TokenClass::CopyN(2), ClassSizeBound::for_vocab_size(4));
    bounds.insert(TokenClass::CopyN(3), ClassSizeBound::for_vocab_size(8));
    bounds.insert(TokenClass::Content, ClassSizeBound::for_vocab_size(1000));

    let report = gap.annotate_with_class_bounds(&classes, &bounds);
    assert_eq!(report.len(), 3, "CopyN(2), CopyN(3), Content = 3 distinct");

    let cn2 = report.row_for(TokenClass::CopyN(2)).unwrap();
    assert_eq!(cn2.count, 2);
    assert!(approx(cn2.mean_gap, 0.5));
    assert!(approx(cn2.log_v_tau, 4.0f32.ln()));

    let cn3 = report.row_for(TokenClass::CopyN(3)).unwrap();
    assert_eq!(cn3.count, 2);
    assert!(approx(cn3.mean_gap, 0.5));
    assert!(approx(cn3.log_v_tau, 8.0f32.ln()));
}

#[test]
fn annotate_classgaprow_is_copy_and_send_sync() {
    // Compile-time assertions: the report types must be Copy/Send/Sync so
    // consumers can pass them across threads and by-value.
    fn assert_copy<T: Copy>() {}
    fn assert_send_sync<T: Send + Sync>() {}
    assert_copy::<ClassGapRow>();
    assert_send_sync::<ClassGapRow>();
    assert_send_sync::<ClassGapReport>();
}
