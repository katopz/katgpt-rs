#![cfg(feature = "tf_loop")]
//! Issue 1241 T1 — the `copy_recall` fixture's gates (arXiv:2611.13378 §5.3
//! + App. H; fixture + scorer in `tests/common/copy_recall.rs`).
//!
//! 1. **Scorer liveness (no model):** an oracle copier reads the full copy
//!    gain (exact 1.0, ≈ 0 bits on the copy span vs the control's miss) and
//!    a uniform predictor reads exactly zero gain — the scorer can SEE recall
//!    and invents none.
//! 2. **Offset discipline:** fewer than 3 offsets is refused; the same block
//!    lands at a different absolute position per offset.
//! 3. **The decorative-loop control (the pre-registered bar):** the
//!    single-token-bench posture — random weights, the production tf_loop
//!    window at k = 4 — reads ≈ 0 copy gain against the MATCHED-POSITION
//!    control stream, offset-averaged, within 3 standard errors. A decorative
//!    loop showing recall gain would mean the fixture measures format
//!    continuation, not cross-position state (the falsifier).
//!
//! A real gain > 0 on a contraction-trained checkpoint is the fixture's
//! liveness proof — Issue 1241 T4, gated on riir-train Issue 706.
//!
//! ```bash
//! cargo test --release --features tf_loop --test issue_1241_t1_copy_recall -- --nocapture
//! ```

#[path = "common/copy_recall.rs"]
mod copy_recall;

use copy_recall::{CopyRecallFixture, MIN_OFFSETS, OffsetAveraged, RecallScore};
use katgpt_rs::transformer::{
    ForwardContext, MultiLayerKVCache, TransformerWeights, forward_training_free_loop,
};
use katgpt_rs::types::{
    CacheStrategy, Config, IterationMode, Rng, SubStepStrategy, TrainingFreeLoopConfig,
};

/// The issue's block: a seeded 1024-token block repeated at 1024–2047
/// (shifted by the offset).
const BLOCK: usize = 1024;
/// Three offsets (the ≥ 3 floor) — the block's absolute position moves.
const OFFSETS: [usize; MIN_OFFSETS] = [0, 37, 101];
const SEED: u64 = 1241;

/// Decorative-control bar, in standard errors of the gain: a model with no
/// recall reads |mean copy gain| ≤ Z·SE on both axes. Re-registered with
/// the matched control (the first draft's fixed 0.05-bit band was set
/// against the confounded first-span comparison, and the repeat-vs-control
/// difference is over DIFFERENT tokens — its noise must set the bar).
const CONTROL_Z: f64 = 3.0;

// ── 1. scorer liveness ──────────────────────────────────────────────

/// Logits that put `peak` on one token and 0 elsewhere.
fn peaked(vocab: usize, at: usize, peak: f32) -> Vec<f32> {
    let mut row = vec![0.0; vocab];
    row[at] = peak;
    row
}

#[test]
fn i1241_t1_scorer_sees_an_oracle_copy_and_invents_no_gain() {
    let vocab = 27;
    let fx = CopyRecallFixture::new(SEED, vocab, 64, 5);
    let n = fx.tokens.len();
    // Oracle copier: past the first block it predicts token t from t − block
    // — on each stream's OWN history (the control's history has no copy).
    let oracle = |stream: &[usize]| -> Vec<Vec<f32>> {
        (0..n - 1)
            .map(|p| {
                let t = p + 1;
                if t >= fx.offset + fx.block {
                    peaked(vocab, stream[t - fx.block], 40.0)
                } else {
                    vec![0.0; vocab]
                }
            })
            .collect()
    };
    let r = RecallScore::of(&fx, &oracle(&fx.tokens), &oracle(&fx.control));
    assert_eq!(r.copy.exact, 1.0);
    assert!(r.copy.bits < 1e-12, "{}", r.copy.bits);
    assert!(r.control.exact < 0.2, "the control has nothing to copy");
    assert!(r.gain_exact() > 0.8 && r.gain_bits() > 10.0, "{r:?}");
    assert!(
        r.gain_bits() > CONTROL_Z * r.gain_bits_se(),
        "an oracle clears the bar"
    );

    // A uniform predictor: both streams read log2(V) bits → exactly zero gain.
    let flat = vec![vec![0.0f32; vocab]; n - 1];
    let r = RecallScore::of(&fx, &flat, &flat);
    let uniform_bits = (vocab as f64).log2();
    assert!((r.copy.bits - uniform_bits).abs() < 1e-12);
    assert_eq!(r.gain_bits(), 0.0);
    assert!(r.copy.bits_se < 1e-12, "constant bits have no spread");
}

// ── 2. offset discipline ────────────────────────────────────────────

#[test]
fn i1241_t1_offsets_move_the_same_block_and_fewer_than_three_refuse() {
    let a = CopyRecallFixture::new(SEED, 27, 32, 0);
    let b = CopyRecallFixture::new(SEED, 27, 32, 7);
    assert_eq!(a.tokens[..32], b.tokens[7..39], "the same block, moved");
    assert_eq!(
        b.tokens[7..39],
        b.tokens[39..71],
        "the copy repeats the block"
    );
    assert_eq!(b.copy_span(), 40..71);
    assert_eq!(
        b.control[..39],
        b.tokens[..39],
        "the control shares the prefix"
    );
    assert_ne!(
        b.control[39..71],
        b.tokens[39..71],
        "the control does not repeat"
    );
    let flat = vec![vec![0.0; 27]; a.tokens.len() - 1];
    let one = RecallScore::of(&a, &flat, &flat);
    assert!(OffsetAveraged::of(vec![one; MIN_OFFSETS - 1]).is_none());
    let avg = OffsetAveraged::of(vec![one; MIN_OFFSETS]).expect("three offsets");
    assert_eq!(avg.spread_gain_bits, 0.0);
}

// ── 3. the decorative-loop control ──────────────────────────────────

fn decode(
    config: &Config,
    weights: &TransformerWeights,
    tf: &TrainingFreeLoopConfig,
    tokens: &[usize],
) -> Vec<Vec<f32>> {
    let mut ctx = ForwardContext::new(config);
    let mut cache = MultiLayerKVCache::new(config);
    tokens[..tokens.len() - 1]
        .iter()
        .enumerate()
        .map(|(pos, &tok)| {
            forward_training_free_loop(
                &mut ctx,
                weights,
                &mut cache,
                tok,
                pos,
                config,
                tf,
                #[cfg(feature = "loop_guidance")]
                None,
            )
            .to_vec()
        })
        .collect()
}

#[test]
fn i1241_t1_decorative_loop_reads_no_copy_gain() {
    let mut config = Config::micro();
    config.n_layer = 3;
    config.block_size = OFFSETS.iter().max().unwrap() + 2 * BLOCK;
    let weights = TransformerWeights::new(&config, &mut Rng::new(SEED));
    let tf = TrainingFreeLoopConfig {
        window_start: 1,
        window_end: 1,
        loop_count: 4,
        strategy: SubStepStrategy::KStageRK { beta: 0.5 },
        iteration_mode: IterationMode::Block,
        cache_strategy: CacheStrategy::Last,
    };
    let per_offset: Vec<RecallScore> = OFFSETS
        .iter()
        .map(|&off| {
            let fx = CopyRecallFixture::new(SEED, config.vocab_size, BLOCK, off);
            let logits = decode(&config, &weights, &tf, &fx.tokens);
            let control = decode(&config, &weights, &tf, &fx.control);
            let r = RecallScore::of(&fx, &logits, &control);
            println!(
                "[i1241 T1] offset {off:>3}: copy exact {:.4} bits {:.4} | control exact {:.4} bits {:.4} | gain exact {:+.4}±{:.4} bits {:+.4}±{:.4}",
                r.copy.exact, r.copy.bits, r.control.exact, r.control.bits,
                r.gain_exact(), r.gain_exact_se(), r.gain_bits(), r.gain_bits_se()
            );
            r
        })
        .collect();
    let avg = OffsetAveraged::of(per_offset).expect("three offsets");
    println!(
        "[i1241 T1] decorative k=4 control: gain exact {:+.4} (spread {:.4}, se {:.4}), bits {:+.4} (spread {:.4}, se {:.4})",
        avg.mean_gain_exact,
        avg.spread_gain_exact,
        avg.se_gain_exact,
        avg.mean_gain_bits,
        avg.spread_gain_bits,
        avg.se_gain_bits
    );
    assert!(
        avg.mean_gain_exact.abs() <= CONTROL_Z * avg.se_gain_exact,
        "a decorative loop shows exact-match recall gain — the fixture measures format continuation"
    );
    assert!(
        avg.mean_gain_bits.abs() <= CONTROL_Z * avg.se_gain_bits,
        "a decorative loop shows bits recall gain — the fixture measures format continuation"
    );
}
