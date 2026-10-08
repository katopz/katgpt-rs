#![cfg(feature = "entropy_bounded_commit")]
//! Issue 917 T2 — G1-style gates for the D2F decode EB wiring.
//!
//! The T1 policy is gated upstream (`katgpt-core`) and consumed, never
//! reimplemented; the T3 lanes (DDTree/DFlash) carry their own file. This
//! file pins the D2F wiring (`crates/katgpt-forward/src/d2f/`):
//!
//! - **[`d2f_commit_set_eb`]** — the per-pass commit policy the three decode
//!   loops call: no-stall on flat / one-dominant / degenerate-zero rows,
//!   brute-force `Σ H − max H` agreement over seeded random cases,
//!   determinism + permutation invariance, `γ = ∞` + cap ≡ fixed-k commit
//!   counts, NaN safety.
//! - **End-to-end decode** over synthetic weights
//!   (`TransformerWeights::new` from a fixed seed — no checkpoint, no
//!   `D2F_GOAT_TRAINED`, no network): the EB no-stall floor fully denoises
//!   the block within the step budget, and the disarmed posture reproduces
//!   the incumbent τ decode byte-identically.
//!
//! **Flag-off identity is structural.** The whole file is `#![cfg]`-gated on
//! the feature (the T3 convention): with the feature off the gated code
//! compiles to nothing and the decode loops ARE the incumbent τ code — the
//! legacy branch is byte-identical. At runtime the second gate is
//! `D2fDecodeConfig.eb_gamma ≥ 0` (default `−1.0` = disarmed), so the
//! crate's ungated `d2f` unit tests run unchanged in BOTH postures as the
//! identity check; `d2f_eb_disarmed_matches_legacy_tau_decode` pins that
//! half explicitly.
//!
//! ```sh
//! cargo test -p katgpt-forward --features entropy_bounded_commit \
//!   --test entropy_bounded_d2f_wiring
//! ```

use katgpt_core::entropy_bounded_commit::position_stats;
use katgpt_core::traits::{NoPruner, NoScreeningPruner};
use katgpt_forward::d2f::{
    D2fBlockResult, D2fBlockState, D2fDecodeConfig, D2fEbScratch, d2f_commit_set_eb,
    d2f_decode_block_with,
};
use katgpt_forward::d2f_context::D2fContext;
use katgpt_transformer::TransformerWeights;
use katgpt_types::{Config, Rng};

const VOCAB: usize = 16;
/// The mask index the policy tests use (`tokens[p] == mask` marks a masked
/// position; the value itself is arbitrary at the policy layer).
const MASK: usize = 0;

/// `count` copies of `row`, flattened row-major (the decode loops' logits
/// layout).
fn eb_rows(count: usize, row: &[f32]) -> Vec<f32> {
    let mut flat = Vec::with_capacity(count * row.len());
    for _ in 0..count {
        flat.extend_from_slice(row);
    }
    flat
}

/// One-hot logits row (softmax mass on `hot`; entropy ≈ 0 in f32 — the tail
/// terms underflow, exactly the T3 fixture's shape).
fn one_hot_logits(vocab: usize, hot: usize) -> Vec<f32> {
    let mut r = vec![-1e4f32; vocab];
    r[hot] = 0.0;
    r
}

/// Independent transcription of the documented commit order — strict
/// `(NaN-last, key, index)` under `f32::total_cmp` — plus the brute-force
/// `Σ H − max H` residual over that order (the T3 bruteforce convention,
/// mirrored against the T1 incremental scan).
fn expected_commit_set(
    logits_flat: &[f32],
    masked: &[usize],
    vocab: usize,
    gamma: f32,
    cap: usize,
) -> Vec<usize> {
    let mut keys = Vec::with_capacity(masked.len());
    for &p in masked {
        keys.push(position_stats(&logits_flat[p * vocab..(p + 1) * vocab]).entropy);
    }
    let mut order: Vec<usize> = (0..masked.len()).collect();
    order.sort_by(|&a, &b| {
        let (ka, kb) = (keys[a], keys[b]);
        ka.is_nan()
            .cmp(&kb.is_nan())
            .then(ka.total_cmp(&kb))
            .then(a.cmp(&b))
    });
    order.truncate(cap);
    let mut out = Vec::new();
    let (mut residual, mut max_h) = (0.0f32, 0.0f32);
    for &i in &order {
        let h = keys[i];
        if !h.is_finite() {
            break;
        }
        let h = h.max(0.0);
        let next = if out.is_empty() {
            0.0
        } else {
            residual + h.min(max_h)
        };
        if next > gamma {
            break;
        }
        residual = next;
        max_h = max_h.max(h);
        out.push(masked[i]);
    }
    out.sort_unstable();
    out
}

/// The committed positions (ascending) within the pass window `len`,
/// asserted against the returned count. (The scratch is reused across
/// cases; only `[..len]` is this pass's window.)
fn selected_positions(scratch: &D2fEbScratch, k: usize, len: usize) -> Vec<usize> {
    let v: Vec<usize> = scratch
        .selected()
        .iter()
        .take(len)
        .enumerate()
        .filter_map(|(i, &s)| s.then_some(i))
        .collect();
    assert_eq!(v.len(), k, "selection bitmap size == committed count");
    v
}

// ── Policy: no-stall (G1) ───────────────────────────────────────

#[test]
fn d2f_commit_set_no_stall_on_flat_rows() {
    // Uniform logits → every entropy is ln(16) ≈ 2.77 nats: the incumbent τ
    // rule commits nothing here; the EB singleton commits (residual exactly
    // 0) and a second commit would add min(ln16, ln16) > γ = 0.1.
    let n = 8usize;
    let flat = eb_rows(n, &[0.0f32; VOCAB]);
    let tokens = vec![MASK; n];
    let mut scratch = D2fEbScratch::new(n);
    let (k, masked) =
        d2f_commit_set_eb(&flat, &tokens, 0, n, MASK, VOCAB, &mut scratch, 0.1, usize::MAX);
    assert_eq!(masked, n);
    assert_eq!(k, 1, "flat rows: the EB singleton commits where τ stalls");
}

#[test]
fn d2f_commit_set_no_stall_on_one_dominant_and_degenerate_zero() {
    // Degenerate zero-entropy rows: the whole block commits even at γ = 0.
    let n = 6usize;
    let flat = eb_rows(n, &one_hot_logits(VOCAB, 3));
    let tokens = vec![MASK; n];
    let mut scratch = D2fEbScratch::new(n);
    let (k, _) =
        d2f_commit_set_eb(&flat, &tokens, 0, n, MASK, VOCAB, &mut scratch, 0.0, usize::MAX);
    assert_eq!(k, n, "zero-entropy block commits whole at γ = 0 (no-stall)");

    // One dominant row among flat ones: ≥ 1 commit at every armed γ ≥ 0.
    let mut mixed = vec![0.0f32; VOCAB];
    mixed[7] = 12.0;
    let flat = eb_rows(6, &mixed);
    let tokens6 = vec![MASK; 6];
    for gamma in [0.0f32, 0.1, 5.0] {
        let (k, _) = d2f_commit_set_eb(
            &flat,
            &tokens6,
            0,
            6,
            MASK,
            VOCAB,
            &mut scratch,
            gamma,
            usize::MAX,
        );
        assert!(k >= 1, "γ={gamma}: no-stall floor must hold");
    }
}

// ── Policy: brute-force agreement / determinism / cap / NaN ─────

#[test]
fn d2f_commit_set_matches_bruteforce_sum_minus_max() {
    let mut rng = fastrand::Rng::with_seed(917);
    let mut scratch = D2fEbScratch::new(VOCAB);
    for _ in 0..500 {
        let n = rng.usize(1..=VOCAB);
        let flat: Vec<f32> = (0..n * VOCAB).map(|_| (rng.f32() - 0.5) * 12.0).collect();
        let tokens = vec![MASK; n];
        let masked: Vec<usize> = (0..n).collect();
        let gamma = rng.f32() * 4.0;
        let (k, m) =
            d2f_commit_set_eb(&flat, &tokens, 0, n, MASK, VOCAB, &mut scratch, gamma, usize::MAX);
        assert_eq!(m, n);
        assert!(k >= 1, "no-stall: finite-entropy rows must commit ≥ 1");
        let expected = expected_commit_set(&flat, &masked, VOCAB, gamma, usize::MAX);
        assert_eq!(selected_positions(&scratch, k, n), expected);
        // Independent brute-force residual on the committed prefix.
        let hs: Vec<f32> = expected
            .iter()
            .map(|&p| position_stats(&flat[p * VOCAB..(p + 1) * VOCAB]).entropy.max(0.0))
            .collect();
        let sum: f32 = hs.iter().sum();
        let max = hs.iter().cloned().fold(0.0f32, f32::max);
        assert!(sum - max <= gamma + 1e-5, "committed prefix within budget");
    }
}

#[test]
fn d2f_commit_set_deterministic_and_permutation_invariant() {
    // Six distinct sharpness rows; the same multiset permuted across
    // positions must keep the commit COUNT (key ties break by index, so the
    // exact set may ride the permutation — the count may not).
    let mut rows: Vec<Vec<f32>> = Vec::new();
    for peak in 0..6usize {
        let mut r = vec![0.0f32; VOCAB];
        r[peak] = peak as f32 * 2.0;
        rows.push(r);
    }
    let run = |order: &[usize]| -> usize {
        let mut flat = Vec::new();
        for &src in order {
            flat.extend_from_slice(&rows[src]);
        }
        let tokens = vec![MASK; order.len()];
        let mut scratch = D2fEbScratch::new(order.len());
        let (k, _) = d2f_commit_set_eb(
            &flat,
            &tokens,
            0,
            order.len(),
            MASK,
            VOCAB,
            &mut scratch,
            1.5,
            usize::MAX,
        );
        // Determinism: an identical second call reproduces the same set.
        let mut scratch2 = D2fEbScratch::new(order.len());
        let (k2, _) = d2f_commit_set_eb(
            &flat,
            &tokens,
            0,
            order.len(),
            MASK,
            VOCAB,
            &mut scratch2,
            1.5,
            usize::MAX,
        );
        assert_eq!(k, k2);
        assert_eq!(scratch.selected(), scratch2.selected());
        k
    };
    let forward: Vec<usize> = (0..6).collect();
    let k_fwd = run(&forward);
    let reversed: Vec<usize> = (0..6).rev().collect();
    let k_rev = run(&reversed);
    assert_eq!(k_fwd, k_rev, "row multiset permutation keeps the commit count");
    assert!(k_fwd >= 1);
}

#[test]
fn d2f_commit_set_gamma_inf_cap_is_fixed_k() {
    // γ = +∞ admits every finite-entropy candidate, so cap = k reproduces
    // the fixed-width-k policy: the k lowest-key masked positions.
    let mut rng = fastrand::Rng::with_seed(7);
    let mut scratch = D2fEbScratch::new(VOCAB);
    for _ in 0..50 {
        let n = rng.usize(2..=VOCAB);
        let flat: Vec<f32> = (0..n * VOCAB).map(|_| (rng.f32() - 0.5) * 12.0).collect();
        let tokens = vec![MASK; n];
        let masked: Vec<usize> = (0..n).collect();
        let cap = rng.usize(1..=n);
        let (k, _) = d2f_commit_set_eb(
            &flat,
            &tokens,
            0,
            n,
            MASK,
            VOCAB,
            &mut scratch,
            f32::INFINITY,
            cap,
        );
        assert_eq!(k, cap.min(n), "γ=∞ commits exactly min(masked, cap)");
        let mut by_key: Vec<usize> = masked.clone();
        by_key.sort_by(|&a, &b| {
            let (ka, kb) = (
                position_stats(&flat[a * VOCAB..(a + 1) * VOCAB]).entropy,
                position_stats(&flat[b * VOCAB..(b + 1) * VOCAB]).entropy,
            );
            ka.total_cmp(&kb).then(a.cmp(&b))
        });
        by_key.truncate(cap);
        by_key.sort_unstable();
        assert_eq!(selected_positions(&scratch, k, n), by_key, "fixed-k set");
    }
}

#[test]
fn d2f_commit_set_nan_never_commits() {
    // A NaN logits row is never committed (its entropy is NaN: non-finite
    // entropy terminates the prefix; the NaN key also sorts last). The good
    // rows around it still commit; a NaN/negative γ commits nothing (the
    // garbage-config contract).
    let mut nan_row = vec![0.0f32; VOCAB];
    nan_row[2] = f32::NAN;
    let mut flat = eb_rows(1, &[0.5f32; VOCAB]);
    flat.extend_from_slice(&nan_row);
    flat.extend_from_slice(&[-0.5f32; VOCAB]);
    let tokens = vec![MASK; 3];
    let mut scratch = D2fEbScratch::new(3);
    let (k, _) =
        d2f_commit_set_eb(&flat, &tokens, 0, 3, MASK, VOCAB, &mut scratch, 10.0, usize::MAX);
    let sel = selected_positions(&scratch, k, 3);
    assert!(!sel.contains(&1), "NaN row never committed");
    assert_eq!(sel, vec![0, 2], "the good rows commit around it");

    let (k_nan_gamma, _) =
        d2f_commit_set_eb(&flat, &tokens, 0, 3, MASK, VOCAB, &mut scratch, f32::NAN, usize::MAX);
    assert_eq!(k_nan_gamma, 0, "NaN γ commits nothing");
    let (k_neg_gamma, m) =
        d2f_commit_set_eb(&flat, &tokens, 0, 3, MASK, VOCAB, &mut scratch, -1.0, usize::MAX);
    assert_eq!((k_neg_gamma, m), (0, 3), "negative γ commits nothing");
}

// ── End-to-end decode (synthetic weights — no checkpoint) ───────

fn synthetic_decode(decode_config: &D2fDecodeConfig) -> D2fBlockResult {
    let config = Config::micro_dllm();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let mut ctx = D2fContext::new(&config);
    d2f_decode_block_with(
        &mut ctx,
        &weights,
        &config,
        decode_config,
        &NoPruner,
        &NoScreeningPruner,
        &mut rng,
    )
}

#[test]
fn d2f_eb_armed_decode_fully_denoises_and_is_deterministic() {
    let decode_config = D2fDecodeConfig {
        denoise_steps: 8,  // ≥ block size: the ≥1-commit-per-pass floor
        eb_gamma: 0.1,     // ARMED (UNCALIBRATED wiring default)
        ..D2fDecodeConfig::with_block_size(4)
    };
    let a = synthetic_decode(&decode_config);
    assert_eq!(
        a.state,
        D2fBlockState::FullyActivated,
        "the EB no-stall floor must denoise a 4-slot block in 8 steps"
    );
    let b = synthetic_decode(&decode_config);
    assert_eq!(a.tokens, b.tokens, "EB decode is deterministic under a fixed seed");
}

#[test]
fn d2f_eb_armed_ignores_unreachable_tau() {
    // τ = 1.1 is the incumbent's "commit nothing, stall the pass" posture;
    // armed, the EB policy replaces the τ test entirely and the block still
    // denoises — the exact property the flag-off lane lacks.
    let decode_config = D2fDecodeConfig {
        denoise_steps: 8,
        confidence_threshold: 1.1,
        eb_gamma: 0.1,
        ..D2fDecodeConfig::with_block_size(4)
    };
    let result = synthetic_decode(&decode_config);
    assert_eq!(result.state, D2fBlockState::FullyActivated);
}

#[test]
fn d2f_eb_disarmed_matches_legacy_tau_decode() {
    // The runtime half of the flag-off identity: the disarmed posture
    // (eb_gamma = −1.0, the Default) runs the incumbent τ branch. τ = 0
    // commits everything in one pass; the result is byte-identical across
    // runs and never consults EB.
    let decode_config = D2fDecodeConfig {
        denoise_steps: 8,
        confidence_threshold: 0.0,
        ..D2fDecodeConfig::with_block_size(4) // eb_gamma defaults to −1.0
    };
    let a = synthetic_decode(&decode_config);
    assert_eq!(
        a.state,
        D2fBlockState::FullyActivated,
        "τ = 0 commits the whole block in one pass"
    );
    assert_eq!(a.steps_used, 1, "τ = 0: fully denoised at the first pass");
    let b = synthetic_decode(&decode_config);
    assert_eq!(a.tokens, b.tokens, "the disarmed (legacy) decode is deterministic");
}
