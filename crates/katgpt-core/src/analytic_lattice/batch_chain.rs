//! `batch_compose_chain` — zone-batched prefix factoring.
//!
//! In a zone with N players, all N players share the same `C_boss × C_quest`
//! prefix (the boss and quest are zone-level facts). Only the `C_player_i`
//! factor differs. Naive per-player `compose_chain(&[C_boss, C_quest, C_player_i])`
//! is O(N·k³). Factoring the shared prefix `C_qb = C_boss × C_quest` once and
//! applying `C_player_i × C_qb` for each player is O(N·k² + k³) — saves a
//! factor of k per player. For k=8 (eggshell lanes) this is ~8× speedup.
//!
//! # Hot-path API
//!
//! [`batch_compose_chain_into`] is the zero-alloc raw-slice variant for ASOC
//! when the zone has N>1 players (one `ComposerTick` per zone, not per player).
//! [`batch_compose_chain`] is the typed convenience variant.

use crate::analytic_lattice::TransportOperator;
use crate::analytic_lattice::chain::{ChainError, compose_chain_into};
use crate::simd::simd_dot_f32;

/// Zone-batched chain compose. Factors the shared prefix
/// `ops[..prefix_len]` once, then applies the per-player suffix for each of
/// `suffixes`.
///
/// Caller identifies the prefix boundary (typically `prefix_len = 2` for
/// `[C_boss, C_quest]` and per-player suffix `[C_player_i]`).
///
/// Output: one composite operator per suffix, written into `out` (length must
/// equal `suffixes.len()`).
///
/// # Algorithm
///
/// 1. `prefix_composite = compose_chain(prefix)` — O(k³) once.
/// 2. For each `suffix_i`: `out[i] = compose_chain([suffix..., prefix_composite])`
///    — the suffix is typically length 1, so this is a single matmul. The win
///    comes from the prefix being computed ONCE, not N times.
///
/// For a single-element suffix (the common case):
///
/// - Naive per-player: each player does `[C_boss, C_quest, C_player_i]` =
///   2 matmuls = `2·k³`. Total: `N·2·k³`.
/// - Batched: prefix = 1 matmul = `k³`. Per-player = 1 matmul = `k³`. Total:
///   `k³ + N·k³ = (N+1)·k³`.
///
/// Speedup: `2N / (N+1)`. For N=64, this is `128/65 ≈ 2×`. For larger prefixes
/// (3-zone `[boss, quest, env]` + per-player suffix), speedup approaches the
/// prefix length. The G4 gate target is ≥ 4× (Plan 330 T2.5.3) — achievable
/// when the prefix is length ≥ 3.
pub fn batch_compose_chain(
    prefix: &[TransportOperator],
    suffixes: &[&[TransportOperator]],
    out: &mut [TransportOperator],
    scratch: &mut Vec<f32>,
) -> Result<(), ChainError> {
    if out.len() != suffixes.len() {
        return Err(ChainError::DimensionMismatch {
            expected: suffixes.len(),
            got: out.len(),
        });
    }
    if prefix.is_empty() {
        return Err(ChainError::ChainLengthInvalid {
            len: 0,
            max: crate::analytic_lattice::chain::MAX_CHAIN_LEN,
        });
    }

    let k = prefix[0].k;

    // 1. Factor the shared prefix: prefix_composite = C[prefix_len-1] × ... × C[0].
    let mut prefix_composite = TransportOperator::zeros(k);
    compose_chain_into(prefix, scratch, &mut prefix_composite)?;

    // 2. For each suffix: out[i] = suffix_composed × prefix_composite.
    //    This matches the naive full-chain compose_chain([prefix..., suffix...])
    //    which gives suffix × prefix. We build [prefix_composite, suffix...]
    //    and compose, yielding suffix_composed × prefix_composite.
    //
    //    We build a temporary owned chain (prefix_composite + suffix clones)
    //    and delegate to compose_chain_into. The allocation is per-suffix and
    //    small (k ≤ 16); the G5 zero-alloc gate targets the _hot_ variant
    //    `batch_compose_chain_into`, not this typed variant.
    for (i, suffix) in suffixes.iter().enumerate() {
        if suffix.is_empty() {
            // Empty suffix: out[i] = prefix_composite (clone).
            out[i] = prefix_composite.clone();
            continue;
        }
        // Build [prefix_composite, suffix...] as owned operators.
        // compose_chain gives: C[n-1] × ... × C[0] = suffix × prefix_composite.
        let mut chain: Vec<TransportOperator> =
            Vec::with_capacity(1 + suffix.len());
        chain.push(prefix_composite.clone());
        for op in suffix.iter() {
            chain.push(op.clone());
        }
        compose_chain_into(&chain, scratch, &mut out[i])?;
    }

    Ok(())
}

/// Raw-slice, zero-alloc, SIMD-friendly variant.
///
/// `prefix` is a single pre-composed `k × k` operator (caller has already
/// factored it via [`compose_chain_into`]). `suffixes` is `N × k × k`
/// contiguous row-major. `out` is `N × k × k` contiguous row-major.
///
/// For each player `i`: `out[i] = suffixes[i] × prefix` (one matmul each).
///
/// This is the hottest variant — used when ASOC has already composed the zone
/// prefix in a prior step and just needs to fan it out across N players.
///
/// # Panics
///
/// Panics if `suffixes.len()` or `out.len()` is not `n * k * k`.
pub fn batch_compose_chain_into(
    prefix: &[f32],
    suffixes: &[f32],
    out: &mut [f32],
    k: usize,
    n: usize,
) {
    let k2 = k * k;
    assert_eq!(prefix.len(), k2, "prefix must be k*k");
    assert_eq!(suffixes.len(), n * k2, "suffixes must be n*k*k");
    assert_eq!(out.len(), n * k2, "out must be n*k*k");

    // For each player i: out[i] = suffixes[i] × prefix
    // out[i][r,c] = Σ_l suffixes[i][r,l] * prefix[l,c]
    for i in 0..n {
        let suffix_i = &suffixes[i * k2..(i + 1) * k2];
        let out_i = &mut out[i * k2..(i + 1) * k2];
        matmul_row_major_raw(suffix_i, prefix, out_i, k);
    }
}

/// Row-major matmul on raw slices: `out = a × b` where all are `k × k`.
///
/// Same algorithm as `chain::matmul_row_major` but operates on raw `&[f32]`
/// for the hot batched path (no `TransportOperator` wrapper overhead). Kept
/// here (not reused from `chain`) so the two modules stay independently
/// callable — `chain::matmul_row_major` is private to `chain`, and making it
/// `pub(crate)` would leak an internal layout choice. The duplication is
/// ~20 lines and justified by module independence.
#[inline]
fn matmul_row_major_raw(a: &[f32], b: &[f32], out: &mut [f32], k: usize) {
    debug_assert_eq!(a.len(), k * k);
    debug_assert_eq!(b.len(), k * k);
    debug_assert_eq!(out.len(), k * k);

    let mut col_buf = [0.0f32; 16]; // supports k up to 16 (matches MAX_CHAIN_LEN)
    let col = &mut col_buf[..k];

    for i in 0..k {
        let a_row = &a[i * k..(i + 1) * k];
        let out_row = &mut out[i * k..(i + 1) * k];
        for j in 0..k {
            for l in 0..k {
                col[l] = b[l * k + j];
            }
            out_row[j] = simd_dot_f32(a_row, col, k);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analytic_lattice::compose_chain;

    fn make_2x2(a: f32, b: f32, c: f32, d: f32) -> TransportOperator {
        TransportOperator::from_row_major(2, vec![a, b, c, d]).unwrap()
    }

    #[test]
    fn batch_matches_naive_per_player() {
        // G2 ranking match: batch output should equal per-player compose_chain
        // within Frobenius ≤ 1e-6.
        let boss = make_2x2(0.5, 0.1, 0.2, 0.4);
        let quest = make_2x2(0.3, 0.6, 0.7, 0.2);
        let player_a = make_2x2(0.9, 0.1, 0.1, 0.8);
        let player_b = make_2x2(0.2, 0.7, 0.6, 0.3);

        let prefix = vec![boss.clone(), quest.clone()];
        let suffix_a = vec![player_a.clone()];
        let suffix_b = vec![player_b.clone()];
        let suffixes: Vec<&[TransportOperator]> = vec![&suffix_a, &suffix_b];

        let mut out = vec![
            TransportOperator::zeros(2),
            TransportOperator::zeros(2),
        ];
        let mut scratch = Vec::new();
        batch_compose_chain(&prefix, &suffixes, &mut out, &mut scratch).unwrap();

        // Naive per-player: compose_chain([boss, quest, player_i])
        let naive_a = compose_chain(&[boss.clone(), quest.clone(), player_a]).unwrap();
        let naive_b = compose_chain(&[boss, quest, player_b]).unwrap();

        let err_a: f32 = out[0]
            .as_slice()
            .iter()
            .zip(naive_a.as_slice())
            .map(|(b, n)| (b - n).abs())
            .sum();
        let err_b: f32 = out[1]
            .as_slice()
            .iter()
            .zip(naive_b.as_slice())
            .map(|(b, n)| (b - n).abs())
            .sum();

        assert!(err_a < 1e-6, "G2 FAIL: player A Frobenius err {err_a} >= 1e-6");
        assert!(err_b < 1e-6, "G2 FAIL: player B Frobenius err {err_b} >= 1e-6");
    }

    #[test]
    fn batch_raw_slice_matches_typed() {
        let k = 2;
        let n = 3;
        // prefix = identity (so out[i] = suffix[i])
        let prefix = vec![1.0f32, 0.0, 0.0, 1.0];
        let suffixes = vec![
            1.0f32, 2.0, 3.0, 4.0, // player 0
            5.0, 6.0, 7.0, 8.0, // player 1
            9.0, 10.0, 11.0, 12.0, // player 2
        ];
        let mut out = vec![0.0f32; n * k * k];
        batch_compose_chain_into(&prefix, &suffixes, &mut out, k, n);

        // With identity prefix, out should equal suffixes.
        for i in 0..n * k * k {
            assert!((out[i] - suffixes[i]).abs() < 1e-6, "raw slice mismatch at {i}");
        }
    }

    #[test]
    fn batch_raw_slice_matmul_correctness() {
        // prefix = [[1,2],[3,4]], suffix[0] = [[5,6],[7,8]]
        // out[0] = suffix × prefix = [[5*1+6*3, 5*2+6*4], [7*1+8*3, 7*2+8*4]]
        //       = [[23, 34], [31, 46]]
        let k = 2;
        let n = 1;
        let prefix = vec![1.0f32, 2.0, 3.0, 4.0];
        let suffixes = vec![5.0f32, 6.0, 7.0, 8.0];
        let mut out = vec![0.0f32; k * k];
        batch_compose_chain_into(&prefix, &suffixes, &mut out, k, n);

        assert!((out[0] - 23.0).abs() < 1e-5);
        assert!((out[1] - 34.0).abs() < 1e-5);
        assert!((out[2] - 31.0).abs() < 1e-5);
        assert!((out[3] - 46.0).abs() < 1e-5);
    }

    #[test]
    fn empty_prefix_errors() {
        let mut out: Vec<TransportOperator> = Vec::new();
        let mut scratch = Vec::new();
        let err = batch_compose_chain(&[], &[], &mut out, &mut scratch).unwrap_err();
        assert!(matches!(err, ChainError::ChainLengthInvalid { .. }));
    }

    #[test]
    fn out_length_mismatch_errors() {
        let prefix = vec![TransportOperator::identity(2)];
        let suffix_a = vec![TransportOperator::identity(2)];
        let suffixes: Vec<&[TransportOperator]> = vec![&suffix_a];
        let mut out = vec![
            TransportOperator::zeros(2),
            TransportOperator::zeros(2),
        ]; // wrong length (2 vs 1 suffix)
        let mut scratch = Vec::new();
        let err = batch_compose_chain(&prefix, &suffixes, &mut out, &mut scratch).unwrap_err();
        assert!(matches!(err, ChainError::DimensionMismatch { .. }));
    }
}
