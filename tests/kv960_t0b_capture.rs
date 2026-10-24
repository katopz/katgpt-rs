//! Issue 960 T0b — the real-forward run half of the KV-quant behavioral-subspace
//! diagnostic (Research 633 / arXiv:2606.09864 PCR translation).
//!
//! Runs ~20 seeded prompts through the REAL forward (`forward_base`) with the
//! `kv_capture` sink installed, feeds the captured per-layer Q/K/V to the T0a
//! probe (`katgpt-spectral::kv_sensitivity`, reduction mode), and answers the
//! two T0b questions:
//!
//! 1. **Rank stability (G1 analog)** — half-split prompt resamples rank channels
//!    the same way (Spearman between the two halves' accumulated-damage ranks).
//! 2. **Structural correlation (the T3 finding)** — does the K-side behavioral
//!    damage ranking correlate with the shipped structural-rank weighting,
//!    projected per raw channel as `Σ_i bits_i · R[i][c]²` (the calibrated
//!    eigenbasis' bit-weighted column energy: `bits_i` = the shipped
//!    two-regime ladder, b_high on the top-`d_eff` rotated dims, b_low on the
//!    tail)? Reported, never gated — "correlate or not" IS the finding.
//!
//! Deterministic by construction: fixed seeds for weights and prompts; the
//! probe is the T0a instrument (fixed-order folds, LOCAL-ONLY reclassification
//! per Issue 940). The thresholds asserted below are pinned from the measured
//! run recorded in `.benchmarks/954_kv960_t0b_capture.md` with margins.
//!
//! Box state at the pinned run: M3 Max, AC power, `--release`, calibration
//! diagnostic — NOT a perf claim (issue 960 G4 exempts the probe from
//! steady-state standards).

use katgpt_core::types::{Config, Rng};
use katgpt_forward::{ForwardContext, KvCaptureBuf, forward_base};
use katgpt_spectral::kv_sensitivity::{
    KeyTensors, ProbeConfig, ProbeScratch, ValueTensors, probe_key_channels, probe_value_channels,
};
use katgpt_spectral::spectral::{BitAllocator, calibrate_eigenbasis};
use katgpt_transformer::{MultiLayerKVCache, TransformerWeights};

/// Prompt count — the paper's ~20-prompt calibration budget.
const N_PROMPTS: usize = 20;
/// Positions per prompt (the causal prefix every query attends over).
const SEQ: usize = 40;
/// Probe bit width (the paper's low-bit regime where the collapse law binds).
const PROBE_BITS: u32 = 2;
/// Average bits for the structural ladder projection (the weighting's SHAPE
/// is set by d_eff + the eigenbasis; this only scales the two regimes).
const STRUCTURAL_AVG_BITS: f32 = 4.0;
/// Weight-seed / prompt-seed base. Fixed = the run is deterministic.
const WEIGHT_SEED: u64 = 960;
const PROMPT_SEED_BASE: u64 = 1000;

/// Small-but-real GQA config: 4 layers, 6 query heads over 3 kv heads
/// (2:1 groups), head_dim 16, seq 40. Every n_embd / head relationship the
/// forward asserts holds (n_embd = n_head × head_dim, kv_dim = n_kv_head × hd).
fn probe_config() -> Config {
    let mut config = Config::micro();
    config.vocab_size = 128;
    config.block_size = 48;
    config.n_embd = 96;
    config.n_head = 6;
    config.head_dim = 16;
    config.n_kv_head = 3;
    config.mlp_hidden = 192;
    config.n_layer = 4;
    config
}

/// Spearman rank correlation between two same-length slices. Ordinal ranks,
/// descending by value, index tie-break (deterministic; exact f32 ties are
/// measure-zero here). The degenerate constant case returns 0.0, not NaN.
fn spearman(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "spearman: length mismatch");
    let n = a.len();
    if n < 2 {
        return 0.0;
    }
    let rank = |v: &[f32]| -> Vec<f64> {
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&i, &j| v[j].total_cmp(&v[i]).then(i.cmp(&j)));
        let mut ranks = vec![0.0f64; n];
        for (r, &i) in idx.iter().enumerate() {
            ranks[i] = r as f64;
        }
        ranks
    };
    let ra = rank(a);
    let rb = rank(b);
    let n_f = n as f64;
    let ma = ra.iter().sum::<f64>() / n_f;
    let mb = rb.iter().sum::<f64>() / n_f;
    let mut cov = 0.0;
    let mut va = 0.0;
    let mut vb = 0.0;
    for i in 0..n {
        let da = ra[i] - ma;
        let db = rb[i] - mb;
        cov += da * db;
        va += da * da;
        vb += db * db;
    }
    if va == 0.0 || vb == 0.0 {
        return 0.0;
    }
    cov / (va.sqrt() * vb.sqrt())
}

#[test]
fn kv960_t0b_real_forward_capture_rank_and_correlate() {
    let config = probe_config();
    let n = config.n_embd;
    let hd = config.head_dim;
    let n_layer = config.n_layer;
    let n_kv = config.n_kv_head;
    assert_eq!(n % config.n_head, 0, "n_embd must divide into heads");
    assert_eq!(
        config.n_head % n_kv,
        0,
        "this harness assumes whole GQA groups"
    );
    let group = config.n_head / n_kv; // query heads per kv head
    let n_q = group * SEQ; // query rows per (layer, kv head)

    let weights = TransformerWeights::new(&config, &mut Rng::new(WEIGHT_SEED));
    let mut ctx = ForwardContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);
    let mut scratch = ProbeScratch::default();
    scratch.ensure(n_q, SEQ, 1);
    let probe_cfg = ProbeConfig {
        bits: PROBE_BITS,
        top_k: 1,
    };

    // Per (layer, kv head, channel): accumulated damage — full corpus, first
    // half only (second half = full − first), and the captured K/V samples
    // for the structural calibration.
    let zeros = || vec![vec![vec![0.0f32; hd]; n_kv]; n_layer];
    let mut k_full = zeros();
    let mut v_full = zeros();
    let mut k_first = zeros();
    let mut v_first = zeros();
    let mut k_samples: Vec<Vec<Vec<Vec<f32>>>> = vec![vec![vec![]; n_kv]; n_layer];
    let cap = n_layer * SEQ * 8; // generous: never binds at this shape

    for prompt in 0..N_PROMPTS {
        let mut rng = Rng::new(PROMPT_SEED_BASE + prompt as u64);
        let tokens: Vec<usize> = (0..SEQ)
            .map(|_| (rng.next() % config.vocab_size as u64) as usize)
            .collect();

        ctx.kv_capture = Some(KvCaptureBuf {
            rows: Vec::new(),
            max_rows: Some(cap),
        });
        for (pos, &tok) in tokens.iter().enumerate() {
            forward_base(
                &mut ctx, &weights, &mut cache, tok, pos, &config, None, None,
            );
        }
        let buf = ctx.kv_capture.take().expect("capture installed");
        assert_eq!(
            buf.rows.len(),
            n_layer * SEQ,
            "capture must hold exactly one row per (layer, position)"
        );

        let first_half = prompt < N_PROMPTS / 2;
        for layer in 0..n_layer {
            for kvh in 0..n_kv {
                // Per-kv-head key/value matrices [SEQ × hd] for this prompt.
                let mut k_mat = vec![0.0f32; SEQ * hd];
                let mut v_mat = vec![0.0f32; SEQ * hd];
                for row in buf.rows.iter().filter(|r| r.layer == layer) {
                    let off = row.pos * hd;
                    k_mat[off..off + hd].copy_from_slice(&row.k[kvh * hd..(kvh + 1) * hd]);
                    v_mat[off..off + hd].copy_from_slice(&row.v[kvh * hd..(kvh + 1) * hd]);
                }
                for pos in 0..SEQ {
                    k_samples[layer][kvh].push(k_mat[pos * hd..(pos + 1) * hd].to_vec());
                }

                // Query matrix: every query head in the group, every position.
                let mut q_mat = vec![0.0f32; n_q * hd];
                let mut qi = 0usize;
                for h in (kvh * group)..(kvh * group + group) {
                    for row in buf.rows.iter().filter(|r| r.layer == layer) {
                        q_mat[qi * hd..qi * hd + hd].copy_from_slice(&row.q[h * hd..(h + 1) * hd]);
                        qi += 1;
                    }
                }
                assert_eq!(qi, n_q, "query row packing drifted");

                // The two damage vecs are INDEPENDENTLY sorted (damage-desc /
                // channel-asc each) — accumulate each by its own channel field.
                let k_damage = probe_key_channels(
                    KeyTensors {
                        q: &q_mat,
                        n_q,
                        k_ref: &k_mat,
                        k_dequant: None,
                        seq: SEQ,
                        d: hd,
                    },
                    &probe_cfg,
                    &mut scratch,
                );
                let v_damage = probe_value_channels(
                    ValueTensors {
                        v_ref: &v_mat,
                        v_dequant: None,
                        n_q,
                        seq: SEQ,
                        d: hd,
                        weights: None,
                        bits: PROBE_BITS,
                    },
                    &mut scratch,
                );
                for kd in &k_damage {
                    k_full[layer][kvh][kd.channel] += kd.damage;
                    if first_half {
                        k_first[layer][kvh][kd.channel] += kd.damage;
                    }
                }
                for vd in &v_damage {
                    v_full[layer][kvh][vd.channel] += vd.damage;
                    if first_half {
                        v_first[layer][kvh][vd.channel] += vd.damage;
                    }
                }
            }
        }
        cache.reset();
    }

    // Second-half accumulators (damage sums are additive across prompts).
    let k_second: Vec<Vec<Vec<f32>>> = k_full
        .iter()
        .zip(&k_first)
        .map(|(f, a)| {
            f.iter()
                .zip(a)
                .map(|(fv, av)| fv.iter().zip(av).map(|(&x, &y)| x - y).collect())
                .collect()
        })
        .collect();
    let v_second: Vec<Vec<Vec<f32>>> = v_full
        .iter()
        .zip(&v_first)
        .map(|(f, a)| {
            f.iter()
                .zip(a)
                .map(|(fv, av)| fv.iter().zip(av).map(|(&x, &y)| x - y).collect())
                .collect()
        })
        .collect();

    // Structural weighting per (layer, kv head): the shipped calibration on
    // that head's full captured key population (the `from_keys` semantics),
    // projected to raw channels as Σ_i bits_i · R[i][c]². V has no shipped
    // structural analog (SpectralQuant calibrates keys) — K side only.
    let alloc = BitAllocator::new(1, 16);
    let mut structural: Vec<Vec<Vec<f32>>> = vec![vec![vec![0.0; hd]; n_kv]; n_layer];
    for layer in 0..n_layer {
        for kvh in 0..n_kv {
            let cal = calibrate_eigenbasis(&k_samples[layer][kvh], hd);
            let d_eff = (cal.d_eff.ceil() as usize).clamp(1, hd);
            let (b_high, b_low) = alloc.allocate(cal.d_eff, STRUCTURAL_AVG_BITS, hd);
            let mut w = vec![0.0f32; hd];
            for (i, row) in cal.eigenvectors.chunks(hd).enumerate() {
                let bits_i = if i < d_eff { b_high } else { b_low };
                for (c, &rc) in row.iter().enumerate() {
                    w[c] += bits_i as f32 * rc * rc;
                }
            }
            structural[layer][kvh] = w;
        }
    }

    // Verdicts: half-split stability + structural correlation per (layer, kvh).
    let mut printed =
        String::from("| layer | kvh | K stab | V stab | K↔structural |\n|---|---|---|---|---|\n");
    let mut k_stab_sum = 0.0f64;
    let mut v_stab_sum = 0.0f64;
    let mut k_struct_sum = 0.0f64;
    for layer in 0..n_layer {
        for kvh in 0..n_kv {
            let ks = spearman(&k_full[layer][kvh], &k_second[layer][kvh]);
            let vs = spearman(&v_full[layer][kvh], &v_second[layer][kvh]);
            let kcorr = spearman(&k_full[layer][kvh], &structural[layer][kvh]);
            k_stab_sum += ks;
            v_stab_sum += vs;
            k_struct_sum += kcorr;
            printed.push_str(&format!(
                "| {layer} | {kvh} | {ks:.3} | {vs:.3} | {kcorr:.3} |\n"
            ));
        }
    }
    let cells = (n_layer * n_kv) as f64;
    let (kstab, vstab, kstruct) = (k_stab_sum / cells, v_stab_sum / cells, k_struct_sum / cells);
    println!(
        "kv960 T0b capture ({N_PROMPTS} prompts, seq {SEQ}, {n_layer} layers × {n_kv} kv-heads, bits {PROBE_BITS}):\n{printed}"
    );
    println!(
        "kv960 T0b verdicts: mean K stability {kstab:.3}, mean V stability {vstab:.3}, mean K↔structural correlation {kstruct:.3}"
    );

    // G1-analog pins (fixed seeds → stable numbers, not flaky tolerances;
    // margins recorded in the bench file). Two disjoint prompt halves must
    // rank the same channels fragile.
    assert!(
        kstab > 0.5,
        "K-side half-split rank stability collapsed: {kstab:.3}"
    );
    assert!(
        vstab > 0.5,
        "V-side half-split rank stability collapsed: {vstab:.3}"
    );
    // Sanity: the probe must discriminate (a flat damage profile would make
    // every ranking meaningless).
    let layer0 = &k_full[0][0];
    let spread = layer0.iter().cloned().fold(f32::MIN, f32::max)
        - layer0.iter().cloned().fold(f32::MAX, f32::min);
    assert!(
        spread > 0.0,
        "K damage ranking has zero spread — probe blind"
    );
}
