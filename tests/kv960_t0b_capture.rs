//! Issue 960 T0b/T1 — the real-forward run half of the KV-quant
//! behavioral-subspace diagnostic (Research 633 / arXiv:2606.09864 PCR
//! translation).
//!
//! **T0b** runs ~20 seeded prompts through the REAL forward (`forward_base`)
//! with the `kv_capture` sink installed, feeds the captured per-layer Q/K/V to
//! the T0a probe (`katgpt-spectral::kv_sensitivity`, reduction mode), and
//! answers:
//!
//! 1. **Rank stability (G1 analog)** — half-split prompt resamples rank
//!    channels the same way (Spearman between the two halves'
//!    accumulated-damage ranks).
//! 2. **Structural correlation (the T3 finding)** — does the K-side behavioral
//!    damage ranking correlate with the shipped structural-rank weighting,
//!    projected per raw channel as `Σ_i bits_i · R[i][c]²`? Reported, never
//!    gated — "correlate or not" IS the finding.
//!
//! **T1** classifies the sensitivity profile into the paper's three PCR
//! failure modes, translated to non-safety vocabulary, and emits the predicted
//! mitigation direction per head — the design input for T2's weight lever:
//!
//! - **behavior-in-outliers** — damage co-located with wide-range channels
//!   (`spearman(damage, range) ≥ +τ`): coarse quantization of the important
//!   channels themselves. Mitigation: FINER GRANULARITY (more bits on those
//!   channels rescues).
//! - **behavior-in-quiet-channels** (the paper's inverse, surfaced by the
//!   per-channel instrument) — damage concentrated in narrow-range channels
//!   (`≤ −τ`): behaviorally-load-bearing precision lives in low-variance
//!   channels, which range/eigenvalue-driven ladders (water-fill) under-serve.
//!   Mitigation: RANGE-BLIND behavioral-damage weighting — exactly T2's
//!   premise.
//! - **multi-layer dilution** — per-layer damage-mass spread
//!   (participation index): diluted (mass flat across layers → the paper's
//!   mode 3, cross-layer allocation only) vs concentrated (one hot layer →
//!   a per-layer fix captures the mass).
//! - ⚠ **outlier-crushes-behavior is EXCLUDED BY CONSTRUCTION here**: the T0a
//!   reduction quantizer is per-channel (each column on its own min–max
//!   grid), so no shared scale exists for outliers to stretch. That mode
//!   needs the attribution path over a real backend store (T2/T3) — a
//!   per-channel-scale backend IS the mitigation for it, which the reduction
//!   instrument already bakes in.
//!
//! Deterministic by construction: fixed seeds for weights and prompts; the
//! probe is the T0a instrument (fixed-order folds, LOCAL-ONLY
//! reclassification per Issue 940). Thresholds asserted below are pinned from
//! the measured run recorded in `.benchmarks/954_kv960_t0b_capture.md`.
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
/// T1 mode-classification band: |ρ(damage, range)| below this reads diffuse
/// (no mode); above it the sign picks the mode. Pinned from the measured run.
const MODE_TAU: f64 = 0.4;
/// T1 dilution bar on the layer-mass delocalization index
/// `(PR − 1)/(n_layer − 1)` ∈ [0, 1] (PR = (Σm)²/Σm² ∈ [1, n]): 1 = mass
/// spread flat across layers (DILUTED — the paper's mode 3, per-layer fixes
/// fail), 0 = one layer carries everything (a per-layer fix on the hot layer
/// captures the mass). At or above this the model reads DILUTED.
const DILUTION_BAR: f64 = 0.5;

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

/// Everything the T0b/T1 verdicts consume, from ONE capture+probe pass.
struct CaptureRun {
    /// K-side accumulated damage per (layer, kv head, channel), full corpus.
    k_full: Vec<Vec<Vec<f32>>>,
    /// V-side accumulated context distortion, full corpus.
    v_full: Vec<Vec<Vec<f32>>>,
    /// First-prompt-half accumulators (second = full − first).
    k_first: Vec<Vec<Vec<f32>>>,
    v_first: Vec<Vec<Vec<f32>>>,
    /// Captured key samples per (layer, kv head): the calibration population.
    k_samples: Vec<Vec<Vec<Vec<f32>>>>,
    /// Shipped structural weighting projected per raw channel
    /// (`Σ_i bits_i · R[i][c]²`), K side only (SpectralQuant calibrates keys).
    structural: Vec<Vec<Vec<f32>>>,
}

/// One forward pass over all prompts with capture + per-(layer, kv head)
/// probes. Deterministic: fixed weight/prompt seeds; ~2 s release.
fn capture_and_probe(config: &Config) -> CaptureRun {
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

    let weights = TransformerWeights::new(config, &mut Rng::new(WEIGHT_SEED));
    let mut ctx = ForwardContext::new(config);
    let mut cache = MultiLayerKVCache::new(config);
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
            forward_base(&mut ctx, &weights, &mut cache, tok, pos, config, None, None);
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

    // Structural weighting per (layer, kv head): the shipped calibration on
    // that head's full captured key population (the `from_keys` semantics),
    // projected to raw channels as Σ_i bits_i · R[i][c]².
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

    CaptureRun {
        k_full,
        v_full,
        k_first,
        v_first,
        k_samples,
        structural,
    }
}

/// Per-channel min–max range of channel `c` over the captured sample rows.
fn channel_range(samples: &[Vec<f32>], c: usize) -> f32 {
    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;
    for row in samples {
        let v = row[c];
        if v < mn {
            mn = v;
        }
        if v > mx {
            mx = v;
        }
    }
    if mn == mx { 0.0 } else { mx - mn }
}

#[test]
fn kv960_t0b_real_forward_capture_rank_and_correlate() {
    let config = probe_config();
    let n_layer = config.n_layer;
    let n_kv = config.n_kv_head;
    let run = capture_and_probe(&config);

    // Second-half accumulators (damage sums are additive across prompts).
    let sub = |full: &Vec<Vec<Vec<f32>>>, first: &Vec<Vec<Vec<f32>>>| -> Vec<Vec<Vec<f32>>> {
        full.iter()
            .zip(first)
            .map(|(f, a)| {
                f.iter()
                    .zip(a)
                    .map(|(fv, av)| fv.iter().zip(av).map(|(&x, &y)| x - y).collect())
                    .collect()
            })
            .collect()
    };
    let k_second = sub(&run.k_full, &run.k_first);
    let v_second = sub(&run.v_full, &run.v_first);

    // Verdicts: half-split stability + structural correlation per (layer, kvh).
    let mut printed =
        String::from("| layer | kvh | K stab | V stab | K↔structural |\n|---|---|---|---|---|\n");
    let mut k_stab_sum = 0.0f64;
    let mut v_stab_sum = 0.0f64;
    let mut k_struct_sum = 0.0f64;
    for layer in 0..n_layer {
        for kvh in 0..n_kv {
            let ks = spearman(&run.k_full[layer][kvh], &k_second[layer][kvh]);
            let vs = spearman(&run.v_full[layer][kvh], &v_second[layer][kvh]);
            let kcorr = spearman(&run.k_full[layer][kvh], &run.structural[layer][kvh]);
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
    let layer0 = &run.k_full[0][0];
    let spread = layer0.iter().cloned().fold(f32::MIN, f32::max)
        - layer0.iter().cloned().fold(f32::MAX, f32::min);
    assert!(
        spread > 0.0,
        "K damage ranking has zero spread — probe blind"
    );
}

/// One T1 failure-mode classification: `KSideMode` from the ρ(damage, range)
/// sign, plus the model-level dilution verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KSideMode {
    /// ρ ≥ +τ: damage co-located with wide-range channels → finer granularity.
    BehaviorInOutliers,
    /// ρ ≤ −τ: damage in narrow-range channels → range-blind weighting (T2).
    BehaviorInQuietChannels,
    /// |ρ| < τ: no channel-range mode → cross-layer / allocation-spread axes.
    Diffuse,
}

#[test]
fn kv960_t1_failure_mode_classification() {
    let config = probe_config();
    let n_layer = config.n_layer;
    let n_kv = config.n_kv_head;
    let hd = config.head_dim;
    let run = capture_and_probe(&config);

    let mode_name = |m: KSideMode| match m {
        KSideMode::BehaviorInOutliers => "behavior-in-outliers",
        KSideMode::BehaviorInQuietChannels => "behavior-in-quiet-channels",
        KSideMode::Diffuse => "diffuse",
    };
    let mitigation = |m: KSideMode| match m {
        KSideMode::BehaviorInOutliers => "finer granularity on the damaged channels",
        KSideMode::BehaviorInQuietChannels => {
            "range-blind behavioral-damage weighting (T2's lever)"
        }
        KSideMode::Diffuse => "allocation-spread axes, not channel selection",
    };

    // Half-split classification agreement (the G1 analog for the classifier):
    // classify on each half independently, count sign agreements.
    let second_of = |first: &Vec<Vec<Vec<f32>>>| -> Vec<Vec<Vec<f32>>> {
        run.k_full
            .iter()
            .zip(first)
            .map(|(f, a)| {
                f.iter()
                    .zip(a)
                    .map(|(fv, av)| fv.iter().zip(av).map(|(&x, &y)| x - y).collect())
                    .collect()
            })
            .collect()
    };
    let k_second = second_of(&run.k_first);

    let classify = |damage: &[f32], samples: &[Vec<f32>]| -> KSideMode {
        let ranges: Vec<f32> = (0..hd).map(|c| channel_range(samples, c)).collect();
        let rho = spearman(damage, &ranges);
        if rho >= MODE_TAU {
            KSideMode::BehaviorInOutliers
        } else if rho <= -MODE_TAU {
            KSideMode::BehaviorInQuietChannels
        } else {
            KSideMode::Diffuse
        }
    };

    let mut printed = String::from(
        "| layer | kvh | ρ(damage,range) | mode (full) | mode (1st half) | mode (2nd half) |\n|---|---|---|---|---|---|\n",
    );
    let mut agree = 0usize;
    let mut mode_counts = [0usize; 3];
    for (layer, (((kf_l, kfirst_l), ksecond_l), samples_l)) in run
        .k_full
        .iter()
        .zip(&run.k_first)
        .zip(&k_second)
        .zip(&run.k_samples)
        .enumerate()
    {
        for (kvh, (((kf, kfl), ks), samples)) in kf_l
            .iter()
            .zip(kfirst_l)
            .zip(ksecond_l)
            .zip(samples_l)
            .enumerate()
        {
            let rho_full = spearman(
                kf,
                &(0..hd)
                    .map(|c| channel_range(samples, c))
                    .collect::<Vec<f32>>(),
            );
            let m_full = classify(kf, samples);
            let m_a = classify(kfl, samples);
            let m_b = classify(ks, samples);
            if m_a == m_b {
                agree += 1;
            }
            mode_counts[match m_full {
                KSideMode::BehaviorInOutliers => 0,
                KSideMode::BehaviorInQuietChannels => 1,
                KSideMode::Diffuse => 2,
            }] += 1;
            printed.push_str(&format!(
                "| {layer} | {kvh} | {rho_full:.3} | {} | {} | {} |\n",
                mode_name(m_full),
                mode_name(m_a),
                mode_name(m_b)
            ));
        }
    }

    // Model-level layer spread: per-layer damage mass m_l = Σ_{kvh,c} damage;
    // participation ratio PR = (Σm)²/Σm² ∈ [1, n_layer] (Cauchy-Schwarz),
    // normalized to the delocalization index (PR − 1)/(n − 1) ∈ [0, 1]:
    // 1 = mass spread flat across layers (DILUTED — mode 3), 0 = one layer
    // dominates (per-layer fix viable).
    let layer_mass: Vec<f64> = (0..n_layer)
        .map(|l| {
            run.k_full[l]
                .iter()
                .flatten()
                .map(|&x| x as f64)
                .sum::<f64>()
        })
        .collect();
    let m_sum: f64 = layer_mass.iter().sum();
    let m_sq: f64 = layer_mass.iter().map(|m| m * m).sum();
    let (delocalization, spread_word) = if m_sq > 0.0 {
        let idx = ((m_sum * m_sum / m_sq) - 1.0) / (n_layer as f64 - 1.0);
        (
            idx,
            if idx >= DILUTION_BAR {
                "DILUTED — multi-layer dilution (mode 3): cross-layer allocation, per-layer fixes fail"
            } else {
                "CONCENTRATED — a per-layer fix on the hot layer captures the mass"
            },
        )
    } else {
        (0.0, "no mass")
    };

    // Model-level range alignment: mean ρ(damage, range) over heads. On
    // random-weight transformers this is EXPECTED positive (the per-channel
    // quant step is range/(2^bits − 1), so damage tracks range mechanically)
    // — the negative control; discriminating power shows on trained models
    // where damage and range dissociate.
    let head_rho = |damage: &[f32], l: usize, kvh: usize| -> f64 {
        let ranges: Vec<f32> = (0..hd)
            .map(|c| channel_range(&run.k_samples[l][kvh], c))
            .collect();
        spearman(damage, &ranges)
    };
    let mut rhos = Vec::new();
    let mut rho_first_v = Vec::new();
    for l in 0..n_layer {
        for kvh in 0..n_kv {
            rhos.push(head_rho(&run.k_full[l][kvh], l, kvh));
            rho_first_v.push(head_rho(&run.k_first[l][kvh], l, kvh));
        }
    }
    let rho_mean = rhos.iter().sum::<f64>() / rhos.len() as f64;
    let rho_stab = spearman(
        &rhos.iter().map(|&x| x as f32).collect::<Vec<_>>(),
        &rho_first_v.iter().map(|&x| x as f32).collect::<Vec<_>>(),
    );

    let masses = layer_mass
        .iter()
        .enumerate()
        .map(|(l, m)| format!("L{l}={m:.1}"))
        .collect::<Vec<_>>()
        .join(" ");
    println!(
        "kv960 T1 verdicts: modes full-corpus behavior-in-outliers {} / quiet-channels {} / diffuse {} of {} heads; layer masses [{masses}]; delocalization index {:.3} → {}; mean ρ(damage,range) {rho_mean:.3}; ρ half-split rank stability {rho_stab:.3}; sub-budget (10-prompt) mode agreement {}/{} — borderline heads at τ flip by design, the budget classifies ONCE at full corpus",
        mode_counts[0],
        mode_counts[1],
        mode_counts[2],
        n_layer * n_kv,
        delocalization,
        spread_word,
        agree,
        n_layer * n_kv,
    );
    println!(
        "kv960 T1 mitigation map: behavior-in-outliers → {}; behavior-in-quiet-channels → {}; diffuse → {}; outlier-crushes-behavior → excluded by construction on the per-channel reduction instrument (attribution-mode lane).",
        mitigation(KSideMode::BehaviorInOutliers),
        mitigation(KSideMode::BehaviorInQuietChannels),
        mitigation(KSideMode::Diffuse)
    );

    // Pins (deterministic seeds; margins in the bench record). The model-level
    // range alignment is the gated quantity (the paper's budget classifies at
    // FULL corpus); sub-budget mode flips are reported, never gated. The
    // concentration index and the per-head modes are findings, reported.
    assert!(
        rho_mean > 0.3,
        "model-level damage-range alignment collapsed: {rho_mean:.3}"
    );
    assert!(
        rho_stab > 0.5,
        "ρ half-split rank stability collapsed: {rho_stab:.3}"
    );
    assert!(
        m_sum > 0.0,
        "zero total damage mass — probe produced nothing"
    );
}
