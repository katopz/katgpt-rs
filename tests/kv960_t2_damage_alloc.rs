//! Issue 960 T2 — behavioral-damage-weighted bit allocation GOAT (Research
//! 633 / arXiv:2606.09864). The T0a probe's damage ranking (calibrated on
//! the 20-prompt corpus — the paper's calibration budget) drives the SHIPPED
//! water-fill allocator
//! (`katgpt-spectral::kv_sensitivity_alloc::allocate_bits`) as its weight
//! input; retention is the JOINT key read
//! (`katgpt-spectral::kv_sensitivity::joint_key_read` — all channels
//! degraded at once at the allocated widths) over 10 HELD-OUT prompts
//! (disjoint seeds — calibration never doubles as the eval population).
//!
//! Three arms at EQUAL average bits (the GOAT premise — `allocate_bits`
//! asserts the budget total):
//!
//! 1. **behavioral** — `D_c = damage_c(b₀) · 4^{b₀}` (the probe damage
//!    extrapolated to the bit-0 scale; the water-fill equalizes predicted
//!    damage `Σ D_c/4^{b_c}`).
//! 2. **structural** — the SHIPPED weighting projected per raw channel
//!    (`Σ_i bits_i · R[i][c]²`, the T0b projection over the head's own
//!    first-half calibration population).
//! 3. **uniform** — flat weights (the water-fill round-robins
//!    channel-ascending).
//!
//! Gates (issue 960 acceptance, the T2 rows):
//!
//! - **G1** — joint read deterministic (repeat-identical); the behavioral
//!   weight ranking stable across prompt halves (Spearman, the T0b 0.828
//!   precedent); equal-bits across arms at every rung.
//! - **G2** — behavioral recovers retention at lower-or-equal average bits
//!   than BOTH controls (bits-equivalence against behavioral's baseline
//!   rung), with a strictly better profile somewhere (a per-rung win
//!   clearing the pinned margin). A NEGATIVE here is a finding about the
//!   model, never about the instrument — the curves print either way.
//! - **G3** — lossless path untouched / feature-off byte-identical: the
//!   feature gates NEW code only (no existing path changes); asserted by
//!   the compile postures in CI, not in this binary.
//! - **G4** — the decode-path surface (`BitTable::bits` lookup) is 0-alloc:
//!   pinned in its own single-test binary
//!   (`crates/katgpt-spectral/tests/kv960_t2_bit_table_alloc.rs`) so the
//!   per-process counting allocator is sound.
//! - **Report-the-floor** — every behavioral number prints beside BOTH
//!   floors (uniform-at-equal-bits and structural-at-equal-bits).
//!
//! Honest caveat (T1, Bench 954): on seeded random weights the behavioral
//! ranking co-locates with channel range (9/12 behavior-in-outliers — the
//! expected negative control), so a G2 win validates the PROCEDURE, not the
//! model-level law; the trained-model read is T3. The structural arm on
//! random weights is near-flat (isotropic-ish covariance) — the interesting
//! comparison is behavioral vs uniform; behavioral vs structural is
//! expected to win by the T0b dissociation (ρ = −0.038).
//!
//! Deterministic by construction: fixed weight/prompt seeds (the T0b
//! seeds — same world, so the T2 read is comparable to the T0b/T1 record);
//! all reductions are fixed-order f64 folds. Calibration diagnostic, NOT a
//! perf claim (the probe/allocation is one-shot, G4-exempt).

use katgpt_core::types::{Config, Rng};
use katgpt_forward::{ForwardContext, KvCaptureBuf, forward_base};
use katgpt_spectral::kv_sensitivity::{
    JointKeyRead, KeyTensors, ProbeConfig, ProbeScratch, joint_key_read, probe_key_channels,
};
use katgpt_spectral::kv_sensitivity_alloc::{
    BitTable, allocate_bits, behavioral_waterfill_weights,
};
use katgpt_spectral::spectral::{BitAllocator, calibrate_eigenbasis};
use katgpt_transformer::{MultiLayerKVCache, TransformerWeights};

/// Calibration prompt count — the paper's ~20-prompt budget (the T0b seed
/// world verbatim).
const N_CAL: usize = 20;
/// Held-out evaluation prompts (disjoint seeds — the paper's protocol:
/// calibration prompts never double as the retention population).
const N_EVAL: usize = 10;
const EVAL_SEED_BASE: u64 = 1500;
/// Positions per prompt (the causal prefix every query attends over).
const SEQ: usize = 40;
/// Probe bit width for the calibration damage (the paper's low-bit regime).
const PROBE_BITS: u32 = 2;
/// Average bits for the structural ladder projection (the weighting's SHAPE
/// is set by d_eff + the eigenbasis; this only scales the two regimes).
const STRUCTURAL_AVG_BITS: f32 = 4.0;
/// Weight-seed / prompt-seed base — the T0b/T1 world verbatim.
const WEIGHT_SEED: u64 = 960;
const PROMPT_SEED_BASE: u64 = 1000;
/// The low-bit ladder (every rung × head_dim 16 is an exact integer budget).
const BIT_LADDER: [f32; 5] = [1.25, 1.5, 2.0, 2.5, 3.0];
/// Baseline rung for the bits-equivalence read (the paper's collapse
/// regime midpoint).
const BASELINE_RUNG: usize = 2;
/// Per-channel floor / cap for every arm.
const MIN_BITS: u8 = 1;
const MAX_BITS: u8 = 8;
/// G1 allocation-stability floor: Spearman between the calibration corpus's
/// FIRST-half accumulated damage ranks and the FULL-corpus ranks (the T0b
/// nested read — 0.828 measured there; disjoint halves would be noisier and
/// is not the claim). Floored well under the T0b record.
const ALLOC_STABILITY_FLOOR: f64 = 0.5;
/// G2 strict-win margin: the largest per-rung win over the better control
/// must clear this (a tie-only profile is not a win). PINNED from the
/// measured run — see `.benchmarks/955_kv960_t2_damage_alloc.md`.
const MIN_WIN_MARGIN: f32 = 0.02;

/// Small-but-real GQA config — the T0b world verbatim (4 layers, 6 query
/// heads over 3 kv heads, head_dim 16, seq 40).
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

/// Spearman rank correlation between two same-length slices (the T0b
/// helper verbatim). Ordinal ranks, descending by value, index tie-break;
/// the degenerate constant case returns 0.0, not NaN.
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

/// One (layer, kv head)'s captured tensors for ONE prompt.
struct HeadCapture {
    /// Query rows `[n_q × hd]` (every query head in the group, every position).
    q: Vec<f32>,
    /// Key rows `[SEQ × hd]`.
    k: Vec<f32>,
}

/// Everything the T2 verdict consumes, from ONE capture+probe pass.
struct CaptureRun {
    /// Per (prompt, layer, kv head) captured tensors — calibration prompts
    /// `[0..N_CAL]`, held-out eval prompts `[N_CAL..N_CAL + N_EVAL]`.
    heads: Vec<Vec<Vec<HeadCapture>>>,
    /// K-side accumulated probe damage per (layer, kv head, channel) over
    /// the calibration corpus: full, first half, second half (the halves
    /// feed the G1 nested resample-stability read).
    k_cal: Vec<Vec<Vec<f32>>>,
    k_cal_first: Vec<Vec<Vec<f32>>>,
    k_cal_second: Vec<Vec<Vec<f32>>>,
}

/// One forward pass over all prompts — 20 seeded calibration prompts (the
/// T0b world, damage accumulated full + per half) then 10 held-out eval
/// prompts (distinct seeds, tensors retained, never probed for weights).
/// Deterministic; ~3 s release.
fn capture_and_probe(config: &Config) -> CaptureRun {
    let hd = config.head_dim;
    let n_layer = config.n_layer;
    let n_kv = config.n_kv_head;
    assert!(
        config.n_embd.is_multiple_of(config.n_head),
        "n_embd must divide into heads"
    );
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

    let zeros = || vec![vec![vec![0.0f32; hd]; n_kv]; n_layer];
    let mut k_cal = zeros();
    let mut k_cal_first = zeros();
    let mut k_cal_second = zeros();
    let mut heads: Vec<Vec<Vec<HeadCapture>>> = Vec::with_capacity(N_CAL + N_EVAL);
    let cap = n_layer * SEQ * 8; // generous: never binds at this shape

    // One prompt's per-(layer, kv head) Q/K matrices from a capture buffer.
    fn extract_heads(
        buf: &katgpt_forward::KvCaptureBuf,
        n_layer: usize,
        n_kv: usize,
        group: usize,
        hd: usize,
        n_q: usize,
    ) -> Vec<Vec<HeadCapture>> {
        let mut prompt_heads: Vec<Vec<HeadCapture>> = Vec::with_capacity(n_layer);
        for layer in 0..n_layer {
            let mut layer_heads: Vec<HeadCapture> = Vec::with_capacity(n_kv);
            for kvh in 0..n_kv {
                // Per-kv-head key matrix [SEQ × hd] for this prompt.
                let mut k_mat = vec![0.0f32; SEQ * hd];
                for row in buf.rows.iter().filter(|r| r.layer == layer) {
                    let off = row.pos * hd;
                    k_mat[off..off + hd].copy_from_slice(&row.k[kvh * hd..(kvh + 1) * hd]);
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
                layer_heads.push(HeadCapture { q: q_mat, k: k_mat });
            }
            prompt_heads.push(layer_heads);
        }
        prompt_heads
    }

    // One seeded prompt through the forward, tensors retained.
    macro_rules! forward_prompt {
        ($seed:expr) => {{
            let mut rng = Rng::new($seed);
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
            buf
        }};
    }

    for prompt in 0..N_CAL {
        let buf = forward_prompt!(PROMPT_SEED_BASE + prompt as u64);
        let prompt_heads = extract_heads(&buf, n_layer, n_kv, group, hd, n_q);
        for layer in 0..n_layer {
            for kvh in 0..n_kv {
                let head = &prompt_heads[layer][kvh];
                let k_damage = probe_key_channels(
                    KeyTensors {
                        q: &head.q,
                        n_q,
                        k_ref: &head.k,
                        k_dequant: None,
                        seq: SEQ,
                        d: hd,
                    },
                    &probe_cfg,
                    &mut scratch,
                );
                let half = if prompt < N_CAL / 2 {
                    &mut k_cal_first
                } else {
                    &mut k_cal_second
                };
                for kd in &k_damage {
                    k_cal[layer][kvh][kd.channel] += kd.damage;
                    half[layer][kvh][kd.channel] += kd.damage;
                }
            }
        }
        heads.push(prompt_heads);
        cache.reset();
    }
    for prompt in 0..N_EVAL {
        let buf = forward_prompt!(EVAL_SEED_BASE + prompt as u64);
        heads.push(extract_heads(&buf, n_layer, n_kv, group, hd, n_q));
        cache.reset();
    }

    CaptureRun {
        heads,
        k_cal,
        k_cal_first,
        k_cal_second,
    }
}

/// The shipped structural weighting projected per raw channel
/// (`Σ_i bits_i · R[i][c]²`) over the calibration corpus — the T0b
/// projection verbatim, same population shape as behavioral's weights.
fn structural_weights(run: &CaptureRun, config: &Config) -> Vec<Vec<Vec<f32>>> {
    let hd = config.head_dim;
    let n_layer = config.n_layer;
    let n_kv = config.n_kv_head;
    let alloc = BitAllocator::new(1, 16);
    let mut structural: Vec<Vec<Vec<f32>>> = vec![vec![vec![0.0; hd]; n_kv]; n_layer];
    for layer in 0..n_layer {
        for kvh in 0..n_kv {
            let samples: Vec<Vec<f32>> = run.heads[..N_CAL]
                .iter()
                .flat_map(|p| p[layer][kvh].k.chunks(hd).map(<[f32]>::to_vec))
                .collect();
            assert_eq!(
                samples.len(),
                N_CAL * SEQ,
                "calibration sample count drifted"
            );
            let cal = calibrate_eigenbasis(&samples, hd);
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
    structural
}

/// Joint corpus read for one arm at one rung: allocate per (layer, kv head)
/// from `weights`, then read the JOINT degradation over every HELD-OUT eval
/// prompt, aggregated in fixed order (f64). Allocation is calibration-class
/// — one table per (rung, arm, head).
fn corpus_joint_read(
    run: &CaptureRun,
    weights: &[Vec<Vec<f64>>],
    avg_bits: f32,
    config: &Config,
    scratch: &mut ProbeScratch,
) -> JointKeyRead {
    let hd = config.head_dim;
    let n_kv = config.n_kv_head;
    let group = config.n_head / n_kv;
    let n_q = group * SEQ;
    let cfg = ProbeConfig {
        bits: PROBE_BITS, // ignored by joint_key_read (the table overrides)
        top_k: 1,
    };
    let tables: Vec<Vec<BitTable>> = weights
        .iter()
        .map(|w_layer| {
            w_layer
                .iter()
                .map(|w| allocate_bits(w, avg_bits, MIN_BITS, MAX_BITS))
                .collect()
        })
        .collect();
    let mut damage_sum = 0.0f64;
    let mut distort_sum = 0.0f64;
    let mut flip_sum = 0.0f64;
    let mut reads = 0.0f64;
    for prompt in N_CAL..(N_CAL + N_EVAL) {
        for (layer_heads, layer_tables) in run.heads[prompt].iter().zip(tables.iter()) {
            for (head, table) in layer_heads.iter().zip(layer_tables.iter()) {
                let read = joint_key_read(
                    KeyTensors {
                        q: &head.q,
                        n_q,
                        k_ref: &head.k,
                        k_dequant: None,
                        seq: SEQ,
                        d: hd,
                    },
                    table.as_slice(),
                    &cfg,
                    scratch,
                );
                damage_sum += read.damage as f64;
                distort_sum += read.mean_rel_distortion as f64;
                flip_sum += read.pick_flip_rate as f64;
                reads += 1.0;
            }
        }
    }
    JointKeyRead {
        damage: (damage_sum / reads) as f32,
        mean_rel_distortion: (distort_sum / reads) as f32,
        pick_flip_rate: (flip_sum / reads) as f32,
    }
}

#[test]
fn kv960_t2_behavioral_damage_bit_allocation_goat() {
    let config = probe_config();
    let hd = config.head_dim;
    let n_layer = config.n_layer;
    let n_kv = config.n_kv_head;

    let run = capture_and_probe(&config);

    // ── Arm weights (calibration corpus only) ───────────────────────────
    // behavioral: D_c = damage_c(b₀) · 4^{b₀} over the 20-prompt corpus;
    // the first-half accumulators feed the G1 nested stability read.
    let w_behavioral: Vec<Vec<Vec<f64>>> = run
        .k_cal
        .iter()
        .map(|layer| {
            layer
                .iter()
                .map(|dmg| behavioral_waterfill_weights(dmg, PROBE_BITS))
                .collect()
        })
        .collect();
    let w_uniform: Vec<Vec<Vec<f64>>> = vec![vec![vec![1.0; hd]; n_kv]; n_layer];
    let w_structural: Vec<Vec<Vec<f64>>> = structural_weights(&run, &config)
        .iter()
        .map(|l| {
            l.iter()
                .map(|h| h.iter().map(|&w| w as f64).collect())
                .collect()
        })
        .collect();

    // ── G1: allocation stability (the T0b nested read: first-half ranks
    // vs the full-corpus ranks the arms actually consume) — gated; the
    // DISJOINT half-vs-half Spearman prints beside it as a disclosure (the
    // noisier number a 10-prompt resample really carries). ─────────────
    let mut stab_sum = 0.0f64;
    let mut disjoint_sum = 0.0f64;
    for layer in 0..n_layer {
        for kvh in 0..n_kv {
            stab_sum += spearman(&run.k_cal_first[layer][kvh], &run.k_cal[layer][kvh]);
            disjoint_sum += spearman(&run.k_cal_first[layer][kvh], &run.k_cal_second[layer][kvh]);
        }
    }
    let heads_f = (n_layer * n_kv) as f64;
    let stab_mean = stab_sum / heads_f;
    println!(
        "G1 alloc-rank stability (Spearman): nested first-half vs full {stab_mean:.3} (floor {ALLOC_STABILITY_FLOOR}) · disjoint half-vs-half {:.3} (disclosed, not gated)",
        disjoint_sum / heads_f
    );
    assert!(
        stab_mean >= ALLOC_STABILITY_FLOOR,
        "G1: behavioral allocation ranking unstable across resamples ({stab_mean:.3} < {ALLOC_STABILITY_FLOOR})"
    );

    // ── G1: joint read determinism (repeat-identical) ───────────────────
    let mut scratch = ProbeScratch::default();
    scratch.ensure(2 * SEQ, SEQ, 1);
    let cfg = ProbeConfig {
        bits: PROBE_BITS,
        top_k: 1,
    };
    {
        let head = &run.heads[0][0][0];
        let table = allocate_bits(
            &w_behavioral[0][0],
            BIT_LADDER[BASELINE_RUNG],
            MIN_BITS,
            MAX_BITS,
        );
        let read = |scratch: &mut ProbeScratch| {
            joint_key_read(
                KeyTensors {
                    q: &head.q,
                    n_q: 2 * SEQ,
                    k_ref: &head.k,
                    k_dequant: None,
                    seq: SEQ,
                    d: hd,
                },
                table.as_slice(),
                &cfg,
                scratch,
            )
        };
        assert_eq!(
            read(&mut scratch),
            read(&mut scratch),
            "G1: joint read not repeat-identical"
        );
    }

    // ── The ladder: three arms × five rungs, equal average bits ─────────
    let arm_weights = [&w_behavioral, &w_structural, &w_uniform];
    let mut curves: [[f64; BIT_LADDER.len()]; 3] = [[0.0; BIT_LADDER.len()]; 3];
    let mut flip_curves: [[f64; BIT_LADDER.len()]; 3] = [[0.0; BIT_LADDER.len()]; 3];
    let mut total_bits_seen: [[usize; BIT_LADDER.len()]; 3] = [[0; BIT_LADDER.len()]; 3];
    for (arm, weights) in arm_weights.iter().enumerate() {
        for (r, &avg) in BIT_LADDER.iter().enumerate() {
            let read = corpus_joint_read(&run, weights, avg, &config, &mut scratch);
            curves[arm][r] = read.damage as f64;
            flip_curves[arm][r] = read.pick_flip_rate as f64;
            // allocate_bits asserts the total equals the budget; record the
            // first head's total for the cross-arm equality assert.
            let t = allocate_bits(&weights[0][0], avg, MIN_BITS, MAX_BITS);
            total_bits_seen[arm][r] = t.total_bits();
        }
    }
    for r in 0..BIT_LADDER.len() {
        assert_eq!(
            total_bits_seen[0][r], total_bits_seen[1][r],
            "equal-bits violated at rung {} (behavioral vs structural)",
            BIT_LADDER[r]
        );
        assert_eq!(
            total_bits_seen[1][r], total_bits_seen[2][r],
            "equal-bits violated at rung {} (structural vs uniform)",
            BIT_LADDER[r]
        );
    }

    // ── Report (the floors print beside every behavioral number) ────────
    println!(
        "\n== Issue 960 T2 GOAT: joint damage vs average bits (held-out eval prompts, K side) =="
    );
    println!(
        "{:<10} {:>6} {:>11} {:>11} {:>11} {:>12}",
        "avg_bits", "total", "behavioral", "structural", "uniform", "flip(behav)"
    );
    for (r, &avg) in BIT_LADDER.iter().enumerate() {
        println!(
            "{:<10} {:>6} {:>11.4} {:>11.4} {:>11.4} {:>12.4}",
            avg, total_bits_seen[0][r], curves[0][r], curves[1][r], curves[2][r], flip_curves[0][r]
        );
    }

    // ── G2: bits-equivalence against behavioral's baseline rung ─────────
    let baseline = curves[0][BASELINE_RUNG];
    let eq_bits = |curve: &[f64; BIT_LADDER.len()]| -> Option<usize> {
        curve.iter().position(|&d| d <= baseline)
    };
    let eq_b = eq_bits(&curves[0]);
    let eq_s = eq_bits(&curves[1]);
    let eq_u = eq_bits(&curves[2]);
    println!(
        "bits-equivalence (first rung reaching behavioral@{} = {baseline:.4}): behavioral {:?} · structural {:?} · uniform {:?} (ladder index; None = never)",
        BIT_LADDER[BASELINE_RUNG], eq_b, eq_s, eq_u
    );
    assert!(
        eq_b.unwrap() <= eq_s.unwrap_or(usize::MAX),
        "G2: behavioral does not recover at lower-or-equal bits than structural"
    );
    assert!(
        eq_b.unwrap() <= eq_u.unwrap_or(usize::MAX),
        "G2: behavioral does not recover at lower-or-equal bits than uniform"
    );

    // ── G2: a strictly better profile somewhere ─────────────────────────
    let max_win = curves[1]
        .iter()
        .zip(curves[2].iter())
        .zip(curves[0].iter())
        .map(|((&s, &u), &b)| s.min(u) - b)
        .fold(0.0f64, f64::max);
    println!(
        "max per-rung win over the better control: {max_win:.4} (margin floor {MIN_WIN_MARGIN})"
    );
    assert!(
        max_win >= MIN_WIN_MARGIN as f64,
        "G2: per-rung win margin {max_win:.4} below the pinned floor {MIN_WIN_MARGIN} — re-pin from the run if the profile moved"
    );

    println!(
        "\nG2 PASS — behavioral-damage weighting recovers retention at lower-or-equal average bits than BOTH floors, with a strictly better profile (see the table above)."
    );
}
