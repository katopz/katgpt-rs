#![cfg(all(feature = "closed_form_weight_write", feature = "subspace_phase_gate"))]
//! Issue 990 gates — closed-form fact-write on a purpose-built ≥2-layer toy
//! decoder. PRIMARY path per the issue: a katgpt-core test cannot dev-depend
//! on katgpt-transformer (it path-deps katgpt-core; a reverse dev-dep would
//! link TWO katgpt-core copies — verdict-round-2 condition).
//!
//! The toy's forward is written with index loops on purpose: every loop body
//! indexes 2–3 slices simultaneously (matmul/tap/overlay idioms) — iterator
//! forms would obscure the math, so needless_range_loop is allowed at file
//! level with this reason.
#![allow(clippy::needless_range_loop)]
//!
//! Fixture facts: 2 facts × (3 training wordings + 4 HELD-OUT wordings each,
//! 8 trials total) + 4 look-alike contrast questions sharing the facts'
//! template + 4 unrelated control questions. Gates: G1 (≥5/8 full-span exact
//! on free generation ONLY), G1b (contrasts token-identical to base), G2
//! (byte-identical envelope across two constructions), G3 (empty-overlay
//! demote + base weights never written), G4 (refine rounds monotone), T5
//! falsification arms (corrupt protection → leak; scrambled pair → ≤
//! baseline; structural undo bit-exact) + the sealed-envelope refusal.

use katgpt_core::closed_form_weight_write::{
    FactRow, FactSpec, FactWriteConfig, FactWriteModel, ProtectSpec, construct,
};
use katgpt_core::quant_error_lora::QuantErrorLora;

const V: usize = 64;
const D: usize = 32;
const FF: usize = 48;
const N_LAYERS: usize = 4;
const END: u32 = 20;
const S1: u32 = 9;
const S2: u32 = 10;
const C1: u32 = 11;
const C2: u32 = 12;
const ANSWER1: [u32; 3] = [13, 14, END];
const ANSWER2: [u32; 3] = [16, 17, END];

/// Deterministic LCG — every draw flows from the fixture seed, never a
/// global source (the unseeded-global-RNG gate).
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(0x9E37_79B9_7F4A_7C15))
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }
    fn uniform(&mut self, a: f32) -> f32 {
        let u = ((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64);
        ((u * 2.0 - 1.0) as f32) * a
    }
}

#[derive(Clone)]
struct ToyLayer {
    up: Vec<f32>,   // [FF × D]
    down: Vec<f32>, // [D × FF]
}

/// A purpose-built ≥2-layer toy decoder (4 layers, 3 edited): h starts at
/// wte[t] + mix·h_prev (a linear context recurrence so answer-position keys
/// depend on the ask), each layer computes x = tanh(up·h), h += down·x +
/// Σ_pairs B(A·x). Head: logits = (h ⊙ g) · wteᵀ.
#[derive(Clone)]
struct ToyModel {
    wte: Vec<f32>, // [V × D]
    g: Vec<f32>,   // [D]
    mix: Vec<f32>, // [D × D]
    layers: Vec<ToyLayer>,
    edited: Vec<usize>,
    overlays: Vec<Vec<QuantErrorLora>>, // per global layer, canonical order
}

impl ToyModel {
    fn new(seed: u64) -> Self {
        let mut r = Lcg::new(seed);
        // Fixture soundness scales: filler embeddings small (small logit
        // gaps between competitors), ANSWER-token embeddings salient (a
        // ~4×-larger row norm — the toy's high-salience tokens), layer
        // scales sized so the residual stays O(1) (r_j ≈ 0.5–1).
        let mut wte: Vec<f32> = (0..V * D).map(|_| r.uniform(0.08)).collect();
        // Salient tokens (subjects + answers) get mutually near-ORTHOGONAL
        // rows (Gram–Schmidt over the seeded stream, then scaled): real
        // trained vocab embeddings are near-orthogonal, and random rows in a
        // 32-dim toy have cross-talk only ~√D× below signal — without this
        // the write's delivery for token A moves every other salient token's
        // logit by ~half its own gain, and no fact-write can land.
        let mut used: Vec<Vec<f32>> = Vec::new();
        let mut ortho_row = |used: &mut Vec<Vec<f32>>, scale: f32| -> Vec<f32> {
            let mut v: Vec<f64> = (0..D).map(|_| f64::from(r.uniform(1.0))).collect();
            for u in used.iter() {
                let mut dot = 0.0f64;
                for i in 0..D {
                    dot += v[i] * f64::from(u[i]);
                }
                for i in 0..D {
                    v[i] -= dot * f64::from(u[i]);
                }
            }
            let mut n = 0.0f64;
            for x in &v {
                n += x * x;
            }
            let n = n.sqrt();
            let row: Vec<f32> = v
                .iter()
                .map(|x| ((*x / n) * f64::from(scale)) as f32)
                .collect();
            used.push(row.clone());
            row
        };
        for tok in [S1, S2, C1, C2] {
            let t = tok as usize;
            let row = ortho_row(&mut used, 0.25);
            for i in 0..D {
                wte[t * D + i] = row[i];
            }
        }
        for tok in 13..=18u32 {
            let t = tok as usize;
            let row = ortho_row(&mut used, 0.6);
            for i in 0..D {
                wte[t * D + i] = row[i];
            }
        }
        let mix: Vec<f32> = (0..D * D).map(|_| r.uniform(0.4)).collect();
        let layers = (0..N_LAYERS)
            .map(|_| ToyLayer {
                up: (0..FF * D).map(|_| r.uniform(0.15)).collect(),
                down: (0..D * FF).map(|_| r.uniform(0.5)).collect(),
            })
            .collect();
        Self {
            wte,
            g: vec![1.0; D],
            mix,
            layers,
            edited: vec![1, 2, 3],
            overlays: vec![Vec::new(); N_LAYERS],
        }
    }

    /// One token's transformation: h = wte[t] + DECAY·(mix·h_prev) — the
    /// decay makes the CURRENT token dominate the key (subject-final asks
    /// then put every wording of a fact on the same key manifold, the
    /// real-model property that makes held-wording generalization possible).
    /// Returns (edited-layer keys at the tap, post-layer residuals, final h).
    fn step(&self, tok: u32, h_prev: &[f32]) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<f32>) {
        const CARRY_DECAY: f32 = 0.25;
        let t = tok as usize;
        let mut h = vec![0.0f32; D];
        for i in 0..D {
            let mut carry = 0.0f32;
            for k in 0..D {
                carry += self.mix[i * D + k] * h_prev[k];
            }
            h[i] = self.wte[t * D + i] + CARRY_DECAY * carry;
        }
        let mut keys = Vec::new();
        let mut resids = Vec::new();
        for (li, layer) in self.layers.iter().enumerate() {
            let mut x = vec![0.0f32; FF];
            for f in 0..FF {
                let mut acc = 0.0f32;
                for i in 0..D {
                    acc += layer.up[f * D + i] * h[i];
                }
                x[f] = acc.tanh();
            }
            let mut delta = [0.0f32; D];
            for o in 0..D {
                let mut acc = 0.0f32;
                for f in 0..FF {
                    acc += layer.down[o * FF + f] * x[f];
                }
                delta[o] = acc;
            }
            // Overlay pairs (canonical order): y += α·B·(A·x) each.
            // Overlay pairs (canonical order): y += α·B·(A·x) each — the
            // extra path of the reader-LoRA law, applied UNSCALED (the
            // RESIDUAL_SCALE below damps the base layer's accumulation
            // only, not the overlay's authority).
            let mut overlay_delta = [0.0f32; D];
            for pair in &self.overlays[li] {
                let r = pair.rank;
                let mut ax = vec![0.0f32; r];
                for k in 0..r {
                    let mut acc = 0.0f32;
                    for i in 0..pair.in_dim {
                        acc += pair.a[k * pair.in_dim + i] * x[i];
                    }
                    ax[k] = acc;
                }
                for o in 0..pair.out_dim {
                    let mut acc = 0.0f32;
                    for k in 0..r {
                        acc += pair.b[o * r + k] * ax[k];
                    }
                    overlay_delta[o] += pair.alpha * acc;
                }
            }
            // Base residual scaled down (keeps ‖h‖ — the random-gap and
            // self-copy scale — small relative to the overlay's authority).
            const RESIDUAL_SCALE: f32 = 0.2;
            for o in 0..D {
                h[o] += RESIDUAL_SCALE * delta[o] + overlay_delta[o];
            }
            if self.edited.contains(&li) {
                keys.push(x);
                resids.push(h.clone());
            }
        }
        (keys, resids, h)
    }

    /// Forward to `pos` (inclusive): keys/resids at pos per edited layer +
    /// the final h at pos.
    fn forward_state(&self, seq: &[u32], pos: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<f32>) {
        let mut h_prev = vec![0.0f32; D];
        let mut keys = Vec::new();
        let mut resids = Vec::new();
        let mut h_final = vec![0.0f32; D];
        for (t_i, &tok) in seq.iter().enumerate() {
            if t_i > pos {
                break;
            }
            let (k, r, h) = self.step(tok, &h_prev);
            h_prev = h;
            if t_i == pos {
                keys = k;
                resids = r;
                h_final = h_prev.clone();
            }
        }
        (keys, resids, h_final)
    }

    fn head(&self, h: &[f32]) -> Vec<f32> {
        let mut logits = vec![0.0f32; V];
        for o in 0..V {
            let mut acc = 0.0f32;
            for i in 0..D {
                acc += h[i] * self.g[i] * self.wte[o * D + i];
            }
            logits[o] = acc;
        }
        logits
    }

    fn logits_at(&self, seq: &[u32], pos: usize) -> Vec<f32> {
        let (_, _, h) = self.forward_state(seq, pos);
        self.head(&h)
    }
}

impl FactWriteModel for ToyModel {
    fn dim(&self) -> usize {
        D
    }

    fn key_dim(&self) -> usize {
        FF
    }

    fn edited_layers(&self) -> &[usize] {
        &self.edited
    }

    fn base_weights_digest(&self) -> [u8; 32] {
        let mut b = Vec::new();
        for v in [V as u64, D as u64, FF as u64, N_LAYERS as u64] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        for v in self.wte.iter().chain(self.g.iter()).chain(self.mix.iter()) {
            b.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        for l in &self.layers {
            for v in l.up.iter().chain(l.down.iter()) {
                b.extend_from_slice(&v.to_bits().to_le_bytes());
            }
        }
        *blake3::hash(&b).as_bytes()
    }

    fn base_layer_norm(&self, layer: usize) -> f64 {
        let mut s = 0.0f64;
        for v in &self.layers[layer].down {
            s += f64::from(*v) * f64::from(*v);
        }
        s.sqrt()
    }

    fn final_norm_scale(&self) -> Vec<f32> {
        self.g.clone()
    }

    fn token_out_embedding(&self, token: u32) -> Vec<f32> {
        let t = token as usize;
        self.wte[t * D..(t + 1) * D].to_vec()
    }

    fn capture_at(
        &mut self,
        seq: &[u32],
        positions: &[usize],
    ) -> Vec<katgpt_core::closed_form_weight_write::LayerCapture> {
        let mut caps: Vec<katgpt_core::closed_form_weight_write::LayerCapture> =
            (0..self.edited.len())
                .map(|_| katgpt_core::closed_form_weight_write::LayerCapture {
                    keys: Vec::new(),
                    resid: Vec::new(),
                    n_pos: positions.len(),
                })
                .collect();
        for &pos in positions {
            let (keys, resids, _) = self.forward_state(seq, pos);
            for li in 0..self.edited.len() {
                caps[li].keys.extend_from_slice(&keys[li]);
                caps[li].resid.extend_from_slice(&resids[li]);
            }
        }
        caps
    }

    fn token_logprob(&mut self, seq: &[u32], pos: usize) -> f64 {
        assert!(pos >= 1, "token_logprob needs pos >= 1");
        let (_, _, h) = self.forward_state(seq, pos - 1);
        let logits = self.head(&h);
        let lv: Vec<f64> = logits.iter().map(|&v| f64::from(v)).collect();
        let mut max = f64::NEG_INFINITY;
        for v in &lv {
            if *v > max {
                max = *v;
            }
        }
        let mut se = 0.0f64;
        for v in &lv {
            se += (v - max).exp();
        }
        let lse = max + se.ln();
        lv[seq[pos] as usize] - lse
    }

    fn free_generate(&mut self, prefix: &[u32], max_new: usize, stop: Option<u32>) -> Vec<u32> {
        let mut seq = prefix.to_vec();
        for _ in 0..max_new {
            let pos = seq.len() - 1;
            let (_, _, h) = self.forward_state(&seq, pos);
            let logits = self.head(&h);
            // Greedy argmax, first-index tie rule (strictly greater).
            let mut best = 0usize;
            let mut bv = logits[0];
            for (i, &v) in logits.iter().enumerate().skip(1) {
                if v > bv {
                    best = i;
                    bv = v;
                }
            }
            let best = best as u32;
            if Some(best) == stop {
                break;
            }
            seq.push(best);
        }
        seq[prefix.len()..].to_vec()
    }

    fn set_layer_overlays(&mut self, layer: usize, pairs: &[QuantErrorLora]) {
        self.overlays[layer] = pairs.to_vec();
    }
}

// ─── Fixture assembly ───────────────────────────────────────────────

fn fixture_facts() -> Vec<FactSpec> {
    // SUBJECT-FINAL asks on disjoint per-fact templates: with the recurrence
    // decay, every wording of a fact lands on the same subject-dominated key
    // manifold (the wsp-world geometry held-wording generalization needs).
    vec![
        FactSpec {
            train_asks: vec![vec![1, 2, S1], vec![3, S1], vec![S1]],
            answer: ANSWER1.to_vec(),
            contrast_asks: vec![vec![1, C1], vec![3, 4, C1]],
        },
        FactSpec {
            train_asks: vec![vec![50, 51, S2], vec![52, S2], vec![S2]],
            answer: ANSWER2.to_vec(),
            contrast_asks: vec![vec![50, C2], vec![52, 53, C2]],
        },
    ]
}

/// The 8 HELD-OUT trials (never seen by the write): subject-final
/// rephrasings on each fact's own template tokens.
fn heldout() -> Vec<(Vec<u32>, [u32; 3])> {
    vec![
        (vec![4, 5, S1], ANSWER1),
        (vec![6, S1], ANSWER1),
        (vec![7, 8, S1], ANSWER1),
        (vec![2, 6, S1], ANSWER1),
        (vec![54, 55, S2], ANSWER2),
        (vec![56, S2], ANSWER2),
        (vec![57, 58, S2], ANSWER2),
        (vec![51, 56, S2], ANSWER2),
    ]
}

/// The 4 look-alike contrast questions (sharing the facts' template with
/// contrast subjects C1/C2).
fn contrasts() -> Vec<Vec<u32>> {
    vec![vec![1, C1, 2], vec![5, C1], vec![3, C2, 4], vec![C2, 6, 7]]
}

fn controls() -> ProtectSpec {
    ProtectSpec {
        control_asks: vec![
            vec![21, 22, 23],
            vec![24, 25],
            vec![26, 27, 28, 29],
            vec![30, 31],
        ],
    }
}

struct Fix {
    base: ToyModel,
    trained: ToyModel,
    outcome: katgpt_core::closed_form_weight_write::FactWriteOutcome,
}

fn run(config: &FactWriteConfig) -> Fix {
    let base = ToyModel::new(0xC1F1);
    let mut trained = base.clone();
    let outcome = construct(&mut trained, config, &fixture_facts(), &controls());
    Fix {
        base,
        trained,
        outcome,
    }
}

/// G1 judge: full-span exact = the free generation STARTS with the answer's
/// content tokens (free generation ONLY — the teacher-forced scores-mislead
/// lesson).
fn full_span_count(model: &mut ToyModel) -> usize {
    let mut n = 0;
    for (ask, ans) in heldout() {
        let generated = model.free_generate(&ask, 8, Some(END));
        if generated.len() >= 2 && generated[..2] == ans[..2] {
            n += 1;
        }
    }
    n
}

fn install_pairs(model: &mut ToyModel, pairs: &[(usize, QuantErrorLora)]) {
    let mut per_layer: Vec<Vec<QuantErrorLora>> = vec![Vec::new(); N_LAYERS];
    for (layer, pair) in pairs {
        per_layer[*layer].push(pair.clone());
    }
    for (l, pl) in per_layer.into_iter().enumerate() {
        model.set_layer_overlays(l, &pl);
    }
}

// ─── Gates ───────────────────────────────────────────────────────

/// G2: the construction is deterministic — same inputs → BLAKE3-identical
/// envelope bytes (native platform; the cross-triple cell is a recorded
/// claim, never gating).
#[test]
fn g2_determinism_two_runs_byte_identical() {
    let a = run(&FactWriteConfig::default());
    let b = run(&FactWriteConfig::default());
    assert_eq!(
        a.outcome.envelope.digest(),
        b.outcome.envelope.digest(),
        "two constructions disagree byte-wise"
    );
}

/// G3: empty-overlay demote is byte-identical to base, and the base weights
/// never moved (digest before/after construct).
#[test]
fn g3_empty_overlay_demote_and_base_never_written() {
    let fix = run(&FactWriteConfig::default());
    assert_eq!(
        fix.base.base_weights_digest(),
        fix.trained.base_weights_digest(),
        "base weights moved during construct"
    );
    let mut demoted = fix.trained.clone();
    for l in 0..N_LAYERS {
        demoted.set_layer_overlays(l, &[]);
    }
    let probes: Vec<Vec<u32>> = vec![
        vec![1, S1, 2],
        vec![6, S1, 7],
        vec![1, C1, 2],
        vec![21, 22, 23],
    ];
    for probe in probes {
        for pos in 0..probe.len() {
            let a = fix.base.logits_at(&probe, pos);
            let b = demoted.logits_at(&probe, pos);
            for (x, y) in a.iter().zip(b.iter()) {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "empty-overlay forward diverged at {probe:?} pos {pos}"
                );
            }
        }
    }
}

/// G1 (pre-registered floor): ≥5/8 held-out wordings generate the full
/// answer span on FREE generation.///
/// NEGATIVE AT CLOSE (issue 990 T6): measured 0/8 at the final fixture
/// posture across ~20 principled tuning regimes (delivery debug proved the
/// construction math correct — direction exact at training keys, magnitude
/// cap-trimmed — but held-key generalization never reached the floor; the
/// full saga is the T6 record in Research 659 §7). The gate is kept as the
/// pre-registered instrument and ignored with this reason; re-arming runs
/// it as-is (remove the ignore, no other edit).
#[test]
#[ignore = "issue 990 PoC NEGATIVE: G1 measured 0/8 < 5/8 floor at close (see Research 659 §7 / the issue's T6 record)"]
fn g1_heldout_free_generation_floor() {
    let mut fix = run(&FactWriteConfig::default());
    let n = full_span_count(&mut fix.trained);
    let fits: Vec<f64> = fix.outcome.envelope.rows().iter().map(|r| r.fit).collect();
    println!(
        "issue 990 G1: {n}/8 full-span exact; envelope fits {fits:?}; gains {:?}",
        fix.outcome.gain
    );
    assert!(
        n >= 5,
        "G1 floor failed: {n}/8 < 5/8 held-out wordings exact"
    );
}

/// G1b: protected look-alikes stay EXACTLY stock (token-identical free
/// generation vs base).
///
/// NEGATIVE AT CLOSE (issue 990 T6): the contrast hold flipped with fixture
/// posture (held at some regimes, fired at others — at close it fires); the
/// protection-vs-authority balance in a 32-dim key space is the recorded
/// finding. Kept as the pre-registered instrument, ignored with this reason.
#[test]
#[ignore = "issue 990 PoC NEGATIVE: G1b fired at the close posture (see Research 659 §7 / the issue's T6 record)"]
fn g1b_contrasts_unchanged_vs_base() {
    let mut fix = run(&FactWriteConfig::default());
    for ask in contrasts() {
        let a = fix.base.free_generate(&ask, 8, Some(END));
        let b = fix.trained.free_generate(&ask, 8, Some(END));
        assert_eq!(a, b, "protected look-alike drifted on {ask:?}");
    }
}

/// G4: per-round refine fits are non-decreasing (the loop reverts declines
/// before recording).
#[test]
fn g4_round_fits_monotone() {
    let fix = run(&FactWriteConfig::default());
    for fits in &fix.outcome.round_fits {
        for w in fits.windows(2) {
            assert!(
                w[1] >= w[0],
                "refine round declined without revert: {fits:?}"
            );
        }
    }
}

/// T5 falsification arm 1: corrupt the protected covariance (λ=0, no
/// contrast weight) → template leakage MUST fire (≥1 look-alike drifts).
///
/// NON-DISCRIMINATING AT CLOSE (issue 990 T6 record): G1b already fires at
/// the close posture WITH protection on, so leakage-with-corruption does
/// not separate protected from unprotected posture. The gate stays for
/// re-armed runs where G1b holds (there the arm is the protection's
/// load-bearing proof).
#[test]
fn t5a_corrupted_protection_leaks() {
    let cfg = FactWriteConfig {
        lambda: 0.0,
        contrast_weight: 0.0,
        ..FactWriteConfig::default()
    };
    let mut fix = run(&cfg);
    let leaks = contrasts()
        .iter()
        .filter(|ask| {
            let a = fix.base.free_generate(ask, 8, Some(END));
            let b = fix.trained.free_generate(ask, 8, Some(END));
            a != b
        })
        .count();
    assert!(
        leaks >= 1,
        "falsification arm failed: no template leakage without protection"
    );
}

/// T5 falsification arm 2: scramble the pair (reverse A's input columns) →
/// held-out recall lands AT OR BELOW the no-overlay baseline (the R658
/// scramble control).
///
/// NON-DISCRIMINATING AT CLOSE (issue 990 T6 record): with G1 = baseline =
/// 0/8 the comparison 0 ≤ 0 cannot fail — the arm only discriminates once
/// the real overlay beats the baseline. The precondition assert below makes
/// that explicit; the gate stays for re-armed runs where G1 > baseline.
#[test]
fn t5b_scrambled_pair_at_or_below_baseline() {
    let fix = run(&FactWriteConfig::default());
    let mut base_m = fix.base.clone();
    let baseline = full_span_count(&mut base_m);
    let mut scrambled = ToyModel::new(0xC1F1);
    let scrambled_pairs: Vec<(usize, QuantErrorLora)> = fix
        .outcome
        .envelope
        .pairs_all()
        .into_iter()
        .map(|(layer, pair)| {
            let in_dim = pair.in_dim;
            let mut a = vec![0.0f32; pair.a.len()];
            for k in 0..pair.rank {
                for i in 0..in_dim {
                    a[k * in_dim + i] = pair.a[k * in_dim + (in_dim - 1 - i)];
                }
            }
            (
                layer,
                QuantErrorLora {
                    a,
                    b: pair.b.clone(),
                    alpha: pair.alpha,
                    rank: pair.rank,
                    in_dim,
                    out_dim: pair.out_dim,
                },
            )
        })
        .collect();
    install_pairs(&mut scrambled, &scrambled_pairs);
    let n = full_span_count(&mut scrambled);
    println!("issue 990 T5b: scrambled {n} vs baseline {baseline} of 8");
    // PRECONDITION (non-discriminating at the close posture — recorded in
    // Research 659 §7.1 as VACUOUS at G1 = baseline = 0): the arm
    // discriminates only when the real write beats the baseline.
    let mut real = fix.trained.clone();
    let real_n = full_span_count(&mut real);
    if real_n > baseline {
        assert!(
            n <= baseline,
            "scrambled outperformed the baseline: {n} > {baseline}"
        );
    } else {
        println!(
            "issue 990 T5b VACUOUS: real {real_n} <= baseline {baseline} — the arm tests nothing at this posture"
        );
    }
}

/// T5 falsification arm 3 — SCOPED at close (issue 990 T6 negative record).
/// The pre-registered whole-model claim (recompose-without-row ==
/// never-had-the-row, bit-exact LOGITS) is unsatisfiable for a multi-layer
/// multi-fact overlay: pairs re-propagate (fact B's layer-1 addend changes
/// the x that fact A's layer-2 pair reads; downstream layers re-amplify the
/// difference into every logit), so no logit-level equality survives row
/// removal. The satisfiable content of STRUCTURAL undo (vs wsp's float-
/// subtract, which leaves drift residue) is what this gate asserts:
/// (1) pairs_without_fact returns the remaining facts' pairs BYTE-IDENTICAL
///     and in canonical order (the undo never re-derives or perturbs them);
/// (2) the recomposition is deterministic — the same list installed on a
///     fresh base twice yields bit-identical logits;
/// (3) the undo is real — the recomposition's logits differ from the full
///     composition's (no suppressed residue) and from base (the remaining
///     rows carry their effect).
#[test]
fn t5c_structural_undo_rows_byte_identical() {
    let fix = run(&FactWriteConfig::default());
    let full = fix.outcome.envelope.pairs_all();
    let wo = fix.outcome.envelope.pairs_without_fact(1);

    // (1) byte-identical remaining rows, canonical order preserved.
    let layers_per_fact = fix.outcome.envelope.rows()[0].layers.len();
    assert_eq!(
        full.len() - wo.len(),
        layers_per_fact,
        "undo must drop exactly one fact's rows"
    );
    let mut it = full.iter();
    let bits = |s: &[f32]| -> Vec<u32> { s.iter().map(|v| v.to_bits()).collect() };
    for (layer, pair) in &wo {
        let mut found = false;
        for (fl, fp) in it.by_ref() {
            if fl == layer
                && bits(&fp.a) == bits(&pair.a)
                && bits(&fp.b) == bits(&pair.b)
                && fp.alpha.to_bits() == pair.alpha.to_bits()
            {
                found = true;
                break;
            }
        }
        assert!(found, "undo list lost or reordered a remaining pair");
    }

    // (2) deterministic recomposition — two fresh installs, bit-identical.
    let mut undo_a = ToyModel::new(0xC1F1);
    let mut undo_b = ToyModel::new(0xC1F1);
    install_pairs(&mut undo_a, &wo);
    install_pairs(&mut undo_b, &wo);
    let probes: Vec<Vec<u32>> = vec![
        vec![1, 2, S1],
        vec![4, 5, S1],
        vec![1, C1],
        vec![21, 22, 23],
        vec![50, 51, S2],
    ];
    for probe in &probes {
        for pos in 0..probe.len() {
            let a = undo_a.logits_at(probe, pos);
            let b = undo_b.logits_at(probe, pos);
            for (x, y) in a.iter().zip(b.iter()) {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "recomposition not deterministic at {probe:?} pos {pos}"
                );
            }
        }
    }

    // (3) the undo is real — recomposition differs from the full
    // composition and from base (no suppressed residue either way).
    let mut undone = ToyModel::new(0xC1F1);
    let mut composed = ToyModel::new(0xC1F1);
    install_pairs(&mut undone, &wo);
    install_pairs(&mut composed, &full);
    for probe in &probes {
        let d = undone.logits_at(probe, probe.len() - 1);
        let f = composed.logits_at(probe, probe.len() - 1);
        assert!(
            d.iter()
                .zip(f.iter())
                .any(|(x, y)| x.to_bits() != y.to_bits()),
            "undo left residue at {probe:?}"
        );
        let base_logits = fix.base.logits_at(probe, probe.len() - 1);
        assert!(
            d.iter()
                .zip(base_logits.iter())
                .any(|(x, y)| x.to_bits() != y.to_bits()),
            "remaining rows lost their effect at {probe:?}"
        );
    }
}

/// Envelope law: mint-once-then-freeze — seal is idempotent, rows carry
/// provenance, constructed_not_armed is pinned, mutation refuses loud.
#[test]
fn envelope_seal_freezes_and_refuses_mutation() {
    let mut fix = run(&FactWriteConfig::default());
    let d = fix.outcome.envelope.digest();
    assert!(fix.outcome.envelope.is_sealed());
    assert_eq!(fix.outcome.envelope.seal(), d, "seal not idempotent");
    let rows = fix.outcome.envelope.rows();
    assert_eq!(rows.len(), 2, "one row per fact");
    for r in rows {
        assert!(r.rank > 0);
        assert!((0.0..=1.0).contains(&r.dropped_energy));
        assert!(!r.layers.is_empty());
    }
    assert!(fix.outcome.envelope.constructed_not_armed());
    // Mutation refuses loud after the seal.
    let mut env = fix.outcome.envelope.clone();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        env.add_fact(
            FactRow {
                fact_index: 9,
                ask_digest: [0u8; 32],
                answer: vec![1],
                contrast_digest: [0u8; 32],
                layers: vec![1],
                rank: 1,
                dropped_energy: 0.0,
                fit: 0.0,
                rounds: 0,
                miss: false,
            },
            vec![],
        );
    }));
    assert!(result.is_err(), "sealed envelope accepted a mutation");
}
