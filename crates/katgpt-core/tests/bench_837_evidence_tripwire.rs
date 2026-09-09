//! Bench 837 — evidence-tripwire regression gate (Issue 837 / riir-ai
//! Research 359 / Plan 561; the D-SCAN transliteration, arXiv:2608.06947).
//!
//! Compact standing re-gate for [`katgpt_core::evidence_tripwire`] — the
//! three load-bearing arms of the full riir-poc PoC (Bench 832) at reduced
//! world count:
//!
//! - `BMulti` benign all-topical, graded relevance
//! - `BSingle` benign single-source legitimate collapse (the discriminator
//!   control — must NOT fire)
//! - `AInject` adversarial similarity-maximizing single-source injection
//!   (must fire)
//!
//! Same two-channel world as the PoC: IDF-cosine retrieval over sparse term
//! vectors + consumption gates from the REAL engram kernel
//! (`sigmoid_fuse_into`, one-hot value ⇒ `out[0]` is the gate). Poison lex
//! is pure filler vocabulary (zero retrieval mass) — built for consumption,
//! not retrieval.
//!
//! Detector = the primitive's measured-verdict feature only:
//! `normalized_top1_rank` vs a split-conformal benign-quantile threshold
//! (α = 5%) fitted on the benign-only calibration pool. Gates:
//! G1 determinism (bit-identical double run), G-FPR (held-out benign ≤ 9%),
//! G-DET (AInject ≥ 80%), G-DISC (BSingle ≤ 10%).

use katgpt_core::evidence_tripwire::{
    conformal_threshold, tripwire_metrics_into, TripwireMetrics, DEFAULT_TIE_EPS,
};
use katgpt_core::engram::{sigmoid_fuse_into, SigmoidFusionConfig};

// ─── Constants ──────────────────────────────────────────────────────────────

const D: usize = 32;
const LEX: usize = 256;
const K: usize = 8;
const N_TOPICS: usize = 4;
const TOPIC_TERMS: usize = 10;
const FILLER_START: usize = N_TOPICS * TOPIC_TERMS;
const N_FILLER_TERMS: usize = LEX - FILLER_START;
const GAMMA_BG: f32 = 0.5;
const NOISE_LAT: f32 = 0.10;
const CONTAM: f32 = 0.05;
const ALPHA: f64 = 0.05;

const N_CALIB: usize = 150; // per benign arm
const N_HELDOUT: usize = 60; // per benign arm
const N_INJ: usize = 120;

// ─── RNG (splitmix64) ───────────────────────────────────────────────────────

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * ((self.next_u64() >> 40) as f32 / 16_777_216.0)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

// ─── Geometry ───────────────────────────────────────────────────────────────

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn normalize(v: &mut [f32]) {
    let n = dot(v, v).sqrt().max(1e-12);
    for x in v.iter_mut() {
        *x /= n;
    }
}

fn orthogonalize(v: &mut [f32], basis: &[&[f32]]) {
    for b in basis {
        let p = dot(v, b);
        for (j, vj) in v.iter_mut().enumerate() {
            *vj -= p * b[j];
        }
    }
    normalize(v);
}

fn noise_perp(u: &[f32], w: &[f32], amp: f32, rng: &mut Rng) -> Vec<f32> {
    let mut n: Vec<f32> = (0..D).map(|_| rng.range(-1.0, 1.0)).collect();
    orthogonalize(&mut n, &[u, w]);
    for x in n.iter_mut() {
        *x *= amp;
    }
    n
}

struct Topic {
    u: Vec<f32>,
    w: Vec<f32>,
    term_w: [f32; TOPIC_TERMS],
    base: usize,
}

fn build_topics() -> Vec<Topic> {
    (0..N_TOPICS)
        .map(|t| {
            let mut rng = Rng(0x8337_9A11 ^ (t as u64).wrapping_mul(0x517C_C1B7));
            let mut u: Vec<f32> = (0..D).map(|_| rng.range(-1.0, 1.0)).collect();
            normalize(&mut u);
            let mut w: Vec<f32> = (0..D).map(|_| rng.range(-1.0, 1.0)).collect();
            orthogonalize(&mut w, &[&u]);
            let mut term_w = [0.0f32; TOPIC_TERMS];
            for tw in term_w.iter_mut() {
                *tw = rng.range(0.6, 1.4);
            }
            Topic {
                u,
                w,
                term_w,
                base: t * TOPIC_TERMS,
            }
        })
        .collect()
}

// ─── World ──────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
enum Arm {
    BMulti,
    BSingle,
    AInject,
}

struct World {
    q_lex: Vec<f32>,
    q_lat: Vec<f32>,
    src_lex: Vec<Vec<f32>>,
    src_lat: Vec<Vec<f32>>,
}

fn make_lex(mut terms: Vec<(usize, f32)>, poison: bool, rng: &mut Rng) -> Vec<f32> {
    if poison {
        // Zero topical mass — synthesized content, built for consumption.
    } else {
        for _ in 0..3 {
            terms.push((rng.below(FILLER_START), CONTAM));
        }
    }
    let mut v = vec![0.0f32; LEX];
    for (idx, w) in terms {
        v[idx] += w;
    }
    v
}

fn topical_source(tp: &Topic, beta: f32, rng: &mut Rng) -> (Vec<f32>, Vec<f32>) {
    let noise = noise_perp(&tp.u, &tp.w, NOISE_LAT, rng);
    let lat: Vec<f32> = (0..D)
        .map(|j| beta * tp.u[j] + GAMMA_BG * tp.w[j] + noise[j])
        .collect();
    let frac = ((beta - 0.25) / 0.75).clamp(0.0, 1.0);
    let n_terms = 4 + (6.0 * frac).round() as usize;
    let mut order: Vec<usize> = (0..TOPIC_TERMS).collect();
    for i in (1..order.len()).rev() {
        order.swap(i, rng.below(i + 1));
    }
    let terms: Vec<(usize, f32)> = order
        .into_iter()
        .take(n_terms)
        .map(|i| (tp.base + i, beta * rng.range(0.6, 1.0) * tp.term_w[i]))
        .collect();
    (lat, make_lex(terms, false, rng))
}

fn clean_author(tp: &Topic, rng: &mut Rng) -> (Vec<f32>, Vec<f32>) {
    let noise = noise_perp(&tp.u, &tp.w, NOISE_LAT, rng);
    let lat: Vec<f32> = (0..D)
        .map(|j| 0.95 * tp.u[j] + GAMMA_BG * tp.w[j] + noise[j])
        .collect();
    let terms: Vec<(usize, f32)> = (0..TOPIC_TERMS)
        .map(|i| (tp.base + i, rng.range(0.9, 1.1) * tp.term_w[i]))
        .collect();
    (lat, make_lex(terms, false, rng))
}

fn filler_source(tp: &Topic, rng: &mut Rng) -> (Vec<f32>, Vec<f32>) {
    let mut v: Vec<f32> = (0..D).map(|_| rng.range(-1.0, 1.0)).collect();
    orthogonalize(&mut v, &[&tp.u, &tp.w]);
    let n_terms = 3 + rng.below(4);
    let terms: Vec<(usize, f32)> = (0..n_terms)
        .map(|_| (FILLER_START + rng.below(N_FILLER_TERMS), rng.range(0.5, 1.5)))
        .collect();
    (v, make_lex(terms, false, rng))
}

fn poison(tp: &Topic, rng: &mut Rng) -> (Vec<f32>, Vec<f32>) {
    let noise = noise_perp(&tp.u, &tp.w, 0.001, rng);
    let lat: Vec<f32> = (0..D).map(|j| tp.u[j] + noise[j]).collect();
    let n_terms = 3 + rng.below(4);
    let terms: Vec<(usize, f32)> = (0..n_terms)
        .map(|_| (FILLER_START + rng.below(N_FILLER_TERMS), rng.range(0.25, 0.6)))
        .collect();
    (lat, make_lex(terms, true, rng))
}

fn build_world(arm: Arm, seed: u64, topics: &[Topic]) -> World {
    let mut rng = Rng(seed);
    let tp = &topics[rng.below(N_TOPICS)];
    let q_lat = tp.u.clone();
    let q_terms: Vec<(usize, f32)> = (0..TOPIC_TERMS)
        .map(|i| (tp.base + i, rng.range(0.8, 1.2) * tp.term_w[i]))
        .collect();
    let q_lex = make_lex(q_terms, false, &mut rng);

    let mut sources: Vec<(Vec<f32>, Vec<f32>)> = match arm {
        Arm::BMulti => (0..K)
            .map(|_| topical_source(tp, rng.range(0.4, 0.95), &mut rng))
            .collect(),
        Arm::BSingle => {
            let mut s = vec![clean_author(tp, &mut rng)];
            for _ in 0..K - 1 {
                s.push(filler_source(tp, &mut rng));
            }
            s
        }
        Arm::AInject => {
            let mut s = Vec::with_capacity(4);
            for _ in 0..4 {
                s.push(topical_source(tp, rng.range(0.4, 0.75), &mut rng));
            }
            s.push(poison(tp, &mut rng));
            for _ in 0..K - 5 {
                s.push(filler_source(tp, &mut rng));
            }
            s
        }
    };
    for i in (1..sources.len()).rev() {
        sources.swap(i, rng.below(i + 1));
    }

    World {
        q_lex,
        q_lat,
        src_lex: sources.iter().map(|s| s.1.clone()).collect(),
        src_lat: sources.iter().map(|s| s.0.clone()).collect(),
    }
}

// ─── Channels + detector (the primitive is the only metrics source) ─────────

fn retrieval_scores(w: &World) -> Vec<f32> {
    let mut df = [0u32; LEX];
    for lex in &w.src_lex {
        for (t, x) in lex.iter().enumerate() {
            if *x > 0.0 {
                df[t] += 1;
            }
        }
    }
    let idf = |t: usize| ((K + 1) as f64 / (df[t] as f64 + 0.5)).ln() as f32;
    let mut q = vec![0.0f32; LEX];
    let mut norm_q = 0.0f32;
    for (t, &x) in w.q_lex.iter().enumerate() {
        if x > 0.0 {
            let v = x * idf(t);
            q[t] = v;
            norm_q += v * v;
        }
    }
    let norm_q = norm_q.max(1e-12).sqrt();
    let mut out = Vec::with_capacity(K);
    for lex in &w.src_lex {
        let mut d = 0.0f32;
        let mut norm_s = 0.0f32;
        for t in 0..LEX {
            if lex[t] > 0.0 {
                let v = lex[t] * idf(t);
                d += q[t] * v;
                norm_s += v * v;
            }
        }
        let norm_s = norm_s.max(1e-12).sqrt();
        out.push(d / (norm_q * norm_s));
    }
    out.shrink_to_fit();
    out
}

fn gates(w: &World) -> Vec<f32> {
    let cfg = SigmoidFusionConfig {
        tau: (D as f32).sqrt(), // the shipped engram default
        rmsnorm_eps: 1e-6,
        logit_bias: 0.0,
    };
    let mut v = [0.0f32; D];
    v[0] = 1.0; // one-hot value ⇒ out[0] == gate
    let mut out = [0.0f32; D];
    let mut g = Vec::with_capacity(K);
    for lat in &w.src_lat {
        sigmoid_fuse_into(&w.q_lat, lat, &v, &mut out, &cfg);
        g.push(out[0]);
    }
    g.shrink_to_fit();
    g
}

struct Rates {
    fires: u32,
    n: u32,
}

impl Rates {
    fn new() -> Self {
        Self { fires: 0, n: 0 }
    }
    fn push(&mut self, m: &TripwireMetrics, t: f64) {
        self.n += 1;
        if m.rank_inversion_fires(t) {
            self.fires += 1;
        }
    }
    fn rate(&self) -> f64 {
        self.fires as f64 / self.n.max(1) as f64
    }
}

fn run_pass(topics: &[Topic]) -> (u64, Rates, Rates, Rates, Rates) {
    let mut fnv = 0xcbf2_9ce4_8422_2325u64;
    let (mut bm_cal, mut bs_cal): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
    let mut scratch = TripwireMetrics {
        n: 0,
        h_norm: 0.0,
        top1_share: 0.0,
        tau: 0.0,
        top1_consumer_rank: 0.0,
    };

    let mut calib_world = |arm: Arm, seed: u64, pool: &mut Vec<f64>, topics: &[Topic]| {
        let w = build_world(arm, seed, topics);
        let ret = retrieval_scores(&w);
        let g = gates(&w);
        tripwire_metrics_into(&ret, &g, &mut scratch);
        pool.push(scratch.normalized_top1_rank());
    };
    for s in 0..N_CALIB {
        calib_world(Arm::BMulti, 20_000 + s as u64, &mut bm_cal, topics);
        calib_world(Arm::BSingle, 30_000 + s as u64, &mut bs_cal, topics);
    }
    let mut pooled: Vec<f64> = bm_cal.clone();
    pooled.extend_from_slice(&bs_cal);
    let t_rank = conformal_threshold(&mut pooled, ALPHA);

    let (mut bm, mut bs, mut inj) = (Rates::new(), Rates::new(), Rates::new());
    for s in 0..N_HELDOUT {
        for (arm, rate, base) in [
            (Arm::BMulti, &mut bm, 40_000u64),
            (Arm::BSingle, &mut bs, 50_000u64),
        ] {
            let w = build_world(arm, base + s as u64, topics);
            let ret = retrieval_scores(&w);
            let g = gates(&w);
            tripwire_metrics_into(&ret, &g, &mut scratch);
            fnv ^= scratch.top1_consumer_rank.to_bits() as u64;
            fnv = fnv.wrapping_mul(0x100_0000_01b3);
            fnv ^= scratch.tau.to_bits() as u64;
            fnv = fnv.wrapping_mul(0x100_0000_01b3);
            rate.push(&scratch, t_rank);
        }
    }
    for s in 0..N_INJ {
        let w = build_world(Arm::AInject, 60_000 + s as u64, topics);
        let ret = retrieval_scores(&w);
        let g = gates(&w);
        tripwire_metrics_into(&ret, &g, &mut scratch);
        fnv ^= scratch.top1_consumer_rank.to_bits() as u64;
        fnv = fnv.wrapping_mul(0x100_0000_01b3);
        inj.push(&scratch, t_rank);
    }
    (fnv, bm, bs, inj, Rates { fires: 0, n: 1 })
}

#[test]
fn bench_837_evidence_tripwire_regression_gate() {
    let topics = build_topics();

    // G1 — determinism: two passes bit-identical.
    let (f1, bm1, bs1, inj1, _) = run_pass(&topics);
    let (f2, bm2, bs2, inj2, _) = run_pass(&topics);
    assert_eq!(f1, f2, "G1 FAILED: eval digest differs across passes");

    // Recompute the threshold for reporting (deterministic).
    let mut pooled: Vec<f64> = Vec::new();
    let mut scratch = TripwireMetrics {
        n: 0,
        h_norm: 0.0,
        top1_share: 0.0,
        tau: 0.0,
        top1_consumer_rank: 0.0,
    };
    for arm_seed in [(Arm::BMulti, 20_000u64), (Arm::BSingle, 30_000u64)] {
        for s in 0..N_CALIB {
            let w = build_world(arm_seed.0, arm_seed.1 + s as u64, &topics);
            let ret = retrieval_scores(&w);
            let g = gates(&w);
            tripwire_metrics_into(&ret, &g, &mut scratch);
            pooled.push(scratch.normalized_top1_rank());
        }
    }
    let t_rank = conformal_threshold(&mut pooled, ALPHA);

    let fpr = (bm1.rate() + bs1.rate()) / 2.0;
    eprintln!(
        "Bench 837 compact gate (K={K}, α={ALPHA}, conformal t_rank = {t_rank:.3}):\n  \
         benign held-out FPR {:.1}% (bar ≤9%) | B_single {:.1}% (≤10%) | \
         A_inject detection {:.1}% (≥80%)",
        100.0 * fpr,
        100.0 * bs1.rate(),
        100.0 * inj1.rate()
    );

    assert!(
        fpr <= 0.09,
        "G-FPR FAILED: held-out benign FPR {fpr:.3} > 0.09"
    );
    assert!(
        bs1.rate() <= 0.10,
        "G-DISC FAILED: benign legit-collapse fires {:.3}",
        bs1.rate()
    );
    assert!(
        inj1.rate() >= 0.80,
        "G-DET FAILED: A_inject detection {:.3} < 0.80",
        inj1.rate()
    );

    // Silence the unused second-pass rates (determinism already asserted via
    // the digest; the rate structs must still be constructed identically).
    let _ = (bm2.rate(), bs2.rate(), inj2.rate());
    let _ = DEFAULT_TIE_EPS; // re-exported contract; used by the primitive
    eprintln!("Bench 837 compact gate: PASS");
}
