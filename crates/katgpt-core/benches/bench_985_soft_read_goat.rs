//! Bench 985 — soft-read GOAT cell (Issue 985 T4–T7; Research 656, the
//! Memorilla distillation's modelless floor).
//!
//! The preregistered bet: a sigmoid-gated query-conditioned fixed-K soft
//! read with NO learned weights must (a) match-or-beat hard cosine top-16 —
//! the adversarial DISCRETE baseline (the paper's nearest untrained ablation
//! captured +4.9 of +15.1, so the floor must beat discrete selection), and
//! (b) beat the empty-memory floor — the paper's random-init result (0.153
//! vs 0.186 no-memory) proves untrained latents can lose to NOTHING, and
//! that must fail loudly here too.
//!
//! # The composition-synthetic fixture (Fig 4(c)'s collapse shape)
//!
//! Multi-hop fact chains: `f_{j+1} = normalize(0.5·f_j + 0.85·u_j)` — hop-j
//! cosine to hop-1 decays 1, 0.5, 0.25, 0.125. The QUERY is a hop-1 mention
//! (`q = normalize(f_1 + 0.1·noise)`); gold for composition level c is
//! `f_c`. Same-topic DISTRACTORS sit at cosine ≈ 0.55 to the query — ABOVE
//! hop-2's 0.5 — so hard top-16 fills with hop-1 rows and distractors and
//! never reaches hop-2+: the discrete read's answer quality COLLAPSES as c
//! grows (the paper's 78.7 → 51.8 shape). The slot bank (farthest-point
//! anchors over the store) spreads along the chain; the conditioned heads
//! project the query toward every anchor, so some slot reads the hop-c
//! neighborhood the raw query cannot rank. If the fixture fails to
//! collapse the hard arm, the fixture is rigged — that is a printed
//! FAILURE (the W2 discipline).
//!
//! # Arms (amendment 3 — recorded in the issue BEFORE the rerun)
//!
//! - `soft16` — SoftRead, FPS slots, K=16, τ=0.1, **b=−2**; per-slot
//!   TOP-GATE row (argmax g_ik = argmax cosine — gate-scale-free), best
//!   over slots. Same 16-candidate budget as hard's top-16, so the only
//!   difference is the conditioning — the mechanism-isolating comparison.
//! - `soft-avg` — the gated weighted average's best-slot direction
//!   (PRINTED reference, never gated: an aggregate average is structurally
//!   capped below an exact-row answer even when the read is perfect).
//! - `soft-orth` — the seeded-orthonormal slot posture (comparison arm,
//!   printed not gated: the GOAT posture is the on-manifold FPS bank).
//! - `hard16` — top-16 by cosine(q, z); score = best of the 16 by cosine
//!   to gold. `hard16-mean` (printed): the aggregate mean of the top-16.
//! - `empty` — no primitive, the empty-memory floor: score 0 by
//!   definition (no read, no answer).
//!
//! # Gates (all preregistered; amendment 3 recorded in the issue)
//!
//! - **G1 composition bar:** soft16 ≥ hard16 at every level; soft16 at the
//!   deepest level > empty (0); the collapse exists: hard16(deepest) <
//!   hard16(1) − 0.1 (fixture-discipline).
//! - **G2 store-size arm:** the bar held at N ∈ {1, 16, 512}.
//! - **G3 Issue 984 arms (quality direction):** replicate means over the
//!   N=512 fixture — real ≥ noise + margin, noise ≤ empty + tol,
//!   shuffled ≤ real (the noise/shuffled stores re-derive their OWN banks;
//!   the whole primitive runs on the corrupted store).
//! - **G4 second-signal cell (T6):** the shared Bench 845 W1/W2 worlds
//!   with `DistanceSoftGate` — W1: AURC(soft) < AURC(score) (the signal
//!   carries OOD information); W2: AURC(soft) > AURC(score) (fixture
//!   discrimination). The existing max-sim arm re-runs as a regression
//!   witness (its G2 relationship fused > score must still hold).
//!
//! Posture: opt-in (`soft_read`). **MEASURED NEGATIVE (2026-11-06)** — the
//! amended GOAT cell failed the preregistered bar; the grade is struck
//! (this commit) and Research 656 records the negative. This bench is the
//! REPORT-ONLY replication instrument: gates print, one live assert
//! remains (fixture validity — if the hard16 collapse vanishes, the
//! recorded negative is void and the cell re-opens).
//!
//! # Run
//!
//! ```bash
//! CARGO_TARGET_DIR=/tmp/bench985 cargo bench -p katgpt-core \
//!   --features soft_read,distance_abstain --no-default-features \
//!   --bench bench_985_soft_read_goat -- --nocapture
//! ```

#![cfg(feature = "soft_read")]

#[path = "../tests/common/mod.rs"]
mod common;
#[path = "../tests/common/store_transforms.rs"]
mod store_transforms;
#[path = "../tests/common/bench845_worlds.rs"]
mod worlds;

use katgpt_core::soft_read::{
    SlotMode, SoftRead, SoftReadScratch, distance_soft::DistanceSoftGate,
};
use std::hint::black_box;
use worlds::{
    D, MID, SCALE, SEL30_RATE, aurc, build_corpus, build_world, selective_accuracy_at, w1_true_p,
    w2_true_p,
};

/// Deterministic `SplitMix64` (the workspace-bench idiom — no rand dep).
struct SplitMix64(u64);
impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn next_unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn next_sym(&mut self) -> f32 {
        self.next_unit() * 2.0 - 1.0
    }
}

fn unit(v: [f32; D]) -> [f32; D] {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        let inv = 1.0 / n;
        v.map(|x| x * inv)
    } else {
        v
    }
}

const CHAIN_LEN: usize = 4; // composition levels 1..=4
const HOP_DECAY: f32 = 0.5; // ⟨f_c, f_1⟩ = HOP_DECAY^(c−1)
const HOP_FRESH: f32 = 0.85; // the fresh-direction weight (unit renorm)
const DISTRACTOR_W: f32 = 0.55; // distractor cosine to the query — above hop-2's 0.5
const DISTRACTORS_PER_CHAIN: usize = 18; // amendment 3: ≥19 same-topic rows above hop-2 saturate hard's top-16
const QUERY_NOISE: f32 = 0.1;
const K_SLOTS: usize = 16;
const N_REPS: usize = 4; // query-noise replicates
/// Amendment 3: the gate bias (the preregistered b_k dial, armed). At b=0
/// every zero-cosine row holds gate σ(0)=0.5 and the gated average
/// collapses toward the global store mean (measured: soft best-answer
/// cosine 0.54/0.27 flat — non-selective). −2 puts the 0.5-gate point at
/// cosine 0.2.
const GATE_BIAS: f32 = -2.0;

/// One chain: hop facts f_1..f_CHAIN_LEN + same-topic distractors.
struct Chain {
    hops: Vec<[f32; D]>,
    distractors: Vec<[f32; D]>,
}

fn build_chain(seed: u64, n_distractors: usize) -> Chain {
    let mut rng = SplitMix64(seed);
    let f1 = unit(std::array::from_fn(|_| rng.next_sym()));
    let mut hops = vec![f1];
    for _ in 1..CHAIN_LEN {
        let prev = *hops.last().expect("seeded hop");
        let u = unit(std::array::from_fn(|_| rng.next_sym()));
        hops.push(unit(std::array::from_fn(|i| {
            HOP_DECAY * prev[i] + HOP_FRESH * u[i]
        })));
    }
    let mut distractors = Vec::with_capacity(n_distractors);
    for _ in 0..n_distractors {
        let u = unit(std::array::from_fn(|_| rng.next_sym()));
        distractors.push(unit(std::array::from_fn(|i| {
            DISTRACTOR_W * f1[i] + (1.0 - DISTRACTOR_W) * 1.5 * u[i]
        })));
    }
    Chain { hops, distractors }
}

/// The store for one size arm: chain 0 (hops, then distractors — clamped
/// to n_target so N=1 is a one-row store), then more chains, then filler.
/// Returns the store and the number of composition LEVELS the store
/// actually supports (min(CHAIN_LEN, n_target)).
fn build_store(seed: u64, n_target: usize) -> (Vec<[f32; D]>, Vec<Chain>, usize) {
    let mut rng = SplitMix64(seed ^ 0xB00C);
    let mut rows: Vec<[f32; D]> = Vec::with_capacity(n_target);
    let mut chains = Vec::new();
    // Chain 0 first — the queried chain — hops before distractors so tiny
    // stores keep as many composition levels as the budget allows.
    let c0 = build_chain(seed ^ 0xC4A1, DISTRACTORS_PER_CHAIN);
    for h in &c0.hops {
        if rows.len() < n_target {
            rows.push(*h);
        }
    }
    for d in &c0.distractors {
        if rows.len() < n_target {
            rows.push(*d);
        }
    }
    chains.push(c0);
    let mut chain_i = 1u64;
    while rows.len() + CHAIN_LEN + DISTRACTORS_PER_CHAIN <= n_target {
        let c = build_chain(seed ^ 0xC4A1 ^ (chain_i << 8), DISTRACTORS_PER_CHAIN);
        rows.extend(&c.hops);
        rows.extend(&c.distractors);
        chains.push(c);
        chain_i += 1;
    }
    while rows.len() < n_target {
        rows.push(unit(std::array::from_fn(|_| rng.next_sym())));
    }
    let levels = CHAIN_LEN.min(n_target);
    (rows, chains, levels)
}

/// One query replicate: the hop-1 mention + noise, per level.
struct QueryRep {
    queries: Vec<[f32; D]>, // per level c−1: normalize(f1 + noise)
}

fn build_query_rep(seed: u64, chains: &[Chain]) -> QueryRep {
    let mut rng = SplitMix64(seed);
    let f1 = chains[0].hops[0];
    let q = unit(std::array::from_fn(|i| {
        f1[i] + QUERY_NOISE * rng.next_sym()
    }));
    QueryRep {
        queries: vec![q; CHAIN_LEN],
    }
}

/// The hard top-16 read: rows ranked by cosine(q, z); best-of-K by cosine
/// to gold (the discrete read's best extraction), plus the aggregate mean
/// cosine (printed reference).
fn hard16_scores(q: &[f32; D], rows: &[[f32; D]], gold: &[f32; D], k: usize) -> (f32, f32) {
    let gold = unit(*gold);
    let mut idx: Vec<usize> = (0..rows.len()).collect();
    let dots: Vec<f32> = rows
        .iter()
        .map(|r| r.iter().zip(q).map(|(a, b)| a * b).sum())
        .collect();
    idx.sort_by(|a, b| dots[*b].total_cmp(&dots[*a]));
    let top = &idx[..k.min(idx.len())];
    let mut best = f32::NEG_INFINITY;
    let mut mean = [0.0f32; D];
    for &i in top {
        let r = unit(rows[i]);
        let c: f32 = r.iter().zip(&gold).map(|(a, b)| a * b).sum();
        if c > best {
            best = c;
        }
        for (m, x) in mean.iter_mut().zip(&r) {
            *m += x;
        }
    }
    let mean = unit(mean);
    let mean_c: f32 = mean.iter().zip(&gold).map(|(a, b)| a * b).sum();
    (best, mean_c)
}

/// Evaluate one size arm: replicate-mean best-answer cosine per level for
/// every arm. The GATED arm is `soft16` (per-slot top-gate row, best over
/// slots — amendment 3's mechanism-isolating, budget-fair semantic);
/// `soft-avg` (the aggregate read's direction) is a printed reference.
struct ArmMeans {
    soft: [f64; CHAIN_LEN],
    soft_avg: [f64; CHAIN_LEN],
    soft_orth: [f64; CHAIN_LEN],
    hard: [f64; CHAIN_LEN],
    hard_mean: [f64; CHAIN_LEN],
}

fn evaluate_size_arm(n_target: usize) -> (ArmMeans, f32, usize) {
    let mut soft = [0.0f64; CHAIN_LEN];
    let mut soft_avg = [0.0f64; CHAIN_LEN];
    let mut soft_orth = [0.0f64; CHAIN_LEN];
    let mut hard = [0.0f64; CHAIN_LEN];
    let mut hard_mean = [0.0f64; CHAIN_LEN];
    let (rows, chains, levels) = build_store(0x985_FEED, n_target);
    let mut scratch = SoftReadScratch::<D, K_SLOTS>::new();
    let soft_fps = SoftRead::<D, K_SLOTS>::new(&rows).with_biases([GATE_BIAS; K_SLOTS]);
    let soft_orthogonal = SoftRead::<D, K_SLOTS>::with_mode(&rows, SlotMode::Orthonormal)
        .with_biases([GATE_BIAS; K_SLOTS]);
    let entropy = soft_fps.bank().coverage_entropy(&rows);
    for rep in 0..N_REPS {
        let qr = build_query_rep(0x985_0100 + rep as u64, &chains);
        for c in 0..levels {
            let gold = chains[0].hops[c];
            let q = qr.queries[c];
            let r1 = soft_fps.read_into(&q, &rows, &mut scratch);
            soft[c] += soft_fps.best_gated_row_cosine(&rows, &scratch, &gold) as f64;
            soft_avg[c] += SoftRead::<D, K_SLOTS>::best_answer_cosine(&r1, &gold) as f64;
            let r2 = soft_orthogonal.read_into(&q, &rows, &mut scratch);
            let _ = r2; // scratch holds the heads the extraction reads
            soft_orth[c] += soft_orthogonal.best_gated_row_cosine(&rows, &scratch, &gold) as f64;
            let (hb, hm) = hard16_scores(&q, &rows, &gold, K_SLOTS);
            hard[c] += hb as f64;
            hard_mean[c] += hm as f64;
            black_box(&q);
        }
    }
    (
        ArmMeans {
            soft: soft.map(|s| s / N_REPS as f64),
            soft_avg: soft_avg.map(|s| s / N_REPS as f64),
            soft_orth: soft_orth.map(|s| s / N_REPS as f64),
            hard: hard.map(|s| s / N_REPS as f64),
            hard_mean: hard_mean.map(|s| s / N_REPS as f64),
        },
        entropy,
        levels,
    )
}

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Bench 985 — soft-read GOAT cell (Issue 985; Research 656)");
    println!(
        "  (K={K_SLOTS} slots, τ=0.1, b={GATE_BIAS}, chain decay {HOP_DECAY}, distractors {DISTRACTORS_PER_CHAIN}/chain, reps {N_REPS})"
    );
    println!("═══════════════════════════════════════════════════════════════");

    // ── G1 — the composition bar at N=512 ────────────────────────────────
    println!();
    println!(
        "G1 — composition bar (N=512; best-answer cosine per level c, higher is better; soft16 = per-slot top-gate row, same 16-candidate budget as hard16):"
    );
    println!("  c       soft16    soft-avg   soft-orth    hard16   hard16-mean   (empty=0)");
    let (means512, entropy, levels512) = evaluate_size_arm(512);
    let mut g1 = true;
    for c in 0..levels512 {
        println!(
            "  {}     {:-9.4} {:-9.4} {:-9.4} {:-9.4} {:-9.4}",
            c + 1,
            means512.soft[c],
            means512.soft_avg[c],
            means512.soft_orth[c],
            means512.hard[c],
            means512.hard_mean[c]
        );
        // The preregistered bar: soft ≥ hard16 AND > empty (0).
        if means512.soft[c] < means512.hard[c] {
            g1 = false;
        }
    }
    let last = levels512 - 1;
    if means512.soft[last] <= 0.0 {
        g1 = false;
    }
    // Fixture discipline: the hard arm must actually collapse.
    let collapse = means512.hard[last] < means512.hard[0] - 0.1;
    println!(
        "  soft16 ≥ hard16 at every level: {}   soft16(c={}) > 0: {}   hard collapse ({} → {}, needs ≥0.1 drop): {}",
        if g1 { "OK" } else { "FAIL" },
        last + 1,
        if means512.soft[last] > 0.0 {
            "OK"
        } else {
            "FAIL"
        },
        means512.hard[0],
        means512.hard[last],
        if collapse {
            "OK"
        } else {
            "FAIL — fixture rigged"
        }
    );
    println!(
        "  slot-coverage entropy (even tiling, floor 0.5): {entropy:.3} {}",
        if entropy >= 0.5 { "OK" } else { "FAIL" }
    );
    let g1 = g1 && collapse && entropy >= 0.5;

    // ── G2 — the store-size arm N ∈ {1, 16, 512} ─────────────────────────
    println!();
    println!("G2 — store-size arm (the bar must hold at every N; levels clamp to the store):\n");
    let mut g2 = true;
    for n in [1usize, 16] {
        let (m, _, levels) = evaluate_size_arm(n);
        for c in 0..levels {
            let ok = m.soft[c] >= m.hard[c];
            g2 &= ok;
            println!(
                "  N={n:<4} c={}  soft16 {:-7.4} vs hard {:-7.4}  {}",
                c + 1,
                m.soft[c],
                m.hard[c],
                if ok { "ok" } else { "FAIL" }
            );
        }
    }
    println!("  N=512 covered by G1: {}", if g1 { "ok" } else { "FAIL" });
    println!(
        "  note: N=16 makes hard16 trivially 1.0 (top-16 of a 16-row store is EVERY row) — the N=16 arm bounds soft16 from below and cannot distinguish the mechanisms"
    );
    let g2 = g2 && g1;

    // ── G3 — Issue 984 store arms on the composition metric ────────────
    // Verdict-narrowed instrument (a842c972): arms run at the DEEPEST
    // level (gated, as preregistered) AND the shallowest non-trivial level
    // (disclosed); hard16 runs UNDER the corrupted arms too (is the level
    // at chance for everyone?); unique-candidate count and head-alignment
    // spread measured per store (correlated picks and conditioning
    // strength are MEASURED here, not inferred).
    println!();
    println!("G3 — Issue 984 arms (soft16 + hard16 best-answer cosine, quality direction; N=512):");
    let (rows512, chains512, levels3) = build_store(0x985_FEED, 512);
    let noise_rows = store_transforms::noise_store(&rows512);
    let shuffled_rows = store_transforms::shuffled_store(&rows512);
    let mut scratch = SoftReadScratch::<D, K_SLOTS>::new();
    let mut arm_probe = |store: &[[f32; D]], level: usize| -> [f64; 4] {
        // [soft16, hard16, unique-candidates, mean head-alignment cos(q_k,q)]
        let soft = SoftRead::<D, K_SLOTS>::new(store).with_biases([GATE_BIAS; K_SLOTS]);
        let mut sums = [0.0f64; 4];
        for rep in 0..N_REPS {
            let qr = build_query_rep(0x985_0100 + rep as u64, &chains512);
            let q = qr.queries[level];
            let gold = chains512[0].hops[level];
            let _ = soft.read_into(&q, store, &mut scratch);
            sums[0] += soft.best_gated_row_cosine(store, &scratch, &gold) as f64;
            let (hb, _) = hard16_scores(&q, store, &gold, K_SLOTS);
            sums[1] += hb as f64;
            // Unique candidates across the 16 slots + conditioning spread.
            let mut picked: Vec<usize> = Vec::with_capacity(K_SLOTS);
            let mut align_sum = 0.0f32;
            let qu = unit(q);
            for head in &scratch.heads {
                let mut top = 0usize;
                let mut td = f32::NEG_INFINITY;
                for (i, r) in store.iter().enumerate() {
                    let d: f32 = head.iter().zip(r).map(|(a, x)| a * x).sum();
                    if d > td {
                        td = d;
                        top = i;
                    }
                }
                picked.push(top);
                let hc: f32 = head.iter().zip(&qu).map(|(a, b)| a * b).sum();
                align_sum += hc;
            }
            picked.sort_unstable();
            picked.dedup();
            sums[2] += picked.len() as f64;
            sums[3] += (align_sum / K_SLOTS as f32) as f64;
        }
        sums.map(|s| s / N_REPS as f64)
    };
    let margin = 0.05;
    let tol = 0.05;
    let mut g3 = true;
    for (label, level) in [
        ("c=2 (shallowest non-trivial)", 1usize),
        ("c=4 (deepest, GATED)", levels3 - 1),
    ] {
        let [real, real_hard, real_uniq, real_align] = arm_probe(&rows512, level);
        let [noise, noise_hard, noise_uniq, noise_align] = arm_probe(&noise_rows, level);
        let [shuffled, shuf_hard, shuf_uniq, shuf_align] = arm_probe(&shuffled_rows, level);
        println!("  {label}:");
        println!("    store     soft16   hard16   uniq-cand   head-align");
        println!(
            "    real      {:-8.4} {:-8.4} {:-11.1} {:+.4}",
            real, real_hard, real_uniq, real_align
        );
        println!(
            "    noise     {:-8.4} {:-8.4} {:-11.1} {:+.4}",
            noise, noise_hard, noise_uniq, noise_align
        );
        println!(
            "    shuffled  {:-8.4} {:-8.4} {:-11.1} {:+.4}",
            shuffled, shuf_hard, shuf_uniq, shuf_align
        );
        let empty = 0.0f64; // no store → no answer → the empty floor is 0
        let i1 = real >= noise + margin;
        let i2 = noise <= empty + tol;
        let i3 = shuffled <= real - margin;
        let gated = level == levels3 - 1;
        println!(
            "    I1 real ≥ noise + {margin}: {}   I2 noise ≤ empty + {tol}: {}   I3 shuffled ≤ real − {margin}: {}{}",
            if i1 { "OK" } else { "FAIL" },
            if i2 { "OK" } else { "FAIL" },
            if i3 { "OK" } else { "FAIL" },
            if gated {
                "  (gates G3)"
            } else {
                "  (disclosed, not gated)"
            }
        );
        if gated {
            g3 = i1 && i2 && i3;
        }
    }
    println!(
        "  note: E[max cos of 16 random draws] ≈ 0.22 at D=64 — a noise arm materially above that is head-ALIGNED selection, not pure random draw"
    );

    // ── G4 — the T6 second-signal cell over the shared W1/W2 worlds ─────
    println!();
    println!("G4 — CorpusDistanceGate second signal (shared bench 845 W1/W2 worlds):");
    let corpus = build_corpus();
    let gate = DistanceSoftGate::<D, K_SLOTS>::new(&corpus, MID, SCALE);
    let mut w1_ok = true;
    let mut w2_ok = true;
    for (seed, is_w1) in [(0x845_0100u64, true), (0x845_0200u64, false)] {
        let t_p: fn(f32, bool) -> f32 = if is_w1 { w1_true_p } else { w2_true_p };
        let world = build_world(seed, &corpus, t_p);
        let errors: Vec<bool> = world.iter().map(|a| !a.correct).collect();
        let score_sigs: Vec<f32> = world.iter().map(|a| a.p).collect();
        let soft_sigs: Vec<f32> = world.iter().map(|a| gate.soft_confidence(&a.q)).collect();
        let ood_seen = world.iter().filter(|a| a.ood).count();
        let a_score = aurc(&score_sigs, &errors);
        let a_soft = aurc(&soft_sigs, &errors);
        let sel = selective_accuracy_at(&soft_sigs, &errors, SEL30_RATE);
        println!(
            "  {}  AURC score {a_score:.4} · soft {a_soft:.4} · soft sel30 {sel:.4} (OOD {}/{} = {:.0}%)",
            if is_w1 { "W1" } else { "W2" },
            ood_seen,
            world.len(),
            100.0 * ood_seen as f32 / world.len() as f32
        );
        if is_w1 {
            w1_ok = a_soft < a_score;
        } else {
            w2_ok = a_soft > a_score;
        }
    }
    println!(
        "  W1 soft beats score-only: {}   W2 soft loses to score-only (fixture discriminates): {}",
        if w1_ok { "OK" } else { "FAIL" },
        if w2_ok { "OK" } else { "FAIL" }
    );
    let g4 = w1_ok && w2_ok;

    // ── Verdict ──────────────────────────────────────────────────────
    // MEASURED NEGATIVE (2026-11-06): report-only replication instrument.
    // The grade is struck — see the header and Issue 985. The one live
    // assert is fixture validity: if the hard arm stops collapsing, the
    // recorded negative is void and the cell re-opens.
    println!();
    let all = g1 && g2 && g3 && g4;
    println!("GATES: G1 {g1} · G2 {g2} · G3 {g3} · G4 {g4}");
    println!(
        "{}",
        if all {
            "✓ Bench 985 GOAT PASSED — the modelless soft read earns the opt-in lane"
        } else {
            "✗ Bench 985 GOAT FAILED — MEASURED NEGATIVE (2026-11-06): the deterministic-slot modelless floor loses to hard top-16; the grade is struck, the feature stays opt-in, Research 656 records the negative"
        }
    );
    assert!(
        collapse,
        "fixture validity: the hard16 collapse must exist or the recorded negative is void"
    );
}
