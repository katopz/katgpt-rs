//! Bench 986 — slot placement × effective conditioning (Issue 986; the
//! Issue 985 negative's two preregistered hypotheses, one cell, no tuning
//! passes).
//!
//! Fixture: the Bench 985 composition fixture UNCHANGED (chain decay 0.5,
//! 18 distractors/chain, N=512, K=16, τ=0.1, b=−2, 4 reps, seeds as
//! committed). The bar is the 985 bar verbatim per arm: soft16 ≥ hard16 at
//! every level AND > empty (0), plus Issue 984's arms at the deepest level.
//!
//! Arms:
//! - **A (placement)** — k-means++-seeded anchors (`SlotMode::KMeansPlusPlus`),
//!   conditioning UNCHANGED (the shipped β·⟨q,d⟩·d residual).
//! - **B (effective conditioning)** — FPS anchors unchanged, the fixed-length
//!   residual posture (`with_head_cos_target(0.90)` — preregistered exactly
//!   0.90, never swept).
//! - **C (both)**.
//! - **Mechanism-liveness gate** (the 985 finding, preregistered): mean
//!   cos(q_k,q) ≤ 0.95 AND unique-candidate count ≥ K/2 are PRECONDITIONS
//!   for interpreting an arm — an arm whose heads don't differ re-measures
//!   985's 1-candidate collapse, not placement.
//!
//! Report posture per the 985 precedent: gates print, exit 0, ONE live
//! assert on fixture validity (the hard16 collapse must exist or the cell
//! is void).

#[path = "../tests/common/mod.rs"]
mod common;
#[path = "../tests/common/store_transforms.rs"]
mod store_transforms;
#[path = "../tests/common/bench845_worlds.rs"]
mod worlds;

use katgpt_core::soft_read::{SlotMode, SoftRead, SoftReadScratch};
use std::hint::black_box;
use worlds::D;

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

const CHAIN_LEN: usize = 4;
const HOP_DECAY: f32 = 0.5;
const HOP_FRESH: f32 = 0.85;
const DISTRACTOR_W: f32 = 0.55;
const DISTRACTORS_PER_CHAIN: usize = 18;
const QUERY_NOISE: f32 = 0.1;
const K_SLOTS: usize = 16;
const N_REPS: usize = 4;
const GATE_BIAS: f32 = -2.0;
/// The preregistered effective-conditioning target (Arm B / C): mean
/// cos(q_k, q) ≈ 0.90 — exactly this value, never swept.
const HEAD_COS_TARGET: f32 = 0.90;
/// The mechanism-liveness ceiling (the 985 finding): an arm whose mean
/// head-cos exceeds this has not built 16 readers and cannot test
/// placement.
const LIVENESS_HEAD_COS: f64 = 0.95;
/// The interpretability floor on unique candidates (K/2).
const LIVENESS_MIN_UNIQUE: f64 = K_SLOTS as f64 / 2.0;

/// One chain: hop facts + same-topic distractors (verbatim from 985).
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

/// The N=512 store (verbatim from 985).
fn build_store(seed: u64, n_target: usize) -> (Vec<[f32; D]>, Vec<Chain>, usize) {
    let mut rng = SplitMix64(seed ^ 0xB00C);
    let mut rows: Vec<[f32; D]> = Vec::with_capacity(n_target);
    let mut chains = Vec::new();
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

struct QueryRep {
    queries: Vec<[f32; D]>,
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

/// The hard top-16 read (verbatim from 985).
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

/// One arm's per-level soft16 means + the liveness diagnostics.
struct ArmResult {
    /// best-gated-row cosine per level (replicate means).
    soft: [f64; CHAIN_LEN],
    /// mean cos(q_k, q) over slots, queries, reps (the liveness read).
    head_align: f64,
    /// mean unique top-gate candidates (the liveness read).
    unique: f64,
    /// slot-coverage entropy of the arm's bank.
    entropy: f32,
}

fn evaluate_arm(
    mode: SlotMode,
    effective: bool,
    rows: &[[f32; D]],
    chains: &[Chain],
    levels: usize,
) -> ArmResult {
    let mut soft = [0.0f64; CHAIN_LEN];
    let mut align_sum = 0.0f64;
    let mut unique_sum = 0.0f64;
    let mut scratch = SoftReadScratch::<D, K_SLOTS>::new();
    let mut read = SoftRead::<D, K_SLOTS>::with_mode(rows, mode).with_biases([GATE_BIAS; K_SLOTS]);
    if effective {
        read = read.with_head_cos_target(HEAD_COS_TARGET);
    }
    let entropy = read.bank().coverage_entropy(rows);
    let mut reads_count = 0usize;
    for rep in 0..N_REPS {
        let qr = build_query_rep(0x985_0100 + rep as u64, chains);
        for c in 0..levels {
            let gold = chains[0].hops[c];
            let q = qr.queries[c];
            let _ = read.read_into(&q, rows, &mut scratch);
            soft[c] += read.best_gated_row_cosine(rows, &scratch, &gold) as f64;
            // Liveness diagnostics per read.
            let qu = unit(q);
            let mut picked: Vec<usize> = Vec::with_capacity(K_SLOTS);
            let mut a = 0.0f32;
            for head in &scratch.heads {
                let mut top = 0usize;
                let mut td = f32::NEG_INFINITY;
                for (i, r) in rows.iter().enumerate() {
                    let d: f32 = head.iter().zip(r).map(|(x, y)| x * y).sum();
                    if d > td {
                        td = d;
                        top = i;
                    }
                }
                picked.push(top);
                a += head.iter().zip(&qu).map(|(x, y)| x * y).sum::<f32>();
            }
            picked.sort_unstable();
            picked.dedup();
            unique_sum += picked.len() as f64;
            align_sum += (a / K_SLOTS as f32) as f64;
            reads_count += 1;
            black_box(&q);
        }
    }
    ArmResult {
        soft: soft.map(|s| s / N_REPS as f64),
        head_align: align_sum / reads_count as f64,
        unique: unique_sum / reads_count as f64,
        entropy,
    }
}

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Bench 986 — slot placement × effective conditioning (Issue 986)");
    println!(
        "  (fixture = 985 verbatim: K={K_SLOTS}, τ=0.1, b={GATE_BIAS}, decay {HOP_DECAY}, {DISTRACTORS_PER_CHAIN} distractors, reps {N_REPS}; head-cos target {HEAD_COS_TARGET})"
    );
    println!("═══════════════════════════════════════════════════════════════");

    let (rows, chains, levels) = build_store(0x985_FEED, 512);
    let last = levels - 1;

    // The hard16 reference (unchanged code path).
    let mut hard = [0.0f64; CHAIN_LEN];
    for rep in 0..N_REPS {
        let qr = build_query_rep(0x985_0100 + rep as u64, &chains);
        for c in 0..levels {
            let (hb, _) = hard16_scores(&qr.queries[c], &rows, &chains[0].hops[c], K_SLOTS);
            hard[c] += hb as f64;
        }
    }
    for v in hard.iter_mut() {
        *v /= N_REPS as f64;
    }

    // Fixture validity (the ONE live assert): the hard collapse must exist.
    let collapse = hard[last] < hard[0] - 0.1;

    println!();
    println!("G0 — hard16 reference (best-answer cosine per level; empty=0):");
    for c in 0..levels {
        println!("  c={}  hard {:-7.4}", c + 1, hard[c]);
    }
    println!(
        "  hard collapse ({} → {}, needs ≥0.1 drop): {}",
        hard[0],
        hard[last],
        if collapse {
            "OK"
        } else {
            "FAIL — fixture rigged"
        }
    );

    // ── The three arms ───────────────────────────────────────────────
    println!();
    println!(
        "G1 — the three arms (soft16 per level; bar = soft16 ≥ hard16 at every level AND > 0):"
    );
    let arms: [(&str, SlotMode, bool); 3] = [
        ("A placement (k-means++)", SlotMode::KMeansPlusPlus, false),
        ("B conditioning (FPS+eff)", SlotMode::FarthestPoint, true),
        ("C both", SlotMode::KMeansPlusPlus, true),
    ];
    let mut results = Vec::new();
    for (label, mode, effective) in arms {
        let r = evaluate_arm(mode, effective, &rows, &chains, levels);
        results.push((label, r));
    }
    // The 985 FPS reference row (shipped posture) for context.
    let fps_ref = evaluate_arm(SlotMode::FarthestPoint, false, &rows, &chains, levels);

    println!("  level     hard16     985-FPS        A          B          C");
    for c in 0..levels {
        println!(
            "  c={:<5} {:-9.4} {:-9.4} {:-9.4} {:-9.4} {:-9.4}",
            c + 1,
            hard[c],
            fps_ref.soft[c],
            results[0].1.soft[c],
            results[1].1.soft[c],
            results[2].1.soft[c]
        );
    }

    // ── The mechanism-liveness gate (preregistered precondition) ─────
    println!();
    println!(
        "G2 — mechanism liveness (mean cos(q_k,q) ≤ {LIVENESS_HEAD_COS} AND unique ≥ {LIVENESS_MIN_UNIQUE}; a failed arm re-measures 985's collapse, not placement):"
    );
    let mut all_live = true;
    for (label, r) in &results {
        let live = r.head_align <= LIVENESS_HEAD_COS && r.unique >= LIVENESS_MIN_UNIQUE;
        all_live &= live;
        println!(
            "  {label:<26} head-cos {:+.4}  unique {:>5.1}  entropy {:.3}  {}",
            r.head_align,
            r.unique,
            r.entropy,
            if live { "LIVE" } else { "NOT INTERPRETABLE" }
        );
    }
    println!(
        "  {:<26} head-cos {:+.4}  unique {:>5.1}  entropy {:.3}  (985 reference row)",
        "985-FPS shipped", fps_ref.head_align, fps_ref.unique, fps_ref.entropy
    );

    // ── The bar, per arm (only live arms gate) ───────────────────────
    println!();
    println!("G3 — the bar per arm (soft16 ≥ hard16 at every level AND > 0):");
    let mut any_bar_pass = false;
    for (label, r) in &results {
        let beats_hard = (0..levels).all(|c| r.soft[c] >= hard[c]);
        let above_empty = r.soft[last] > 0.0;
        let bar = beats_hard && above_empty;
        any_bar_pass |= bar;
        println!(
            "  {label:<26} ≥hard16: {}   >0 at c={}: {}   → {}",
            if beats_hard { "OK" } else { "FAIL" },
            last + 1,
            if above_empty { "OK" } else { "FAIL" },
            if bar { "PASS" } else { "fail" }
        );
    }

    // ── Issue 984 store arms at the deepest level, per arm ───────────
    // The evidence instrument: does each arm's read depend on its STORE?
    // Real vs noise vs shuffled stores under the arm's own posture.
    println!();
    println!("G4 — Issue 984 store arms at the deepest level (quality direction, per arm):");
    let noise_rows = store_transforms::noise_store(&rows);
    let shuffled_rows = store_transforms::shuffled_store(&rows);
    let margin = 0.05;
    let mut g4 = true;
    for (label, mode, effective) in arms {
        let probe = |store: &[[f32; D]]| -> f64 {
            evaluate_arm(mode, effective, store, &chains, levels).soft[last]
        };
        let real = probe(&rows);
        let noise = probe(&noise_rows);
        let shuffled = probe(&shuffled_rows);
        let i1 = real >= noise + margin;
        let i2 = noise <= 0.0 + margin;
        let i3 = shuffled <= real - margin;
        let ok = i1 && i2 && i3;
        g4 &= ok;
        println!(
            "  {label:<26} real {real:.4}  noise {noise:.4}  shuffled {shuffled:.4}  I1 {} I2 {} I3 {}",
            if i1 { "OK" } else { "FAIL" },
            if i2 { "OK" } else { "FAIL" },
            if i3 { "OK" } else { "FAIL" },
        );
    }

    // ── Verdict (report posture; exit 0) ─────────────────────────────
    println!();
    let reopening = all_live && any_bar_pass && g4;
    println!(
        "GATES: collapse {} · liveness {all_live} · any-arm-bar {any_bar_pass} · 984-arms {g4}",
        if collapse { "OK" } else { "FAIL" }
    );
    println!(
        "{}",
        if reopening {
            "→ VERDICT INPUT: at least one live arm cleared the bar — the modelless floor lane RE-OPENS (Issue 985's negative gets the placement+conditioning footnote)"
        } else {
            "→ VERDICT INPUT: no arm cleared the bar (liveness or bar failed) — the query-conditioned aggregate axis CLOSES at the floor level; the trained lane (riir-train Plan 488 R1) proceeds regardless"
        }
    );
    assert!(
        collapse,
        "fixture validity: the hard16 collapse must exist or this cell is void"
    );
}
