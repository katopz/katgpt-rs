//! Plan 634 T2.4 — the F4 well-mixed tie battery (the DEMOTE tripwire),
//! RESTATED per the two-round delegated verdict (session
//! `6c7f5a02-48ea-4836-b7d7-57180b5d2c61`; round 1 REVISE → B′,
//! round 2 **AGREE** → B″).
//!
//! **The original red run (verbatim, 2026-11-12, one seed, 600 steps):**
//! batch-colinearity alg4 +0.569476 · srw +0.708531 · jump +0.279631;
//! coverage 1.0000/1.0000/1.0000 (saturated — tests nothing). The literal
//! "tie BOTH baselines" demand is structurally unsatisfiable for any
//! walk: consecutive visits are graph-adjacent (diffusion-similar by
//! construction), so chunk-autocorrelation cannot equal both the
//! lingering-walk anchor (0.709) and the iid anchor (0.280) at once.
//!
//! **B″ arms (bands/lag/seeds/budgets pre-registered round 1, unchanged —
//! this is NOT a post-run band widen):**
//! (i)  raw colinearity no-regression vs the SRW: `alg4 ≤ srw + 0.05`;
//! (ii) THINNED colinearity (lag 20 ≥ the C48 relaxation time ~17):
//!      CONTROL `|thinned_srw − thinned_jump| ≤ 0.05` first — measured
//!      0.0496, **held at the edge** (margin 0.0004, inside the ±0.05–0.08
//!      seed half-ranges: enough to treat walk/iid colinearity as not
//!      comparable, WEAK evidence, recorded as such; seeds NOT re-run to
//!      buy margin — that is retuning) — then the tripwire
//!      `|thinned_alg4 − thinned_jump| ≤ 0.05`;
//! (iii) coverage no-regression **vs the SRW** (the paper-faithful
//!      walk-shaped baseline) at non-saturating budgets
//!      steps ∈ {24, 48, 96}, band 0.10 unchanged. Jump coverage is
//!      **REPORTED-ONLY** here: round 1's tie-vs-jump demand was the
//!      round-1 verdict's own mis-specification — at fixed budget, iid
//!      sampling covers `n·(1−(1−1/n)^t)` nodes and NO nearest-neighbor
//!      walk expands that fast; the SRW itself misses the 0.10 band vs
//!      jump at 48/96 (0.159/0.200 — printed verbatim below). A gate the
//!      reference walk fails measures walks-vs-iid, the same conflation
//!      round 1 rejected for colinearity.
//!
//! **The measured trade-off (read BOTH numbers together):** against the
//! SRW, Alg-4 yields LOWER batch colinearity (0.598 vs 0.690 — more
//! diverse batches) AND LOWER distinct coverage — the absolute deficit
//! GROWS with budget (0.051 → 0.060 → 0.068 at 24/48/96) while only the
//! RELATIVE deficit shrinks (17% → 13% → 10%). The diversity win is paid
//! in revisit rate; neither number stands alone.
//!
//! **The quasi-cycling watch item (report-only arm below):** the
//! stateless per-node QP distribution makes the walk quasi-cycle
//! transiently; the floor mass breaks it asymptotically. The prediction:
//! the absolute alg4−srw coverage gap turns around at larger budgets. If
//! it keeps growing there, "coverage regresses vs the SRW at longer
//! horizons" is recorded as a known limitation — every gate still passes.
//!
//! FAILURE of (i), (ii)'s tripwire (control holding), or (iii) ⇒ the
//! walk-time selector stays opt-in-report-only permanently (the plan
//! records the demote). Failure of the control arm ⇒ back to the verdict
//! with the evidence, never a silent band widen.

#[path = "plan634/mod.rs"]
mod plan634_common;

use katgpt_core::diversity::diffusion::{
    DiffusionEmbedding, LocalScratch, neighbor_diversity_pick,
};
use plan634_common::{Xorshift64, circulant_expander};

const N: usize = 48;
const OFFSETS: [usize; 3] = [1, 2, 4];
const LONG_WALK: usize = 600;
const THIN_LAG: usize = 20; // ≥ the relaxation time (~17)
const RAW_BATCH: usize = 12;
const THIN_BATCH: usize = 10;
const TAU: usize = 2;
const ETA: f32 = 0.05;
/// 16 odd seeds (distinct xorshift streams — `new` ORs the low bit).
const SEEDS: [u64; 16] = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31];
/// Pre-registered round 1 (unchanged round 2 — not a post-run widen).
const COLINEARITY_BAND: f64 = 0.05;
const COVERAGE_BAND: f64 = 0.10;
const COVERAGE_BUDGETS: [usize; 3] = [24, 48, 96]; // n/2, n, 2n
const WATCH_BUDGET: usize = 192; // the quasi-cycling report-only check (4n)

#[derive(Clone, Copy, PartialEq)]
enum Rule {
    Alg4,
    Srw,
    Jump,
}

/// One walk of `steps` steps; returns the visited node sequence.
fn walk(rule: Rule, seed: u64, steps: usize) -> Vec<usize> {
    let g = circulant_expander(N, &OFFSETS);
    let mut rng = Xorshift64::new(seed);
    let mut scratch = LocalScratch::new(N);
    let mut visited = Vec::with_capacity(steps);
    let mut cur = rng.next_index(N);
    for _ in 0..steps {
        visited.push(cur);
        cur = match rule {
            Rule::Jump => rng.next_index(N),
            Rule::Srw => {
                let (cols, _) = g.row(cur);
                cols[rng.next_index(cols.len())]
            }
            Rule::Alg4 => {
                let (cols, _) = g.row(cur);
                let kk = cols.len();
                let mut mu = vec![0.0_f32; kk];
                neighbor_diversity_pick(&g, cur, TAU, ETA, &mut scratch, &mut mu);
                let pick = rng.next_f64();
                let mut acc = 0.0_f64;
                let mut chosen = kk - 1;
                for (i, &p) in mu.iter().enumerate() {
                    acc += p as f64;
                    if pick < acc {
                        chosen = i;
                        break;
                    }
                }
                cols[chosen]
            }
        };
    }
    visited
}

/// Mean pairwise e_τ cosine within consecutive `batch`-sized chunks.
fn chunk_cosine(visited: &[usize], emb: &DiffusionEmbedding, batch: usize) -> f64 {
    let mut cos_sum = 0.0_f64;
    let mut batches = 0_usize;
    for chunk in visited.chunks(batch) {
        if chunk.len() < 2 {
            continue;
        }
        let mut s = 0.0_f64;
        let mut cnt = 0_usize;
        for i in 0..chunk.len() {
            for j in (i + 1)..chunk.len() {
                let a = emb.spectral_coordinates::<8>(chunk[i], TAU);
                let b = emb.spectral_coordinates::<8>(chunk[j], TAU);
                let dot: f64 = a
                    .iter()
                    .zip(b.iter())
                    .map(|(x, y)| (*x as f64) * (*y as f64))
                    .sum();
                let na: f64 = a.iter().map(|x| *x as f64 * *x as f64).sum::<f64>().sqrt();
                let nb: f64 = b.iter().map(|x| *x as f64 * *x as f64).sum::<f64>().sqrt();
                if na > 0.0 && nb > 0.0 {
                    s += dot / (na * nb);
                    cnt += 1;
                }
            }
        }
        if cnt > 0 {
            cos_sum += s / cnt as f64;
            batches += 1;
        }
    }
    cos_sum / batches as f64
}

/// Distinct-visit share over the WHOLE walk (call at the budget's length).
fn coverage(visited: &[usize]) -> f64 {
    let mut seen = std::collections::HashSet::new();
    for &v in visited {
        seen.insert(v);
    }
    seen.len() as f64 / N as f64
}

/// mean and half-range of a per-seed statistic.
fn mean_halfrange(vals: &[f64]) -> (f64, f64) {
    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
    let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    (mean, (max - min) / 2.0)
}

#[test]
fn t24_f4_well_mixed_tie_both_baselines() {
    let g = circulant_expander(N, &OFFSETS);
    let emb = DiffusionEmbedding::build(&g);
    let _ = &g;

    // ── Arm (i): RAW colinearity, no-regression vs the SRW ─────────────
    let mut raw_alg4 = Vec::with_capacity(SEEDS.len());
    let mut raw_srw = Vec::with_capacity(SEEDS.len());
    let mut raw_jump = Vec::with_capacity(SEEDS.len());
    for &seed in &SEEDS {
        raw_alg4.push(chunk_cosine(
            &walk(Rule::Alg4, seed, LONG_WALK),
            &emb,
            RAW_BATCH,
        ));
        raw_srw.push(chunk_cosine(
            &walk(Rule::Srw, seed, LONG_WALK),
            &emb,
            RAW_BATCH,
        ));
        raw_jump.push(chunk_cosine(
            &walk(Rule::Jump, seed, LONG_WALK),
            &emb,
            RAW_BATCH,
        ));
    }
    let (m_alg4, hr_alg4) = mean_halfrange(&raw_alg4);
    let (m_srw, hr_srw) = mean_halfrange(&raw_srw);
    let (m_jump, hr_jump) = mean_halfrange(&raw_jump);
    println!(
        "F4(i) raw chunk-cosine (lag 1 — the ORIGINAL conflated statistic, reported \
         only across walk/iid):\n  alg4 {m_alg4:+.6} ±{hr_alg4:.4} · srw {m_srw:+.6} \
         ±{hr_srw:.4} · jump {m_jump:+.6} ±{hr_jump:.4}"
    );
    assert!(
        m_alg4 <= m_srw + COLINEARITY_BAND,
        "F4(i) NO-REGRESSION FAILED: raw colinearity alg4 {m_alg4:+.6} exceeds srw \
         {m_srw:+.6} + band {COLINEARITY_BAND} — the selector makes the walk's own \
         batch diversity WORSE than the plain random walk; the plan records the demote"
    );

    // ── Arm (ii): THINNED colinearity — control first, then the tripwire ─
    let thin = |rule: Rule| -> Vec<f64> {
        SEEDS
            .iter()
            .map(|&seed| {
                let visited = walk(rule, seed, LONG_WALK);
                let thinned: Vec<usize> = visited.iter().step_by(THIN_LAG).copied().collect();
                chunk_cosine(&thinned, &emb, THIN_BATCH)
            })
            .collect()
    };
    let thin_alg4 = thin(Rule::Alg4);
    let thin_srw = thin(Rule::Srw);
    let thin_jump = thin(Rule::Jump);
    let (t_alg4, thr_alg4) = mean_halfrange(&thin_alg4);
    let (t_srw, thr_srw) = mean_halfrange(&thin_srw);
    let (t_jump, thr_jump) = mean_halfrange(&thin_jump);
    println!(
        "F4(ii) thinned chunk-cosine (lag {THIN_LAG} ≥ relaxation ~17):\n  \
         alg4 {t_alg4:+.6} ±{thr_alg4:.4} · srw {t_srw:+.6} ±{thr_srw:.4} · \
         jump {t_jump:+.6} ±{thr_jump:.4}"
    );
    // The CONTROL validates the thinning premise. Round-2 condition 2: the
    // measured margin was 0.0004 — HELD AT THE EDGE, weak evidence,
    // recorded as such (seeds are NOT re-run to buy margin: retuning).
    let control_delta = (t_srw - t_jump).abs();
    assert!(
        control_delta <= COLINEARITY_BAND,
        "F4(ii) CONTROL FAILED: thinned srw {t_srw:+.6} vs thinned jump {t_jump:+.6} \
         (|Δ| {control_delta:.4} > band {COLINEARITY_BAND}) — lag {THIN_LAG} did not \
         decorrelate the plain walk to iid levels; the thinning premise is \
         undemonstrated. Take this to a verdict round with the numbers, never \
         widen the band silently"
    );
    println!(
        "  control |thinned_srw − thinned_jump| = {control_delta:.4} ≤ \
         {COLINEARITY_BAND} — held{}",
        if COLINEARITY_BAND - control_delta < 0.01 {
            " AT THE EDGE (margin < 0.01, within seed noise: weak evidence, recorded)"
        } else {
            ""
        }
    );
    // The RESTATED jump arm — the demote tripwire lives here.
    assert!(
        (t_alg4 - t_jump).abs() <= COLINEARITY_BAND,
        "F4(ii) THINNED TIE FAILED (control held): thinned alg4 {t_alg4:+.6} vs \
         thinned jump {t_jump:+.6} (|Δ| {:.4} > band {COLINEARITY_BAND}) — at \
         matched revisit profiles the selector's samples are still more colinear \
         than iid sampling; a REAL primitive defect. The plan records the demote",
        (t_alg4 - t_jump).abs()
    );

    // ── Arm (iii): coverage no-regression vs the SRW (walk-shaped bar) ──
    // Jump coverage prints REPORT-ONLY beside it (the round-1
    // mis-specification: the walk class itself misses the band — the
    // structural walk/iid locality gap, disclosed verbatim per budget).
    println!(
        "F4(iii) coverage (bar = no-regression vs SRW, band {COVERAGE_BAND}; jump \
         REPORTED-ONLY — round-1 mis-specification, the walk class itself misses it):"
    );
    let mut deficits = Vec::with_capacity(COVERAGE_BUDGETS.len());
    for &steps in &COVERAGE_BUDGETS {
        let mut cov_alg4 = Vec::with_capacity(SEEDS.len());
        let mut cov_srw = Vec::with_capacity(SEEDS.len());
        let mut cov_jump = Vec::with_capacity(SEEDS.len());
        for &seed in &SEEDS {
            cov_alg4.push(coverage(&walk(Rule::Alg4, seed, steps)));
            cov_srw.push(coverage(&walk(Rule::Srw, seed, steps)));
            cov_jump.push(coverage(&walk(Rule::Jump, seed, steps)));
        }
        let (ca, ha) = mean_halfrange(&cov_alg4);
        let (cs, hs) = mean_halfrange(&cov_srw);
        let (cj, hj) = mean_halfrange(&cov_jump);
        // Condition 1: BOTH deficit forms recorded — the absolute deficit
        // GROWS with budget, only the RELATIVE one shrinks.
        let abs_deficit = cs - ca;
        let rel_deficit = abs_deficit / cs;
        deficits.push((steps, abs_deficit, rel_deficit));
        println!(
            "  @{steps:>3} steps: alg4 {ca:.4} ±{ha:.4} · srw {cs:.4} ±{hs:.4} · \
             jump {cj:.4} ±{hj:.4}  [alg4−srw abs {abs_deficit:+.4} rel {:.1}% · \
             srw−jump {:+.4} REPORTED]",
            rel_deficit * 100.0,
            cs - cj
        );
        assert!(
            (ca - cs).abs() <= COVERAGE_BAND,
            "F4(iii) coverage no-regression vs SRW FAILED @ {steps} steps: alg4 \
             {ca:.4} vs srw {cs:.4} (|Δ| {:.4} > band {COVERAGE_BAND})",
            (ca - cs).abs()
        );
    }

    // ── Report-only watch item: the quasi-cycling hypothesis at 4n ─────
    // Prediction (pre-stated round 2, condition 3): if the stateless-QP
    // quasi-cycling hypothesis holds, the ABSOLUTE alg4−srw gap turns
    // around here; if it keeps growing, "coverage regresses vs the SRW at
    // longer horizons" is recorded as a known limitation (every gate
    // still passes — never a demotion from this arm).
    let mut w_alg4 = Vec::with_capacity(SEEDS.len());
    let mut w_srw = Vec::with_capacity(SEEDS.len());
    for &seed in &SEEDS {
        w_alg4.push(coverage(&walk(Rule::Alg4, seed, WATCH_BUDGET)));
        w_srw.push(coverage(&walk(Rule::Srw, seed, WATCH_BUDGET)));
    }
    let (wa, _) = mean_halfrange(&w_alg4);
    let (ws, _) = mean_halfrange(&w_srw);
    let gap_here = ws - wa;
    let (last_steps, last_abs, _) = deficits[deficits.len() - 1];
    let turned = gap_here <= last_abs;
    println!(
        "F4(watch) coverage @{WATCH_BUDGET} (report-only): alg4 {wa:.4} · srw {ws:.4} \
         → abs deficit {gap_here:+.4} vs {last_abs:+.4} @{last_steps} — the \
         quasi-cycling prediction {}",
        if turned {
            "TURNS AROUND (floor mass mixes; hypothesis supported)"
        } else {
            "DOES NOT turn (recorded as a known limitation: coverage regresses vs \
             the SRW at longer horizons)"
        }
    );

    println!("F4: B″ arms hold ✓ (demote tripwire armed, not fired)");
}
