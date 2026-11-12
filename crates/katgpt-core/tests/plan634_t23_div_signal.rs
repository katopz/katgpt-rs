//! Plan 634 T2.3 — the F3 Div-signal loop (REPORT-ONLY this phase): a
//! CGSP-shaped cycle loop on the planted clique-graph — per cycle, a
//! seeded pool; `Div_P^τ` of the pool's uniform distribution; the
//! batch-admission rate the pool yields; Spearman rank correlation of
//! `Div_t` against the NEXT cycle's admission rate, under a
//! BreakevenDifficultyFilter-shaped band (middle two Div quartiles) and a
//! ColinearityBatchGate-shaped non-degeneracy filter (admitted batch's
//! mean pairwise e_τ cosine < 0.95).
//!
//! NO admission bar — the correlation coefficient is PRINTED (the pass/fail
//! bar is a later phase gated on riir-ai Plan 646 data). The asserts pin
//! only the harness's own contract: determinism (identical re-run) and a
//! non-vacuous report (n_used ≥ 8).

#[path = "plan634/mod.rs"]
mod plan634_common;

use katgpt_core::diversity::diffusion::{DiffusionEmbedding, batch_select_diffusion};
use katgpt_core::set_admission::admit_into;
use plan634_common::{Xorshift64, planted_clique_graph, spearman_rho};

const K: usize = 8;
const M: usize = 6;
const N_CYCLES: usize = 40;
const POOL_LEN: usize = 12;
const TAU: usize = 2;
const SEED: u64 = 0x0634_00A3_u64;
const COLINEARITY_GATE: f32 = 0.95;

/// One cycle's record.
struct CycleRow {
    div: f64,
    admission_rate: f64,
    batch_mean_cosine: f64,
}

fn run_cycles() -> Vec<CycleRow> {
    let (g, _, _, _, _) = planted_clique_graph(K, M);
    let n = g.n;
    let emb = DiffusionEmbedding::build(&g);
    let mut rng = Xorshift64::new(SEED);
    let mut q = vec![0.0_f32; n];
    let mut r = vec![0.0_f32; n];
    let mut rows = Vec::with_capacity(N_CYCLES);
    for _ in 0..N_CYCLES {
        // Seeded pool with GENUINELY VARYING spread: choose s ∈ 2..=8
        // clusters (deterministic partial shuffle), then fill the pool
        // round-robin from those clusters — the pool's spread (and so its
        // Div) differs cycle to cycle, which is the signal the correlation
        // reads (a same-shape sampler every cycle collapses Div to a
        // constant and the band to a point — the vacuous-report trap the
        // first run hit).
        let s = 2 + rng.next_index(7);
        let mut chosen: Vec<usize> = (0..K).collect();
        for i in (1..chosen.len()).rev() {
            let j = rng.next_index(i + 1);
            chosen.swap(i, j);
        }
        chosen.truncate(s);
        let mut pool = Vec::with_capacity(POOL_LEN);
        let mut cursor = 0_usize;
        while pool.len() < POOL_LEN {
            let c = chosen[cursor % s];
            cursor += 1;
            let node = c * M + rng.next_index(M);
            if !pool.contains(&node) {
                pool.push(node);
            }
        }
        // Div of the pool's uniform distribution — mass ON the pool
        // nodes only (a whole-graph-uniform μ ignores the pool entirely
        // and collapses Div to a per-cycle constant — the vacuous-report
        // trap the first run hit).
        let mut mu = vec![0.0_f32; n];
        for &node in &pool {
            mu[node] = 1.0 / POOL_LEN as f32;
        }
        let div = g.div_tau(&mu, TAU, &mut q, &mut r) as f64;
        // The pool's admission rate through the shipped batch selector.
        let mut out = [u16::MAX; POOL_LEN];
        let admitted = batch_select_diffusion::<8>(&g, &pool, TAU, 0.0, &mut out);
        let admission_rate = admitted as f64 / POOL_LEN as f64;
        // ColinearityBatchGate-shaped non-degeneracy statistic: mean
        // pairwise cosine of the ADMITTED nodes' e_τ coordinates.
        let mut cos_sum = 0.0_f64;
        let mut cos_n = 0_usize;
        for i in 0..admitted {
            for j in (i + 1)..admitted {
                let a = emb.spectral_coordinates::<8>(pool[out[i] as usize], TAU);
                let b = emb.spectral_coordinates::<8>(pool[out[j] as usize], TAU);
                let dot: f64 = a
                    .iter()
                    .zip(b.iter())
                    .map(|(x, y)| (*x as f64) * (*y as f64))
                    .sum();
                let na: f64 = a.iter().map(|x| *x as f64 * *x as f64).sum::<f64>().sqrt();
                let nb: f64 = b.iter().map(|x| *x as f64 * *x as f64).sum::<f64>().sqrt();
                if na > 0.0 && nb > 0.0 {
                    cos_sum += dot / (na * nb);
                    cos_n += 1;
                }
            }
        }
        let mean_cos = if cos_n > 0 {
            cos_sum / cos_n as f64
        } else {
            0.0
        };
        rows.push(CycleRow {
            div,
            admission_rate,
            batch_mean_cosine: mean_cos,
        });
    }
    rows
}

#[test]
fn t23_f3_div_signal_report_only() {
    // A first `admit_into` reference keeps the set_admission consumer in
    // the loop (the admission-rate lane the bar will eventually read);
    // the selector above is the batch operator, this the raw kernel —
    // both compile-time-checked on the same feature set.
    let probe_pool: Vec<[f32; 8]> = vec![[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; 4];
    let mut probe_out = [u16::MAX; 4];
    let probe_cfg = katgpt_core::set_admission::SetAdmissionConfig {
        alpha_align: 0.0,
        kappa_div: 1.0,
        ..katgpt_core::set_admission::SetAdmissionConfig::default()
    };
    let mut probe_scratch = katgpt_core::set_admission::AdmissionScratchCore::<8>::new();
    let probe_admitted = admit_into(
        &probe_cfg,
        &probe_pool,
        &[0.0; 4],
        &[0.0; 8],
        &mut probe_out,
        &mut probe_scratch,
    );
    assert_eq!(
        probe_admitted, 1,
        "collinear probe pool must admit exactly one"
    );

    let rows = run_cycles();
    // Determinism: a second run is byte-identical.
    let rows2 = run_cycles();
    for (a, b) in rows.iter().zip(rows2.iter()) {
        assert_eq!(
            a.div.to_bits(),
            b.div.to_bits(),
            "Div must be bit-reproducible"
        );
        assert_eq!(
            a.admission_rate.to_bits(),
            b.admission_rate.to_bits(),
            "admission rate must be bit-reproducible"
        );
    }

    // BreakevenDifficultyFilter-shaped band: keep the middle two Div
    // quartiles (the band the eventual bar will live inside).
    let mut divs: Vec<f64> = rows.iter().map(|r| r.div).collect();
    divs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q1 = divs[N_CYCLES / 4];
    let q3 = divs[3 * N_CYCLES / 4];

    let mut div_series = Vec::new();
    let mut next_rate_series = Vec::new();
    for w in rows.windows(2) {
        let (cur, nxt) = (&w[0], &w[1]);
        let in_band = cur.div >= q1 && cur.div <= q3;
        let nondegenerate = cur.batch_mean_cosine < COLINEARITY_GATE as f64;
        if in_band && nondegenerate {
            div_series.push(cur.div);
            next_rate_series.push(nxt.admission_rate);
        }
    }
    let n_used = div_series.len();
    assert!(
        n_used >= 8,
        "report is vacuous: only {n_used}/{N_CYCLES} cycles survived the band+gate \
         (q1={q1:.6} q3={q3:.6}) — widen the fixture, never the assert"
    );
    let rho = spearman_rho(&div_series, &next_rate_series);
    let mean_div = div_series.iter().sum::<f64>() / n_used as f64;
    let mean_rate = next_rate_series.iter().sum::<f64>() / n_used as f64;
    println!(
        "F3 REPORT (no bar this phase): Spearman(Div_t, admission_rate_t+1) = {rho:+.4} \
         over n_used={n_used}/{N_CYCLES} cycles\n  \
         band = Div ∈ [{q1:.6}, {q3:.6}], mean Div {mean_div:.6}, mean next rate {mean_rate:.6}\n  \
         a POSITIVE ρ supports the Lemma-5.3-shaped read (spreadier pools admit better next \
         cycle); the bar waits for riir-ai Plan 646 data"
    );
}
