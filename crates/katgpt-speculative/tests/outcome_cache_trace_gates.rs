//! Issue 976 — pre-registered gate battery for the JetSpec++ outcome cache,
//! run on a FROZEN teacher-forced trace (Bench 990).
//!
//! Trace (modelless, deterministic, BLAKE3-pinned input):
//! - corpus: `tests/data/austen_pride_prejudice_tail.txt` (public domain,
//!   already tracked for Plan 601), lowercased word tokens;
//! - drafter: `bigram_markov` greedy chain fitted on tokens `[0, 40%)`;
//! - verifier: the corpus itself (greedy-lossless teacher forcing — the
//!   accepted prefix is the leading run of drafted tokens equal to the text,
//!   the bonus is the text token after it);
//! - calibration window: rounds over `[40%, 70%)`; test window `[70%, 100%)`.
//!
//! Gates (Issue 976 §Gates): Prop 2b paired hit rate (per-round top-B ≥
//! both fixed allocations at every B), conformal coverage at nominal rate,
//! E[accepted] forecast vs the conformal-naive floor (Report the Floor),
//! Issue-940 determinism, M6 staleness ladder (stale → predicted → true).
//!
//! ```sh
//! cargo test -p katgpt-speculative --features outcome_cache,bigram_markov \
//!   --test outcome_cache_trace_gates --release -- --nocapture
//! ```

use katgpt_core::conformal::{
    PredictiveInterval, PredictiveOutput, UqPrimitiveUnderTest, run_floor_comparison,
};
use katgpt_speculative::bigram_markov::{BigramMarkovBuilder, BigramMarkovTable};
use katgpt_speculative::outcome_cache::*;
use std::collections::HashMap;

const CORPUS: &str = include_str!("../../../tests/data/austen_pride_prejudice_tail.txt");
/// BLAKE3 of the corpus bytes — the frozen-trace pin.
const CORPUS_BLAKE3: &str = "4318b9aff93523c9108d46f07c2030fb99f659b40da35246c6b54d34ac22f748";
const BUDGETS: [usize; 5] = [1, 2, 4, 8, 16];
const ALPHA: f32 = 0.1;

#[derive(Clone, Copy)]
struct Round {
    conf: [f32; GAMMA],
    n: usize,
    k: usize,
    bonus: u32,
    /// Drafter-residual bonus rows for this round (round-conditioned p̂(a|k)).
    rows: BonusRows,
}

fn tokenize(text: &str) -> (Vec<u32>, usize) {
    let mut ids: HashMap<String, u32> = HashMap::new();
    let mut out = Vec::new();
    for w in text
        .split(|c: char| !c.is_ascii_alphabetic())
        .filter(|w| !w.is_empty())
    {
        let w = w.to_ascii_lowercase();
        let next = ids.len() as u32;
        out.push(*ids.entry(w).or_insert(next));
    }
    (out, ids.len())
}

fn rounds(table: &BigramMarkovTable, toks: &[u32], lo: usize, hi: usize) -> Vec<Round> {
    let mut out = Vec::new();
    let mut t = lo.max(1);
    while t + GAMMA < hi {
        let mut conf = [0.0f32; GAMMA];
        let mut draft = [0u32; GAMMA];
        let mut prev = toks[t - 1];
        let mut n = 0;
        while n < GAMMA {
            let Some((succ, probs)) = table.successors(prev) else {
                break;
            };
            draft[n] = succ[0];
            conf[n] = probs[0];
            prev = succ[0];
            n += 1;
        }
        let k = (0..n).take_while(|&i| draft[i] == toks[t + i]).count();
        let mut rows = BonusRows::default();
        for j in 0..=n {
            let prev = if j == 0 { toks[t - 1] } else { draft[j - 1] };
            let rejected = (j < n).then(|| draft[j]);
            if let Some((succ, probs)) = table.successors(prev) {
                rows.set_residual_row(j, succ, probs, rejected);
            }
        }
        out.push(Round {
            conf,
            n,
            k,
            bonus: toks[t + k],
            rows,
        });
        t += k + 1;
    }
    out
}

struct Calibrated {
    pod: BonusHistogramPod,
    recal: ConfidenceRecalibrator,
    freq: [u32; GRID],
    lambda: f32,
    ladder_r: f32,
    kl: f32,
}

fn pk_of(cal: &Calibrated, r: &Round, recalibrate: bool) -> [f32; K_SLOTS] {
    let mut pk = [0.0f32; K_SLOTS];
    if recalibrate {
        let mut c = [0.0f32; GAMMA];
        cal.recal.apply(&r.conf[..r.n], &mut c);
        accept_count_distribution(&c[..r.n], &mut pk);
    } else {
        accept_count_distribution(&r.conf[..r.n], &mut pk);
    }
    pk
}

fn calibrate(cal_rounds: &[Round]) -> Calibrated {
    let mut b = HistogramBuilder::new();
    let mut recal = ConfidenceRecalibrator::default();
    for r in cal_rounds {
        b.observe(r.k, r.bonus);
        recal.observe(&r.conf[..r.n], r.k);
    }
    let (pod, kl) = match b.try_freeze(0.05, true) {
        FreezeVerdict::Converged(p, kl) | FreezeVerdict::Forced(p, kl) => (p, kl),
        v => panic!("freeze: {v:?}"),
    };
    let mut freq = [0u32; GRID];
    let mut accepted = 0usize;
    let mut reached = 0usize;
    for r in cal_rounds {
        freq[cell(r.k, pod.slot_of(r.bonus))] += 1;
        accepted += r.k;
        reached += (r.k + 1).min(r.n);
    }
    let mut cal = Calibrated {
        pod,
        recal,
        freq,
        lambda: 0.0,
        ladder_r: accepted as f32 / reached.max(1) as f32,
        kl,
    };
    let mut realized: Vec<f32> = cal_rounds
        .iter()
        .map(|r| {
            let mut s = [0.0f32; GRID];
            let rows = r.rows.with_backoff(&cal.pod);
            outcome_scores_rows(&pk_of(&cal, r, true), &rows, &mut s);
            s[cell(r.k, rows.slot_of(r.k, r.bonus))]
        })
        .collect();
    cal.lambda = conformal_threshold(&mut realized, ALPHA);
    cal
}

#[derive(Debug, PartialEq, Clone, Copy, Default)]
struct Row {
    ctx: usize,
    ctx_ceiling: usize,
    ctx_stale: usize,
    per_round_raw: usize,
    per_round: usize,
    conformal: usize,
    conformal_size: usize,
    stat: usize,
    ladder: usize,
    stale: usize,
    ceiling: usize,
    wins_vs_static: usize,
    losses_vs_static: usize,
}

fn evaluate(cal: &Calibrated, test: &[Round], b: usize, backoff: bool) -> Row {
    let mut row = Row::default();
    let mut st = [0u16; GRID];
    let ns = static_top_b(&cal.freq, b, &mut st);
    let mut ld = [0u16; GRID];
    let nl = geometric_ladder(&cal.pod, cal.ladder_r, b, &mut ld);
    let mut prev_sel = [0u16; GRID];
    let mut prev_n = 0usize;
    let mut prev_ctx: [(usize, u32); GRID] = [(usize::MAX, 0); GRID];
    let mut prev_ctx_n = 0usize;
    for r in test {
        // Round-conditioned posterior (drafter residual rows) — the Prop 2b arm.
        let mut sc = [0.0f32; GRID];
        let mut selc = [0u16; GRID];
        let rows = if backoff {
            r.rows.with_backoff(&cal.pod)
        } else {
            r.rows
        };
        outcome_scores_rows(&pk_of(cal, r, true), &rows, &mut sc);
        let nc = select_top_b(&sc, b, &mut selc);
        let rc = cell(r.k, rows.slot_of(r.k, r.bonus));
        let h_ctx = hit(&selc[..nc], rc);
        row.ctx += usize::from(h_ctx);
        row.ctx_ceiling += usize::from(!is_fallback(rc));
        row.ctx_stale += usize::from(prev_ctx[..prev_ctx_n].contains(&(r.k, r.bonus)));
        for (dst, &c) in prev_ctx.iter_mut().zip(&selc[..nc]) {
            let (k, slot) = (c as usize / ROW, c as usize % ROW);
            *dst = (k, rows.tokens[k][slot]);
        }
        prev_ctx_n = nc;
        let mut cs = [0u16; GRID];
        let ncs = select_conformal(&sc, cal.lambda, b, &mut cs);
        row.conformal += usize::from(hit(&cs[..ncs], rc));
        row.conformal_size += ncs;
        let realized = cell(r.k, cal.pod.slot_of(r.bonus));
        let mut s = [0.0f32; GRID];
        let mut sel = [0u16; GRID];
        outcome_scores(&pk_of(cal, r, false), &cal.pod, &mut s);
        let n = select_top_b(&s, b, &mut sel);
        row.per_round_raw += usize::from(hit(&sel[..n], realized));
        outcome_scores(&pk_of(cal, r, true), &cal.pod, &mut s);
        let n = select_top_b(&s, b, &mut sel);
        let h_pr = hit(&sel[..n], realized);
        row.per_round += usize::from(h_pr);
        let h_st = hit(&st[..ns], realized);
        row.stat += usize::from(h_st);
        row.ladder += usize::from(hit(&ld[..nl], realized));
        row.stale += usize::from(hit(&prev_sel[..prev_n], realized));
        row.ceiling += usize::from(!is_fallback(realized));
        row.wins_vs_static += usize::from(h_ctx && !h_st);
        row.losses_vs_static += usize::from(!h_ctx && h_st);
        prev_sel = sel;
        prev_n = n;
    }
    row
}

struct Fixture {
    corpus_digest: String,
    cal: Calibrated,
    test: Vec<Round>,
    n_cal: usize,
}

fn fixture() -> Fixture {
    let corpus_digest = blake3::hash(CORPUS.as_bytes()).to_hex().to_string();
    let (toks, vocab) = tokenize(CORPUS);
    let fit_end = toks.len() * 4 / 10;
    let cal_end = toks.len() * 7 / 10;
    let mut bb = BigramMarkovBuilder::new();
    bb.add_sequence(&toks[..fit_end]);
    let table = bb.build(vocab, 16);
    let cal_rounds = rounds(&table, &toks, fit_end, cal_end);
    let test = rounds(&table, &toks, cal_end, toks.len());
    let cal = calibrate(&cal_rounds);
    Fixture {
        corpus_digest,
        cal,
        test,
        n_cal: cal_rounds.len(),
    }
}

/// Adapter: the k-marginal of the outcome distribution as a predictive
/// interval for the realized accepted count (point = E[k]).
struct KForecast<'a> {
    cal: &'a Calibrated,
    rounds: &'a [Round],
    i: usize,
}

impl UqPrimitiveUnderTest for KForecast<'_> {
    fn name(&self) -> &str {
        "outcome_cache_k_marginal"
    }
    fn predict_next(&mut self) -> PredictiveOutput {
        let Some(r) = self.rounds.get(self.i) else {
            return PredictiveOutput::empty();
        };
        let pk = pk_of(self.cal, r, true);
        let (mut lo, mut hi, mut cdf) = (None, None, 0.0f32);
        for (k, &p) in pk.iter().enumerate() {
            cdf += p;
            if lo.is_none() && cdf >= ALPHA / 2.0 {
                lo = Some(k as f32);
            }
            if hi.is_none() && cdf >= 1.0 - ALPHA / 2.0 {
                hi = Some(k as f32);
            }
        }
        let hi = hi.unwrap_or(r.n as f32);
        PredictiveOutput::from_interval(PredictiveInterval::new(
            lo.unwrap_or(0.0),
            expected_accepted(&pk),
            hi,
            ALPHA,
        ))
    }
    fn observe(&mut self, _y: f32) {
        self.i += 1;
    }
}

#[test]
fn issue976_gate_battery() {
    let f = fixture();
    println!("corpus blake3 = {}", f.corpus_digest);
    assert_eq!(f.corpus_digest, CORPUS_BLAKE3, "frozen trace input drifted");
    let cal = &f.cal;
    let n_test = f.test.len();
    let mean_k = f.test.iter().map(|r| r.k).sum::<usize>() as f32 / n_test as f32;
    println!(
        "rounds cal={} test={} mean_k(test)={mean_k:.3} ladder_r={:.3} freeze converged={} kl={:.4} λ={:.5} pod_blake3={}",
        f.n_cal,
        n_test,
        cal.ladder_r,
        cal.pod.converged,
        cal.kl,
        cal.lambda,
        blake3::Hash::from(cal.pod.blake3_digest()).to_hex()
    );
    assert!(n_test > 1000 && f.n_cal > 1000, "trace too thin");

    // G-freeze: wire round-trip + digest refusal.
    let back = BonusHistogramPod::from_commit_bytes(&cal.pod.commit_bytes()).unwrap();
    assert!(back.verify_against(&cal.pod.blake3_digest()));

    // G-Prop2b: paired hit rate, strict ≥ vs BOTH fixed allocations at equal B.
    println!(
        "   B | ctx (stale/ceil) | pod-raw pod-recal | conformal(size) | static ladder | pod-stale pod-ceil | W/L ctx vs static"
    );
    for b in BUDGETS {
        let nb = evaluate(cal, &f.test, b, false);
        println!(
            "  {b:>2} | ctx WITHOUT backoff: {:>6.2}% (ceil {:.2}) — ablation",
            100.0 * nb.ctx as f32 / n_test as f32,
            100.0 * nb.ctx_ceiling as f32 / n_test as f32
        );
        let r = evaluate(cal, &f.test, b, true);
        let pct = |x: usize| 100.0 * x as f32 / n_test as f32;
        println!(
            "  {b:>2} | {:>6.2}% ({:.2}/{:.2}) | {:>6.2}% {:>6.2}% | {:>6.2}% ({:.2}) | {:>6.2}% {:>6.2}% | {:>6.2}% {:>6.2}% | {}/{}",
            pct(r.ctx),
            pct(r.ctx_stale),
            pct(r.ctx_ceiling),
            pct(r.per_round_raw),
            pct(r.per_round),
            pct(r.conformal),
            r.conformal_size as f32 / n_test as f32,
            pct(r.stat),
            pct(r.ladder),
            pct(r.stale),
            pct(r.ceiling),
            r.wins_vs_static,
            r.losses_vs_static,
        );
        assert!(r.ctx >= r.stat, "Prop 2b vs static_top_b FAILED at B={b}");
        assert!(
            r.ctx >= r.ladder,
            "Prop 2b vs geometric_ladder FAILED at B={b}"
        );
        // M6 staleness ladder: stale ≤ predicted ≤ true ceiling.
        assert!(r.ctx_stale <= r.ctx && r.ctx <= r.ctx_ceiling);
    }

    // G-coverage: split-conformal at nominal rate (finite-sample slack 0.02 —
    // consecutive text rounds are not exactly exchangeable; disclosed).
    let covered = f
        .test
        .iter()
        .filter(|r| {
            let mut s = [0.0f32; GRID];
            let rows = r.rows.with_backoff(&cal.pod);
            outcome_scores_rows(&pk_of(cal, r, true), &rows, &mut s);
            s[cell(r.k, rows.slot_of(r.k, r.bonus))] >= cal.lambda
        })
        .count() as f32
        / n_test as f32;
    println!(
        "conformal coverage = {covered:.4} (nominal {:.2})",
        1.0 - ALPHA
    );
    assert!(covered >= 1.0 - ALPHA - 0.02, "coverage below nominal");

    // G-floor: E[accepted] forecast vs the conformal-naive floor (reported;
    // the verdict is the bench row, not an assertion — Report the Floor).
    let ks: Vec<f32> = f.test.iter().map(|r| r.k as f32).collect();
    let mut fc = KForecast {
        cal,
        rounds: &f.test,
        i: 0,
    };
    let rep = run_floor_comparison(&mut fc, &ks, ALPHA, 50, "austen_tail_k");
    let mae_fc: f32 = f
        .test
        .iter()
        .skip(50)
        .map(|r| (expected_accepted(&pk_of(cal, r, true)) - r.k as f32).abs())
        .sum::<f32>()
        / (n_test - 50) as f32;
    let mae_naive: f32 = f
        .test
        .windows(2)
        .skip(49)
        .map(|w| (w[0].k as f32 - w[1].k as f32).abs())
        .sum::<f32>()
        / (n_test - 50) as f32;
    println!(
        "floor: prim crps={:.4} cov={:.3} wink={:.3} | floor crps={:.4} cov={:.3} wink={:.3} | crps_ratio={:.3} wink_ratio={:.3} verdict={:?} | MAE E[k] {mae_fc:.4} vs naive {mae_naive:.4}",
        rep.primitive.mean_crps_interval,
        rep.primitive.coverage,
        rep.primitive.mean_winkler,
        rep.floor.mean_crps_interval,
        rep.floor.coverage,
        rep.floor.mean_winkler,
        rep.crps_ratio,
        rep.winkler_ratio,
        rep.overall
    );
    assert!(rep.crps_ratio.is_finite() && rep.n_scored > 1000);

    // G-determinism (Issue 940): a second full run is byte-identical.
    let g = fixture();
    assert_eq!(g.cal.pod.blake3_digest(), cal.pod.blake3_digest());
    assert_eq!(g.cal.lambda.to_bits(), cal.lambda.to_bits());
    for b in BUDGETS {
        assert_eq!(
            evaluate(&g.cal, &g.test, b, true),
            evaluate(cal, &f.test, b, true)
        );
    }
}
