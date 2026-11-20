//! Issue 1229 — G1 gates for the Lepski agreement cascade (Research 681).
//!
//! Operationalisation pinned in the issue BEFORE the first run (4a6f85780):
//! structure (≥ 80% within one grid step of the oracle k* AND Spearman ≥ 0.5),
//! property ((A+2)·τ̂_{k*} on ≥ 99% of queries, σ̂ estimated), discrete
//! (σ̂ = √(D/2) within 5%, copied MAD reads 0, boundary tracking, ≥ 95% Bayes
//! agreement). The falsifiers kill; they are never re-tuned here.
//!
//! Gated by `required-features = ["estimators"]` (never `#![cfg]` — a
//! cfg-empty binary reports a green zero).

use katgpt_core::lepski_cascade::{
    A_DEFAULT, A_THEORY, geometric_grid_into, lepski_cascade, prefix_means_into,
    sigma_hat_consecutive, sigma_hat_disagreement, sigma_hat_from_abs_diffs, tau_hat_into,
};

const N: usize = 2000;
const SIGMA: f64 = 0.1;
const QUERIES: usize = 200;
/// Cell 1 (A = 4.5, pinned 4a6f85780) — the recorded negative.
const SEEDS_CELL1: [u64; 5] = [
    0x1229_0001,
    0x1229_0002,
    0x1229_0003,
    0x1229_0004,
    0x1229_0005,
];
/// Cell 2 (A = 1, fresh seeds, pinned f19d541da before its run).
const SEEDS_CELL2: [u64; 5] = [
    0x1229_0101,
    0x1229_0102,
    0x1229_0103,
    0x1229_0104,
    0x1229_0105,
];

// ---- deterministic fixture RNG (splitmix64 + Box–Muller) ----------------

struct SplitMix(u64);

impl SplitMix {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(f64::MIN_POSITIVE);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * core::f64::consts::PI * u2).cos()
    }
}

// ---- fixtures -------------------------------------------------------------

fn doppler(x: f64) -> f64 {
    (x * (1.0 - x)).sqrt() * (2.1 * core::f64::consts::PI / (x + 0.05)).sin()
}

fn piecewise(x: f64) -> f64 {
    match x {
        x if x < 0.25 => 0.0,
        x if x < 0.5 => 1.0,
        x if x < 0.75 => -0.5,
        _ => 0.5,
    }
}

/// Sample sorted by covariate: (xs, noisy ys).
fn sample(seed: u64, n: usize, f: fn(f64) -> f64, sigma: f64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = SplitMix(seed);
    let mut xs: Vec<f64> = (0..n).map(|_| rng.uniform()).collect();
    xs.sort_by(f64::total_cmp);
    let ys = xs.iter().map(|&x| f(x) + sigma * rng.normal()).collect();
    (xs, ys)
}

/// Indices of the sample ranked by |X − q| (1-D two-pointer merge outward),
/// first `k_max` only. Ties break toward the left neighbour (fixed rule).
fn ranked_indices(xs: &[f64], q: f64, k_max: usize, out: &mut Vec<usize>) {
    out.clear();
    let split = xs.partition_point(|&x| x < q);
    let (mut lo, mut hi) = (split, split); // lo is exclusive-left cursor
    while out.len() < k_max && (lo > 0 || hi < xs.len()) {
        let take_left = match (lo > 0, hi < xs.len()) {
            (true, true) => (q - xs[lo - 1]) <= (xs[hi] - q),
            (true, false) => true,
            _ => false,
        };
        if take_left {
            lo -= 1;
            out.push(lo);
        } else {
            out.push(hi);
            hi += 1;
        }
    }
}

fn grid() -> Vec<usize> {
    let mut g = [0usize; 16];
    let n = geometric_grid_into(4, 512, &mut g);
    g[..n].to_vec()
}

fn spearman(a: &[f64], b: &[f64]) -> f64 {
    fn ranks(v: &[f64]) -> Vec<f64> {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&i, &j| v[i].total_cmp(&v[j]));
        let mut r = vec![0.0; v.len()];
        let mut i = 0;
        while i < idx.len() {
            let mut j = i;
            while j + 1 < idx.len() && v[idx[j + 1]] == v[idx[i]] {
                j += 1;
            }
            let avg = (i + j) as f64 / 2.0;
            for &t in &idx[i..=j] {
                r[t] = avg;
            }
            i = j + 1;
        }
        r
    }
    let (ra, rb) = (ranks(a), ranks(b));
    let n = ra.len() as f64;
    let (ma, mb) = (ra.iter().sum::<f64>() / n, rb.iter().sum::<f64>() / n);
    let (mut c, mut va, mut vb) = (0.0, 0.0, 0.0);
    for (x, y) in ra.iter().zip(&rb) {
        c += (x - ma) * (y - mb);
        va += (x - ma) * (x - ma);
        vb += (y - mb) * (y - mb);
    }
    if va == 0.0 || vb == 0.0 {
        0.0
    } else {
        c / (va * vb).sqrt()
    }
}

struct Continuous {
    within_one_step: f64,
    spearman: f64,
    property_rate: f64,
    mean_log2_khat: f64,
    upper_binding: f64,
}

fn run_continuous(f: fn(f64) -> f64, a: f64, seeds: &[u64]) -> Continuous {
    let g = grid();
    let jn = g.len();
    let k_max = *g.last().unwrap();
    let (mut within, mut prop_ok, mut binding, mut total) = (0usize, 0usize, 0usize, 0usize);
    let (mut khat_all, mut kstar_all) = (Vec::new(), Vec::new());
    let mut sum_log2 = 0.0;
    let mut ranked = Vec::with_capacity(k_max);
    let mut ys_ranked = vec![0.0; k_max];
    let mut f0_ranked = vec![0.0; k_max];
    let (mut est, mut bias_means, mut tau_hat, mut tau_true) =
        (vec![0.0; jn], vec![0.0; jn], vec![0.0; jn], vec![0.0; jn]);
    let mut scratch = vec![0.0; N];
    for &seed in seeds {
        let (xs, ys) = sample(seed, N, f, SIGMA);
        let sigma_hat = sigma_hat_consecutive(&ys, &mut scratch).unwrap();
        tau_hat_into(sigma_hat, N, &g, &mut tau_hat);
        tau_hat_into(SIGMA, N, &g, &mut tau_true);
        for qi in 0..QUERIES {
            let q = qi as f64 / (QUERIES - 1) as f64;
            ranked_indices(&xs, q, k_max, &mut ranked);
            for (r, &i) in ranked.iter().enumerate() {
                ys_ranked[r] = ys[i];
                f0_ranked[r] = f(xs[i]);
            }
            assert_eq!(prefix_means_into(&ys_ranked, &g, &mut est), jn);
            assert_eq!(prefix_means_into(&f0_ranked, &g, &mut bias_means), jn);
            let truth = f(q);
            // Oracle k*: initial segment of b_ℓ ≤ τ_ℓ (true σ); k₀ if none.
            let mut kstar_j = 0;
            for j in 0..jn {
                if (bias_means[j] - truth).abs() <= tau_true[j] {
                    kstar_j = j;
                } else {
                    break;
                }
            }
            let sel = lepski_cascade(&est, &tau_hat, a).unwrap();
            let khat_j = sel.index;
            total += 1;
            if khat_j.abs_diff(kstar_j) <= 1 {
                within += 1;
            }
            if sel.upper_end_binding() {
                binding += 1;
            }
            if (est[khat_j] - truth).abs() <= (a + 2.0) * tau_hat[kstar_j] {
                prop_ok += 1;
            }
            khat_all.push(khat_j as f64);
            kstar_all.push(kstar_j as f64);
            sum_log2 += (g[khat_j] as f64).log2();
        }
    }
    Continuous {
        within_one_step: within as f64 / total as f64,
        spearman: spearman(&khat_all, &kstar_all),
        property_rate: prop_ok as f64 / total as f64,
        mean_log2_khat: sum_log2 / total as f64,
        upper_binding: binding as f64 / total as f64,
    }
}

fn report(name: &str, a: f64, c: &Continuous) {
    println!(
        "[i1229 G1] {name:<10} A={a:<4} within1step={:.3} spearman={:.3} property={:.4} \
         mean_log2_khat={:.2} upper_binding={:.3}",
        c.within_one_step, c.spearman, c.property_rate, c.mean_log2_khat, c.upper_binding
    );
}

fn structure_pass(c: &Continuous) -> bool {
    c.within_one_step >= 0.80 && c.spearman >= 0.5
}

/// Cell 1 — A = 4.5. Recorded result: property PASS both fixtures,
/// structure PASS piecewise, structure FAIL Doppler. Pinned so a change in
/// either direction is seen, never silently absorbed.
#[test]
fn g1_cell1_a45_recorded() {
    let d = run_continuous(doppler, A_THEORY, &SEEDS_CELL1);
    report("doppler", A_THEORY, &d);
    let p = run_continuous(piecewise, A_THEORY, &SEEDS_CELL1);
    report("piecewise", A_THEORY, &p);
    assert!(
        d.property_rate >= 0.99 && p.property_rate >= 0.99,
        "cell-1 property moved"
    );
    assert!(structure_pass(&p), "cell-1 piecewise structure moved");
    assert!(
        !structure_pass(&d),
        "cell-1 Doppler structure FAIL is the recorded verdict"
    );
}

/// Cell 2 — A = 1 on fresh seeds: the deciding gate (Issue 1229).
fn gate_cell2(name: &str, f: fn(f64) -> f64) {
    let c = run_continuous(f, A_DEFAULT, &SEEDS_CELL2);
    report(name, A_DEFAULT, &c);
    assert!(
        c.property_rate >= 0.99,
        "{name}: property gate FAIL — (A+2)·τ̂ held on {:.4} < 0.99 (indicts σ̂/statistic)",
        c.property_rate
    );
    assert!(
        structure_pass(&c),
        "{name}: structure gate FAIL — within1step {:.3} (bar 0.80), spearman {:.3} (bar 0.5)",
        c.within_one_step,
        c.spearman
    );
}

#[test]
fn g1_cell2_doppler() {
    gate_cell2("doppler", doppler);
}

#[test]
fn g1_cell2_piecewise_constant() {
    gate_cell2("piecewise", piecewise);
}

#[test]
fn g1_sigma_hat_recovers_planted_sigma() {
    // Sanity on the continuous σ̂ itself (not a pre-registered gate row —
    // its failure would explain a property-gate failure).
    let mut scratch = vec![0.0; N];
    for &seed in SEEDS_CELL1.iter().chain(&SEEDS_CELL2) {
        let (_, ys) = sample(seed, N, doppler, SIGMA);
        let s = sigma_hat_consecutive(&ys, &mut scratch).unwrap();
        println!("[i1229 G1] sigma_hat doppler seed={seed:#x} -> {s:.4} (planted {SIGMA})");
        assert!(
            (s - SIGMA).abs() / SIGMA < 0.15,
            "σ̂ {s} far from planted {SIGMA}"
        );
    }
}

struct Discrete {
    near: f64,
    far: f64,
    bayes_agreement: f64,
}

fn run_discrete(a: f64, seeds: &[u64]) -> Discrete {
    const ND: usize = 4000;
    let p_of = |x: f64| if x < 0.5 { 0.2 } else { 0.8 };
    let planted = (0.2_f64 * 0.8).sqrt();
    let g = grid();
    let jn = g.len();
    let k_max = *g.last().unwrap();
    let (mut near_sum, mut near_n, mut far_sum, mut far_n) = (0.0, 0usize, 0.0, 0usize);
    let (mut agree, mut total) = (0usize, 0usize);
    let mut ranked = Vec::with_capacity(k_max);
    let mut votes = vec![0.0; k_max];
    let (mut est, mut tau) = (vec![0.0; jn], vec![0.0; jn]);
    let mut diffs = vec![0.0; ND];
    for &seed in seeds {
        let mut rng = SplitMix(seed ^ 0xD15C);
        let mut xs: Vec<f64> = (0..ND).map(|_| rng.uniform()).collect();
        xs.sort_by(f64::total_cmp);
        let ys: Vec<f64> = xs
            .iter()
            .map(|&x| if rng.uniform() < p_of(x) { 1.0 } else { 0.0 })
            .collect();
        let disagree = ys.windows(2).filter(|w| w[0] != w[1]).count();
        let s = sigma_hat_disagreement(disagree, ND - 1).unwrap();
        for (d, w) in diffs.iter_mut().zip(ys.windows(2)) {
            *d = (w[1] - w[0]).abs();
        }
        let mad_copy = sigma_hat_from_abs_diffs(&mut diffs[..ND - 1]).unwrap();
        println!(
            "[i1229 G1 discrete] seed={seed:#x} D={:.4} sigma_hat={s:.4} (planted {planted:.4}) \
             mad_copy={mad_copy}",
            disagree as f64 / (ND - 1) as f64
        );
        assert!(
            (s - planted).abs() / planted <= 0.05,
            "discrete σ̂ {s} vs {planted}"
        );
        assert_eq!(
            mad_copy, 0.0,
            "the copied MAD form must read the named degeneracy"
        );
        tau_hat_into(s, ND, &g, &mut tau);
        for qi in 0..QUERIES {
            let q = qi as f64 / (QUERIES - 1) as f64;
            ranked_indices(&xs, q, k_max, &mut ranked);
            for (r, &i) in ranked.iter().enumerate() {
                votes[r] = ys[i];
            }
            prefix_means_into(&votes, &g, &mut est);
            let sel = lepski_cascade(&est, &tau, a).unwrap();
            let k = g[sel.index] as f64;
            let dist = (q - 0.5).abs();
            if dist < 0.05 {
                near_sum += k;
                near_n += 1;
            } else if dist > 0.2 {
                far_sum += k;
                far_n += 1;
            }
            let vote = est[sel.index] >= 0.5;
            let bayes = p_of(q) >= 0.5;
            total += 1;
            if vote == bayes {
                agree += 1;
            }
        }
    }
    let (near, far) = (near_sum / near_n as f64, far_sum / far_n as f64);
    let acc = agree as f64 / total as f64;
    println!(
        "[i1229 G1 discrete] A={a} mean_khat near={near:.1} far={far:.1} bayes_agreement={acc:.4}"
    );
    Discrete {
        near,
        far,
        bayes_agreement: acc,
    }
}

/// Cell 1 — A = 4.5. Recorded: σ̂ + degeneracy + Bayes PASS, tracking FAIL
/// (the cascade never rejects on a [0,1] vote share at this n).
#[test]
fn g1_cell1_a45_discrete_recorded() {
    let d = run_discrete(A_THEORY, &SEEDS_CELL1);
    assert!(d.bayes_agreement >= 0.95, "cell-1 Bayes agreement moved");
    assert!(
        d.near >= d.far,
        "cell-1 discrete tracking FAIL is the recorded verdict"
    );
}

/// Cell 2 — A = 1 on fresh seeds: the deciding discrete gate.
#[test]
fn g1_cell2_discrete() {
    let d = run_discrete(A_DEFAULT, &SEEDS_CELL2);
    assert!(
        d.near < d.far,
        "discrete tracking FAIL: near {} ≥ far {}",
        d.near,
        d.far
    );
    assert!(
        d.bayes_agreement >= 0.95,
        "discrete Bayes agreement {} < 0.95",
        d.bayes_agreement
    );
}
