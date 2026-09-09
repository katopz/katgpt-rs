//! Issue 707 Phase 3 — TPR binding-algebra GOAT gate (Research 527,
//! arXiv:2608.29530 McCoy/Soulos/Linzen/Smolensky 2026).
//!
//! # Modelless construction — read this first
//!
//! Every corpus here is a **planted TPR**: states are generated as
//! `e = W·(Σ_p r_p ⊗ f_v) + b` from a fixed random `W` / filler table. There
//! is no trained model, no gradient step in the primitive, and no learned
//! checkpoint — the fit under test is closed-form ridge-ALS. The one gradient
//! loop in this file is the **G2c baseline arm**, a full-batch GD fit of the
//! SAME objective, present only so "ALS is faster than gradient descent" is a
//! measured claim rather than an assumption. It is bench-local and never
//! ships.
//!
//! # Gates
//!
//! - **G1 correctness:** planted recovery (residual energy fraction below
//!   `G1_ENERGY_BAR`), double-fit **bit-identical** artifacts, monotone
//!   certificate clean, holdout unbind cosine ≥ `G1_COS_BAR`, surgery
//!   additive to f32 noise.
//! - **G2 perf:** surgery p99 < 1 µs at D ∈ {64, 256, 768}; projection ≤ 2×
//!   its own irreducible two-GEMV floor; ALS wall-clock ≤ the GD baseline's
//!   time to reach the same objective.
//! - **G3 isolation:** the feature is opt-in, and `RIIR_TPR=0` disables every
//!   op — verified by re-executing THIS binary in a child process with the
//!   variable set, because the switch is `OnceLock`-cached and an in-process
//!   check after any other gate would test the cache, not the switch.
//! - **G4 alloc:** zero steady-state allocations across bind / unbind /
//!   surgery / project.
//! - **G8 systematicity:** withheld-`(role, filler)`-pair top-1 beats the
//!   atomic-dictionary null by ≥ `G8_MARGIN_PP` points — and the null's
//!   IN-DISTRIBUTION coverage is checked first, because a null that fails ID
//!   too is vacuous and its OOD zero certifies nothing (the measured
//!   healer-corpus failure, riir-clippy `.benchmarks/062_withheld_pair_ood.md`).

use katgpt_core::tpr::{
    AlsConfig, AlsInput, AtomicNull, TprArtifact, TprBindings, TprScheme, TprScratch, als_fit,
    bind_into, encode_into, project_into, state_from_core_into, surgery_delta_into, unbind_into,
    validate_bindings, withheld_pair_top1_report,
};
use std::time::Instant;

const G1_ENERGY_BAR: f32 = 1e-6;
const G1_COS_BAR: f32 = 0.999;
const G2_SURGERY_P99_NS: u128 = 1_000;
const G2_PROJECT_FLOOR_MULT: f64 = 2.0;
const G4_CALLS: usize = 20_000;
const G8_MARGIN_PP: f32 = 20.0;
const CHILD_ARG: &str = "--killswitch-child";

// ─── Fixture ───────────────────────────────────────────────────────────────

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn sym(&mut self) -> f32 {
        let u = (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32;
        u.mul_add(2.0, -1.0)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n.max(1) as u64) as usize
    }
}

struct Corpus {
    dim: usize,
    d: usize,
    m: usize,
    n_fillers: usize,
    states: Vec<f32>,
    bindings: Vec<TprBindings>,
}

fn planted(dim: usize, d: usize, m: usize, n_fillers: usize, n: usize, seed: u64) -> Corpus {
    let k = m * d;
    let mut rng = Rng::new(seed);
    let w: Vec<f32> = (0..dim * k).map(|_| rng.sym()).collect();
    let bias: Vec<f32> = (0..dim).map(|_| 0.25 * rng.sym()).collect();
    let mut fillers = vec![0.0f32; n_fillers * d];
    for v in 0..n_fillers {
        let row = &mut fillers[v * d..(v + 1) * d];
        for x in row.iter_mut() {
            *x = rng.sym();
        }
        let nrm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in row.iter_mut() {
            *x /= nrm.max(1e-6);
        }
    }
    let mut states = vec![0.0f32; n * dim];
    let mut bindings = Vec::with_capacity(n);
    let mut core = vec![0.0f32; k];
    for s in 0..n {
        core.fill(0.0);
        let mut b = TprBindings::default();
        for p in 0..m {
            let v = rng.below(n_fillers);
            b.roles.push(p as u16);
            b.fillers.push(v as u16);
            for j in 0..d {
                core[p * d + j] += fillers[v * d + j];
            }
        }
        for i in 0..dim {
            let mut acc = bias[i];
            for j in 0..k {
                acc = w[i * k + j].mul_add(core[j], acc);
            }
            states[s * dim + i] = acc;
        }
        bindings.push(b);
    }
    Corpus {
        dim,
        d,
        m,
        n_fillers,
        states,
        bindings,
    }
}

fn input(c: &Corpus) -> AlsInput<'_> {
    AlsInput {
        dim: c.dim,
        n_fillers: c.n_fillers,
        states: &c.states,
        bindings: &c.bindings,
    }
}

fn fit(c: &Corpus) -> TprArtifact {
    let cfg = AlsConfig::new(c.d, TprScheme::Orthogonal { arity: c.m });
    als_fit(input(c), &cfg).expect("fit").0
}

fn percentile(v: &mut [u128], q: f64) -> u128 {
    v.sort_unstable();
    let idx = (((v.len() - 1) as f64) * q).round() as usize;
    v[idx.min(v.len() - 1)]
}

// ─── G1 ────────────────────────────────────────────────────────────────────

fn run_g1() -> bool {
    println!("\n── G1 correctness ──");
    let c = planted(64, 8, 4, 12, 512, 0x6070_0001);
    let cfg = AlsConfig::new(c.d, TprScheme::Orthogonal { arity: c.m });
    let (a, rep) = als_fit(input(&c), &cfg).expect("fit");
    let (b, _) = als_fit(input(&c), &cfg).expect("fit");

    let identical = a.to_bytes() == b.to_bytes();
    println!(
        "  planted recovery: energy fraction {:e} (bar {:e}), ssr {:e}, {} sweeps",
        rep.residual_energy_fraction, G1_ENERGY_BAR, rep.final_ssr, rep.sweeps
    );
    println!(
        "  determinism: double fit bit-identical = {identical} (commitment {})",
        hex8(&a.commitment)
    );
    let best = rep
        .ssr_per_sweep
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let is_best = a.fit_objective <= best * (1.0 + 1e-9) + 1e-12;
    println!(
        "  monotone: {} rejected proposal(s); returned objective {:e} == trajectory min {:e} → {is_best}",
        rep.monotone_violations, a.fit_objective, best
    );

    let mut scratch = TprScratch::new(&a);
    let (hs, hb) = holdout(&a, &c, 128, 0x6070_0002);
    let vr = validate_bindings(&a, &hs, &hb, &mut scratch).expect("validate");
    let scale = hs.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1.0);
    println!(
        "  holdout: unbind cos min {:.6} mean {:.6}; surgery max |Δ| {:e} (scale {scale:.2})",
        vr.unbind_cos_min, vr.unbind_cos_mean, vr.surgery_max_abs_err
    );

    let pass = rep.residual_energy_fraction < G1_ENERGY_BAR
        && identical
        && is_best
        && vr.unbind_cos_min > G1_COS_BAR
        && vr.surgery_max_abs_err <= 1e-4 * scale;
    println!("  G1 {}", verdict(pass));
    pass
}

fn hex8(b: &[u8; 32]) -> String {
    b[..4].iter().map(|x| format!("{x:02x}")).collect()
}

fn holdout(art: &TprArtifact, c: &Corpus, n: usize, seed: u64) -> (Vec<f32>, Vec<TprBindings>) {
    let mut rng = Rng::new(seed);
    let mut scratch = TprScratch::new(art);
    let mut states = vec![0.0f32; n * art.dim];
    let mut bindings = Vec::with_capacity(n);
    let mut out = vec![0.0f32; art.dim];
    for s in 0..n {
        let mut b = TprBindings::default();
        for p in 0..c.m {
            b.roles.push(p as u16);
            b.fillers.push(rng.below(c.n_fillers) as u16);
        }
        encode_into(art, &b, &mut scratch, &mut out).expect("encode");
        states[s * art.dim..(s + 1) * art.dim].copy_from_slice(&out);
        bindings.push(b);
    }
    (states, bindings)
}

// ─── G2 ────────────────────────────────────────────────────────────────────

fn run_g2() -> bool {
    println!("\n── G2 perf ──");
    let mut ok = true;

    for &dim in &[64usize, 256, 768] {
        let c = planted(dim, 8, 4, 12, 256, 0x6070_0100 + dim as u64);
        let art = fit(&c);
        let mut scratch = TprScratch::new(&art);
        let mut state = vec![0.0f32; dim];
        encode_into(&art, &c.bindings[0], &mut scratch, &mut state).expect("encode");
        let f_old = art.fillers[..art.d].to_vec();
        let f_new = art.fillers[art.d..2 * art.d].to_vec();

        for _ in 0..2_000 {
            surgery_delta_into(&art, &mut state, 0, &f_old, &f_new, &mut scratch).expect("surgery");
        }
        let mut samples = Vec::with_capacity(20_000);
        for _ in 0..20_000 {
            let t = Instant::now();
            surgery_delta_into(&art, &mut state, 0, &f_old, &f_new, &mut scratch).expect("surgery");
            samples.push(t.elapsed().as_nanos());
        }
        let p50 = percentile(&mut samples, 0.50);
        let p99 = percentile(&mut samples, 0.99);
        let hit = p99 < G2_SURGERY_P99_NS;
        ok &= hit;
        println!(
            "  surgery D={dim:<4} p50 {p50:>4} ns  p99 {p99:>4} ns  (bar {G2_SURGERY_P99_NS} ns) {}",
            verdict(hit)
        );
    }

    // Projection vs its own irreducible floor: the two D×K GEMVs it must do
    // regardless (Wᵀ(e−b) and W·x). Anything above 2× that is the Cholesky
    // solve costing more than the data movement it rides on.
    let c = planted(256, 8, 4, 12, 256, 0x6070_0200);
    let art = fit(&c);
    let mut scratch = TprScratch::new(&art);
    let mut state = vec![0.0f32; art.dim];
    encode_into(&art, &c.bindings[0], &mut scratch, &mut state).expect("encode");
    let mut out = vec![0.0f32; art.dim];
    let core: Vec<f32> = (0..art.core_len()).map(|i| (i as f32) * 0.01).collect();

    for _ in 0..2_000 {
        project_into(&art, &state, &mut scratch, &mut out).expect("project");
        state_from_core_into(&art, &core, &mut out).expect("decode");
    }
    let iters = 20_000;
    let t = Instant::now();
    for _ in 0..iters {
        project_into(&art, &state, &mut scratch, &mut out).expect("project");
    }
    let proj_ns = t.elapsed().as_nanos() as f64 / iters as f64;
    let t = Instant::now();
    for _ in 0..iters {
        state_from_core_into(&art, &core, &mut out).expect("decode");
        state_from_core_into(&art, &core, &mut out).expect("decode");
    }
    let floor_ns = t.elapsed().as_nanos() as f64 / iters as f64;
    let ratio = proj_ns / floor_ns.max(1e-9);
    let hit = ratio <= G2_PROJECT_FLOOR_MULT;
    ok &= hit;
    println!(
        "  project D=256 {proj_ns:.0} ns vs two-GEMV floor {floor_ns:.0} ns → {ratio:.2}× (bar {G2_PROJECT_FLOOR_MULT:.1}×) {}",
        verdict(hit)
    );

    // ALS vs a full-batch GD fit of the same objective.
    let c = planted(64, 8, 4, 12, 512, 0x6070_0300);
    let cfg = AlsConfig::new(c.d, TprScheme::Orthogonal { arity: c.m });
    let t = Instant::now();
    let (_art, rep) = als_fit(input(&c), &cfg).expect("fit");
    let als_ms = t.elapsed().as_secs_f64() * 1e3;
    let (gd_ms, gd_ssr, gd_reached) = gd_baseline(&c, rep.final_ssr);
    let hit = als_ms <= gd_ms;
    ok &= hit;
    println!(
        "  fit: ALS {als_ms:.1} ms → ssr {:e}   |   GD {gd_ms:.1} ms → ssr {gd_ssr:e} (reached ALS objective: {gd_reached}) {}",
        rep.final_ssr,
        verdict(hit)
    );
    println!("  G2 {}", verdict(ok));
    ok
}

/// Full-batch gradient descent on the same ridge objective — the LOSER arm of
/// G2c, here only to price ALS against the thing it replaces. Steps `W`, `b`
/// and the filler table jointly with a fixed learning rate and a backtracking
/// halving on divergence; stops when it reaches `target_ssr` or burns its
/// iteration budget.
fn gd_baseline(c: &Corpus, target_ssr: f64) -> (f64, f64, bool) {
    let (dim, d, m) = (c.dim, c.d, c.m);
    let k = m * d;
    let n = c.bindings.len();
    let mut rng = Rng::new(0x6070_0400);
    let mut w: Vec<f32> = (0..dim * k).map(|_| 0.1 * rng.sym()).collect();
    let mut bias = vec![0.0f32; dim];
    let mut fillers: Vec<f32> = (0..c.n_fillers * d).map(|_| rng.sym()).collect();
    let mut cores = vec![0.0f32; n * k];
    let mut resid = vec![0.0f32; dim];
    let mut lr = 1e-3f32;
    let max_iters = 20_000usize;

    let t = Instant::now();
    let mut ssr = f64::INFINITY;
    let mut reached = false;
    for _ in 0..max_iters {
        cores.fill(0.0);
        for (s, b) in c.bindings.iter().enumerate() {
            for (&p, &v) in b.roles.iter().zip(b.fillers.iter()) {
                let off = s * k + p as usize * d;
                for j in 0..d {
                    cores[off + j] += fillers[v as usize * d + j];
                }
            }
        }
        let mut gw = vec![0.0f32; dim * k];
        let mut gb = vec![0.0f32; dim];
        let mut gf = vec![0.0f32; c.n_fillers * d];
        let mut cur = 0.0f64;
        for (s, b) in c.bindings.iter().enumerate() {
            let core = &cores[s * k..(s + 1) * k];
            for i in 0..dim {
                let mut acc = bias[i];
                for j in 0..k {
                    acc = w[i * k + j].mul_add(core[j], acc);
                }
                resid[i] = acc - c.states[s * dim + i];
                cur += (resid[i] as f64) * (resid[i] as f64);
            }
            for i in 0..dim {
                let r = resid[i];
                gb[i] += 2.0 * r;
                for j in 0..k {
                    gw[i * k + j] += 2.0 * r * core[j];
                }
            }
            for (&p, &v) in b.roles.iter().zip(b.fillers.iter()) {
                let blk = p as usize * d;
                for j in 0..d {
                    let mut g = 0.0f32;
                    for i in 0..dim {
                        g = w[i * k + blk + j].mul_add(2.0 * resid[i], g);
                    }
                    gf[v as usize * d + j] += g;
                }
            }
        }
        if cur > ssr {
            lr *= 0.5;
        }
        ssr = cur;
        if ssr <= target_ssr {
            reached = true;
            break;
        }
        let scale = lr / n as f32;
        for i in 0..dim * k {
            w[i] -= scale * gw[i];
        }
        for i in 0..dim {
            bias[i] -= scale * gb[i];
        }
        for i in 0..c.n_fillers * d {
            fillers[i] -= scale * gf[i];
        }
    }
    (t.elapsed().as_secs_f64() * 1e3, ssr, reached)
}

// ─── G3 ────────────────────────────────────────────────────────────────────

fn run_g3() -> bool {
    println!("\n── G3 isolation ──");
    let exe = match std::env::current_exe() {
        Ok(e) => e,
        Err(e) => {
            println!("  G3 SKIP — cannot locate this binary to re-exec: {e}");
            println!("  G3 {}", verdict(false));
            return false;
        }
    };
    let out = std::process::Command::new(&exe)
        .arg(CHILD_ARG)
        .env("RIIR_TPR", "0")
        .output();
    match out {
        Err(e) => {
            println!("  G3 FAIL — child process did not start: {e}");
            false
        }
        Ok(o) => {
            let text = String::from_utf8_lossy(&o.stdout);
            let disabled = text.contains("KILLSWITCH_OK");
            print!("  child (RIIR_TPR=0): {text}");
            println!(
                "  every op refused with TprError::Disabled = {disabled}; feature is opt-in \
                 (`tpr` is absent from katgpt-core's default list)"
            );
            println!("  G3 {}", verdict(disabled && o.status.success()));
            disabled && o.status.success()
        }
    }
}

/// Child mode: assert every runtime op refuses while `RIIR_TPR=0`.
fn killswitch_child() -> ! {
    let c = planted(32, 4, 2, 6, 32, 0x6070_0500);
    // The fit is an OFFLINE path and is deliberately NOT gated — build the
    // artifact with the switch honoured only on the runtime ops.
    let art = fit(&c);
    let mut scratch = TprScratch::new(&art);
    let mut out = vec![0.0f32; art.dim];
    let mut fout = vec![0.0f32; art.d];
    let core = vec![0.0f32; art.core_len()];
    let f = art.fillers[..art.d].to_vec();
    let refused = bind_into(&art, 0, &f, &mut out).is_err()
        && unbind_into(&art, &core, 0, &mut fout).is_err()
        && surgery_delta_into(&art, &mut out, 0, &f, &f, &mut scratch).is_err()
        && project_into(&art, &out.clone(), &mut scratch, &mut out).is_err()
        && encode_into(&art, &c.bindings[0], &mut scratch, &mut out).is_err();
    if refused { println!("KILLSWITCH_OK — bind/unbind/surgery/project/encode all refused") } else { println!("KILLSWITCH_LEAK — an op ran with RIIR_TPR=0") }
    std::process::exit(if refused { 0 } else { 1 });
}

// ─── G4 ────────────────────────────────────────────────────────────────────

fn run_g4() -> bool {
    use std::alloc::{GlobalAlloc, Layout};
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingAllocator;
    static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
    unsafe impl GlobalAlloc for CountingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            unsafe { std::alloc::System.alloc(layout) }
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { std::alloc::System.dealloc(ptr, layout) }
        }
    }
    #[global_allocator]
    static A: CountingAllocator = CountingAllocator;

    println!("\n── G4 alloc ──");
    let c = planted(256, 8, 4, 12, 256, 0x6070_0600);
    let art = fit(&c);
    let mut scratch = TprScratch::new(&art);
    let mut state = vec![0.0f32; art.dim];
    let mut out = vec![0.0f32; art.dim];
    let mut fout = vec![0.0f32; art.d];
    let f_old = art.fillers[..art.d].to_vec();
    let f_new = art.fillers[art.d..2 * art.d].to_vec();
    let bindings = c.bindings[0].clone();
    encode_into(&art, &bindings, &mut scratch, &mut state).expect("encode");

    for _ in 0..100 {
        bind_into(&art, 0, &f_old, &mut out).expect("bind");
        project_into(&art, &state, &mut scratch, &mut out).expect("project");
    }

    let before = ALLOC_COUNT.load(Ordering::Relaxed);
    let mut sink = 0.0f32;
    for _ in 0..G4_CALLS {
        bind_into(&art, 0, &f_old, &mut out).expect("bind");
        surgery_delta_into(&art, &mut state, 0, &f_old, &f_new, &mut scratch).expect("surgery");
        let r = project_into(&art, &state, &mut scratch, &mut out).expect("project");
        unbind_into(&art, scratch.core(), 0, &mut fout).expect("unbind");
        sink += r + fout[0] + out[0];
    }
    let allocs = ALLOC_COUNT.load(Ordering::Relaxed) - before;
    println!("  {allocs} allocations across {G4_CALLS} op quadruples (sink {sink:e})");
    let pass = allocs == 0;
    println!("  G4 {}", verdict(pass));
    pass
}

// ─── G8 ────────────────────────────────────────────────────────────────────

fn run_g8() -> bool {
    println!("\n── G8 systematicity (withheld-pair OOD vs the atomic null) ──");
    let c = planted(64, 8, 3, 8, 900, 0x6070_0700);
    // Withhold every state whose role-0 filler is 0 AND role-1 filler is 1 —
    // the pair combination the fit never sees.
    let held: Vec<usize> = (0..c.bindings.len())
        .filter(|&s| c.bindings[s].fillers[0] == 0 && c.bindings[s].fillers[1] == 1)
        .collect();
    let train: Vec<usize> = (0..c.bindings.len()).filter(|s| !held.contains(s)).collect();
    let (tr_s, tr_b) = subset(&c, &train);
    let (te_s, te_b) = subset(&c, &held);
    println!("  corpus: {} train / {} withheld states", tr_b.len(), te_b.len());

    let cfg = AlsConfig::new(c.d, TprScheme::Orthogonal { arity: c.m });
    let tr_in = AlsInput {
        dim: c.dim,
        n_fillers: c.n_fillers,
        states: &tr_s,
        bindings: &tr_b,
    };
    let (art, _) = als_fit(tr_in, &cfg).expect("fit");
    let mut scratch = TprScratch::new(&art);

    let mut pool: Vec<TprBindings> = Vec::new();
    for t in &te_b {
        for v in 0..c.n_fillers {
            let mut cand = t.clone();
            cand.fillers[0] = v as u16;
            if !pool.iter().any(|x| x == &cand) {
                pool.push(cand);
            }
        }
    }

    let null = AtomicNull::fit(c.dim, &tr_s, &tr_b);
    let id_cov = null.coverage(&tr_b);
    // The ID arm must use a pool of the SAME SHAPE as the OOD arm (truth +
    // role-0 filler variants) over a comparable number of states. Scoring the
    // memorizer against a 700-candidate pool while TPR faces 56 would price
    // the pool, not the arm.
    let id_states: Vec<usize> = (0..te_b.len().min(tr_b.len())).collect();
    let (id_s, id_b) = subset_of(&tr_s, &tr_b, c.dim, &id_states);
    let mut id_pool: Vec<TprBindings> = Vec::new();
    for t in &id_b {
        for v in 0..c.n_fillers {
            let mut cand = t.clone();
            cand.fillers[0] = v as u16;
            if !id_pool.iter().any(|x| x == &cand) {
                id_pool.push(cand);
            }
        }
    }
    let null_id = null.top1(&id_s, &id_b, &id_pool);
    let null_ood = null.top1(&te_s, &te_b, &pool) * 100.0;
    let ood = withheld_pair_top1_report(&art, &te_s, &te_b, &pool, &mut scratch).expect("ood");
    let tpr_ood = ood.top1 * 100.0;
    let chance = 100.0 / pool.len() as f32;

    println!("  null: ID coverage {:.1}% / ID top-1 {:.1}%", id_cov * 100.0, null_id * 100.0);
    let informative = id_cov > 0.99 && null_id > 0.5;
    if informative { println!("  null is INFORMATIVE in-distribution — its OOD zero is a real failure") } else { println!(
            "  null is VACUOUS (it cannot fit its own training set) — the OOD comparison \
             certifies NOTHING; see riir-clippy .benchmarks/062"
        ); }
    println!(
        "  OOD top-1: TPR {tpr_ood:.1}%  vs  null {null_ood:.1}%  (chance {chance:.1}%, pool {})",
        pool.len()
    );
    // Issue 711: that G8's corpus is "planted and healthy" was asserted in
    // prose and is now MEASURED — the covariate the OOD number's
    // interpretation depends on, plus the pool ceiling it must be read
    // against. A degenerate universe would make the number above
    // uninterpretable rather than wrong, so the gate refuses instead of
    // reporting a margin it cannot read.
    println!(
        "  composition: roles/filler max={} mean={:.3} over {} filler(s), {} testable; \
         pool coverage {:.1}% -> {}",
        ood.spread.max,
        ood.spread.mean,
        ood.spread.fillers,
        ood.spread.multi_role_fillers,
        ood.coverage * 100.0,
        match ood.verdict() {
            Some(v) => format!("verdict {:.1}%", v * 100.0),
            None => "WITHHELD — role = f(filler), no unseen pair to generalize to".to_string(),
        }
    );
    let readable = ood.verdict().is_some() && ood.coverage > 0.99;
    if !readable {
        println!(
            "  G8 corpus is NOT interpretable (readable = false) — refusing rather \
             than scoring an unreadable OOD number"
        );
    }
    let margin = tpr_ood - null_ood;
    let pass = informative && readable && margin >= G8_MARGIN_PP && tpr_ood > chance;
    println!("  margin {margin:.1} pp (bar {G8_MARGIN_PP:.1} pp) → G8 {}", verdict(pass));
    pass
}

fn subset_of(
    states: &[f32],
    bindings: &[TprBindings],
    dim: usize,
    idx: &[usize],
) -> (Vec<f32>, Vec<TprBindings>) {
    let mut out = Vec::with_capacity(idx.len() * dim);
    let mut b = Vec::with_capacity(idx.len());
    for &s in idx {
        out.extend_from_slice(&states[s * dim..(s + 1) * dim]);
        b.push(bindings[s].clone());
    }
    (out, b)
}

fn subset(c: &Corpus, idx: &[usize]) -> (Vec<f32>, Vec<TprBindings>) {
    let mut states = Vec::with_capacity(idx.len() * c.dim);
    let mut bindings = Vec::with_capacity(idx.len());
    for &s in idx {
        states.extend_from_slice(&c.states[s * c.dim..(s + 1) * c.dim]);
        bindings.push(c.bindings[s].clone());
    }
    (states, bindings)
}

// ─── Main ──────────────────────────────────────────────────────────────────

fn verdict(ok: bool) -> &'static str {
    if ok { "PASS" } else { "FAIL" }
}

fn main() {
    if std::env::args().any(|a| a == CHILD_ARG) {
        killswitch_child();
    }
    println!("Issue 707 — TPR binding-algebra GOAT gate (Research 527)");
    let g1 = run_g1();
    let g2 = run_g2();
    let g3 = run_g3();
    let g4 = run_g4();
    let g8 = run_g8();
    println!("\n── verdict ──");
    println!("  G1 {} | G2 {} | G3 {} | G4 {} | G8 {}", verdict(g1), verdict(g2), verdict(g3), verdict(g4), verdict(g8));
    if g1 && g2 && g3 && g4 && g8 { println!("Issue 707 GOAT gate: ALL PASS") } else {
            println!("Issue 707 GOAT gate: FAIL");
            std::process::exit(1);
        }
}
