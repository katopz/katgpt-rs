//! Issue 932 T1+T3 — belief-views viability probe (Research 619 MSAR
//! time-depth transpose). Measures, on the SHIPPED per-tick pipeline
//! (`route → accumulate → evolve_belief`, the riir-engine
//! `HabituatedPerceptor::sense_tick` shape), whether the belief trajectory
//! carries the multi-scale, history-destroying structure that weight-free
//! multi-view temporal retrieval would need:
//!
//! - **T3 (E1, viability decider) — lossiness:** can `b_{t-1}` be recovered
//!   from `b_t` and the tick's own evidence? `leaky_step` has NO self-dynamics
//!   (its delta reads the evidence only), so the only many-to-one op is the
//!   `[-1, 1]` clamp. Loss = the fraction of channel-ticks where the clamp binds.
//! - **T1 (E2) — time scale:** per-channel autocorrelation half-life τ of the
//!   post-warm-up trajectory.
//!
//! Synthetic corpus: seeded (inline splitmix — no global RNG) per-tick sense
//! activations with a dominant-kind regime switch every `REGIME` ticks.
//!
//! Run: `cargo test -p katgpt-sense --test belief_views_viability_probe -- --nocapture`

use katgpt_sense::reconstruction::ReconstructionState;

const TICKS: usize = 2000;
const WARMUP: usize = 200;
const REGIME: usize = 250;
const SEEDS: u64 = 16;

struct SplitMix(u64);
impl SplitMix {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        (z >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// One tick of sense activations: noise on every kind plus a strong dominant
/// kind that switches every `REGIME` ticks (a context change).
fn activations(rng: &mut SplitMix, tick: usize) -> [f32; 6] {
    let dominant = (tick / REGIME) % 6;
    let mut a = [0.0f32; 6];
    for (m, v) in a.iter_mut().enumerate() {
        *v = 0.3 * rng.next_f32();
        if m == dominant {
            *v += 0.7;
        }
    }
    a
}

/// The shipped per-tick pipeline; returns the trajectory.
fn trajectory(seed: u64) -> Vec<[f32; 8]> {
    let mut rng = SplitMix(seed);
    let mut st = ReconstructionState::new([0.0; 8]);
    let mut out = Vec::with_capacity(TICKS);
    for t in 0..TICKS {
        let a = activations(&mut rng, t);
        let sel = st.route(&a);
        st.accumulate(&sel, &a);
        st.evolve_belief();
        out.push(*st.belief());
    }
    out
}

/// Autocorrelation half-life (ticks) of one channel over `xs`; `None` when
/// the channel is constant (zero variance — no structure at any scale).
fn acf_half_life(xs: &[f32]) -> Option<usize> {
    let n = xs.len();
    let mean = xs.iter().sum::<f32>() / n as f32;
    let var: f32 = xs.iter().map(|x| (x - mean) * (x - mean)).sum();
    if var < 1e-12 {
        return None;
    }
    for lag in 1..n / 2 {
        let c: f32 = (0..n - lag).map(|i| (xs[i] - mean) * (xs[i + lag] - mean)).sum();
        if c / var < 0.5 {
            return Some(lag);
        }
    }
    Some(n / 2)
}

#[test]
fn belief_views_viability_probe() {
    let mut clamped = 0usize;
    let mut interior = 0usize;
    let mut constant_channels = 0usize;
    let mut half_lives = Vec::new();
    // Two initial beliefs, identical evidence: with no self-dynamics their
    // difference is preserved exactly until a clamp binds (closed-form T3).
    let mut max_interior_drift = 0.0f32;
    for s in 0..SEEDS {
        let seed = 0x932 + s * 0x9e37;
        let traj = trajectory(seed);
        for b in &traj[WARMUP..] {
            for &v in b {
                if v.abs() >= 1.0 {
                    clamped += 1;
                } else {
                    interior += 1;
                }
            }
        }
        for ch in 0..8 {
            let xs: Vec<f32> = traj[WARMUP..].iter().map(|b| b[ch]).collect();
            match acf_half_life(&xs) {
                Some(h) => half_lives.push(h),
                None => constant_channels += 1,
            }
        }
        // Self-dynamics check on the first 20 ticks (before saturation).
        let mut rng_a = SplitMix(seed);
        let mut rng_b = SplitMix(seed);
        let mut a = ReconstructionState::new([0.1; 8]);
        let mut b = ReconstructionState::new([-0.1; 8]);
        for t in 0..20 {
            let (xa, xb) = (activations(&mut rng_a, t), activations(&mut rng_b, t));
            let (sa, sb) = (a.route(&xa), b.route(&xb));
            a.accumulate(&sa, &xa);
            b.accumulate(&sb, &xb);
            a.evolve_belief();
            b.evolve_belief();
            for ch in 0..8 {
                let (va, vb) = (a.belief()[ch], b.belief()[ch]);
                if va.abs() < 1.0 && vb.abs() < 1.0 {
                    max_interior_drift = max_interior_drift.max(((va - vb) - 0.2).abs());
                }
            }
        }
    }
    let total = clamped + interior;
    let clamp_frac = clamped as f64 / total as f64;
    half_lives.sort_unstable();
    let med = half_lives.get(half_lives.len() / 2).copied();
    println!(
        "[932 probe] post-warm-up channel-ticks {total}: clamped {clamped} ({:.1}%), interior {interior}",
        100.0 * clamp_frac
    );
    println!(
        "[932 probe] channels: {} constant (zero variance) of {}; ACF half-life median {:?} ticks, range {:?}..{:?}",
        constant_channels,
        SEEDS * 8,
        med,
        half_lives.first(),
        half_lives.last()
    );
    println!("[932 probe] interior self-dynamics drift (two inits, same evidence): {max_interior_drift:e}");

    // Comparison arm — the modality-additive kernel (default-on feature, no
    // workspace consumer today): a convex per-dim EMA of a sigmoid drive, so
    // the clamp can never bind and b_{t-1} = (b_t − (1−α)·d_t)/α is exact.
    #[cfg(feature = "modality_additive")]
    {
        let (mut add_clamped, mut add_total) = (0usize, 0usize);
        let mut add_half = Vec::new();
        for s in 0..SEEDS {
            let mut rng = SplitMix(0x932 + s * 0x9e37);
            let mut st = ReconstructionState::new([0.0; 8]);
            let mut traj = Vec::with_capacity(TICKS);
            for t in 0..TICKS {
                st.evolve_belief_additive(&activations(&mut rng, t));
                traj.push(*st.belief());
            }
            for b in &traj[WARMUP..] {
                add_total += 8;
                add_clamped += b.iter().filter(|v| v.abs() >= 1.0).count();
            }
            for ch in 0..8 {
                let xs: Vec<f32> = traj[WARMUP..].iter().map(|b| b[ch]).collect();
                if let Some(h) = acf_half_life(&xs) {
                    add_half.push(h);
                }
            }
        }
        add_half.sort_unstable();
        println!(
            "[932 probe] additive arm: clamped {add_clamped}/{add_total}; non-constant channels {}; ACF half-life median {:?}",
            add_half.len(),
            add_half.get(add_half.len() / 2)
        );
        assert_eq!(add_clamped, 0, "additive kernel is convex — the clamp must never bind");
    }

    // Structural saturation (any evidence stream): once Σ evidence ≥ 1 the
    // divisive delta is lr·(p_i − 0.5) over cumulative kind shares Σp = 1, so
    // at most ONE kind (≤ 2 dims via KIND_MAP) can drift non-negative; ≥ 6 of
    // 8 dims ramp monotonically into the −1 clamp and stay there.
    assert!(
        constant_channels as u64 >= SEEDS * 6,
        "divisive kernel no longer saturates ≥ 6/8 dims: {constant_channels}"
    );

    // Pinned premise: `leaky_step` has no self-dynamics — the interior map is
    // a pure translation, so b_{t-1} = b_t − δ_t exactly (T3 closed form).
    assert!(max_interior_drift < 1e-5, "evolve_belief gained self-dynamics: {max_interior_drift}");
}
