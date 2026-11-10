//! Plan 578 — kinematic rollout GOAT gate (bench 680).
//!
//! - **G1**: multi-seed end-to-end determinism (bit-identical pipeline
//!   hashes) + the exactness fixtures live in the module's lib tests
//!   (`kinematics::tests` — uniform/parabola exactly-0 at k ∈ {1,10,100,1000},
//!   const-jerk at k ∈ {1,10,100} + the documented k=1000 mantissa band) —
//!   re-pinned here via the ID-OOD gap ≡ 0 table.
//! - **G2**: ns-cost table. The `< 10 ns` single-target extrapolate budget is
//!   **release-locked** (`#[cfg_attr(debug_assertions, ignore)]` — the
//!   ugc_g1b/funcattn precedent); debug runs still print the table.
//! - **T3.2**: the PhyWorld ID-OOD gap ≡ 0 table (paper ranges, dyadic grids).
//!
//! Run:
//!
//! ```bash
//! cargo test -p katgpt-core --test bench_680_kinematic_rollout_goat \
//!   --features kinematic_rollout --release -- --nocapture
//! ```

use katgpt_core::kinematics::fixture::{self, RangeLabel};
use katgpt_core::kinematics::perception::{
    Eps, RegimeClassifier, RegimeConfig, RegimeSnapshot, ResidualConfig, ResidualMonitor,
    closest_approach, extrapolation_horizon, time_to_contact,
};
use katgpt_core::kinematics::{
    KinState, Sched, kinematic_extrapolate_into, reference_chain_extrapolate_into,
};
use std::hint::black_box;
use std::time::Instant;

fn best_of_3<F: FnMut()>(iters: usize, mut f: F) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..3 {
        let t0 = Instant::now();
        for _ in 0..iters {
            f();
        }
        let ns = t0.elapsed().as_nanos() as f64 / iters as f64;
        if ns < best {
            best = ns;
        }
    }
    best
}

/// G1: end-to-end pipeline hash (extrapolate + regime + monitor) must be
/// bit-identical across independent runs, across seeds.
#[test]
fn g1_multi_seed_determinism() {
    fn pipeline_hash(seed: u64, label: RangeLabel) -> u64 {
        let fix = fixture::generate(seed, label);
        let mut st = KinState::<2>::new(1.0).unwrap();
        let mut clf = RegimeClassifier::new(RegimeConfig {
            g: fix.params.g,
            ..RegimeConfig::default()
        });
        let mut mon = ResidualMonitor::new(ResidualConfig::default());
        let mut prev_vel = [0.0f32; 2];
        let mut running_acc = 0.0f32;
        let mut prev_extent = fix.frames[0].extent;
        let mut predicted = [0.0f32; 2];
        let mut h: u64 = 0x9E37_79B9_7F4A_7C15;
        for f in fix.frames.iter() {
            let sigma_rate = f.extent - prev_extent;
            prev_extent = f.extent;
            kinematic_extrapolate_into(&st, 8, &Sched::Measured, &mut predicted).ok();
            let vel_before = st.vel;
            st.observe_into(&f.pos, f.tick).unwrap();
            let snap =
                RegimeSnapshot::from_state(&st, &prev_vel, running_acc, f.extent, sigma_rate);
            let v = clf.classify(&snap);
            running_acc += 0.1 * (snap.acc_mag - running_acc);
            prev_vel = st.vel;
            h = h
                .wrapping_mul(31)
                .wrapping_add(predicted[0].to_bits() as u64)
                .wrapping_add(predicted[1].to_bits() as u64)
                .wrapping_add(format!("{v:?}").len() as u64);
            if let Some(ev) = mon.update(predicted[0] - f.pos[0], &vel_before, &st.vel, 1.0) {
                h = h
                    .wrapping_mul(33)
                    .wrapping_add(format!("{ev:?}").len() as u64);
            }
        }
        h
    }
    for seed in [1u64, 2, 3, 7, 42, 0xDEAD_BEEF] {
        for label in [RangeLabel::Id, RangeLabel::Ood] {
            let a = pipeline_hash(seed, label);
            let b = pipeline_hash(seed, label);
            assert_eq!(a, b, "seed {seed} {label:?}: pipeline hash differs");
        }
    }
}

/// G1 support: closed-form ≡ reference-chain bit-identity on the exactness
/// family at scale (module tests pin the per-family anchors; this sweeps
/// every fixture × every k in the gap table's horizon set).
#[test]
fn g1_closed_form_equals_chain_across_fixtures() {
    let ks = [1u32, 8, 31];
    let seg_sched = [
        Sched::Measured,
        Sched::Measured,
        Sched::Measured,
        Sched::Measured,
        Sched::GeometricDrag { rho: 0.5 },
    ];
    for seed in 1..=6u64 {
        for label in [RangeLabel::Id, RangeLabel::Ood] {
            let fix = fixture::generate(seed, label);
            for (seg, sched) in seg_sched.iter().enumerate() {
                let start = fix.segments[seg];
                let end = fix
                    .segments
                    .get(seg + 1)
                    .copied()
                    .unwrap_or(fix.frames.len());
                let mut st = KinState::<2>::new(1.0).unwrap();
                let mut closed = [0.0f32; 2];
                let mut chained = [0.0f32; 2];
                for (i, f) in fix.frames[start..end].iter().enumerate() {
                    st.observe_into(&f.pos, start as u32 + i as u32).unwrap();
                    if st.n_obs < 4 {
                        continue;
                    }
                    // Drag arm: the bit-identity claim covers the resolvable
                    // regime — deep in the drag tail the emitted positions
                    // have frozen at their ULP grid and the two float paths
                    // round differently (the same floor the gap table uses).
                    if matches!(sched, Sched::GeometricDrag { .. }) {
                        let acc_mag = st.acc[0].abs().max(st.acc[1].abs());
                        if acc_mag <= katgpt_core::kinematics::perception::DRAG_ACC_FLOOR {
                            continue;
                        }
                    }
                    for &k in ks.iter() {
                        kinematic_extrapolate_into(&st, k, sched, &mut closed).unwrap();
                        reference_chain_extrapolate_into(&st, k, sched, &mut chained).unwrap();
                        if matches!(sched, Sched::GeometricDrag { .. }) {
                            // Drag: bit-identity holds while the decaying
                            // tail stays above the accumulator's half-ULP;
                            // beyond that the chain drops sub-ULP increments
                            // the closed form keeps — assert the measured
                            // ULP band instead (documented in the bench doc).
                            for ch in 0..2 {
                                let denom = chained[ch].abs().max(1.0);
                                let rel = ((closed[ch] - chained[ch]) / denom).abs();
                                assert!(
                                    rel < 1e-6,
                                    "seed {seed} {label:?} seg {seg} k={k} ch={ch}: \
                                     drag divergence {rel:e}"
                                );
                            }
                        } else {
                            assert_eq!(
                                closed[0].to_bits(),
                                chained[0].to_bits(),
                                "seed {seed} {label:?} seg {seg} k={k}: closed != chain"
                            );
                            assert_eq!(closed[1].to_bits(), chained[1].to_bits());
                        }
                    }
                }
            }
        }
    }
}

/// T3.2: the ID-OOD gap ≡ 0 table — on the analytic family, extrapolation
/// error is bit-exactly zero for BOTH the paper's ID and OOD parameter
/// ranges (the provable strengthening of the paper's empirical ~20× gap).
#[test]
fn t32_id_ood_gap_table() {
    let ks = [1u32, 8, 31];
    let seg_sched = [
        Sched::Measured,
        Sched::Measured,
        Sched::Measured,
        Sched::Measured,
        Sched::GeometricDrag { rho: 0.5 },
    ];
    let seg_names = ["uniform", "parabolic", "bounce", "looming", "drag"];
    println!("== PhyWorld ID-OOD gap table (T=31, paper ranges, dyadic grids) ==");
    println!("horizon | ID max err | OOD max err | gap");
    let mut id_worst = vec![0.0f32; ks.len()];
    let mut ood_worst = vec![0.0f32; ks.len()];
    let mut per_seg_id = vec![vec![0.0f32; ks.len()]; seg_names.len()];
    let mut per_seg_ood = vec![vec![0.0f32; ks.len()]; seg_names.len()];
    for seed in 1..=6u64 {
        for (label, worst, per_seg) in [
            (RangeLabel::Id, &mut id_worst, &mut per_seg_id),
            (RangeLabel::Ood, &mut ood_worst, &mut per_seg_ood),
        ] {
            let fix = fixture::generate(seed, label);
            for (seg, sched) in seg_sched.iter().enumerate() {
                let errs = fixture::extrapolation_errors(&fix, seg, &ks, sched);
                for (ki, e) in errs.iter().enumerate() {
                    worst[ki] = worst[ki].max(*e);
                    per_seg[seg][ki] = per_seg[seg][ki].max(*e);
                }
            }
        }
    }
    for (ki, k) in ks.iter().enumerate() {
        println!(
            "k={k:>2}     | {:+.3e}  | {:+.3e}  | {}",
            id_worst[ki],
            ood_worst[ki],
            if id_worst[ki] == 0.0 && ood_worst[ki] == 0.0 {
                "≡ 0 (bit-exact both arms)"
            } else {
                "NONZERO — see bench doc"
            }
        );
    }
    println!("\nPer-segment breakdown (max |err| over seeds × anchors × channels):");
    for (si, name) in seg_names.iter().enumerate() {
        for (ki, k) in ks.iter().enumerate() {
            assert_eq!(
                per_seg_id[si][ki], 0.0,
                "segment {name} k={k}: ID error nonzero"
            );
            assert_eq!(
                per_seg_ood[si][ki], 0.0,
                "segment {name} k={k}: OOD error nonzero"
            );
        }
        println!(
            "  {name:>9}: ID {:?} OOD {:?}",
            per_seg_id[si], per_seg_ood[si]
        );
    }
    // The paper's numbers for contrast: ID-OOD error gap ratio ~23.9×
    // (single-task, 256²). Ours: both arms identically zero — the ratio is
    // 0/0, undefined; the gap itself is identically zero.
    println!(
        "\npaper: LDR ID-OOD gap ratio ~23.9x (empirical, pixel MSE). Ours: gap ≡ 0 by construction (position units, bit-exact)."
    );
}

/// G2: ns-cost table. Budgets asserted in RELEASE only.
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "G2 budgets are release-mode (the ugc_g1b/funcattn precedent)"
)]
fn g2_ns_cost_table() {
    // ── single-target extrapolate: d=4, k=100, ZeroJerk ──────────────
    let mut st = KinState::<4>::new(1.0).unwrap();
    let mut x = [0.0f32; 4];
    for t in 0..8u32 {
        for (ch, slot) in x.iter_mut().enumerate() {
            *slot = 0.5 * t as f32 + 0.25 * ch as f32;
        }
        st.observe_into(&x, t).unwrap();
    }
    let sched = Sched::ZeroJerk;
    let mut out = [0.0f32; 4];
    let ns_extrap = best_of_3(1_000_000, || {
        kinematic_extrapolate_into(
            black_box(&st),
            black_box(100),
            black_box(&sched),
            black_box(&mut out),
        )
        .unwrap();
        black_box(out[0]);
    });
    println!("G2 single-target extrapolate (d=4, k=100, ZeroJerk): {ns_extrap:.2} ns");
    assert!(
        ns_extrap < 10.0,
        "single-target extrapolate over budget: {ns_extrap:.2} ns ≥ 10 ns"
    );

    // ConstJerk arm (adds the t3 term).
    let sched_cj = Sched::ConstJerk { j: 0.25 };
    let ns_cj = best_of_3(1_000_000, || {
        kinematic_extrapolate_into(
            black_box(&st),
            black_box(100),
            black_box(&sched_cj),
            black_box(&mut out),
        )
        .unwrap();
        black_box(out[0]);
    });
    println!("G2 single-target extrapolate (d=4, k=100, ConstJerk): {ns_cj:.2} ns");
    assert!(ns_cj < 10.0, "ConstJerk arm over budget: {ns_cj:.2} ns");

    // GeometricDrag arm (powf cost — separate budget).
    let sched_gd = Sched::GeometricDrag { rho: 0.5 };
    let ns_gd = best_of_3(200_000, || {
        kinematic_extrapolate_into(
            black_box(&st),
            black_box(100),
            black_box(&sched_gd),
            black_box(&mut out),
        )
        .unwrap();
        black_box(out[0]);
    });
    println!("G2 single-target extrapolate (d=4, k=100, GeometricDrag): {ns_gd:.2} ns");
    assert!(ns_gd < 50.0, "GeometricDrag arm over budget: {ns_gd:.2} ns");

    // ── 1000-target batch row (d=4, k=100, zero-alloc loop) ─────────
    const N: usize = 1000;
    let mut states: Vec<KinState<4>> = Vec::with_capacity(N);
    for i in 0..N {
        let mut s = KinState::<4>::new(1.0).unwrap();
        for t in 0..8u32 {
            for (ch, slot) in x.iter_mut().enumerate() {
                *slot = 0.5 * t as f32 + 0.01 * i as f32 + 0.25 * ch as f32;
            }
            s.observe_into(&x, t).unwrap();
        }
        states.push(s);
    }
    let mut outs = vec![[0.0f32; 4]; N];
    let ns_batch = best_of_3(2_000, || {
        for (s, o) in states.iter().zip(outs.iter_mut()) {
            kinematic_extrapolate_into(s, 100, &sched, o).unwrap();
        }
        black_box(outs[N - 1][0]);
    });
    println!(
        "G2 1000-target batch (d=4, k=100, ZeroJerk): {ns_batch:.0} ns total ({:.3} ns/target)",
        ns_batch / N as f64
    );
    assert!(ns_batch < 10_000.0, "batch over budget: {ns_batch:.0} ns");

    // ── perception operators ─────────────────────────────────────────
    let ns_ttc = best_of_3(1_000_000, || {
        black_box(time_to_contact(
            black_box(0.75),
            black_box(0.8125),
            black_box(1.0),
        ));
    });
    println!("G2 time_to_contact: {ns_ttc:.2} ns");
    assert!(ns_ttc < 10.0, "ttc over budget: {ns_ttc:.2} ns");

    let eps = Eps::from_obs_noise(0.05, 1.0);
    let ns_horizon = best_of_3(100_000, || {
        black_box(extrapolation_horizon(
            black_box(1.0),
            black_box(&eps),
            black_box(10.0),
        ));
    });
    println!("G2 extrapolation_horizon (scan to k*): {ns_horizon:.1} ns");
    assert!(
        ns_horizon < 2_000.0,
        "horizon over budget: {ns_horizon:.1} ns"
    );

    let p1 = [0.0f32, 0.0];
    let v1 = [1.0f32, 0.25];
    let p2 = [10.0f32, 5.0];
    let v2 = [-2.0f32, 0.0];
    let ns_ca = best_of_3(1_000_000, || {
        black_box(closest_approach(
            black_box(&p1),
            black_box(&v1),
            black_box(&p2),
            black_box(&v2),
        ));
    });
    println!("G2 closest_approach (d=2): {ns_ca:.2} ns");
    assert!(ns_ca < 20.0, "closest_approach over budget: {ns_ca:.2} ns");

    // Regime classification per tick.
    let fix = fixture::generate(1, RangeLabel::Id);
    let mut st2 = KinState::<2>::new(1.0).unwrap();
    let mut clf = RegimeClassifier::new(RegimeConfig {
        g: fix.params.g,
        ..RegimeConfig::default()
    });
    let prev_vel = [0.0f32; 2];

    // Warm one snapshot.
    let f0 = &fix.frames[10];
    st2.observe_into(&f0.pos, f0.tick).unwrap();
    let snap = RegimeSnapshot::from_state(&st2, &prev_vel, 0.25, f0.extent, -0.01);
    let ns_regime = best_of_3(500_000, || {
        black_box(clf.classify(black_box(&snap)));
    });
    println!("G2 regime classify (d=2): {ns_regime:.2} ns");
    assert!(
        ns_regime < 50.0,
        "regime classify over budget: {ns_regime:.2} ns"
    );

    // Residual monitor update (d=2).
    let mut mon = ResidualMonitor::new(ResidualConfig::default());
    let vb = [1.0f32, 0.5];
    let va = [1.0f32, 0.5];
    let ns_mon = best_of_3(500_000, || {
        black_box(mon.update(
            black_box(0.01),
            black_box(&vb),
            black_box(&va),
            black_box(1.0),
        ));
    });
    println!("G2 residual monitor update (d=2): {ns_mon:.2} ns");
    assert!(ns_mon < 50.0, "monitor over budget: {ns_mon:.2} ns");

    println!("\nG2 ALL PASS (release)");
}
