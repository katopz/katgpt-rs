//! Plan 619 T1.4/T1.6 gates — the LatticeMemory hot-path allocation ceiling
//! — plus the T1.9 freeze/thaw gates (snapshot/restore/commitment, the Plan
//! 199 T1.B consumer surface).
//!
//! The G4 law: after construction, every write/read/score is heap-free (the
//! slab, the cell table and the delta-rule scratch are allocated once in
//! `new`; the read path is direct row-wise accumulation needing no scratch).
//! Counted with the shared `TrackingAllocator` per-thread counters.
//!
//! Run (dev or the Issue-741 release posture):
//!
//! ```sh
//! cargo test -p katgpt-core --features lattice_memory --test lattice_memory_gates
//! cargo test -p katgpt-core --features lattice_memory,alloc_tracking \
//!     --release --test lattice_memory_gates
//! ```
//!
//! The allocator arms live behind
//! `any(debug_assertions, feature = "alloc_tracking")` (Issue 741 — the
//! profile is not a capability, the feature is); without either, the
//! disclosure test below still runs and prints the LOUD skip — a green run
//! that measured nothing is not a G4 pass.
#![cfg(feature = "lattice_memory")]

#[cfg(any(debug_assertions, feature = "alloc_tracking"))]
#[global_allocator]
static GATE_ALLOC: katgpt_core::alloc::TrackingAllocator =
    katgpt_core::alloc::TrackingAllocator;

use katgpt_core::lattice_memory::{
    BumpKernel, LatticeCell, LatticeConfig, LatticeMemory, LatticeSnapshot, LatticeSnapshotError,
};

fn config() -> LatticeConfig {
    LatticeConfig {
        d: 16,
        d_k: 4,
        d_v: 4,
        grid: [64, 64],
        w: [2.0, 2.0],
        seed: 42,
        byte_budget: None,
        warp: None,
        bump_kernel: BumpKernel::Tent,
    }
}

/// Always runs: names the posture that measured nothing, so a green count
/// over zero tests can't be read as a G4 pass (the bench_818 convention).
#[test]
fn alloc_gate_posture_disclosure() {
    if cfg!(any(debug_assertions, feature = "alloc_tracking")) {
        println!("alloc gate: ARMED (TrackingAllocator installed)");
    } else {
        println!(
            "G4 alloc-free ⛔ NOT MEASURED — this profile compiles no allocator. \
             Re-run with --features alloc_tracking (release) or in dev; \
             a green run without it is not a G4 pass."
        );
    }
}

#[test]
#[cfg(any(debug_assertions, feature = "alloc_tracking"))]
fn hot_path_is_alloc_free() {
    let mut lat = LatticeMemory::new(config()).unwrap();
    let x = [0.35_f32; 16];
    let k = [0.5_f32, 0.5, 0.5, 0.5];
    let v = [1.0_f32, 0.0, 0.0, 0.0];
    let mut out = [0.0_f32; 4];
    // Warm both paths (first-touch must not count against the hot loop).
    lat.write_delta(&x, &k, &v, 1.0);
    lat.read_cells(&x, &k, &mut out);
    lat.read_blend(&x, &mut out);
    let _ = lat.read_scored(&x, &k, &mut out, 4.0);
    let mut w = v;
    katgpt_core::alloc::reset_alloc_stats();
    for i in 0..1000_u32 {
        w[0] = (i & 0xFF) as f32;
        lat.write_delta(&x, &k, &w, 0.5);
        lat.write_value(&x, &w);
        lat.read_cells(&x, &k, &mut out);
        lat.read_blend(&x, &mut out);
        let score = lat.read_scored(&x, &k, &mut out, 4.0);
        std::hint::black_box(score);
        std::hint::black_box(&out);
    }
    let (count, bytes) = katgpt_core::alloc::get_alloc_stats();
    assert_eq!(
        count, 0,
        "hot path allocated {count} times ({bytes} B) — the G4 law is 0 after construction"
    );
}

#[test]
#[cfg(any(debug_assertions, feature = "alloc_tracking"))]
fn sized_constructor_reports_but_hot_path_stays_clean() {
    // The sizer projects + fits at CONSTRUCTION time (documented); the hot
    // path after construction is the same law as above.
    let mut rng = fastrand::Rng::with_seed(9);
    let samples: Vec<Vec<f32>> = (0..64)
        .map(|_| (0..16).map(|_| rng.f32() * 2.0 - 1.0).collect())
        .collect();
    let refs: Vec<&[f32]> = samples.iter().map(|s| s.as_slice()).collect();
    let mut lat =
        LatticeMemory::sized(16, 4, 4, 7, &refs, 64, 0.1, None).expect("sized lattice");
    let x = samples[0].clone();
    let k = [0.5_f32; 4];
    let mut out = [0.0_f32; 4];
    lat.write_delta(&x, &k, &[1.0; 4], 1.0);
    lat.read_cells(&x, &k, &mut out);
    katgpt_core::alloc::reset_alloc_stats();
    for (i, s) in samples.iter().enumerate().skip(1) {
        let mut vv = [0.0_f32; 4];
        vv[0] = i as f32;
        lat.write_delta(s, &k, &vv, 1.0);
        lat.read_cells(s, &k, &mut out);
        std::hint::black_box(&out);
    }
    let (count, _) = katgpt_core::alloc::get_alloc_stats();
    assert_eq!(count, 0, "sized-lattice hot path allocated {count} times");
}

// ── T1.9 — freeze/thaw (Plan 199 T1.B consumer surface) ───────────────

#[test]
fn snapshot_roundtrip_restores_exact_state() {
    let mut lat = LatticeMemory::new(config()).unwrap();
    let k = [0.5_f32; 4];
    let xa = [0.35_f32; 16];
    let xb =
        [0.9_f32, 0.1, 0.0, 0.8, 0.2, 0.0, 0.7, 0.3, 0.0, 0.6, 0.4, 0.0, 0.5, 0.5, 0.0, 0.4];
    lat.write_delta(&xa, &k, &[1.0, 0.0, 0.0, 0.0], 1.0);
    lat.write_delta(&xb, &k, &[0.0, 1.0, 0.0, 0.0], 0.5);
    lat.write_value(&xa, &[2.0, 0.0, 0.0, 0.0]);
    let mut blend_a = [0.0_f32; 4];
    let mut cells_b = [0.0_f32; 4];
    lat.read_blend(&xa, &mut blend_a);
    lat.read_cells(&xb, &k, &mut cells_b);
    let written_at_snap = lat.written_cells();
    let cursor_at_snap = lat.cursor();

    let snap = lat.snapshot();

    // Mutate past the snapshot; the thaw must undo it exactly.
    lat.write_delta(&xb, &k, &[0.0, 0.0, 1.0, 0.0], 1.0);

    // The artifact survives a serde round trip untouched.
    let bytes = serde_json::to_vec(&snap).expect("serialize snapshot");
    let thawed: LatticeSnapshot = serde_json::from_slice(&bytes).expect("deserialize snapshot");
    assert_eq!(snap.cells, thawed.cells, "serde round trip altered the cell table");
    assert_eq!(snap.slab, thawed.slab, "serde round trip altered the slab");

    lat.restore(&thawed).expect("restore");
    let mut blend_after = [0.0_f32; 4];
    let mut cells_after = [0.0_f32; 4];
    lat.read_blend(&xa, &mut blend_after);
    lat.read_cells(&xb, &k, &mut cells_after);
    assert_eq!(blend_a, blend_after, "blend must be bitwise identical after thaw");
    assert_eq!(cells_b, cells_after, "cell read must be bitwise identical after thaw");
    assert_eq!(lat.written_cells(), written_at_snap);
    assert_eq!(lat.cursor(), cursor_at_snap, "thaw must restore the slab cursor exactly");

    // Writes continue from the restored cursor: a post-thaw write to a
    // FRESH cell (not one of the two already-written primaries) claims slab
    // from exactly where the snapshot froze it.
    let pa = lat.primary_cell(&xa);
    let pb = lat.primary_cell(&xb);
    let mut fresh = None;
    for t in 1..64_u32 {
        let mut cand = [0.0_f32; 16];
        cand[0] = t as f32 * 0.13;
        cand[1] = 0.7;
        cand[2] = -0.4;
        let pc = lat.primary_cell(&cand);
        if pc != pa && pc != pb {
            fresh = Some(cand);
            break;
        }
    }
    let xc = fresh.expect("no fresh primary cell in 64 probes");
    lat.write_delta(&xc, &k, &[0.0, 0.0, 0.0, 1.0], 1.0);
    assert_eq!(lat.written_cells(), written_at_snap + 1, "fresh cell not claimed post-thaw");
    assert!(lat.cursor() > cursor_at_snap, "cursor did not continue past thaw");
}

#[test]
fn commitment_binds_content_and_geometry() {
    let build = || {
        let mut lat = LatticeMemory::new(config()).unwrap();
        lat.write_delta(&[0.35_f32; 16], &[0.5_f32; 4], &[1.0, 0.0, 0.0, 0.0], 1.0);
        lat
    };
    let a = build();
    let b = build();
    assert_eq!(a.commitment(), b.commitment(), "identical state must commit identically");

    // Empty lattices of different geometry never share a commitment (the
    // geometry header), even before any write.
    let empty_a = LatticeMemory::new(config()).unwrap();
    let mut small_cfg = config();
    small_cfg.grid = [32, 32];
    let empty_small = LatticeMemory::new(small_cfg).unwrap();
    assert_ne!(empty_a.commitment(), empty_small.commitment(), "geometry is in the header");

    // One extra write moves the commitment; the freeze/thaw identity holds.
    let x2 =
        [0.9_f32, 0.1, 0.0, 0.8, 0.2, 0.0, 0.7, 0.3, 0.0, 0.6, 0.4, 0.0, 0.5, 0.5, 0.0, 0.4];
    let mut c = build();
    let frozen = c.commitment();
    let snap = c.snapshot();
    c.write_delta(&x2, &[0.5_f32; 4], &[0.0, 1.0, 0.0, 0.0], 0.5);
    assert_ne!(frozen, c.commitment(), "a written byte must move the commitment");
    c.restore(&snap).expect("restore");
    assert_eq!(frozen, c.commitment(), "thaw must return the frozen commitment");
}

#[test]
fn restore_refuses_mismatched_geometry_and_corrupt_snapshots() {
    let mut lat = LatticeMemory::new(config()).unwrap();
    let snap = lat.snapshot();

    // Different geometry → refusal naming both sides.
    let mut other_cfg = config();
    other_cfg.grid = [32, 32];
    let mut other = LatticeMemory::new(other_cfg).unwrap();
    match other.restore(&snap) {
        Err(LatticeSnapshotError::GeometryMismatch { expected_cells, found_cells, .. }) => {
            assert_eq!(expected_cells, 32 * 32);
            assert_eq!(found_cells, snap.cells.len());
        }
        res => panic!("expected GeometryMismatch, got {res:?}"),
    }

    // Corrupt cursor → refusal.
    let mut bad_cursor = snap.clone();
    bad_cursor.cursor = bad_cursor.slab.len() + 1;
    assert!(matches!(
        lat.restore(&bad_cursor),
        Err(LatticeSnapshotError::CursorOutOfBounds { .. })
    ));

    // A cell claiming past the slab → refusal (a thawed lie can never
    // reach the hot path's slab indexing).
    let mut liar = snap.clone();
    liar.cells[0] = LatticeCell { offset: liar.slab.len() as u32 - 1, len: 8, writes: 1 };
    assert!(matches!(
        lat.restore(&liar),
        Err(LatticeSnapshotError::CellOutOfBounds { .. })
    ));

    // Every refusal happens before any write: the target is untouched.
    let untouched = lat.commitment();
    lat.restore(&bad_cursor).unwrap_err();
    lat.restore(&liar).unwrap_err();
    assert_eq!(untouched, lat.commitment(), "a refused restore must not tear state");
}

// ── Issue 938 T1.3 — the measured-erosion gate ─────────────────────────

/// λ rungs of the survival sweep: below, at, and past the pinned capacity
/// (0.35·32 = 11.2).
const EROSION_RUNGS: [u32; 6] = [2, 6, 11, 16, 22, 32];

/// One cell's survival measurement, in a FRESH lattice (the measurement
/// must not inherit earlier rungs' channels — their values are orthogonal
/// only to THEIR v₀, and the inherited v₀-components would drown the
/// signal): write (k₀, v₀), then λ−1 overwrites with fresh keys AND fresh
/// values from the ladders (each overwrite its own association — writing
/// the SAME value would re-add (k₀ᵀkw)·v along the k₀ channel every step
/// and the curve would converge to 1 instead of eroding), then read along
/// k₀. The read blends, so the survival is the RATIO of the v₀-projections
/// (the read vector is φ·(k₀ᵀS); projecting onto v₀ isolates the
/// association — a raw coordinate mixes every later value's components and
/// drowns the signal).
fn survival_after(
    cfg: &LatticeConfig,
    x: &[f32],
    k0: &[f32],
    v0: &[f32],
    lam: u32,
    keys_for: &mut dyn FnMut(u32) -> Vec<f32>,
    values_for: &mut dyn FnMut(u32) -> Vec<f32>,
) -> f32 {
    let mut lat = LatticeMemory::new(cfg.clone()).expect("fresh survival lattice");
    let proj = |out: &[f32]| -> f32 { out.iter().zip(v0.iter()).map(|(a, b)| a * b).sum() };
    lat.write_delta(x, k0, v0, 1.0);
    let mut out = [0.0_f32; 16];
    lat.read_cells(x, k0, &mut out);
    let base = proj(&out);
    // base = φ·‖v₀‖² > 0 always (v₀ is unit); the guard stays defensive.
    assert!(base.abs() > 1e-4, "pre-overwrite read vanished (base = {base})");
    for i in 1..lam {
        let kw = keys_for(i);
        let vw = values_for(i);
        lat.write_delta(x, &kw, &vw, 1.0);
    }
    lat.read_cells(x, k0, &mut out);
    proj(&out) / base
}

/// Mean pairwise cos² over the first `d_k` components — the erosion the
/// prediction consumes (self-measured on OUR fixture; never μLattice's
/// 0.083, per the issue's T1.3 law).
fn mean_pairwise_cos2(keys: &[Vec<f32>], d_k: usize) -> f32 {
    let mut sum = 0.0_f64;
    let mut count = 0_u64;
    for (a, ka) in keys.iter().enumerate() {
        for kb in keys.iter().skip(a + 1) {
            let mut dot = 0.0_f32;
            let mut na = 0.0_f32;
            let mut nb = 0.0_f32;
            for i in 0..d_k {
                dot += ka[i] * kb[i];
                na += ka[i] * ka[i];
                nb += kb[i] * kb[i];
            }
            let denom = na.sqrt() * nb.sqrt();
            if denom > 1e-12 {
                let c = dot / denom;
                sum += f64::from(c * c);
                count += 1;
            }
        }
    }
    (sum / count as f64) as f32
}

/// One Gaussian draw (the `address.rs` shape, test-side), unit-normalized.
fn draw_gaussian_unit(rng: &mut fastrand::Rng, d: usize) -> Vec<f32> {
    let mut k: Vec<f32> = (0..d)
        .map(|_| {
            let u1 = rng.f32().max(f32::EPSILON);
            let u2 = rng.f32();
            (-2.0 * u1.ln()).sqrt() * (core::f32::consts::TAU * u2).cos()
        })
        .collect();
    let n: f32 = k.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    for x in &mut k {
        *x /= n;
    }
    k
}

/// A `d_v`-wide unit value.
fn draw_value_unit(rng: &mut fastrand::Rng, d_v: usize) -> Vec<f32> {
    let mut v: Vec<f32> = (0..d_v).map(|_| draw_gaussian_unit(rng, 3)[0]).collect();
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    for x in &mut v {
        *x /= n;
    }
    v
}

/// `v` orthogonalized against `v0` and renormalized — the overwrite values
/// of the survival fixture. With v_w ⊥ v0 the v₀-projection of every add
/// term is exactly zero, so the measured survival is the erosion product
/// Π(1 − eᵢ²) itself (the capacity law's premise: associations independent
/// of the probed one) — not that product buried under add-term noise.
fn orthogonal_to(vw: &[f32], v0: &[f32]) -> Vec<f32> {
    let mut v = vw.to_vec();
    let d: f32 = v.iter().zip(v0.iter()).map(|(a, b)| a * b).sum();
    for (vv, &b) in v.iter_mut().zip(v0.iter()) {
        *vv -= d * b;
    }
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    for vv in &mut v {
        *vv /= n;
    }
    v
}

#[test]
fn erosion_survival_brackets_the_predicted_curve() {
    // T1.3 (G1): sweep λ vs survival on OUR fixtures, random keys AND
    // correlated keys. The predicted pure-erosion curve p(λ) = (1−e)^(λ−1)
    // with e self-measured must bracket the measured survival within the
    // stated tolerance (±0.12 — first-order law, cross-terms erode a little
    // faster); the shipped promise is that survival AT the pinned capacity
    // (0.35·d_k = 11.2 → rung 11) stays ≥ 0.5 on random keys, and that the
    // correlated fixture measures STRICTLY faster erosion (the sizing
    // hazard is real on our keys, measured our way).
    const D: usize = 32;
    const D_V: usize = 16;
    const CELLS: u32 = 8;
    let cfg = LatticeConfig {
        d: D,
        d_k: D,
        d_v: D_V,
        grid: [CELLS, CELLS],
        w: [1.0, 1.0],
        seed: 0x938_0133,
        byte_budget: None,
        warp: None,
        bump_kernel: BumpKernel::Tent,
    };

    // Random unit keys: e ≈ 1/d_k.
    let mut rng = fastrand::Rng::with_seed(0x938_0134);

    // The random arm: one fresh address per (trial, rung); keys drawn
    // per-overwrite. Median over 24 trials kills the per-cell variance.
    let trials = 24_u32;
    let mut random_survival: Vec<Vec<f32>> = Vec::new();
    let mut ladder_keys: Vec<Vec<f32>> = Vec::new();
    for _ in 0..32 {
        ladder_keys.push(draw_gaussian_unit(&mut rng, D));
    }
    let e_random = mean_pairwise_cos2(&ladder_keys, D);
    assert!(
        (e_random - 1.0 / D as f32).abs() < 0.02,
        "fixture keys must sit at the random-key erosion: e = {e_random}"
    );
    // The value ladder: 40 unit values. Each rung orthogonalizes its picks
    // against that rung's v₀ (see `orthogonal_to`) — the measured survival
    // is then the erosion product itself.
    let value_ladder: Vec<Vec<f32>> =
        (0..40).map(|_| draw_value_unit(&mut rng, D_V)).collect();
    for _ in 0..trials {
        let mut row = Vec::with_capacity(EROSION_RUNGS.len());
        for (ri, &lam) in EROSION_RUNGS.iter().enumerate() {
            // A fresh address per rung: a grid point + key-derived offset.
            let mut x = [0.0_f32; D];
            let u = draw_gaussian_unit(&mut rng, D);
            for (xi, &u_i) in x.iter_mut().zip(u.iter()) {
                *xi = u_i * 0.9;
            }
            let k0 = draw_gaussian_unit(&mut rng, D);
            let v0r = draw_value_unit(&mut rng, D_V);
            let rung_values: Vec<Vec<f32>> = value_ladder
                .iter()
                .map(|vw| orthogonal_to(vw, &v0r))
                .collect();
            let s = survival_after(&cfg, &x, &k0, &v0r, lam, &mut |i| {
                // Deterministic per-overwrite key from the pre-drawn ladder.
                ladder_keys[(i as usize + ri * 7) % ladder_keys.len()].clone()
            }, &mut |i| rung_values[(i as usize + ri * 5) % rung_values.len()].clone());
            row.push(s);
        }
        random_survival.push(row);
    }
    // Median across trials per rung.
    let median = |col: usize, rows: &[Vec<f32>]| -> f32 {
        let mut vals: Vec<f32> = rows.iter().map(|r| r[col]).collect();
        vals.sort_by(f32::total_cmp);
        vals[vals.len() / 2]
    };
    let predicted = |lam: u32, e: f32| (1.0 - e).powi((lam - 1) as i32);
    for (ri, &lam) in EROSION_RUNGS.iter().enumerate() {
        let s = median(ri, &random_survival);
        let p = predicted(lam, e_random);
        assert!(
            s <= p + 0.12 && s >= p * p - 0.12,
            "random arm λ={lam}: survival {s:.3} outside the erosion bracket \
             [{:.3}, {:.3}] (e = {e_random:.4})",
            p * p - 0.12,
            p + 0.12
        );
    }
    // The shipped promise at the pinned capacity rung (11.2 → 11).
    let s_at_cap = median(2, &random_survival);
    assert!(
        s_at_cap >= 0.5,
        "the 0.35·d_k promise broke on OUR random keys: survival {s_at_cap:.3} at λ=11"
    );

    // The correlated arm: same protocol, keys clustered around one center
    // per trial (cos² ≈ 0.1–0.2 — the vocabulary-style shape). The measured
    // e feeds the SAME prediction; the survival must sit measurably lower
    // than the random arm at the capacity rung.
    let mut clustered_survival = Vec::with_capacity(trials as usize);
    let mut clusters: Vec<Vec<Vec<f32>>> = Vec::with_capacity(trials as usize);
    for t in 0..trials {
        let center = draw_gaussian_unit(&mut rng, D);
        let mut cluster: Vec<Vec<f32>> = Vec::new();
        for _ in 0..32 {
            let mut k: Vec<f32> = center.iter().map(|&c| c * 0.62).collect();
            let noise = draw_gaussian_unit(&mut rng, D);
            for (k_i, &n_i) in k.iter_mut().zip(noise.iter()) {
                *k_i += n_i * 0.78;
            }
            let n: f32 = k.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            for x in &mut k {
                *x /= n;
            }
            cluster.push(k);
        }
        let e_hat = mean_pairwise_cos2(&cluster, D);
        if t == 0 {
            assert!(
                e_hat > 3.0 / D as f32,
                "correlated fixture must erode faster than random: e = {e_hat}"
            );
        }
        let mut row = Vec::with_capacity(EROSION_RUNGS.len());
        for &lam in EROSION_RUNGS.iter() {
            let mut x = [0.0_f32; D];
            let u = draw_gaussian_unit(&mut rng, D);
            for (xi, &u_i) in x.iter_mut().zip(u.iter()) {
                *xi = u_i * 0.9;
            }
            let k0 = cluster[0].clone();
            let v0r = draw_value_unit(&mut rng, D_V);
            let rung_values: Vec<Vec<f32>> = value_ladder
                .iter()
                .map(|vw| orthogonal_to(vw, &v0r))
                .collect();
            let s = survival_after(&cfg, &x, &k0, &v0r, lam, &mut |i| {
                // Never the k₀ slot itself — a same-key rewrite resets the
                // erosion and would flatter the curve.
                cluster[(i as usize - 1) % (cluster.len() - 1) + 1].clone()
            }, &mut |i| rung_values[(i as usize + t as usize * 3) % rung_values.len()].clone());
            row.push(s);
        }
        clustered_survival.push((e_hat, row));
        clusters.push(cluster);
    }
    // The correlated arm's honest prediction: the EXACT per-trial algebra —
    // the effective key recursion k_eff ← k_eff − (k_effᵀkw)·kw per
    // overwrite, survival = k₀ᵀk_eff — which includes the key ROTATION the
    // naive product Π(1 − cos²) ignores: under a cluster the removals share
    // a direction, the overlaps shrink as k_eff rotates away, and survival
    // SATURATES (measured ≈ 0.30 at λ=11, e = 0.184 — the aggregate sizer's
    // law (1 − ē)^(λ−1) predicts 0.131 there, i.e. the shipped eroded
    // refusal surface OVER-refuses parallel-key corpora: the safe
    // direction, recorded as the gate's conservatism finding).
    let mut agg_below_exact = 0_u32;
    for (ti, ((e_hat, row), cluster)) in clustered_survival.iter().zip(clusters.iter()).enumerate()
    {
        let _ = e_hat;
        let k0 = &cluster[0];
        let mut exact_run = 0_u32;
        for (ri, &lam) in EROSION_RUNGS.iter().enumerate() {
            // Fresh cell per rung: the recursion restarts at k_eff = k₀.
            let mut k_eff = k0.clone();
            let mut exact = 1.0_f32; // k₀ᵀk_eff
            for i in 1..lam {
                let kw = &cluster[(i as usize - 1) % (cluster.len() - 1) + 1];
                let coef: f32 = k_eff.iter().zip(kw.iter()).map(|(a, b)| a * b).sum();
                let k0_dot_kw: f32 = k0.iter().zip(kw.iter()).map(|(a, b)| a * b).sum();
                for (ke, &w) in k_eff.iter_mut().zip(kw.iter()) {
                    *ke -= coef * w;
                }
                exact -= coef * k0_dot_kw;
            }
            let s = row[ri];
            assert!(
                (s - exact).abs() <= 0.08,
                "correlated arm trial {ti} λ={lam}: survival {s:.3} vs the exact \
                 key-recursion algebra {exact:.3} — the delta-rule drifted"
            );
            if predicted(lam, *e_hat) < exact - 0.02 {
                agg_below_exact += 1;
                exact_run += 1;
            }
        }
        let _ = exact_run;
    }
    // The recorded direction: the aggregate law sat BELOW the exact algebra
    // at essentially every (trial, rung) — conservatism, quantified.
    let total_cells = clustered_survival.len() * EROSION_RUNGS.len();
    assert!(
        agg_below_exact as f32 / total_cells as f32 > 0.9,
        "the aggregate law's conservatism direction moved: {agg_below_exact}/{total_cells}"
    );
    // The hazard, measured our way: correlated survival at the capacity
    // rung sits strictly below the random arm's (median over trials).
    let mut cs: Vec<f32> = clustered_survival.iter().map(|(_, r)| r[2]).collect();
    cs.sort_by(f32::total_cmp);
    let cs_med = cs[cs.len() / 2];
    assert!(
        cs_med < s_at_cap - 0.05,
        "correlated erosion must measure faster: {cs_med:.3} vs random {s_at_cap:.3}"
    );
}
