#![cfg(all(feature = "estimators", feature = "sink_aware_attn"))]
//! Plan 638 T2.7 — the G5 incumbent kill-switch (ω-sweep amendment).
//!
//! **Amendment (delegated verdict round 1, session 8ba704f4, REVISE):**
//! the original arm-1 plant (`sin(dot)`, dot ~ N(0,1)) is ~85% linear by
//! variance — a NULL regime where the spectral incumbents cannot lose —
//! and the arm ran at 32 dims where the plan's D3 prose says 64. This
//! gate now runs the pre-fixed frequency sweep.
//!
//! **PRE-REGISTERED READ (pinned BEFORE the sweep ran — plan T3.1):**
//! B = `(ω·dot).sin()` through 16 fixed seeded directions, u ∈ R⁴
//! gaussian; SPACE_D = 64 (the plan's prose, disclosed); n = 128 per
//! group; τ = 0.9. ω ∈ {1, 2, 3}: ω = 1, 2 are descriptive rungs; **the
//! DECISION arm is ω = 3** (linear fraction ≈ 0.2% — the regime the arm
//! was designed for).
//!
//! - **HOLD** (complementary signal exists) iff at ω = 3 ALL of:
//!   incumbents mis-order (PR_B > PR_A) AND |d̂_B − 4| ≤ 1.5 AND
//!   d̂_A ≥ d̂_B + 1. Re-open triggers then = a labeled real shape + the
//!   G2 bar.
//! - **DEMOTE** otherwise (the incumbents still order the plant right at
//!   ω = 3, or cond_id breaks its own bars there) — now with a real
//!   counter-instance.
//!
//! The gate asserts finiteness of every reading and prints the decision
//! line; the verdict itself is recorded in the plan (T3.1), never
//! asserted here — a self-verdicting test proves nothing.
//!
//! **Arm 2 — the real-shape calibration (disclosed, never pass/fail):**
//! the frozen rule embed (`../riir-refine/assets/rule_embed_frozen_v1.bin`,
//! BLAKE3-verified before parse; the integrity contract stays refine-side
//! `decode_verified`). 268 rules × 64 dims, v1 layout. No domain labels
//! are derivable from the kebab-case rule ids, so this arm records each
//! estimator's reading on the real population — the dense-core contrast
//! (64 rows nearest the centroid vs the rest) plus the full population —
//! and NEVER a pass/fail on a label contrast. Disclosed deviation: the
//! Plan-638 prose calls this fixture "8-D"; the shipped v1 artifact is
//! 64-DIM — the gate reads the real file.
//!
//! ⚠ Run at `--release` (O(n²·d) arms; the debug wall proves nothing).

#[path = "fixture_cond_geometry/mod.rs"]
mod fixture;

use fixture::Lcg;
use katgpt_core::data_probe::cond_geometry::{
    BallCountScratch, MIN_GROUP_TOKENS, ball_counts, cond_id,
};
use katgpt_core::data_probe::geometry::within_class_effective_rank;
use katgpt_core::subspace_phase_gate::{
    SvdResultScratch, SvdScratch, numerical_rank, participation_ratio, thin_svd_into,
};

const GROUP: usize = 128;
const SPACE_D: usize = 64;
const TAU: f32 = 0.9;

// ── Arm 1 — the pre-fixed frequency sweep (the decision arm is ω = 3) ─

/// Group A: isotropic LINEAR cloud in 8 dims (spread dim = intrinsic dim),
/// embedded in SPACE_D.
fn linear_cloud(seed: u64) -> Vec<f32> {
    let mut pts = vec![0.0f32; GROUP * SPACE_D];
    let mut rng = Lcg::new(seed);
    for i in 0..GROUP {
        let row = &mut pts[i * SPACE_D..(i + 1) * SPACE_D];
        for (c, v) in row.iter_mut().enumerate() {
            *v = if c < 8 { rng.next_gauss() } else { 0.0 };
        }
    }
    pts
}

/// Group B: the CURVED 4-manifold — u ∈ R⁴ gaussian, embedded through 16
/// fixed seeded sinusoidal directions at frequency `omega`. At ω = 1 the
/// plant is ~85% linear by variance (the null regime the round-1 verdict
/// named); at ω = 3 it is ≈ 0.2% linear — genuinely curved.
fn curved_manifold(seed: u64, omega: f64) -> Vec<f32> {
    let mut rng = Lcg::new(seed);
    // 16 fixed directions w_j ∈ R⁴, unit-normalized.
    let mut dirs = [[0.0f64; 4]; 16];
    for w in dirs.iter_mut() {
        let mut n2 = 0.0;
        for v in w.iter_mut() {
            *v = rng.next_gauss() as f64;
            n2 += *v * *v;
        }
        for v in w.iter_mut() {
            *v /= n2.sqrt();
        }
    }
    let mut pts = vec![0.0f32; GROUP * SPACE_D];
    for i in 0..GROUP {
        let mut u = [0.0f64; 4];
        for v in u.iter_mut() {
            *v = rng.next_gauss() as f64;
        }
        let row = &mut pts[i * SPACE_D..(i + 1) * SPACE_D];
        for (c, w) in dirs.iter().enumerate() {
            let dot: f64 = u.iter().zip(w.iter()).map(|(a, b)| a * b).sum();
            row[c] = (omega * dot).sin() as f32;
        }
    }
    pts
}

fn cond_id_of_group(pts: &[f32]) -> f32 {
    let n = pts.len() / SPACE_D;
    let all: Vec<u32> = (0..n as u32).collect();
    let mut scratch = BallCountScratch::with_capacity(n);
    ball_counts(pts, n, SPACE_D, &all, TAU, &mut scratch).expect("counts");
    cond_id(&scratch, &all).expect("estimate")
}

/// The incumbent spectral readings on a group: centered thin SVD →
/// eigenvalues (σ²) → participation ratio + numerical rank (η = 0.95).
fn spectral_incumbents(pts: &[f32], dim: usize) -> (f32, usize) {
    let n = pts.len() / dim;
    // Center the group.
    let mut mat = pts.to_vec();
    for c in 0..dim {
        let mean: f32 = (0..n).map(|i| mat[i * dim + c]).sum::<f32>() / n as f32;
        for i in 0..n {
            mat[i * dim + c] -= mean;
        }
    }
    let mut res = SvdResultScratch::with_capacity(n, dim);
    let mut work = SvdScratch::with_capacity(dim, n);
    thin_svd_into(&mat, n, dim, &mut res, &mut work);
    let mut eig: Vec<f32> = res.singular_values().iter().map(|s| s * s).collect();
    eig.sort_by(|a, b| b.partial_cmp(a).expect("finite singulars"));
    (participation_ratio(&eig), numerical_rank(&eig, 0.95))
}

#[test]
fn g5_frequency_sweep_and_incumbent_record() {
    const {
        assert!(GROUP >= MIN_GROUP_TOKENS);
    }

    // ── the sweep: ω ∈ {1, 2, 3}; the DECISION arm is ω = 3 (pre-fixed) ──
    let mut decision = String::new();
    for (omega, seed_a, seed_b) in [
        (1.0f64, 0x638_5001u64, 0x638_5002u64),
        (2.0, 0x638_5011, 0x638_5012),
        (3.0, 0x638_5021, 0x638_5022),
    ] {
        let a = linear_cloud(seed_a);
        let b = curved_manifold(seed_b, omega);
        let d_a = cond_id_of_group(&a);
        let d_b = cond_id_of_group(&b);

        let mut combined = a.clone();
        combined.extend_from_slice(&b);
        let labels: Vec<usize> = (0..GROUP).map(|_| 0).chain((0..GROUP).map(|_| 1)).collect();
        let eff = within_class_effective_rank(&combined, SPACE_D, &labels);
        let (pr_a, nr_a) = spectral_incumbents(&a, SPACE_D);
        let (pr_b, nr_b) = spectral_incumbents(&b, SPACE_D);

        println!(
            "G5 ω={omega}: d̂_A={d_a:.3} d̂_B={d_b:.3} | PR_A={pr_a:.2} PR_B={pr_b:.2} | nr A={nr_a} B={nr_b} | eff_rank={eff:.2}"
        );
        assert!(
            d_a.is_finite() && d_b.is_finite(),
            "ω={omega}: non-finite reading"
        );

        if omega == 3.0 {
            let incumbents_misorder = pr_b > pr_a;
            let cond_id_holds_b = (f64::from(d_b) - 4.0).abs() <= 1.5;
            let cond_id_separates = f64::from(d_a) >= f64::from(d_b) + 1.0;
            decision = if incumbents_misorder && cond_id_holds_b && cond_id_separates {
                "HOLD — complementary signal exists (incumbents mis-order, cond_id holds its bars)"
                    .into()
            } else {
                "DEMOTE — no complementary signal (the incumbents still order the plant right, or cond_id breaks its bars)".into()
            };
            println!(
                "G5 DECISION ARM ω=3 (pre-fixed read): incumbents mis-order={} (PR_B {pr_b:.2} vs PR_A {pr_a:.2}) | |d̂_B−4|={:.3}≤1.5={} | d̂_A≥d̂_B+1={} → {}",
                incumbents_misorder,
                (f64::from(d_b) - 4.0).abs(),
                cond_id_holds_b,
                cond_id_separates,
                decision
            );
        }
    }
    println!("G5 T2.7 verdict (recorded in the plan): {decision}");
    assert!(!decision.is_empty(), "the decision arm never ran");

    // ── Arm 2 — the real-shape calibration (disclosed, never pass/fail) ──
    real_shape_calibration();
}

// ── Arm 2 — the real-shape calibration row ─────────────────────────────

const REAL_DIM: usize = 64;

/// Parse + BLAKE3-verify the frozen artifact; return the ψ rows
/// ([n_rules × 64], f32 dequantized) and the rule count.
fn load_rule_embed() -> Option<(Vec<f32>, usize)> {
    let path = std::env::var("RULE_EMBED_FIXTURE").unwrap_or_else(|_| {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../riir-refine/assets/rule_embed_frozen_v1.bin"
        )
        .to_string()
    });
    let Ok(bytes) = std::fs::read(&path) else {
        println!(
            "G5 real shape: fixture absent at {path} — SKIPPED LOUD (a deferral, never a green)"
        );
        return None;
    };
    if bytes.len() < 60 {
        panic!("G5 real shape: artifact too short ({} bytes)", bytes.len());
    }
    let (payload, hash) = bytes.split_at(bytes.len() - 32);
    let got: [u8; 32] = hash.try_into().expect("32-byte hash");
    let want = blake3::hash(payload);
    assert_eq!(
        want.as_bytes(),
        &got,
        "G5 real shape: blake3 mismatch — corrupt or tampered fixture"
    );
    let u32_at = |p: &[u8], o: usize| -> usize {
        u32::from_le_bytes(p[o..o + 4].try_into().unwrap()) as usize
    };
    assert_eq!(&payload[..4], b"RLEM", "bad magic");
    let version = u32_at(payload, 4);
    assert_eq!(version, 1, "format version {version} != 1");
    let buckets = u32_at(payload, 8);
    let dim = u32_at(payload, 12);
    let tau = f32::from_le_bytes(payload[16..20].try_into().unwrap());
    let n_rules = u32_at(payload, 20);
    assert!(tau.is_finite() && tau > 0.0, "tau {tau}");
    assert_eq!(dim, REAL_DIM, "artifact dim {dim} != {REAL_DIM}");
    let mut pos = 28 + buckets * dim;
    let mut psi_rows = Vec::with_capacity(n_rules);
    let mut ids = Vec::with_capacity(n_rules);
    for _ in 0..n_rules {
        let id_len = u32_at(payload, pos);
        pos += 4;
        let id = String::from_utf8(payload[pos..pos + id_len].to_vec()).expect("rule id utf-8");
        pos += id_len;
        let scale = f32::from_le_bytes(payload[pos..pos + 4].try_into().unwrap());
        pos += 4;
        assert!(scale.is_finite() && scale > 0.0, "ψ scale {scale}");
        let row: Vec<f32> = payload[pos..pos + dim]
            .iter()
            .map(|&b| ((b as i8) as f32 / 127.0) * scale)
            .collect();
        pos += dim;
        ids.push(id);
        psi_rows.extend_from_slice(&row);
    }
    let trailing = payload.len() - pos;
    assert_eq!(trailing, 0, "trailing bytes: {trailing}");
    assert_eq!(ids.len(), n_rules, "rule count pin");
    Some((psi_rows, n_rules))
}

fn real_shape_calibration() {
    let Some((rows, n_rules)) = load_rule_embed() else {
        return;
    };
    println!(
        "G5 real shape: {n_rules} rules × {REAL_DIM} dims (v1 fixture; ids unprefixed — no domain labels derivable)"
    );

    // The full-population reading.
    let all: Vec<u32> = (0..n_rules as u32).collect();
    let mut scratch = BallCountScratch::with_capacity(n_rules);
    ball_counts(&rows, n_rules, REAL_DIM, &all, TAU, &mut scratch).expect("real counts");
    let d_full = cond_id(&scratch, &all).expect("real estimate");
    println!("G5 real shape: cond_id(full) = {d_full:.3}");

    // The dense-core contrast: the 64 rows nearest the centroid.
    let mut centroid = vec![0.0f64; REAL_DIM];
    for c in 0..REAL_DIM {
        centroid[c] = (0..n_rules)
            .map(|i| f64::from(rows[i * REAL_DIM + c]))
            .sum::<f64>()
            / n_rules as f64;
    }
    let mut ranked: Vec<(f64, usize)> = (0..n_rules)
        .map(|i| {
            let d2: f64 = (0..REAL_DIM)
                .map(|c| {
                    let d = f64::from(rows[i * REAL_DIM + c]) - centroid[c];
                    d * d
                })
                .sum();
            (d2, i)
        })
        .collect();
    ranked.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("finite"));
    let core: Vec<u32> = ranked[..64].iter().map(|&(_, i)| i as u32).collect();
    let d_core = cond_id(&scratch, &core).expect("core estimate");
    println!("G5 real shape: cond_id(dense core 64) = {d_core:.3}");

    // The incumbent calibration on the real shape (2-class core/rest
    // labels; one reading each).
    let mut labels = vec![1usize; n_rules];
    for &c in &core {
        labels[c as usize] = 0;
    }
    let eff = within_class_effective_rank(&rows, REAL_DIM, &labels);
    println!("G5 real shape: within_class_effective_rank(core vs rest) = {eff:.2}");
    let (pr, nr) = spectral_incumbents(&rows, REAL_DIM);
    println!("G5 real shape: participation_ratio = {pr:.2}, numerical_rank = {nr}");
}
