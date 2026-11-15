#![cfg(all(feature = "estimators", feature = "sink_aware_attn"))]
//! Plan 638 T2.7 — the G5 incumbent kill-switch.
//!
//! **Arm 1 — the synthetic decision (the planted contrast):** group A is
//! an isotropic LINEAR 8-dim cloud; group B is a CURVED 4-dim manifold
//! (the image of R⁴ through 16 fixed seeded sinusoidal directions — an
//! immersion, so the local dimension stays 4 while the covariance SPREAD
//! spans up to 16 directions). Planted truth: A is higher-dimensional
//! than B even though B's linear spread is wider. `cond_id` must recover
//! the plant: |d̂_A − 8| ≤ 1.5, |d̂_B − 4| ≤ 1.5, d̂_A ≥ d̂_B + 1.
//! The incumbents' readings are the RECORD (printed, never asserted):
//! `within_class_effective_rank` reads SPREAD — on this shape it is
//! expected to order the groups the WRONG way round (B above A), which
//! is exactly the complementary-signal case; if it instead orders them
//! right, the pre-registered demotion clause of Plan 638 T2.7 fires and
//! the plan record adjudicates.
//!
//! **Arm 2 — the real-shape calibration (disclosed):** the frozen rule
//! embed (`../riir-refine/assets/rule_embed_frozen_v1.bin`, BLAKE3-
//! verified before parse; the integrity contract stays refine-side
//! `decode_verified`). 268 rules × 64 dims, v1 layout (magic | ver |
//! buckets | dim | tau | n_rules | span_scale | span tower | rules |
//! blake3). No domain labels are derivable from the kebab-case rule ids,
//! so this arm records each estimator's reading on the real population —
//! the dense-core contrast (64 rows nearest the centroid vs the rest)
//! plus the full population — and NEVER a pass/fail on a label
//! contrast. Disclosed deviation: the Plan-638 prose calls this fixture
//! "8-D"; the shipped v1 artifact is 64-DIM — the gate reads the real
//! file (the plan's dimension note predates the shipped fixture).
//!
//! ⚠ Run at `--release` (O(n²·d) arms; the debug wall proves nothing).

#[path = "fixture_cond_geometry/mod.rs"]
mod fixture;

use fixture::Lcg;
use katgpt_core::data_probe::cond_geometry::{
    ball_counts, cond_id, BallCountScratch, MIN_GROUP_TOKENS,
};
use katgpt_core::data_probe::geometry::within_class_effective_rank;
use katgpt_core::subspace_phase_gate::{
    numerical_rank, participation_ratio, thin_svd_into, SvdResultScratch, SvdScratch,
};

const GROUP: usize = 128;
const SPACE_D: usize = 32;
const TAU: f32 = 0.9;

// ── Arm 1 — the synthetic planted contrast ─────────────────────────────

/// Group A: isotropic LINEAR cloud in 8 dims (spread dim = intrinsic dim).
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
/// fixed seeded sinusoidal directions (an immersion: local dim stays 4;
/// linear spread spans up to 16 directions).
fn curved_manifold(seed: u64) -> Vec<f32> {
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
            row[c] = dot.sin() as f32;
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
fn g5_planted_contrast_and_incumbent_record() {
    const {
        assert!(GROUP >= MIN_GROUP_TOKENS);
    }

    // ── the planted contrast, cond_id's claim ──
    let a = linear_cloud(0x638_5001);
    let b = curved_manifold(0x638_5002);
    let d_a = cond_id_of_group(&a);
    let d_b = cond_id_of_group(&b);
    println!("G5 cond_id: d̂(linear-8) = {d_a:.3}   d̂(curved-4) = {d_b:.3}");
    assert!(
        (d_a as f64 - 8.0).abs() <= 1.5,
        "G5 FAILED: cond_id missed the linear-8 plant (d̂_A = {d_a})"
    );
    assert!(
        (d_b as f64 - 4.0).abs() <= 1.5,
        "G5 FAILED: cond_id missed the curved-4 plant (d̂_B = {d_b})"
    );
    assert!(
        d_a >= d_b + 1.0,
        "G5 FAILED: cond_id does not separate the planted contrast (Δ = {})",
        d_a - d_b
    );

    // ── the incumbent record (printed, never asserted) ──
    let mut combined = a.clone();
    combined.extend_from_slice(&b);
    let labels: Vec<usize> = (0..GROUP).map(|_| 0).chain((0..GROUP).map(|_| 1)).collect();
    let eff = within_class_effective_rank(&combined, SPACE_D, &labels);
    let (pr_a, nr_a) = spectral_incumbents(&a, SPACE_D);
    let (pr_b, nr_b) = spectral_incumbents(&b, SPACE_D);
    println!(
        "G5 incumbents: within_class_effective_rank(A vs B labels) = {eff:.2} | participation_ratio A={pr_a:.2} B={pr_b:.2} | numerical_rank A={nr_a} B={nr_b}"
    );
    let incumbent_orders_right = pr_a > pr_b && nr_a >= nr_b;
    println!(
        "G5 incumbent verdict: {} (the pre-registered demotion clause fires only on BOTH shapes ordering right — see the plan record)",
        if incumbent_orders_right {
            "ORDERS THE PLANT RIGHT — demotion-candidate reading"
        } else {
            "ORDERS THE PLANT WRONG (spread ≠ dimension) — the complementary-signal case"
        }
    );

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
