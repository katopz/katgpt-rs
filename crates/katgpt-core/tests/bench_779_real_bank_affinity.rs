//! Issue 779 T3 / Bench doc 767 — real-model (layer, activation, label)
//! bank affinity measurement over `katgpt_core::subspace_intervention`.
//!
//! Consumes the bank captured by the riir-ai example
//! `future_probe_bank_capture` (gemma-2-2b-it, 6 behavior classes × 8 topics
//! × 8 shells; train/test split holds out shells 6+7 so a template-token
//! shortcut cannot transfer). This test answers the T3 question:
//!
//! - does a measurable layer-affinity structure exist for behavior-intent
//!   probes in a REAL transformer residual stream (vs the planted synthetic
//!   bank of Issue 778)?
//! - if a peak emerges: which layer per class should `FutureBehaviorProbe`
//!   consumers read (the re-pin candidate)?
//! - if the curve is flat: the hand-picked layer is fine — a legitimate
//!   negative close (Issue 779 outcome criteria).
//!
//! Two-floor discipline (UQ "Report the Floor" rule): every accuracy row is
//! printed beside BOTH floors — uniform chance (1/classes) and the train
//! majority-class base rate. A layer row only "counts" above both.
//!
//! # Prerequisites
//!
//! ```sh
//! # 1. capture (riir-ai, ~35 min on M3 release):
//! cargo run --release --features latent_steering_bridge \
//!   --example future_probe_bank_capture            # writes bank779.bin/.json
//! # 2. analyze (katgpt-rs, release — the per-layer SVD is heavy):
//! BANK779_BIN=…/bank779.bin BANK779_JSON=…/bank779.json \
//!   cargo test --release -p katgpt-core \
//!   --test bench_779_real_bank_affinity -- --ignored --nocapture
//! ```
//!
//! Assertions are instrument-liveness only (shapes, sweep bounds, floors
//! printed) — the affinity VERDICT is a measurement, recorded in the bench
//! doc, never hard-asserted (a flat curve is a legitimate outcome).

#![cfg(not(target_arch = "wasm32"))]

use katgpt_core::subspace_intervention::{
    InterventionScratch, affinity_sweep, eval_head_into, ridge_probe_fit_into, three_arm_eval,
};

const MAGIC: [u8; 4] = *b"BK77";
/// λ = lambda_scale · mean(diag(XᵀX)) — the module convention. The sweep
/// reads the same bank at four regularization strengths (span-stability axis
/// from R557: conclusions must not hang on one λ).
const LAMBDAS: [f32; 4] = [0.003, 0.01, 0.03, 0.1];
/// Projection ranks for the three-arm contrast at the best layer. The last
/// entry equals the probe rank cap (`classes.min(d)` = 6) — the projection-
/// identity point (aligned@rank == full).
const KS: [usize; 4] = [1, 2, 4, 6];

struct Bank {
    n_total: usize,
    n_layers: usize,
    d: usize,
    classes: usize,
    acts: Vec<f32>,
    labels: Vec<usize>,
    train_idx: Vec<usize>,
    test_idx: Vec<usize>,
    class_names: Vec<String>,
}

fn load_bank() -> Bank {
    let bin_path = std::env::var("BANK779_BIN")
        .unwrap_or_else(|_| panic!("set BANK779_BIN to the captured bank (see module doc)"));
    let json_path = std::env::var("BANK779_JSON")
        .unwrap_or_else(|_| panic!("set BANK779_JSON to the capture meta (see module doc)"));
    let bytes = std::fs::read(&bin_path).unwrap_or_else(|e| panic!("cannot read {bin_path}: {e}"));
    assert!(bytes.len() > 24, "bank file too small: {}", bytes.len());
    assert_eq!(&bytes[0..4], &MAGIC, "bad bank magic");
    let rd_u32 = |off: usize| u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as usize;
    let version = rd_u32(4);
    assert_eq!(version, 1, "unsupported bank version {version}");
    let n_total = rd_u32(8);
    let n_layers = rd_u32(12);
    let d = rd_u32(16);
    let classes = rd_u32(20);
    let acts_off = 24;
    let acts_bytes = n_layers * n_total * d * 4;
    assert_eq!(
        bytes.len(),
        acts_off + acts_bytes + n_total,
        "bank length mismatch (header {n_total}/{n_layers}/{d}/{classes})"
    );
    let mut acts = vec![0.0_f32; n_layers * n_total * d];
    for (i, v) in acts.iter_mut().enumerate() {
        *v = f32::from_le_bytes(
            bytes[acts_off + i * 4..acts_off + i * 4 + 4]
                .try_into()
                .unwrap(),
        );
    }
    let labels: Vec<usize> = bytes[acts_off + acts_bytes..]
        .iter()
        .map(|&b| b as usize)
        .collect();
    assert_eq!(labels.len(), n_total);

    let meta: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&json_path)
            .unwrap_or_else(|e| panic!("cannot read {json_path}: {e}")),
    )
    .expect("meta json parse");
    let arr = |key: &str| -> Vec<usize> {
        meta[key]
            .as_array()
            .unwrap_or_else(|| panic!("meta missing {key}"))
            .iter()
            .map(|v| v.as_u64().expect("u64 idx") as usize)
            .collect()
    };
    let class_names: Vec<String> = meta["classes"]
        .as_array()
        .expect("meta classes")
        .iter()
        .map(|v| v.as_str().expect("class str").to_string())
        .collect();
    assert_eq!(class_names.len(), classes, "meta/bin class count mismatch");
    assert_eq!(meta["n_total"].as_u64().unwrap() as usize, n_total);
    assert_eq!(meta["n_layers"].as_u64().unwrap() as usize, n_layers);
    assert_eq!(meta["d"].as_u64().unwrap() as usize, d);

    Bank {
        n_total,
        n_layers,
        d,
        classes,
        acts,
        labels,
        train_idx: arr("train_idx"),
        test_idx: arr("test_idx"),
        class_names,
    }
}

/// Both floors: uniform chance and the train majority-class base rate.
fn floors(bank: &Bank) -> (f32, f32) {
    let chance = 1.0 / bank.classes as f32;
    let mut counts = vec![0_usize; bank.classes];
    for &i in &bank.train_idx {
        counts[bank.labels[i]] += 1;
    }
    let majority =
        counts.iter().copied().max().unwrap_or(0) as f32 / bank.train_idx.len().max(1) as f32;
    (chance, majority)
}

#[test]
#[ignore = "requires the captured real-model bank (run riir-ai example future_probe_bank_capture — see module doc / Bench 767)"]
fn real_bank_layer_affinity_measurement() {
    let bank = load_bank();
    let (chance, majority) = floors(&bank);
    println!(
        "[i779] bank: {} samples, {} layers, d={}, {} classes | floors: chance={chance:.3} train-majority={majority:.3}",
        bank.n_total, bank.n_layers, bank.d, bank.classes
    );
    println!(
        "[i779] train={} test={}",
        bank.train_idx.len(),
        bank.test_idx.len()
    );

    let n_test = bank.test_idx.len();
    let n_train = bank.train_idx.len();
    let mut scratch = InterventionScratch::new(bank.d, bank.classes, n_train, n_test);

    let mut best_layer = 0_usize;
    let mut best_acc = f32::NEG_INFINITY;
    let mut best_lambda = LAMBDAS[0];

    for &lambda in &LAMBDAS {
        let mut full_acc = vec![0.0_f32; bank.n_layers];
        let mut recall = vec![0.0_f32; bank.n_layers * bank.classes];
        let mut peaks = vec![0_usize; bank.classes];
        let mut layer_buf = vec![0.0_f32; bank.classes];
        let best = affinity_sweep(
            &bank.acts,
            &bank.labels,
            &bank.train_idx,
            &bank.test_idx,
            bank.n_total,
            bank.d,
            bank.classes,
            bank.n_layers,
            lambda,
            &mut scratch,
            &mut full_acc,
            &mut recall,
            &mut peaks,
            &mut layer_buf,
        );
        println!("\n[i779] affinity sweep @ lambda_scale={lambda}: best layer {best}");
        println!("[i779] layer | acc | vs floors");
        for (l, acc) in full_acc.iter().enumerate() {
            let mark = if *acc > chance.max(majority) {
                "*"
            } else {
                " "
            };
            println!(
                "[i779]  L{l:02}{mark}    | {:.3} | {}",
                acc,
                if *acc > chance.max(majority) {
                    "ABOVE floors"
                } else {
                    "below"
                }
            );
        }
        println!("[i779] per-class peak layers @lambda={lambda}:");
        for (c, name) in bank.class_names.iter().enumerate() {
            let l = peaks[c];
            println!(
                "[i779]   {name:<10} L{l:02} (recall {:.3})",
                recall[l * bank.classes + c]
            );
        }
        if full_acc[best] > best_acc {
            best_acc = full_acc[best];
            best_layer = best;
            best_lambda = lambda;
        }
    }

    println!(
        "\n[i779] BEST layer {best_layer} acc {best_acc:.3} @lambda={best_lambda} (chance {chance:.3}, majority {majority:.3})"
    );

    // ── Three-arm contrast at the best layer (frozen probe head) ──────────
    let y_train: Vec<usize> = bank.train_idx.iter().map(|&i| bank.labels[i]).collect();
    let y_test: Vec<usize> = bank.test_idx.iter().map(|&i| bank.labels[i]).collect();
    let mut x_train = vec![0.0_f32; n_train * bank.d];
    let mut x_test = vec![0.0_f32; n_test * bank.d];
    let layer =
        &bank.acts[best_layer * bank.n_total * bank.d..(best_layer + 1) * bank.n_total * bank.d];
    for (r, &i) in bank.train_idx.iter().enumerate() {
        x_train[r * bank.d..(r + 1) * bank.d].copy_from_slice(&layer[i * bank.d..(i + 1) * bank.d]);
    }
    for (r, &i) in bank.test_idx.iter().enumerate() {
        x_test[r * bank.d..(r + 1) * bank.d].copy_from_slice(&layer[i * bank.d..(i + 1) * bank.d]);
    }
    let mut w = vec![0.0_f32; bank.classes * bank.d];
    ridge_probe_fit_into(
        &x_train,
        &y_train,
        n_train,
        bank.d,
        bank.classes,
        best_lambda,
        &mut scratch,
        &mut w,
    );
    let mut recall_buf = vec![0.0_f32; bank.classes];
    let mut eval_hits = vec![0_usize; bank.classes];
    let mut eval_total = vec![0_usize; bank.classes];
    let full = eval_head_into(
        &x_test,
        &y_test,
        n_test,
        bank.d,
        &w,
        bank.classes,
        &mut recall_buf,
        &mut eval_hits,
        &mut eval_total,
    );
    println!("[i779] three-arm at L{best_layer}: full acc {full:.3}");

    let mut aligned = vec![0.0_f32; KS.len()];
    let mut random = vec![0.0_f32; KS.len()];
    let mut residual = vec![0.0_f32; KS.len()];
    let mut tri_recall = vec![0.0_f32; bank.classes];
    let rank = three_arm_eval(
        &x_test,
        &y_test,
        n_test,
        bank.d,
        &w,
        bank.classes,
        &KS,
        0x5EED_0779,
        &mut scratch,
        &mut aligned,
        &mut random,
        &mut residual,
        &mut tri_recall,
    );
    println!("[i779] probe rank {rank}; k | aligned | random | residual");
    for (ki, &k) in KS.iter().enumerate() {
        println!(
            "[i779]   k={k:02} | {:.3} | {:.3} | {:.3}",
            aligned[ki], random[ki], residual[ki]
        );
    }

    // ── Instrument-liveness assertions (NOT outcome asserts) ──────────────
    assert!(
        best_layer < bank.n_layers,
        "sweep returned an out-of-range layer"
    );
    assert!(
        (0.0..=1.0).contains(&full),
        "full-head accuracy out of [0,1]: {full}"
    );
    for a in &aligned {
        assert!((0.0..=1.0).contains(a), "aligned arm out of [0,1]: {a}");
    }
    // Projection identity (mathematical, not an outcome): at k == rank the
    // aligned projection is lossless for the head — must equal `full`.
    assert!(
        (aligned.last().copied().unwrap_or(0.0) - full).abs() < 1e-6,
        "projection identity failed: aligned@rank {} vs full {full}",
        aligned.last().copied().unwrap_or(0.0)
    );
}
