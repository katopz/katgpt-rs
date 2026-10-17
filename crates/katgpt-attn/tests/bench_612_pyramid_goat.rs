//! Plan 612 T2.2–T2.5 — the PISA pyramid selection GOAT gate (REAL tensors).
//!
//! Replays REAL captured qwen38-27B full-attention-layer Q/K tensors
//! (`tests/fixtures/pyramid_612/`, schema `pyramid_612_capture_v1`;
//! Ternary-Bonsai blocked at Issue 908 — qwen38 is the secondary) and
//! answers the slot's load-bearing question: does pyramid-LSE selection
//! hold single-level-LSE quality at a log-linear selection-latency slope?
//!
//! # Inputs
//!
//! - Committed subset (always runs): 2 FA layers (3, 63) × 2 kv heads
//!   (0, 3) × 4 source lengths (4096/16384/32768/65536) × 5 query
//!   positions. L > 4096 bins are stride-subsampled to ~4097 rows — the
//!   SEQUENCE the selector sees is ~4K real post-RoPE keys (disclosed:
//!   stride decorrelates block neighbours vs the true contiguous set).
//! - Full set (env `PYRAMID_612_FULL_DIR`, every file BLAKE3-verified
//!   against the manifest): all 4 kv heads × both layers at TRUE
//!   contiguous lengths — the length axis (16K/32K/64K) and every 32K+
//!   claim live here. Skip-loud when absent, never a green zero.
//!
//! # Protocol (per family = layer × kv head × source length × q position)
//!
//! The query at position p sees the causal PREFIX K[0..=p] only — the
//! selector, the reference, and the forced-block policy all operate on
//! the truncated sequence. Reference = full softmax attention mass per
//! q-head (std `exp`), summed over the GQA group sharing the kv head
//! (kv = q_head·n_kv/n_head, the kernel's own mapping), accumulated per
//! C=64 block. True top-K = argmax block mass. All arms get the same
//! group-summed query u, the same K budget, and the same NSA forced
//! leaves (first/previous/current):
//! - `single_mean`  — full-scan leaf scoring at the Mean rung (BSA class)
//! - `single_lse`   — full-scan leaf scoring at the ExactLse rung
//! - `pyramid_*`    — [`coarse_to_fine_select`] at all three rungs
//!
//! Full-scan leaf scores derive from ONE pass of u·K logits (dot
//! linearity: mean-of-logits == dot(u, block mean)), so single-level and
//! pyramid arms consume IDENTICAL leaf evidence — the comparison is
//! purely full scan vs coarse-to-fine walk.
//!
//! # Pins (theorems, asserted per family)
//!
//! - Jensen: `mean + ln(cnt) ≤ true LSE` per block (the ladder's
//!   ordering — catches a ln_z-only leaf arm).
//! - Captured mass ≤ true-top-K mass for EVERY selector (argmax
//!   definition — catches a selector emitting candidate-array positions
//!   instead of leaf ids).
//!
//! # Pre-registered bars (the bench_397 HGA convention: empirical
//! comparisons are COMPUTED + PRINTED verdicts — a FAIL is a documented
//! negative and the feature stays opt-in; only theorems and the latency
//! complexity claims are hard-asserted)
//!
//! - ISO-QUALITY at 32K+: mean paired Recall@8 diff
//!   (pyramid_lse − single_lse) ≥ −0.01 over non-trivial families.
//!   **MEASURED 2026-09-30: FAIL at −0.133 (Bench 612, documented
//!   negative — the walk's per-level pruning loses recall as depth
//!   grows; the feature stays opt-in).**
//! - LATENCY (release-only; debug prints and defers): aggregate slope
//!   gap ≥ 0.5 (single-level aggregate = N queries × O(N) scan → slope
//!   ≈2; pyramid aggregate → slope ≈1) AND pyramid_lse median per-query
//!   < single_lse median at N = 65 536.
//! - G4 lives in `bench_612_alloc_check` (its own binary — a global
//!   counting allocator cannot share a binary with parallel tests).
//! - Canary: the real-tensor position-0 family (prefix = 1 key) selects
//!   exactly `[0]` — the N ≤ C degenerate contract on live data.
//!
//! The paper's 90.95% Recall@8 is THEIR checkpoint's number and gates
//! nothing here — every quality figure is measured on OUR tensors
//! (Research 595 §3, never quoted).

#![cfg(feature = "pyramid_topk")]

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use katgpt_attn::dash_attn::block_topk::argtopk_with_scratch;
use katgpt_attn::dash_attn::pyramid_topk::{
    FORCED_LEAF_SLOTS, PYRAMID_BLOCK_SIZE, PyramidKeyHierarchy, PyramidLevels, PyramidScoreMode,
    PyramidScorer, PyramidScratch, coarse_to_fine_select,
};

const TOP_K: usize = 8;
/// Non-trivial family floor: enough leaves that K + forced ≠ everything.
const MIN_LEAVES: usize = 2 * TOP_K;

// ---------------------------------------------------------------------------
// Fixture loading
// ---------------------------------------------------------------------------

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pyramid_612")
}

fn manifest() -> serde_json::Value {
    let raw = fs::read_to_string(fixture_dir().join("manifest.json"))
        .expect("pyramid_612 manifest is committed — a missing manifest is a repo defect");
    serde_json::from_str(&raw).expect("pyramid_612 manifest parses")
}

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let (chunks, _) = bytes.as_chunks::<4>();
    chunks.iter().map(|c| f32::from_le_bytes(*c)).collect()
}

fn read_u32(path: &Path) -> Vec<u32> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let (chunks, _) = bytes.as_chunks::<4>();
    chunks.iter().map(|c| u32::from_le_bytes(*c)).collect()
}

fn verify_blake3(path: &Path, expected: &str) {
    let bytes =
        fs::read(path).unwrap_or_else(|e| panic!("read {} for BLAKE3 verify: {e}", path.display()));
    let got = blake3::hash(&bytes).to_hex().to_string();
    assert_eq!(
        got,
        expected,
        "BLAKE3 mismatch for {} — fixture corruption",
        path.display()
    );
}

/// Map a manifest-relative path ("commit/x" / "full/x") onto a real dir.
fn resolve(rel: &str, full_dir: Option<&Path>) -> PathBuf {
    let name = rel
        .trim_start_matches("commit/")
        .trim_start_matches("full/");
    match rel.split('/').next() {
        Some("full") => {
            let dir = full_dir.unwrap_or_else(|| {
                panic!("manifest names a full/ file but no full dir is configured: {rel}")
            });
            dir.join(name)
        }
        _ => fixture_dir().join(name),
    }
}

struct KBin {
    layer: usize,
    head: usize,
    /// Rows in THIS bin (committed subset ≈ 4 096/4 097; full = true length).
    rows: usize,
    /// The SOURCE sequence length L (distinguishes the four committed
    /// stride geometries; == rows for full bins).
    src_l: usize,
    data: Vec<f32>,
    /// Committed subsets carry their exact row set (original positions,
    /// ascending). `None` = contiguous 0..rows.
    positions: Option<Vec<u32>>,
}

struct QMat {
    layer: usize,
    position: usize,
    #[allow(dead_code)] // shape witness; asserted at load, kept for report completeness
    heads: usize,
    data: Vec<f32>,
}

fn load_k_bins(m: &serde_json::Value, full_dir: Option<&Path>, use_full: bool) -> Vec<KBin> {
    let key = if use_full { "k_full" } else { "k_commit" };
    let mut out = Vec::new();
    for len in m["lengths"].as_array().expect("lengths") {
        let src_l = len["L"].as_u64().unwrap() as usize;
        for e in len[key].as_array().expect("k entries per length") {
            let path = resolve(e["file"].as_str().unwrap(), full_dir);
            verify_blake3(&path, e["blake3"].as_str().unwrap());
            let data = read_f32(&path);
            let row_len = e["row_len_f32"].as_u64().unwrap() as usize;
            let rows = e["rows"].as_u64().unwrap() as usize;
            assert_eq!(
                data.len(),
                rows * row_len,
                "bin shape for {}",
                path.display()
            );
            let positions = if use_full {
                None
            } else {
                let ppath = resolve(e["positions_file"].as_str().unwrap(), full_dir);
                verify_blake3(&ppath, len["commit"]["positions_blake3"].as_str().unwrap());
                let p = read_u32(&ppath);
                assert_eq!(p.len(), rows, "positions table row count");
                Some(p)
            };
            out.push(KBin {
                layer: e["layer"].as_u64().unwrap() as usize,
                head: e["head"].as_u64().unwrap() as usize,
                rows,
                src_l,
                data,
                positions,
            });
        }
    }
    out.shrink_to_fit();
    out
}

fn load_q(m: &serde_json::Value, full_dir: Option<&Path>, use_full: bool) -> Vec<QMat> {
    let arr = if use_full {
        m["q_full"].as_array().expect("q_full")
    } else {
        m["q_commit"].as_array().expect("q_commit")
    };
    let mut out = Vec::new();
    for e in arr {
        let path = resolve(e["file"].as_str().unwrap(), full_dir);
        verify_blake3(&path, e["blake3"].as_str().unwrap());
        let data = read_f32(&path);
        let heads = e["heads"].as_u64().unwrap() as usize;
        let row_len = e["row_len_f32"].as_u64().unwrap() as usize;
        assert_eq!(data.len(), heads * row_len);
        out.push(QMat {
            layer: e["layer"].as_u64().unwrap() as usize,
            position: e["position"].as_u64().unwrap() as usize,
            heads,
            data,
        });
    }
    out
}

fn full_dir() -> Option<PathBuf> {
    std::env::var_os("PYRAMID_612_FULL_DIR").map(PathBuf::from)
}

// ---------------------------------------------------------------------------
// Selection math (reference + single-level arms + metrics)
// ---------------------------------------------------------------------------

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

/// NSA forced leaves (first / previous / current), deduped, clamped — the
/// lib's `ForcedBlocks::at` mirrored for the single-level arms.
fn forced_set(query_idx: usize, n_leaves: usize) -> Vec<usize> {
    if n_leaves == 0 {
        return Vec::new();
    }
    let cur = (query_idx / PYRAMID_BLOCK_SIZE).min(n_leaves - 1);
    let mut v = Vec::with_capacity(FORCED_LEAF_SLOTS);
    for cand in [0, cur.saturating_sub(1), cur] {
        if !v.contains(&cand) {
            v.push(cand);
        }
    }
    v.shrink_to_fit();
    v
}

/// Single-level full scan over per-leaf scores: argtopk-K of the non-forced
/// leaves ∪ forced (the lib's slot discipline, mirrored).
fn single_level_select(scores: &[f32], forced: &[usize]) -> Vec<usize> {
    let n_leaves = scores.len();
    let mut nonforced: Vec<usize> = Vec::with_capacity(n_leaves);
    let mut nonforced_scores: Vec<f32> = Vec::with_capacity(n_leaves);
    for (b, &s) in scores.iter().enumerate() {
        if !forced.contains(&b) {
            nonforced.push(b);
            nonforced_scores.push(s);
        }
    }
    let k = TOP_K.min(nonforced_scores.len());
    let mut idx = Vec::new();
    let mut pairs = Vec::new();
    argtopk_with_scratch(&nonforced_scores, k, &mut idx, &mut pairs);
    let mut out: Vec<usize> = idx[..k].iter().map(|&i| nonforced[i]).collect();
    for &f in forced {
        if !out.contains(&f) {
            out.push(f);
        }
    }
    out.sort_unstable();
    out
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Arm {
    SingleMean,
    SingleLse,
    PyramidMean,
    PyramidHalfVar,
    PyramidLse,
}

const ARMS: [Arm; 5] = [
    Arm::SingleMean,
    Arm::SingleLse,
    Arm::PyramidMean,
    Arm::PyramidHalfVar,
    Arm::PyramidLse,
];

impl Arm {
    fn label(self) -> &'static str {
        match self {
            Arm::SingleMean => "single_mean",
            Arm::SingleLse => "single_lse",
            Arm::PyramidMean => "pyramid_mean",
            Arm::PyramidHalfVar => "pyramid_halfvar",
            Arm::PyramidLse => "pyramid_lse",
        }
    }
}

struct FamilyOutcome {
    layer: usize,
    head: usize,
    /// SOURCE length L (== the true N for full bins).
    length: usize,
    #[allow(dead_code)] // provenance; grouped tables key on layer/head/L
    position: usize,
    n_leaves: usize,
    recall: [f32; 5],
    mass_ratio: [f32; 5],
}

struct EvalCtx<'a> {
    head_dim: usize,
    n_kv_head: usize,
    n_head: usize,
    scale: f32,
    storage: &'a mut [f32],
}

/// Evaluate ONE family: reference mass, true top-K, all arms, both metrics.
/// Asserts the two per-family theorem pins inline (Jensen + captured mass).
fn eval_family(
    ctx: &mut EvalCtx<'_>,
    kbin: &KBin,
    prefix_rows: usize,
    qmat: &QMat,
    query_idx: usize,
) -> FamilyOutcome {
    let d = ctx.head_dim;
    let keys = &kbin.data[..prefix_rows * d];
    let n_leaves = prefix_rows.div_ceil(PYRAMID_BLOCK_SIZE);
    let group = ctx.n_head / ctx.n_kv_head;
    let q0 = kbin.head * group;

    // -- reference: per-q-head softmax mass, group-summed per block -------
    let mut block_mass = vec![0.0f32; n_leaves];
    let mut u = vec![0.0f32; d];
    for h in 0..group {
        let q = &qmat.data[(q0 + h) * d..(q0 + h + 1) * d];
        let mut logits = vec![0.0f32; prefix_rows];
        let mut mx = f32::NEG_INFINITY;
        for j in 0..prefix_rows {
            let l = dot(q, &keys[j * d..(j + 1) * d]) * ctx.scale;
            logits[j] = l;
            if l > mx {
                mx = l;
            }
        }
        let mut z = 0.0f32;
        for l in logits.iter_mut() {
            *l = (*l - mx).exp();
            z += *l;
        }
        for (j, &p) in logits.iter().enumerate() {
            block_mass[j / PYRAMID_BLOCK_SIZE] += p / z;
        }
        for (uv, &qv) in u.iter_mut().zip(q.iter()) {
            *uv += qv;
        }
    }

    // u logits: ONE pass, the shared leaf evidence for every arm.
    let mut u_logits = vec![0.0f32; prefix_rows];
    for j in 0..prefix_rows {
        u_logits[j] = dot(&u, &keys[j * d..(j + 1) * d]) * ctx.scale;
    }

    // Per-leaf ladder values + the Jensen pin (theorem).
    let mut mean_score = vec![0.0f32; n_leaves];
    let mut lse_score = vec![0.0f32; n_leaves];
    for b in 0..n_leaves {
        let base = b * PYRAMID_BLOCK_SIZE;
        let cnt = (prefix_rows - base).min(PYRAMID_BLOCK_SIZE);
        let lg = &u_logits[base..base + cnt];
        let mut s = 0.0f32;
        for &x in lg {
            s += x;
        }
        let mean = s / cnt as f32;
        let mx = lg.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut se = 0.0f32;
        for &l in lg {
            se += (l - mx).exp();
        }
        let lse = mx + se.ln();
        mean_score[b] = mean;
        lse_score[b] = lse;
        assert!(
            mean + (cnt as f32).ln() <= lse + 1e-3 + 1e-4 * lse.abs(),
            "Jensen pin violated: L{}/layer{}/head{} leaf {b}: mean+ln(c)={:.4} > lse={:.4}",
            prefix_rows,
            kbin.layer,
            kbin.head,
            mean + (cnt as f32).ln(),
            lse
        );
    }

    // True top-K by reference mass, plus the per-size mass bound (a
    // selector's budget is K + FORCED leaves, so the argmax bound for an
    // m-block selection is the top-m mass — the theorem, stated exactly).
    let mut truth_idx = Vec::new();
    let mut truth_pairs = Vec::new();
    argtopk_with_scratch(
        &block_mass,
        TOP_K.min(n_leaves),
        &mut truth_idx,
        &mut truth_pairs,
    );
    let truth: Vec<usize> = truth_idx.to_vec();
    let mut mass_desc = block_mass.clone();
    mass_desc.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let mut mass_prefix = Vec::with_capacity(mass_desc.len() + 1);
    mass_prefix.push(0.0f32);
    for &mval in &mass_desc {
        mass_prefix.push(*mass_prefix.last().unwrap() + mval);
    }
    let mass_bound = |m: usize| mass_prefix[m.min(mass_prefix.len() - 1)];

    let forced = forced_set(query_idx, n_leaves);

    // -- arms ---------------------------------------------------------------
    let mut selections: [Vec<usize>; 5] = Default::default();
    selections[0] = single_level_select(&mean_score, &forced);
    selections[1] = single_level_select(&lse_score, &forced);

    let need = PyramidKeyHierarchy::required_len(prefix_rows, d);
    let hier = PyramidKeyHierarchy::build(keys, prefix_rows, d, &mut ctx.storage[..need]);
    let mut scratch = PyramidScratch::new();
    for (slot, mode) in [
        (2usize, PyramidScoreMode::Mean),
        (3, PyramidScoreMode::MeanPlusHalfVar),
        (4, PyramidScoreMode::ExactLse),
    ] {
        let scorer = PyramidScorer {
            mode,
            scale: ctx.scale,
        };
        let heads: Vec<&[f32]> = (0..group)
            .map(|h| &qmat.data[(q0 + h) * d..(q0 + h + 1) * d])
            .collect();
        coarse_to_fine_select(&hier, keys, &heads, query_idx, TOP_K, scorer, &mut scratch);
        let mut out = scratch.out.clone();
        out.sort_unstable();
        assert!(
            out.len() <= TOP_K + FORCED_LEAF_SLOTS,
            "pyramid output shape: {out:?}"
        );
        assert!(
            out.windows(2).all(|w| w[0] < w[1]),
            "sorted+deduped: {out:?}"
        );
        for &f in &forced {
            assert!(
                out.contains(&f),
                "forced leaf {f} missing from pyramid {out:?}"
            );
        }
        selections[slot] = out;
    }

    let mut outcome = FamilyOutcome {
        layer: kbin.layer,
        head: kbin.head,
        length: kbin.src_l,
        position: query_idx,
        n_leaves,
        recall: [0.0; 5],
        mass_ratio: [0.0; 5],
    };
    for (slot, sel) in selections.iter().enumerate() {
        let hits = sel.iter().filter(|b| truth.contains(b)).count() as f32;
        outcome.recall[slot] = hits / truth.len() as f32;
        let mass: f32 = sel.iter().map(|&b| block_mass[b]).sum();
        let bound = mass_bound(sel.len());
        // PIN (argmax definition): no m-block selection captures more than
        // the true top-m — catches candidate-position-as-leaf-id outputs.
        assert!(
            mass <= bound * (1.0 + 1e-5) + 1e-6,
            "captured-mass pin violated ({}): mass {mass:.6} > top-{} mass {bound:.6}",
            ARMS[slot].label(),
            sel.len()
        );
        outcome.mass_ratio[slot] = mass / bound;
    }
    outcome
}

/// Original position → row index in this bin (identity for contiguous bins).
fn row_of(kbin: &KBin, position: usize) -> usize {
    match &kbin.positions {
        None => position,
        Some(p) => match p.binary_search(&(position as u32)) {
            Ok(i) => i,
            Err(_) => panic!(
                "q position {position} not in the committed subset rows (L{})",
                kbin.src_l
            ),
        },
    }
}

fn q_positions_for(m: &serde_json::Value, src_l: usize) -> Vec<usize> {
    for len in m["lengths"].as_array().unwrap() {
        if len["L"].as_u64().unwrap() as usize == src_l {
            return len["q_positions"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
        }
    }
    panic!("no manifest length entry for L={src_l}")
}

// ---------------------------------------------------------------------------
// G1 — selection quality on real tensors
// ---------------------------------------------------------------------------

fn run_quality(use_full: bool) -> Vec<FamilyOutcome> {
    let m = manifest();
    let fd = full_dir();
    let fd = if use_full {
        let d = fd.unwrap_or_else(|| panic!("PYRAMID_612_FULL_DIR not set"));
        assert!(
            d.is_dir(),
            "PYRAMID_612_FULL_DIR={} is not a dir",
            d.display()
        );
        Some(d)
    } else {
        None
    };
    let head_dim = m["config"]["head_dim"].as_u64().unwrap() as usize;
    let n_head = m["config"]["n_head"].as_u64().unwrap() as usize;
    let n_kv_head = m["config"]["n_kv_head"].as_u64().unwrap() as usize;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let kbins = load_k_bins(&m, fd.as_deref(), use_full);
    let qs = load_q(&m, fd.as_deref(), use_full);
    println!(
        "[612] quality source={} k_bins={} q_mats={}",
        if use_full { "full" } else { "commit" },
        kbins.len(),
        qs.len()
    );

    let max_rows = kbins.iter().map(|k| k.rows).max().unwrap();
    let mut storage = vec![0.0f32; PyramidKeyHierarchy::required_len(max_rows, head_dim) + 8];

    let mut out = Vec::with_capacity(kbins.len());
    for kbin in &kbins {
        for &p in &q_positions_for(&m, kbin.src_l) {
            let prefix = row_of(kbin, p) + 1;
            let qmat = qs
                .iter()
                .find(|q| q.layer == kbin.layer && q.position == p)
                .unwrap_or_else(|| panic!("no Q bin for layer {} pos {p}", kbin.layer));
            let mut ctx = EvalCtx {
                head_dim,
                n_kv_head,
                n_head,
                scale,
                storage: &mut storage,
            };
            out.push(eval_family(&mut ctx, kbin, prefix, qmat, prefix - 1));
        }
    }
    out.shrink_to_fit();
    out
}

fn print_report(families: &[FamilyOutcome], source: &str) {
    let mut keys: Vec<(usize, usize, usize)> = Vec::new();
    for f in families {
        let k = (f.layer, f.head, f.length);
        if !keys.contains(&k) {
            keys.push(k);
        }
    }
    keys.sort_unstable();
    println!(
        "\n[612] == G1 quality ({source}) — mean recall@{TOP_K}/mass-ratio per group (non-trivial families) =="
    );
    let header = ARMS
        .iter()
        .map(|a| format!("{:>11}", a.label()))
        .collect::<Vec<_>>()
        .join(" ");
    println!("[612] {:>11} | {header} | n", "layer/head/L");
    for (l, h, len) in &keys {
        let grp: Vec<&FamilyOutcome> = families
            .iter()
            .filter(|f| {
                f.layer == *l && f.head == *h && f.length == *len && f.n_leaves >= MIN_LEAVES
            })
            .collect();
        if grp.is_empty() {
            continue;
        }
        let cells: Vec<String> = (0..5)
            .map(|slot| {
                let r: f32 = grp.iter().map(|f| f.recall[slot]).sum::<f32>() / grp.len() as f32;
                let mr: f32 =
                    grp.iter().map(|f| f.mass_ratio[slot]).sum::<f32>() / grp.len() as f32;
                format!("{:5.3}/{:5.3}", r, mr)
            })
            .collect();
        println!(
            "[612] {:>4}/{:>2}/{:>5} | {} | {}",
            l,
            h,
            len,
            cells.join(" "),
            grp.len()
        );
    }
    let nt: Vec<&FamilyOutcome> = families
        .iter()
        .filter(|f| f.n_leaves >= MIN_LEAVES)
        .collect();
    if !nt.is_empty() {
        println!(
            "[612] overall over {} non-trivial families (leaves >= {MIN_LEAVES}):",
            nt.len()
        );
        for (slot, arm) in ARMS.iter().enumerate() {
            let r: f32 = nt.iter().map(|f| f.recall[slot]).sum::<f32>() / nt.len() as f32;
            let mr: f32 = nt.iter().map(|f| f.mass_ratio[slot]).sum::<f32>() / nt.len() as f32;
            println!(
                "[612]   {:>15} recall {:6.4}  mass-ratio {:6.4}",
                arm.label(),
                r,
                mr
            );
        }
        let diffs: Vec<f32> = nt.iter().map(|f| f.recall[4] - f.recall[1]).collect();
        let mean = diffs.iter().sum::<f32>() / diffs.len() as f32;
        let var = diffs.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / diffs.len() as f32;
        println!(
            "[612]   paired recall diff (pyramid_lse - single_lse): mean {mean:+.4} sd {:.4} n {}",
            var.sqrt(),
            diffs.len()
        );
    }
}

#[test]
fn g1_quality_committed_real_tensors() {
    let families = run_quality(false);
    assert!(
        !families.is_empty(),
        "committed fixture set is in-repo — an empty family set is a loader defect, not a pass"
    );
    print_report(&families, "commit");
}

#[test]
fn g1_quality_full_set_and_iso_quality_bar() {
    if full_dir().is_none() {
        println!(
            "[612] SKIP-LOUD: PYRAMID_612_FULL_DIR unset — the true-length quality axis and the 32K+ iso-quality bar are DEFERRED (committed-subset quality ran in its own test)"
        );
        return;
    }
    let families = run_quality(true);
    assert!(
        !families.is_empty(),
        "full set configured but empty — a loader defect"
    );
    print_report(&families, "full");

    let nt: Vec<&FamilyOutcome> = families
        .iter()
        .filter(|f| f.length >= 32768 && f.n_leaves >= MIN_LEAVES)
        .collect();
    assert!(
        !nt.is_empty(),
        "no non-trivial 32K+ families — the iso-quality read cannot be reported"
    );
    let mean: f32 = nt.iter().map(|f| f.recall[4] - f.recall[1]).sum::<f32>() / nt.len() as f32;
    let var = nt
        .iter()
        .map(|f| {
            let dv = f.recall[4] - f.recall[1] - mean;
            dv * dv
        })
        .sum::<f32>()
        / nt.len() as f32;
    // Verdict — COMPUTED + PRINTED, not hard-asserted (the bench_397 HGA
    // pattern): a FAIL is a documented negative, the feature stays opt-in,
    // and the promotion bar is re-run only on a deliberate change. The
    // per-family pins inside eval_family stay hard-asserted everywhere.
    let verdict = if mean >= -0.01 { "PASS" } else { "FAIL" };
    println!(
        "[612] ISO-QUALITY VERDICT: {verdict} — mean paired recall diff (pyramid_lse - single_lse) at L>=32K = {mean:+.4} (sd {:.4}) over {} families (bar: >= -0.01)",
        var.sqrt(),
        nt.len()
    );
    if verdict == "FAIL" {
        println!(
            "[612]   documented negative: the walk's per-level K=8 pruning loses more recall than the exact scan as depth grows — feature stays opt-in, no promotion (Bench 612)"
        );
    }
}

// ---------------------------------------------------------------------------
// G2 — selection-latency slope + crossover (release)
// ---------------------------------------------------------------------------

/// (N, pyramid_lse µs, single_lse µs, single_mean µs).
type LatRow = (usize, f64, f64, f64);

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

#[test]
fn g2_latency_slope_and_crossover() {
    if full_dir().is_none() {
        println!(
            "[612] SKIP-LOUD: PYRAMID_612_FULL_DIR unset — the latency-slope axis is DEFERRED (needs true-length tensors)"
        );
        return;
    }
    let release = cfg!(not(debug_assertions));
    if !release {
        println!(
            "[612] debug build: latency REPORT-ONLY — asserts are release-gated (a latency gate in debug measures an unoptimized binary)"
        );
    }

    let m = manifest();
    let fd = full_dir().unwrap();
    let head_dim = m["config"]["head_dim"].as_u64().unwrap() as usize;
    let n_head = m["config"]["n_head"].as_u64().unwrap() as usize;
    let n_kv_head = m["config"]["n_kv_head"].as_u64().unwrap() as usize;
    let group = n_head / n_kv_head;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let kbins = load_k_bins(&m, Some(&fd), true);
    let qs = load_q(&m, Some(&fd), true);
    let mut rows: Vec<LatRow> = Vec::new();
    for kbin in kbins.iter().filter(|k| k.layer == 3 && k.head == 0) {
        let prefix = kbin.rows;
        let d = head_dim;
        let keys = &kbin.data;
        let p_last = prefix - 1;
        let qmat = qs
            .iter()
            .find(|q| q.layer == 3 && q.position == p_last)
            .expect("deepest q position exists");
        let heads: Vec<&[f32]> = (0..group).map(|h| &qmat.data[h * d..(h + 1) * d]).collect();
        // Precomputed group-summed query (the quality path folds it inside
        // the selector; the scan arms need the same vector by hand).
        let mut u = vec![0.0f32; d];
        for h in &heads {
            for (uv, &qv) in u.iter_mut().zip(h.iter()) {
                *uv += qv;
            }
        }

        let mut storage = vec![0.0f32; PyramidKeyHierarchy::required_len(prefix, d) + 8];
        let hier = PyramidKeyHierarchy::build(keys, prefix, d, &mut storage);
        let mut scratch = PyramidScratch::new();
        let scorer = PyramidScorer {
            mode: PyramidScoreMode::ExactLse,
            scale,
        };
        let n_leaves = prefix.div_ceil(PYRAMID_BLOCK_SIZE);
        let leaf_means = hier.level_rows(0).to_vec();

        let reps = if release { 30 } else { 5 };
        let warm = 3;
        let mut t_pyramid = Vec::with_capacity(reps);
        let mut t_single_lse = Vec::with_capacity(reps);
        let mut t_single_mean = Vec::with_capacity(reps);
        let mut blackhole: f32 = 0.0;
        let mut lg = vec![0.0f32; PYRAMID_BLOCK_SIZE];
        for r in 0..reps + warm {
            // pyramid arm
            let t0 = Instant::now();
            blackhole +=
                coarse_to_fine_select(&hier, keys, &heads, p_last, TOP_K, scorer, &mut scratch)
                    as f32;
            let t1 = Instant::now();

            // single_lse arm: full scan — exact LSE per leaf over every key.
            let t2 = Instant::now();
            let mut acc = 0.0f32;
            for b in 0..n_leaves {
                let base = b * PYRAMID_BLOCK_SIZE;
                let cnt = (prefix - base).min(PYRAMID_BLOCK_SIZE);
                let mut mx = f32::NEG_INFINITY;
                for i in 0..cnt {
                    let l = dot(&u, &keys[(base + i) * d..(base + i + 1) * d]) * scale;
                    lg[i] = l;
                    if l > mx {
                        mx = l;
                    }
                }
                let mut se = 0.0f32;
                for &l in lg[..cnt].iter() {
                    se += (l - mx).exp();
                }
                acc += mx + se.ln();
            }
            let t3 = Instant::now();

            // single_mean arm: full scan over (precomputed) leaf means.
            let t4 = Instant::now();
            let mut acc2 = 0.0f32;
            for b in 0..n_leaves {
                acc2 += dot(&u, &leaf_means[b * d..(b + 1) * d]) * scale;
            }
            let t5 = Instant::now();
            blackhole += acc + acc2;

            if r >= warm {
                t_pyramid.push(t1.duration_since(t0).as_secs_f64() * 1e6);
                t_single_lse.push(t3.duration_since(t2).as_secs_f64() * 1e6);
                t_single_mean.push(t5.duration_since(t4).as_secs_f64() * 1e6);
            }
        }
        let _ = blackhole;
        rows.push((
            prefix,
            median(&mut t_pyramid.clone()),
            median(&mut t_single_lse.clone()),
            median(&mut t_single_mean.clone()),
        ));
    }
    rows.sort_by_key(|r| r.0);
    println!(
        "\n[612] == G2 per-query selection latency (µs, median of {} reps, release={release}) ==",
        if release { 30 } else { 5 }
    );
    println!(
        "[612] {:>7} {:>12} {:>12} {:>12}",
        "N", "pyramid_lse", "single_lse", "single_mean"
    );
    for (n, p, sl, sm) in &rows {
        println!("[612] {n:>7} {p:>12.1} {sl:>12.1} {sm:>12.1}");
    }

    // log2-log2 slopes of the PREFILL AGGREGATE (N queries × per-query).
    let fit = |sel: &dyn Fn(&LatRow) -> f64| -> f64 {
        let xs: Vec<f64> = rows.iter().map(|r| (r.0 as f64).log2()).collect();
        let ys: Vec<f64> = rows.iter().map(|r| (sel(r) * r.0 as f64).log2()).collect();
        let n = xs.len() as f64;
        let mx = xs.iter().sum::<f64>() / n;
        let my = ys.iter().sum::<f64>() / n;
        let sxy = xs
            .iter()
            .zip(ys.iter())
            .map(|(x, y)| (x - mx) * (y - my))
            .sum::<f64>();
        let sxx = xs.iter().map(|x| (x - mx) * (x - mx)).sum::<f64>();
        sxy / sxx
    };
    let slope_p = fit(&|r| r.1);
    let slope_sl = fit(&|r| r.2);
    let slope_sm = fit(&|r| r.3);
    println!(
        "[612] aggregate slope (t_query × N vs N, log-log): pyramid_lse {slope_p:.2}  single_lse {slope_sl:.2}  single_mean {slope_sm:.2} (claim: pyramid ≈ 1 vs single ≈ 2)"
    );
    if let Some(&(nmax, p, sl, _)) = rows.last() {
        println!(
            "[612] at N={nmax}: pyramid_lse {p:.1}µs vs single_lse {sl:.1}µs ({:.2}×)",
            sl / p
        );
        if release {
            assert!(
                p < sl,
                "G2 latency FAIL: pyramid_lse {p:.1}µs >= single_lse {sl:.1}µs at N={nmax} — document the negative"
            );
            assert!(
                slope_sl - slope_p >= 0.5,
                "G2 slope FAIL: single_lse {slope_sl:.2} - pyramid {slope_p:.2} < 0.5 — document the negative"
            );
        }
    }
    if !release {
        println!("[612] debug build: G2 latency asserts DEFERRED to release");
    }
}

// ---------------------------------------------------------------------------
// T2.5 — canary
// ---------------------------------------------------------------------------

#[test]
fn t25_canary_position_zero_degenerate() {
    let m = manifest();
    let head_dim = m["config"]["head_dim"].as_u64().unwrap() as usize;
    let n_head = m["config"]["n_head"].as_u64().unwrap() as usize;
    let n_kv_head = m["config"]["n_kv_head"].as_u64().unwrap() as usize;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let kbins = load_k_bins(&m, None, false);
    let qs = load_q(&m, None, false);
    // The position-0 family: prefix = 1 key — N ≤ C degenerate.
    let kbin = kbins.iter().find(|k| k.layer == 3 && k.head == 0).unwrap();
    let qmat = qs.iter().find(|q| q.layer == 3 && q.position == 0).unwrap();
    let mut storage = vec![0.0f32; PyramidKeyHierarchy::required_len(kbin.rows, head_dim) + 8];
    let mut ctx = EvalCtx {
        head_dim,
        n_kv_head,
        n_head,
        scale,
        storage: &mut storage,
    };
    let out = eval_family(&mut ctx, kbin, 1, qmat, 0);
    assert_eq!(
        out.n_leaves, 1,
        "position-0 family is the single-leaf degenerate"
    );
    for (slot, arm) in ARMS.iter().enumerate() {
        assert_eq!(
            out.recall[slot],
            1.0,
            "{} must select the only leaf in the degenerate family",
            arm.label()
        );
    }
    println!(
        "[612] T2.5 canary: position-0 real-tensor family — all arms select exactly [0] (N<=C contract on live data)"
    );
}
