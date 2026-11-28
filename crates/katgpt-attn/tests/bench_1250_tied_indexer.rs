//! Issue 1250 arm A — the YASA tied indexer under the Bench-612 block
//! contract (Research 698; the claim-lease record rides `.issues/1250`).
//!
//! Replays the SAME real captured qwen38-27B FA-layer Q/K tensors the
//! Bench-612 gate uses (`tests/fixtures/pyramid_612/`, schema
//! `pyramid_612_capture_v1`), same C=64 blocks, same top-8, same
//! softmax-mass reference — so the numbers are directly comparable to the
//! recorded 612 bars. This file never edits the 612 gate; it is its own
//! `[[test]]` target behind `tied_indexer`.
//!
//! # The two recall columns (never pooled)
//!
//! - **same-group** — each family's OWN group scores itself with the tied
//!   scorer (its heads, its kv head's keys): the scorer-quality column.
//! - **cross-group** — GROUP-0's tied selection evaluated against every
//!   OTHER group's reference top-8: YASA's actual deployment mechanism
//!   (one group's selection shared by all groups). The 612 harness never
//!   measured this; Research 698 §5 names it the honest prior's second
//!   axis (prior against us: Bench 612's coarse-first deficit AND the
//!   cross-group tie).
//!
//! # Pre-registered bars (i1250 T4 — computed + printed verdicts, the
//! bench_397 HGA convention; a FAIL is a documented negative, the feature
//! stays opt-in; the per-family mass pins stay hard-asserted)
//!
//! - **PASS** = paired diff (tied_same − single_lse) ≥ −0.01 at no more
//!   scoring cost than single_lse (Bench 612's ORIGINAL iso bar — NOT
//!   "beats −0.133": that bar only means "beat a failed arm").
//! - **Plan-filing** additionally requires the 95% lower bound of
//!   (tied − single_mean) > 0, PER LENGTH BIN, never pooled (single_mean
//!   is essentially BSA: 0.6475 vs single_lse 0.7793 full-set — beating
//!   −0.133 alone would re-measure BSA).
//! - The ARM verdict is the CONSERVATIVE read: the deployment shape is
//!   cross-group, so a same-group-only pass records
//!   "SAME-GROUP-ONLY — cross-group fails the bar" and closes the arm for
//!   league purposes.
//! - Everything else closes the arm negative (Research 698 grade struck,
//!   issue closed + demoted in the same commit).
//!
//! # Cost cell (T3 — measured, never inherited)
//!
//! The YASA 1/8 claim is THEIR geometry (12-of-48 heads × no value sum).
//! Ours: 6-of-24 heads × (N/64 block means) vs single_lse's (1 × N raw
//! keys). The cell times all three arms release-only, prints the analytic
//! dot-work ratio beside the measured medians, and the bar condition
//! (tied ≤ single_lse) is a computed verdict.
//!
//! # Skip laws
//!
//! - `PYRAMID_612_FULL_DIR` unset: the true-length bar axis is SKIP-LOUD
//!   (the committed subset still runs — never a green zero on the bar;
//!   the bar is DEFERRED, disclosed).
//! - Q/K are UNSCALED in the fixtures; the reference applies
//!   `1/sqrt(head_dim)` internally exactly as 612 does. The tied arm is
//!   unscaled by design (ReLU-mean needs no cross-key normalization; a
//!   positive scale is rank-invariant).

#![cfg(feature = "tied_indexer")]

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use katgpt_attn::dash_attn::block_topk::argtopk_with_scratch;
use katgpt_attn::dash_attn::tied_indexer::tied_indexer_block_scores_into;

/// The landed lib scorer, adapted to this harness's (heads, means, n, d,
/// out) call shape with the harness's own scale — one call site per arm.
/// Writes into the caller's slice through a scratch Vec (the lib API is
/// Vec-owned; the adaptation keeps the harness's `&mut [f32]` buffers).
fn tied_scores_into(
    heads: &[&[f32]],
    leaf_means: &[f32],
    n_leaves: usize,
    head_dim: usize,
    scale: f32,
    out: &mut [f32],
) {
    let mut buf: Vec<f32> = out.to_vec();
    tied_indexer_block_scores_into(leaf_means, n_leaves, head_dim, heads, scale, &mut buf);
    out[..n_leaves].copy_from_slice(&buf[..n_leaves]);
}

const TOP_K: usize = 8;
const BLOCK: usize = 64;
/// Non-trivial family floor (the 612 convention): enough leaves that
/// K + forced ≠ everything.
const MIN_LEAVES: usize = 2 * TOP_K;

// ---------------------------------------------------------------------------
// Fixture loading (the 612 loader pattern, self-contained — a [[test]]
// target shares no code with the 612 gate by the repo's test-independence
// convention)
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
    rows: usize,
    src_l: usize,
    data: Vec<f32>,
    /// Committed subsets carry their exact row set; `None` = contiguous.
    positions: Option<Vec<u32>>,
}

struct QMat {
    layer: usize,
    position: usize,
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
            data,
        });
    }
    out
}

fn full_dir() -> Option<PathBuf> {
    std::env::var_os("PYRAMID_612_FULL_DIR").map(PathBuf::from)
}

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
// Selection math (reference + reference-shape arms, all in-harness)
// ---------------------------------------------------------------------------

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

/// NSA forced leaves (first / previous / current), deduped, clamped —
/// the shared slot discipline.
fn forced_set(query_idx: usize, n_leaves: usize) -> Vec<usize> {
    if n_leaves == 0 {
        return Vec::new();
    }
    let cur = (query_idx / BLOCK).min(n_leaves - 1);
    let mut v = Vec::with_capacity(3);
    for cand in [0, cur.saturating_sub(1), cur] {
        if !v.contains(&cand) {
            v.push(cand);
        }
    }
    v.shrink_to_fit();
    v
}

fn select_topk(scores: &[f32], n_leaves: usize, forced: &[usize]) -> Vec<usize> {
    let mut nonforced: Vec<usize> = Vec::with_capacity(n_leaves);
    let mut nonforced_scores: Vec<f32> = Vec::with_capacity(n_leaves);
    for (b, &s) in scores[..n_leaves].iter().enumerate() {
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

/// A family's shared context — the params that stay fixed across the
/// reference/arms calls (the too-many-arguments clippy bound).
struct FamilyCtx<'a> {
    qmat: &'a QMat,
    group: usize,
    n_kv_head: usize,
    head_dim: usize,
    scale: f32,
}

/// Reference per-q-head softmax mass, group-summed per block; returns
/// (block_mass, group_summed_query u). Exactly the 612 reference.
fn reference_mass(
    ctx: &FamilyCtx<'_>,
    keys: &[f32],
    prefix_rows: usize,
    q_head_base: usize,
    u_out: &mut [f32],
) -> Vec<f32> {
    let head_dim = ctx.head_dim;
    let n_leaves = prefix_rows.div_ceil(BLOCK);
    let mut block_mass = vec![0.0f32; n_leaves];
    for v in u_out[..head_dim].iter_mut() {
        *v = 0.0;
    }
    for h in 0..ctx.group {
        let q = &ctx.qmat.data[(q_head_base + h) * head_dim..(q_head_base + h + 1) * head_dim];
        let mut logits = vec![0.0f32; prefix_rows];
        let mut mx = f32::NEG_INFINITY;
        for j in 0..prefix_rows {
            let l = dot(q, &keys[j * head_dim..(j + 1) * head_dim]) * ctx.scale;
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
            block_mass[j / BLOCK] += p / z;
        }
        for (uv, &qv) in u_out[..head_dim].iter_mut().zip(q.iter()) {
            *uv += qv;
        }
    }
    block_mass
}

/// Block mean keys into a reusable buffer (row-major n_leaves × d).
fn block_means_into(keys: &[f32], prefix_rows: usize, head_dim: usize, buf: &mut [f32]) -> usize {
    let n_leaves = prefix_rows.div_ceil(BLOCK);
    for b in 0..n_leaves {
        let base = b * BLOCK;
        let cnt = (prefix_rows - base).min(BLOCK);
        let cntf = cnt as f32;
        for i in 0..head_dim {
            let mut s = 0.0f32;
            for r in base..base + cnt {
                s += keys[r * head_dim + i];
            }
            buf[b * head_dim + i] = s / cntf;
        }
    }
    n_leaves
}

/// single_lse leaf scores (the exact-LSE reference arm) from u logits.
fn lse_leaf_scores(
    u: &[f32],
    keys: &[f32],
    prefix_rows: usize,
    head_dim: usize,
    scale: f32,
    scores: &mut [f32],
    logits: &mut [f32],
) -> usize {
    let n_leaves = prefix_rows.div_ceil(BLOCK);
    for (score, window) in scores
        .iter_mut()
        .take(n_leaves)
        .zip(keys.chunks(BLOCK * head_dim))
    {
        let cnt = window.len() / head_dim;
        let mut mx = f32::NEG_INFINITY;
        for (i, krow) in window.chunks(head_dim).enumerate() {
            let l = dot(u, krow) * scale;
            logits[i] = l;
            if l > mx {
                mx = l;
            }
        }
        let mut se = 0.0f32;
        for &l in logits[..cnt].iter() {
            se += (l - mx).exp();
        }
        *score = mx + se.ln();
    }
    n_leaves
}

/// single_mean leaf scores: dot(u, block mean) — the BSA-class arm.
fn mean_leaf_scores(
    u: &[f32],
    leaf_means: &[f32],
    _n_leaves: usize,
    head_dim: usize,
    scale: f32,
    scores: &mut [f32],
) {
    for (b, score) in scores.iter_mut().enumerate() {
        *score = dot(u, &leaf_means[b * head_dim..(b + 1) * head_dim]) * scale;
    }
}

struct FamilyOutcome {
    layer: usize,
    head: usize,
    length: usize,
    n_leaves: usize,
    /// Own-group reference recall for: tied_same, single_lse, single_mean.
    recall_same: [f32; 3],
    mass_ratio_same: [f32; 3],
    /// Cross-group: GROUP-0's tied selection vs THIS family's own truth,
    /// measured on families whose own group ≠ 0.
    recall_cross_mean: f32,
    /// The family's single_lse recall on the SAME truth — the paired
    /// baseline for the cross column's diff (never compare an absolute
    /// recall against a paired-diff bar).
    recall_cross_baseline: f32,
    /// Number of other groups measured (committed subset: 1 — head 3 only).
    cross_groups: usize,
}

/// Evaluate ONE family. `head` is the kv head of `kbin` (== the family's
/// own group). Cross-group needs group-0's KEYS: found among `kbins` by
/// (layer, head 0) at the same length.
fn eval_family(
    ctx: &FamilyCtx<'_>,
    kbin: &KBin,
    kbin_head0: Option<&KBin>,
    prefix_rows: usize,
    scratch_buf: &mut [f32],
) -> FamilyOutcome {
    let head_dim = ctx.head_dim;
    let group = ctx.group;
    let scale = ctx.scale;
    let qmat = ctx.qmat;
    let keys = &kbin.data[..prefix_rows * head_dim];
    let n_leaves = prefix_rows.div_ceil(BLOCK);
    let group0_base = 0usize; // kv head 0 ↔ q heads [0..group)
    let own_base = kbin.head * group;

    // -- references ---------------------------------------------------------
    let mut u_own = vec![0.0f32; head_dim];
    let mass_own = reference_mass(ctx, keys, prefix_rows, own_base, &mut u_own);
    let mut truth_idx = Vec::new();
    let mut truth_pairs = Vec::new();
    argtopk_with_scratch(
        &mass_own,
        TOP_K.min(n_leaves),
        &mut truth_idx,
        &mut truth_pairs,
    );
    let truth: Vec<usize> = truth_idx.to_vec();
    let mut mass_desc = mass_own.clone();
    mass_desc.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let mut mass_prefix = Vec::with_capacity(mass_desc.len() + 1);
    mass_prefix.push(0.0f32);
    for &mv in &mass_desc {
        mass_prefix.push(mass_prefix.last().unwrap() + mv);
    }
    let mass_bound = |m: usize| mass_prefix[m.min(mass_prefix.len() - 1)];

    let forced = forced_set(prefix_rows - 1, n_leaves);

    // -- leaf evidence ------------------------------------------------------
    let need = n_leaves * head_dim;
    let leaf_means = &mut scratch_buf[..need];
    let produced = block_means_into(keys, prefix_rows, head_dim, leaf_means);
    assert_eq!(produced, n_leaves);

    let mut tied_scores = vec![0.0f32; n_leaves];
    let heads: Vec<&[f32]> = (0..group)
        .map(|h| &qmat.data[(own_base + h) * head_dim..(own_base + h + 1) * head_dim])
        .collect();
    tied_scores_into(
        &heads,
        leaf_means,
        n_leaves,
        head_dim,
        scale,
        &mut tied_scores,
    );

    let mut lse_scores = vec![0.0f32; n_leaves];
    let mut u_logits = vec![0.0f32; prefix_rows.min(BLOCK) * n_leaves];
    lse_leaf_scores(
        &u_own,
        keys,
        prefix_rows,
        head_dim,
        scale,
        &mut lse_scores,
        &mut u_logits,
    );

    let mut mean_scores = vec![0.0f32; n_leaves];
    mean_leaf_scores(
        &u_own,
        leaf_means,
        n_leaves,
        head_dim,
        scale,
        &mut mean_scores,
    );

    let sel_tied = select_topk(&tied_scores, n_leaves, &forced);
    let sel_lse = select_topk(&lse_scores, n_leaves, &forced);
    let sel_mean = select_topk(&mean_scores, n_leaves, &forced);

    // -- same-group metrics + the argmax-definition pin ----------------------
    let mut out = FamilyOutcome {
        layer: kbin.layer,
        head: kbin.head,
        length: kbin.src_l,
        n_leaves,
        recall_same: [0.0; 3],
        mass_ratio_same: [0.0; 3],
        recall_cross_mean: f32::NAN,
        recall_cross_baseline: f32::NAN,
        cross_groups: 0,
    };
    let sels = [&sel_tied, &sel_lse, &sel_mean];
    for (slot, sel) in sels.iter().enumerate() {
        let hits = sel.iter().filter(|b| truth.contains(b)).count() as f32;
        out.recall_same[slot] = hits / truth.len() as f32;
        let mass: f32 = sel.iter().map(|&b| mass_own[b]).sum();
        let bound = mass_bound(sel.len());
        assert!(
            mass <= bound * (1.0 + 1e-5) + 1e-6,
            "captured-mass pin violated (slot {slot}): mass {mass:.6} > top-{} bound {bound:.6}",
            sel.len()
        );
        out.mass_ratio_same[slot] = mass / bound;
    }

    // -- cross-group: group-0's tied selection vs each other group's truth --
    if let Some(k0) = kbin_head0 {
        assert_eq!(
            k0.layer, kbin.layer,
            "cross-group pair must share the layer"
        );
        assert_eq!(
            k0.src_l, kbin.src_l,
            "cross-group pair must share the length"
        );
        let keys0 = &k0.data[..prefix_rows * head_dim];
        let mut u0 = vec![0.0f32; head_dim];
        let _ = reference_mass(ctx, keys0, prefix_rows, group0_base, &mut u0);
        let leaf_means0 = &mut scratch_buf[..need];
        let produced0 = block_means_into(keys0, prefix_rows, head_dim, leaf_means0);
        assert_eq!(produced0, n_leaves);
        let heads0: Vec<&[f32]> = (0..group)
            .map(|h| &qmat.data[h * head_dim..(h + 1) * head_dim])
            .collect();
        let mut tied0 = vec![0.0f32; n_leaves];
        tied_scores_into(&heads0, leaf_means0, n_leaves, head_dim, scale, &mut tied0);
        let sel0 = select_topk(&tied0, n_leaves, &forced);

        // Cross-group transfer: group-0's tied selection evaluated against
        // THIS family's own truth — measured on every family whose own group
        // is NOT group 0 (head-0 families' truth IS group 0's, nothing to
        // transfer; each other kv head's family contributes its own 0→g
        // read — committed: the 0→3 pair; full set: 0→1, 0→2, 0→3).
        if kbin.head != 0 && kbin.head < ctx.n_kv_head {
            let hits = sel0.iter().filter(|b| truth.contains(b)).count() as f32;
            out.cross_groups = 1;
            out.recall_cross_mean = hits / truth.len() as f32;
            out.recall_cross_baseline = out.recall_same[1]; // single_lse, same truth
        }
    }
    out
}

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
    let group = n_head / n_kv_head;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let kbins = load_k_bins(&m, fd.as_deref(), use_full);
    let qs = load_q(&m, fd.as_deref(), use_full);
    println!(
        "[1250] quality source={} k_bins={} q_mats={} (group={group}, d={head_dim})",
        if use_full { "full" } else { "commit" },
        kbins.len(),
        qs.len()
    );

    let max_rows = kbins.iter().map(|k| k.rows).max().unwrap();
    let max_leaves = max_rows.div_ceil(BLOCK);
    let mut scratch_buf = vec![0.0f32; max_leaves * head_dim + head_dim];
    let ctx = FamilyCtx {
        qmat: &qs[0],
        group,
        n_kv_head,
        head_dim,
        scale,
    };

    let mut out = Vec::with_capacity(kbins.len());
    for kbin in &kbins {
        // Group-0's keys at the same (layer, length) — the cross-group
        // scorer's input; absent only when the family IS group 0.
        let k0 = kbins
            .iter()
            .find(|k| k.layer == kbin.layer && k.head == 0 && k.src_l == kbin.src_l);
        for &p in &q_positions_for(&m, kbin.src_l) {
            let prefix = row_of(kbin, p) + 1;
            let qmat = qs
                .iter()
                .find(|q| q.layer == kbin.layer && q.position == p)
                .unwrap_or_else(|| panic!("no Q bin for layer {} pos {p}", kbin.layer));
            let ctx_here = FamilyCtx { qmat, ..ctx };
            out.push(eval_family(&ctx_here, kbin, k0, prefix, &mut scratch_buf));
        }
    }
    out.shrink_to_fit();
    out
}

const ARM_LABELS: [&str; 3] = ["tied_same", "single_lse", "single_mean"];

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
        "\n[1250] == G1 quality ({source}) — mean recall@{TOP_K}/mass per group (non-trivial) =="
    );
    let header = ARM_LABELS
        .iter()
        .map(|l| format!("{:>11}", l))
        .collect::<Vec<_>>()
        .join(" ");
    println!(
        "[1250] {:>11} | {header} | cross(0→g) mean/min | n",
        "layer/head/L"
    );
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
        let cells: Vec<String> = (0..3)
            .map(|slot| {
                let r: f32 =
                    grp.iter().map(|f| f.recall_same[slot]).sum::<f32>() / grp.len() as f32;
                let mr: f32 =
                    grp.iter().map(|f| f.mass_ratio_same[slot]).sum::<f32>() / grp.len() as f32;
                format!("{:5.3}/{:5.3}", r, mr)
            })
            .collect();
        let cross: Vec<String> = grp
            .iter()
            .filter(|f| f.cross_groups > 0)
            .map(|f| format!("{:.3}/{:.3}", f.recall_cross_mean, f.recall_cross_baseline))
            .collect();
        let cross_cell = if cross.is_empty() {
            "    —    ".to_string()
        } else {
            cross.join(",")
        };
        println!(
            "[1250] {:>4}/{:>2}/{:>5} | {} | {:>17} | {}",
            l,
            h,
            len,
            cells.join(" "),
            cross_cell,
            grp.len()
        );
    }

    let nt: Vec<&FamilyOutcome> = families
        .iter()
        .filter(|f| f.n_leaves >= MIN_LEAVES)
        .collect();
    if nt.is_empty() {
        return;
    }
    println!(
        "[1250] overall over {} non-trivial families (leaves >= {MIN_LEAVES}):",
        nt.len()
    );
    for (slot, label) in ARM_LABELS.iter().enumerate() {
        let r: f32 = nt.iter().map(|f| f.recall_same[slot]).sum::<f32>() / nt.len() as f32;
        let mr: f32 = nt.iter().map(|f| f.mass_ratio_same[slot]).sum::<f32>() / nt.len() as f32;
        println!("[1250]   {label:>11} recall {r:6.4}  mass-ratio {mr:6.4}");
    }
    let cross: Vec<&FamilyOutcome> = nt.iter().filter(|f| f.cross_groups > 0).copied().collect();
    if !cross.is_empty() {
        let cm: f32 = cross.iter().map(|f| f.recall_cross_mean).sum::<f32>() / cross.len() as f32;
        let cb: f32 =
            cross.iter().map(|f| f.recall_cross_baseline).sum::<f32>() / cross.len() as f32;
        println!(
            "[1250]   cross-group (0→g) over {} families: tied-cross recall {cm:6.4}  vs that family's single_lse {cb:6.4}  (paired diff {:+.4})",
            cross.len(),
            cm - cb
        );
    }
}

/// The pre-registered verdicts (computed + printed, never hard-asserted —
/// the bench_397 convention; the per-family mass pins inside eval_family
/// stay hard-asserted everywhere). Returns the ARM verdict string.
fn print_verdicts(families: &[FamilyOutcome], source: &str, min_length: usize) -> String {
    let nt: Vec<&FamilyOutcome> = families
        .iter()
        .filter(|f| f.length >= min_length && f.n_leaves >= MIN_LEAVES)
        .collect();
    if nt.is_empty() {
        println!("[1250] VERDICT DEFERRED ({source}): no non-trivial families at L>={min_length}");
        return "DEFERRED".to_string();
    }
    // paired diffs per family: tied_same − single_lse, and tied − single_mean
    let d_lse: Vec<f32> = nt
        .iter()
        .map(|f| f.recall_same[0] - f.recall_same[1])
        .collect();
    let d_mean: Vec<f32> = nt
        .iter()
        .map(|f| f.recall_same[0] - f.recall_same[2])
        .collect();
    let stats = |d: &[f32]| -> (f32, f32) {
        let mean = d.iter().sum::<f32>() / d.len() as f32;
        let var = d.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / d.len() as f32;
        (mean, var.sqrt())
    };
    let (m_lse, sd_lse) = stats(&d_lse);
    let (m_mean, sd_mean) = stats(&d_mean);
    let se_mean = sd_mean / (d_mean.len() as f32).sqrt();
    let lb95_mean = m_mean - 1.645 * se_mean; // one-sided 95%
    println!(
        "[1250] paired recall diff (tied_same - single_lse) at L>={min_length} [{source}]: mean {m_lse:+.4} sd {sd_lse:.4} n {} (bar: >= -0.01)",
        d_lse.len()
    );
    println!(
        "[1250] paired recall diff (tied_same - single_mean): mean {m_mean:+.4} sd {sd_mean:.4} n {} one-sided LB95 {lb95_mean:+.4} (plan-filing bar: > 0, PER LENGTH BIN)",
        d_mean.len()
    );

    // per-length-bin LB95 (the plan-filing law: never pooled)
    let mut bins: Vec<usize> = Vec::new();
    for f in &nt {
        if !bins.contains(&f.length) {
            bins.push(f.length);
        }
    }
    bins.sort_unstable();
    for b in bins {
        let rows: Vec<&FamilyOutcome> = nt.iter().filter(|f| f.length == b).copied().collect();
        let dm: Vec<f32> = rows
            .iter()
            .map(|f| f.recall_same[0] - f.recall_same[2])
            .collect();
        let (mb, sdb) = stats(&dm);
        let seb = sdb / (dm.len() as f32).sqrt();
        let lbb = mb - 1.645 * seb;
        println!(
            "[1250]   bin L={b:>5}: n {:>3}  tied−single_mean mean {mb:+.4} sd {sdb:.4} LB95 {lbb:+.4}",
            dm.len()
        );
    }

    let pass = m_lse >= -0.01;
    let filing = lb95_mean > 0.0;

    // cross-group column: PAIRED diff vs single_lse on the same truth
    // (the deployment shape's honest read — an absolute recall is never
    // compared against the paired-diff bar).
    let cross: Vec<f32> = nt
        .iter()
        .filter(|f| f.cross_groups > 0)
        .map(|f| f.recall_cross_mean - f.recall_cross_baseline)
        .collect();
    let mut cross_cell = String::from("n/a (no other-group families)");
    let mut cross_pass = false;
    if !cross.is_empty() {
        let (cm, csd) = stats(&cross);
        cross_pass = cm >= -0.01;
        cross_cell = format!(
            "paired diff (tied_cross - single_lse, same truth) {cm:+.4} sd {csd:.4} n {} (bar >= -0.01: {})",
            cross.len(),
            if cross_pass { "PASS" } else { "FAIL" }
        );
    }

    let verdict = match (pass, cross_pass) {
        (true, true) => "PASS",
        (true, false) => "SAME-GROUP-ONLY — cross-group (the YASA deployment shape) fails the bar",
        (false, _) => "FAIL",
    };
    println!(
        "[1250] T4 SAME-GROUP verdict (vs single_lse): {}",
        if pass { "PASS" } else { "FAIL" }
    );
    println!("[1250] T4 CROSS-GROUP verdict: {cross_cell}");
    println!(
        "[1250] T4 plan-filing leg (tied − single_mean LB95 > 0, per length bin): {}",
        if filing {
            "MET (pooled; see bins above — filing requires EVERY bin)"
        } else {
            "NOT MET"
        }
    );
    println!("[1250] ARM VERDICT [{source}, L>={min_length}]: {verdict}");
    verdict.to_string()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn t25_canary_position_zero_degenerate() {
    let m = manifest();
    let head_dim = m["config"]["head_dim"].as_u64().unwrap() as usize;
    let n_head = m["config"]["n_head"].as_u64().unwrap() as usize;
    let n_kv_head = m["config"]["n_kv_head"].as_u64().unwrap() as usize;
    let group = n_head / n_kv_head;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let kbins = load_k_bins(&m, None, false);
    let qs = load_q(&m, None, false);
    let kbin = kbins.iter().find(|k| k.layer == 3 && k.head == 0).unwrap();
    let qmat = qs.iter().find(|q| q.layer == 3 && q.position == 0).unwrap();
    let mut scratch = vec![0.0f32; head_dim + 8];
    let ctx = FamilyCtx {
        qmat,
        group,
        n_kv_head,
        head_dim,
        scale,
    };
    let out = eval_family(&ctx, kbin, None, 1, &mut scratch);
    assert_eq!(
        out.n_leaves, 1,
        "position-0 family is the single-leaf degenerate"
    );
    for (slot, label) in ARM_LABELS.iter().enumerate() {
        assert_eq!(
            out.recall_same[slot], 1.0,
            "{label} must select the only leaf in the degenerate family"
        );
    }
    println!("[1250] T2.5 canary: position-0 real-tensor family — all arms select exactly [0]");
}

#[test]
fn g1_quality_committed_real_tensors() {
    let families = run_quality(false);
    assert!(
        !families.is_empty(),
        "committed fixture set is in-repo — an empty family set is a loader defect, not a pass"
    );
    print_report(&families, "commit");
    // The committed subset spans L ≤ 65 536 stride-subsampled bins; report
    // the pooled committed verdict as a READING (the bar itself lives on
    // the true-length full set — L>=32K contiguous — and is adjudicated
    // there; a committed number never substitutes for it).
    print_verdicts(&families, "commit-reading", 1);
}

#[test]
fn g1_quality_full_set_and_t4_bars() {
    if full_dir().is_none() {
        println!(
            "[1250] SKIP-LOUD: PYRAMID_612_FULL_DIR unset — the true-length T4 bar axis is DEFERRED (committed reading ran in its own test)"
        );
        return;
    }
    let families = run_quality(true);
    assert!(
        !families.is_empty(),
        "full set configured but empty — a loader defect"
    );
    print_report(&families, "full");
    let verdict = print_verdicts(&families, "full", 32768);
    println!(
        "[1250] NOTE: the verdict above carries the recall axis; the COST leg of T4 (tied <= single_lse scoring cost) is the cost cell's verdict, reported in g3_cost_accounting."
    );
    let _ = verdict;
}

// ---------------------------------------------------------------------------
// T3 — cost accounting: measured latency + analytic dot-work ratio
// ---------------------------------------------------------------------------

#[test]
fn g3_cost_accounting() {
    if full_dir().is_none() {
        println!(
            "[1250] SKIP-LOUD: PYRAMID_612_FULL_DIR unset — the cost axis needs true-length tensors, DEFERRED"
        );
        return;
    }
    let release = cfg!(not(debug_assertions));
    if !release {
        println!("[1250] debug build: cost REPORT-ONLY — the cost bar is release-gated");
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

    let mut rows: Vec<(usize, f64, f64, f64)> = Vec::new();
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
        let mut u = vec![0.0f32; d];
        for h in &heads {
            for (uv, &qv) in u.iter_mut().zip(h.iter()) {
                *uv += qv;
            }
        }
        let n_leaves = prefix.div_ceil(BLOCK);
        let mut leaf_means = vec![0.0f32; n_leaves * d];
        block_means_into(keys, prefix, d, &mut leaf_means);

        let reps = if release { 30 } else { 5 };
        let warm = 3;
        let mut t_tied = Vec::with_capacity(reps);
        let mut t_lse = Vec::with_capacity(reps);
        let mut t_mean = Vec::with_capacity(reps);
        let mut blackhole = 0.0f32;
        let mut tied_scores = vec![0.0f32; n_leaves];
        let mut mean_scores = vec![0.0f32; n_leaves];
        let mut logits = vec![0.0f32; BLOCK];
        let mut lse_scores = vec![0.0f32; n_leaves];
        for r in 0..reps + warm {
            let t0 = Instant::now();
            tied_scores_into(&heads, &leaf_means, n_leaves, d, scale, &mut tied_scores);
            let t1 = Instant::now();

            let t2 = Instant::now();
            lse_leaf_scores(&u, keys, prefix, d, scale, &mut lse_scores, &mut logits);
            let t3 = Instant::now();

            let t4 = Instant::now();
            mean_leaf_scores(&u, &leaf_means, n_leaves, d, scale, &mut mean_scores);
            let t5 = Instant::now();
            blackhole += tied_scores[0] + lse_scores[0] + mean_scores[0];

            if r >= warm {
                t_tied.push(t1.duration_since(t0).as_secs_f64() * 1e6);
                t_lse.push(t3.duration_since(t2).as_secs_f64() * 1e6);
                t_mean.push(t5.duration_since(t4).as_secs_f64() * 1e6);
            }
        }
        let _ = blackhole;
        let med = |mut v: Vec<f64>| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let n = v.len();
            if n % 2 == 1 {
                v[n / 2]
            } else {
                (v[n / 2 - 1] + v[n / 2]) / 2.0
            }
        };
        rows.push((prefix, med(t_tied), med(t_lse), med(t_mean)));
    }
    rows.sort_by_key(|r| r.0);

    let analytic = group as f64 / (BLOCK as f64); // H × (N/64) / (1 × N)
    println!(
        "\n[1250] == T3 cost (µs, median of {} reps, release={release}) — analytic dot-work ratio H×(N/C)/(1×N) = {analytic:.4} ==",
        if release { 30 } else { 5 }
    );
    println!(
        "[1250] {:>7} {:>12} {:>12} {:>12}",
        "N", "tied", "single_lse", "single_mean"
    );
    for (n, t, sl, sm) in &rows {
        println!("[1250] {n:>7} {t:>12.1} {sl:>12.1} {sm:>12.1}");
    }
    if let Some(&(nmax, t, sl, _)) = rows.last() {
        println!(
            "[1250] at N={nmax}: tied {t:.1}µs vs single_lse {sl:.1}µs ({:.2}×) — T4 cost leg (tied <= single_lse): {}",
            sl / t,
            if t <= sl { "PASS" } else { "FAIL" }
        );
        if release {
            assert!(
                t <= sl,
                "T4 cost leg FAIL: tied {t:.1}µs > single_lse {sl:.1}µs at N={nmax} — document the negative"
            );
        }
    }
}
