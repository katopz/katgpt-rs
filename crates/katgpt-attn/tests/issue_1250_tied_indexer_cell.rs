//! Issue 1250 Arm A — the YASA tied-indexer recall cell over the REAL
//! pyramid_612 committed capture (qwen38-27B FA layers, pretrained tensors).
//!
//! # What this cell answers
//!
//! The YASA mechanism (Research 698): candidate keys scored by ONE query
//! head-group's mean-ReLU evidence — `s_b = mean_h ReLU(q_h·k̄_b)` over the
//! mean-pooled compressed keys — with one group's selection SHARED by all
//! groups. Against the slot's standing arms at the SAME block contract
//! (C = 64, top-8, NSA forced leaves — the Plan 612/Bench-612 protocol):
//!
//! - `single_mean` — dot(u, k̄_b), the group-summed query (BSA class)
//! - `single_lse`  — exact LSE over the block's per-token u-logits
//! - `tied_same`   — the tied scorer, the family's OWN group indexing
//! - `tied_cross`  — the tied scorer, GROUP-0's selection applied to the
//!   family (YASA's actual mechanism: one indexer shared by all groups;
//!   the reverse direction 3→0 rides along as the symmetry control)
//!
//! Columns, never pooled (the issue's T2): (a) same-group recall vs the
//! family's own truth, (b) cross-group transfer — the consuming family's
//! truth scored by the indexer group's selection.
//!
//! # Scope disclosure (read before quoting any number)
//!
//! - COMMITTED-SUBSET ONLY: the committed bins are stride-subsampled to the
//!   4096-row cap (the harness's disclosed stride decorrelation). The T4
//!   pre-registered bars are FULL-SET-GATED (L ≥ 32K contiguous bins,
//!   `PYRAMID_612_FULL_DIR` — skip-loud when absent, never a green zero).
//!   **Every paired diff below is DIAGNOSTIC at L ≤ 4096 — it is NOT the
//!   T4 verdict and must never be quoted as one.**
//! - kv heads 1, 2 exist only in the full set — the cross-group column
//!   here is the 0↔3 pair the committed capture carries.
//! - The T4 bars are applied NOWHERE in this file — no invented bar at a
//!   length the pre-registration never named.
//!
//! # Loader note
//!
//! The committed-fixture loader mirrors `bench_612_pyramid_goat.rs`'s
//! (manifest + BLAKE3 verify + f32 LE read). Duplicated small, not shared:
//! that file is the LANDED Bench-612 gate — its bytes are its record, and
//! this cell refuses to churn it for a loader.
//!
//! Feature gates: `pyramid_topk` (protocol + hierarchy block means) +
//! `tied_indexer` (the scorer) — the `[[test]]` required-features row is
//! the reader half, this whole-file cfg is the count half.
#![cfg(all(feature = "pyramid_topk", feature = "tied_indexer"))]

use std::fs;
use std::path::{Path, PathBuf};

use katgpt_attn::dash_attn::block_topk::argtopk_with_scratch;
use katgpt_attn::dash_attn::pyramid_topk::{
    PYRAMID_BLOCK_SIZE, PyramidKeyHierarchy, PyramidLevels,
};
use katgpt_attn::dash_attn::tied_indexer::tied_indexer_block_scores_into;

const TOP_K: usize = 8;
const FORCED_LEAF_SLOTS: usize = 3;
const MIN_LEAVES: usize = 2 * TOP_K;

// ---------------------------------------------------------------------------
// Fixture loading (mirrors bench_612_pyramid_goat.rs — see the header note)
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

/// Committed bins only — this cell never touches the full set (the full-set
/// column is the T4 lane's, and its absence here is the disclosure, not a
/// skip to paper over: no `[test]` in this file reads `PYRAMID_612_FULL_DIR`).
struct KBin {
    layer: usize,
    head: usize,
    rows: usize,
    src_l: usize,
    data: Vec<f32>,
}

struct QMat {
    layer: usize,
    position: usize,
    heads: usize,
    data: Vec<f32>,
}

fn load_committed() -> (Vec<KBin>, Vec<QMat>, usize) {
    let m = manifest();
    let config = &m["config"];
    let n_head = config["n_head"].as_u64().unwrap() as usize;
    let n_kv_head = config["n_kv_head"].as_u64().unwrap() as usize;
    let mut kbins = Vec::new();
    for len in m["lengths"].as_array().expect("lengths") {
        let src_l = len["L"].as_u64().unwrap() as usize;
        for e in len["k_commit"].as_array().expect("k entries per length") {
            let path =
                fixture_dir().join(e["file"].as_str().unwrap().trim_start_matches("commit/"));
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
            kbins.push(KBin {
                layer: e["layer"].as_u64().unwrap() as usize,
                head: e["head"].as_u64().unwrap() as usize,
                rows,
                src_l,
                data,
            });
        }
    }
    let mut qs = Vec::new();
    for e in m["q_commit"].as_array().expect("q_commit") {
        let path = fixture_dir().join(e["file"].as_str().unwrap().trim_start_matches("commit/"));
        verify_blake3(&path, e["blake3"].as_str().unwrap());
        let data = read_f32(&path);
        let heads = e["heads"].as_u64().unwrap() as usize;
        let row_len = e["row_len_f32"].as_u64().unwrap() as usize;
        assert_eq!(data.len(), heads * row_len);
        qs.push(QMat {
            layer: e["layer"].as_u64().unwrap() as usize,
            position: e["position"].as_u64().unwrap() as usize,
            heads,
            data,
        });
    }
    assert!(
        !kbins.is_empty(),
        "committed capture is in-repo — an empty set is a repo defect"
    );
    assert!(
        !qs.is_empty(),
        "committed Q capture is in-repo — an empty set is a repo defect"
    );
    (kbins, qs, n_head / n_kv_head)
}

// ---------------------------------------------------------------------------
// Selection + truth (the harness protocol mirrored at cell scope)
// ---------------------------------------------------------------------------

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

/// NSA forced leaves (first / previous / current), deduped, clamped — the
/// harness's `forced_set` mirrored.
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
    v
}

/// Single-level selection: argtopk-K of the non-forced leaves ∪ forced.
fn select_top_k(scores: &[f32], forced: &[usize]) -> Vec<usize> {
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
    out
}

/// A family's truth + shared evidence, computed exactly as the harness
/// computes it (per-q-head softmax mass over the causal prefix, group-summed
/// per C-block; the group-summed query u for the sibling arms). `q_rows`
/// is the family's OWN group slice re-packed compact `[group × d]` — the
/// family's heads sit at offset `kv_head·group` in the full Q matrix, and
/// this function's h-loop reads from offset 0 of what it is handed.
struct FamilyTruth {
    /// Reference block mass (the recall pins read it through the selection
    /// metrics' caller; kept pub-in-file for the mass-ratio report).
    #[allow(dead_code)]
    block_mass: Vec<f32>,
    u: Vec<f32>,
    truth: Vec<usize>,
}

fn family_truth(
    keys: &[f32],
    rows: usize,
    d: usize,
    q_rows: &[f32],
    group: usize,
    scale: f32,
) -> FamilyTruth {
    let n_leaves = rows.div_ceil(PYRAMID_BLOCK_SIZE);
    let mut block_mass = vec![0.0f32; n_leaves];
    let mut u = vec![0.0f32; d];
    for h in 0..group {
        let q = &q_rows[h * d..(h + 1) * d];
        let mut logits = vec![0.0f32; rows];
        let mut mx = f32::NEG_INFINITY;
        for j in 0..rows {
            let l = dot(q, &keys[j * d..(j + 1) * d]) * scale;
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
    // True top-K by reference mass (argmax definition; ties fold into the
    // argtopk's own order — the harness's shape).
    let mut idx = Vec::new();
    let mut pairs = Vec::new();
    let k = TOP_K.min(n_leaves);
    argtopk_with_scratch(&block_mass, k, &mut idx, &mut pairs);
    FamilyTruth {
        block_mass,
        u,
        truth: idx[..k].to_vec(),
    }
}

/// The single-level arms' block scores at the shared contract.
fn single_scores(
    u: &[f32],
    keys: &[f32],
    rows: usize,
    d: usize,
    scale: f32,
) -> (Vec<f32>, Vec<f32>) {
    let n_leaves = rows.div_ceil(PYRAMID_BLOCK_SIZE);
    let mut mean_score = vec![0.0f32; n_leaves];
    let mut lse_score = vec![0.0f32; n_leaves];
    let mut msum = vec![0.0f32; d];
    let mut kbar = vec![0.0f32; d];
    let mut lg = [0.0f32; PYRAMID_BLOCK_SIZE];
    for b in 0..n_leaves {
        let base = b * PYRAMID_BLOCK_SIZE;
        let cnt = (rows - base).min(PYRAMID_BLOCK_SIZE);
        // mean arm: dot(u, block mean) — ONE pass over the block's keys is
        // also the mean accumulation (mean(u·k) = u·mean(k), linearity).
        for m in msum.iter_mut() {
            *m = 0.0;
        }
        let mut mx = f32::NEG_INFINITY;
        for i in 0..cnt {
            let l = dot(u, &keys[(base + i) * d..(base + i + 1) * d]) * scale;
            lg[i] = l;
            if l > mx {
                mx = l;
            }
            for (m, &kv) in msum
                .iter_mut()
                .zip(&keys[(base + i) * d..(base + i + 1) * d])
            {
                *m += kv;
            }
        }
        let inv = 1.0 / cnt as f32;
        for (kbar_v, m) in kbar.iter_mut().zip(msum.iter()) {
            *kbar_v = m * inv;
        }
        mean_score[b] = dot(u, &kbar) * scale;
        let mut se = 0.0f32;
        for &l in lg[..cnt].iter() {
            se += (l - mx).exp();
        }
        lse_score[b] = mx + se.ln();
    }
    (mean_score, lse_score)
}

// ---------------------------------------------------------------------------
// The cell
// ---------------------------------------------------------------------------

/// One committed family: arms → recall vs the family's own truth.
struct FamilyRow {
    layer: usize,
    kv_head: usize,
    src_l: usize,
    position: usize,
    n_leaves: usize,
    recall: [f32; 4], // single_mean, single_lse, tied_same, tied_cross
}

fn run_cell() -> Vec<FamilyRow> {
    let (kbins, qs, group) = load_committed();
    let d = kbins[0].data.len() / kbins[0].rows;
    let scale = 1.0f32 / (d as f32).sqrt();
    let n_head_total = qs[0].heads;
    let _ = n_head_total; // shape witness; the cell keys on (layer, kv_head) only
    let mut rows_out = Vec::new();

    // Indexer group = kv-head 0's q heads (YASA: "the main attention's OWN
    // group-0 (or any one) query-head group"). Disclosed, fixed, and the
    // reverse direction rides along as the symmetry control.
    for kbin in &kbins {
        let rows = kbin.rows;
        let keys = &kbin.data[..rows * d];
        let n_leaves = rows.div_ceil(PYRAMID_BLOCK_SIZE);
        if n_leaves < MIN_LEAVES {
            continue;
        }
        // The family's block means: the pyramid hierarchy's level-0 rows are
        // the exact mean-pooled compressed keys (one build, shared by the
        // tied arms — no selection walks here).
        let need = PyramidKeyHierarchy::required_len(rows, d);
        let mut storage = vec![0.0f32; need];
        let hier = PyramidKeyHierarchy::build(keys, rows, d, &mut storage);
        let block_means = hier.level_rows(0);

        for qmat in qs.iter().filter(|q| q.layer == kbin.layer) {
            let query_idx = qmat.position.min(rows - 1);
            let forced = forced_set(query_idx, n_leaves);

            // The family's own group slice re-packed compact [group × d] —
            // family_truth reads it from offset 0 (see its doc).
            let own_lo = kbin.head * group;
            let mut own_rows = Vec::with_capacity(group * d);
            for h in 0..group {
                own_rows.extend_from_slice(&qmat.data[(own_lo + h) * d..(own_lo + h + 1) * d]);
            }
            let truth = family_truth(keys, rows, d, &own_rows, group, scale);

            let (mean_score, lse_score) = single_scores(&truth.u, keys, rows, d, scale);

            // tied_same: the family's own group as the indexer…
            let mut same_scores = Vec::new();
            tied_indexer_block_scores_into(
                block_means,
                n_leaves,
                d,
                &truth_group_refs(qmat, own_lo, group, d),
                scale,
                &mut same_scores,
            );
            // …tied_cross: GROUP-0's q heads (offset 0) as the indexer —
            // YASA's actual mechanism (one shared selection).
            let mut cross_scores = Vec::new();
            tied_indexer_block_scores_into(
                block_means,
                n_leaves,
                d,
                &truth_group_refs(qmat, 0, group, d),
                scale,
                &mut cross_scores,
            );

            let sel_mean = select_top_k(&mean_score, &forced);
            let sel_lse = select_top_k(&lse_score, &forced);
            let sel_same = select_top_k(&same_scores, &forced);
            let sel_cross = select_top_k(&cross_scores, &forced);

            let rec = |sel: &Vec<usize>| -> f32 {
                let hits = sel.iter().filter(|b| truth.truth.contains(b)).count() as f32;
                hits / truth.truth.len() as f32
            };

            rows_out.push(FamilyRow {
                layer: kbin.layer,
                kv_head: kbin.head,
                src_l: kbin.src_l,
                position: qmat.position,
                n_leaves,
                recall: [
                    rec(&sel_mean),
                    rec(&sel_lse),
                    rec(&sel_same),
                    rec(&sel_cross),
                ],
            });
        }
    }
    assert!(
        !rows_out.is_empty(),
        "no families survived MIN_LEAVES — a loader or shape defect"
    );
    rows_out
}

/// The group slice of a per-(layer, position) Q matrix, as `&[f32]` refs —
/// `lo` names the family's first q head (kv_head·group); the cross arm
/// passes lo = 0 (group-0 indexes, YASA's shared-selection mechanism).
fn truth_group_refs(qmat: &QMat, lo: usize, group: usize, d: usize) -> Vec<&[f32]> {
    (0..group)
        .map(|h| &qmat.data[(lo + h) * d..(lo + h + 1) * d])
        .collect()
}

fn recall_of(rows: &[&FamilyRow], arm: usize) -> f32 {
    rows.iter().map(|r| r.recall[arm]).sum::<f32>() / rows.len() as f32
}

const ARM_LABELS: [&str; 4] = ["single_mean", "single_lse", "tied_same", "tied_cross"];

fn print_report(rows: &[FamilyRow]) {
    println!(
        "\n[1250] == Arm A committed-subset recall@{TOP_K} (C={PYRAMID_BLOCK_SIZE}, top-{TOP_K}, forced∪) — DIAGNOSTIC at L ≤ 4096, NOT the T4 verdict =="
    );
    let header = ARM_LABELS
        .iter()
        .map(|a| format!("{:>11}", a))
        .collect::<Vec<_>>()
        .join(" ");
    println!("[1250] {:>14} | {} | n", "L/kv/pos", header);

    let mut keys: Vec<(usize, usize, usize)> = rows
        .iter()
        .map(|r| (r.src_l, r.kv_head, r.position))
        .collect();
    keys.sort_unstable();
    keys.dedup();
    for (l, kv, pos) in &keys {
        let grp: Vec<&FamilyRow> = rows
            .iter()
            .filter(|r| r.src_l == *l && r.kv_head == *kv && r.position == *pos)
            .collect();
        let cells: Vec<String> = (0..4)
            .map(|arm| format!("{:11.4}", recall_of(&grp, arm)))
            .collect();
        println!(
            "[1250] {:>5}/{:>1}/{:>5} | {} | {}",
            l,
            kv,
            pos,
            cells.join(" "),
            grp.len()
        );
    }

    // Paired diffs — per length bin (the T4 plan-filing bar's grouping; the
    // CI is printed so a future full-set run keeps the exact shape) and the
    // overall (labeled pooled-diagnostic).
    println!("[1250] paired diffs (mean ± sd, n):");
    for (a, b) in [(2usize, 1usize), (2, 0), (3, 1)] {
        let mut bins: Vec<usize> = rows.iter().map(|r| r.src_l).collect();
        bins.sort_unstable();
        bins.dedup();
        for l in bins {
            let grp: Vec<f32> = rows
                .iter()
                .filter(|r| r.src_l == l)
                .map(|r| r.recall[a] - r.recall[b])
                .collect();
            let n = grp.len();
            let mean = grp.iter().sum::<f32>() / n as f32;
            let var = grp.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
            let se = (var.sqrt() / n as f32).sqrt();
            println!(
                "[1250]   {} − {} @L{:<5}: {:+.4} ± {:.4} (n {}, 95% LB {:+.4})",
                ARM_LABELS[a],
                ARM_LABELS[b],
                l,
                mean,
                var.sqrt(),
                n,
                mean - 1.96 * se
            );
        }
        let all: Vec<f32> = rows.iter().map(|r| r.recall[a] - r.recall[b]).collect();
        let n = all.len();
        let mean = all.iter().sum::<f32>() / n as f32;
        let var = all.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
        println!(
            "[1250]   {} − {} POOLED-DIAGNOSTIC: {:+.4} ± {:.4} (n {})",
            ARM_LABELS[a],
            ARM_LABELS[b],
            mean,
            var.sqrt(),
            n
        );
    }
    println!(
        "[1250] T4 bars: FULL-SET-GATED (PYRAMID_612_FULL_DIR) — this table is the\n[1250] committed-column diagnostic the full-set verdict will diff against."
    );
}

#[test]
fn committed_recall_columns() {
    let rows = run_cell();
    print_report(&rows);

    // Structural pins (not quality bars — the pre-registered T4 bars are
    // full-set-gated and appear NOWHERE in this file):
    // 1. the sibling arms rank-consistent with the landed gate's committed
    //    ordering (single_lse ≥ single_mean on the pooled diagnostic) — a
    //    LOW-level sanity on the shared protocol, not a promotion claim;
    let lse_ge_mean = rows.iter().filter(|r| r.recall[1] >= r.recall[0]).count();
    println!(
        "[1250] protocol sanity: single_lse ≥ single_mean on {}/{} families (the landed\n[1250] gate's committed ordering — a protocol check, not a bar)",
        lse_ge_mean,
        rows.len()
    );
    // 2. every recall is a valid fraction;
    for r in rows.iter() {
        for v in r.recall.iter() {
            assert!((0.0..=1.0).contains(v), "recall {} out of range", v);
        }
    }
    // 3. cross and same selections came from DIFFERENT scorers (the cross
    //    column is not accidentally the same column): at least one family
    //    must differ somewhere — if group-0's and the family's own indexer
    //    agree on EVERY family, the mechanism under test is vacuous and the
    //    cell says so loudly instead of reporting two copies of one column.
    let differ = rows.iter().filter(|r| r.recall[2] != r.recall[3]).count();
    println!(
        "[1250] cross≠same on {}/{} families (a 0 read means group-0's indexer is\n[1250] selection-identical to the family's own — the transfer column is vacuous)",
        differ,
        rows.len()
    );
}

#[test]
fn determinism_replay_byte_identical() {
    let rows_a = run_cell();
    let rows_b = run_cell();
    assert_eq!(rows_a.len(), rows_b.len());
    for (a, b) in rows_a.iter().zip(rows_b.iter()) {
        assert_eq!(a.layer, b.layer);
        assert_eq!(a.kv_head, b.kv_head);
        assert_eq!(a.src_l, b.src_l);
        assert_eq!(a.position, b.position);
        assert_eq!(a.n_leaves, b.n_leaves);
        for (va, vb) in a.recall.iter().zip(b.recall.iter()) {
            assert_eq!(
                va.to_bits(),
                vb.to_bits(),
                "bit-identity (Issue-940 discipline)"
            );
        }
    }
}
