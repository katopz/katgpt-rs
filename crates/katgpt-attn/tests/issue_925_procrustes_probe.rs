//! Issue 925 — LoT-fitted extent pooling as a routing summary operator (probe).
//!
//! Source: [arXiv:2610.05816](https://arxiv.org/abs/2610.05816) — *Level-of-Token
//! (LoT) Diffusion* (Nakayama et al., Oct 2026), PASS-redirected from the
//! Research 208 / 379 verdicts; probe charter: `.issues/925_lot_fitted_pooling_routing_probe.md`.
//!
//! # The question
//!
//! This slot's leaf-block summary operators have only been explored
//! ANALYTICALLY (Mean / MeanPlusHalfVar / ExactLse — Bench 612, and
//! `MixedRopeSummarizer` from Plan 397). The LoT paper's fitted extent-pooling
//! stack (orthogonal Procrustes bases, Eq. 14; RMS rescale, Eq. 15) is a
//! data-FITTED operator: a fixed orthogonal W fitted offline on real tensors,
//! one matvec at query time, summary dim d equal to the analytic rivals'.
//! The paper's own ablations say merge-operator quality is load-bearing at
//! compression — this probe measures whether a fitted operator beats the
//! shipped analytic ones on ROUTING (Recall@k of the true top block), never
//! LoT's reconstruction objective (irrelevant here).
//!
//! # The fit (T1) — closed form, no GD
//!
//! For each block b with keys {k_j}, the pooled group is the mean
//! `μ_b = mean_j k_j`. The fit target is the **query-marginalized attention
//! centroid** `k̄_b = Σ_j w_j k_j / Σ_j w_j`, where w_j averages the
//! per-q-head softmax mass of key j over the GQA group sharing the kv head —
//! computed on the FIT partition's own families only. The fit is orthogonal
//! Procrustes over the aligned pairs (μ_b, k̄_b):
//!
//! ```text
//!   M = Σ_b k̄_b μ_bᵀ          (d×d, f64 accumulation)
//!   M = U Σ Vᵀ                (one-sided Jacobi SVD, deterministic)
//!   W = U Vᵀ                  (the orthogonal Procrustes minimizer of ‖WA − B‖_F)
//! ```
//!
//! W is fit PER LAYER shared across that layer's kv heads (key space is a
//! per-layer object). Eval summary: `s_b = W μ_b`; the paper's RMS rescale
//! (Eq. 15) variant: `s_b = W μ_b · τ / rms(W μ_b)` with the fit-time constant
//! `τ = mean rms(W μ_b)` over the fit pairs (per-block rescale = the ranking
//! change the rescale can make; τ is fit-partition-only information).
//!
//! The literal alternative reading — pairs (k_j, μ_b), each real key aligned
//! to its block mean — is dead by THEOREM, not by measurement:
//! `M_lit = Σ_b Σ_{j∈b} k_j μ_bᵀ = Σ_b (Σ_j k_j) μ_bᵀ = C Σ_b μ_b μ_bᵀ`,
//! a symmetric PSD matrix. Its SVD has U = V, so `W_lit = U Uᵀ = I` on the
//! matrix's range — and every μ_b the operator is applied to lies in exactly
//! that range, so `W_lit μ_b = μ_b` identically: **the literal fitted
//! operator collapses into the Mean arm for any rank**. The harness prints
//! the fit scatter's σ_min/σ_max as the premise evidence.
//!
//! # Partitions + leak discipline
//!
//! Leave-one-(layer, kv-head)-out over the fixture's (layer, head) groups —
//! 4 disjoint fit/eval folds on the committed subset, 8 on the full set; every
//! family is evaluated EXACTLY ONCE (under its own held-out fold), so the
//! pooled recall is directly comparable to Bench 612's. W for layer ℓ is fit
//! only from that layer's non-held-out heads' families; the held-out group's
//! K bins and queries are never read by the fit. DISCLOSED: the Issue 908
//! fixture is ONE prompt stream (single token-stream BLAKE3 in the manifest),
//! so prompt-level disjointness is impossible on this capture; the strictest
//! available split is the head-held-out one, and tokens overlap across heads
//! by construction (same stream, different kv projections). Bonsai PQ2_0
//! tensors remain follow-up work gated on the ternary capture lane (issue 925).
//!
//! # Slot discipline (binding, mirrored from `pyramid_topk.rs` + issue 925)
//!
//! - The random-key NIAH harness is BANNED here: everything replays the
//!   Issue 908 (`778de9af8`) qwen38-27B Q4_K_M FA-layer captures.
//! - Committed subset (16 K bins + 30 Q mats) always runs; the full set
//!   (`PYRAMID_612_FULL_DIR`, BLAKE3-verified) is the gate posture.
//! - Reference = full softmax attention mass per q-head (std `exp`),
//!   group-summed per C=64 block; true top-8 = argmax block mass; every arm
//!   shares the group-summed query u, K=8, and the NSA forced leaves
//!   (first / previous / current).
//!
//! # Arms and cost accounting
//!
//! | arm | summary | query-time cost per block |
//! |---|---|---|
//! | `mean` | μ_b (BSA class; == Bench 612 `single_mean` by dot linearity) | 1 dot(d) |
//! | `mixed_rope` | shipped `MixedRopeSummarizer` (Plan 397) | 1 dot(d) |
//! | `fit` | W μ_b (Procrustes, no rescale) | 1 dot(d) |
//! | `fit_rms` | W μ_b RMS-rescaled (LoT Eq. 14+15 stack) | 1 dot(d) |
//! | `halfvar` (ceiling) | mean + ½·Var of the block's per-key logits | per-key logits: cnt dot(d) |
//! | `lse` (ceiling) | exact LSE of the block's per-key logits | per-key logits: cnt dot(d) |
//!
//! All four summary arms are scored from PRECOMPUTED block summaries — at a
//! deployed pyramid's INTERNAL levels that is one matvec per node, which the
//! two ceiling arms cannot match (they need the per-key pass; that is exactly
//! the cost PISA-2/PISA pay at the leaves). `mixed_rope`'s summaries are also
//! query-independent and precomputed offline — equal query cost.
//!
//! `mixed_rope` layout adapter (data-faithful, not rival tuning): the
//! captured qwen35 keys are post-RoPE in the ROTATE-HALF layout with partial
//! rotary (rotary_dim 64 of head_dim 256; pair i rotates dims (i, i+32),
//! dims ≥64 unrotated — the substrate's own rope construction,
//! `inv_freq[i] = theta_base^(−2i/rotary_dim)`). `MixedRopeSummarizer`
//! indexes ADJACENT pairs (2i, 2i+1), so each block is permuted into the
//! adjacent-pair layout (pair i ← dims (i, i+32) for i<32; pairs ≥32 carry
//! the unrotated dims), and `inv_freq` is zero-padded for the unrotated
//! pairs (θ=0 ⇒ the operator's low-frequency rule degenerates to the raw
//! mean — exactly right for dims that never rotate). The query is permuted
//! identically; the inner product is basis-consistent.
//!
//! # Pins (theorems, hard-asserted) vs the pre-registered verdict
//!
//! Hard-asserted per family: Jensen (`mean + ln cnt ≤ true LSE` per block),
//! captured-mass (`mass(sel) ≤ top-|sel| mass` for every arm), the
//! degenerate canary (a position-0 family selects exactly `[0]`), and the
//! W orthonormality pin (‖WᵀW − I‖_F ≤ 1e-3). Also hard-asserted: the
//! harness-comparability invariant — this file's `mean` / `lse` arms run
//! Bench 612's protocol verbatim, so their pooled recall must reproduce
//! 612's published `single_mean` / `single_lse` numbers (±5e-3; the fixture
//! is BLAKE3-pinned, the protocol identical, deterministic math).
//!
//! Pre-registered verdict rule (issue 925 gate 4–5; COMPUTED + PRINTED, a
//! FAIL is a documented negative — never a panic): the fitted arm GAINS iff
//! its paired pooled Recall@8 delta over `mixed_rope` (and over `mean`) is
//! positive AND exceeds the ACROSS-FOLD spread (sd of per-fold means) of the
//! rival baseline. No gain ⇒ the kill criterion fires and the slot's ledger
//! records the negative.
//!
//! # Comparability targets (Bench 612, same families, same protocol)
//!
//! committed: `single_mean` 0.7695 / `single_lse` 0.8906 (64 non-trivial
//! families); full: 0.6475 / 0.7793 (128 non-trivial families).

#![cfg(all(feature = "pyramid_topk", feature = "hga"))]

use std::fs;
use std::path::{Path, PathBuf};

use katgpt_attn::dash_attn::block_topk::argtopk_with_scratch;
use katgpt_attn::dash_attn::pyramid_topk::PYRAMID_BLOCK_SIZE;
use katgpt_core::MixedRopeSummarizer;

const TOP_K: usize = 8;
/// Non-trivial family floor — the Bench 612 convention (leaves ≥ 2K).
const MIN_LEAVES: usize = 2 * TOP_K;

/// Bench 612's published recall@8 numbers (the comparability invariant).
const SIX12_COMMIT_MEAN: f32 = 0.7695;
const SIX12_COMMIT_LSE: f32 = 0.8906;
const SIX12_FULL_MEAN: f32 = 0.6475;
const SIX12_FULL_LSE: f32 = 0.7793;
const SIX12_TOL: f32 = 0.005;

// ---------------------------------------------------------------------------
// Fixture loading (mirrors bench_612; committed subset + env-gated full set)
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
    let bytes = fs::read(path)
        .unwrap_or_else(|e| panic!("read {} for BLAKE3 verify: {e}", path.display()));
    let got = blake3::hash(&bytes).to_hex().to_string();
    assert_eq!(
        got, expected,
        "BLAKE3 mismatch for {} — fixture corruption",
        path.display()
    );
}

fn resolve(rel: &str, full_dir: Option<&Path>) -> PathBuf {
    let name = rel.trim_start_matches("commit/").trim_start_matches("full/");
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
    /// Shape witness — asserted at load against the manifest.
    #[allow(dead_code)]
    rows: usize,
    src_l: usize,
    data: Vec<f32>,
    /// Row → original token position (identity for contiguous full bins).
    positions: Vec<usize>,
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
            assert_eq!(data.len(), rows * row_len, "bin shape for {}", path.display());
            let positions: Vec<usize> = if use_full {
                (0..rows).collect()
            } else {
                let ppath = resolve(e["positions_file"].as_str().unwrap(), full_dir);
                verify_blake3(&ppath, len["commit"]["positions_blake3"].as_str().unwrap());
                read_u32(&ppath).into_iter().map(|p| p as usize).collect()
            };
            assert_eq!(positions.len(), rows, "positions table row count");
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

fn row_of(kbin: &KBin, position: usize) -> usize {
    match kbin.positions.binary_search(&(position)) {
        Ok(i) => i,
        Err(_) => panic!(
            "q position {position} not in the bin rows (L{})",
            kbin.src_l
        ),
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

// ---------------------------------------------------------------------------
// Deterministic one-sided Jacobi SVD (Hestenes) — f64 internals, f32 surface.
// ---------------------------------------------------------------------------

/// SVD of a square n×n row-major matrix: `m = U Σ Vᵀ`.
/// Returns `(u, v, sigma)` row-major n×n each. Deterministic (cyclic pair
/// order, fixed convergence schedule).
fn jacobi_svd_f64(m: &[f64], n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    assert_eq!(m.len(), n * n);
    let mut a = m.to_vec();
    let mut v = vec![0.0f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    let eps = 1e-15;
    for _sweep in 0..60 {
        let mut rotated = false;
        for p in 0..n {
            for q in (p + 1)..n {
                let (mut alpha, mut beta, mut gamma) = (0.0f64, 0.0f64, 0.0f64);
                for i in 0..n {
                    let ap = a[i * n + p];
                    let aq = a[i * n + q];
                    alpha += ap * ap;
                    beta += aq * aq;
                    gamma += ap * aq;
                }
                if alpha == 0.0 || beta == 0.0 || gamma.abs() <= eps * (alpha * beta).sqrt() {
                    continue;
                }
                rotated = true;
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                for i in 0..n {
                    let ap = a[i * n + p];
                    let aq = a[i * n + q];
                    a[i * n + p] = c * ap - s * aq;
                    a[i * n + q] = s * ap + c * aq;
                    let vp = v[i * n + p];
                    let vq = v[i * n + q];
                    v[i * n + p] = c * vp - s * vq;
                    v[i * n + q] = s * vp + c * vq;
                }
            }
        }
        if !rotated {
            break;
        }
    }
    let mut sigma = vec![0.0f64; n];
    let mut u = vec![0.0f64; n * n];
    for j in 0..n {
        let mut nrm = 0.0f64;
        for i in 0..n {
            nrm += a[i * n + j] * a[i * n + j];
        }
        nrm = nrm.sqrt();
        sigma[j] = nrm;
        if nrm > 0.0 {
            for i in 0..n {
                u[i * n + j] = a[i * n + j] / nrm;
            }
        }
    }
    (u, v, sigma)
}

/// Complete U's zero singular columns (rank-deficient M) with deterministic
/// Gram-Schmidt extensions of the identity, in column order.
fn complete_basis(u: &mut [f64], sigma: &[f64], n: usize) {
    let smax = sigma.iter().cloned().fold(0.0f64, f64::max);
    let tol = smax * 1e-12;
    let mut filled = vec![false; n];
    for j in 0..n {
        filled[j] = sigma[j] > tol;
    }
    for j in 0..n {
        if filled[j] {
            continue;
        }
        for k in 0..n {
            let mut w = vec![0.0f64; n];
            w[k] = 1.0;
            for jj in 0..n {
                if !filled[jj] {
                    continue;
                }
                let mut dp = 0.0f64;
                for i in 0..n {
                    dp += u[i * n + jj] * w[i];
                }
                for i in 0..n {
                    w[i] -= dp * u[i * n + jj];
                }
            }
            let nrm: f64 = w.iter().map(|x| x * x).sum();
            let nrm = nrm.sqrt();
            if nrm > 1e-3 {
                for i in 0..n {
                    u[i * n + j] = w[i] / nrm;
                }
                filled[j] = true;
                break;
            }
        }
        assert!(filled[j], "SVD basis completion failed at column {j}");
    }
}

/// Orthogonal Procrustes: minimize ‖W A − B‖_F over orthogonal W where
/// `M = B Aᵀ = Σ k̄ μᵀ`. Returns W (d×d f32, row-major), also enforcing
/// orthonormality via basis completion on rank-deficient fits.
fn procrustes_w(m: &[f64], d: usize) -> Vec<f32> {
    let (mut u, v, sigma) = jacobi_svd_f64(m, d);
    complete_basis(&mut u, &sigma, d);
    let mut w = vec![0.0f32; d * d];
    for i in 0..d {
        for j in 0..d {
            let mut acc = 0.0f64;
            for k in 0..d {
                acc += u[i * d + k] * v[j * d + k];
            }
            w[i * d + j] = acc as f32;
        }
    }
    w
}

fn frob_orthonormality_gap(w: &[f32], d: usize) -> f64 {
    let mut acc = 0.0f64;
    for i in 0..d {
        for j in 0..d {
            let mut dp = 0.0f64;
            for k in 0..d {
                dp += (w[k * d + i] * w[k * d + j]) as f64;
            }
            acc += (dp - if i == j { 1.0 } else { 0.0 }).powi(2);
        }
    }
    acc.sqrt()
}

// ---------------------------------------------------------------------------
// mixed_rope layout adapter (rotate-half + partial rotary → adjacent pairs)
// ---------------------------------------------------------------------------

/// Map adjacent-pair slot 2i / 2i+1 → source data dim. Pair i < rotary_pairs
/// carries the rotated dims (i, i + rotary_pairs); pairs ≥ rotary_pairs carry
/// consecutive unrotated dims from rotary_dim up.
fn rope_pair_map(head_dim: usize, rotary_pairs: usize) -> Vec<usize> {
    let rotary_dim = 2 * rotary_pairs;
    assert!(rotary_dim <= head_dim);
    let half = head_dim / 2;
    assert_eq!(head_dim % 2, 0);
    let mut map = vec![0usize; head_dim];
    for i in 0..rotary_pairs {
        map[2 * i] = i;
        map[2 * i + 1] = i + rotary_pairs;
    }
    for j in 0..(head_dim - rotary_dim) / 2 {
        let slot = 2 * (j + rotary_pairs);
        map[slot] = rotary_dim + 2 * j;
        map[slot + 1] = rotary_dim + 2 * j + 1;
    }
    assert_eq!(map.len(), head_dim);
    assert_eq!(half * 2, head_dim);
    map
}

fn permute_rows(src: &[f32], map: &[usize], rows: usize, d: usize, out: &mut [f32]) {
    for r in 0..rows {
        let base = r * d;
        for (dst, &srccol) in map.iter().enumerate() {
            out[base + dst] = src[base + srccol];
        }
    }
}

/// The model-faithful inv_freq table in the ADJACENT-pair indexing the
/// summarizer consumes: `theta^(−2i/rotary_dim)` for the rotary pairs, 0.0
/// for the unrotated tail (θ=0 ⇒ no phase ⇒ the operator's own low-frequency
/// rule degenerates to the raw mean).
fn rope_inv_freq_perm(head_dim: usize, rotary_pairs: usize, rope_theta: f32) -> Vec<f32> {
    let half = head_dim / 2;
    let mut inv = vec![0.0f32; half];
    for (i, slot) in inv.iter_mut().enumerate().take(rotary_pairs) {
        let exponent = 2.0 * i as f32 / (2 * rotary_pairs) as f32;
        *slot = rope_theta.powf(-exponent);
    }
    inv
}

// ---------------------------------------------------------------------------
// Family reference (mirrors bench_612 exactly) + fit
// ---------------------------------------------------------------------------

/// Per-q-head softmax reference, group-summed per C=64 block; also the
/// per-key attention weight (group-averaged softmax prob) the fit consumes.
struct FamilyRef {
    block_mass: Vec<f32>,
    u: Vec<f32>,
    w: Vec<f32>,
}

fn family_reference(
    keys: &[f32],
    prefix_rows: usize,
    d: usize,
    qmat: &QMat,
    q0: usize,
    group: usize,
    scale: f32,
) -> FamilyRef {
    let mut block_mass = vec![0.0f32; prefix_rows.div_ceil(PYRAMID_BLOCK_SIZE)];
    let mut u = vec![0.0f32; d];
    let mut w = vec![0.0f32; prefix_rows];
    for h in 0..group {
        let q = &qmat.data[(q0 + h) * d..(q0 + h + 1) * d];
        let mut logits = vec![0.0f32; prefix_rows];
        let mut mx = f32::NEG_INFINITY;
        for j in 0..prefix_rows {
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
            w[j] += p / z / group as f32;
        }
        for (uv, &qv) in u.iter_mut().zip(q.iter()) {
            *uv += qv;
        }
    }
    FamilyRef { block_mass, u, w }
}

/// One layer's fitted operator + its diagnostics.
#[derive(Clone)]
struct LayerFit {
    w: Vec<f32>,
    tau: f32,
    pairs: usize,
    skipped: usize,
    /// mean ‖Wμ − k̄‖ / ‖k̄‖ over the fit pairs (in-sample, informational).
    in_rel: f64,
    /// mean ‖μ − k̄‖ / ‖k̄‖ (the unrotated baseline for `in_rel`).
    raw_rel: f64,
    /// σ_min/σ_max of the fit block-mean scatter Σ μμᵀ (the variant-B
    /// identity theorem's premise evidence).
    scatter_ratio: f64,
}

struct FamilySlice<'a> {
    keys: &'a [f32],
    prefix_rows: usize,
    qmat: &'a QMat,
    /// The kv head the keys belong to (picks the q group inside `qmat`).
    head: usize,
}

fn fit_layer(families: &[FamilySlice<'_>], d: usize, group: usize, scale: f32) -> LayerFit {
    let mut m = vec![0.0f64; d * d];
    let mut scatter = vec![0.0f64; d * d];
    let mut mus: Vec<Vec<f32>> = Vec::new();
    let mut kbars: Vec<Vec<f32>> = Vec::new();
    let mut skipped = 0usize;
    for fam in families {
        let n_leaves = fam.prefix_rows.div_ceil(PYRAMID_BLOCK_SIZE);
        let q0 = fam.head * group;
        let fr = family_reference(fam.keys, fam.prefix_rows, d, fam.qmat, q0, group, scale);
        for b in 0..n_leaves {
            let base = b * PYRAMID_BLOCK_SIZE;
            let cnt = (fam.prefix_rows - base).min(PYRAMID_BLOCK_SIZE);
            let sw: f32 = fr.w[base..base + cnt].iter().sum();
            if sw < 1e-20 {
                skipped += 1;
                continue;
            }
            let mut mu = vec![0.0f32; d];
            let mut kbar = vec![0.0f32; d];
            for j in 0..cnt {
                let row = base + j;
                let wj = fr.w[row];
                for (i, (mv, kv)) in mu.iter_mut().zip(kbar.iter_mut()).enumerate() {
                    *mv += fam.keys[row * d + i] / cnt as f32;
                    *kv += fam.keys[row * d + i] * wj;
                }
            }
            for kv in kbar.iter_mut() {
                *kv /= sw;
            }
            for i in 0..d {
                for j in 0..d {
                    m[i * d + j] += kbar[i] as f64 * mu[j] as f64;
                    scatter[i * d + j] += mu[i] as f64 * mu[j] as f64;
                }
            }
            mus.push(mu);
            kbars.push(kbar);
        }
    }
    let pairs = mus.len();
    assert!(pairs > 0, "fit partition produced zero aligned pairs");
    let w = procrustes_w(&m, d);
    let orth = frob_orthonormality_gap(&w, d);
    assert!(orth <= 1e-3, "W orthonormality pin: ‖WᵀW−I‖_F = {orth:.2e}");
    let mut tau = 0.0f64;
    for mu in &mus {
        let s = apply_w(&w, mu, d);
        let rms = (s.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>() / d as f64).sqrt();
        tau += rms;
    }
    tau /= pairs as f64;
    let (mut in_acc, mut raw_acc) = (0.0f64, 0.0f64);
    for (mu, kbar) in mus.iter().zip(kbars.iter()) {
        let s = apply_w(&w, mu, d);
        let knorm = (kbar.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>() + 1e-30).sqrt();
        let fit_err = s
            .iter()
            .zip(kbar.iter())
            .map(|(a, b)| ((*a - *b) as f64).powi(2))
            .sum::<f64>();
        let raw_err = mu
            .iter()
            .zip(kbar.iter())
            .map(|(a, b)| ((*a - *b) as f64).powi(2))
            .sum::<f64>();
        in_acc += fit_err.sqrt() / knorm;
        raw_acc += raw_err.sqrt() / knorm;
    }
    let (_, _, ssig) = jacobi_svd_f64(&scatter, d);
    let smin = ssig.iter().cloned().fold(f64::INFINITY, f64::min);
    let smax = ssig.iter().cloned().fold(0.0f64, f64::max);
    LayerFit {
        w,
        tau: tau as f32,
        pairs,
        skipped,
        in_rel: in_acc / pairs as f64,
        raw_rel: raw_acc / pairs as f64,
        scatter_ratio: if smax > 0.0 { smin / smax } else { 0.0 },
    }
}

fn apply_w(w: &[f32], x: &[f32], d: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; d];
    for (i, o) in out.iter_mut().enumerate() {
        let row = &w[i * d..(i + 1) * d];
        *o = dot(row, x);
    }
    out
}

// ---------------------------------------------------------------------------
// Selection (mirrors bench_612) + arms
// ---------------------------------------------------------------------------

/// NSA forced leaves (first / previous / current), deduped, clamped.
fn forced_set(query_idx: usize, n_leaves: usize) -> Vec<usize> {
    if n_leaves == 0 {
        return Vec::new();
    }
    let cur = (query_idx / PYRAMID_BLOCK_SIZE).min(n_leaves - 1);
    let mut v = Vec::with_capacity(3);
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
    Mean,
    MixedRope,
    Fit,
    FitRms,
    HalfVar,
    Lse,
}

const ARMS: [Arm; 6] = [
    Arm::Mean,
    Arm::MixedRope,
    Arm::Fit,
    Arm::FitRms,
    Arm::HalfVar,
    Arm::Lse,
];

const MEAN: usize = 0;
const MIXED: usize = 1;
const FIT: usize = 2;
const FIT_RMS: usize = 3;
const HALFVAR: usize = 4;
const LSE: usize = 5;

impl Arm {
    fn label(self) -> &'static str {
        match self {
            Arm::Mean => "mean",
            Arm::MixedRope => "mixed_rope",
            Arm::Fit => "fit",
            Arm::FitRms => "fit_rms",
            Arm::HalfVar => "halfvar*",
            Arm::Lse => "lse*",
        }
    }
}

struct FamilyOutcome {
    layer: usize,
    head: usize,
    length: usize,
    n_leaves: usize,
    recall: [f32; 6],
    mass_ratio: [f32; 6],
}

/// Per-run constants shared by every family's evaluation.
struct EvalCtx {
    head_dim: usize,
    n_head: usize,
    n_kv_head: usize,
    scale: f32,
    rope_theta: f32,
    rotary_pairs: usize,
}

/// Evaluate ONE family against the layer's fitted operator.
fn eval_family(
    kbin: &KBin,
    prefix_rows: usize,
    qmat: &QMat,
    query_idx: usize,
    fit: &LayerFit,
    ctx: &EvalCtx,
) -> FamilyOutcome {
    let head_dim = ctx.head_dim;
    let n_head = ctx.n_head;
    let n_kv_head = ctx.n_kv_head;
    let scale = ctx.scale;
    let rope_theta = ctx.rope_theta;
    let rotary_pairs = ctx.rotary_pairs;
    let d = head_dim;
    let keys = &kbin.data[..prefix_rows * d];
    let n_leaves = prefix_rows.div_ceil(PYRAMID_BLOCK_SIZE);
    let group = n_head / n_kv_head;
    let q0 = kbin.head * group;
    let pair_map = rope_pair_map(d, rotary_pairs);
    let inv_freq = rope_inv_freq_perm(d, rotary_pairs, rope_theta);

    // -- reference (truth) + fit weights + group-summed query ---------------
    let fr = family_reference(keys, prefix_rows, d, qmat, q0, group, scale);
    let u = &fr.u;

    // -- u logits: ONE pass, the shared leaf evidence ------------------------
    let mut u_logits = vec![0.0f32; prefix_rows];
    for j in 0..prefix_rows {
        u_logits[j] = dot(u, &keys[j * d..(j + 1) * d]) * scale;
    }

    // -- per-leaf ladder values + the Jensen pin (theorem) -------------------
    let mut mean_score = vec![0.0f32; n_leaves];
    let mut halfvar_score = vec![0.0f32; n_leaves];
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
        let mut v = 0.0f32;
        for &l in lg {
            v += (l - mean) * (l - mean);
        }
        v /= cnt as f32;
        let mx = lg.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut se = 0.0f32;
        for &l in lg {
            se += (l - mx).exp();
        }
        let lse = mx + se.ln();
        mean_score[b] = mean;
        halfvar_score[b] = mean + 0.5 * v;
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

    // -- block means (raw layout) + fitted summaries --------------------------
    let mut fit_score = vec![0.0f32; n_leaves];
    let mut fit_rms_score = vec![0.0f32; n_leaves];
    {
        let mut wmu = Vec::with_capacity(d);
        for b in 0..n_leaves {
            let base = b * PYRAMID_BLOCK_SIZE;
            let cnt = (prefix_rows - base).min(PYRAMID_BLOCK_SIZE);
            let mut mu = vec![0.0f32; d];
            for j in 0..cnt {
                let row = base + j;
                for (i, mv) in mu.iter_mut().enumerate() {
                    *mv += keys[row * d + i] / cnt as f32;
                }
            }
            wmu.clear();
            wmu.extend(apply_w(&fit.w, &mu, d));
            fit_score[b] = dot(u, &wmu);
            let rms = (wmu.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>() / d as f64)
                .sqrt() as f32;
            if rms > 1e-12 {
                let g = fit.tau / rms;
                fit_rms_score[b] = dot(u, &wmu.iter().map(|x| x * g).collect::<Vec<f32>>());
            } else {
                fit_rms_score[b] = fit_score[b];
            }
        }
    }

    // -- mixed_rope: shipped summarizer over the adjacent-pair permutation ----
    let mut mixed_score = vec![0.0f32; n_leaves];
    {
        let u_perm: Vec<f32> = pair_map.iter().map(|&src| u[src]).collect();
        let mut block_perm = vec![0.0f32; PYRAMID_BLOCK_SIZE * d];
        for (b, ms) in mixed_score.iter_mut().enumerate() {
            let base = b * PYRAMID_BLOCK_SIZE;
            let cnt = (prefix_rows - base).min(PYRAMID_BLOCK_SIZE);
            permute_rows(&keys[base * d..(base + cnt) * d], &pair_map, cnt, d, &mut block_perm);
            let pos = &kbin.positions[base..base + cnt];
            let span = pos[cnt - 1] - pos[0];
            let summarizer = MixedRopeSummarizer::new(d, inv_freq.clone(), span);
            let summary = summarizer.summarize(&block_perm, pos, 0, cnt);
            *ms = dot(&u_perm, &summary);
        }
    }

    // -- truth top-K by reference mass + the per-size mass bound --------------
    let mut truth_idx = Vec::new();
    let mut truth_pairs = Vec::new();
    argtopk_with_scratch(
        &fr.block_mass,
        TOP_K.min(n_leaves),
        &mut truth_idx,
        &mut truth_pairs,
    );
    let truth: Vec<usize> = truth_idx.to_vec();
    let mut mass_desc = fr.block_mass.clone();
    mass_desc.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let mut mass_prefix = Vec::with_capacity(mass_desc.len() + 1);
    mass_prefix.push(0.0f32);
    for &mv in &mass_desc {
        mass_prefix.push(*mass_prefix.last().unwrap() + mv);
    }
    let mass_bound = |m: usize| mass_prefix[m.min(mass_prefix.len() - 1)];

    let forced = forced_set(query_idx, n_leaves);

    // -- selections per arm ----------------------------------------------------
    let mut selections: [Vec<usize>; 6] = Default::default();
    selections[MEAN] = single_level_select(&mean_score, &forced);
    selections[MIXED] = single_level_select(&mixed_score, &forced);
    selections[FIT] = single_level_select(&fit_score, &forced);
    selections[FIT_RMS] = single_level_select(&fit_rms_score, &forced);
    selections[HALFVAR] = single_level_select(&halfvar_score, &forced);
    selections[LSE] = single_level_select(&lse_score, &forced);

    if n_leaves == 1 {
        for sel in &selections {
            assert_eq!(sel, &[0], "degenerate canary: position-0 family must select [0]");
        }
    }

    let mut outcome = FamilyOutcome {
        layer: kbin.layer,
        head: kbin.head,
        length: kbin.src_l,
        n_leaves,
        recall: [0.0; 6],
        mass_ratio: [0.0; 6],
    };
    for (slot, sel) in selections.iter().enumerate() {
        let hits = sel.iter().filter(|b| truth.contains(b)).count() as f32;
        outcome.recall[slot] = hits / truth.len() as f32;
        let mass: f32 = sel.iter().map(|&b| fr.block_mass[b]).sum();
        let bound = mass_bound(sel.len());
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

// ---------------------------------------------------------------------------
// Folds + report
// ---------------------------------------------------------------------------

struct FoldRow {
    layer: usize,
    head: usize,
    n_nontrivial: usize,
    mean_recall: [f64; 6],
}

struct ProbeRun {
    outcomes: Vec<FamilyOutcome>,
    folds: Vec<FoldRow>,
}

fn run_probe(use_full: bool) -> ProbeRun {
    let m = manifest();
    let fd = full_dir();
    let fd = if use_full {
        let d = fd.unwrap_or_else(|| panic!("PYRAMID_612_FULL_DIR not set"));
        assert!(d.is_dir(), "PYRAMID_612_FULL_DIR={} is not a dir", d.display());
        Some(d)
    } else {
        None
    };
    let head_dim = m["config"]["head_dim"].as_u64().unwrap() as usize;
    let n_head = m["config"]["n_head"].as_u64().unwrap() as usize;
    let n_kv_head = m["config"]["n_kv_head"].as_u64().unwrap() as usize;
    let rope_theta = m["config"]["rope_theta"].as_f64().unwrap() as f32;
    let rotary_pairs = m["config"]["rotary_dim"].as_u64().unwrap() as usize / 2;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let group = n_head / n_kv_head;

    let kbins = load_k_bins(&m, fd.as_deref(), use_full);
    let qs = load_q(&m, fd.as_deref(), use_full);
    let ctx = EvalCtx {
        head_dim,
        n_head,
        n_kv_head,
        scale,
        rope_theta,
        rotary_pairs,
    };
    println!(
        "[925] probe source={} k_bins={} q_mats={} d={head_dim} rotary_pairs={rotary_pairs} rope_theta={rope_theta:e}",
        if use_full { "full" } else { "commit" },
        kbins.len(),
        qs.len()
    );

    // Eval groups: the fixture's (layer, head) pairs, in deterministic order.
    let mut groups: Vec<(usize, usize)> = Vec::new();
    for kbin in &kbins {
        let g = (kbin.layer, kbin.head);
        if !groups.contains(&g) {
            groups.push(g);
        }
    }
    groups.sort_unstable();

    let mut outcomes = Vec::new();
    let mut folds = Vec::new();

    for (gi, (glayer, ghead)) in groups.iter().enumerate() {
        // Fit families: same layer, other heads.
        let mut fit_families: Vec<FamilySlice> = Vec::new();
        for kbin in kbins.iter().filter(|k| k.layer == *glayer && k.head != *ghead) {
            for &p in &q_positions_for(&m, kbin.src_l) {
                let prefix = row_of(kbin, p) + 1;
                let qmat = qs
                    .iter()
                    .find(|q| q.layer == kbin.layer && q.position == p)
                    .unwrap_or_else(|| panic!("no Q bin for layer {} pos {p}", kbin.layer));
                fit_families.push(FamilySlice {
                    keys: &kbin.data[..prefix * head_dim],
                    prefix_rows: prefix,
                    qmat,
                    head: kbin.head,
                });
            }
        }
        let fit = fit_layer(&fit_families, head_dim, group, scale);
        println!(
            "[925] fold {gi} layer{glayer}/head{ghead}: fit pairs {} (skipped {}) in-sample |Wμ−k̄|/|k̄| {:.4} vs raw {:.4} scatter σmin/σmax {:.2e} τ {:.4}",
            fit.pairs, fit.skipped, fit.in_rel, fit.raw_rel, fit.scatter_ratio, fit.tau
        );

        // Eval families: the held-out group.
        let mut fold_recalls = [0.0f64; 6];
        let mut n_nt = 0usize;
        for kbin in kbins.iter().filter(|k| k.layer == *glayer && k.head == *ghead) {
            for &p in &q_positions_for(&m, kbin.src_l) {
                let prefix = row_of(kbin, p) + 1;
                let qmat = qs
                    .iter()
                    .find(|q| q.layer == kbin.layer && q.position == p)
                    .unwrap_or_else(|| panic!("no Q bin for layer {} pos {p}", kbin.layer));
                let outcome = eval_family(kbin, prefix, qmat, prefix - 1, &fit, &ctx);
                if outcome.n_leaves >= MIN_LEAVES {
                    n_nt += 1;
                    for (s, fr) in fold_recalls.iter_mut().enumerate() {
                        *fr += outcome.recall[s] as f64;
                    }
                }
                outcomes.push(outcome);
            }
        }
        if n_nt > 0 {
            for fr in fold_recalls.iter_mut() {
                *fr /= n_nt as f64;
            }
        }
        folds.push(FoldRow {
            layer: *glayer,
            head: *ghead,
            n_nontrivial: n_nt,
            mean_recall: fold_recalls,
        });
    }

    ProbeRun { outcomes, folds }
}

fn mean_of(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn sd_of(xs: &[f64]) -> f64 {
    let mu = mean_of(xs);
    (xs.iter().map(|x| (x - mu) * (x - mu)).sum::<f64>() / xs.len() as f64).sqrt()
}

fn print_report(run: &ProbeRun, source: &str) {
    let nt: Vec<&FamilyOutcome> = run
        .outcomes
        .iter()
        .filter(|f| f.n_leaves >= MIN_LEAVES)
        .collect();
    println!(
        "\n[925] == per-fold mean recall@{TOP_K} (non-trivial families) — {source} =="
    );
    let header = ARMS
        .iter()
        .map(|a| format!("{:>10}", a.label()))
        .collect::<Vec<_>>()
        .join(" ");
    println!("[925] {:>14} | {header} | n", "fold layer/head");
    for f in &run.folds {
        let cells = f
            .mean_recall
            .iter()
            .map(|r| format!("{:10.4}", r))
            .collect::<Vec<_>>()
            .join(" ");
        println!(
            "[925] {:>6}/{:>3}     | {} | {}",
            f.layer, f.head, cells, f.n_nontrivial
        );
    }

    println!("\n[925] == group table (layer/head/L pooled over folds, non-trivial) ==");
    let mut keys: Vec<(usize, usize, usize)> = Vec::new();
    for f in run
        .outcomes
        .iter()
        .filter(|f| f.n_leaves >= MIN_LEAVES)
    {
        let k = (f.layer, f.head, f.length);
        if !keys.contains(&k) {
            keys.push(k);
        }
    }
    keys.sort_unstable();
    println!("[925] {:>6}/{:>3}/{:>6} | {header} | n", "layer", "head", "L");
    for (l, h, len) in &keys {
        let grp: Vec<&FamilyOutcome> = run
            .outcomes
            .iter()
            .filter(|f| f.layer == *l && f.head == *h && f.length == *len && f.n_leaves >= MIN_LEAVES)
            .collect();
        if grp.is_empty() {
            continue;
        }
        let cells = (0..6)
            .map(|s| {
                format!(
                    "{:>10.4}",
                    grp.iter().map(|f| f.recall[s] as f64).sum::<f64>() / grp.len() as f64
                )
            })
            .collect::<Vec<_>>()
            .join(" ");
        println!("[925] {:>6}/{:>3}/{:>6} | {} | {}", l, h, len, cells, grp.len());
    }

    println!("\n[925] == length-axis breakdown (pooled over folds, non-trivial) ==");
    let mut lens: Vec<usize> = run
        .outcomes
        .iter()
        .filter(|f| f.n_leaves >= MIN_LEAVES)
        .map(|f| f.length)
        .collect();
    lens.sort_unstable();
    lens.dedup();
    println!("[925] {:>6} | {header} | n", "L");
    for l in &lens {
        let grp: Vec<&FamilyOutcome> = run
            .outcomes
            .iter()
            .filter(|f| f.length == *l && f.n_leaves >= MIN_LEAVES)
            .collect();
        let cells = (0..6)
            .map(|s| {
                format!(
                    "{:10.4}",
                    grp.iter().map(|f| f.recall[s] as f64).sum::<f64>() / grp.len() as f64
                )
            })
            .collect::<Vec<_>>()
            .join(" ");
        println!("[925] {:>6} | {} | {}", l, cells, grp.len());
    }

    println!("\n[925] == overall (pooled; every family evaluated exactly once) ==");
    println!("[925] {:>10} | {:>8} {:>8} | n", "arm", "recall@8", "mass-r");
    for (slot, arm) in ARMS.iter().enumerate() {
        let r = nt.iter().map(|f| f.recall[slot] as f64).sum::<f64>() / nt.len() as f64;
        let mr = nt.iter().map(|f| f.mass_ratio[slot] as f64).sum::<f64>() / nt.len() as f64;
        println!("[925] {:>10} | {:8.4} {:8.4} | {}", arm.label(), r, mr, nt.len());
    }

    // Paired deltas + the pre-registered verdict rule.
    let pooled = |slot: usize| -> f64 {
        nt.iter().map(|f| f.recall[slot] as f64).sum::<f64>() / nt.len() as f64
    };
    let folds_of = |slot: usize| -> Vec<f64> {
        run.folds
            .iter()
            .filter(|f| f.n_nontrivial > 0)
            .map(|f| f.mean_recall[slot])
            .collect()
    };
    println!("\n[925] == verdict (pre-registered: gain iff Δ>0 AND Δ>baseline across-fold sd) ==");
    for (probe_slot, rival_slot) in [(FIT, MIXED), (FIT_RMS, MIXED), (FIT, MEAN)] {
        let delta = pooled(probe_slot) - pooled(rival_slot);
        let rival_folds = folds_of(rival_slot);
        let spread = sd_of(&rival_folds);
        let verdict = if delta > 0.0 && delta > spread {
            "GAIN"
        } else {
            "NO-GAIN"
        };
        println!(
            "[925]   Δ({} − {}) = {:+.4}  baseline across-fold sd = {:.4}  → {}",
            ARMS[probe_slot].label(),
            ARMS[rival_slot].label(),
            delta,
            spread,
            verdict
        );
    }
    println!(
        "[925]   ceiling echo: Δ(lse − mean) = {:+.4} (Bench 612 measured −0.1211 committed / −0.1318 full for the pyramid walk; here both are single-level)",
        pooled(LSE) - pooled(MEAN)
    );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn svd_procrustes_selftest_pins() {
    // Known answer: W must recover a fixed rotation from noiseless pairs,
    // including the M orientation (M = B Aᵀ, W = U Vᵀ — a transposed
    // construction returns Rᵀ and fails this pin).
    let d = 8usize;
    let th = 0.7f64;
    let (c, s) = (th.cos(), th.sin());
    let mut r = vec![0.0f64; d * d];
    r[0] = c;
    r[1] = -s;
    r[d] = s;
    r[d + 1] = c;
    for i in 2..d {
        r[i * d + i] = 1.0;
    }
    let mut m = vec![0.0f64; d * d];
    let n_pairs = 64;
    for p in 0..n_pairs {
        let mut mu = vec![0.0f64; d];
        for (i, mv) in mu.iter_mut().enumerate() {
            *mv = ((p * 7 + i * 13) % 11) as f64 / 11.0 - 0.5;
        }
        // M += (Rμ) μᵀ
        let mut rmu = vec![0.0f64; d];
        for i in 0..d {
            rmu[i] = (0..d).map(|k| r[i * d + k] * mu[k]).sum::<f64>();
        }
        for i in 0..d {
            for j in 0..d {
                m[i * d + j] += rmu[i] * mu[j];
            }
        }
    }
    let w = procrustes_w(&m, d);
    let orth = frob_orthonormality_gap(&w, d);
    assert!(orth <= 1e-3, "orthonormality pin: {orth:.2e}");
    let mut max_err = 0.0f64;
    for i in 0..d {
        for j in 0..d {
            max_err = max_err.max((w[i * d + j] as f64 - r[i * d + j]).abs());
        }
    }
    assert!(max_err < 1e-6, "Procrustes known-answer pin: max |W−R| = {max_err:.2e}");

    // Rank-deficient completion: a rank-1 M still yields an orthogonal W.
    let mut m1 = vec![0.0f64; d * d];
    let mut mu = vec![0.0f64; d];
    mu[0] = 1.0;
    for i in 0..d {
        for j in 0..d {
            m1[i * d + j] = 2.0 * mu[i] * mu[j];
        }
    }
    let w1 = procrustes_w(&m1, d);
    let orth1 = frob_orthonormality_gap(&w1, d);
    assert!(orth1 <= 1e-3, "rank-deficient orthonormality pin: {orth1:.2e}");
    // W acts as identity on the fit's row space (span of μ): Wμ = μ here.
    let muf: Vec<f32> = mu.iter().map(|x| *x as f32).collect();
    let wmu = apply_w(&w1, &muf, d);
    let err: f64 = wmu
        .iter()
        .zip(muf.iter())
        .map(|(a, b)| ((*a - *b) as f64).abs())
        .sum::<f64>();
    assert!(err < 1e-4, "range-action pin: Wμ must equal μ on the fit range ({err:.2e})");

    // rope_pair_map sanity: rotary pairs hit (i, i+rp); tail pairs are consecutive.
    let map = rope_pair_map(256, 32);
    assert_eq!(map[0], 0);
    assert_eq!(map[1], 32);
    assert_eq!(map[62], 31);
    assert_eq!(map[63], 63);
    assert_eq!(map[64], 64);
    assert_eq!(map[65], 65);
    assert_eq!(map[254], 254);
    assert_eq!(map[255], 255);
    let mut seen = vec![false; 256];
    for &s in &map {
        assert!(!seen[s], "rope_pair_map is a bijection");
        seen[s] = true;
    }
}

#[test]
fn probe_925_committed_real_tensors() {
    let run = run_probe(false);
    assert!(
        !run.outcomes.is_empty(),
        "committed fixture set is in-repo — an empty family set is a loader defect, not a pass"
    );
    print_report(&run, "commit");
    let nt: Vec<&FamilyOutcome> = run
        .outcomes
        .iter()
        .filter(|f| f.n_leaves >= MIN_LEAVES)
        .collect();
    assert_eq!(
        nt.len(),
        64,
        "committed non-trivial family count drifted from Bench 612's 64"
    );
    // Harness-comparability invariant: the mean/lse arms ARE Bench 612's
    // single-level protocol; their pooled recall must reproduce 612's
    // published numbers (BLAKE3-pinned fixture, identical code path).
    let pooled = |slot: usize| -> f64 {
        nt.iter().map(|f| f.recall[slot] as f64).sum::<f64>() / nt.len() as f64
    };
    let mean_gap = (pooled(MEAN) as f32 - SIX12_COMMIT_MEAN).abs();
    let lse_gap = (pooled(LSE) as f32 - SIX12_COMMIT_LSE).abs();
    assert!(
        mean_gap <= SIX12_TOL && lse_gap <= SIX12_TOL,
        "comparability invariant FAILED vs Bench 612: mean {:.4} (want {:.4}±{}) lse {:.4} (want {:.4}±{})",
        pooled(MEAN),
        SIX12_COMMIT_MEAN,
        SIX12_TOL,
        pooled(LSE),
        SIX12_COMMIT_LSE,
        SIX12_TOL
    );
    println!(
        "[925] comparability vs 612 committed: mean {:.4} (612 {:.4}) lse {:.4} (612 {:.4}) — within ±{}",
        pooled(MEAN), SIX12_COMMIT_MEAN, pooled(LSE), SIX12_COMMIT_LSE, SIX12_TOL
    );
}

#[test]
fn probe_925_full_set() {
    let Some(_) = full_dir() else {
        println!("[925] full set not configured (PYRAMID_612_FULL_DIR unset) — SKIP LOUD (a deferral, never a green)");
        return;
    };
    let run = run_probe(true);
    print_report(&run, "full");
    let nt: Vec<&FamilyOutcome> = run
        .outcomes
        .iter()
        .filter(|f| f.n_leaves >= MIN_LEAVES)
        .collect();
    let pooled = |slot: usize| -> f64 {
        nt.iter().map(|f| f.recall[slot] as f64).sum::<f64>() / nt.len() as f64
    };
    let mean_gap = (pooled(MEAN) as f32 - SIX12_FULL_MEAN).abs();
    let lse_gap = (pooled(LSE) as f32 - SIX12_FULL_LSE).abs();
    assert!(
        mean_gap <= SIX12_TOL && lse_gap <= SIX12_TOL,
        "comparability invariant FAILED vs Bench 612 full: mean {:.4} (want {:.4}) lse {:.4} (want {:.4})",
        pooled(MEAN),
        SIX12_FULL_MEAN,
        pooled(LSE),
        SIX12_FULL_LSE
    );
    println!(
        "[925] comparability vs 612 full: mean {:.4} (612 {:.4}) lse {:.4} (612 {:.4}) — within ±{}",
        pooled(MEAN), SIX12_FULL_MEAN, pooled(LSE), SIX12_FULL_LSE, SIX12_TOL
    );
}
