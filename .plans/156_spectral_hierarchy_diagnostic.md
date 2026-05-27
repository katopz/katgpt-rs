# Plan 156: Spectral Hierarchy Diagnostic — KG Extraction Validation

**Date:** 2026-05-27
**Research:** 121 (Hierarchical Concept Geometry Emerges from Co-occurrence)
**Related:** Research 010 (KG × HLA), Research 011 (PEIRA), Research 111 (Analogy), Research 012 (LEO), Plan 149 (Dirichlet Energy), Plan 151 (KG Role Transport)
**Feature Gate:** `spectral_hierarchy` (default-OFF, opt-in)
**Verdict:** LOW-GAIN implementation — theory validates existing pipeline, adds one diagnostic utility

---

## Why This Plan Exists

Research 121 proves that hierarchical splitting geometry in co-occurrence Gram matrices is **theoretically guaranteed** under our decay assumptions. This means our KG extraction pipeline (Research 010) has a firm mathematical foundation. The implementation is minimal — a spectral diagnostic utility + wiring into existing Dirichlet Energy probes.

---

## Tasks

- [ ] T1: Add `spectral_hierarchy` feature gate to `katgpt-rs-core/Cargo.toml`
- [ ] T2: Implement `eigenspace_alignment(gram: &[Vec<f32>], reference: &[Vec<f32>], k: usize) -> f32` — top-k eigenspace alignment g(k) metric
- [ ] T3: Implement `haar_wavelet_basis(depth: usize) -> (Vec<Vec<f32>>, Vec<Vec<Vec<f32>>>)` — scaling + wavelet modes for binary trees
- [ ] T4: Implement `cauchy_interlacing_check(eigenvalues: &[Vec<f32>]) -> bool` — validate nested split block interlacing
- [ ] T5: Wire `eigenspace_alignment` into `data_probe/dirichlet_energy.rs` as complementary metric
- [ ] T6: GOAT proof — synthetic binary tree with exponential kernel → verify coarse-to-fine eigenvector ordering + interlacing + g(k) > 0.9
- [ ] T7: Document in `27_mmo_goat_pillars_decision_matrix.md` that Research 010 KG extraction is now theoretically grounded (add cross-reference to Research 121)

---

## Feature Gate

```toml
[features]
spectral_hierarchy = []  # KG extraction diagnostic — eigenspace alignment, Haar wavelets, Cauchy interlacing
```

Default-OFF because:
- Offline analysis tool, not inference path
- Zero perf impact when disabled
- If proven useful at inference time (attention conditioning) → promote to default-on

---

## GOAT Proof Criteria (T6)

Synthetic test with depth-3 binary tree, exponential kernel f(d) = α·e^{-βd}:
1. Eigenvectors separate into scaling + wavelet modes (Theorem 1) ✅
2. Wavelet eigenvalues ordered coarse-to-fine (Theorem 2) ✅
3. Cauchy interlacing holds across nested blocks ✅
4. g(k) > 0.9 between theoretical and empirical Gram matrices for k ≤ 5 ✅

---

## What's NOT in This Plan

- Game-specific co-occurrence matrix construction → riir-ai private
- KG node extraction → Plan 151 (KG Role Transport)
- PEIRA training integration → Plan 153
- LEO goal hierarchy → Plan 155
