# Plan 298: Smear-Aware FaithfulnessProbe — Ternary Smear Classifier

**Date:** 2026-06-21
**Research:** [katgpt-rs/.research/277_DiffusionGemma_Transparency_Smearing_Faithfulness.md](../.research/277_DiffusionGemma_Transparency_Smearing_Faithfulness.md)
**Source paper:** [arXiv:2606.20560](https://arxiv.org/abs/2606.20560) — Engels et al., "How Transparent is DiffusionGemma?", DeepMind, Jun 2026
**Target:** `katgpt-rs/crates/katgpt-core/src/faithfulness/smear.rs` (new module) + Cargo feature `smear_classifier` (depends on `faithfulness_probe`)
**Status:** Active — Phase 1 complete, Phases 2-4 pending

---

## Goal

Add a **ternary smear classifier** that extends Plan 278's binary `FaithfulnessProbe` (faithful / unfaithful) to distinguish three classes of latent mass distribution:

1. **`CoherentSingle`** — one dominant hypothesis. Mass concentrated on a single direction at a single site.
2. **`TokenSmear { span }`** — high mass on one direction spread across adjacent *sites* (positions). Benign positional uncertainty. Faithful.
3. **`SequenceSmear { n_hypotheses, semantic_distance }`** — mass split across ≥2 *semantically distinct* directions at one site. Potentially unfaithful: multi-hypothesis superposition requiring disambiguation before commitment.

This is the one extractable primitive from Research 277. The paper names the phenomena; we ship the classifier.

## Why this is a real (Gain-tier) extension, not a duplicate

- Plan 278's `FaithfulnessProbe` is **binary** via causal intervention + attribution.
- MUX (Plan 178) **generates** superposition but does not **classify** it.
- BoMSampler (Plan 281) **samples** K hypotheses but does not **classify** the resulting distribution.
- **No shipped code classifies a superposition distribution into smear types.** This is the gap.

The classifier lets downstream consumers (riir-ai Cognitive Integrity Layer, anti-cheat, sync integrity) react differently to benign positional uncertainty vs potentially-unfaithful multi-hypothesis computation.

## Design

### Types

```rust
/// Smear classification of a latent mass distribution (Research 277, Plan 298).
///
/// Extends Plan 278's binary faithful/unfaithful signal with a vocabulary
/// for *how* the latent mass is distributed, per arXiv:2606.20560 §5.2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SmearClass {
    /// Mass concentrated on a single direction at a single site.
    /// The "faithful single-hypothesis" case.
    CoherentSingle = 0,
    /// Mass on one direction spread across `span` adjacent sites.
    /// Benign positional uncertainty (paper §5.2.1). Faithful.
    TokenSmear = 1,
    /// Mass split across ≥2 semantically distinct directions at one site.
    /// Potentially unfaithful multi-hypothesis superposition (paper §5.2.2).
    SequenceSmear = 2,
}

/// Per-classification detail (for diagnostics + GOAT gate evidence).
#[derive(Debug, Clone, Copy)]
pub struct SmearReport {
    pub class: SmearClass,
    /// Number of distinct sites carrying significant mass (≥1 for TokenSmear).
    pub span: u8,
    /// Number of distinct semantic directions carrying significant mass at the
    /// dominant site (≥2 for SequenceSmear).
    pub n_hypotheses: u8,
    /// Max pairwise cosine distance among the significant directions at the
    /// dominant site. Higher = more semantically distinct = more concerning.
    pub semantic_distance: f32,
}

pub trait SmearClassifier {
    /// Classify a `[k][d]` row-major slice of MUX superposition weights or
    /// BoM K-hypothesis beliefs.
    ///
    /// - `weights` — `[k * d]` flat slice, row-major (k rows of d elements).
    /// - `k` — number of hypotheses/sites.
    /// - `d` — dimensionality of each hypothesis/site.
    /// - `scratch` — caller-allocated scratch buffer of length `k` (norms) +
    ///   `k * (k-1) / 2` (pairwise cosine). Reused across calls; zero-alloc.
    fn classify(&self, weights: &[f32], k: usize, d: usize, scratch: &mut [f32]) -> SmearReport;
}
```

### Decision logic (paper §5.2.1 vs §5.2.2, operationalized)

Given `k` hypothesis vectors `w_0, ..., w_{k-1} ∈ ℝ^d`:

1. **Compute norms** `‖w_i‖` into `scratch[0..k]`.
2. **Threshold**: drop hypotheses with `‖w_i‖ < ε` (configurable, default `1e-3`). Call the survivors `S`.
3. **If `|S| == 1`** → `CoherentSingle`.
4. **Compute pairwise cosine** `cos(w_i, w_j) = (w_i · w_j) / (‖w_i‖ ‖w_j‖)` for `i, j ∈ S` into `scratch[k..]`.
5. **Compute `semantic_distance = max(1 - min cosine)`** — the worst-case pairwise distance.
6. **Decision**:
   - If `semantic_distance < τ_same` (default `0.1` — near-parallel) → **`TokenSmear { span: |S| }`**. The hypotheses are positional variants of the same direction.
   - Else → **`SequenceSmear { n_hypotheses: |S|, semantic_distance }`**. The hypotheses are semantically distinct.

### Why this is correct per the paper

- Paper §5.2.1 token smearing: "model places probability mass for a single token across multiple adjacent positions simultaneously." In our framing: same direction (token), spread across adjacent sites (positions). → `TokenSmear`.
- Paper §5.2.2 sequence smearing: "two or more semantically distinct candidate sequences held in superposition." In our framing: mass split across semantically distinct directions at one site. → `SequenceSmear`.
- The cosine-distance threshold `τ_same` operationalizes "semantically distinct" — the paper uses top-50 cosine neighbors as the criterion; we use a configurable threshold on max pairwise distance.

### Default impl: `CosineSmearClassifier`

```rust
pub struct CosineSmearClassifier {
    pub epsilon: f32,    // norm threshold for "significant" hypothesis. Default 1e-3.
    pub tau_same: f32,   // max pairwise cosine distance for TokenSmear. Default 0.1.
}

impl Default for CosineSmearClassifier {
    fn default() -> Self { Self { epsilon: 1e-3, tau_same: 0.1 } }
}
```

## Phase 1 — Core Skeleton (trait + CosineSmearClassifier + tests)

### Tasks

- [x] **T1.1** Create `crates/katgpt-core/src/faithfulness/smear.rs` with `SmearClass`, `SmearReport`, `SmearClassifier` trait, `CosineSmearClassifier` impl. Behind `smear_classifier` feature (depends on `faithfulness_probe`).
- [x] **T1.2** Implement zero-alloc `classify`: caller passes `&mut [f32]` scratch of length `k + k*(k-1)/2`. Use `simd_dot_f32` for the inner products. No allocations in the hot path.
- [x] **T1.3** Use `#[repr(u8)]` on `SmearClass` per AGENTS.md rule (1-byte enum).
- [x] **T1.4** Unit tests:
  - `coherent_single_one_dominant_direction` — single non-zero hypothesis → `CoherentSingle`.
  - `token_smear_parallel_directions_across_sites` — 3 parallel hypotheses (cosine > 0.9) → `TokenSmear { span: 3 }`.
  - `sequence_smear_orthogonal_directions_one_site` — 2 orthogonal hypotheses → `SequenceSmear { n_hypotheses: 2, semantic_distance ≈ 1.0 }`.
  - `epsilon_filters_low_norm_hypotheses` — hypotheses below `epsilon` are dropped before classification.
  - `tau_same_boundary` — cosine distance exactly at `tau_same` → `TokenSmear` (≤).
  - `deterministic_for_fixed_input` — same input → bit-identical `SmearReport`.
- [x] **T1.5** Feature gate wiring: `smear_classifier = ["faithfulness_probe"]` in `crates/katgpt-core/Cargo.toml` + root `Cargo.toml`. Zero symbols when feature is off (verified via `nm`).

## Phase 2 — FaithfulnessProbe integration

### Tasks

- [x] **T2.1** Extend `DefaultFaithfulnessProbe` (Plan 278) with an optional `smear: Option<Box<dyn SmearClassifier>>` field. When `Some`, the probe's audit cadence additionally classifies the latent mass at the intervention site and emits a `SmearReport` alongside the existing binary verdict.
- [x] **T2.2** The `SmearReport` flows into the existing faithfulness event stream — no new sync dependency, no new chain commit. The report is a diagnostic; the existing `TriggeredInjectionGate` (default-on, Plan 278) decides whether to act on it.
- [x] **T2.3** Document in `.docs/faithfulness_probe.md` that `SmearClass::SequenceSmear` with high `semantic_distance` is the "potentially unfaithful multi-hypothesis" signal that warrants Cognitive Integrity Layer attention (riir-ai `.research/129`).
- [x] **T2.4** *(new)* API decision: chose `probe_intervention_full(...) -> InterventionOutcome<D>` + `faithfulness_profile_full(...) -> FaithfulnessProfileFull<D>` (cleaner than `(Delta, Option<SmearReport>)` tuple — the struct carries field names + doc for the `smear: None` fallback). Trait `FaithfulnessProbe` stays minimal and binary-only — the smear surface is inherent on `DefaultFaithfulnessProbe` (matches the spec's preferred option).
- [x] **T2.5** *(new)* Added `SmearSource` trait for consumers that carry superposition (MUX/BoM). Documented that plain-autoregressive consumers should NOT implement it — the probe returns `smear: None` when the source is absent (correct: they are always `CoherentSingle` by construction).
- [x] **T2.6** *(new)* Unblocked the workspace parse: Plan 299 left two missing-file stub registrations (`examples/engram_demo.rs`, `tests/bench_299_engram_goat.rs`). Created minimal placeholder files so `cargo` can parse the root manifest. Plan 299's agent will overwrite them with the real implementations.

## Phase 3 — GOAT Gate (G1/G2/G3)

The GOAT gate must prove the ternary classification is *useful*, not just correct.

### GOAT Proofs Required

| # | Metric | Threshold | Measurement |
|---|--------|-----------|-------------|
| **G1** | Determinism + correctness | Bit-identical `SmearReport` for fixed input; the 3 hand-constructed cases classify correctly. | Unit tests (T1.4). |
| **G2** | **Useful discrimination** — the ternary classifier produces a measurably different downstream decision than the binary probe on a synthetic workload. | On a synthetic workload with known ground-truth smear types, `SequenceSmear`-flagged interventions have ≥2× the unfaithfulness rate of `TokenSmear`-flagged interventions. | Synthetic harness: generate (a) coherent-single, (b) token-smear, (c) sequence-smear latent distributions; inject each into a Plan 278-style causal intervention; measure unfaithfulness rate per class. |
| **G3** | Latency | `classify(k=8, d=32)` ≤ 200 ns on Apple Silicon arm64 SIMD plasma path. | `benches/smear_classifier_bench.rs` extension. Budget allows 28 pairwise dots of dim-32 = 896 muladds ≈ 150 ns at 6 GFLOP/s scalar, <100 ns SIMD. |

### Tasks

- [x] **T3.1** G1 unit tests — all pass (T1.4 covers this).
- [x] **T3.2** G2 synthetic harness: `tests/bench_298_smear_classifier_goat.rs`. Construct the three smear types via MUX-style superposition weights (deterministic). Inject into a Plan 278-style attribution probe. Measure per-class unfaithfulness rate. Target: `SequenceSmear` rate ≥ 2× `TokenSmear` rate.
- [x] **T3.3** G3 latency bench: `benches/smear_classifier_bench.rs`. Measure `classify` cost at `k∈{2,4,8}`, `d∈{8,16,32}`. Verify plasma-tier budget.
- [x] **T3.4** Honest assessment: G2 passes (ratio 2.11× ≥ 2.0×) and G3 passes (107.6 ns ≤ 200 ns target). The classifier stays opt-in — it's a diagnostic, and the synthetic workload is constructed to make the discrimination measurable. Promotion to default-on would require real-workload evidence (riir-ai Cognitive Integrity Layer integration, deferred to Plan 308/T4.3).
- [x] **T3.5** Written `.benchmarks/298_smear_classifier_goat.md` with before/after evidence and promotion decision.

## Phase 4 — Documentation + cross-refs

### Tasks

- [x] **T4.1** Update `katgpt-rs/README.md` with a `SmearClassifier` entry under the Faithfulness section (new row in the GOAT-Proved Additions table + new subsection under the FaithfulnessProbe showcase). Coordinated with Plan 299 agent: only edited Faithfulness-related content, no Engram sections touched.
- [x] **T4.2** Updated `.docs/faithfulness_probe.md` with the ternary classification vocabulary + wiring guide + paper citations (covered alongside T2.3).
- [ ] **T4.3** **SKIPPED** — the riir-ai `.research/129` cross-repo update is out of scope for this single-repo coding task. Remains TODO for the orchestrator. The katgpt-rs docs already cross-link to `.research/129` and cite arXiv:2606.20560; the riir-ai guide itself needs the vocabulary adoption ("opaque serial depth" + "smearing") + paper citation as external validation of the top-k scalar bridge design.
- [x] **T4.4** Cross-links added in `.docs/faithfulness_probe.md`: Research 277 ↔ Plan 298 ↔ Plan 278 ↔ Plan 178 (MUX) ↔ Plan 281 (BoM). Also cross-linked in `.benchmarks/298_smear_classifier_goat.md`.

## Optimization constraints (per AGENTS.md)

- **Zero-alloc hot path**: `classify` takes caller-allocated scratch; no `Vec` allocation.
- **SIMD**: inner products via `simd_dot_f32`. The K(K-1)/2 pairwise dot loop auto-vectorizes.
- **Fixed-size enum**: `#[repr(u8)]` on `SmearClass` for 1-byte sync-friendly output.
- **Plasma-tier latency target**: <200 ns for k=8, d=32.
- **No new sync dependency**: the classifier reads local latent state (MUX/BoM weights), emits a raw u8 enum. ✅ Raw-at-boundary, latent-locally.

## Risks

| Risk | Mitigation |
|------|------------|
| G2 fails — ternary classification is not measurably better than binary | Demote to opt-in Gain per T3.4. The classifier is still a useful diagnostic even if it doesn't improve downstream decisions. |
| `tau_same` threshold is hard to calibrate | Make configurable per-classifier-instance. Default 0.1 is conservative (near-parallel). Sweep in G2 harness. |
| Cosine distance is expensive for large k | Cap k at 16 in the trait contract. For k>16, subsample or use a cheaper proxy (max-norm hypothesis). Document the cap. |
| Feature interacts unexpectedly with Plan 278's default-on `triggered_injection` | Phase 2 wires `SmearClassifier` as `Option<Box<dyn>>` — `None` by default → zero behavior change when feature is on but classifier is not provided. |

## Dependencies

- Plan 278 (FaithfulnessProbe — the host module).
- Plan 178 (MUX — the superposition generator whose output we classify; soft dep — classifier works on any `[k][d]` slice).
- Plan 281 (BoMSampler — same as MUX; soft dep).
- `simd_dot_f32` from `crates/katgpt-core/src/simd/`.

## TL;DR

Plan 298 adds a **ternary smear classifier** (`SmearClass::CoherentSingle` / `TokenSmear` / `SequenceSmear`) extending Plan 278's binary `FaithfulnessProbe`, distilled from arXiv:2606.20560 (DiffusionGemma transparency paper, Research 277). The classifier distinguishes benign positional uncertainty (token smearing — paper §5.2.1) from potentially-unfaithful multi-hypothesis superposition (sequence smearing — paper §5.2.2) via max pairwise cosine distance among significant hypothesis directions, with a configurable `tau_same` threshold. Zero-alloc, SIMD-backed, `#[repr(u8)]` sync-friendly output. Opt-in behind `smear_classifier` feature (depends on `faithfulness_probe`). **GOAT gate G2 is the load-bearing test:** the ternary classification must produce measurably better downstream decisions than the existing binary probe (≥2× unfaithfulness rate in `SequenceSmear` vs `TokenSmear` on a synthetic harness). If G2 fails, demote to opt-in Gain per T3.4. Research verdict was Gain (not Super-GOAT) — Q1 fails on prior art across all six paper mechanisms; this plan ships the one small diagnostic primitive worth extracting. **riir-ai `.research/129` doc update is the only cross-repo action** (cite paper as external validation + adopt vocabulary; no new guide).
