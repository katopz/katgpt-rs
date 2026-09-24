# Issue 879 — MAttr budget primitives: `exact_mass_admit` exact-mass admission + log-frontier budget tracker (POC)

**Status:** OPEN — filed 2026-09-24 from [Research 584](../.research/584_Matryoshka_Attribution_Sigmoid_Topk_Budget_Primitives.md) (arXiv:2609.25518 "Matryoshka attribution" distill). Two small katgpt-core primitives; feature-gated, GOAT gate required before any default promotion. (Rev 2 per verdict round: renamed off the `gate_sigmoid_topk` collision; out-param API; 4th consumer seam.)

## What

Two modelless budget primitives distilled from MAttr (paper math is public; no code copied — the upstream repo carries no license):

1. **`exact_mass_admit` — exact-cardinality sigmoid admission** (the paper's "sigmoid top-k" operator). Given scores `S: &[f32]`, budget `k: f32`, temperature `T`: bisect `τ` over `[min(S)−10T, max(S)+10T]` (fixed ≤50 iters, branch-free `where`-style updates) until `Σ expit((S−τ)/T) ≈ k`; emit `mᵢ = expit((sᵢ−τ)/T)`. Properties to pin in tests: **sum-to-k** (within bracket tolerance), **shift-invariance** (S+c → identical mask — only relative scores matter), **nestedness** (mask at k+1 ≥ mask at k, elementwise), zero-alloc.
   **Naming (load-bearing):** `katgpt-spectral` already ships `gate_sigmoid_topk`/`gate_sigmoid_topk_into` (Plan 279 family) = per-expert independent sigmoids + selection-sort **hard cut**, uncalibrated mass. One token apart, opposite mass semantics — this primitive is `exact_mass_admit`/`exact_mass_admit_into` everywhere, and the module doc states the disambiguation.
2. **`log_frontier` tracker — modelless budget auto-tuner** (from upstream `schedules.py::AdaptiveLogK`; not in the paper text). Single scalar `k_max_log`; sample k log-uniformly in `(1, k_max]`; at probe_frac=0.25 probe exactly `k_max` and take a **sign step** `±lr` toward `acc == target`; clamped to `[log floor, log total]`. Consumes ONLY a scalar accuracy — offline calibration sweeps today (KV `DensityBudget` ladder boundaries, `thermal_lod` attention_k tier elbows), runtime quality signals later.

## Why

- The workspace ships no exact-cardinality soft selection (`Σσ = k`): `entropic_tilt` = KL/information budget on softmax advantages; `set_admission` = greedy DPP + θ-ladder; `argtopk` = hard cut; `block_topk` (katgpt-attn) and `gate_sigmoid_topk` (katgpt-spectral) = hard cut + **uncalibrated** sigmoid weights. Consumer seams (each measured in the GOAT gate, not assumed): (a) **`katgpt-spectral::gate_sigmoid_topk` / `quantile_balance_router::route_with_bias`** — in-repo, cheapest: calibrated-mass upgrade posture for the router gates; (b) `katgpt-attn/block_topk` calibrated gate mass; (c) `cs_kv_probe::GatedKvSlice` exact-k log-bias sweep; (d) `memory_soup` SSC top-k gates (riir-ai, cross-repo consumer posture).
- No adaptive budget controller ships anywhere (`DensityBudget::k_for` is a static dial). The tracker is ~30 LOC, zero-alloc, and is the modelless extraction of MAttr's `AdaptiveLogK`.

## Tasks

- [ ] **T1** `katgpt-core` module `exact_mass_admit` behind feature `exact_mass_admit`: zero-alloc bisection; **out-param API** (the `gate_sigmoid_topk_into`/`argtopk_with_scratch` house pattern — no owned-array returns): `exact_mass_admit_into(scores: &[f32], k: f32, T: f32, out: &mut [f32]) -> f32` (returns τ) + an allocating convenience wrapper; tests: sum-to-k, shift-invariance, nested-in-k, edge cases (k=1, k=N−1, degenerate scores), no-alloc (`#[cfg(feature = "alloc_tracking")]` guard).
- [ ] **T2** `katgpt-core` module `log_frontier` (same feature or nested): `LogFrontier::new(total, target, lr, probe_frac, floor)`, `sample(&mut self, rng) -> f32`, `observe(&mut self, acc: f32)`; tests: sign-step direction both sides of target, clamping, determinism under seeded rng, probe cadence.
- [ ] **T3** GOAT gate (G1–G4): correctness (T1/T2 tests), perf bench vs the hard-cut+sigmoid baseline family (`gate_sigmoid_topk_into` and `argtopk`+per-element-sigmoid) at N ∈ {1e3, 1e5, 1e7}, no-regression (consumers unchanged with feature off), alloc-free. **Promotion decision per bench outcome — no assumed win** (the calibration gain is the hypothesis, not a result).
- [ ] **T4** If G2 shows no consumer gain: record negative result here, keep primitives opt-in, close.

## Cross-refs

- Fusion consumers: riir-clippy [Issue 133](../../riir-clippy/.issues/133_matryoshka_k_supervision_rule_embed.md) (rule_embed reopen lane); riir-train parameter-delta attribution fusion — **UNFILED-PENDING**: the issue file is written on disk at `riir-train/.issues/571_mattr_parameter_delta_attribution.md` (untracked, no highwater bump — that checkout is behind 9 with a concurrent session's dirty `Cargo.lock`; `test -e` is the check; the next riir-train committer lands it with the highwater bump).
- Upstream: `aryamanarora/matryoshka-attribution` @ `360ff6af9f202396131231f59ab6bcc48f609ab0` (no license — read-only distillation, `.raw/` clone removed).
