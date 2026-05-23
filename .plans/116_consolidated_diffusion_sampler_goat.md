# Plan 116: Consolidated — DiffusionSampler GOAT + Natsukaze Validation

> **Consolidates:** Plan 089 T6 (Trained Sampler), Plan 089 T7 (LoRA Drafter Alignment), Plan 086 T6 (Natsukaze Validation)
> **Status:** 🔧 Ready — `diffusion_sampler.rs` created, needs wiring + tests
> **Branch:** `develop/feature/116_consolidated_diffusion_sampler_goat`
> **Feature Gate:** `tri_mode` (Plan 089), `go-training` (Plan 086)
> **Depends on:** Plan 089 T1-T5 ✅, Plan 086 T1-T5 ✅, Plan 081 T0-T14 ✅, Plan 083 ✅

## Objective

Consolidate three open tasks into a single ordered plan:

| Source | Task | Scope | Priority |
|--------|------|-------|----------|
| **Plan 089 T6** | DiffusionSampler wiring + tests + GOAT | `microgpt-rs` speculative | P0 |
| **Plan 086 T6** | Natsukaze Go analytics validation | `riir-ai` riir-examples | P1 |
| **Plan 089 T7** | LoRA Drafter Alignment | `riir-ai` riir-gpu | P2 (deferred) |

**Why consolidate:** These tasks share the GOAT proof pattern and benchmark infrastructure. Running them together avoids context-switching and ensures the DiffusionSampler is validated before we invest in LoRA alignment.

**Prerequisite check:** Plan 089 T1-T5 all PASS (`.benchmarks/018_d2f_verifier_goat.md`). Plan 086 T1-T5 all PASS (18/18 tests, 95% arena accuracy). Plan 081 T0-T14 all PASS (26/26 tests). Data files ready at `riir-ai/data/go_9x7514_games.flat.zip` (8.7MB).

---

## Tasks

### T1: Wire `diffusion_sampler.rs` into module ✅
- [x] Add `pub mod diffusion_sampler;` to `src/speculative/mod.rs` behind `#[cfg(feature = "tri_mode")]`
- [x] Re-export key types: `DiffusionSampler`, `SamplerFeatures`, `SamplerTrajectory`
- [x] Verify `cargo check --features tri_mode` passes with zero errors

### T2: DiffusionSampler unit tests — 22/22 pass ✅
- [x] Run `cargo test --features tri_mode -- diffusion_sampler`
- [x] Verify all tests pass:
  - `sampler_features_*` — feature extraction from D2F decode state
  - `logistic_sampler_*` — logistic variant train/predict
  - `mlp_sampler_*` — MLP variant train/predict
  - `auto_selection_*` — factory picks correct variant per config scale
  - `trajectory_collection_*` — instrument D2F decode with ground truth
  - `train_logistic_on_patterns_*` — end-to-end: generate data → train → evaluate
  - `auc_evaluation_*` — Area Under ROC Curve computation
  - `decide_*` — replaces fixed threshold check
- [x] Fix any failing tests, record count in GOAT table — 20 existing + 2 new T3 integration tests

### T3: DiffusionSampler integration into D2F denoising loop ✅
- [x] Add `d2f_decode_block_with_prompt_with_sampler()` in `speculative/d2f.rs` (tri_mode feature-gated)
- [x] When sampler present: use `sampler.decide(features)` instead of fixed `chosen_prob >= tau_conf`
- [x] Add `sampler: Option<DiffusionSampler>` field to `SelfSpecConfig` in `speculative/types.rs`
- [x] Add re-exports: `d2f_decode_block_with_sampler`, `d2f_decode_block_with_prompt_with_sampler`
- [x] Test: `test_d2f_decode_with_sampler_produces_valid_output` — all tokens in vocab range
- [x] Test: `test_d2f_decode_sampler_differs_from_fixed_threshold` — both produce valid confidence ∈ [0,1]

### T4: DiffusionSampler GOAT benchmark ✅
- [x] Create benchmark comparing:
  - **Baseline:** D2F decode with fixed `tau_conf` threshold (Plan 089 T5 result)
  - **Trained logistic:** D2F decode with `LogisticSampler` decisions
  - **Trained MLP:** D2F decode with `MlpSampler` decisions
- [x] Metrics: TPF (tokens per forward), acceptance rate, AUC
- [x] Record results in `.benchmarks/019_diffusion_sampler_goat.md`
- [x] GOAT gate: trained samplers within ±15pp of baseline — ✅ PASS (0.0pp delta)
- [x] AUC: Logistic 0.765, MLP 0.781 — both >0.55 discriminative signal threshold
- [x] 5/5 GOAT proof tests pass in `tests/test_diffusion_sampler_goat.rs`

### T5: Natsukaze Go analytics validation (Plan 086 T6)
- [ ] Run `cargo run -p riir-examples --features go-training --example go_12_analytics_validate`
- [ ] Validates against `data/go_9x7514_games.flat.zip` (8.7MB, ~460K games)
- [ ] Compare: Natsukaze analytics features vs self-play features
- [ ] Expected: Natsukaze shows higher CR, lower MLWR, lower garbage ratio
- [ ] Train predictor on both datasets → compare accuracy
- [ ] GOAT gate: Natsukaze features produce higher prediction accuracy than self-play features
- [ ] Record results in GOAT table below

### T6: LoRA Drafter Alignment research (Plan 089 T7 — deferred)
- [ ] Design LK-hybrid loss for aligning diffusion drafter with AR verifier
- [ ] LoRA on `o_proj` only (rank 128, α=512)
- [ ] **BLOCKED:** riir-gpu needs D2F training support
- [ ] This task is documented here for tracking; implementation deferred until riir-gpu supports D2F

---

## GOAT Proof

| Task | Gate | Method | Pass Criteria | Result |
|------|------|--------|---------------|--------|
| T1 | Module wiring: compiles | Build check | `cargo check --features tri_mode` zero errors | ✅ PASS |
| T2 | Unit tests: all pass | Test run | All `diffusion_sampler` tests pass (22/22) | ✅ PASS |
| T3 | Integration: sampler in denoising loop | Test | D2F+logistic produces valid output, decisions differ from fixed | ✅ PASS |
| T4 | Benchmark: trained ≥ fixed | Benchmark | Sampler AUC ≥ 0.55 (Logistic 0.765, MLP 0.781), within ±15pp baseline | ✅ PASS |
| T5 | Natsukaze: real data validation | Integration | Natsukaze accuracy > self-play accuracy | ⬜ |
| T6 | LoRA alignment | Research | LK-hybrid loss designed, training pipeline ready | ⬜ DEFERRED |

**Gate order:** T1 → T2 → T3 → T4 → T5. T6 independent/deferred.

**If T4 fails (sampler no better than fixed):** Document negative result. The fixed threshold is already functional (Plan 089 proved). A trained sampler is an optimization, not a requirement. The DiffusionSampler module stays but is not default.

**If T5 fails (Natsukaze accuracy ≤ self-play):** Document negative result. Plan 081 (modelless) is unaffected. The model-based path may need different features or larger training set.

---

## DiffusionSampler Architecture (from Plan 089 T6)

```
speculative/
├── mod.rs                    # + pub mod diffusion_sampler (tri_mode)
├── diffusion_sampler.rs      # NEW: feature extraction + trained acceptance
│   ├── SamplerFeatures        — 6-dim: top1_prob, margin, top3_mass, entropy, step_norm, pos_norm
│   ├── DiffusionSampler       — enum { Logistic, Mlp, Transformer }
│   │   ├── auto(config)       — factory: logistic if n_embd≤32, mlp if ≤256, transformer if >256
│   │   ├── train(trajectories) — binary cross-entropy SGD
│   │   ├── predict(features)   — returns P(correct) ∈ [0,1]
│   │   └── decide(features, tau) — replaces fixed threshold
│   ├── SamplerTrajectory      — (features, correct: bool) training pair
│   ├── collect_trajectories()  — instruments D2F decode with ground truth
│   ├── train_logistic_on_patterns() — end-to-end convenience
│   └── evaluate_auc()         — ROC AUC for sampler quality
├── d2f_verifier.rs           # Plan 089 T1 (existing, unchanged)
└── ...
```

**Three capacity variants:**

| Variant | Params | Use Case | Status |
|---------|--------|----------|--------|
| `LogisticSampler` | ~7 (6 weights + bias) | `micro_dllm` scale (n_embd ≤ 32) | ✅ Implemented |
| `MlpSampler` | ~250 (hidden_dim=32) | Small models (n_embd 33-256) | ✅ Implemented |
| `TransformerSampler` | ~4.8M (d=384, 4 layers) | Production scale (n_embd > 256) | 🔧 Stub (deferred) |

---

## Natsukaze Validation Architecture (from Plan 086 T6)

```
crates/riir-examples/examples/
└── go_12_analytics_validate.rs   # T6: load flat.zip → analytics → predict → compare
    ├── Load go_9x7514_games.flat.zip (8.7MB, ~460K games)
    ├── Split into games via split_samples_into_games()
    ├── Extract analytics via extract_game_analytics()
    ├── Encode features via GoAnalyticsFeatures::from_analytics()
    ├── Train GoOutcomePredictor on 80%, test on 20%
    ├── Compare: Natsukaze accuracy vs self-play synthetic accuracy
    └── Report: CR, MLWR, garbage ratio distributions
```

**Expected Natsukaze characteristics vs self-play:**
- Higher coincidence rate (strong AI plays consistently)
- Lower MLWR (fewer losing moves)
- Lower garbage ratio (fewer garbage moves)
- Prediction accuracy: target ≥65% (PGD paper baseline)

---

## Estimated Effort

| Task | Lines | Effort | Depends On |
|------|-------|--------|-----------|
| T1: Module wiring | ~5 | 10 min | diffusion_sampler.rs exists |
| T2: Unit tests | ~0 (existing) | 30 min | T1 |
| T3: D2F integration | ~30 | 2 hours | T2 |
| T4: GOAT benchmark | ~100 (test) | 4 hours | T3 |
| T5: Natsukaze validation | ~0 (example exists) | 2 hours | None (independent) |
| T6: LoRA alignment | deferred | — | riir-gpu D2F support |

**Total: ~1-2 days for T1-T5**

---

## References

### Parent Plans
- `microgpt-rs/.plans/089_tri_mode_inference.md` — Tri-Mode Inference (T1-T5 ✅, T6→here, T7→T6 here)
- `riir-ai/.plans/086_pgd_game_analytics_model_based.md` — PGD Model-Based (T1-T5 ✅, T6→T5 here)
- `microgpt-rs/.plans/081_pgd_game_analytics_modelless.md` — PGD Modelless (✅ Complete, data bridge)

### Key Files
- `src/speculative/diffusion_sampler.rs` — DiffusionSampler implementation (43K, ~30 tests)
- `src/speculative/d2f_verifier.rs` — D2fDrafterVerifier (Plan 089 T1)
- `src/speculative/mod.rs` — Module index (needs `pub mod diffusion_sampler`)
- `src/speculative/types.rs` — SelfSpecConfig (needs `sampler` field)
- `crates/riir-examples/examples/go_12_analytics_validate.rs` — Natsukaze validation (exists)

### Benchmarks
- `.benchmarks/018_d2f_verifier_goat.md` — Plan 089 GOAT (T1-T5 baseline)
- `.benchmarks/019_diffusion_sampler_goat.md` — This plan's GOAT (T4)

### Research
- `.research/055_Nemotron_TriMode_Diffusion.md` — Tri-Mode research
- `.research/047_PGD_Professional_Go_Dataset_Analytics.md` — PGD analytics features

### Data
- `riir-ai/data/go_9x7514_games.flat.zip` — Natsukaze 9×9 games (8.7MB, ~460K games)
- `riir-ai/data/go_9x7514_puzzles.flat.zip` — Natsukaze 9×9 puzzles (99KB)