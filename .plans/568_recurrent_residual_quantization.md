# Plan 568: Recurrent Residual Quantization (RRQ) — Single-Checkpoint Multi-Precision Weights

**Date:** 2026-08-06
**Research:** [katgpt-rs/.research/467_Recurrent_Residual_Quantization.md](../.research/467_Recurrent_Residual_Quantization.md)
**Source paper:** [arXiv:2608.04048](https://arxiv.org/abs/2608.04048) — Luo, Dong, Cheng, Shen (Intel), "Recurrent Residual Quantization: A Progressive Multi-Precision Representation for LLMs", Aug 2026
**Target:** `katgpt-rs/crates/katgpt-core/src/rrq_quant.rs` (new module) + Cargo feature `rrq_quant`
**Status:** Active — Phase 1 + Phase 2 + Phase 3 COMPLETE (2026-08-07); Phase 3 G2 = honest negative (fused LUT kernel is correct substrate but not a perf win); Phase 4 is P3 (no concrete consumer today)

---

## Goal

Ship a **modelless, calibration-free, single-checkpoint multi-precision weight quantization primitive** behind feature flag `rrq_quant`, default-off. The primitive represents an LLM weight matrix as `W̃(t) = Ŵ0 + Σ_{k=1..t} R̂k` — a low-bit quantized base plus a sequence of 2-bit quantized residual corrections. Each prefix of stages is a usable model at a distinct effective bit-width (2/4/6/8-bit for the default 2+2+2+2 config).

**Why now, why default-off:** zero shipped multi-precision weight representation exists in our stack (Research 467 §2.1 confirms zero prior art for `RRQ | residual_quant | Matryoshka | MatGPTQ | multi-precision weight`). The closest cousins — `quant_error_lora.rs` (single SVD correction), QJL residual in TurboQuant (activation level), and `multi_precision_npc` (FAILED training-time, now in riir-train) — all cover related but distinct problems. **No concrete consumer needs this today.** Ship the primitive + benchmark, default-off, revisit when (a) we serve a multi-precision LLM at runtime, (b) per-NPC expert routing (`quant_expert_goat.rs`) wants to share a multi-precision base, or (c) the §3.4 freeze/thaw incremental-upgrade fusion finds a consumer.

**GOAT gate (promotion criterion):**
- **G1** (correctness): prefix-t reconstruction is bit-identical to a reference sum-of-stages; load-time PMR selector classifies Llama (mild) vs Qwen (severe) outlier profiles correctly. **Phase 1 + Phase 2 DONE.**
- **G2** (perf): fused 4-stage LUT dequant+dot at parity (within 1.05×) with single-stage 8-bit LUT path at the 8-bit prefix. **Phase 3 HONEST NEGATIVE — FAILS at 6.374×** (4× gather overhead dominates the L1-residency benefit at realistic LLM-tile scale). The arithmetic `prefix_dot_into` remains the recommended hot path; the LUT kernel stays as correct substrate (G1 PASS).
- **G3** (no-regression): default features + `--features rrq_quant` both build clean; clippy zero warnings; existing tests unchanged.
- **G4** (alloc-free): prefix-t reconstruction + dot have 0 steady-state allocations.
- **Promotion to default-on: REQUIRES a concrete consumer.** Default-off until then.

---

## Phase 1 — Unblocking Skeleton (CORE, P0)

### Tasks

- [x] **T1.1** Create `katgpt-rs/crates/katgpt-core/src/rrq_quant.rs` behind `#[cfg(feature = "rrq_quant")]`. Add `pub mod rrq_quant;` to `lib.rs` behind the same gate. Add `rrq_quant = []` to `[features]` in `katgpt-rs/crates/katgpt-core/Cargo.toml`.
- [x] **T1.2** Define the storage type: `RrqStage` (2-bit packed codes `Vec<u8>` + per-group `scales: Vec<f16>` + `zero_points: Vec<f16>` + `n_elements` + `group_size`) + `RrqWeights` (base + `Vec<RrqStage>` residuals + rows/cols).
- [x] **T1.3** Implement `RrqWeights::from_weights_rtn(weights, rows, cols, n_stages, group_size)` — pure RTN, calibration-free. Per the paper Algorithm 1: base = RTN of weights; for k=1..=n_stages: residual = full − recon; stage = RTN of residual; recon += stage_dequant.
- [x] **T1.4** Implement the inference primitives: `prefix_reconstruct_into(t, out)` (additive sum of stages) + `prefix_dot_into(t, x, out, scratch)` (exploits matmul linearity: x·W̃(t) = Σ per-stage GEMVs).
- [x] **T1.5** Implement `RrqStage::dequant_into(out)` + `RrqStage::dot_acc_into(cols, x, out)` (accumulate one stage's contribution).
- [x] **T1.6** **G1 gate tests** — **DEVIATION (documented):** G1 tests live in lib unit tests (`src/rrq_quant.rs::tests`, 7 tests) rather than `tests/rrq_quant_goat.rs`. This matches the codebase convention (most primitives put G1 correctness in `mod tests`; only G4 alloc-count goes in a separate binary because it needs the global `CountingAllocator`). The 7 G1 tests:
  - `g1_stage_quantize_matches_reference`: RTN quantize matches a hand-rolled reference (codes + scale + zero_point).
  - `g1_code_packing_roundtrip`: 2-bit packing/unpacking round-trips; dequant is monotone on monotone input.
  - `g1_prefix_reconstruct_matches_reference`: prefix-t reconstruction is bit-identical to an independent sum-of-stages reference path.
  - `g1_dot_matches_reconstruct_then_dot`: `prefix_dot_into(t, ...)` matches reconstruct-then-matmul within 1 ULP.
  - `g1_more_stages_lower_error`: more residual stages → monotonically lower reconstruction error; 8-bit error < 50% of 2-bit error.
  - `g1_constant_weights_exact`: constant weights → zero residual → exact reconstruction.
  - `g4_prefix_dot_smoke_zero_vec_growth`: smoke test (100 calls, no panic) — the real alloc-count gate is T1.7.
- [x] **T1.7** **G4 alloc-free test** (`tests/rrq_quant_alloc_check.rs`): thread-local `CountingAllocator`; 1000 steady-state `prefix_dot_into` + `dot_acc_into` calls; **0 allocations after warmup**.
- [x] **T1.8** **G3 no-regression**: `cargo clippy -p katgpt-core --all-targets --features rrq_quant` zero warnings; `cargo test -p katgpt-core --lib` (default features, 1840 passed) + `--features rrq_quant` (1847 passed = 1840 + 7 new) — zero regressions.
- [-] **T1.9** Update `katgpt-rs/README.md` Feature Showcase — **DEFERRED.** The README feature showcase is a curated marquee list; RRQ is opt-in with no consumer, so adding it now would be premature. Revisit when promotion to default-on lands (requires a concrete consumer per the GOAT gate).
- [x] **T1.10** Commit on `develop` with `feat:` prefix.

---

## Phase 2 — Load-Time PMR + KS Quant-Strategy Router (P1)

### Tasks

- [x] **T2.1** Implement `peak_to_mean_ratio(weights: &[f32], group_size: usize) -> f32` — paper §3 PMR metric: `max|x| / mean|x|` per group, **max across groups** (the conservative worst case — flag a layer if ANY group has severe outliers). Degenerate all-zero group → 1.0 (flat). Zero allocation (single pass, no scratch). Also ships `PMR_THRESHOLD_2_2 = 9.0` (paper §3.4 default for the 2-bit base + 2-bit residual split, `K > 9r`).
- [x] **T2.2** Implement the strategy router (`QuantStrategy` enum + `select_quant_strategy`). **DEVIATION (documented):** the KS D-statistic is **consumed as a scalar parameter**, NOT recomputed here. Substrate-first gate found the existing `katgpt_spectral::ks_d_statistic` (Plan 224 OAQG substrate, Research 200) — `katgpt-core` is the leaf and must not depend on `katgpt-spectral`, so the router takes `ks_d_stat: f32` as input and the caller bridges the value (dependency inversion). Ships `KS_FLAG_THRESHOLD = 0.25` + `DEFAULT_DIRECT_RTN_BITS = 4`. Decision table: KS > 0.25 → FlagForReview (security override); else PMR > threshold → Rrq { n_stages: DEFAULT_N_STAGES }; else DirectRtn { bits: 4 }.
- [x] **T2.3** **G1 gate test** for the selector (7 new lib unit tests):
  - `g1_pmr_uniform_distribution_is_one`: all |x| equal → PMR = 1.
  - `g1_pmr_single_spike_equals_group_size`: one spike among zeros → PMR = n (finite, = group size).
  - `g1_pmr_all_zero_group_is_flat`: 0/0 → 1.0.
  - `g1_pmr_takes_max_across_groups`: flat group (PMR 1) + spike group (PMR 4) → max = 4.
  - `g1_pmr_classifies_llama_vs_qwen`: synthetic Llama-like (outlier factor 5, PMR ~5 < 9) → DirectRtn; synthetic Qwen-like (outlier factor 30, PMR ~24 > 9) → Rrq. The paper's Table 9 K/MAE numbers (26.5 / 116) are a different normalization and serve as context, not exact targets — the synthetic distributions are tuned to our PMR scale.
  - `g1_ks_overrides_pmr`: KS > 0.25 → FlagForReview regardless of PMR (Qwen-like profile, would otherwise be Rrq).
  - `g1_router_boundary_ks_exactly_at_threshold_is_not_flagged`: KS == 0.25 (strict `>`) → falls through to the PMR decision.
- [x] **T2.4** Commit on `develop`. **G3:** 1840 (off) / 1854 (on, +14 new = 7 Phase 1 + 7 Phase 2), zero regressions; clippy zero warnings (off + on). **G4:** the existing Phase 1 alloc-check test still passes; the router + PMR run once at load (not hot path), so G4 alloc-free does not strictly apply, but both functions are zero-allocation by construction (single pass over the slice, no Vec growth).

---

## Phase 3 — Fused Multi-Stage LUT Dequant+Dot Kernel (P2)

### Tasks

- [x] **T3.1** Extend `simd_lut_dequant` (Plan 452) with a multi-stage variant: `dequant_dot_via_lut_multi_stage_slice` (slice-based core) + `dequant_dot_via_lut_multi_stage<L: QuantLut>` (typed-LUT wrapper) + scalar/NEON/AVX2 backends. The kernel sums N stage contributions in registers: `acc += (Σ_k lut_k[code_k[i]]) · x[i]` — single FMA per element after summing stages. Codes are pre-unpacked raw indices (no shift/mask); the caller (RRQ) unpacks 2-bit packed codes once at load via `RrqStage::codes_unpacked_into`. Also shipped: `RrqStage::group_lut_at(g) -> [f32; 4]` (builds the RRQ-affine LUT: `lut[code] = zp + code·scale`, handling the degenerate `scale=0` case naturally unlike `QuantLut::build`) + `RrqWeights::prefix_dot_lut_into` (gated by BOTH `rrq_quant` + `simd_lut_dequant`; requires `cols` divisible by `group_size` — the common LLM case; falls back to `prefix_dot_into` otherwise).
- [x] **T3.2** **G2 latency gate — HONEST NEGATIVE RESULT.** The test `g2_4stage_lut_vs_single_8bit_documented_negative` measures the 4-stage 2-bit LUT path vs single-8-bit LUT path at the 8-bit prefix (256×256 matrix, gs=128, 200 iterations, release mode, Apple Silicon NEON). **Result: 4-stage LUT is ~6× SLOWER** (84560ns vs 13266ns; ratio 6.374×; gate was ≤ 1.05×). Root cause: the 4× code-read + 4× gather overhead dominates the L1-residency benefit. At gs=128, a 256×256 matrix needs only 2 LUTs per row (2KB total) — well within L1, so there is no spilling to amortize. The hypothesis ("amortized gather from tiny 4-entry LUTs staying hot in L1 while the 1KB 8-bit LUT spills") fails because the 8-bit LUTs DON'T spill at realistic tile scales. The negative is structural: the only regime where the hypothesis could hold (>32KB of LUT data → tens of thousands of groups) requires matrices far larger than any single-layer tile, and at that scale cold-LUT gather latency hurts both paths equally. **Consequence:** the fused multi-stage kernel is correct substrate (G1 PASS: 5 new tests — multi-stage matches scalar reference, single-stage≡single-stage-kernel, zero-stages→0, empty→0, alignment boundaries) but NOT a perf win. `prefix_dot_into` (arithmetic path from Phase 1) remains the recommended hot path. The LUT path stays available as opt-in substrate for consumers whose LUT construction is already amortized (e.g. a future hardware StreamDQ analog where the gather is near-memory). The G2 test passes (documents the negative rather than panicking) so the ratio is visible in CI logs. This matches the plan's risk-table mitigation: "Honest negative result. Phase 1 still ships the additive primitive (just not the fused kernel)."
- [x] **T3.3** Commit on `develop`. **G1:** 5 new simd_lut_dequant tests + 3 new rrq_quant tests (group_lut_matches_rrq_affine, codes_unpacked_roundtrip, prefix_dot_lut_matches_arithmetic). **G3:** 1845 (default) / 1862 (rrq_quant+simd_lut_dequant) lib tests pass, zero regressions; clippy zero warnings. **G4:** existing alloc-check still passes; the LUT path is zero-alloc by construction (per-group `[f32;4]` LUTs on stack, kernel accumulator register-resident).

---

## Phase 4 — Prefix-t as Tier Dispatch (STRETCH, P3, deferred until consumer)

### Tasks

- [-] **T4.1** (DEFERRED) Wire `RrqWeights::prefix_dot_into(t, ...)` into a tier dispatch: Plasma tier (2-bit base only) → Hot tier (+1 stage, 4-bit) → Warm tier (+2 stages, 6-bit). Same checkpoint, three tiers, the tier transition is "include one more stage in the sum".
- [-] **T4.2** (DEFERRED) Freeze/thaw integration (riir-neuron-db `MerkleFrozenEnvelope`): each stage is its own shard, the prefix-t view is a runtime composition. This is the Super-GOAT angle from Research 467 §3.4 — only pursue when a consumer needs incremental precision upgrades.
- [-] **T4.3** (DEFERRED) G3 no-regression on existing tier tests.

**Phase 4 deferral rationale:** no concrete consumer needs incremental precision upgrades today. The `quant_expert_goat.rs` per-expert precision routing uses fixed precision per expert; the per-NPC personality divergence story is handled by `CommittedFieldBlend` (different axis). Revisit when one of those consumers wants to share a multi-precision base, or when we serve a multi-precision LLM at runtime.

---

## Open questions / risks

| Risk | Impact | Mitigation |
|---|---|---|
| No consumer materializes; primitive rots unused | Low | Default-off; benchmark proves it works; revisit at quarterly audit. Cost is one feature flag + ~300 LOC. |
| Stage-compounded scale overhead makes RRQ larger than direct 8-bit at the 8-bit prefix | Low | Paper Appendix G shows ~4–5% larger than MatGPTQ, which is acceptable for the multi-precision capability. Document in the GOAT gate. |
| Fused multi-stage LUT path slower than single 8-bit LUT (T3.2 FAIL) | Low | Honest negative result. Phase 1 still ships the additive primitive (just not the fused kernel). |
| PMR threshold (paper §3.4, `~9r` for 2+2) doesn't generalize to our codecs | Low | Phase 2 benchmark calibrates against our existing codecs; the threshold is config, not hardcoded. |
| Small-kernel parameter paradox (Research 463 §2.4.1) applies to RRQ too | Medium | On small CNNs (Moka-scale), each 2-bit residual stage adds 0.5 bits/weight; for a 32×288 conv that's substantial. RRQ is substrate for larger models (LLM weights, future game networks) — same scope caveat as `quant_error_lora`. Document in the module doc. |

---

## Out of scope

- SignRoundV2 learned base (training-method artifact; all-RTN variant is the modelless path)
- GPTQ/AWQ/OmniQuant heterogeneous stage variants (paper §5.4 representationally supported but not empirically evaluated)
- Matryoshka / MatGPTQ nested bit slicing (RRQ explicitly replaces this)
- NVFP4 / FP4 stages (format-specific to NVIDIA Blackwell; verdicted Pass in Research 439)
- riir-train follow-up (RRQ is PTQ, no training; §3.5 check moot)

---

## References

- [Research 467](../.research/467_Recurrent_Residual_Quantization.md) — the parent research note
- [arXiv:2608.04048](https://arxiv.org/abs/2608.04048) — the source paper
- `Plan 452` — SIMD LUT fused dequant+dot (Phase 3 substrate)
- [Plan 100](100_block_diagonal_rotation_quantization.md) — RotorQuant / PlanarQuant / IsoQuant (QJL residual cousin)
- [Research 200](../.research/200_Quantization_Outlier_Collapse_Security.md) — KS D-statistic detector (Phase 2 sibling)
- [Research 463](../.research/463_moka_freeze_thaw_lever_audit.md) — `quant_error_lora` (closest cousin; same `E = W − dequant(W_q)` problem, SVD mechanism)
- [Research 020](../.research/020_TurboQuant_Online_Vector_Quantization.md) — TurboQuant (QJL residual at activation level)
- [Research 418](../.research/418_StreamDQ_SIMD_LUT_DeQuant.md) — StreamDQ → SIMD LUT DeQuant (Phase 3 kernel substrate)
