# Negative Results & Replaced Features

> Features that were researched, implemented, benchmarked, and found to provide no measurable gain.
> Infrastructure kept where reusable for future paths.

## 1. Stepwise Reward Shaping (Plan 054) — NO GAIN

Distilled from [StepCodeReasoner](https://arxiv.org/pdf/2605.11922) (ICML 2026). **Benchmarked, no measurable improvement over flat rewards.** Feature-gated off by default, not in `full`.

| Method | Nodes | PathLen | Goal% | Time |
|--------|-------|---------|-------|------|
| Baseline (BinaryScreen) | 256 | 7 | 100% | 297ms |
| Flat rewards (λ=0) | 256 | 7 | 100% | 356ms |
| **Shaped rewards (λ=0.3)** | **256** | **7** | **100%** | **475ms** |

Same tree, same path, same goal rate — shaped rewards only add +33% latency. The paper's +7-14% gains come from GRPO gradient updates on a 7B model, not from post-hoc reward shaping on a bandit Q-value.

Infrastructure kept for future GRPO integration (G-Zero Phase 2). `stepcode` feature must be explicitly enabled.

Run: `cargo test --features "stepcode" --test bench_stepcode_modelless -- --nocapture`

## 2. δ-Mem Modelless Distillation (Plan 053) — Infrastructure Only

Distills δ-mem's online associative memory (arXiv 2605.12357) into our modelless stack. The delta-rule update `S' = (1-β)S - β(S·k)⊗k + β·v⊗k` is implemented with feature hashing replacing the paper's learned projections.

### Verdict: No DDTree Gain

| Metric | Target | Actual |
|--------|--------|--------|
| DDTree node delta | ≤10% more | 0% ✅ |
| Latency overhead | ≤5% | **+2500%** ❌ |
| Tree quality improvement | ≤5% shorter paths | 0% ❌ |
| Memory convergence | ≤20% error | 18% ✅ |
| Domain isolation | ≤50% interference | 0% ✅ |

**Why no gain:** The paper corrects attention Q/O projections across all layers of a 4B+ param Transformer. We correct a single scalar relevance score in a tree search — the correction surface is too simple. The 26× overhead comes from FeatureHasher + matmul per `relevance()` call (~682 calls/build).

**What works:** Delta-rule math, domain isolation, bounded state, snapshots. **What doesn't:** DDTree quality or latency. The value prop is for Transformer attention correction, not tree scoring.

**Feature gate:** `delta_mem = ["bandit"]` — **off by default**, not in `default` features.

📖 See [`.plans/053_delta_mem_modelless.md`](../../.plans/053_delta_mem_modelless.md) for full plan, [`.research/024_Delta_Mem_Online_Associative_Memory.md`](../../.research/024_Delta_Mem_Online_Associative_Memory.md) for paper analysis.

## 3. SDAR Gated Distillation — Negative Result

Adapts SDAR's token-level sigmoid gating pattern to our modelless distillation stack. Applies asymmetric trust (endorse positive gaps, attenuate negative) to bandit updates and absorb-compress promotions. No gradients — pure modelless signal gating.

### Asymmetric Trust Principle

- Positive gaps (endorsement) → gate opens → strong update signal
- Negative gaps (rejection) → gate closes → attenuated update signal
- Sigmoid gate: `σ(β·x)` with β=5.0 (paper-validated optimum)

### Component Benchmarks (`.benchmarks/008_sdar_gated_modelless.md`)

| Method | Throughput | Hot-path overhead |
|--------|-----------|-------------------|
| `sdar_gate()` (pure sigmoid) | 2.4T/sec | — |
| `SdarBanditPruner::update()` | 118M/sec | ~0% (inlined) |
| `SdarGatedAbsorbCompress::observe()` | 112M/sec | +0.4% (inlined) |

| Benefit ratio targeting (β=5.0) | Promotions | Rate |
|-------------------------------|-----------|------|
| High BR (1.5–2.0) | 195/200 | 97.5% |
| Neutral BR (0.9–1.1) | 102/200 | 51.0% |
| Low BR (0.0–0.4) | 0/0 | 0.0% |

### Arena Results (`.benchmarks/010_sdar_arena.md`) — ⚠️ Negative Result

**Bomber** (7 players, 5 matchups × 50 games):

| Rank | Player | ELO | Win% |
|------|--------|-----|------|
| 4 | GZero | 981 | 7.0% |
| 5 | Rubric | 955 | 5.0% |
| 6 | **SDAR** | **954** | **6.0%** |

**FFT** (7 strategies, 42 matchups × 20 games): SDAR draws 100% vs GZero and Rubric (40 games each). Win matrix identical — same action distributions.

**Verdict:** SDAR modelless gating does **not** improve arena performance. The sigmoid gate modulates reward signal intensity (convergence rate), not action selection. In short tournament series, SDAR produces the same action distributions as Rubric and GZero.

The infrastructure (sigmoid gate primitive, bandit wrapper, absorb wrapper) is production-quality and reusable for the gradient-based path (Plan 073).

**Feature gate:** `sdar_gate = []` — off by default.

## 4. RMSD — Relevance-Masked Self-Distillation — NO GOAT

Two-step relevance mask on top of SDAR: pre-filter T=20 actions by |ΔQ| magnitude → select S=5 most informative → only those receive SDAR sigmoid gating. Adds `TeacherContinuation` (student → teacher snapshot on plateau).

### Arena Results (`.benchmarks/037_rmsd_goat.md`) — ❌ NO GOAT

**Bomber** (1000 games, RMSD + Random vs SDAR + Random): RMSD within 10% relative gap of SDAR. Same conclusion as SDAR — the relevance mask affects convergence rate, not action selection.

**Verdict:** RMSD does **not** improve arena performance over SDAR (which itself doesn't improve over GZero/Rubric). Negative arena result = NO GOAT, regardless of 46 structural proofs passing. The two-step filter concentrates learning signal on high-magnitude actions, but in short tournament series both RMSD and SDAR produce the same action distributions.

The infrastructure (relevance filter, magnitude judge, continuation, top-K KL approximation, `rmsd_loss`) is production-quality and reusable for the gradient-based path.

| Component | Throughput | Hot-path overhead |
|-----------|-----------|-------------------|
| `RmsdRelevanceFilter::filter_actions()` | ~50M/sec | — |
| `rmsd_loss()` | ~100M/sec | — |
| `RmsdPlayer::select_action()` | ~10K/sec | +~5% vs SDAR |

46 structural proofs (34 unit + 2 arena + 10 pipeline) — code correctness only, not GOAT. Feature gate: `rmsd_distill` — **off by default**, excluded from `full`.

```rust
use katgpt_rs::pruners::rmsd_relevance::{RmsdConfig, RmsdRelevanceFilter, rmsd_loss};
use katgpt_rs::pruners::bomber::RmsdPlayer;

let player = RmsdPlayer::new(0);

// Or use the filter directly
let filter = RmsdRelevanceFilter::new(20, 5);
let (selected, metrics) = filter.filter_actions(&teacher_q, &student_q);
let loss = rmsd_loss(&selected, &teacher_q, &student_q, 5.0);
```

📖 See `.benchmarks/037_rmsd_goat.md` for full results (NO GOAT — negative arena).
Paper: [Relevance-Masked Self-Distillation](https://www.appliedcompute.com/research/relevance-masked-self-distillation) — Applied Compute, 2026

## 5. Alien Sampler (Plan 311) — GOAT FAILED 2/4

Distills ["The Alien Space of Science" (Artiles et al., arXiv:2603.01092)](https://arxiv.org/abs/2603.01092) into `AlienSampler<V, C, A>`: a within-pool z-scored linear fusion `(1−β)·zC + β·zU` of coherence × unavailability, plus `MedianTopMAvailability` implementing the paper's load-bearing median-of-top-m cosine community-aggregation rule.

### Verdict: 2/4 PASS → DEMOTE (opt-in, not default)

| Gate | Target | Result | Verdict |
|------|--------|--------|---------|
| G1 motif collapse | Arm C / Arm B ≤ 0.50 | 0.5010 (β=0.7) | ❌ BORDERLINE (within 0.2% of threshold) |
| G2 quality preservation | Arm C / Arm A ≥ 0.90 | 0.6722 (β=0.7) | ❌ FAIL |
| G3 perf | C/B ≤ 5.0× | 4.56× (post-rayon, 16 cores) | ✅ PASS |
| G4 latent boundary | no Vec<f32> escapes rank() | type-system enforced | ✅ PASS |

### Why it fails the gate (scenario limitation, not primitive limitation)

The synthetic coherence surface has a **single dominant peak** (archetype 0). This creates a **sharp phase transition at β≈0.4**: either the availability signal is too weak (β<0.4 → concentration=1.0, full collapse) or too strong (β>0.4 → quality drops to 0.65-0.74). **No β satisfies both gates.** The paper's real-world coherence surface (research-paper quality scores) is presumably flatter and multi-modal — multiple "good" research topics with comparable coherence. Transfer to synthetic NPC populations is unvalidated, exactly as the plan's risk register predicted.

### What works (mechanism validated)

At β=0.7, concentration drops from 0.9978 → 0.4999 — a **2× reduction** in motif collapse. The paper's analog was 95.7%→34.3% (2.8× reduction). Same mechanism, slightly weaker effect on this scenario.

G3 was originally 38.42× (FAIL) but closed to 4.56× via rayon NPC-parallelization (commit `60e4e50d`). The primitive ships with correct parallel-friendly architecture (per-NPC cosine scratch, deterministic RNG split).

**Feature gate:** `alien_sampler` — **off by default**, opt-in for paper reproduction and future research on flatter coherence surfaces. SIMD inner-loop optimization is incremental (G3 already closed via rayon; SIMD would be a marginal gain on top).

📖 Plan: [`.plans/311_alien_sampler_primitive.md`](../../.plans/311_alien_sampler_primitive.md), Benchmark: [`.benchmarks/311_alien_sampler_goat.md`](../../.benchmarks/311_alien_sampler_goat.md), Research: [`.research/293_Alien_Science_Coherence_Availability_Frontier.md`](../../.research/293_Alien_Science_Coherence_Availability_Frontier.md)

## 7. Modelless KV Cache Consolidation (Plan 420) — QUALITY GAIN REFUTED

Distills [Bottlenecked Transformers (arXiv:2505.16950)](https://arxiv.org/abs/2505.16950) into a modelless KV cache consolidation primitive that periodically rewrites KV value vectors in-place at surprise-triggered step boundaries, using a deterministic sigmoid-gated mean-shift toward the recent step's value centroid. The IB argument: autoregressive training makes the KV cache minimally compressive of input (high I(X;Z)); periodic consolidation reduces I(X;Z) while preserving I(Z;Y).

### §3.6 Defend-Wrong PoC — QUALITY GAIN REFUTED

| Competitor | Token Acc | NLL |
|---|---|---|
| Baseline (vanilla KV) | 0.0133 | 7.8628 |
| Modelless consolidation | 0.0133 (−0.06pp) | 7.8629 (+0.0001) |
| Random-rewrite control | 0.0133 | 7.8633 (+0.0005) |

**Verdict:** Consolidation ≈ baseline (Δtoken_acc = −0.06pp, ΔNLL = +0.0001). Consolidation ≈ random-rewrite (ΔNLL = +0.0005). The modelless mean-shift has **no effect** on an untrained model — as expected (the IB argument requires a trained model whose KV cache carries learned extraneous detail).

**Hyperparameter sweep:** zero sensitivity to `g_max ∈ {0.1, 0.3, 0.5}`, `k ∈ {16, 32, 64}` — all configs produce identical token_acc (0.0133) and near-identical NLL (7.8627–7.8630).

**riir-train confirmation:** riir-train Plan 313 independently confirmed the refutation on a **TRAINED** model (31% accuracy, 0.00pp gain). The paper's quality benefit is inseparable from its TRAINED Cache Processor; the modelless mean-shift does not capture it.

**No feature flag ships.** Phases 2–4 permanently shelved. The PoC bench (`bench_420_kv_consolidation_poc.rs`) is retained as the negative-result record.

📖 Plan: [`.plans/420_kv_consolidation_modelless.md`](../../.plans/420_kv_consolidation_modelless.md), Research: [`.research/401_Bottlenecked_Transformer_KV_Cache_Consolidation.md`](../../.research/401_Bottlenecked_Transformer_KV_Cache_Consolidation.md)

## 8. Replaced / Fell Behind / No Gain (Full Audit)

| Feature | Source | Verdict | Why |
|---------|--------|---------|-----|

| **TurboQuant** (`turboquant`) | [TurboQuant (Zandieh 2025)](https://arxiv.org/pdf/2504.19874) | **Demoted to legacy baseline** | SpectralQuant dominates at calibrated quality (0.9917 cosine, 9.1× compression). OCTOPUS dominates at data-oblivious quality (0.9870 cosine at 3-bit, -70% MSE vs TQ). TQ kept for comparison/education only (Bench 013, 022). |
| **StepCode** (`stepcode`) | Plan 054 Bi-Level GRPO | **NO GAIN proven** | Mathematically correct but paper's 7-14% gains come from training 7B model on dense stepwise rewards — modelless path only improves heuristic signal quality. Off by default, not in `full`. |
| **δ-Mem** (`delta_mem`) | Plan 053 Associative Memory | **NO GAIN for DDTree** | Delta-rule converges (cosine ≤0.20 error after 200 updates), domain isolation works. BUT: **26× latency overhead** (682 calls/build). Corrections too small to flip branch ordering. |
| **SDAR Arena** (`sdar_gate`) | Plan 072 Asymmetric Trust | **Negative arena result** | ELO 954 ≈ Rubric 955 — no improvement. 28% higher bandit regret. SDAR draws 100% vs GZero and Rubric in FFT. Reward modulation ≠ selection improvement. |
| **RMSD** (`rmsd_distill`) | Plan 125 Relevance-Masked Self-Distillation | **Negative arena result — NO GOAT** | 46/46 structural proofs pass (code correctness), but RMSD within 10% of SDAR over 1000 bomber games — no improvement. Same fate as SDAR: reward signal modulation does not improve action selection. Infrastructure reusable for gradient-based path. |
| **Fast BLT** | [Fast BLT Research 17](https://arxiv.org/abs/2605.09959) | **Explicitly rejected** | Architecture mismatch: we use BPE tokens not bytes, no hierarchical architecture, already have `LeviathanVerifier` for speculative decoding. |
| **AutoTTS** | [AutoTTS Research 16](https://arxiv.org/abs/2605.09959) | **Not implemented** | Manual `tree_budget` in `Config` serves same purpose. β parameterization was planned but never built. |
| **EMO MoE** | [EMO Research 09](https://arxiv.org/abs/2406.08732) | **Concept only** | `domains.toml` exists as placeholder. No `PromptRouter`, no `ExpertRegistry`, no MoE architecture at our model scale. |
| **Attractor Models** | [Attractor Research 35](https://arxiv.org/abs/2605.09959) | **Not implemented** | Fixed-point solver on DDTree already disproved (Plan 053). Bandit refinement serves propose+refine function. |
| **rust-gpu** | [Rust GPU Feasibility Research 29](https://arxiv.org/abs/2605.09959) | **DEFERRED** | Nightly requirement, `spirv-std` API gaps, no CPU fallback. SIMD-first validated instead: ~3.6M tok/s on Apple M-series. |
| **Dual-cutoff** | [FFO Research 30 P1](https://arxiv.org/abs/2605.09959) | **Harmful** | Cutoff=0.2 masks 17/27 arms (-49% relevance), eliminates exploration signal. UCB1 exploration bonus inflates low-Q scores. |
| **KPop Binary KL** | [KPop Research 119](https://ringtech.notion.site/kpop) | **No gain — future reference** | Online RL (GRPO/PPO) train/infer mismatch technique for MoE. We don't do online RL, no MoE, no train/infer split. "70-80% tokens redundant" validates existing pruning philosophy. Stored for future if we add game LoRA online RL. |
| **GDSD Pruner** (`gdsd_distill`) | [GDSD Research 151](https://arxiv.org/abs/2605.08605) | **NO GAIN proven** | GOAT 0/3 gain gates. G1: +0.00% acceptance improvement (identical to baseline). G3: +181.5% overhead (nearly 3× cost). Correct implementation (7/7 structural) but zero measured benefit. |
| **MPNS** (`multi_precision_npc`) | riir-ai Plan 252 T5 | **Negative arena result — NO GOAT** | 12/12 unit tests pass, but arena proves zero quantization robustness advantage. React weights collapse to all -1.0 (ternary kills gradient diversity). Dream weights quantize to identity (same magnitude). Root cause: simplified SGD (`loss * sigmoid(w)`) insufficient. Needs STE + adaptive optimizer. |
| **Alien Sampler** (`alien_sampler`) | Plan 311 Coherence × Availability | **GOAT FAILED 2/4** | G1+G2 fail (β phase-transition at β≈0.4, no β satisfies both motif-collapse and quality). G3 PASS post-rayon (4.56×). G4 PASS. Mechanism validated (2× concentration reduction); domain transfer to synthetic NPC populations unvalidated. Module retained opt-in for paper reproduction. |
| **AC-Prefix** (`ac_prefix`) | Plan 313 Arbitrary-Conditional Prefix | **GOAT PARTIAL — original G1 FAILED, then modelless-unblocked** | G1-original (paper equivalence to iterative-MLM at 1e-4) FAILED at 7.5e-4 on untrained micro-GPT. Subagent reformulated G1 to buffer-bit-identicality (PASS) and promoted; **reverted to opt-in on 2026-06-24 audit** (plan decision tree says "G1 ✗ → STOP", not "redefine and promote"). **Re-promoted to DEFAULT-ON same day** via §3.5 modelless unblock (Issue 003 Phase 0 RESOLVED): Path 2 `attends_dedup` eliminates the doubled-signal bias bit-identically to iterative-MLM on single-layer micro-GPT (0.0 diff). G2/G3/G4 PASS (27.258× speedup, 0 mismatches, 0 allocs). Primitive correct as modelless mask builder; multi-layer equivalence remains a non-blocking riir-train follow-up. |
| **KV Consolidation** | Plan 420 Bottlenecked Transformers | **QUALITY GAIN REFUTED** | §3.6 PoC: Δtoken_acc = −0.06pp, ΔNLL = +0.0001; zero hyperparameter sensitivity. riir-train Plan 313 confirmed on TRAINED model (31% accuracy, 0.00pp gain). Paper's quality benefit is inseparable from TRAINED Cache Processor; modelless mean-shift is inert. No feature flag ships. |

## 9. MCTS State-Action Cache (Plan 451) — OPT-IN-FOREVER (G2 FAILED)

Distilled the UnMaskFork state-action pair cache for MCTS over deterministic inference actions — memoize `(state, action) → child` transitions so repeated rollouts skip re-applying the same `apply()` call. Correct idea for dLLM-scale MCTS where each `apply` is a full forward pass.

**G2 FAILED on re-gate (Issue 044, 2026-07-07).** The synthetic domain's `apply` is a 4-token array write (~ns), so cache hits don't translate to meaningful NFE (number of forward evaluations) savings:

| Sub-gate | Target | Result | Verdict |
|----------|--------|--------|---------|
| **G2a** reward-convergence | ≥1 strict win | 0/6 (both arms converge to 1.000 at NFE=256) | ❌ FAIL |
| **G2b** NFE-savings | ≥1.4× expansion | 1.01–1.03× (avg_rollout_depth 0.6–0.7 — tree reaches terminal depth before cache helps) | ❌ FAIL |

**Root cause:** the synthetic domain is too cheap for cache hits to matter. Only a real dLLM PoC where each `apply` is a full forward pass can show the budget-expansion benefit. Stays **opt-in-forever** until a real dLLM PoC re-validates. Infrastructure (correctness G1 PASS) is reusable. 📖 Plan: [`.plans/451_mcts_state_action_cache_unmaskfork.md`](../../.plans/451_mcts_state_action_cache_unmaskfork.md), Issue 044 (re-gate — file removed per noise-reduction rule).

## 10. SAR × QuasiMoTTo Fusion (Issue 151) — CONCENTRATION REFUTED @ 1.5B

**Fusion hypothesis:** SAR (weight-delta purification, widens the reachable problem set) × QuasiMoTTo (QMC lattice sampling, covers a fixed set with 50% fewer samples) → compound Pass@k gain. The fusion operates at LLM scale (4096×4096 weight matrices) where SAR concentration is supposed to hold (paper's Fig 2, AIME 2024).

**Phase 1 PoC REFUTED concentration at 1.5B scale (2026-07-15).** The foundational assumption — SAR produces a **concentrated** weight-delta spectrum where off-manifold drift is removed — does not hold empirically:

- **Test:** 196 layers of a 1.5B model, measure `on_manifold_fraction` (target >0.8 for concentration to hold).
- **Result:** **0/196 layers exceed 0.8.** The concentration phenomenon claimed by the paper is not reproducible at 1.5B scale in our setup.

Without concentration, the compound gain cannot exist (SAR cannot widen the reachable set if it doesn't first concentrate the delta). Issue closed; the fusion does not survive the PoC gate. This mirrors the earlier `spectral_rewire` G1b failure (Issue 123, NPC-scale ≤64×64, on_manifold_fraction in [0.27, 0.58]) — the concentration phenomenon is not reproducible in either regime. 📖 Plan: [`.plans/423_spectral_rewire_primitive.md`](../../.plans/423_spectral_rewire_primitive.md), PoC result: `riir-train/.scratch/sar_qmc_fusion_poc/PHASE1_RESULTS.md` (Issue 151 closed + removed per noise-reduction rule; git `b7ca596c`).

## 11. SAR Spectral Backdoor Detection (Issue 152) — IMPRACTICAL (FATAL SCOPE)

**Hypothesis:** `spectral_rewire`'s rewiring matrix `M = UᵀΔWV` may expose a spectral signature of a planted backdoor that the backdoor's construction (R422) deliberately hides from uniform-norm tests. R422 proves the backdoor is *statistically undetectable* in TV-distance; the open question was whether SAR's *directional* decomposition breaks that undetectability.

**CLOSED impractical (2026-07-15).** The fatal scope problem (identified during open-questions analysis) is confirmed: SAR operates on a **weight delta** `ΔW`, but a backdoor detector must operate on the **base weights `W`** (you don't have the honest delta to subtract). SAR's purification needs a reference point; a backdoored-in-from-scratch model has no honest baseline to purify against. The detection surface SAR provides exists only on the delta, not the deployed weights. R422's PASS verdict (backdoor is statistically undetectable) stands unbroken. (Issue 152 closed + removed per noise-reduction rule; git `2524918b`.)

## 12. Shard Embedding / JL Projection (Plan 230) — DEPRECATED (Issue 139)

**Hypothesis:** Johnson-Lindenstrauss random orthogonal projection `[f32;64]→[f32;8]` provides O(1) cosine similarity shard lookup at ~90% NN preservation (the JL guarantee).

**DEPRECATED 2026-07-16 (Issue 139).** The projection dimension m=8 is **mathematically unsound** — it violates the Johnson-Lindenstrauss lower bound by over 200×:

- JL requires m ≥ (8·ln n)/ε² for n points at distortion ε. For n=100, ε=0.5: **m ≥ 554**; the code uses **m=8**.
- Empirically measures **1.4–6% NN preservation** vs the documented 90% target.
- **Zero runtime consumers** at deprecation time: SenseModule uses TernaryDir, BFCF uses region centroids.

Option D (deprecate, mark `#[deprecated]`) was chosen over Option B (PCA rescue) because PCA requires a real-data intrinsic-rank measurement that cannot be done modellessly without the corpus existing (per the §3.5 modelless-unblock protocol — a training-dependent measurement path is not a clean modelless gain). `JlProjectionMatrix` + `ShardEmbedding` are kept for back-compat but emit deprecation warnings; bench_230 + diag_230 tests removed. 📖 Plan: [`.plans/230_shard_embedding_projection.md`](../../.plans/230_shard_embedding_projection.md) (close-out note preserves empirical evidence). (Issue 139 closed + removed per noise-reduction rule; git `3e33d7d8`.)

## 13. RoVE — Rotary Value Embeddings Retrofit (Plan 557) — RETROFIT HURTS

**Hypothesis:** Rotary Value Embeddings (arXiv:2606.11275 García-Castellanos/Weiler/Bekkers, Jul 2026) extend RoPE from Q/K to the V projection + inverse-rotate the aggregated output, yielding an attentive convolution ỹ_i = Σ_j A_ij · (R_{j−i} · W_V) · x_j with offset-indexed block-Toeplitz kernel ψ_δ = R_δ·W_V. Parameter-free, FlashAttention-compatible. The open question was whether applying RoVE's V-rotation **at inference time** onto an already-RoPE-trained checkpoint recovers the paper's equivalence.

**RETROFIT HURTS (2026-07-22, Phase 5 A/B on gemma-2-2b-it).** The paper's RoPE↔RoVE equivalence is a **training-time** result: a model must be *trained* under RoVE for the V-rotation to compose correctly with the learned attention pattern. Applying the V-rotation at inference onto a RoPE-trained checkpoint:

- short text (65 tok): loss **+12.5%**
- longer text (162 tok): **+153% perplexity**

All 7 GOAT gates PASS (the substrate is correct + parameter-free + FlashAttention-compatible) — the failure is the *application mode*, not the primitive. **Feature `rotary_value_embedding` stays opt-in for forward-compat** (a future RoVE-trained checkpoint would compose correctly). Implies `position_group_action` (first hot-path consumer of GRAPE's `PositionGroupAction` trait).

📖 Plan: [`.plans/557_rotary_value_embeddings.md`](../../.plans/557_rotary_value_embeddings.md). Benchmark: [`.benchmarks/557_rotary_value_embedding_goat.md`](../../.benchmarks/557_rotary_value_embedding_goat.md) + [`.benchmarks/557_rove_retrofit_poc.md`](../../.benchmarks/557_rove_retrofit_poc.md). Research: [`.research/452_RoVE_Rotary_Value_Embeddings_Attentive_Convolution.md`](../../.research/452_RoVE_Rotary_Value_Embeddings_Attentive_Convolution.md).

## 14. Variable-Rank Domain Expert Clusters (Plan 558) — G2 FAIL (STAYS OPT-IN)

**Hypothesis:** LatentMoE-style variable-rank domain expert clusters (arXiv:2601.18089, Research 453) — `pick_domain` + `project_guided` + `VariableRankRouter` over heterogeneous-rank `CommittedFieldBlend` clusters — distribute per-NPC cognition across move/combat/quest domains with per-cluster rank budgets. PoC confirmed 1.63× entropy gain.

**G2 FAIL — stays opt-in (2026-07-22).** The entropy gain is **real and larger than the PoC**: 2.63× archetype-utilization entropy vs uniform `<3,32>` baseline at iso `K×D=96` compute (G3 PASS, exceeds PoC's 1.63×). But the **latency cost is also real**: ~2× slower per tick (2.224× at 1K NPCs, 1.990× at 10K) — trait-object dispatch (`Box<dyn ErasedCluster>`) + per-NPC `override_pi` virtual calls dominate (~50 ns of the 63 ns overhead vs a 51 ns baseline).

Gate results: G1/G3/G4/G5 PASS; **G2 FAIL** (target ≤1.0×). Absolute numbers (52–104 ns/NPC) are well under the 500 µs/tick MMO budget, and latency scales linearly 1K→10K (MMO-scalable), but the 2× relative cost fails the G2 gate.

**The promotion path** (Plan 558 §Honest Risks #1) is the **macro monomorphization escape hatch** (`variable_rank_router_static!`, Issue 189): generate a per-domain-count monomorphized router enum instead of `Box<dyn ErasedCluster>`, eliminating the vtable tax. Shipped as a macro so consumers opt in to the codegen; the ergonomic `Box<dyn>` path remains for prototyping.

📖 Plan: [`.plans/558_variable_rank_domain_expert_clusters.md`](../../.plans/558_variable_rank_domain_expert_clusters.md). Benchmark: [`.benchmarks/558_variable_rank_domain_expert_goat.md`](../../.benchmarks/558_variable_rank_domain_expert_goat.md). Research: [`.research/453_Variable_Rank_Domain_Expert_Clusters.md`](../../.research/453_Variable_Rank_Domain_Expert_Clusters.md). Monomorphization escape hatch: [`.docs/08_performance/variable_rank_monomorphization.md`](../08_performance/variable_rank_monomorphization.md) + Issue 189.

## 15. Canonical Intent Space — Cross-Arch Super-GOAT PERMANENTLY DEMOTED (Proposal 009 / Research 459)

**Hypothesis:** A canonical intent space — `CanonicalIntent { tag, direction }` + a `ModelAdapter` trait (Procrustes / Subspace / Mask) — would let a steering direction mined on one base model (Gemma2-2B) be re-applied to a different architecture (MiniCPM5-1B, Llama) with no retraining. Plug-and-play any base model. The Super-GOAT headline: swap Gemma → Llama without retraining overlays.

**Cross-arch Super-GOAT claim PERMANENTLY DEMOTED (2026-07-27, Bench 427).** Four hidden-state construction methods all failed the G6 cross-architecture discrimination gate (≥0.5 mean cosine agreement on Rust-idiom direction across architectures; floor is a good system prompt per Research 322 "Report the Floor" rule):

| Phase | Method | Best cross-arch agreement | Threshold | Verdict |
|---|---|---|---|---|
| P3 (Bench 424) | Per-model centroid + Procrustes | **−0.33** | ≥0.5 | ❌ FAIL — Procrustes aligns shape, not location; per-model centroids point in opposite directions after rotation |
| P3 (Bench 424) | Difference-of-means `d_diff` | +0.46 | ≥0.5 | ❌ borderline FAIL — JS discrimination negative on both models (−0.32 / −0.03); apparent signal was a token-count confound |
| P3b (Bench 425) | Intermediate-layer probe (Git Re-Basin hypothesis) | +0.19 (layer 0) | ≥0.5 | ❌ FAIL — Git Re-Basin contradicted: layer 0 discriminates best, monotonic decrease 0→25; the centroid captures surface/lexical features, not semantic Rust-idiom-ness |
| P3c (Bench 426) | Length-detrended `d_diff` | **−0.15** (Python) | ≥0.5 | ❌ FAIL — length detrending REVERSES Python discrimination (+0.19 → −0.15); the apparent Rust-idiom signal was prompt length |
| Recipe D (Bench 427) | Length-matched corpus, k ∈ {2,4,8,16} sweep | **+0.009** (k=16) | ≥0.5 | ❌ FAIL — length-matching works (detrend passes) but cross-arch agreement never crosses +0.01; failure is STRUCTURAL cross-arch disagreement, not length, not noise |

**The failure pattern rules out the obvious escapes:**
- Recipe E (gradient descent stitching) NOT opened — failure is structural cross-arch disagreement (not non-linearity), so a trained stitcher would face the same wall.
- More Python data (10 → 30 prompts) barely moved agreement (+0.4645 → +0.4755, +0.011 gain) — the ceiling is fundamental.
- The modelless path is **declared exhausted**. Reopens only on a **non-hidden-state construction** (AST/clippy/ownership-graph features — see Proposal 010, draft).

**What STILL ships (the intra-arch + substrate GOAT, Bench 562):** the `katgpt-canon` crate (3 adapters: Procrustes / Subspace / Mask) carries a measured G1/G2/G4 GOAT stamp (17/17 PASS). The SubspaceAdapter preserves the cross-arch ALIGNMENT result (Bench 423: G5 GO at k∈{2,4}, mean cosine +0.87/+0.75 on Gemma ↔ MiniCPM held-out). The cross-arch DIRECTION claim is what failed — the substrate is useful, just narrower than the Super-GOAT headline.

**Features stay opt-in** (`canon`, `canon_subspace`, `canon_mask`, default-off). Promotion to default-on would require a new proposal re-arguing the substrate's value proposition post-demotion.

📖 Proposal: [`.proposals/009_canonical_intent_space.md`](../../.proposals/009_canonical_intent_space.md) (status header reflects permanent demotion). Research: [`.research/459_canonical_intent_space_plug_and_play.md`](../../.research/459_canonical_intent_space_plug_and_play.md) (CLOSED). Substrate GOAT: [`.benchmarks/562_katgpt_canon_goat.md`](../../.benchmarks/562_katgpt_canon_goat.md). Cross-arch demotion benches: [`.benchmarks/424_gdn_tree_verify_goat.md`](../../.benchmarks/) (P3) → riir-train `.benchmarks/425`, `426`, `427` (P3b/P3c/Recipe D — the cross-arch probes ran on real model weights via riir-train's forward-trace substrate). Non-hidden-state follow-up: [`.proposals/010_non_hidden_state_canonical_construction.md`](../../.proposals/010_non_hidden_state_canonical_construction.md) (draft, HIGHLY SPECULATIVE).

## 16. Composition Imbalance Diagnostic PoC (Issue 199) — FAILURE MODE DOES NOT TRANSFER

**Hypothesis (the open question after the Twins PASS verdict).** arXiv:2607.22531 (Twins: Focal Loss for unified ViT+VAE) proves that composing two heterogeneous latent spaces into one representation causes "optimization imbalance" during training — the model silently underfits the high-intrinsic-dimension / high-frequency component because the low-ID component dominates the loss landscape. The unproven transfer claim was that our inference-time sigmoid-gated blends of heterogeneous direction fields (`CommittedFieldBlend`, `PersonalityWeightedComposition`, `BranchBank`) suffer an **analogous** imbalance class, and that our existing diagnostics (`within_class_effective_rank`, `subspace_phase_gate`, `effective_rank`) would need to catch it.

**CLOSED 2026-07-28 — the failure mode does not transfer to inference.** The PoC ran three deliberately-mismatched configs through `CommittedFieldBlend<3, 32>::apply_blended` (LowId+2x HighId / LowId+HighId+HighFreq / magnitude-asymmetric `pi=[5,0,0]`) + a 3x-LowId sanity control. Result:

| Config | erank | wc_erank | part_ratio | low_freq |
|---|---|---|---|---|
| Balanced (3x HighId) | 24.68 | 24.68 | 30.14 | 0.508 |
| Mismatched ID | 24.39 | 24.39 | 30.59 | 0.528 |
| Mismatched spectral | 25.61 | 25.61 | 30.12 | 0.503 |
| Mismatched magnitude | 24.90 | 24.90 | 30.28 | 0.505 |
| Sanity: 3x LowId (rank-≤6 expected) | **1.00** | **1.00** | **10.54** | 0.415 |

The diagnostics flag the sanity config (rank collapse detected: erank 24.68 → 1.00) but MISS on all three "mismatched" configs — **by construction, not by failure**. The Twins paper's optimization imbalance is a training-dynamics phenomenon (gradient descent underfits). At inference time `apply_blended` is a closed-form weighted sum — no gradient descent, no underfitting. When heterogeneous fields are summed, the highest-ID field's contribution dominates the output covariance by linear algebra, not by failure. The sigmoid gates normalize per-field contribution magnitude; they do not (and need not) rebalance spectral / ID coverage.

**The original Twins PASS verdict stands — honest justification corrected.** The original justification ("covered by shipped primitives") was architecturally sloppy (§3.6 violation: architectural coverage ≠ quality parity). The honest justification is: **the failure mode is training-specific and does not transfer to inference-time sigmoid-gated blends**. The diagnostic primitives are not load-bearing for this verdict — the closed-form math is. Filing a plan for `composition_imbalance_diagnostic` would be solving a problem we don't have (T7 N/A).

**The PoC remains as a permanent regression check** at `riir-ai/crates/riir-poc/benches/composition_imbalance_modelless.rs` (the "defend-wrong" R&D crate per research skill §3.6). `PersonalityWeightedComposition` and `BranchBank` use the same sigmoid-gated-sum shape; the same closed-form argument applies (T8 N/A).

📖 Issue (removed, this entry is the durable home): closed 2026-07-28 commit `e30d2b45`. Triggering PASS verdict: arXiv:2607.22531 (Twins Focal Loss) — research notes 279, 394, 409 carry the PASS-Redirects line. Site 1: [`committed_field_blend.rs:188-224`](../../crates/katgpt-core/src/committed_field_blend.rs). Site 3: [`branching/bank.rs`](../../crates/katgpt-core/src/branching/bank.rs).

## 17. f16 Weight-Only Forward Quantization (Issue 200) — G2 FAIL: 1.7–3.0× SLOWER

**Hypothesis.** The production `forward_base` path is 95% matmul (GEMV) at seq=1. f32 GEMV arithmetic intensity is 0.5 FLOP/byte → firmly memory-bandwidth-bound → no kernel-level optimization can beat the bandwidth ceiling. The actionable path to ~2× speedup: **halve weight bandwidth by storing weights as f16** (`f32 8 bytes/elem → f16 4 bytes/elem`, 50% reduction). `matmul_f16` + `simd_matmul_f16_f32_rows` already ship in `katgpt-types` (f16 weights × f32 activations, dequant-on-load).

**G2 FAIL (2026-07-29, Apple Silicon / aarch64).** f16 weight-only forward is **1.7–3.0× slower** than f32 across all configs. Root-cause analysis (the durable value):

1. **The activation `x` is f32, not f16.** The hypothesis assumed weight-only bandwidth reduction. Reality: per dot-product element, f32 = 4 bytes (weight) + 4 bytes (activation) = 8 bytes; f16 = 2 bytes (weight) + 4 bytes (activation) = 6 bytes. **Actual bandwidth reduction: 25%, not 50%.** The halved-weight hypothesis double-counts by ignoring the f32 activation.
2. **f16→f32 dequantization is not free.** Even with hardware FCVT (1-2 cycle latency on Apple Silicon), the conversion sits on the critical path between weight load and FMA. Combined with the only-25% bandwidth reduction, the conversion latency more than eats the bandwidth savings.
3. **Scalar `to_f32()` (which LLVM vectorizes to FCVT) is strictly better than hand-rolled NEON bit-manipulation.** A WIP attempt (`convert_4x_f16_to_f32`, ~10 manual NEON ops per 4-element conversion) was 3× slower vs the committed 2.5× slower — discarded during G2 isolation.
4. **Hardware FCVTL via inline asm** improved conversion speed (0.40× → 0.574×) by using the hardware instruction directly, but still net-negative because the store-to-stack-and-reload pattern (forced by `asm!` not being able to pass NEON registers directly to `vfmaq_f32`) adds a round-trip the FMA pipeline can't fully hide. A fully-inlined asm implementation (conversion + FMA in one block) might close the gap, but at that point you're writing the entire dot product in assembly — a maintenance burden disproportionate to a still-sub-2× potential win.

| Gate | Target | Result |
|---|---|---|
| **G1** approximate correctness | max rel err <20% | ✅ **PASS** — **0.03%** on medium config, seq_len=16. f16 dequant path is numerically correct. |
| **G2** perf ≥1.5× at seq=1 | f16 ≥1.5× f32 | ❌ **FAIL** — f16 is **1.7–3.0× SLOWER** (0.574× speedup even with FCVTL inline asm). Refutes the core hypothesis. |
| **G3** no-regression | cargo test clean | ✅ PASS — f16 path is additive, doesn't touch f32 path. |
| **G4** alloc-free steady state | by construction | ✅ PASS — `forward_base_f16` reuses `ForwardContext` scratch buffers. |

**The honest takeaway.** f16 weight quantization for bandwidth-bound GEMV is **not a modelless perf win on this hardware class**. The hypothesis only holds when (a) activations are also f16, OR (b) f16→f32 conversion is zero-latency. Neither is true on Apple Silicon. This is a valid negative result in the sense of Research 003 / Issue 356 — the GOAT gate did its job by catching a wrong hypothesis before it reached production. The root-cause analysis is the durable value: it prevents future agents from re-attempting the same hypothesis.

**Code retained as reference.** `forward_base_f16` + `forward_f16` in `crates/katgpt-forward/src/forward.rs` (L806-941) ship as `pub` opt-in paths that no internal caller dispatches to — preserved for future hardware where the hypothesis holds (e.g. AVX-512_FP16 x86, or future Apple Silicon with free FCVT), or as a reference for a full-f16 (weights + activations) follow-up.

**Re-opens only on** (a) hardware where f16 loads are genuinely cheaper than f32 loads AND a hardware FCVT-equivalent is free, OR (b) a full-f16 forward context (Issue 201's line, which also failed — see section 18 below).

📖 Issue 200 (CLOSED + removed per noise-reduction rule 2026-08-05; the negative result lives here + in Bench 563). Substrate: `crates/katgpt-forward/src/forward.rs:774-932` (`forward_base_f16` + `forward_f16`, opt-in `pub` paths with no internal caller). Perf doc: [`.docs/08_performance/engineering.md`](../08_performance/engineering.md) §'What We Don't Do' (updated with empirical verdict 2026-07-29 commit `9bd1eed2`).

## 18. Full f16 Forward FHM Investigation (Issue 201) — G2 FAIL: 1.31× < 1.5× gate

**Hypothesis.** Successor to Issue 200 (weight-only f16). The bandwidth ceiling wall hits weight-only f16 because the activation is f32 — the real fix is **full f16 (weights + activations)** so per dot-product element is 2 bytes + 2 bytes = 4 bytes (true 50% reduction). Apple Silicon's widening FMA (`fmlalb`/`fmlalt`, a.k.a. `fmlal`/`fmlal2`) does f16×f16→f32 in a single instruction (FHM = Full Half-precision Multiply-add), eliminating the explicit FCVT from Issue 200's critical path while achieving the full 50% bandwidth reduction.

**G2 FAIL (2026-07-29, Apple Silicon / aarch64).** Best L3-exceeding speedup of `simd_dot_f16_f16` (FHM widening FMA) vs `simd_dot_f32` = **1.31×**, under the 1.5× gate. Phase 2 (full forward path) NOT pursued — depends on the Phase 1 gate, which failed.

Root causes (full detail in Bench 563):

1. **f32 is already near the bandwidth ceiling** (~95–110 GB/s at L3-exceeding sizes on Apple Silicon). f16 improves this to ~120–130 GB/s equivalent — only ~25–30% gain, NOT the theoretical 50%.
2. **The dot kernel is NOT purely bandwidth-bound.** FMA throughput, load-use latency, and accumulator-reduction overhead consume a meaningful fraction even at L3-exceeding sizes. Halving bandwidth doesn't halve runtime.
3. **FHM FMA throughput is the limiter** (not FCVT latency like Issue 200). The 4-accumulator unroll that hides f32 FMA latency doesn't fully hide FHM FMA's different pipeline characteristics.
4. **f16 accumulation drift grows with vector length.** rel_err ~0.1% at in-cache sizes but **6.2% at 16M elements** — even a marginal perf pass would have needed a separate precision gate.
5. **The L1 vs DRAM paradox.** f16 wins **1.71× at L1 (cache-resident)** but only 1.31× at DRAM — the gate fails specifically when weights spill to DRAM where f32 is already near the bandwidth ceiling. No architecture-level fix exists; the gate is structurally infeasible on this hardware class.

**Toolchain note.** FHM is inaccessible on stable Rust 1.93.0 (intrinsics unstable #136306; LLVM 21.1.8 assembler rejects the `fmlalb`/`fmlalt` mnemonic in every arrangement form). The Phase 1 measurement was done via nightly toolchain's unstable intrinsics (`vfmlalq_low_f16` / `vfmlalq_high_f16`), verified correct on a known input. Production code on stable would need verified `.inst` encodings — moot given the gate failed.

**Outcome.** Valid negative result (Research 003 / Issue 356 sense). The GOAT gate did its job a second time on the f16 line (Issue 200 weight-only, Issue 201 full-f16), preventing a perf-regressing "optimization" from reaching production. **f32 stays the production dtype** for `forward_base` GEMV on Apple Silicon.

**The remaining quantization path with a plausible ≥1.5× win is INT8 with INT8 activations** (different dequant path: scale + zero-point). But on Apple Silicon it hits the same bandwidth-ceiling argument — INT8 GEMV arithmetic intensity is still bandwidth-bound at L3-exceeding sizes, and the dequant overhead makes it worse. Filed as a **non-goal** in both Issues 200 and 201.

📖 Issue 201 (CLOSED + removed per noise-reduction rule 2026-08-05; the negative result lives here). **Canonical evidence:** [Bench 563](../../.benchmarks/563_issue201_f16_f16_fhm_negative.md) (the full-f16 FHM measurement, covering both Issue 200 weight-only + Issue 201 full-f16 root-cause analysis). Substrate (nightly-only WIP, not on develop): `crates/katgpt-types/src/simd/dot.rs` `simd_dot_f16_f16` (sibling-agent investigation; see git history if needed).

## 19. Quant-Error LoRA G5 (Issue 565) — DECISIVELY NEGATIVE: 0% PUCT Win-Rate

**Hypothesis.** A deterministically-constructed low-rank reader-LoRA (`QuantErrorLora`) compensates for ternary quantization error on the 105K-param Moka CNN, recovering enough accuracy to make the ternary path competitive with int8 for PUCT-based Go inference. Weight-space SVD (`E = W − dequant(W_q) ≈ A·B`) is modelless (closed-form, no gradient descent).

**G1 PASS (surprise).** Strategy A (weight-space SVD rank-32) achieves cosine 0.9939 vs f32 (+0.023 over the ternary baseline's 0.9706). The pre-PoC prediction ("rank-8 fails on small CNNs per the Small-Kernel Parameter Paradox") was too pessimistic — the Paradox is a **cost** issue (27.8% param overhead on small convs vs 0.39% on LLM linears), not a **quality** issue at rank-16+.

**G5 DECISIVELY FAIL (2026-08-01).** PUCT win-rate (n=20, budget=50, vs greedy f32):

| Strategy | Win-rate |
|---|---|
| f32 (baseline) | 100% (20/20) |
| B2 (ternary-only) | **0% (0/20)** |
| A rank-32 (ternary+wSVD LoRA) | **0% (0/20)** |

Not just "below int8's 95%" but **0%** — the ternary+LoRA path collapses PUCT strength entirely. Root cause: PUCT's budget=50 simulations amplify the small policy perturbations (cosine 0.9939 leaves residual structure that PUCT amplifies through search-tree exploration). The value head residual error causes excessive passing (26+ passes/game vs ~17 actual moves). int8 (0.97 cosine, 85-95% win-rate) works because its error is **uniform** (small, symmetric); ternary's error is **structured** (large, biased) and the LoRA removes the bias but leaves residual structure.

**The generalized lesson (durable value).** Cosine ≥ 0.99 is **necessary but not sufficient** for PUCT parity. PUCT's search amplifies residual errors, setting a higher bar than greedy-move parity. int8 is a surprising outlier that works because its error is uniform. This lesson prevents future agents from assuming "good cosine = good PUCT" — the gate between them is PUCT amplification.

**Outcome.** Modelless quant-error-compensating LoRA is unviable for the ternary path on Moka v1. Issue 565 CLOSED (removed per noise-reduction rule). The `quant_error_lora` primitive ships as reusable substrate for larger models where the error manifold is genuinely low-rank AND the target is greedy-move parity (not PUCT parity) — e.g., Gemma 2 2B (Proposal 008) if aggressively quantized for edge deployment. The trained-projection path (riir-train) is the only remaining option for Moka.

📖 Issue: 565 (removed per noise-reduction rule — resolution above), Research: [`.research/463_moka_freeze_thaw_lever_audit.md`](../../.research/463_moka_freeze_thaw_lever_audit.md) §7 (PoC Addendum 2026-08-01), Cross-reference: [opt-in catalog §32](opt_in_features.md#32-quant-error-lora--quantization-error-compensating-reader-lora-issue-565-research-463), PoC bench: `riir-poc/tests/quant_error_lora_poc.rs` (cross-repo permanent regression check), Substrate: `crates/katgpt-core/src/quant_error_lora.rs`.

## 20. PUCT WASM Batched MCTS (Issue 205) — G2 FAIL: 1.09× << 3-5× estimate

**Hypothesis.** Batched MCTS (the AlphaZero production technique) evaluates K leaves in ONE batched forward pass instead of K separate passes, dropping the 50-pass budget at b50 to ~7 batches. With K=8, IF the batched pass achieves weight-cache locality (each weight slice loaded once, used K times), the per-batch cost grows sub-linearly — estimated b50 from **29.8 ms → ~8-12 ms/move** (3-5× speedup).

**G2 FAIL (2026-07-31, V8 JIT / wasm32).** K=8 gives **1.09×** (33.7→30.8ms/move at b50) — far below the 3-5× estimate. Even K=50 (one giant batch covering the entire budget) only reaches **1.19×** (33.7→28.2ms). The batched path is correct (G1 bit-identical to K sequential passes; 12 tests pass) but the gain doesn't justify the complexity (virtual loss, leaf queue, batched expansion/backprop).

Root causes (the durable value):

1. **Forward pass is compute-bound, not cache-bound.** The Moka net (~100KB weights) fits in L2 cache, so sequential passes already benefit from cache residency. The SIMD dot kernel (`wasm32_simd128_dot_f32`, 4 accumulators × 4 lanes = 16-wide unroll) saturates the FPU per call.
2. **Batching doesn't reduce FLOPs.** K samples through the same weight slice reuse weight reads, but the total arithmetic is identical. The only savings are per-call setup overhead + cache pressure relief — both small for a 100KB net that already fits in L2.
3. **The 84% forward-pass fraction is structural.** ~0.50 ms/pass × 50 nodes = 25 ms. Tree overhead is ~15%. Tree-side micro-optimizations cannot move the needle — the only path to dramatic improvement is reducing the per-pass cost or the pass count, and batching only helps the count if the per-batch cost is sub-linear.
4. **wasmi parity preserves K=1 default.** The K=1 sequential path is bit-identical to pre-batch code; wasmi (the browser fallback runtime) cannot run batched code efficiently. Keeping K=1 default preserves wasmi parity.

**Outcome.** Valid negative result. The batched path is **not promoted** — K=1 stays default, batched is opt-in via `PuctPlayer::with_batch_k`. The honest takeaway: for a small in-cache network, AlphaZero-style batching is not the right lever. The next lever is reducing per-pass cost (INT8 activations, Issue 206/207 — which DID succeed) or restructuring the conv2d via im2col+GEMM (filed as a non-goal, doubles the implementation surface).

**Post-hoc vindication (Issue 206/207, 2026-07-31).** The INT8+INT8 dot path via WASM SDOT DID break the 30ms PUCT WASM floor (26.8ms b50) + was promoted to default-on. This confirms the root-cause diagnosis: forward-pass cost was the lever, not search-side batching. The INT8 win uses a different execution unit (`i8x16.dot_s`) than f32 FMA on V8, sidestepping the FPU saturation that blocked the batched path.

📖 Issue: 205 (removed per noise-reduction rule — resolution above), Benchmark: [`.benchmarks/205_puct_wasm_batched_mcts_latency.md`](../../.benchmarks/205_puct_wasm_batched_mcts_latency.md), Substrate: `crates/katgpt-moka-wasm/src/puct.rs` (`PuctPlayer::with_batch_k`), Positive sibling (overturns the batched-path non-goal at a different lever): Issue 206/207 + [`.benchmarks/565_int8_int8_sdot_positive.md`](../../.benchmarks/565_int8_int8_sdot_positive.md).

## 21. Moka v1 on Apple Neural Engine via CoreML (Issue 564) — G2 FAIL: ANE 4.66× SLOWER than CPU

**Hypothesis.** `crates/katgpt-backend/src/ane.rs` is a real CoreML/Apple Neural Engine execution backend (currently transformer-only, used for one `lm_head` linear layer). Moka's conv net is unused on ANE — authoring the ~40-layer graph might unlock a latency win beyond what CPU SIMD can deliver (Plan 563 got CPU to ~2.8ms/move via kernel tuning).

**G2 FAIL (2026-07-29, same day as filed — Apple Silicon / aarch64).** Direct measurement of a scoped probe (stem conv + one full non-global residual block, 9 layers: 5 convs, 3 ReLU, 1 residual add) as a real CoreML graph:

| | Latency (9-layer slice) |
|---|---|
| CPU (`conv2d_into` + `simd_dot_f32`, Plan 563) | **56 µs** |
| ANE (CoreML `predict()`, `ComputeUnits::All`) | **261 µs** |
| Ratio | **ANE is 4.66× slower** |

Correctness was proven (not the bottleneck): `max_abs_diff = 5.96×10⁻⁸` (f32 machine epsilon) — the CoreML output and CPU reference agree to the bit, confirming the `[out,kh,kw,in]`↔`[out,in,kh,kw]` weight transpose + HWC↔CHW activation transpose are both correct.

Root cause (exactly the risk flagged before writing any code):

1. **The model is too small (105K params).** CoreML's fixed per-call dispatch/marshalling overhead (FFI bridging, tensor copy in/out, Neural Engine driver handoff) dominates compute at this scale — not the conv arithmetic. That overhead is roughly constant regardless of layer count per `predict()` call.
2. **No amortization headroom.** The *whole* CPU network only costs ~450 µs. A fixed per-call overhead of a couple hundred microseconds can't amortize away across so little total compute. Authoring all ~50 layers into one larger graph would reduce overhead-per-layer share but cannot close a 4.7× gap.

**Outcome.** Valid negative result. Decision: do NOT build the remaining layers — a probe-first negative result stops the investigation rather than continuing to a full 40+-layer graph for a technique already shown to lose at this scale. The layer-mapping + transpose work in `ane_probe` is not wasted — it's a correct, reusable starting point if revisited on a much larger model or batched inference across many positions.

**Re-opens only on** (a) a much larger model where compute dominates the fixed CoreML dispatch overhead, OR (b) batched inference across many positions (batch > 1 amortizes the per-call cost).

📖 Issue: 564 (removed per noise-reduction rule — resolution above), Substrate: `crates/katgpt-pruners/src/go/moka_net.rs::ane_probe` (scoped probe, retained for reuse), ANE backend: `crates/katgpt-backend/src/ane.rs` (transformer-only, unchanged).

## 22. Loop Injection Regime-Split PoC (Issue 568) — NO TRANSFER (planning at chance)

**Hypothesis.** Paper "Towards Looped Models Done Right" (Huang et al. IFM/USC/CMU, Aug 2026) Q2 finding: persistent input injection (`z̃_t = (1-α)·z_t + α·e`, diagonal operator writing the fixed prelude into the recurrent stream each step) **helps retrieval/context/code** (MMLU +2.53, HumanEval+ +5.49, BBH-CoT +6.63) but **hurts math/reasoning** (MATH500 −3.60, GSM8K −2.51). The falsifiable prediction: this regime split transfers to a toy belief kernel (hand-designed 8-dim recurrent state, NOT a trained looped transformer) — always-inject > baseline on retrieval, < baseline on planning, regime-gated ≥ both.

**Verdict: NO TRANSFER (2026-08-01, empirically validated PASS).** PoC setup: D=8, K=16, trials=500, α=0.3.

| Strategy | Retrieval Cos | Planning Acc | Verdict |
|---|---|---|---|
| Baseline | 0.0143 | 0.5145 | ─ |
| AlwaysInject(0.3) | 0.0246 | 0.5270 | helps_retrieval=YES  hurts_planning=HELPS |
| RegimeGated | 0.0246 | 0.5145 | best_of_both=YES |

Injection helps retrieval (+0.0103 cosine, as predicted ✓) but the planning task stays at chance (0.5145 baseline, within ±0.03 of 0.50) for ALL conditions. The "hurts planning" prediction is **untestable** on our substrate.

Root cause (the durable value):

1. **A fixed-random untrained kernel cannot do multi-step accumulation above chance.** tanh saturation + random `W_z` mixing destroy the signal across K=16 steps. The baseline can't plan, so there's nothing to degrade — the "injection hurts math" prediction requires a model that can actually do math first.
2. **Two planning tasks were tried, both at chance.** Parity (XOR, hardest Boolean — baseline 0.4967, pure coin flip) AND weighted-sum-direction (linear accumulation, much easier — baseline 0.5145). Still at chance. Untestable on either.
3. **The retrieval improvement is real but tiny** (+0.0103 cosine, both baseline + injection near-zero cosine ~0.01–0.03) — not actionable as a Gain-tier primitive.

**The generalized lesson.** The regime split is irreducibly a **trained-model phenomenon**. The "injection hurts math" prediction is substrate-specific to trained looped transformers — it requires a model that can actually do math (accuracy well above chance) as the precondition for injection to measurably degrade it. This empirically validates the PASS verdict on the paper: the cautionary flag belongs in riir-train (training-time architecture decisions), not katgpt-core (modelless primitives). A modelless regime-gated injection primitive would be cargo-culting a phenomenon that doesn't exist on untrained substrates.

**Outcome.** No plan filed. The PoC source remains in `riir-poc/` as a permanent regression check — its job was to settle the dispute, and it should keep settling it if the belief kernel is later trained/structured (the trained case is where the regime split could re-emerge as actionable).

📖 Issue: 568 (removed per noise-reduction rule — resolution above), PoC source: `riir-ai/crates/riir-poc/src/loop_injection_poc.rs` + `benches/loop_injection_regime_split.rs` (cross-repo permanent regression check), Related research: [073](../../.research/073_LT2_Linear_Time_Looped_Transformers.md) (LT2), [097](../../.research/097_Training_Free_Looped_Transformers.md), [414](../../.research/414_Fully_Looped_Transformer_Readout_Blind_Spot.md), [048](../../.research/048_HRM_Text_Hierarchical_Recurrent_Pretraining.md).

## 23. CompressionDrafter — Hot-Tier Modelless LZ4 Drafter (Plan 285) — G1+G2 BOTH FAIL

**Hypothesis.** A Hot-tier drafter that uses LZ4-compressed corpus as a model (nathan.rs/gzip-lm pattern) could provide a modelless quest-grammar drafter for the Hot tier — zero model weights, deterministic, training-free.

**G1+G2 FAIL (two independent runs, both failed).** Per `.plans/285_compression_drafter_quest_grammar.md`:

| Run | G1 (speedup ≥3×) | G2 (overhead ≤2×) | Verdict |
|---|---|---|---|
| Phase 3 | **0.12×** (8× SLOWER) | 407× over target | FAIL |
| Phase 7 | **1.50×** | 1077× over target | FAIL |

Root cause: quest-grammar strings are **too short and too few** for byte-level LZ4 compression to find meaningful matches. The compressor needs long-redundant text to shine (web crawl, log files); quest records are 50–200 byte structured items with minimal byte-level redundancy across entries.

**Outcome.** Stays opt-in, demoted per Plan 285 verdict. `TernaryDraftModel` remains default-on for Hot-tier quest grammar. The failure mirrors BabelCodec (§24) — both hit the wall that **Hot-tier quest/KG text is structurally compact**, and neither byte-level nor rule-level compression finds 2× on already-compact data.

**Re-opens only on** a Hot-tier consumer with long-redundant text (natural-language prose, unstructured logs). For quest/KG/config structured data, this approach is dead.

📖 Plan: [285](../../.plans/285_compression_drafter_quest_grammar.md), Feature: `compression_drafter` (opt-in, dep:lz4_flex), Research: [256](../../.research/256_GzipLM_Compression_Drafter.md).

## 24. BabelCodec — Readability-Relaxed Semantic Codec (Plan 331) — G2 FAIL: 1.14× << 2× target

**Hypothesis.** The deterministic BT-P8 fixed-rule subset of BabelTele (arXiv:2606.19857) could compress structured KG/config/quest text to 2×+ — the modelless subset of a paper claiming 3.6× compression.

**G2 FAIL (2026-06-26).** Same wall as CompressionDrafter (§23), mirror-image reason:

| Gate | Target | Result | Verdict |
|---|---|---|---|
| G1 (fidelity) | bit-identical round-trip | 1500/1500 | ✅ PASS |
| **G2 (compression)** | **≥ 2× (ratio ≤ 0.5)** | **1.14× (ratio 0.8805)** | ❌ **FAIL** |
| G3 (latency) | < 200 ns/msg | 125 ns | ✅ PASS |
| G4 (alloc-free) | 0 allocs | 0/1000 | ✅ PASS |
| G5 (determinism) | BLAKE3 reproducible | 0 mismatches | ✅ PASS |

Root cause: structured canonical forms are **already too dense** for symbolic rewrite. The verbose form `Config[negotiation]: patience_required = 10(turns)` compresses to `Config[negotiation]:patience_required=10(turns)` — saving 2 spaces around `=` ≈ 4%. The paper's 3.6× requires **LLM-prompted omnilingual lexical selection** (riir-train territory, not modelless).

**Outcome.** Stays opt-in. Ships as a correct, tested, bijective codec useful for deterministic BT-P8 ↔ verbose round-tripping with BLAKE3 commitment. But it does not promote — G2 failed, matching the CompressionDrafter precedent.

**The generalized lesson (the §23 + §24 wall).** Hot-tier quest/KG text is structurally compact. Neither byte-level (LZ4) nor rule-level (BT-P8) compression finds 2× on already-compact data. The 2×+ opportunity lives in verbose natural-language prose, which neither codec handles deterministically. Stop filing modelless Hot-tier text compression plans on structured data.

📖 Bench: [331](../../.benchmarks/331_babel_codec_goat.md), Plan: [331](../../.plans/331_babel_codec_readability_relaxed_semantic_codec.md), Feature: `babel_codec` (opt-in), Research: [312](../../.research/312_BabelTele_Readability_Relaxed_Semantic_Codec.md).

## 25. Sink-Aware Attention Per-Call Gate (Plan 287) — G3 STRUCTURAL FAIL: 1671% overhead

**Hypothesis.** The dual-mechanism sink classifier from Fesser et al. (arXiv:2606.08105) — classifying each attention sink as NOP (gate it) or Broadcast (preserve it) — could ship as a per-call production attention policy that beats the default Uniform gate.

**G3 STRUCTURAL FAIL (2026-06-17).** Per `.benchmarks/059_sink_aware_goat.md`:

| Gate | Target | Result | Verdict |
|---|---|---|---|
| G1 (correctness) | 8/8 unit tests | 8/8 | ✅ PASS |
| G2 (synthetic) | Broadcast preservation | 2/2 | ✅ PASS |
| **G3 (per-call latency)** | **≤ 5% overhead** | **1000–3000% overhead** at n=128/512 d_h=64 | ❌ **STRUCTURAL FAIL** |
| G3 (cached cadence=16) | ≤ 5% overhead | ≤ 5% steady-state | ✅ PASS |
| G3-flat (Plan 288) | flat ≥ Vec<Vec> | 1.8–5.1× faster | ✅ PASS |
| G2 (real-ViT) | effective_rank preserved | — | ⏸ DEFERRED |

Root cause: the classifier is **memory-bandwidth bound**. It reads the full attention matrix (n²) + values (n·d) to classify each sink, while the default Uniform policy is just an n·d memcpy. The classifier fundamentally cannot beat a memcpy — it reads strictly more memory. Issue 001 optimizations brought the standalone `classify_sink_at` from 3.125µs → 0.625µs at n=128, but `apply_dual_policy_gate` still has the col_sums scan + value_norm scan.

**Outcome.** Per-call G3 structurally infeasible → NOT promoted to default. The cached variant (`apply_dual_policy_gate_cached`, audit cadence=16) meets the latency target in steady state. Default `SinkAwarePolicy::Uniform` stays; `DualPolicy` remains a research-grade opt-in diagnostic. Ships as opt-in `sink_aware_attn`.

**Re-opens only on** (a) a real-ViT G2 gate that demonstrates the dual-policy preserves `effective_rank` better than Uniform on a trained model (currently DEFERRED), AND (b) the consumer uses the cached variant (not per-call).

📖 Bench: [059](../../.benchmarks/059_sink_aware_goat.md), Plan: [287](../../.plans/287_sink_aware_attention.md), Feature: `sink_aware_attn` (opt-in), Research: [258](../../.research/258_Attention_Sink_Dual_Mechanism_NOP_Broadcast.md).

## 26. Hierarchical Global Attention vs DashAttention (Plan 397) — G2 FAIL: loses to default-on

**Hypothesis.** HGA's chunk→group→token routing with RoPE-aware mixed-frequency summaries (arXiv:2606.30709, BMW Group) could beat the default-on DashAttention primitive on long-context NIAH retrieval at 32K–64K context.

**G2-proxy FAIL (2026-07-05).** Per `.plans/397_hierarchical_global_attention.md`:

The head-to-head G2 gate against DashAttention on a synthetic NIAH (Needle-in-a-Haystack) routing comparison — **HGA LOSES to the default-on DashAttention**. The group tier + mixed-RoPE summary construction does not improve retrieval routing over DashAttention's entmax chunk-level routing.

G5 latency PASS at 1.12× (the routing summary construction is cheap), but G2 is the load-bearing gate and it lost. Per Plan 397 T3.3, this is a documented negative result — HGA stays opt-in.

Root cause: DashAttention's entmax routing is already a strong sparse-attention summary. Adding a sub-chunk group tier with mixed-RoPE summaries does not improve routing quality on the NIAH benchmark — the chunk-level summary captures the needle-vs-haystack signal adequately. The group tier adds hierarchy without improving the retrieval decision.

**Outcome.** Stays opt-in as documented negative result. The forward path (needs entmax) lives in `katgpt-attn/hga_forward.rs`; the katgpt-core half ships the routing summary construction primitives. DashAttention remains the default-on long-context routing primitive.

**Re-opens only on** a long-context task where chunk-level routing is insufficient AND sub-chunk group routing demonstrably helps — the NIAH benchmark didn't surface this. A different benchmark (multi-document QA, code understanding) might, but that's speculative.

📖 Plan: [397](../../.plans/397_hierarchical_global_attention.md), Feature: `hga` (opt-in), Research: [379](../../.research/379_Hierarchical_Global_Attention_Chunk_Group_Routing.md).

## 27. VFD Velocity-Field Disagreement — G2 FAIL: UQ Floor (Issue 010) NOT Met

**Hypothesis.** The VFD estimator from Römer et al. (arXiv:2606.18043 §4, Theorem 4.1) — consuming M frozen velocity fields, integrating each member trajectory independently, accumulating pairwise disagreement weighted by `κ_s = s/(1−s)` — could ship as a calibrated epistemic-UQ primitive that upgrades `velocity_field_ensemble`'s UQ.

**G2 FAIL (2026-07-13).** Per `.benchmarks/432_vfd_goat.md` — the UQ-bearing primitive GOAT gate extension (the "Report the Floor" rule, Issue 010) requires VFD-calibrated intervals to beat `ConformalIntervalCalibrator<SeasonalNaiveForecaster>` m=1 on CRPS/coverage/Winkler. **VFD does NOT meet the conformal-naive floor** — `λ*=0` on both AR(1) + 1D Bimodal corpora (the optimal calibration weight on the VFD disagreement signal is zero, meaning the floor's raw residuals dominate).

| Gate | Result |
|---|---|
| G1 (mechanics) — exact analytic match for constant-disagreement fields | ✅ PASS |
| **G2 (UQ floor per Issue 010)** | ❌ **FAIL** (λ*=0 on both corpora) |
| G3 (no-regression) | ✅ PASS |
| G4 (latency) — `vfd_score_into` ≤ 50µs at M=2 D=8 N_s=10 B=5 | ✅ PASS — 10.43µs (4.8× margin) |
| G5 (QGF integration) — `VfdVarianceSignal: QgfVarianceSignal` | ✅ PASS |

**Outcome.** `velocity_field_disagreement` stays **opt-in**. VFD ships as a **non-UQ disagreement score** — useful for CLR L1 gating, sleep-time prioritization, runtime failure detection (paper §6.4), but with **no calibrated-UQ claim**. The velocity-field ensemble (Plan 376) remains the UQ-bearing primitive; VFD does not upgrade it.

**The generalized lesson (Issue 010 pattern).** This is the same failure mode as the conformal floor catches in other UQ primitives: a disagreement signal that *correlates* with error is not the same as a calibrated predictive interval. The floor's `SeasonalNaiveForecaster` with split conformal is a surprisingly strong baseline; beating it requires the UQ signal to add information the naive residuals don't already carry. VFD's disagreement signal, on the test corpora, did not.

**Re-opens only on** a corpus where the velocity-field disagreement genuinely carries calibrated-UQ information that the conformal-naive floor misses — e.g. a regime where ensemble member disagreement is the *only* available error signal (no historical residuals). The test corpora (AR(1) + Bimodal) had historical residuals available, which is the floor's strength.

📖 Bench: [432 GOAT](../../.benchmarks/432_vfd_goat.md) + [432 UQ floor](../../.benchmarks/432_vfd_uq_floor.md), Plan: [432](../../.plans/432_vfd_velocity_field_disagreement_primitive.md), Feature: `velocity_field_disagreement` (opt-in), Research: [420](../../.research/420_VFD_Velocity_Field_Disagreement_Epistemic_UQ.md).

## 28. FlowField × DualLeoMixer Pre-Max Fusion (Plan 459) — G5 FAIL: Nonlinearity Washes Out α-Mix

**Hypothesis.** Mixing LEO teacher + UVFA student Q-slices via `DualLeoMixer::combine_into` before the `max-over-actions` step would produce a better flow-field navigation potential than the LEO-only baseline (≥30% stuck-NPC reduction).

**G5 FAIL (2026-07-18).** Per `.benchmarks/459_flow_field_dual_leo_mixer_goat.md` — no α in {0.1..0.9} meets the 30% gate. Best α=0.10 achieves only 25.9% reduction; the paper's default α=0.3 achieves only 3.7%.

| Gate | Result |
|---|---|
| G1 (bit-identity — LeoOnly matches single-head) | ✅ PASS |
| G2 (perf — cache-miss ≤ 1.5×) | ✅ PASS — 1.11× |
| **G5 (≥30% stuck-NPC reduction at some α)** | ❌ **FAIL** — best 25.9% at α=0.10 |

**Root cause.** `max_a (α·Q_leo[a] + (1−α)·Q_uvfa[a]) ≠ α·max_a Q_leo[a] + (1−α)·max_a Q_uvfa[a]`. The pre-max α-mix is washed out by the nonlinear max-pool *before* the FFT smoothing sees it. The LEO decoy peak survives the max even at low α because the mix is per-action, pre-max.

**Outcome + the correction.** Plan 459's pre-max `get_or_compute_dual` stays landed (G1+G2 pass, opt-in callers unaffected) but is demoted to "compatibility / parity with QGF pre-max mix". **Plan 460 is the correction** — it moves the blend to *post-max* potentials (`α·potential_leo + (1−α)·potential_uvfa`), which is linear in the FFT's input. Plan 460's postmax path G5' PASSES at α=0.10 (31.5% reduction, +5.6pp over pre-max). The pipeline-stage change is the difference between FAIL and PASS at essentially identical perf cost.

**The real-network caveat (Issue 549).** Plan 460's synthetic G5' PASS did NOT survive contact with untrained CivLeoNet + CivLeoUVFA — drops to 3.3% at α=0.10. The postmax mechanism is correct (G1 bit-identity holds on both synthetic + untrained real); the gain requires **trained** networks. Tracked in `riir-ai/.issues/552`.

📖 Catalog entry: [opt_in_features.md §41](opt_in_features.md#41-flowfield--dualleomixer-fusion---post-max-potential-blending-plan-459--plan-460). Benches: [459](../../.benchmarks/459_flow_field_dual_leo_mixer_goat.md) + [460](../../.benchmarks/460_flow_field_dual_leo_postmax_goat.md). Plans: [459](../../.plans/459_flow_field_dual_leo_mixer_fusion.md) + [460](../../.plans/460_flow_field_dual_leo_postmax_fusion.md). Features: `flow_field_nav` + `dual_leo` (both DEFAULT-ON; dual methods opt-in via API choice).

## 29. ReMax Expected-Max Aggregation (Plan 374) — NOT A MODELLESS EXPLORATION MECHANISM

**Hypothesis.** The ReMax Expected Improvement operator (Nishimori et al. ICML 2026, arXiv:2606.00151) — `expected_max_over_m(q)` + `expected_improvement(q, current_best)` — could ship as a per-arm deterministic selection score that provides modelless exploration at inference time.

**G2 PASS-but-not-modelless (2026-07-03).** Per `.benchmarks/374_remax_goat.md` — the primitive is correct + fast (G1 MC validation + analytic recurrence; G4 max=603ns at K=128). But the headline finding: **argmax EI = argmax q, by monotonicity**. The ReMax Expected Improvement operator, when used as a per-arm deterministic selection score, is **provably equivalent to greedy selection**. ReMax's exploration is a *training-time* phenomenon — it emerges from policy gradient on `J_m(π, q)` (the RePPO algorithm), not from inference-time action selection.

| Gate | Result |
|---|---|
| G1 (correctness — MC validation + analytic recurrence) | ✅ PASS |
| G2 (bandit regret) | ⚠️ PASS (theorem) — ReMax = Greedy, by proof + empirical confirmation |
| G3 (no-regression) | ✅ N/A (opt-in, no existing code depends) |
| G4 (latency — max=603ns K=128, per_action=11.7µs O(K²)) | ✅ PASS |
| G5 (feature isolation) | ✅ PASS |

**Outcome.** Keep `remax_aggregation` as **opt-in**. The primitive is a correct building block for RePPO training (riir-train Plan 304, feature `remax_ppo`), NOT a standalone modelless exploration mechanism. Per AGENTS.md §"Promotion requires modelless gain": the exploration gain requires training (policy gradient on `J_m`), so this primitive stays opt-in and is not promoted to default-on.

**The generalized lesson.** A primitive can have correct math + fast latency + clean feature isolation and STILL not be a modelless gain. The modelless mandate (katgpt-rs/AGENTS.md) requires the *gain* to be achievable without training — not just the *mechanism*. ReMax's mechanism is closed-form arithmetic (modelless); its exploration *benefit* is a training-time phenomenon (not modelless). This is the same distinction as the AC-Prefix G1 lesson (Plan 313): the algorithm's correctness is necessary but not sufficient for promotion.

📖 Bench: [374](../../.benchmarks/374_remax_goat.md), Plan: [374](../../.plans/374_remax_expected_max_aggregation_primitive.md), Feature: `remax_aggregation` (opt-in), Research: [373](../../.research/373_ReMax_Expected_Max_Retry_Aggregation.md). Training algorithm: `riir-train` Plan 304 (`remax_ppo`).

## 30. ShardKV — RoPE-Removal + Hadamard KV Compression (Plan 147) — CONDITIONAL: Combined Fidelity WORST of All KV Methods

**Hypothesis.** RoPE removal + PCA on keys + Hadamard transform + uniform quant on values would achieve ≥8× compression with best-in-class K fidelity (the RoPE-removal insight: rotating keys to remove RoPE's position-dependent rotation concentrates the eigenvalue distribution, d_eff 5.90 → 2.00, enabling better PCA compression).

**CONDITIONAL — not promoted (2026-05-26).** Per `.benchmarks/045_shard_kv_goat.md`:

| Gate | Target | Result | Verdict |
|---|---|---|---|
| G1 (RoPE-removal d_eff ratio) | < 0.7 | 0.339 (5.90 → 2.00) | ✅ PASS |
| G5 (compression ratio) | ≥ 8× | 9.7× at d=128 | ✅ PASS |
| G6 (K cosine) | 0.995 | 0.9880 | ⚠️ CONDITIONAL (met min 0.985) |
| G7 (V cosine) | 0.98 | 0.9407 | ⚠️ CONDITIONAL (met min 0.93) |
| **Cross-method combined fidelity** | best | **1.9373 (WORST)** | ❌ **FAIL** |

| Method | cos_k | cos_v | Combined | Compression |
|---|---|---|---|---|
| ShardKV(K=4,V=2) | **0.9957** | 0.9416 | 1.9373 | 9.0× |
| SpectralQuant(avg=3bit) | 0.9855 | 0.9847 | 1.9703 | 9.1× |
| TurboQuant(K=3,V=3) | 0.9646 | 0.9834 | 1.9480 | 9.1× |
| HybridOCTPQ(K=3,V=3) | 0.9862 | 0.9866 | **1.9728** | 9.1× |

**Root cause.** The V path is too lossy — Hadamard + 2-bit uniform can't match OCTOPUS's 0.99 triplet encoding. ShardKV wins on K fidelity (the RoPE-removal + PCA path works) but loses on V fidelity badly enough that combined fidelity is worst of all methods. The asymmetric K=4/V=2 allocation is also dominated by symmetric K=3/V=3 (+0.031 combined).

**Outcome.** Stays opt-in (`shard_kv`). The RoPE-removal insight is validated + should be evaluated as a standalone enhancement to SpectralQuant (future work). The V path needs rework — replace Hadamard+uniform with OCTOPUS triplet encoding to close the quality gap.

**Niche use case.** Long-context memory-bound workloads where K fidelity matters more than V (attention is more sensitive to key quality than value quality in some regimes). Not a general-purpose default.

**Re-opens only on** (a) a V-path rework that closes the combined-fidelity gap to HybridOCTPQ, OR (b) a workload where the asymmetric K-favoring allocation demonstrably beats symmetric allocation on downstream task quality.

📖 Bench: [045](../../.benchmarks/045_shard_kv_goat.md), Plan: [147](../../.plans/147_shard_asymmetric_kv_cache.md), Feature: `shard_kv` (opt-in, in `katgpt-kv`).

## 31. Factorized Transition Action Abstraction — G2b+G3 FAIL (STAYS OPT-IN)

**Plan:** [375](../../.plans/375_factorized_transition_action_abstraction.md) · **Bench:** [375](../../.benchmarks/375_factorized_action_goat.md) · **Research:** [374](../../.research/374_OTF_LAM_Factorized_Transition_Primitives.md) · **Paper:** [arXiv:2606.30544](https://arxiv.org/abs/2606.30544) (Nam et al., Brown, 2026-06-30) · **Feature:** `factorized_action` (opt-in)

**What it is.** Frozen codebook of K=128 D-dim effect primitives + Top-1 patch assignment + sigmoid relevance gate + normalized weighted average → compact action latent. The factorized/compositional cousin of the shipped monolithic `latent_functor`. Codebook constructed modellessly via Lloyd's k-means with k-means++ init.

**Gate results (4/6 PASS, 2 FAIL):**

| Gate | Result | Detail |
|---|---|---|
| G1 (factorized MSE ≤ monolithic) | ✅ PASS | 0.029 ≤ 0.140 (4.9× improvement) |
| G2a (distractor suppression < 0.7× mono) | ✅ PASS | 0.066 < 0.126 (63% improvement) |
| **G2b (gate adds value over uniform mean)** | ❌ **FAIL** | Gate 0.066 == Mean 0.066 (ratio 1.000 — modelless L2-norm gate is at parity with uniform) |
| **G3 (cross-carrier transfer)** | ❌ **FAIL** | factorized drop 7.9× vs monolithic −0.05× (k-means overfits source distribution) |
| G4 (latency + alloc-free) | ✅ PASS | p50 169–300 ns, 0 allocs/100 calls |
| G5/G6 | ✅ PASS | sigmoid never softmax; feature isolation clean |

**Root cause.** The factorization mechanism is GOAT (G1+G2a crush the monolithic baseline), but the **modelless relevance gate adds zero value** over uniform aggregation. Without FiLM conditioning, the factor token is just the raw centroid — two codes with equal L2 norm get equal gate output. The paper's `GateNetwork` is a **4-layer FiLM-conditioned MLP** that learns state-aware relevance; the modelless L2-norm can't replicate this. G3 (transfer) fails because modelless k-means overfits digit-{0–4}-specific patterns; the paper's trained VQ-VAE transfers well.

**Outcome.** Stays opt-in. The modelless-unblock check (§3.5) was performed — all three paths exhausted. G2b failure is not a modelless-correctable bias but a missing capability (state-aware relevance scoring requires learned FiLM projections). Trained VQ-VAE + GateNetwork → riir-train follow-up.

**Lesson.** When a paper's key component is a learned gating network, the modelless L2-norm analog is structurally insufficient — norm is not a discriminative enough proxy for state-conditioned relevance. This is a systematic failure class for factorized/VQ-VAE-style primitives.

## 32. DFlare Modelless Inference Trio — 3× GOAT-FAILED (STAYS OPT-IN)

**Plan:** [174](../../.plans/174_dflare_modelless_inference.md) · **Research:** [154](../../.research/154_DFlare_Layer_Wise_Fusion_Block_Diffusion.md) · **Paper:** [arXiv:2606.02091](https://arxiv.org/abs/2606.02091) (DFlare) · **Features:** `dflare_fusion`, `dflare_kv_routing`, `dflare_progressive_budget` (all opt-in)

**What it is.** Three modelless inference-time adaptations of DFlare's layer-wise fusion for block diffusion: (1) Marginal Fusion — multi-source conditioning blend; (2) Pruner-Confidence KV Routing — confidence-gated KV selection between target-conditioned and unconditioned; (3) Position-Weighted DDTree Budget — exponential decay allocation biased toward early positions.

**Gate results.** Status from the plan: **"Structural GOAT ✅, Improvement GOAT ❌"** — all three compile and pass structural unit tests (T1–T3, T7), but the improvement GOAT (T4–T6) failed to meet acceptance-length thresholds:

| Idea | Metric | Threshold | Outcome |
|---|---|---|---|
| D2 Marginal Fusion | Acceptance length vs single-conditioning | ≥ 5% improvement | ❌ FAIL |
| D3 KV Routing | Acceptance length with confidence gating | ≥ 3% improvement | ❌ FAIL |
| D4 Progressive Budget | Acceptance length vs uniform budget | ≥ 2% improvement | ❌ FAIL |

**Root cause.** On the micro-transformer test corpus, the multi-source conditioning blend, pruner-confidence routing, and position-weighted budget all produced **no measurable acceptance-length improvement** over the single-conditioning baseline. The DFlare paper's gains require the full block-diffusion training loop (the marginal fusion is trained jointly with the diffusion model); modelless application of the inference-time adaptation alone doesn't capture the trained coupling.

**Outcome.** All three features stay opt-in as substrate for completeness — the Cargo.toml comments explicitly mark them "GOAT-FAILED; kept as substrate for completeness". No runtime consumer wires them.

**Lesson.** Inference-time adaptations of training-time techniques (layer-wise fusion, confidence routing, budget allocation) require the trained coupling to produce acceptance-length gains. The modelless inference-only half is structurally insufficient when the paper's gains come from joint training.

## 33. Linking-Fold Detector — G2 Budget FAIL (OPT-IN, AUDIT-CADENCE)

**Plan:** [410](../../.plans/410_linking_fold_primitive.md) · **Research:** [391](../../.research/391_Low_Dimensional_Topology_Linking_Number.md) · **Paper:** [arXiv:2606.31856](https://arxiv.org/abs/2606.31856) (Ren & Lim, ICML 2026) · **Feature:** `linking_fold_detector` (opt-in); sibling `linking_fold_fold` is DEFAULT-ON

**What it is.** Algorithm 1 (PCA-3D + ε-kNN + fundamental cycle basis via BFS spanning forest + Gauss linking integral) detects whether two point clouds are topologically linked (link≠0). The cold-path diagnostic companion to the DEFAULT-ON `linking_fold_fold` correction primitive.

**Gate results:**

| Gate | Status | Detail |
|---|---|---|
| G1 (correctness) | ✅ PASS | 9/9 unit tests (Hopf = −1, unlinked = 0, fold unlinks) |
| **G2 (perf)** | ❌ **FAIL original budget** | Target: 50ms @ n=2×1000. Measured: 408ms @ n=2×200 (pre-opt) → 59ms @ n=2×200 (post-opt, 6.9× speedup via BB-skip + SoA auto-vec). Minutes extrapolated @ n=2×1000. |
| G3/G4/G5 | ✅ PASS | Feature isolation clean; alloc-free on fold hot-path (not detector); bit-identical |

**Root cause.** The brute-force O(β²) Gauss linking integral over cycle-basis pairs is structurally too slow at n=2×1000 point clouds. Issue 050 optimization (2026-07-07) cut 408→59ms via bounding-sphere pre-check (skips 84.5% of cycle pairs whose Gauss integral provably rounds to 0) + full-SoA segment layout for auto-vectorization. The original 50ms @ n=2×1000 target remains unreachable with brute-force.

**Outcome.** **Option C split executed:** the fold correction (`linking_fold_fold`) is DEFAULT-ON (passes every gate modellessly); the detector (`linking_fold_detector`) stays opt-in. Issue 050 resolved via Option A — the audit-cadence budget of 500ms @ n=2×200 is accepted as fit-for-purpose (detector runs once per session/sleep-cycle, not per-tick; zero in-tree consumers). Option B (optimize to 50ms @ n=2×500 via batch bbox early-exit + cycle pruning) remains a non-blocking follow-up.

**Lesson.** When a diagnostic primitive has a structural O(n²) scaling cliff, the right response is to split: promote the per-tick correction (closed-form, fast) and keep the per-session diagnostic (brute-force, slow) opt-in at audit cadence. Don't block the valuable primitive on the expensive diagnostic's budget.

## 34. RECOS — Rearrangement-Inequality Cosine Similarity (G1 FAIL)

**Plan:** [437](../../.plans/437_recos_rearrangement_bound_similarity.md) · **Bench:** [437](../../.benchmarks/437_recos_goat.md) · **Research:** [421](../../.research/421_Recos_Rearrangement_Bound_Similarity.md) · **Paper:** [arXiv:2602.05266](https://arxiv.org/abs/2602.05266) (Ai 2026, "Beyond Cosine Similarity") · **Feature:** `recos` (opt-in)

**What it is.** RECOS saturates at 1.0 under ordinal concordance (monotonic relationship) — a strictly wider capture range than cosine (which needs linear dependence). Always |recos| ≥ |cos| in abs value (Corollary 2). Operates on fixed `[f32;8]` (HLA dim, stack-sort, alloc-free) for hot path + arbitrary-len slices for cold path.

**Gate results:**

| Gate | Result | Detail |
|---|---|---|
| **G1 (quality)** | ❌ **FAIL** | recall@1 cosine=0.948 vs recos=0.783 (Δ=−16.5pp); recall@5 cosine=0.997 vs recos=0.985 (Δ=−1.2pp). Win rate 0% across 12 seeds (bar ≥80%). |
| G2 (latency) | ℹ️ informational | 40–160× slower than cosine (two d=8 sorts per call). Moot given G1 FAIL. |

**Root cause.** The paper's 98.6% win rate is on **semantic textual similarity (STS)** — a *matching* task. Our use case (`ShardIndex::query`) is **retrieval** — a *discrimination* task. Two mechanisms defeat recos on retrieval:

1. **Corollary 2 inflation of distractor scores.** `|recos| ≥ |cos|` holds for ALL pairs, including distractors. recos inflates both correct and distractor scores; net discrimination doesn't improve.
2. **Noise sensitivity.** recos relies on ordinal structure (component ranking). Gaussian noise flips close-valued component orders, breaking ordinal concordance. Cosine measures linear correlation, which degrades gracefully.

**Diagnostic confirmation:** With exact power-law query (no noise), recos discrimination = 0.332 vs cosine 0.321 — recos is *slightly better*. But this advantage vanishes with any realistic noise (σ ≥ 0.1).

**Outcome.** G1 FAIL → do NOT promote. Stays opt-in as a diagnostic metric for future embeddings where ordinal concordance is the dominant signal and noise is low. NOT UQ-bearing — the "Report the Floor" rule does not apply.

**Lesson.** A better *matching* metric is not necessarily a better *retrieval* metric. When a paper's headline gain is on a matching task (STS), verify the gain transfers to a discrimination task (retrieval/ranking) before promoting. The monotonic-concordance capture range that helps matching can inflate distractor scores and hurt discrimination.

## 35. Stokes Calculus Wrappers — G-C STRUCTURAL FAIL + G-A Runtime FAIL (STAYS OPT-IN)

**Plan:** [314](../../.plans/314_stokes_calculus_wrappers.md) · **Bench:** [314](../../.benchmarks/314_stokes_calculus_goat.md) · **Feature:** `stokes_calculus` (opt-in root alias for `katgpt-core/dec_operators`)

**What it is.** Four Stokes-theorem wrappers on top of the DEC operators (§55): `belief_mass_divergence` (Fokker-Planck validator), `boundary_flux_mass` (region mass from boundary flux), `line_integral` (trajectory energy), `circulation_integral` (turn-count via rank-2 cochain). The headline claim was that line-integral smoothness would serve as a path-quality metric for NPC navigation.

**Gate results (3 gates, 1 PASS + 2 FAIL):**

| Gate | Target | Result | Status |
|---|---|---|---|
| **G-B** (boundary-flux mass) | ≥3× faster, error < 5% | **5.36× faster**, error 3.78% | ✅ **PASS** |
| **G-C** (line integral) | ≥20% fewer reversals | discriminates paths (Δ=1.872) but **cannot encode turn penalties** | ⚠️ **STRUCTURAL FAIL** |
| **G-A** (Fokker-Planck) | ≥1.5× earlier / ≥2× cheaper | riir-ai Plan 334: **9.5× slower, 36% lower F1** | ❌ **FAIL** |

**Root cause (G-C).** Rank-1 cochains (scalar fields on edges) cannot encode turn count — the line integral along a path is independent of how many times the path turns. The `circulation_integral` was added to address this (rank-2 cochain encoding orientation), but the structural limitation holds for the rank-1 `line_integral` primitive. This was confirmed empirically: `line_integral` discriminates paths by length (Δ=1.872) but not by turn structure.

**Root cause (G-A).** The Fokker-Planck validator (`belief_mass_divergence`) was measured in riir-ai Plan 334 against live HLA branching events. It was 9.5× slower than the direct event scan and produced 36% lower F1 — the divergence signal correlates with branching but doesn't improve detection quality over the simpler direct scan.

**Outcome.** Stays opt-in. The 4 primitives are all correct (15 unit tests, Stokes identities hold by construction); the boundary-flux mass (G-B) is a genuine 5.36× perf win. But the headline claims (line-integral path quality, Fokker-Planck anomaly detection) don't hold empirically.

**Lesson.** When the mathematical identity holds (Stokes' theorem is proven), the question is whether the *application* of that identity produces a downstream quality gain. A correct theorem ≠ a useful feature. The boundary-flux-as-region-mass trick works because it's purely computational (O(boundary) vs O(volume)); the Fokker-Planck-as-anomaly-detector fails because divergence correlation is weaker than direct event scanning.

📖 Feature: `stokes_calculus` (opt-in; root alias for `katgpt-core/dec_operators`).

## 36. MSA Blockwise Sparse Attention Family — 3× GOAT FAILED (STAYS OPT-IN PERMANENTLY)

**Plan:** [256](../../.plans/256_msa_blockwise_sparse_distillation.md) · **Research:** [225](../../.research/225_MSA_Blockwise_Sparse_Attention_Distillation.md) · **Features:** `msa_sparse`, `msa_per_group`, `msa_kv_outer`, `msa_adaptive_k` (all opt-in, permanently)

**What it is.** Distillation of MSA's (Blockwise Sparse Attention) key inference-time mechanisms into katgpt-rs's existing VortexFlow framework. Three trivial wins shipped (max-pool scoring, exp-free TopK, max+stddev scorer) plus three GOAT-gate experiments (per-GQA-group independent top-k, KV-outer sparse prefill, adaptive-k budget via sigmoid gate).

**Gate results (3 Phase-2 GOAT gates, ALL FAILED):**

| Gate | Metric | Result | Target | Status |
|---|---|---|---|---|
| **Per-group** | Coverage ratio | 1.003× | ≥ 1.5× | ❌ FAIL |
| **KV-outer** | Speedup @ 128K | 1.14× | ≥ 1.5× | ❌ FAIL |
| **KV-outer** | Speedup @ 512K | 0.83× | ≥ 1.5× | ❌ FAIL (regression) |
| **Adaptive-k** | Recall ratio | 0.629 | ≥ 0.90 | ❌ FAIL |
| Adaptive-k | Compute savings | 37.1% | ≥ 25% | ✅ PASS (but gate is AND) |

**Root cause (per-group).** Per-GQA-group independent top-k selection produces near-identical coverage to the shared top-k path (ratio 1.003× — essentially no difference at modelless scale). The grouping benefit only materializes with trained attention patterns that diverge across groups.

**Root cause (KV-outer).** KV-outer sparse prefill wins at short context (2.02× at 32K) but **regresses** at long context (0.83× at 512K — slower than dense). The overhead of the sparse selection scan grows with sequence length, eventually exceeding the savings.

**Root cause (adaptive-k).** The recall is **mathematically bounded**: recall normalized by fixed k is bounded by k_adaptive/k_fixed ≈ 20.14/32 = 0.63. The two GOAT criteria (≥25% savings → avg k ≤ 24, AND ≥90% recall → requires avg k ≥ 28.8) are in direct tension — they cannot both be satisfied simultaneously. A precision/weighted-recall metric would better reflect selection quality.

**Outcome.** All four `msa_*` features stay opt-in permanently. The Phase 3 arena benchmark (which would require trained weights + RULER) was deferred to Issue 014 (closed+removed) — not feasible modellessly. The three Phase 2 micro-benchmarks serve as modelless RULER proxies; their failures predict the arena would also fail.

**Lesson.** Two distinct failure modes: (1) **inference-time adaptations of training-time sparse patterns produce no quality gain** without the trained attention divergence (per-group); (2) **savings/recall criteria in direct tension** (adaptive-k) cannot both pass — the GOAT gate design itself was flawed, not just the implementation. When designing a dual-criterion gate (AND of two metrics), verify the criteria aren't structurally incompatible.

> **PASS-Redirects (synthesis):** Jiang et al. [arXiv:2407.02490 "MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention"] + Lai et al. [arXiv:2502.20766 "FlexPrefill: A Context-Aware Sparse Attention Framework for Long-Context LLM Prefilling"] — both target the KV-outer root cause (selection-scan overhead growing with P): MInference by hoisting patterns to a one-time offline calibration (the RT-Turbo/PFlash shape, `.docs/05_adaptation/lucebox_techniques.md` Technique 8), FlexPrefill by cheaper context-aware per-prompt selection. Neither changes this verdict's endpoints — re-open only if a runtime-selected variant beats the 512K regression under the same gate.

📖 Features: `msa_sparse` + 3 sub-features (all opt-in permanently). Phase 12 (2026-07-04): primitives moved to `katgpt-attn`.

## 37. Binned Blend Estimator — REAL ARENA STRICTLY HARMFUL (STAYS OPT-IN)

**Plan:** 436 / Issue 428 (removed) · **Bench:** [006](../../.benchmarks/006_shared_vs_independent_hl.md) · **Features:** `binned_blend`, `kernel_blend`, `contextual_bandit` (Plan 436 family — all opt-in)

**What it is.** Three modelless blend estimators for HLPlayer (Bomberman HL Arena, Plan 033): `binned_blend` (5-bin discretization on blast_proximity, per-(bin,arm) Q-table), `kernel_blend` (Nadaraya-Watson kernel estimator, 128-entry ring buffer, σ=0.15), and `contextual_bandit` (contextual bandit baseline). The goal was to replace the n-armed bandit with a modelless nonlinear estimator.

**Gate results:**

| Feature | Micro-env (synthetic) | Real Arena | Verdict |
|---|---|---|---|
| `binned_blend` | +8.7pp over n-armed ✅ | mean delta −93.0 vs baseline −11.67 (**8× WORSE**), survival 78.7% vs 80.7% | ❌ **STRICTLY HARMFUL** |
| `kernel_blend` | — | mean delta +78.5 (CI [+26.3, +130.8] at 99.9%), survival 81.7% (+1.2pp), Welch t=3.61 (p≈0.0003) | ✅ **RECOMMENDED** |
| `contextual_bandit` | — | baseline | — |

**Root cause (binned_blend).** The 5-bin discretization on `blast_proximity` is too coarse for real game features — the micro-env's synthetic distribution doesn't capture the real arena's feature geometry. The estimator overfits to the bin boundaries, producing catastrophic miscalibration in the real game (mean reward 8× worse than the simple n-armed baseline).

**Root cause (kernel_blend success).** The Nadaraya-Watson kernel estimator adapts to the actual feature density without discretization artifacts. The 20-seed wider study (Benchmark 432 reference in Cargo.toml, actual analysis in Bench 006) confirmed the gain holds at 99.9% CI.

**Outcome.** `binned_blend` stays opt-in as **documented evidence of a negative result** — do NOT use it in production. `kernel_blend` is the RECOMMENDED HLPlayer estimator (stays opt-in to preserve tournament A/B reproducibility — both n-armed and kernel paths must remain independently runnable).

**Lesson.** A modelless estimator that passes a synthetic micro-env GOAT can fail catastrophically (8× worse) on the real arena. The 5-bin discretization is the smoking gun — discretization resolution must match the real feature distribution, not the synthetic one. Always validate estimator gains on the real target distribution before promoting.

📖 Features: `binned_blend` (opt-in, harmful — do not use), `kernel_blend` (opt-in, recommended), `contextual_bandit` (opt-in, baseline). All gated behind `bomber`.

## 38. Block-Contiguous Ternary Layout (AoS) — G2 FAIL on Metal: 0.82× vs SoA Simdgroup (KEPT FOR CUDA)

**Issue:** 650 (riir-ai, resolved + removed) · **Bench:** [riir-ai 645](../../../riir-ai/.benchmarks/645_ternary_gemm_simdgroup.md) follow-up #5 · **Feature:** `ternary_group_scale` (same gate — additive types, no new flag)

**What it is.** `TernaryBlockAoS` — a `#[repr(C)]` 34-byte block co-locating `{u16 scale, [u8;16] pos_bits, [u8;16] neg_bits}` per 128-weight group, replacing the SoA three-`Vec` layout (`pos_bits` / `neg_bits` / `group_scale` in three distant allocations) to eliminate its 3× global-memory-access overhead in GPU kernels. Matches the on-disk `block_q2_0` footprint exactly. Shipped as CPU foundation in `katgpt-types/src/ternary_group.rs` (commit `a8e73737`): `to_block_contiguous()` conversion + `TernaryBlockContiguousWeights::matvec()` reference kernel.

**Gate results:**

| Gate | Metric | Measured | Verdict |
|---|---|---|---|
| **G1** | Correctness — AoS matvec vs `ternary_group_matvec_scalar` | Bit-identical across 6 shapes incl. real Bonsai dims (48×512, 1024×512, 128×4096, 512×17408) | ✅ PASS |
| **G2** | GEMM vs sequential (M3 Metal) | **3.12×** | ✅ PASS (gate ≥3×) |
| **G2** | GEMM vs SoA simdgroup kernel (M3 Metal) | **0.82×** — SLOWER | ❌ **FAIL** |

**Root cause.** The block-contiguous layout has worse memory coalescing on Metal than the SoA simdgroup kernel it aimed to beat — the 1-fetch-per-group win does not offset the lost coalescing on this backend.

**Bonus finding that closed the issue.** The motivating premise was Bench 645's 1.89× simdgroup ceiling. A re-run on M3 Metal measured the SoA simdgroup kernel at **5.89× (8×8) / 6.27× (8×32)** vs sequential — the 1.89× is not reproducible (different device, GPU contention per Bench 649, or pre-8×32-optimization measurement). The investigation target evaporated.

**Outcome.** No promotion. The CPU foundation + GPU upload infra stay in-tree as validated code for **potential CUDA (4090) use**, where the coalescing trade-off may differ. The `ternary_gemm_simdgroup` re-promotion question is filed separately in riir-ai.

**Lesson.** Re-measure the baseline before optimizing against it — a stale ceiling (1.89×, likely contention- or device-skewed) can motivate a whole layout redesign that the current kernel already beats (5.89–6.27×). Mirrors the AGENTS.md GPU-exclusivity rule: a gate measured under contention is not evidence.


## 39. UGC Certified Unmasking Schedules — G1b Promotion FAIL (substrate lands always-on)

**Feature flag: NONE — deliberately.** The estimator + schedule construction are
correct and land as an always-on diagnostic substrate
(`katgpt-core/src/ugc_schedule.rs`, pure math + a denoiser trait, zero deps beyond
`crate::types::Rng`); the `ugc_schedule` feature-flag plan opens ONLY if a future
G1b gate passes — it did not.

**What failed:** the promotion gate G1b (dllm, release, 5 cells —
[Bench 659](../../.benchmarks/659_ugc_certified_schedule_poc.md), Research 485,
Issue 664). UGC's predicted optimal schedule length N*=48 (the complexity formula
saturates at ε≈0 there) vs the actual ε(N) curve measured **flat at the MC-noise
floor** for all N ≥ 3 — the certified-complexity estimator sees structure the loss
cannot feel. Cell verdicts: bs=8 → **−8.6%** (worse than preset-8), bs=15 →
**+1.4%** (noise). Preset-8 wins outright (2.49/2.84 vs early-exit baselines).

**What passed:** G1 / G1-cert / G1(exact) / G4 ALL PASS — the estimator is
faithful to the paper, the schedules are constructible, and the arithmetic is
exact. This is a **regime negative**, not an implementation bug: the GOAT track is
CLOSED. Reopen only on a dllm regime with measurably non-flat ε(N).


## 40. Pair-Scored Path Selection — G2 FAIL: the two signals are collinear (Issue 671)

**Feature flag:** rides `bigram_markov` in katgpt-speculative (the substrate
`katgpt_speculative::pair_select` ships inside the existing opt-in feature; no
new flag). Issue 671 RESOLVED-NEGATIVE + removed 2026-08-19; research record
[Research 490](../../.research/490_DFlash2_Pair_Scored_Path_Selection.md), gate
[riir-ai Bench 699](../../../riir-ai/.benchmarks/699_issue671_pair_scored_selection_gate.md).

**The hypothesis:** DFlash 2's path selector lifts parallel drafting by
composing per-position marginal evidence `U_t(b)` + adjacent-pair coherence.
The modelless instantiation — `S_t(a,b) = ln U_t(b) + λ_t · ln P_bigram(b|a)`
with `U_t` from t-step forward propagation of the same bigram table, walked
greedily, three λ gates (Flat/Entropy/Margin) — should lift chain acceptance
from 0.2987 toward the 0.5–0.7 break-even band without training and without
tree verification.

**What passed:** G1 — λ=0 ≡ argmax-of-marginals **bit-identical** on all
2,000 subsampled positions at full vocab (10 unit tests, t-step propagation
pinned vs dense matrix powers). The substrate is correct.

**What failed:** G2 — best pair-scored acceptance **0.2855 < 0.5** (chain full
0.2987 / sub 0.2845, argmax-of-marginals 0.2730, tree-oracle ceiling 0.8770,
unigram control 0.1310 degenerates as designed). The entire λ × gate matrix
(λ ∈ {0.5, 1, 2} × {Flat, Entropy, Margin}) spans 0.2780–0.2845 — 0.98×–1.00×
of the chain. **Root cause: collinearity** — `U_t` is forward-propagated from
the *same* transition table the pair term scores with; no orthogonal evidence
enters the selection, so the λ sweep interpolates between two arms that both
sit at ~0.27–0.28. DFlash 2's selector works because its `U` (a neural
drafter's context-conditioned logits) is evidence *independent* of the trained
pair coherence; with bigram-only statistics the composition is a re-weighting,
not an information gain.

**Two secondary findings:** (1) the entropy gate TIES flat λ exactly (0.2845 =
0.2845) — the `H(h_t)` analog is dead weight at this operating point; (2) the
0.877 tree ceiling is an *oracle containment* measure, not single-path
realizable by any scorer built from the table's own statistics — the G2 target
(0.5) was calibrated on DFlash 2's numbers where a neural drafter's Recall@16
is 99.5%, vs the modelless table's ~22% top-1 rate (Bench 664); the headroom
decomposition does not transfer to a drafter this weak.

**Redirect:** the selection headroom (0.28 → 0.87 containment) needs a real
drafter's context-conditioned `U` — riir-train lineage / the Issue 717
tree-verify path. Reopen only with an information source orthogonal to the
pair table.

## 41. Distributional Steering — FK-vs-gradient separation claim NOT reproduced (Plan 577, G1 FAIL-partial)

**Feature flag:** `distributional_steering` (opt-in, stays opt-in). Plan 577
COMPLETE 2026-08-26; gate [Bench 682](../../.benchmarks/682_distributional_steering_goat.md),
research [Research 505](../../.research/505_Mean_Field_Distributional_Steering.md)
(Howard & Nüsken, arXiv:2608.08770).

**What the paper claims:** on their 1-D toy, Feynman-Kac (FK) steering
achieves a markedly better optimality-gap/λ trade-off than gradient-only
steering — the FK correction is the differentiating mechanism.

**What reproduced:** the targeting minimum at **λ\*=5 on both noise
schedules** (clean V-curves over the λ grid) and the exact-J trade-off
structure; λ\*=10 held on one of two schedules (the other's curve is flat
at the 8-seed noise floor ≈±0.013).

**What failed:** the **separation claim**. In the 1-D broad-kernel Langevin
regime, gradient-only ≈ FK — the optimality gaps differ in the **third
decimal**. Mechanism: with a broad kernel the position steering dominates
and the FK log-weight correction is a small additive term, not a different
driver. The paper's separation regime is the diffusion-sampler setting
(where FK tilts the *sampling* measure); a modelless weights-only Langevin
harness cannot exhibit it.

**Secondary negatives:** G2's sub-µs/particle gate is **structurally
infeasible** for exact O(N²) MMD (measured 9045 ns/particle/step @ N=1000;
the kernel build alone is 10⁶ `fast_exp` — the paper's "Picard = 0.04% of
runtime" is relative to network evals a modelless stack doesn't have);
weights-only degenerates to ESS→1 by λ≈7.5 over 30 steps without resampling
(a real property, clip-bounded — decisive for sampling consumers, harmless
for crowd-salience).

**Load-bearing positives kept:** the Research 505 **Table-2 MMD sign slip**
corrected in-module (`Ψ = 2[emb_ν − emb_μ]`, finite-difference-pinned) and
the **Picard damping law `α = min(1, 2/λ)`** (iteration Jacobian ≈ 0.2λ —
damping 1.0 diverges for λ≳5 regardless of iteration count).

**Reopen paths:** a diffusion-sampler-shaped harness (the prerequisite for
the riir-ai crowd-targeting plan, Guide 344 — deliberately NOT filed while
this stays closed); approximate kernel features for the G2 threshold;
N≲300 populations are already sub-µs per particle.
