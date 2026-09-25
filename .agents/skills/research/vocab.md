# Vocabulary Translation Tables — Supporting Reference for the Research Skill

> **When to read this file:** the research skill §1 step 2 (vocabulary translation before grepping) requires these tables. Read the section(s) relevant to your paper's domain. Papers and our codebase use different words for the same mechanism — these tables are the translation layer that prevents false-novelty claims from paper-vocabulary-only greps.

## 1. Standing latent-state vocabulary (ALWAYS include, even for non-latent papers)

Most architecture/training papers have a latent-space reframing. Always grep BOTH the paper terms AND these codebase equivalents:

| Paper vocabulary | Codebase vocabulary |
|---|---|
| "residual stream" / "hidden state" / "activation" | "belief state", "latent subspace", "sense projection" |
| "layer" / "depth" / "stage" | "decision stage", "functor application", "cgsp cycle", "consolidation tick" |
| "width" / "dimension" / "capacity" | "latent subspace", "active projection channel", "sense channel" |
| "carry-forward" / "bypass" / "skip" | "leaky integrator", "dormant subspace", "decay gate", "persistence" |
| "collapse" / "degeneration" / "valley" | "coherence decay", "re-estimation trigger", "staleness" |
| "bottleneck" / "narrowing" | "subspace projection", "channel selection", "zone gating" |
| "fixed-point" / "deterministic" / "committed" | "LatCal", "lattice calculus", "BLAKE3 commitment", "raw scalar bridge" |
| "divergence" / "flux" / "∇·F" / "density change" | "codifferential", "δ", "DEC divergence", "belief_mass_divergence" |
| "curl" / "vorticity" / "∇×F" / "circulation" | "d₁", "DEC curl", "exterior_derivative rank 1→2" |
| "boundary" / "∂M" / "frontier" / "perimeter" | "exterior_derivative", "d", "coboundary operator", "boundary_flux_mass" |
| "line integral" / "trajectory energy" / "path cost" | "line_integral", "edge field sum", "rank-1 cochain path sum" |
| "Stokes theorem" / "divergence theorem" / "Green's/Gauss" | "DEC identity d∘d=0", "curl(grad)=0", "div(curl)=0", "hodge_decompose" |
| "Hodge decomposition" / "exact/coexact/harmonic" / "Helmholtz" | "hodge_decompose", "DecFlowField", "exact_flow/coexact_flow/harmonic_flow" |
| "Fokker-Planck" / "continuity equation" / "mass conservation" | "belief_mass_divergence", "codifferential on belief cochain" |
| "cell complex" / "mesh" / "simplicial" / "cubical" | "CellComplex", "CochainField", "grid_2d" |

## 2. Standing per-NPC runtime / freeze-thaw / personality vocabulary

ALWAYS include when the paper touches per-entity state, memory, personality, swap, evaluator/judge/critic, or selective erasure/forgetting. The `riir-ai/.research/` corpus is SATURATED in this space — paper-vocabulary-only greps produce false novelty claims (see R320 failure in the main skill).

| Paper vocabulary | Codebase vocabulary |
|---|---|
| "personality swap" / "personality drift" / "character shift" | "committed personality", "freeze/thaw cadence", `ArchetypeBlendShard`, `KarcShard` |
| "selective forgetting" / "memory erasure on swap" | "non-interference branches", "branch-local", `BranchBank`, "orthogonal subspace projection" |
| "survives swap" / "replay-deterministic personality" | "sampling invariance", "quorum-verifiable", "bit-identical across nodes", "FAME Proposition 3" |
| "epoch boundary" / "checkpoint replacement" / "non-stationary utility" | "freeze/thaw cadence", "consolidation sleep-cycle", `tau_reest`, "coherence < tau" |
| "co-evolution" / "evaluator replacement" / "moving target" | "personality divergence", "direction vector drift", "emergent personality at crowd scale" |
| "evaluator" / "judge" / "critic" / "verifier" (per-entity) | "claim verifier", "CLR vote", "Salience Tri-Gate", `ConstraintPruner` |
| "frozen snapshot" / "frozen artifact" (generic) | NAME THE CONCRETE TYPE: `KarcShard`, `ArchetypeBlendShard`, `BranchBank` snapshot, `ZoneGeometryPod`, `MerkleFrozenEnvelope`, `SleepAnticipationShard` |
| "cache invalidation on swap" / "dependent records" | `DecCache::mark_face_destroyed`, `ZoneGeometryCache::invalidate`, `topology_version` bump |

## 3. Standing compute-unit translation (MANDATORY for agent/LLM papers — R368 lesson)

Papers increasingly use "LLM forward pass" / "LLM call" as the compute unit for a decision. Our codebase uses different compute units. ALWAYS translate the compute unit, not just the semantic name:

| Paper vocabulary | Codebase vocabulary |
|---|---|
| "LLM decides what to write/record" | `SpeculativeGenerator` (draft) + `ScreeningPruner` (relevance) + `ConstraintPruner` (validity) |
| "LLM decides what to retrieve/read" | `SpeculativeGenerator` + `ScreeningPruner` + AnyRAG escalation gate |
| "LLM judges/verifies/critiques a claim" | CLR vote + SalienceTriGate + Claim Rubric L1/L2/L3 |
| "LLM reviews trajectory + rewrites code/prompts" | Raven/δ-Mem consolidation + MAPE-K self-healing (architectural analog; quality parity needs PoC) |
| "meta-LLM generates novel semantic content" | **NO modelless analog** — genuine NO-GAIN if the value IS the generation |

## 4. Standing substrate-translation vocabulary (MANDATORY for hardware/accelerator/NMP/PIM/ASIC/system papers — R418 lesson)

Papers framed in hardware vocabulary describe techniques whose *implementation substrate* is hardware, but whose *value* is substrate-independent. We simulate hardware dequant/accelerator techniques in software SIMD (Research 110 Ciot). ALWAYS translate the substrate. Before PASS-ing on a hardware paper, grep BOTH sets:

| Paper vocabulary (hardware) | Codebase vocabulary (software SIMD) |
|---|---|
| DQB / near-memory processing unit / PE | SIMD dequant fn, `dequant_via_lut`, `simd_lut_dequant` |
| HBM / pseudo-channel / memory controller | L1 cache, register file, SIMD lane |
| channel-aware layout / interleaved | cache-line-aligned, `AlignedWeightMatrix`, struct-of-arrays |
| wire-mapping FP-to-FP / 2:1 mux type conversion | bit-cast, `f16::from_bits`, SIMD shuffle/permute |
| LUT-based INT-to-FP / sign extension | pre-computed lookup table, `[f32; 256]`, SIMD gather |
| sideband tag / 3-bit tag / mode select | `QuantFormat` enum, runtime dispatch, `match` on format |
| S/Z buffer / scaling-factor cache | L1-resident scale cache, `BlockQ4K { d, dmin, scales, qs }` |
| fused DQ-GEMM kernel | fused dequantize-matmul, `simd_dot_f16_f32`, dequant-in-register |
| kernel selection / batch threshold | breakeven routing, Plan 218, memory-bound vs compute-bound |
| tensor core / CUDA core / SM | CPU SIMD lane, FPU, NEON/AVX2 lane (the *role*, not hardware) |
| io_uring / DPDK / RDMA / zero-copy bypass | zero-allocation hot path, `_buf` pattern, `pool.rs` |
| TPU systolic array / PE mesh | blocked matmul, `simd_matmul_rows`, tiled GEMM |

## 5. Standing database-substrate translation (MANDATORY for database/systems/storage/agent-state papers — R300 lesson)

Papers framed in database-engine vocabulary describe **access patterns** whose *implementation substrate* is a database engine, but whose *value* is the access pattern. We ship a database substrate (`riir-neuron-db`: Pod + lock-free `ShardIndex` + Merkle + MAPE-K + Raven/δ-Mem + `ItemEmbedIndex` + vibe KG) WITHOUT a SQL/Cypher engine. **Conflating "we don't have a fused query planner" with "we don't have a database" is the R300 false-PASS root cause.**

| Paper vocabulary (database) | Codebase vocabulary (neuron-db) |
|---|---|
| experience graph / reward-bearing search tree / UCB node | `NeuronShard` + lineage, `TrialRecord`, `KarcShard`, `EpisodeBuffer` |
| materialized view / training-data extraction | `consolidation.rs` Raven/δ-Mem output, `ShardCompactor` cold-tier product |
| vector-seeded graph traversal / ANN + relational + graph | `ItemEmbedIndex::query` (latent cosine ANN) composed with `vibe.rs` KgTripleTemplate edges |
| AS-OF temporal query / bi-temporal / time travel | `MerkleFrozenEnvelope` version chain, `PersonalityLedger` ring, `AuditLog` backward replay |
| frontier selection / UCB / visit_count + reward | `BanditPruner`, `ManifoldBandit` (Plan 370), `MCTS`, `GZeroPlayer` |
| governed view maintenance / retraction / invalidate derived | `MapeKLoop` self-heal cascade, `BackupMirror::integrity_check_3way` |
| change log / CDC / logical step number | `PersonalityLedger` ring, `AuditLog` entries, `TrialLog` hash chain |
| concurrent tree search atomicity / MCTS backprop race | `FunctorEntry` atomic Arc-swap, `LoRAWeightVersion` lock-free read |
| episode buffer / experience replay | `EpisodeBuffer` FIFO, Raven/δ-Mem input queue |
| unified query planner / SQL/Cypher/Vector fused | **NO direct analog** — individual access patterns map to latent ops + neuro-symbolic edges + raw commitment |

## 6. Standing coefficient / blend-shape vocabulary (MANDATORY when the paper's mechanism is a fitted/adaptive/mixture coefficient — Le Critique lesson, canonical failure #3, 2026-08-20)

Papers about adaptive blending, mixture coefficients, ensemble weighting, baseline interpolation, or "how much to trust estimator A vs B" ship in our codebase as **hand-tuned constants** with "Weight on the X axis" style doc comments — never under the words "blend"/"mixture"/"adaptive". And the shape ships where you don't expect it: the Le Critique miss was a blend-vocab grep scoped to katgpt-core while the hit sat in riir-clippy's `self_evolve.rs`.

**Grep set (run repo-wide across ALL repos, never scoped to the expected home):**

`W_[A-Z]|const [A-Z_]+: f32|_WEIGHT|weight: f32|alpha: f32|trust: f32|blend_factor`

| Paper vocabulary | Codebase vocabulary (shipped instances) |
|---|---|
| "adaptive mixture coefficient" / "fitted blend weight" / "interpolation ratio" | `W_EVO = 0.6 / W_RATE = 0.4` (riir-clippy `select_best_candidate` — the Le Critique direct hit, ρ hand-pinned at 0.4); `alpha: f32` (`DualLeoMixer`, fixed 0.3); `latent_weight` (retrieval fusion — measured INERT, T3.5); `RetrievalWeights.trust = 0.2` (EvidenceTier); `KvRoutingConfig::blend_factor` (fixed thresholds — EVPO-style) |
| "ensemble weighting" / "combine predictors by measured accuracy" | the same constants — if FIXED, the paper's contribution is usually making them data-fit (TETHER ρ*) |
| "realized outcome" / "return" / "reward stream" (what the coefficient is fit against) | `EvolveRecorder::record_outcome` (riir-clippy, fires on every applied fix), trajectory success counts, bench verdict streams — if the feed already fires, the adaptive version is an A/B away |

**Rule:** a paper whose core mechanism is FITTING a coefficient structurally matches every hand-pinned constant implementing the same convex combination. Coverage requires reading the **containing expression** (the §3.6 granularity rule in the main skill) — and checking whether the outcome stream it would be fit against already fires.

## 7. Standing inference-substrate vocabulary (MANDATORY for quant/kernel/loader/encoder papers — riir-infer)

Papers about quantization formats, kernel ports, checkpoint loaders, or encoder forwards describe mechanisms whose substrate home is now `riir-infer` (public, carved from riir-ai 2026-09-22). Grep BOTH sets before any "already ships" / routing verdict in this space:

| Paper vocabulary | Codebase vocabulary (riir-infer) |
|---|---|
| "quantization format" / "weight-only quant" / "GPTQ/AWQ/EXL3-class" | `quant/` module, the EXL3-trellis lane (CPU reference = oracle + fast arm + `riir-infer-gpu` kernels), GDN quant certification (`deltanet/`) |
| "dequant kernel" / "fused dequant-GEMM" | `simd/` dequant forwards, fused dequant-matmul, dequant-in-register |
| "checkpoint loader" / "GGUF/safetensors" / "tokenizer" | `gguf_loader.rs`, `safetensors_loader.rs`, `tokenizer.rs` (BPE vs SentencePiece selection) |
| "architecture port" / "model forward" | `transformer/`, `gemma_layer.rs`, `llama_layer.rs`, `ternary_layer.rs`, `rope.rs`, `wall.rs` |
| "rotation" / "Hadamard" / "incoherence processing" | the Bonsai2 rotation lane (`bonsai2_hadamard`) |
| "kernel port" / "backend port" / "Metal/CUDA/Triton port" | the laya lane discipline: flat `Vec<f32>` ops, `crates/riir-infer-laya` MSL kernels, `crates/riir-infer-gpu` (wgpu/CubeCL), `laya-riir-cuda` |
| "bit-identical port" / "parity gate" / "golden forward" | the G5 gate: top-1 ≥ 99.9% + p-drift ≤ 1e-3 vs frozen captures, per checkpoint, BEFORE any published number |

## 8. Standing decision-serving vocabulary (MANDATORY for calibration/abstention/selective-prediction/serving papers — riir-reflex)

Papers about calibrated decision serving, abstention, selective prediction, or comparison arenas have a concrete public home: `riir-reflex` (public 2026-09-23). Grep BOTH sets before any novelty or routing verdict in this space:

| Paper vocabulary | Codebase vocabulary (riir-reflex) |
|---|---|
| "selective prediction" / "abstention" / "reject option" | the `noul` answer, fused abstain (calibration score + `CorpusDistanceGate`), abstention first-class in `decision_wire` |
| "confidence calibration" / "temperature scaling" / "conformal" | `SigmoidGateCalibrator`, sigmoid normalization (never softmax), per-suite fitted birth thresholds (harness cal-slice percentile) |
| "corpus as model" / "retrieval-augmented zero-shot" / "non-parametric classifier" | `Lz4FlexDrafter` corpus-is-the-model option scoring, `pick_domain` corpus routing over hashed-feature embedding (`embed.rs`) |
| "decision API" / "typed decision endpoint" | `decision_wire` (`choice`/`score`/`noul`), stdio line-JSON + HTTP edge (`serve.rs`), `game_heads.rs` fitted serving heads |
| "comparison arena" / "benchmark harness" / "leaderboard" | `harness/` (9 dataset suites, corpus-cap + `--cal-select-cap` levers), `lanes/` (laya CPU/Metal/CUDA, CLM, GLiNER, AgentJev) |
| "distribution shift probe" / "OOD rig" | pair-head A/B probes, held-out rig policies, G5 parity gating every published laya number |

## 9. Unified decision rule — substrate-as-instantiation vs mechanism-as-value

This rule prevents false-PASS/false-redirect across four paper classes (R418 hardware + R368 LLM + R300 database + Flow Sampling training-math):

- **Value = technique / access-pattern / decision-structure / math** (not the substrate) → substrate is one *instantiation* → translate to our substrate → GOAT/Gain candidate.
  - **R418 hardware:** LUT lookup / shared ALU / co-located layout / dispatch tag (StreamDQ — DQB-in-HBM is instantiation, SIMD dequant is ours; shipped 2.3×).
  - **R368 LLM:** decision structure (AutoMem LOG/PLAN — LLM is instantiation, probe/draft/pruner is ours). **When you see "N LLM calls/step", FIRST ask "what decision is each call computing?"** — not "violates 20Hz budget, NO-GAIN". R169 guard applies ONLY to genuine LLM-dependent process.
  - **R300 database:** access pattern (Trellis — Axiom/Velox is instantiation, `riir-neuron-db` Pod + Merkle + vibe KG is ours). `riir-neuron-db` IS a database; prefer latent-first + neuro-symbolic translations over `AnyRAG` external escalation.
  - **Flow Sampling training-math:** closed-form drift / conditional score / Riemannian correction / regression target (training loop is instantiation, dllm + Latent Field Steering + freeze/thaw is ours).
- **Value = substrate-fabrication-advance itself** (new transistor geometry, new query optimizer algorithm, new optimizer like Muon, new loss function, new RL algorithm, semantic code generation) → no modelless analog → PASS or → riir-train.

## 10. Worked examples

**DiPOD paper → riir-ai code:** paper-vocabulary grep misses `latent_functor/reestimation.rs` which ships DiPOD's "interleave self-distillation when ELBO drifts" as "coherence-driven re-estimation scheduler when coherence < tau_reest". Vocabulary translation is the only defense — notes framing can use codebase vocabulary that paper-vocabulary grep misses on BOTH layers.

**Stokes paper → katgpt-rs DEC code:** paper-vocabulary grep for `stokes|divergence theorem|fokker-planck` returns ZERO hits. Codebase-vocabulary grep for `codifferential|exterior_derivative|hodge_decompose|DecFlowField` hits `katgpt-dec/src/operators.rs`, `katgpt-dec/src/hodge.rs`, `katgpt-dec/src/flow.rs`. The Generalized Stokes' theorem machinery ships as DEC operators but no note framed it in Stokes vocabulary.
