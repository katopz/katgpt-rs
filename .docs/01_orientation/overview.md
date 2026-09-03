# katgpt-rs: Overview

## What It Is

A from-scratch Rust implementation of a GPT-2 style transformer with speculative decoding, designed as an educational/performance research vehicle. No ML frameworks — just `Vec<f32>`, matmul, and hand-tuned attention kernels.

## Project Goals

- CPU-first inference engine with zero-allocation hot paths
- Speculative decoding pipeline (DDTree + DFlash + Leviathan verification)
- Domain-specific constraint pruning (Sudoku, Rust AST via Validator)
- BPE tokenizer + SynPruner for Rust syntax validation
- Sub-millisecond inference on Apple Silicon
- Discrete Diffusion Forcing (dLLM) research with block-parallel denoising

## Current Capabilities

- Single-token autoregressive generation: ~900K tok/s (micro config)
- DFlash marginal prediction: ~4.2M tok/s
- DDTree build: ~431K trees/s
- Speculative decoding: ~1.64M tok/s (AR Draft)
- forward_raven (16 slots): ~1.6M trees/s
- raven_recall (1000 noise): ~9.3M tok/s
- SIMD-accelerated matmul/HLA kernels: 15.6M ops/s [16×16] NEON (Plan 060)
- forward_hla: ~939K tok/s (single-core, 30K CCU feasible)
- forward_ahla: ~1.2M tok/s (single-core)
- TurboQuant 3-bit KV cache: 5.3× compression, 0.99 attention correlation (legacy baseline)
- OCTOPUS octahedral triplet KV cache: 12.2× compression, 0.9512 cosine at 2-bit, -22% to -49% MSE vs SQ — primary KV compression, zero calibration (Plan 099, default-on)
- SpectralQuant calibrated KV cache: 9.1× compression, 0.9917 cosine — secondary KV compression, per-dimension water-fill (Plan 077, default-on)
- ELF SDE noise injection: 10-22× path diversity, logit-normal schedule (Plan 079, default-on)
- CNA Steering: contrastive neuron attribution + sparse modulation, GOAT proved (Plan 087, default-on)
- Deep Manifold: L2/KL residual fixed-point scoring, GOAT 6/6 (Plan 085, default-on)
- Federation: symmetric KL boundary alignment between experts (Plan 085, default-on)
- dLLM Discrete Diffusion Forcing: block-parallel denoising (behind `"dllm"` feature, Plan 066)
- SP-KV self-pruned KV attention: 3-10× KV reduction with utility prediction (behind `"sp_kv"` feature, Plan 070)
- PFlash block-sparse prefill: up to 21.3× sequence reduction, 100% NIAH retrieval
- MaxSim late-interaction scoring: 7.46× SIMD speedup (behind `"maxsim"` feature, Plan 080)
- SimpleTES RPUCG loop: wide>narrow budget scaling (behind `"tes_loop"` feature, Plan 086)
- GDN2 Gated DeltaNet-2: O(1) recurrent attention with decoupled erase/write gates (Plan 105, GOAT 14/14, default-on)
- DashAttention: adaptive sparse hierarchical attention via α-entmax routing (Plan 106, GOAT 9/9, default-on)
- Auto-Dreamer: offline memory consolidation with cadence scheduler + Q-value clustering (Plan 107, GOAT 8/8, default-on)
- LT2 Looped Inference: weight-shared T-pass loop with hybrid SDPA+AHLA dispatch (Plan 108, GOAT 8/8, default-on)
- DMax Soft Parallel Decode: hybrid token/mask embeddings with contiguous prefix promotion (Plan 109, GOAT 7/7, default-on)
- EqR Convergence Selection: Top1Converged picks smallest marginal-change residual (Plan 119, default-on)
- Subterranean Procedure Compilation: user-defined token-rewriting procedures compiled to zero-cost native code (Plan 110, default-on)
- SR²AM Configurator Bandit: per-turn planning regulation via UCB1 (Plan 112, default-on)
- Data Gate: self-play stability via task-level filtering (Plan 111, default-on)
- Plasma Path: ternary SIMD matvec with bit-plane ternary weights, GOAT 5/5 (Plan 117, default-on)
- Parallel-Probe 2D: consensus-based parallel branch control for N branches, GOAT 7/7 (Plan 133, default-on)
- Training-Free Loop: ODE-motivated damped sub-stepping for inference-time refinement, GOAT 4/4 (Plan 136, default-on)
- Newton-Schulz Orthogonalization: 5-iteration cubic fixed-point for Muon-family optimizer weight matrices, GOAT 25/25 (Plan 152, default-on)
- River-Valley Diagnostics: subspace ratios, effective rank, update cosine similarity for convergence analysis, GOAT 25/25 (Plan 152, default-on)
- Sleep Consolidation: offline recursive memory consolidation at KV eviction into GDN2 fast weights, GOAT 14/14 (Plan 154, default-on)
- Spectral Hierarchy: eigenspace alignment + Haar wavelets + Cauchy interlacing for KG extraction validation (Plan 156, default-on)
- Roofline Cost Model: GPU operator runtime prediction via calibrated peak throughput, ~5µs CPU estimate (Plan 159, default-on)
- Tiled Attention: tiled online-softmax flash attention for CPU SIMD (Plan 115)
- Parallax Attention: streaming covariance-corrected local linear attention (Plan 135, opt-in)
- CODA Fusion: fused SIMD kernels matmul+residual+rmsnorm+activation (Plan 103)
- MoA Inference: token-adaptive Mixture-of-Activations SwiGLU over 7-activation dictionary (Plan 158, default-on, GOAT)
- LEO All-Goals: goal-conditioned Q-value trait framework — LeoHead + vectorized Bellman (Plan 155, default-on, SUPER GOAT)
- Dual LEO: teacher-student Q-value mixing + autocurriculum sampling (Plan 155, default-on, SUPER GOAT)
- Sigmoid Margin: SigLIP-style softplus margin loss + dimension sufficiency bound (Plan 157, default-on, GOAT 7/7)
- Kog CPU Fusion: RMSNorm gamma folding + QKV interleaving for monokernel throughput (Plan 160, opt-in)
- Hybrid OCT+PQ: default KV codec — OCT triplet + PlanarQuant 2D Givens rotation (Plan 101, default-on)
- FlashAR Consensus: dual-path ternary thermal routing for consensus tri-mode (Plan 166, GOAT 9/9, default-on)
- Budget Adaptation: compression-adaptive decode budget scaling (Plan 167, GOAT 8/8, default-on)
- Hydra Budget: emergent self-repair layer skipping (Plan 165, GOAT 4/4, default-on)
- GEPA-D Reflective: Pareto bandit config evolution (Plan 164, GOAT 4/4, default-on)
- PhraseBoost: context trie phrase boosting for DDTree (Plan 164, GOAT 5/5, default-on)
- 740+ tests passing (245 test files), 178 examples across 25 groups
- Shared `katgpt-core` crate: types (Config, enums, math utilities), SIMD kernels — extracted for multi-crate reuse
- `QwenDeltaNet` model architecture: hybrid DeltaNet/Attention per-layer config (Plan 182)
- AND-OR DDTree decomposition: relevance-signal hierarchical goal decomposition with memoized subgoals (Plan 190)
- MUX superposition tree search: MuxSpanPruner + MuxDdTree + MuxBfs + mux_demux verifier + MuxBanditWidth arm selector (mux_pruner, mux_ddtree, mux_bfs, mux_demux features)
- LinOSS + ModalSpec drafter: oscillatory state-space cell + Fourier modal speculative drafting (modal_spec feature)
- RiM reasoning buffer slots: K×M reasoning blocks prepended to input, zero-cost slot reuse (rim_slots feature, Plan 172)
- Wall attention: W_g gate projection per KV head dimension, sigmoid-gated attention bypass (wall_attention feature, Plan 173)
- ManifoldPruner: ManifoldE point-to-manifold soft validity scoring + kernel-tricked relevance (behind `"manifold_pruner"` feature, Plan 234, GOAT G1 FAIL — demoted, opt-in only)
- `traits.rs` module in katgpt-core: GameState, RolloutPolicy, StateHeuristic, ActionSpaceLog, ConstraintPruner, ScreeningPruner, SpeculativeGenerator, CollapseDetector, DominoPruner, CompletionHorizon, PartialScorer, ProblemMutator, BestBuddyAligner, DataGate, LeoHead, DualLeoMixer, AutocurriculumSampler, GenerativeConstraintPruner traits
- Sense Composition: KG Latent Octree NPC sense modules with ternary bit-plane projection, GM override dispatch, lock-free hot-swap, and bandit-quality feedback (Plan 221)
- SLoD Spectral Level-of-Detail Pruner: Poincaré ball hyperbolic geometry + heat diffusion on kNN Laplacians for multi-scale KG resolution control (Plan 235, default-ON, GOAT G1–G6)
- Schema Centroid: Per-class embedding centroids for informed KG entity initialization with controlled perturbation (Plan 237, default-ON, GOAT 7/7)
- BAKE Precision-Gated Bayesian Embedding: Per-dimension precision tracking for KG embeddings with O(8) arithmetic, zero-alloc (Plan 236, opt-in, GOAT 10/10 but marginal drift 4.7%)
- Shard Embedding: Johnson-Lindenstrauss random orthogonal projection [f32;64]→[f32;8] for O(1) cosine similarity shard lookup (Plan 230) — **🪦 DEPRECATED (Issue 139):** m=8 violates JL lower bound 200×; marked `#[deprecated]`, zero runtime consumers
- NFCoT FlowScore Drafter: Inference-time normalizing flow density scoring for speculative candidates, zero training (Plan 229)
- Union Bound Confidence: Additive branch confidence via Boole's inequality (Plan 231, default-ON, GOAT 6/6)
- PathwayTracker: Intrinsic pathway stability detection (Plan 231, default-ON, GOAT 7/7)
- FederationComposer: Explicit pruning with residual early termination (Plan 231, default-ON, GOAT 7/7)
- Closed-Unit Compaction Gate (CUCG): generic rubric-gated trajectory compaction primitive (SelfCompact, arxiv 2606.23525) — fires at structurally-safe moments instead of fixed token thresholds. evaluate() 8.91ns, 112.9M/s. Super-GOAT: trajectory compaction and shard freeze are the same primitive (G7). Default-on (Plan 333, 2026-06-25).
- Phase Separation Probe: per-entity minimum circular distance on a phase circle, distilled from the Lonely Runner Conjecture (Plan 571, arxiv 0710.4495). O(N log N) production path (`phase_separation_sorted`). LRC guarantees (proven N≤7) every entity cycles through `phase_separation ≥ 1/N` — a coverage guarantee no existing primitive provides. Think-brain primitive for zone-attention routing / curiosity / coverage scoring. Default-on (Plan 571, 2026-08-07, G1–G4 ALL PASS).

## Module Structure

> **Snapshot caveat (2026-07-04):** the per-module listing below is a frozen
> pre-Phase-7 snapshot. The canonical crate layout lives in
> [`README.md` § Crate Dependency DAG](../README.md#crate-dependency-dag) and
> the migration history lives in
[`.proposals/003_src_consolidation_master.md`](../../.proposals/003_src_consolidation_master.md)
> (Phases 0–11 DONE; Phase 12 final sweep pending). Notable moves not reflected
> below: Phase 8 (`closure_wire`, `screening` → `katgpt-pruners`; `rerank` →
> `katgpt-attn-match`), Phase 9 (`mbu`, `tf_loop`, `dense_mesh`, `swir` →
> `katgpt-transformer`), Phase 10 (`cce`, `salience`, `trigger_gate`,
> `skill_opt`, `ssd_block`, `cumprodsum`, `alloc`, `llmexec_guard`,
> `memory_soup_lora`, `mux_demux`, `channel_simd` → `katgpt-core`), Phase 11
> (5 new crates: `katgpt-band`, `katgpt-validator`, `katgpt-sparse`,
> `katgpt-claim`, `katgpt-ruliology`). All moves preserve `katgpt_rs::*`
> import paths via root re-export shims.

```
crates/
  katgpt-core/    Shared types + SIMD kernels (multi-crate reuse):
    types.rs        Config (all presets + with_overrides + validate), Rng, HlaMode, AttentionMode (Causal/Bidirectional/BlockCausal/SpKv/SpKvQuant/DashAttn), ModelArchitecture (Generic/Gemma2/Llama/QwenDeltaNet), WeightDtype (F32/F16/BF16), InferenceOverrides, InferenceResult, DashAttnConfig, DeltaRoutingConfig, DeltaRoutingMode, ConvergenceSelector, LoopMode, HybridPattern, SdpaOutputGate, ResidualGate, PlanningDecision, ConfiguratorContext, DataGate, GateDecision, ProposerTask, TaskType, kv_dim, softmax, softmax_scaled, rmsnorm, rmsnorm_with_gamma, rmsnorm_with_gamma_eps, gegelu, gegelu_tanh, matmul, matmul_relu, sparse_matmul, sample_token, LoraAdapter, LoraPair, DomainLatent
    simd.rs         SimdLevel (Scalar/Neon/Avx2), simd_level(), simd_dot_f32, simd_dot_f16_f32, simd_fma_row, simd_outer_product_acc, simd_matvec, simd_matmul_rows, simd_matmul_rows_parallel, simd_matmul_relu_rows, simd_matmul_f16_f32_rows, simd_matmul_f16_f32_rows_parallel, simd_sparse_dot_f32, simd_sparse_matmul_rows, simd_scale_inplace, simd_fused_decay_write, simd_scale_mul_inplace, simd_exp_inplace, maxsim_score, maxsim_score_packed
    lib.rs          Feature gates: tiled_attention, coda_fusion, parallax_attn, leo_all_goals, dual_leo, questbench, tf_loop, plasma_path, peira_distill, dirichlet_energy, spectral_hierarchy, sigmoid_margin, dual_gram_pca, roofline_cost, domain_latent, sr2am_configurator, data_gate, sparse_mlp, modal_spec, mux_pruner, and_or_dtree, rim_slots, wall_attention, cgsp, action_bridge, sense_composition, slod, spectral_pruner, merkle_octree, flow_field_nav, dec_operators, gpart_adapter, dendritic_gate, qgf_oracle, qgf_projector, qgf_drafter, qgf_adaptive
    traits.rs       ConstraintPruner, DominoPruner, CompletionHorizon, ScreeningPruner, CollapseDetector, GameState, StateHeuristic, RolloutPolicy, LeoHead, AllGoalsUpdate, DualLeoMixer, AutocurriculumSampler, SpeculativeGenerator, GenerativeConstraintPruner, QGradientOracle, PartialScorer, ProblemMutator, BestBuddyAligner + NoPruner, NoScreeningPruner, BinaryScreeningPruner, RandomRolloutPolicy, ActionSpaceLog, GameTrace (Plan 107 Phase 0, consolidated from both crates)
    induced_cwm/    Induced CWM kernel primitive — InducedCwmKernel: GameState marker + CwmCommitment (BLAKE3) + BeliefInferenceFn<S> + TransitionUnitTest + ismcts_search_with_inference + ValueFnTournament + InducedCwmSlot (Plan 296, Research 275, arxiv 2510.04542) ⎗
    mop/            MOP value-iteration primitive — reward-free optimal policy: Eq. 7 fixed point in log-space LSE form over frozen tabular kernels, absorbing-state pinning, persistent stochastic π\* (Plan 573, behind "mop_path_entropy" feature, implies cgsp for entropy_nats)
    attention.rs    Tiled online-softmax flash attention for CPU SIMD (Plan 115, behind "tiled_attention" feature)
    coda.rs         CODA fused SIMD kernels: simd_matmul_rmsnorm_swiglu, simd_matmul_residual, simd_matmul_rmsnorm_rope, simd_matmul_rmsnorm_activation, GateActivation (Plan 103, behind "coda_fusion" feature)
    peira.rs        PEIRA inter-view regressor alignment — EMA cross-view/within-view covariance, closed-form predictor (Plan 153, behind "peira_distill" feature) ⚛
    dirichlet.rs    Dirichlet Energy structural alignment diagnostic — E(E) = Σ A_ij ‖h_i − h_j‖² (Research 111, behind "dirichlet_energy" feature)
    spectral_hierarchy.rs  Spectral hierarchy diagnostic — eigenspace alignment, Haar wavelets, Cauchy interlacing (Plan 156, behind "spectral_hierarchy" feature) ⊕
    questbench.rs   QuestBench underspecification scoring — normalized entropy from ScreeningPruner relevance (Plan 110)
    roofline.rs     Roofline cost model — GPU operator runtime prediction via calibrated peak throughput (Plan 159, behind "roofline_cost" feature) ⊏
    parallax_attn.rs Parallax parameterized local linear attention — streaming covariance correction (Plan 135, behind "parallax_attn" feature) ⊔
    linoss.rs        LinOSS oscillatory state-space cell + ModalSpec drafter — Fourier modal speculative drafting (behind "modal_spec" feature)
    sense/             KG Latent Octree Sense Composition — NPC sense modules with ternary bit-plane projection (Plan 221, behind "sense_composition" feature)
      brain.rs          NpcBrain composition + GM override + HLA projection
      octree.rs         SenseOctreeBuilder — KG→bit-plane octree builder
      gm.rs             GM action dispatch API (pin_sense, disable_autonomous, inject_kg)
      hotswap.rs        SenseHotSwap — lock-free AtomicPtr module replacement
      bandit.rs         SenseTrialLog — bandit trial log + decay direction EMA
      batch.rs          SenseBatch — parallel batch projection (rayon when N>64)
      serialize.rs      SNSE binary format with BLAKE3 verification
      bake.rs           BAKE precision-gated Bayesian embedding update (behind "bake_precision" feature)
      schema_centroid.rs  SchemaCentroidCache — per-class centroid init (behind "schema_centroid" feature)
    shard_embedding.rs  🪦 DEPRECATED (Issue 139): JL random orthogonal projection [f32;64]→[f32;8] — violates JL bound at m=8, marked #[deprecated] (Plan 230)
    slod.rs             SLoD Spectral Level-of-Detail Pruner — Poincaré ball + heat diffusion + tier routing (Plan 235, behind "slod" feature)
    and_or/          AND-OR tree module — AndOrNode<G,S> generic AND-OR tree for hierarchical goal decomposition (behind "and_or_dtree" feature)
      mod.rs        Module root, re-exports AndOrNode
      types.rs      AndOrNode enum (Or/And/Leaf), is_solved, push_child, set_best, set_solution
    mux/             MUX superposition tree search — superposition DD-tree with BFS frontier (behind "mux_pruner" feature)
      mod.rs        Module root — mux_pruner, mux_ddtree, mux_bfs, mux_demux, mux_bandit_width sub-features
      span_pruner.rs  MuxSpanPruner — superposition span validation
      top_k.rs      extract_top_k_peaks — top-K peak extraction from logit distributions
      dd_tree.rs    MuxDdTree, MuxNode — superposition DD-tree with hypothesis coverage
      bfs.rs        MuxBfs — dynamic-width BFS frontier expansion
      demux.rs      mux_demux — deterministic superposition recovery verifier
      bandit_width.rs  MuxBanditWidth — UCB1 arm selector for tree width
      freeze_thaw.rs   MuxTarget, MuxPatternStore — freeze/thaw for superposition patterns

src/
  lib.rs            Module index + debug tracking allocator
  main.rs           Entry point (proof → bench → Percepta bench → plot)
  types.rs          Re-exports katgpt_core::types::* (including DashAttnConfig, DeltaRoutingConfig, ConvergenceSelector, LoopMode, HybridPattern, SdpaOutputGate, ResidualGate, PlanningDecision, ConfiguratorContext, DataGate, GateDecision, ProposerTask, TaskType) + QuantizedKVCache trait (interface for TurboQuant/SpectralQuant KV caches)
  simd.rs          SimdLevel (Scalar/Neon/Avx2), simd_level(), simd_dot_f32, simd_fma_row, simd_outer_product_acc, simd_matvec, simd_matmul_rows, simd_matmul_relu_rows, simd_sparse_dot_f32, simd_sparse_matmul_rows, simd_scale_inplace (Plan 060)
  transformer.rs    TransformerWeights (+ mtp projections), LayerWeights, KVCache, MultiLayerKVCache, KVSnapshot, PagedKVCache, RavenKVCache, ForwardContext (+ sparse buffers + lora_buf + mtp_context_buf + tq_dequant_pos), PrefillContext, DecodeStage, forward, forward_with_domain_latent, forward_prefill, forward_paged, forward_raven, forward_turboquant, forward_looped, forward_coda, forward_decode_stage, depth_route_weights, generate, generate_into, generate_batch, generate_with_prefill, tokens_to_string, project_target_activation, cluster_map_round_robin, cluster_map_from_embeddings, raven_compute_router, raven_update, raven_readout, preload_kv_cache
  weights.rs        ContiguousWeights — single-buffer 64-byte aligned weight layout (Plan 102)
  feedback.rs       FeedbackConfig, send_feedback ⌁
  percepta/         Percepta 2D Convex Hull Attention + Computation Graph:
    mod.rs          Module declarations, re-exports (15+ submodules)
    types.rs        TieBreak, HullMeta, Vec2 (f64), constants (HARD_K, BIG, EPS)
    legacy.rs       Vec2 (f32), KVCache2D (Graham Scan), Sudoku9x9, SymbolicValidator, StreamingSolver, SolveEvent
    cht.rs          Line, CHT — dynamic convex hull trick / line container
    hull.rs         AttentionResult, HullHalf, HardAttentionHead (dual-hull O(log N)), BruteAttentionHead
    encoding.rs     encode_key, encode_query, clear_key, hard_scale, hard_scale_query
    cumsum.rs       CumSum — cumulative sum via uniform attention
    standard_cache.rs  StandardCache — O(n) softmax attention KV cache
    gates.rs        reglu, stepglu, multiply — gate primitives; PersistSlot, GateKind
    graph/          Computation Graph DSL:
      mod.rs        Module root, re-exports
      types.rs      Expression (sparse linear combo), DimensionKind, Dimension, LookUp, ProgramGraph, GraphBuilder, ValidationError
    weights.rs      TransformerWeights, LayerWeights, AttentionWeights, FfnWeights, HeadInfo, build_weights
    transformer.rs  TransformerConfig, TransformerVocab, GenerationResult, VanillaTransformer
    evaluator.rs    GraphEvaluator — step/predict/evaluate/compare_with_reference
    specialize.rs   SpecializationError, SpecializationReduction, SpecializedModel, UniversalModel
    scheduler.rs    OpKey, Phase, StdLayer, DepGraph, Schedule, build_dep_graph, milp_schedule
    runner.rs       RunnerError, BuildResult, Runner — compile/build/run/evaluate/specialize/full_pipeline
    compile.rs      compile_program, CompiledProgram — C source → WASM → lowered bytecode → token prefix (behind "percepta_compile")
    wasm/           WASM MVP decoder + lowering + interpreter (Futamura projection):
      mod.rs        Module root
      decoder.rs    WasmModule, FuncType, FuncBody, WasmInstr, decode
      lower.rs      lower_hard_ops, check_basic_only
      interpreter/  WASM interpreter as computation graph:
        mod.rs      Module root
        arithmetic.rs  Arithmetic ops dispatch
        dispatch.rs    Instruction dispatch table
        tokens.rs      Token mapping
  tf_loop.rs        Training-Free Loop — ODE-motivated damped sub-stepping inference-time refinement (Plan 136) ⊛---
  newton_schulz.rs  Newton-Schulz orthogonalization + Muon momentum — 5-iteration cubic fixed-point (Plan 152) ☊
  river_valley.rs   River-valley diagnostic metrics — subspace ratios, effective rank, update cosine similarity (Plan 152) ☊
  ega_attn.rs       Energy-Gated Attention — spectral salience gating with z-normalized sigmoid gate (Plan 139) ⍰
  shard_kv/         ShardKV asymmetric K/V compression (Plan 147) ⎘:
    mod.rs          Module root (re-exports)
    types.rs        ShardKV layer + config types
    rope.rs         RoPE undo for PCA rotation path
    kv_cache.rs     ShardKV KV cache impl (K: PCA+water-fill, V: Hadamard+K-means)
  sleep/            Sleep Consolidation — offline recursive memory consolidation at eviction (Plan 154) ☽:
    mod.rs          Module root, re-exports
    types.rs        SleepConfig, SleepLayer, SleepSnapshot
    consolidation.rs N-pass offline recurrent consolidation into GDN2 fast weights
    eviction.rs     KV cache eviction + consolidation pipeline
  distill/          PEIRA distillation (Plan 153) ⚛:
    mod.rs          Module root (behind "peira_distill" feature)
    peira.rs        PEIRA inter-view regressor alignment — collapse-free modelless distillation
    ilc.rs           ILC (Iterative Latent Clustering) Distillation — synonym-aware DDTree pruning (behind "ilc_distill" feature) ⚛+
  benchmark.rs      BenchCategory, BenchResult, run_all, run_all_parallel, save_results_csv, append_timeseries_csv, generate_batch, bench_hla_vs_flat_cache, bench_hla_memory, bench_hla_quality, bench_simd, bench_sparse_mlp
  plot.rs           plot_results → PNG, plot_timeseries
  rerank.rs         RerankMethod (Cosine/MaxSim), RerankedDoc, ndcg_at, SymmetricBoundaryPair (behind "maxsim" + "bt_rank" features)

  speculative/      SOLID decomposition:
    mod.rs          Re-exports
    types.rs        TreeNode, DraftResult, ConstraintPruner trait, ScreeningPruner trait, NoPruner, NoScreeningPruner, BinaryScreeningPruner, SpeculativeContext, DDTreeBranchCache, RejectionReason, DraftEvent, PrefillMode, FlashPrefillConfig, BlockScores
    sampling.rs     sample_from_distribution, sample_residual_distribution, sample_residual_distribution_into
    dd_tree.rs      build_dd_tree, build_dd_tree_pruned, build_dd_tree_screened, build_dd_tree_screened_with_schedule (thinking_prune), build_dd_tree_balanced, TreeBuilder, extract_parent_tokens, extract_parent_tokens_into, extract_best_path, extract_best_path_into, build_inference_result, merge_retrieved_branches
    dflash.rs       dflash_predict, dflash_predict_with, dflash_predict_ar, dflash_predict_ar_with, dflash_predict_conditioned, dflash_predict_conditioned_with, dflash_predict_parallel
    verifier.rs     SpeculativeVerifier trait, SimulatedVerifier, LeviathanVerifier
    step.rs         speculative_step, speculative_step_verifier, speculative_step_rollback, speculative_step_rollback_with, speculative_step_conditioned, speculative_step_conditioned_with, speculative_step_rollback_paged
    prefill.rs      PrefillScorer trait, AttentionScorer, BlockAttentionScorer, compress_prompt, compress_prompt_blocks, block_select, block_select_grid, should_compress, speculative_prefill, speculative_prefill_block, speculative_prefill_adaptive
    flow_pruner.rs  FlowPruner<P> — GFlowNet-inspired stop-probability regularization ♭
    d2f_verifier.rs D2fDrafterVerifier — D2F diffusion drafts, AR verifies (Tri-Mode, Plan 089) ⓘ
    d2f.rs          D2fBlockState, D2fDecodeConfig, D2fBlockResult, D2fPipelineBlock, D2fPipeline, D2fPipelineResult, d2f_decode_block* (behind "dllm" feature)
    alpha.rs        AlphaTarget, alpha_intersect, is_consistent — LDT α-intersection pruning + conflict detection (behind "lattice_deduction" feature, Plan 088) ⎌
    ppot/           PPoT (Plans 026 + 027) ○
      mod.rs        Module root, public API re-exports
      types.rs      TokenRule enum, PpotConfig
      entropy.rs    token_entropy, identify_high_entropy_positions, identify_positions_by_rule, identify_positions_adaptive
      resample.rs   ppot_resample, ppot_resample_with_support, ppot_resample_different_value, ppot_resample_multi_strategy, ppot_rescue, ppot_rescue_adaptive, ppot_rescue_reviewed
      knowledge.rs  RejectionInsight, ErrorKind, SessionKnowledge
      rank.rs       rank_by_consistency, rank_by_consistency_weighted, select_best_variant, select_best_variant_weighted
      flashar_anchor.rs  FlashAR Strided Anchor-Then-Fill D2F Decoding (Plan 166 T11, behind "flashar_anchor" feature) ⚓
      flashar_consensus.rs  FlashAR Consensus Tri-Mode with Ternary Thermal Paths (Plan 166, behind "flashar_consensus" feature) ⚖
      budget.rs        Compression-adaptive decode budget (Plan 167, behind "budget_adaptation" feature) 💰
      budget_compat.rs  Budget adaptation integration helpers (Plan 167 Phase 2)

  pruners/          Domain-specific constraint pruners:
    mod.rs          Re-exports
    pathfinder.rs   Target, find_path, find_distance, reachable_positions, enumerate_targets, terrain_cost, manhattan
    tactical_pruner.rs  GameState, TacticalPruner (grid-based tactical puzzle)
    dungeon_pruner.rs   FloorGrid, StairConnection, DungeonMap, DungeonState, DungeonPruner (multi-floor)
    dungeon_pathfinder.rs  DungeonAction, MultiFloorTarget, find_path_on_floor, find_path_multifloor, enumerate_multifloor_targets
    map_generator.rs  GeneratedMap, GeneratedDungeon, MapGenerator (procedural generation)
    sudoku_pruner.rs  SudokuPruner *
    bandit.rs       BanditStrategy, BanditStats, BanditPruner<P>, BanditSession, BanditEvent, BanditResult, BanditEnv trait, BernoulliEnv, GaussianEnv, SharedBanditStats ♭
    trial_log.rs    TrialRecord, TrialSummary, TrialLog ♭
    absorb_compress.rs  CompressConfig, AbsorbCompress trait, AbsorbCompressLayer<P> ♭
    hot_swap.rs     HotSwapPruner<P> — blake3 hash comparison reload ♭
    regression.rs   GoldenTrace, RegressionFailure, RegressionResult, RegressionSuite, ReplayReward trait ♭
    review_metrics.rs  ReviewSummary, ReviewMetrics, ReviewStrategy, EntropyAnomalySummary ♭
    stepcode.rs     PathStep, ShapedPath, shape_path, path_consistency ≋
    variance_minimizer.rs  VarianceMinimizer, VarianceMinimizerConfig (Plan 078) ☀
    freeze.rs       save_frozen, load_frozen — shared freeze/thaw disk I/O for repr(C) bandit knowledge structs (Plan 092)
    game_state/     GameState forward model trait + generic MCTS (Plan 056) ⎗
      mcts_search   mcts_search — Monte Carlo Tree Search
                    StateHeuristic trait, ActionSpaceLog
    bomber/         Bomberman 4-player HL arena (bevy_ecs) ⍟
      mod.rs        BomberAction, PowerUpKind, Cell, ECS components/resources, GameEvent
      arena.rs      ArenaGrid — 13×13 grid generation + presets
      players.rs    BomberPlayer trait, RandomPlayer, GreedyPlayer, ValidatorPlayer, HLPlayer, LoraPlayer, LoraWasmPlayer, NNPlayer
      g_zero_player.rs  GZeroPlayer — G-Zero self-play with template proposer + delta bandit
      tft_player.rs  TftPlayer — Tit-for-Tat with provocation detection
      rubric_player.rs  RubricPlayer — rubric-vector reward (Plan 071 T9)
      sdar_player.rs  SdarBomberPlayer — SDAR sigmoid-gated reward (Plan 072)
      arena_runner.rs  BomberArenaConfig, run_bomber_game, run_bomber_matchup (Plan 076)
      replay.rs     ReplaySample, ReplayWriter — JSONL replay persistence
      replay_backward.rs  BackwardSample, ReplayBackwardWalker — GFlowNet backward policy
      systems.rs    init_world, spawn_players, run_tick
      wasm_pruner.rs  BomberWasmPruner — WASM batch validation
      wasm_state.rs  serialize_game_state, ZeroCopyStateBuffer
    monopoly/       Monopoly board game engine (bevy_ecs) ✦
      mod.rs        PropertyGroup, SquareKind, TurnPhase, GameEvent (30+ variants), Player, Property, Board, etc.
      board.rs      build_board, shuffle_decks, group_squares
      players.rs    MonopolyPlayer trait, RandomPlayer, GreedyPlayer, ValidatorPlayer, HLPlayer, DecisionContext, Strategy
      systems.rs    init_world, spawn_players, execute_turn, run_game, calculate_rent, transfer_assets
    fft/            FFT Tactics Arena — ATB battle engine ✧
      mod.rs        Module root, re-exports
      types.rs      Class (6), Team, ActionType (9), Stats, Pos, Unit, Action, GameEvent
      battle.rs     BattleState, resolve_action, should_forgive
      status.rs     StatusEffect (9), ActiveEffect, apply_tick_effects, can_cast, can_act, ct_fill_rate
      players.rs    FftPlayer trait, GreedyFFTPlayer, ValidatorFFTPlayer, HLFFTPlayer
      g_zero_player.rs  GZeroFFTPlayer — G-Zero self-play for FFT
      rubric_player.rs  RubricFFTPlayer — rubric-vector reward (Plan 071 T10)
      sdar_player.rs  SdarFFTPlayer — SDAR sigmoid-gated reward (Plan 072)
      arena_runner.rs  FftArenaConfig, run_fft_battle, run_fft_matchup (Plan 076)
      tft_player.rs  TftFFTPlayer — Tit-for-Tat FFT player
    go/             Go GameState + AutoGo API bridge + tournament ⛩
      mod.rs        Module root, re-exports
      types.rs      GoAction (Place, Pass), GoCell (Empty, Black, White)
      state.rs      GoState — flat array board, simple ko, Tromp-Taylor scoring, GameState trait, GoHeuristic
      autogo_client.rs  AutoGoClient — REST API bridge to AutoGo play.py server
      replay.rs     GoReplay, MoveRecord — game recording + deterministic playback
      players.rs    GoPlayer trait, GoRandomPlayer, GoGreedyPlayer, GoValidatorPlayer, GoHLPlayer, GoGZeroPlayer, GoMctsPlayer
      tournament.rs GoTournamentConfig, GoTournamentResult, AutoGoProxyPlayer, run_tournament
      g_zero_player.rs  GoGZeroSelfPlay — HintDelta + absorb-compress self-play
      autoresearch.rs   AutoResearchLoop — UCB1 bandit over config arms, early stopping
      analytics.rs  cross-domain analysis, scaling laws, player tier comparison
    delta_mem/      δ-Mem modelless distillation — associative bandit memory ⌘
      mod.rs        Module root, re-exports
      state.rs      DeltaMemoryConfig, DeltaMemoryState, DeltaMemorySnapshot
      hash.rs       FeatureHasher, ContextFeatures, OutcomeFeatures
      pruner.rs     CorrectionMode, WriteGranularity, MemorySteeredPruner<P>
      multi.rs      AggregationStrategy, MultiDomainMemory
      multi_pruner.rs  MultiDomainMemoryPruner<P>
    g_zero/         G-Zero self-play distillation — verifier-free self-evolution ǂ
      mod.rs        Module root, re-exports
      types.rs      HintDelta, LogProbResult
      template_proposer.rs  QueryTemplate, GeneratedPair, TemplateProposer
      bomber_templates.rs  BomberTemplate (8 strategies), BomberTemplateProposer
      delta_bandit.rs  DeltaBanditPruner<P>
      delta_absorb.rs  DeltaGatedConfig, DeltaGatedAbsorbCompress<P>
      fft_templates.rs  FFTTemplate (10 strategies), FFTTemplateProposer

    dreamer/        Auto-Dreamer offline memory consolidation (Plan 107, behind "dreamer" feature) ∞:
      mod.rs          Module root, re-exports
      types.rs        DreamerConfig, CadenceSchedule, QCluster
      scheduler.rs    cadence scheduler — when to consolidate
      consolidator.rs offline Q-value consolidation pass
      pipeline.rs     DreamerPipeline — full consolidation pipeline
      counterfactual.rs  counterfactual replay generation
      decay.rs        exponential decay for stale memories
      frozen.rs       frozen memory snapshot I/O
    subterranean/   Procedure graph compilation — compiling workflows into weights (Plan 110, behind "subterranean" feature) ≬:
      mod.rs          Module root, re-exports
      types.rs        ProcedureGraph, ProcedureNode, CompiledProcedure
      cost_model.rs   procedure cost estimation
      path_enumerator.rs  enumerate procedure paths
      path_sampler.rs     sample procedure paths
      training_mode.rs    training mode dispatch
      bandit_bridge.rs    bridge to bandit infrastructure
      game_bridge.rs      bridge to game state trait
      bomber_procedure.rs Bomberman procedure definitions
      go_procedure.rs     Go procedure definitions

    arena/           Cross-arena tournament infrastructure (Plan 076):
      mod.rs        Module root + re-exports
      types.rs      ArenaKind, GameResult, MatchupResult, Ranking, Leaderboard, EloCalculator
      scheduler.rs  Matchup, round_robin_pairs, full_field_matchups
    ropd_rubric/     ROPD rubric modelless distillation (Plan 071):
      mod.rs           Module root + re-exports
      template.rs      RubricCriterion, RubricTemplate (bomber/fft/generic)
      types.rs         RubricVector (weighted_score, gap_vs_references)
      scorer.rs        RubricScorer trait, PatternScorer, score_with_references
      rubric_absorb.rs RubricGatedAbsorbCompress<P> (per-criterion gated absorb)
      rubric_bandit.rs RubricBanditPruner<P> (rubric-weighted reward bandit)

    sdar_gate.rs     SDAR sigmoid gate primitives (sdar_gate, sdar_modulate, sdar_gated_reward)
    bt_rank.rs       BtOutcome, BtComparison, BtConfig, BtScores, bt_fit, bt_fit_from_fn, bt_sigmoid — Bradley-Terry pairwise ranking ⊞
    cna.rs           CnaNeuron, CnaCircuit, CnaDiscoveryConfig, CnaModulator, CnaScreeningPruner, cna_discover, cna_modulate — Contrastive Neuron Attribution 🔬
    manifold_residual.rs  KlResidualScorer, L2ResidualScorer, ManifoldResidual, ResidualRelevanceScorer — Deep Manifold fixed-point scoring ∇
    boundary_alignment.rs  BoundaryAlignment trait, KlBoundaryAligner — federated KL coupling ≋
    tes_loop.rs      TesLoop trait, SimpleTesLoop<E>, TrajectoryPruner — SimpleTES RPUCG loop ⟳
    hydra_budget.rs  Hydra-Aware Adaptive Layer Budget (behind "hydra_budget" feature) 🐉
    gepa_reflective.rs  GEPA-D Reflective Config Evolution (behind "gepa_reflective" feature) 🪞
    phrase_boost.rs  PhraseBoost context trie phrase boosting (behind "phrase_boost" feature) 📝
    phrase_trie.rs   Compact token-level trie for phrase boosting (behind "phrase_boost" feature) 🌳

    sdar/            SDAR gated distillation — modelless (Plan 072):
      mod.rs           Module root + re-exports
      sdar_bandit.rs   SdarBanditPruner<P> (sigmoid-gated reward updates)
      sdar_absorb.rs   SdarGatedAbsorbCompress<P> (soft sigmoid promotion)

  tokenizer/        BPE tokenizer (encode/decode/train, Config::bpe())
    mod.rs          Re-exports: BpeTokenizerImpl, BpeTrainer, BpeTokenizer, MergeRule
    types.rs        BpeTokenizer, MergeRule
    bpe.rs          BpeTokenizerImpl (encode/decode), BpeTrainer (train)

  validator/        SynPruner + partial parser ‡
    mod.rs          Module root
    types.rs        PruneResult, ErrorKind, CompilerFeedback
    partial_parser.rs  PartialParser — bracket balance DFA (Tier 0)
    syn_pruner.rs   SynPruner — two-tier pruner (DFA + syn parse)

  turboquant/      TurboQuant KV cache compression — legacy baseline for bench/educate only (arXiv:2504.19874):
    mod.rs          Module root (re-exports)
    types.rs        TurboQuantCodebook, TurboQuantLayer, TurboQuantKVCacheConfig
    codebook.rs     Lloyd-Max codebook (compute_codebook, quantize, dequantize)
    rotation.rs     QR-based orthogonal rotation + QJL projection
    kv_cache.rs     TurboQuantKVCache (store_key, store_value, dequantize, bit-pack)
    forward.rs      attention_turboquant, dequantize_keys_flat/values_flat, cosine_similarity

  octopus/         OCTOPUS octahedral triplet KV compression — primary default (Plan 099) ⊛:
    mod.rs          Module root (re-exports)
    types.rs        OctopusConfig, OctopusLayer, OctopusCodebook, TripletIndices
    octahedral.rs   oct_encode, oct_decode — S² ↔ [-1,1]² equal-area parameterization
    triplet.rs      Triplet, decompose, recompose, recompose_into — 3-block grouping
    codebook.rs     ScalarCodebook, build_norm_codebook, build_oct_codebook — Lloyd-Max codebooks
    encode.rs       encode_triplet, joint_3x3_round, bit-pack/unpack — triplet encoder
    kv_cache.rs     OctopusKVCache — QuantizedKVCache trait impl
    forward.rs      maxsim_score_octopus, dequantize-to-flat — score-path decode (behind "maxsim" feature)

  hybrid_oct_pq/   Hybrid OCT triplet + PlanarQuant rotation — default KV codec (Plan 101) ⊛+:
    mod.rs          Module root (re-exports)
    types.rs        HybridOctPqConfig, HybridOctPqLayer
    kv_cache.rs     HybridOctPqKVCache — QuantizedKVCache trait impl
  planar_quant/    PlanarQuant 2D Givens rotation KV cache (Plan 100, behind "planar_quant" feature) ⊕:
    mod.rs          Module root (re-exports)
    types.rs        PlanarQuantConfig, PlanarQuantLayer
    rotation.rs     2D Givens rotation — O(d) vs TQ O(d²)
    kv_cache.rs     PlanarQuantKVCache — QuantizedKVCache trait impl
  iso_quant/       IsoQuant 4D quaternion rotation KV cache (Plan 100, behind "iso_quant" feature) ⊕+:
    mod.rs          Module root (re-exports)
    types.rs        IsoQuantConfig, IsoQuantLayer
    rotation.rs     4D quaternion rotation — O(d) vs TQ O(d²)
    kv_cache.rs     IsoQuantKVCache — QuantizedKVCache trait impl

  spectralquant/   SpectralQuant calibrated KV compression — secondary, per-dimension water-fill (Plan 077) ⊛:
    mod.rs          Module root (re-exports)
    types.rs        LloydMaxCodebook, SpectralQuantCalibration, WaterfillAllocation, SpectralQuantLayer, SpectralQuantKVCacheConfig
    spectral.rs     calibrate_eigenbasis, waterfill_bits, participation_ratio, spectral_gap, LloydMaxQuantizer
    nonuniform_quant.rs  NonUniformQuantizer, CompressedVector — Lloyd-Max scalar quantizer
    spectral_rotation.rs  SpectralRotation — eigenbasis rotation, RandomRotation (turboquant compat)
    spectral_kv_cache.rs  SpectralQuantKVCache, DequantizeScratch — full quantized KV cache implementation
    forward.rs      attention_spectralquant, dequantize_spectral_keys_flat/values_flat, par_maxsim_score_spectralquant (behind "maxsim" feature)

  dllm.rs          NoiseSchedule, D2fContext, DenoiseConstraint trait, corrupt_block, forward_bidirectional_positions, forward_block_causal_positions, denoise_loop, denoising_accuracy ⌂
  dash_attn/       DashAttention adaptive sparse hierarchical attention (Plan 106, behind "dash_attn" feature) ∹
    mod.rs          Module root, re-exports
    entmax.rs       α-entmax sparse attention activation
    routing.rs      chunk-level routing + importance scoring
    chunk_summary.rs  chunk summary statistics
    forward.rs      forward_dash_attn, forward_dash_attn_with_config
    tests.rs        unit tests
  gdn2/            Gated DeltaNet-2 recurrent attention (Plan 105, behind "gdn2_attention" feature) ◉
    mod.rs          Module root, re-exports
    types.rs        Gdn2Config, Gdn2State, Gdn2Gate
    kernel.rs       simd_fused_decay_write-based recurrent update
    forward.rs      forward_gdn2, forward_gdn2_with_state
  hla/             Higher-order Linear Attention — O(1) inference cache (Plan 057, SIMD Plan 060) ⎔
    mod.rs          Module root
    types.rs        HlaQHeadState, HlaLayerState, MultiLayerHlaCache, AhlaQHeadState, AhlaLayerState, MultiLayerAhlaCache, HlaVariant
    kernel.rs       hla_state_update, hla_readout, hla_denom, ahla_step, ahla_denom, hla_layer_update, hla_layer_readout, ahla_layer_step
    forward.rs      forward_hla, forward_ahla, generate_hla_into, generate_ahla_into
  sp_kv/           Self-Pruned Key-Value Attention (Plan 070) §
    mod.rs          Module root
    types.rs        SpKvGateMode, SpKvConfig, SpKvLayerCache, SpKvCache, UtilityPredictorWeights, SpKvPredictors, GateBiasBuffer
    utility_predictor.rs  predict, predict_single_head, soft_gate_bias, hard_gate_bias, tahg_gate_bias, UtilityAggregation
    forward.rs      SpKvForwardContext, BiasProvider trait, attention_head_core, attention_head_gated, forward_sp_kv

  unit_distance/    Unit Distance GOAT proof — number-theoretic lattice constructions (Plan 090, behind "unit_distance" feature) 📏:
    mod.rs          Module root, re-exports
    types.rs        LatticePoint, DistanceProof
    cm_field.rs     CM-field constructions
    minkowski.rs    Minkowski bound computations
    pigeonhole.rs   Pigeonhole principle proofs

  data_probe/      Data Probe Diagnostics — information-theoretic validation (Plan 141, behind "data_probe" feature) 🔍:
    mod.rs          Module root
    markov.rs       Dirichlet-sampled Markov chain generator
    nll.rs          NLL computation against known chain
    typical_set.rs  Three-way regime classification
    dirichlet_energy.rs  Dirichlet Energy structural alignment diagnostic
    claim.rs        Claim card infrastructure for C1-C4 validation
    geometry.rs     Representation geometry diagnostics (Plan 151)
  skill_opt/       SkillOpt text-space skill optimization (Plan 144, behind "skill_opt" feature) ✎:
    mod.rs          Module root
    edit.rs         Edit operations and SkillEdit struct
    apply.rs        Deterministic text patching engine
    gate.rs         Validation gate
    schedule.rs     Edit budget schedules
    buffer.rs       FIFO ring buffer for rejected edits
    optimizer.rs    SkillOptimizer trait
  proof_cert/      Hierarchical GOAT Proof Certificates (Plan 145, behind "proof_cert" feature) 🏆:
    mod.rs          Module root
    certificate.rs  Certificate types (ProofCertificate, ProofEvidence, ProofProperty, ProofResult)
    chain.rs        Certificate chain verification
    macros.rs       Declarative proof macros
    serde_impls.rs  Serde serialization + checksum
    wasm_certificates.rs  WASM certificate generation
  cache_prune/     CachePrune SAT + rolling hash + sensitivity (Plan 140, behind "cache_prune" feature) ✂:
    mod.rs          Module root
    rolling_hash.rs Rolling hash for O(n) variable-length segment matching
    sat.rs          Summed-Area Table for O(1) rectangular attention queries
    sensitivity.rs  Generic SensitivityDetector trait

  alloc.rs          Debug-only TrackingAllocator, reset_alloc_stats, get_alloc_stats (debug builds)

  * behind --features sudoku
  ∘ behind --features sparse_mlp    (default)
  ○ behind --features ppot           (default)
  ‡ behind --features validator
  ♭ behind --features bandit         (default)
  ⍟ behind --features bomber         (bevy_ecs + bandit)
  ✦ behind --features monopoly       (bevy_ecs + bandit)
  ✧ behind --features fft            (bandit)
  ⛩ behind --features go             (bandit + reqwest)
  ⌘ behind --features delta_mem      (bandit)
  ǂ behind --features g_zero         (bandit)
  ⌁ behind --features feedback
  ⎔ behind --features hla_attention
  § behind --features sp_kv
  ⌂ behind --features dllm
  ≋ behind --features stepcode
  ⎗ behind --features game_state
  ⊛ behind --features spectral_quant  (default)
  ☀ behind --features replaid_schedules
  ⊞ behind --features bt_rank         (default)
  ⊘ behind --features sdar_gate
  ⊡ behind --features ropd_rubric     (bandit)
  ⚡ behind --features elf_sde         (default)
  🔬 behind --features cna_steering    (default)
  ∇ behind --features deep_manifold    (default)
  ≋ behind --features federation       (default)
  ⟳ behind --features tes_loop         (bandit)
  ⬡ behind --features maxsim
  ▣ behind --features percepta          (ordered-float)
  ▣+ behind --features percepta_gates   (percepta)
  ▣++ behind --features percepta_graph  (percepta_gates)
  ▣+++ behind --features percepta_wasm  (percepta_graph)
  ▣++++ behind --features percepta_compile (percepta_wasm + good_lp)
  ⎌ behind --features lattice_deduction
  ⊛+ behind --features hybrid_oct_pq (default)
  ⊕ behind --features planar_quant
  ⊕+ behind --features iso_quant
  ∹ behind --features dash_attn (default)
  ◎ behind --features mls_aggregate (default)
  ◉ behind --features gdn2_attention (default)
  ∞ behind --features dreamer (default)
  ↻ behind --features lt2_looped (default)
  ⊞+ behind --features dmax_spd (default)
  ERRQ behind --features eqr_convergence (default)
  ≬ behind --features subterranean (default)
  ⚙ behind --features sr2am_configurator (default)
  ⊇ behind --features data_gate (default)
  ◧ behind --features tiled_attention
  ⨍ behind --features coda_fusion
  📏 behind --features unit_distance
  📊 behind --features stability_metrics (default)
  ⎗+ behind --features decode_specialize
  ⓘ behind --features tri_mode (dllm)
  ⊛- behind --features plasma_path   (default)
  ⊛-- behind --features parallel_probe (default)
  ⊛--- behind --features tf_loop      (default)
  ☊ behind --features newton_schulz    (default)
  ☊ behind --features river_valley    (default)
  ⍰ behind --features ega_attn        (opt-in)
  ⎘ behind --features shard_kv        (opt-in)
  ☽ behind --features sleep_consolidation (default)
  ⚛ behind --features peira_distill   (default)
  ⚛+ behind --features ilc_distill
  ⊕ behind --features spectral_hierarchy (default)
  ⊏ behind --features roofline_cost    (default)
  ⊔ behind --features parallax_attn   (opt-in)
  ⚓ behind --features flashar_anchor    (dllm)
  ⚖ behind --features flashar_consensus (tri_mode, plasma_path)
  💰 behind --features budget_adaptation
  🐉 behind --features hydra_budget     (default)
  🪞 behind --features gepa_reflective  (bandit, memo_reflections, default)
  📝 behind --features phrase_boost     (default)
  Plans 137-145 modules are opt-in, see Feature Flags table
```

## Feature Flags

| Flag | Dependencies | Description |
|------|-------------|-------------|
| `sparse_mlp` | — | TwELL-inspired sparse MLP matmul (Plan 022) |
| `ppot` | — | PPoT logit-parameterized CPU resampling + adaptive rescue (Plans 026 + 027) |
| `domain_latent` | — | Free Transformer mid-layer domain conditioning (Plan 038) |
| `bandit` | — | Multi-armed bandit + HL infrastructure: TrialLog, AbsorbCompress, HotSwapPruner, RegressionSuite, ReviewMetrics (Plans 030–032) |
| `sudoku` | — | SudokuPruner constraint pruning + examples |
| `validator` | `syn`, `proc-macro2` | SynPruner + partial parser |
| `delta_mem` | `bandit` | δ-Mem modelless distillation — associative bandit memory (Plan 053) |
| `g_zero` | `bandit` | G-Zero self-play distillation — Hint-δ gated absorb + bandit (Plan 049) |
| `hla_attention` | — | HLA/AHLA streaming attention kernels (Plan 057, SIMD-accelerated in Plan 060) |
| `fft` | `bandit` | FFT Tactics Arena — ATB battle engine with status effects (Plan 053) |
| `bomber` | `bevy_ecs`, `bandit` | Bomberman HL arena (Plan 033) |
| `bomber-wasm` | `bomber`, `wasmtime`, `papaya` | WASM bomber validator loader + batch pool (Plans 034 + 037) |
| `monopoly` | `bevy_ecs`, `bandit` | Monopoly board game engine (Plan 035) |
| `feedback` | — | E2E feedback loop — sends inference results to REST endpoint (Plan 042) |
| `rest` | — | REST bridge test + merge stub (Plan 009, client lives in riir-ai/riir-rest) |
| `embedding_router` | — | Semantic embedding routing (Plan 024) |
| `game_domain` | `domain_latent` | Alias for domain_latent — game-specific Config presets (Plan 040) |
| `language_domain` | — | Language domain: BPE vocab, LLM models (Plan 040) |
| `gpu` | — | Placeholder — GPU training lives in riir-ai/riir-gpu |
| `go` | `bandit`, `reqwest` | Go GameState + AutoGo API bridge + tournament + G-Zero self-play + AutoResearch loop (Plan 065) |
| `sp_kv` | — | SP-KV self-pruned key-value attention (Plan 070) |
| `dllm` | — | D2F Discrete Diffusion Forcing — mini dLLM research (Plan 066) |
| `stepcode` | `bandit` | Path shaping + consistency scoring (Plan 054, infrastructure only, no perf gain) |
| `ropd_rubric` | `bandit` | ROPD rubric modelless distillation — multi-criteria reward vectors, per-criterion gap targeting (Plan 071) |
| `sdar_gate` | — | SDAR sigmoid-gated distillation — asymmetric trust for bandit updates + soft absorb promotion (Plan 072) |
| `bt_rank` | — | Bradley-Terry pairwise ranking for DDTree selection (OpenDeepThink distillation) |
| `spectral_quant` | — | SpectralQuant calibrated eigenbasis + water-fill bit allocation — secondary KV compression, useful for per-dimension water-fill (Plan 077, default-on) |
| `octopus` | — | OCTOPUS octahedral triplet codec — data-oblivious, primary KV compression: -22% to -49% MSE vs SQ, zero calibration (Bench 022, Plan 099, default-on) |
| `turboquant` | — | TurboQuant rotation + uniform codebook — legacy baseline for bench/educate only (Plan 063) |
| `replaid_schedules` | — | RePlaid variance-minimized adaptive schedules — experimental, off by default (Plan 078) |
| `elf_sde` | — | ELF SDE noise injection + logit-normal schedule — 10-22× path diversity (Plan 079, default-on) |
| `cna_steering` | `bandit` | CNA contrastive neuron attribution — sparse circuit discovery + runtime modulation (Plan 087, default-on, GOAT proved) |
| `deep_manifold` | — | Deep Manifold L2/KL residual fixed-point scoring — ResidualRelevanceScorer (Plan 085, default-on, GOAT 6/6) |
| `federation` | `bandit` | Deep Manifold federated KL boundary alignment — KlBoundaryAligner, no data exchange (Plan 085, default-on, GOAT 6/6) |
| `tes_loop` | `bandit` | SimpleTES RPUCG loop — trajectory credit, TrajectoryPruner (Plan 086) |
| `maxsim` | — | MaxSim late-interaction scoring — Σ max_j dot, SIMD-accelerated (Plan 080) |
| `bomber-agent` | `bomber` | Coding agent validator loop (Issue 052) |
| `game_state` | `bomber` | GameState forward model trait + generic MCTS (Plan 056) |
| `bandit_mcts` | `game_state` | Bandit-guided MCTS rollout policy — NFSP/MCTS duality (Plan 067) |
| `percepta` | `ordered-float` | CHT hull cache: upper+lower, HullMeta, tie-break, cumsum |
| `percepta_gates` | `percepta` | + ReGLU, stepglu, multiply, persist primitives |
| `percepta_graph` | `percepta_gates` | + Expression/Dimension DSL, ProgramGraph |
| `percepta_wasm` | `percepta_graph` | + WASM decoder + lowering + interpreter (pure Rust) |
| `percepta_compile` | `percepta_wasm`, `good_lp` | + MILP scheduling + weights + transformer + Futamura |
| `lattice_deduction` | — | LDT Lattice Deduction Transformer — α-intersection pruning, conflict detection, asymmetric elimination (Plan 088, default-on, GOAT 7/7) |
| `delta_routing` | — | Delta Block cross-layer routing — residual block importance routing (Plan 097, default-on, GOAT 6/6) |
| `hybrid_oct_pq` | `planar_quant`, `octopus` | Default KV codec — OCT triplet + PQ 2D Givens rotation (Plan 101, default-on) |
| `planar_quant` | `turboquant` | PlanarQuant 2D Givens rotation KV cache — O(d) vs TQ O(d²) (Plan 100) |
| `iso_quant` | `turboquant` | IsoQuant 4D quaternion rotation KV cache — O(d) vs TQ O(d²) (Plan 100) |
| `dash_attn` | — | DashAttention adaptive sparse hierarchical attention via α-entmax routing (Plan 106, default-on, GOAT 9/9) |
| `mls_aggregate` | — | MLS Multi-Layer Sum aggregation of last K layer residuals (Plan 104, default-on, GOAT 6/6) |
| `gdn2_attention` | — | GDN2 Gated DeltaNet-2 recurrent attention — O(1) decode (Plan 105, default-on, GOAT 14/14) |
| `dreamer` | `bandit` | Auto-Dreamer offline memory consolidation with cadence scheduler + Q-value clustering (Plan 107, default-on, GOAT 8/8) |
| `lt2_looped` | `hla_attention` | LT2 looped inference — weight-shared T-pass loop with hybrid SDPA+AHLA dispatch (Plan 108, default-on, GOAT 8/8) |
| `dmax_spd` | `dllm` | DMax Soft Parallel Decode — hybrid token/mask embeddings with contiguous prefix promotion (Plan 109, default-on, GOAT 7/7) |
| `eqr_convergence` | `elf_sde` | EqR convergence-based rollout selection — Top1Converged picks smallest marginal-change residual (Plan 119, default-on) |
| `subterranean` | `bandit` | Procedure graph compilation — user-defined token-rewriting procedures compiled to zero-cost native code (Plan 110, default-on) |
| `sr2am_configurator` | `bandit`, `g_zero` | SR²AM Configurator Bandit — learned per-turn planning regulation via UCB1 (Plan 112, default-on) |
| `data_gate` | `bandit` | Task-level data gating for self-play training stability (Plan 111, default-on) |
| `tiled_attention` | — | Tiled online-softmax flash attention for CPU SIMD (Plan 115) |
| `parallax_attn` | `tiled_attention`, `newton_schulz`, `katgpt-core/parallax_attn` | Parallax parameterized local linear attention — streaming covariance correction (Plan 135, opt-in) |
| `coda_fusion` | — | CODA fused SIMD kernels — matmul+residual+rmsnorm+activation in single-pass (Plan 103) |
| `moa_inference` | `coda_fusion`, `katgpt-core/moa_inference` | MoA Mixture of Activations — token-adaptive activation mixing over 7-activation dictionary (Plan 158, default-on, GOAT) |
| `stability_metrics` | — | Per-step execution stability instrumentation — P50/P99/CV/stability_score (Plan 102, default-on) |
| `decode_specialize` | — | Stage-specialized decode paths for speculative decoding (Plan 102) |
| `tri_mode` | `dllm` | Tri-Mode inference — AR + Diffusion + Self-Speculation, D2F Drafter Verifier (Plan 089) |
| `unit_distance` | — | Unit Distance GOAT proof — number-theoretic lattice constructions (Plan 090) |
| `plasma_path` | `katgpt-core/plasma_path` | Ternary SIMD matvec — bit-plane ternary weights for SIMD-accelerated matmul (Plan 117, default-on, GOAT 5/5) |
| `parallel_probe` | — | Parallel-Probe 2D — consensus-based parallel branch control for N parallel reasoning branches (Plan 133, default-on, GOAT 7/7) |
| `tf_loop` | `katgpt-core/tf_loop`, `lt2_looped` | Training-Free Loop — pure inference-time mid-stack looping with ODE-motivated damped sub-stepping (Plan 136, default-on, GOAT 4/4) |
| `safe_bandit` | `bandit` | PrudentBanker Safe-Phased Bandit — delay-calibrated safe exploration with bounded regret (Plan 137, opt-in) |
| `stiff_anomaly` | — | Stiff/Soft Subspace Anomaly Gate — eigenvalue decomposition anomaly detection (Plan 138, opt-in) |
| `ega_attn` | — | Energy-Gated Attention — spectral salience gating (Plan 139, opt-in) |
| `cache_prune` | — | CachePrune — SAT + rolling hash + sensitivity masking for KV cache pruning (Plan 140, opt-in) |
| `data_probe` | — | Data Probe Diagnostics — information-theoretic validation with Markov chain analysis (Plan 141, opt-in) |
| `state_source` | `bandit` | State-Source Modelless Distillation — state-visitation tracking + P-UCB selector (Plan 142, opt-in) |
| `skill_opt` | — | SkillOpt — text-space skill optimization framework (Plan 144, opt-in) |
| `proof_cert` | — | Hierarchical GOAT Proof Certificates — formal verification methodology with certificate chains (Plan 145, opt-in) |
| `nexus_elo` | `state_source`, `bandit` | Nexus Elo — Plackett-Luce + P-UCB + goal cache for DDTree/SR²AM (Plan 143, opt-in) |
| `mech_attribution` | `cna_steering`, `ropd_rubric`, `bandit` | Mechanistic Data Attribution — catalyst pattern detection + influence proxy (Plan 111, opt-in) |
| `event_log` | `bandit` | Event-sourced game traces with fork-and-diff (Plan 124, GOAT 22/22) |
| `epiplexity_scoring` | `bandit` | Epiplexity structural information scoring — prequential coding estimator (Plan 130, opt-in) |
| `leo_all_goals` | `katgpt-core/leo_all_goals` | LEO All-Goals Q-value trait framework — `LeoHead`, `AllGoalsUpdate`, `sigmoid_bounded_q` (Plan 155, default-on, SUPER GOAT) |
| `dual_leo` | `leo_all_goals`, `katgpt-core/dual_leo` | Dual LEO teacher-student mixing — `DualLeoMixer` + `AutocurriculumSampler` (Plan 155, default-on, SUPER GOAT) |
| `sigmoid_margin` | `katgpt-core/sigmoid_margin` | Sigmoid margin loss + retrieval margin diagnostic — SigLIP-style softplus, `dim_sufficiency_bound` (Plan 157, Research 123, default-on, GOAT 7/7) |
| `newton_schulz` | — | Newton-Schulz orthogonalization + Muon momentum — 5-iteration cubic fixed-point for optimizer weight matrices (Plan 152, default-on, GOAT 25/25) |
| `river_valley` | — | River-valley diagnostic metrics — subspace ratios, effective rank, update cosine similarity (Plan 152, default-on, GOAT 25/25) |
| `proof_sketch_evolution` | `bandit` | Proof Sketch Evolution — Elo-rated proof population + global goal cache for DDTree/SR²AM (Plan 128, Research 088, opt-in) |
| `datrie_vocab` | — | Double-array trie vocab lookup — zero-alloc trie for ToaST tokenizer (Research 137, opt-in, pending benchmark) |
| `kog_cpu_fusion` | — | Kog AI monokernel CPU fusion — RMSNorm gamma folding + QKV interleaving (Plan 160, Research 139, default-on, GOAT 3/3 Gemma 2 scale) |
| `flashar_anchor` | `dllm` | FlashAR strided anchor-then-fill D2F decoding (Plan 166 T11, opt-in) |
| `flashar_consensus` | `tri_mode`, `plasma_path` | FlashAR consensus tri-mode with ternary thermal paths (Plan 166). **DEMOTED from default-on** (Issue 136, 2026-07-12, removed, see git history): Plan 485 benchmark showed KL 2.9-6.5 (100× worse than Leviathan baseline 0.03). DSpark entropy-skip hybrid dominates on both axes. Opt-in. |
| `budget_adaptation` | — | Compression-adaptive decode budget (Plan 167, default-on) |
| `ilc_distill` | — | ILC iterative latent clustering distillation — synonym-aware DDTree pruning (Research 136 GOAT 6/6, default-on) |
| `hydra_budget` | — | Hydra-aware adaptive layer budget — emergent self-repair layer skipping (Plan 165, default-on) |
| `gepa_reflective` | `bandit` | GEPA-D reflective config evolution — Pareto bandit config evolution (Plan 164, default-on) |
| `phrase_boost` | — | PhraseBoost context trie phrase boosting for DDTree (Plan 164, default-on) |
| `shard_kv` | `spectral_quant`, `turboquant` | ShardKV asymmetric K/V compression — undo RoPE + PCA K path, Hadamard + K-means V path (Plan 147, opt-in) |
| `sleep_consolidation` | `lt2_looped`, `gdn2_attention` | Sleep Consolidation — offline recursive memory consolidation at KV eviction into GDN2 fast weights (Plan 154, default-on, GOAT 14/14) |
| `spectral_hierarchy` | `katgpt-core/spectral_hierarchy` | Spectral hierarchy diagnostic — eigenspace alignment, Haar wavelets, Cauchy interlacing for KG extraction validation (Plan 156, default-on, GOAT) |
| `dual_gram_pca` | `katgpt-core/dual_gram_pca` | Dual-Gram PCA routing for short-sequence calibration (Plan 159, default-on, GOAT) |
| `roofline_cost` | `katgpt-core/roofline_cost` | Roofline cost model for GPU operator runtime prediction — compute/memory/launch bottleneck estimation (Plan 159, default-on, GOAT) |
| `peira_distill` | `katgpt-core/peira_distill`, `bandit` | PEIRA inter-view regressor alignment — collapse-free modelless distillation via EMA covariance (Plan 153 GOAT 7/7, default-on) |
| `parallax_attn` | `tiled_attention`, `newton_schulz`, `katgpt-core/parallax_attn` | Parallax parameterized local linear attention — streaming covariance correction (Plan 135, opt-in) |
| `freq_bandit` | `bandit` | FreqBandit — oscillatory spectral bandit for cyclic pattern detection to adaptive speculative decode (Plan 189, default-on, GOAT 7/7 G189=GAIN) |
| `belief_drafter` | `katgpt-core/belief_drafter`, `papaya` | NextLat Belief-State Speculative Drafter — lightweight 3-layer residual MLP recursive hidden-state prediction for variable-length self-speculative decoding + LatentTransitionCache + BeliefRankPruner (Plan 217, default-on, GOAT 43 tests + 7 benchmarks) |
| `bfcf_lfu_shard` | `bfcf_tree`, `papaya` | BFCF × LFU × Sharding — region-level LFU cache with frequency-aware sharding, batch processing, NeuronShard compound keys, emotion-aware eviction, KG triple transitions (Plan 218, default-on, GOAT 44 tests + 10 benchmarks) |
| `caddtree_budget` | `spec_cost_model` | CaDDTree — Cost-Aware Adaptive DDTree Budget Selection (Plan 219, 7 GOAT tests, default-on) |
| `hardware_aware_scheduler` | — | Hardware-Aware Prefix Scheduler — multi-request verification budget allocator (Plan 339, Issue 003, DSpark §3.2.2). Global sort + greedy admission + non-anticipating early-stop (Appendix A correctness theorem). Opt-in until a real multi-request batch caller exercises the synthetic GOAT gate. |
| `manifold_pruner` | — | ManifoldPruner — ManifoldE point-to-manifold soft validity scoring + kernel-tricked relevance for ScreeningPruner (Plan 234, opt-in, GOAT G1 FAIL) |
| `sense_composition` | `katgpt-core/sense_composition` | KG Latent Octree NPC sense modules — ternary bit-plane projection, GM override, hot-swap, bandit feedback (Plan 221, opt-in) |
| `shard_embedding` | — | 🪦 **DEPRECATED (Issue 139)** — JL random orthogonal projection [f32;64]→[f32;8] for O(1) cosine similarity shard lookup (Plan 230). Violates JL lower bound 200× at m=8; marked `#[deprecated]`, zero runtime consumers. |
| `slod` | `katgpt-core/slod`, `spectral_hierarchy` | SLoD Spectral Level-of-Detail Pruner — Poincaré ball hyperbolic geometry + heat diffusion tier routing (Plan 235, default-on, GOAT G1–G6) |
| `schema_centroid` | `katgpt-core/schema_centroid`, `dep:papaya` | Schema Centroid — per-class embedding centroids for informed KG entity init (Plan 237, default-on, GOAT 7/7) |
| `bake_precision` | `katgpt-core/bake_precision`, `dep:papaya`, `sense_composition` | BAKE Precision-Gated Bayesian Embedding — per-dimension precision tracking, O(8) arithmetic (Plan 236, opt-in, GOAT 10/10 but marginal) |
| `nf_flow_score` | — | NFCoT FlowScore — modelless normalizing flow density scoring for speculative candidates (Plan 229, opt-in) |
| `nf_flow_gate` | `nf_flow_score` | NFCoT adaptive EMA acceptance criterion (Plan 229 T3, opt-in) |
| `nf_flow_budget` | `nf_flow_score` | NFCoT sigmoid-weighted speculative depth allocation (Plan 229 T4, opt-in) |
| `nf_flow` | `nf_flow_score`, `nf_flow_gate`, `nf_flow_budget` | NFCoT parent feature — enables score + gate + budget (Plan 229, opt-in) |
| `union_bound_confidence` | — | Union Bound Confidence — additive branch confidence via Boole's inequality (Plan 231, default-on, GOAT 6/6) |
| `pathway_tracker` | — | PathwayTracker — intrinsic pathway stability detection (Plan 231, default-on, GOAT 7/7) |
| `federation_composer` | — | FederationComposer — explicit pruning with residual early termination (Plan 231, default-on, GOAT 7/7) |
| `collapse_aware_thinking` | `selectivity_router`, `thinking_cot`, `bandit` | Collapse-aware adaptive thinking — runtime reasoning collapse detection + early exit (Plan 212, default-on) |
| `cgsp` | `bandit`, `collapse_aware_thinking`, `data_gate`, `breakeven_routing` | Curiosity-Guided Self-Play — modelless Solver/Conjecturer/Guide triad with collapse recovery + BLAKE3-committed personality snapshots (Plan 274, Research 240 — **opt-in**: GOAT gate run, G2/G3/G4/P2/P3/G6 pass; G1 is informational because CGSP is curiosity-driven not target-seeking — see `.benchmarks/274_cgsp_goat.md`) |
| `substrate_gate` | `katgpt-core/substrate_gate` | SubstrateGate — inference-time routing via substrate conditions (Plan 216, default-on) |
| `llmexec_guard` | — | Entropy-driven verification budgeting (default-on) |
| `outlier_guard` | — | Model-load-time outlier injection detection via KS D-statistic (default-on) |
| `segment_checkpoint` | — | Segment-level checkpoint/rollback for speculative decoding (default-on) |
| `trust_region_spec` | — | Trust-region speculative verification (default-on) |
| `precision_aware_draft` | — | Precision-aware draft selection (default-on) |
| `self_distilling_bandit` | — | Self-distilling bandit arms (default-on) |
| `static_cal_tables` | — | Pre-computed calibration tables for quantization (default-on) |
| `targeted_precision` | — | Targeted precision allocation for KV cache (default-on) |
| `egcs` | — | Expert-gated channel selection (default-on) |
| `nds_proxy` | — | NDS Proxy — normalized difference score proxy for routing (Plan 186, default-on) |
| `rat_plus_bridge` | `katgpt-core/rat_plus_bridge` | RAT+ Recurrence Bridge via GDN2 state for modelless dilated inference (Plan 225, opt-in) |
| `swir_switch_thinking` | `thinking_cot` | SwiR Switch-Thinking — explicit↔latent reasoning mode controller driven by entropy trends, asymmetric dwell windows + switch-count overthinking guard (Plan 275, Research 241, **DEFAULT-ON** since Plan 313 T6.2, 2026-06-27): G2 token-efficiency 1.32×/1.37×/1.43× at n=3/5/10 on Gemma 2 2B + MATH-500 (gate ≥1.3×, all pass) with tuned config (w_e_to_l=32, c_max=64). G1 accuracy blocked by model capability (Gemma 2 2B too small, not a SwiR design flaw); G3–G6 all pass (3.1ns/step, convex-hull 1000/1000, no-regression, kurtosis escape). Token efficiency is the primary value prop. |
| `micro_belief` | `katgpt-core/micro_belief` | MicroRecurrentBeliefState — per-entity recurrent state trait + attractor/leaky/latent-thought kernels + BLAKE3 snapshot + bridge (Plan 276, Research 242, **opt-in**: G1.1/G1.2/G1.3/G1.5 pass; G1.4 latency FAIL ~273ns; G2.1 coherence FAIL — attractor demoted to Gain, trait unification + LeakyIntegrator are the promotable outputs) |
| `sink_aware_attn` | `data_probe` | Sink-Aware Attention — dual-policy sigmoid gate (Plan 287 Phase 3, Research 258, arxiv 2606.08105). Implies data_probe for the classifier primitive. **Opt-in**: default stays Uniform pending G2/G3 GOAT gate. Different paper + mechanism than `depth_invariance` (target-side sink classification vs drafter-side magnitude accumulation). |
| `depth_invariance` | `katgpt-core/depth_invariance` | Depth-Invariance Diagnostic + MagnitudeRegularizedResidual — root-cause counterpart to BeliefRankPruner / GainCostLoopHalter / latent_functor/reestimation / micro_belief/coherence_bench (Plan 306, Research 286, arxiv 2605.09992). Detects DepthSpecificRefinement / Collapsed / DepthInvariant on flattened `&[f32]` state chains. **DEFAULT-ON** (Plan 306 T7.4, 2026-06-23): G1 (8 correctness tests) + G2 (paper finding reproduced on random-init BeliefDrafter) + G3 (negative control on AttractorKernel + positive control on unclamped leaky) + G4 (re-spec to absolute-latency at HLA scale, all PASS) + SIMD inner-loop landed. Zero runtime cost unless invoked. |
| `temporal_deriv` | `katgpt-core/temporal_deriv` | Temporal Derivative Kernel — dual fast/slow EMA surprise signal driving 4 consumers (HLA companion, δ-Mem write gate, collapse detector, derivative curiosity) via a unified α-pair (Plan 277, Research 243, arXiv:2606.08720, **default-on**, GOAT 4/4) |
| `drift_segment` | `katgpt-kv/drift_segment` | DriftSegmentStore — training-free drift-segmented multi-state memory: rising-edge drift boundaries open slots, adjacent-density merge enforces capacity-K; composes temporal_deriv × segment_checkpoint slots × hope_compactor pair-merge (Issue 652, Research 482, arXiv:2606.10650, **opt-in**: Bench 635 GOAT PASS — +46.09pp/+75.00pp needle recall vs fixed-LFU, 12 ns/token, 0 allocs; consumers landed → [`03_memory/drift_segment.md`](../03_memory/drift_segment.md)) |
| `mop_path_entropy` | `katgpt-core/mop_path_entropy` (implies `cgsp`) | MOP value-iteration primitive — reward-free optimal policy from a frozen tabular kernel: the paper's Eq. 7 fixed point in log-space LSE form, absorbing-state pinning (V=0 bit-exact), persistent stochastic π\*, β risk knob (Plan 573, Research 478, arXiv:2205.10316, **opt-in**: Bench 638 GOAT G1–G4 PASS — gridworld 663 µs/solve, 0 allocs; consumer landed: riir-ai `mop_runtime` — Plan 538 COMPLETE, Bench 680 G1–G4 + Bench 681 G8 civ arena ALL PASS) |
| `hippocampal_cache` | `katgpt-core/hippocampal_cache` | HOLA Hippocampal Exact KV Cache — surprise-evicted (β·‖e‖) bounded KV cache with decoupled RMSNorm-γ softmax read, complementing the GDN2 recurrent state (Plan 395, Research 378, arxiv 2607.02303, **opt-in**: G1–G4 modelless PASS, G5 perplexity deferred to riir-train Issue 038) |
| `hga` | `katgpt-core/hga` | Hierarchical Global Attention — chunk→group→token routing with mixed-RoPE summaries + tiered route-and-fetch KV store (Plan 397, Research 379, arxiv 2606.30709, **opt-in**: G1/G3/G5 PASS, G2-proxy FAIL on random-key NIAH — same class as MSA R225 GOAT-FAILED; full G2 transformer loss-gap deferred to riir-train). `TieredKvStore` trait ships always-on as generic route-and-fetch primitive. |
| `faithfulness_probe` | — | FaithfulnessProbe — causal intervention diagnostic for injected memory (Plan 278, Research 244, opt-in) |
| `triggered_injection` | — | TriggeredInjectionGate — sigmoid-thresholded inject/skip hot-path gate (Plan 278, Research 244, default-on, G3 PASS — saves compute, matches quality) |
| `manifold_power_iter_router` | — | Manifold Power Iteration MoE Router — one-shot router-row conditioning at snapshot swap via shared `spectral_retract` helper (Plan 279, Research 246, arxiv 2606.12397, **DEFAULT-ON** since Plan 279 Phase 4 GOAT 9/9 green, G1 λ-alignment + G2 MaxVio reduction + G3 zero per-token overhead) |
| `quantile_balance_router` | — | Quantile Balancing MoE Router — one-shot per-expert bias β at snapshot swap via alternating-coordinate descent on the balanced-assignment LP (Plan 455, Research 447, Su blog Feb 2026 + Marin 32B validation, **DEFAULT-ON** since Plan 455 Phase 3 2026-07-17): G1–G8 12/12 GOAT gate green + Phase 3 head-to-head Case C (composed pipeline `R'=MPI(R) → β=QB(X·R'^T) → top-k(s−β)` strictly Pareto-dominates either alone: λ 0.65→0.99 from MPI, MaxVio 1.84→0.00 from QB on orthogonal axes) |
| `cs_kv_probe` | — | CS-KV-Importance Probe + Density-Budget Interpolator — compressed-sensing KV-group importance via ablation + Lasso, sigmoid-gated top-K application (Plan 280, Research 247, arxiv 2606.13594, **opt-in**: G1 CS-beats-random + G2 sparse-vs-dense duality shape + G3 K(ca) monotone/bounded) |
| `self_advantage_gate` | — | Self-Advantage Recursion Gate — dead-compute detector via pre/post-recursion log-ratio (Plan 283, Research 250, arxiv:2511.16886, default-on, GOAT 4/4 PASS) |
| `funcattn` | `tiled_attention`, `katgpt-core/funcattn` | Functional Attention — closed-form Tikhonov k×k spectral transport operator (Plan 286, Research 257, arxiv 2605.31559, **DEFAULT-ON** since 2026-07-07): dual form `(1-α)·K̃ᵀK̃ + α·I_d` convex-combo regularization, sigmoid-basis default per AGENTS.md. Promoted after G6 LLM-domain gate PASS — the prior FAIL was an Issue 049 test-data-gen artifact (admitted ~12.5% degenerate `a==b` constant sequences at V=8, corrupting the learned basis into a spurious 0.969 plateau). D4 fix rejects degenerates; post-fix FUNCATTN=1.000 SDPA=1.000 @ release 600 FD-SGD steps (debug 40-step: FUNCATTN 0.9297 > SDPA 0.6719). 6/6 GOAT gates green (G1 correctness, G2 perf 10.9× sample-efficiency, G3 no-regression, G4 alloc-free, G5 feature-isolation, G6 LM-domain parity). Modelless (no training). Heavy downstream use: riir-ai plans 318/329/330/309/310/321 consume `C` as transport substrate; **riir-ai Issue 533 (2026-07-17)** consumes `funcattn_forward` + the `k` parameter as the attention-rank LOD row of the Thermal LOD coordinator (tier → k, data-derived from Plan 332 k-sweep elbow). |
| `functional_substitution_gate` | `katgpt-core/functional_substitution_gate`, `funcattn`, `faithfulness_probe` | Head Substitution Gate — IoU cheap-proxy + cached FaithfulnessProfile veto decision wrapper around FuncAttn (Plan 353, Research 353, arxiv 2606.19317). **Opt-in Gain-tier**: G1+G3+G4 green, G2 synthetic green (Spearman ρ ≤ −0.9 across seeds/sizes), G2 real-head deferred to riir-ai. Gate wrapper around FuncAttn; not a new primitive (the original `ProgramSynthesizedHead` draft was dropped after re-review identified FuncAttn as the existing primitive surface). |
| `chain_fold` | `thinking_cot` | ThoughtFold chain folding — inference-time CoT step pruning (Plan 195 GOAT 16/16, Plan 228 78% reduction) |
| `clr` | — | CLR Claim-Level Reliability runtime — `(mean_m v_k,m)^M` nonlinear reliability vote over claim embeddings + Long2Short brevity tiebreak + learning potential + MGPO sampling weight (Plan 284, Research 255, arxiv 2606.16140, **default-on**, GOAT G1–G5: CLR beats majority +78pp, ECE 0.0087, K=32 vote 4–5µs, zero-alloc vote internals, feature-isolated) |
| `ict_branching` | `katgpt-core/ict_branching` | ICT Distributional Branching-Point Detector — `collision_purity(π) = Σ π²` (proven unconditionally monotone, ICT §A.2.5 — H₁ wrong below π > e⁻¹≈0.37) + Jensen-Shannon divergence to group mean + `BranchingDetector` top-k% selector (Plan 294, Research 270, arxiv 2606.19771, **opt-in until G3+G8 pass**): G1 PASS paper Fig 1a bifurcation; G2 BORDERLINE-FAIL median 37.5% (paper's 10% is LLM-token-specific — sweep `k_percent` per-domain); G3 ⭐ PASS Spearman ρ(H₁, JS-uniqueness) = 0.0652 95% CI [-0.017, 0.150] < 0.5 — JS structurally orthogonal to H₁ (Super-GOAT proceeds); G4 PASS 1.96µs/call (target ≤50µs); G5 PASS 0 allocs; G6 PASS feature-isolated; G10 PASS Bebop H₁→H₂ acceptance-forecast upgrade MAE 0.402<0.423 on long-tail. Stays opt-in pending G8 (riir-ai Plan 324 runtime fusion). |
| `induced_cwm` | `katgpt-core/induced_cwm` | Induced Code World Model kernel primitive — `InducedCwmKernel: GameState` marker + `CwmCommitment` (BLAKE3) + `BeliefInferenceFn<S>` + `TransitionUnitTest` (Plan 296, Research 275, arxiv 2510.04542, **opt-in**: G1–G4 GOAT 4/4 PASS, ready for downstream consumption; LLM-induction pipeline is private — riir-ai Plan 326) |
| `induced_cwm_ismcts` | `katgpt-core/induced_cwm_ismcts`, `induced_cwm` | Information-Set MCTS over an induced CWM + belief fn (Plan 296 Phase 2) |
| `induced_cwm_tournament` | `katgpt-core/induced_cwm_tournament`, `induced_cwm` | Value Function Tournament — round-robin arena-play selector over `StateHeuristic` candidates (Plan 296 Phase 3) |
| `subspace_phase_gate` | `katgpt-core/subspace_phase_gate` | Participation ratio + numerical rank + N≥d phase-transition gate + runtime Jacobian SVD (Plan 301, Research 279, arxiv 2409.02426). Pure numeric substrate; consumed by Plan 312. **DEFAULT-ON** (Plan 301 Phase 5 T5.1, 2026-07-02) — G1 PASS + Phases 3–5 complete; also pulled transitively via `viable_manifold_graph` (DEFAULT-ON). |
| `alien_sampler` | `katgpt-core/alien_sampler` | Coherence × Availability frontier ranking — `AlienSampler<V,C,A>` z-scored fusion + `MedianTopMAvailability` median-of-top-m cosine rule (Plan 311, Research 293, arxiv 2603.01092). **🪦 GOAT FAILED 2/4** — G1+G2 fail (β phase-transition at β≈0.4); G3 PASS post-rayon (4.56×); G4 PASS. Opt-in for paper reproduction. |
| `viable_manifold_graph` | `katgpt-core/viable_manifold_graph`, `subspace_phase_gate` | Discrete safe-manifold navigation — `pullback_volume` + `SafeManifoldGraph` (CSR adjacency) + `manifold_geodesic` / `manifold_random_walk` / `manifold_curiosity_walk` (Plan 312, Research 294, arxiv 2206.00106). **DEFAULT-ON** — G1–G7 correctness all PASS + perf bench PASS post-CSR (random walk 7.10 ns/step, 14× under 100 ns target). |
| `ac_prefix` | `katgpt-core/ac_prefix` | AC-GPT arbitrary-conditional prefix — mask builder + sequence augmenter turning any causal Transformer forward into single-pass `p(xe | xc)` (Plan 313, Research 295, arxiv 2606.14943). Three-region attention rule, branch-free `attends(i,j)`, bit-packed `AcPrefixMask`. **DEFAULT-ON** — G1 (buffer construction bit-identical) + G2 (27.46× speedup vs iterative-MLM) + G3 (empty-prefix no-regression) + G4 (alloc-free hot path) all PASS. G1 reformulated: original "matches iterative-MLM to 1e-4" is a trained-model property (holds post-LoRA, riir-train's job); modelless G1 tests buffer construction. |
| `closed_unit_compaction` | `katgpt-core/closed_unit_compaction` | Closed-Unit Compaction Gate (CUCG) — generic rubric-gated trajectory compaction primitive (Plan 333, Research 300, arxiv 2606.23525 SelfCompact). Fires summarization at structurally-safe moments (closed-unit ∧ summarizable ∧ progress ∧ ¬stuck) via sigmoid projections + `FireRule` Boolean tree + `Backstop` token-pct + optional `skip_if_reliable` CLR fuse, instead of fixed token thresholds. **DEFAULT-ON** — 7/7 GOAT gates PASS: G1 recall=1.000/FDR=0.000, G2 50% suppression, G3 probe ratio=1.00, G4 zero-alloc, G5 feature isolation, G6 0 softmax, G7 `can_freeze` isomorphism (Super-GOAT: trajectory compaction and shard freeze are the same primitive, proven structurally). `evaluate()` 8.91ns <50ns, 112.9M/s ≥50M. Zero runtime cost unless a caller invokes evaluate. |
| `committed_field_blend` | `katgpt-core/committed_field_blend` | CommittedFieldBlend — sampling-invariant per-entity MoE: frozen sigmoid blend of N archetype operator fields, weights computed ONCE from a trajectory summary + BLAKE3-committed (Plan 321, Research 302, arXiv:2510.00621 FAME). Defining property: **sampling invariance** (FAME Prop. 3) — dense vs sparse observation of same trajectory → identical committed `pi` + identical dynamics. Implies `personality_composition` (reuses sigmoid + `simd::simd_fused_scale_acc`). Closed-form Lipschitz safety bound (`max_k sigmoid(pi_k/tau)·L_k`, FAME Lemma 1). **DEFAULT-ON** (2026-06-28) — G1–G5 GOAT gate ALL PASS (G2 sampling invariance holds across 100 entities, worst-case Δpi=1.19e-6; G4 zero-alloc on apply + commit; G5 BLAKE3 reproducible + tamper-detecting). Runtime validation also shipped: riir-ai Plan 336 G6a–G6e + G7a ALL PASS (2026-06-26). Modelless gain (closed-form sigmoid projection + BLAKE3 commit, no training). Zero runtime cost unless a caller invokes commit/apply_blended. |
| `qgf` | `qgf_oracle`, `qgf_projector` | QGF Test-Time Q-Guided Flow parent — test-time Q-gradient guidance (Plan 268, Research 236, arxiv 2606.11087). Parent feature; enables the trait + projector only. The drafter (F1) and adaptive (F4) are separate sub-features. |
| `qgf_oracle` | — | `QGradientOracle` trait — drop-Jacobian critic gradient `∇_a Q(s, â_1)` (Plan 268 F3). Four shipped impls: `NoGuidanceOracle` (freeze-tier no-op, zero confidence), `LeoHeadOracle<H>` (Plan 268, single LEO teacher head — `leo_all_goals`), `FlowFieldOracle` (Plan 268, owned `FlowField` `(dx,dy)` lookup — `flow_field_nav`), and **`DualLeoOracle<H1,H2,M>`** (Plan 467 / Proposal 007, LEO teacher + UVFA student α-mix at the gradient level — `leo_all_goals + dual_leo`; encodes the Plan 460 "no operator between mix and consumer" invariant by construction). **Plan 467 G1–G4 PASS mechanistically; G5 measured FAIL on synthetic data** (riir-ai Bench 553, 2026-07-18: dual 0.00% vs single 0.50% on T7 Go puzzles, but the correctness invariant `b ≡ a` held bit-identically — mechanism correct, quality gate FAILs because synthetic data produces near-flat Q-fields). **G5 measured FAIL on civ real networks too** (riir-ai Bench 558, 2026-07-19: dual +2.69% vs single 35.68% → 36.64% on civ action-prediction, ≥3% gate — fourth-axis stop rule; the civ dual-LEO investigation is fully closed per Research 322 — the alternative-critic escape hatch was category-confused). Stays opt-in with documented unproven G5 across both synthetic and civ real-network regimes; reopens only on seal integration gain, new game domain positive G5, or Q-vs-forecast research breakthrough. |
| `qgf_projector` | `speculative_generator` | `FirstOrderProjector` — one-step Euler projection `â_1` of the final output (Plan 268 F2). |
| `qgf_drafter` | `qgf` | `QGuidedDrafter` — `tilt_logits` hot path (`logits += w · ∇Q`, SIMD AXPY, zero-alloc) + tier-routing policy `route_for` (Plan 268 F1 + Phase 4 T8). **Opt-in**: katgpt-core mechanism gates G1–G5 PASS (G1 non-circular via 2 negative controls, G4 0 allocs/2000 calls, G5 sigmoid bounded/finite/monotone); downstream selling-point gates (Sudoku/DDTree/Bomber) deferred to riir-ai. |
| `qgf_adaptive` | `qgf_drafter` | `VarianceAdaptiveGuidance` — per-query sigmoid `1/β = sigmoid(k·(confidence − threshold))` (Plan 268 F4). Novel extension beyond the paper's fixed `1/β`. **Opt-in** until real-world validation on Bomber arena. |
| `cross_stage_relocation` | `causal_head_importance`, `katgpt-core/cross_stage_relocation` | Cross-Stage Residual Relocation Operator + Permeation-Map Diagnostic — Knowing-Using Gap (Plan 431, Research 417, arxiv 2607.08393). Two modelless primitives: (1) `permeation_scan_into` 2D `(src,dst)` intervention heatmap reusing Plan 358's `direct_effect_importance` + two-cluster classification; (2) `RelocateOp` applied operator with paper's fixed `(0.82L→0.45L)+(0.10L→0.45L)` default (`RelocatePair::LateEarly`, 58–75% oracle recovery). **Opt-in** — G1–G6 PASS for katgpt-rs scope (scan 10–25% faster than hand-rolled; operator <0.03% of forward pass; 0 allocs); G7 (58–75% recovery transfer to our substrate) deferred to Phase 3 PoC in `riir-poc/`. Stack slot: intervention/diagnostic. |
| `occupancy_ratio` | — | FORE Occupancy-Ratio Estimator — adjoint-Bellman KL-contraction occupancy-ratio probe (Plan 438, arxiv 2607.05375). **Phase 5 CLOSED (2026-07-15)**: all tasks complete; FORE stays opt-in (Baird-MRP G1 PASS but no downstream consumer wired). Ships `OccupancyRatioEstimator` + KL-projection fit loop (Algorithm 1) + recos MAG `TransferMetric` cold-path diagnostic (Plan 437 Phase 3/4 DONE). |
| `ane_fused_chain` | — | ANE Fused-Chain Cost Model — dependency-aware overlap estimator for chained ANE operations accounting for DMA/Port/Kernel overlap (Plan 439, arxiv 2606.22283). Real M3 Max validation landed (Phase 2.5); consumer integration (riir-engine) shipped (Phase 4). **DEFAULT-ON** (Plan 439 Phase 2 GOAT G1–G5 PASS, promoted 2026-07-14). |
| `multi_agent_path` | — | Lifelong LaCAM Local Guidance Substrate — modelless, training-free, receding-horizon windowed multi-agent pathfinder (Plan 440, Research 424, arXiv:2605.16855 Arita & Okumura AAAI 2026). `LifelongLaCam<P, C, G>` = PIBT one-step generator + per-agent space-time A* guidance (BFS distance field) + warm-start schemes (LLLG_Π / LLLG_Φ / LLLG_∅) + one-step blocking-count hindrance estimator. **Five pluggable seams** (`CostFn`, `LocalGuidanceSource`, `WarmStartScheme`, `HindranceEstimator`, `FlowField`) for the Super-GOAT fusion (riir-ai/318: HLA × Crowd MCGS × P350). Issue 148 upgraded all 4 benchmark maps to real MovingAI files (ht_chantry improved 0.09→0.27); Issue 149 added the `FlowField<P>` seam for 1-wide corridor direction assignment (correct + tested, near-zero effect on real maps — game corridors are 2-wide). Pure heuristic — no training, no backprop. **Opt-in** — G3 (no-regression, 1601 tests) + G4 (latency, 467ms median @ 1000 agents on real MovingAI maps) PASS; G1 PARTIAL (2/4 maps — warehouse/ht_chantry fail, greedy PIBT lacks priority inheritance); G2 FAIL (warm-start not consumable by greedy rollout). Promotion deferred until G1/G2 unblock and riir-ai/489 G5–G7 fusion gates validate the Super-GOAT claim. |
| `flow_field_nav` | `katgpt-core/flow_field_nav` | Fourier-Smoothed Flow Fields for LEO crowd navigation (Plan 242, GOAT PASS 46.9%). `FlowFieldCache::get_or_compute<H: LeoHead>` builds a smoothed navigation field from raw `Q_LEO[:,:,g]` slices via `LeoPotentialGrid::from_q_values` + FFT low-pass + finite-difference gradient + unit-length normalization. **Opt-in** — GOAT 46.9% quality gain on synthetic 2D landscape. **Dual-LEO fusion paths** (composition of `dual_leo` + `flow_field_nav`, both already default-on): Plan 459 ships `get_or_compute_dual` (pre-max Q-slice α-mix — G1–G4 PASS, **G5 FAIL** honestly, 25.9% stuck reduction at α=0.1 short of 30% gate; demoted to compatibility); Plan 460 ships `get_or_compute_dual_postmax` (post-max potential α-mix — **G5' PASS at α=0.10: 31.5% stuck reduction; PROMOTED as the recommended dual path**). "Promoted" = doc-level recommendation, NOT a feature-gate promotion (`flow_field_nav` and `dual_leo` were already default-on; the dual path is a sibling API, not a replacement). Real-network quality gain requires riir-games-civ wiring (CivLeoNet + UVFA wrapper). See [`.benchmarks/459_flow_field_dual_leo_mixer_goat.md`](../../.benchmarks/459_flow_field_dual_leo_mixer_goat.md) + [`.benchmarks/460_flow_field_dual_leo_postmax_goat.md`](../../.benchmarks/460_flow_field_dual_leo_postmax_goat.md). |
| `simd_lut_dequant` | — | Software SIMD LUT-accelerated dequant distilled from StreamDQ's hardware DQB (Plan 452, Research 418 §2.3, arxiv 2607.11262). **Split GOAT decision:** the fused `dequant_dot_via_lut` kernel wins **4.58×** over the two-step path (NEON FMA + no buffer spill) → **DEFAULT-ON**; the plain `dequant_via_lut` is **3.5× slower** than the arithmetic cast on NEON (scalar gather, no native instruction) → opt-in infrastructure for future FP8/INT8. G1 bit-exact, G3 no-regression, G4 0 allocs (stack `[f32; N]` LUT + caller-owned out). Cross-repo: `simd_lut_q4k` default-on in riir-engine (Plan 486 T3.3). |
| `lacam_escalation` | `multi_agent_path` | Bounded one-step LaCAM — the constraint-tree search from Okumura 2023 applied to a single tick, replacing the fake "LaCAM escalation" (shuffled-priority retries). The critical insight (Research 441, from reading `Kei18/lacam`): LaCAM = recursive PIBT **+ constraint tree** — only the recursive PIBT half was tried before (Issues 140/143 collapsed throughput); the constraint tree bounds the recursion. **Opt-in** — G6c collision-freedom 1.000 (37.5%→100%), G-col vertex rate 0.0% (Issue 154 fixed), G-PI no-collapse 0.69, G3/G4 PASS; **G1 3/4 maps** (ht_chantry 0.28 marginal — one-step resolves single-tick collisions but not multi-step maze detours). **Issue 546 (2026-07-18) added the multi-step path** as `EscalationBudget::multistep_default()` (stuck-agent targeting + depth 8 + 100ms/100K-node budget) — opt-in escalation; default behavior remains Plan 453 one-step (bit-identical). Measured +0.6% throughput / +2.5% latency on ht_chantry-real: marginal, **G1 not closed** (corridor-queue deadlocks are structurally too long for any bounded-depth search). Pair with Proposal 006 (bi-directional flow field) for the full G1 close. |
| `interpolation_geometry` | `katgpt-core/interpolation_geometry` | Interpolation Geometry — iMAUVE + 5-way intervention probe for committed latent substrates (Plan-158 / Research 445, Prabhudesai & Geng, *Latent Thought Flows with Text Compression*, Jun 2026). Generic `LatentSpace` trait abstracting over HLA `[f32;8]` / `style_weights[64]` / archetype-blend π / KarcShard / ZoneGeometryPod / MerkleFrozenEnvelope (the six substrates cataloged in Research 445 §2.6). Two protocols: `imauve_score` (nearest-neighbor midpoint coherence — the paper's headline metric, Pearson r=0.99 with downstream quality) + `intervention_battery` (matched/shuffled/zero/mean/noise probe extending Plan 278's FaithfulnessProbe to per-entity committed state). **Opt-in** — three-pressure audit (Q1 summarize-vs-route / Q2 runtime-depends-on-latent / Q3 local-context-vs-bypass) PASS for all six substrates; see [Benchmark 456](../../.benchmarks/456_interpolation_geometry_goat.md) + [`.docs/04_calibration/interpolation_geometry.md`](../04_calibration/interpolation_geometry.md). Pure modelless evaluation methodology — NOT a training primitive. |
| `grapem_rodrigues` | `katgpt-core/grapem_rodrigues` | GRAPE-M Rank-2 Rodrigues Exponential — O(d) closed-form application of `exp(n·ω·L)` for arbitrary rank-2 skew generator `L = abᵀ − baᵀ` (Research 446 / arXiv:2512.07805 §2.3). Pure modelless float arithmetic on a user-supplied plane `(a, b)`. Subsumes `phase_rotation`'s scalar-broadcast 2D rotation as the canonical-basis special case. **Opt-in** — G1 bit-identical to materialized `expm(L)`; G2 latency `< 2× phase_rotation_gate_into`; G4 zero-alloc. See [Benchmark 457](../../.benchmarks/457_grapem_rodrigues_goat.md). Promotion deferred pending a hot-path consumer. |
| `position_group_action` | `katgpt-core/position_group_action`, `grapem_rodrigues` | Unified `PositionGroupAction` trait — RoPE / ALiBi / FoX / Wall / NoPE / GRAPE-M as instances of `G(n) = exp(n·ω·L)` (Research 446 / arXiv:2512.07805 §2.2 + §4.1). Vocabulary bridge for position-encoding-agnostic tooling (KV compaction, attention matching). Hot-path code keeps using `PositionFreeCompactor` / `WallDiagonalGate` directly; the trait is for cold-path interop. **Opt-in** — G3 no-regression (existing RoPE/Wall paths unchanged when feature off). See [Benchmark 458](../../.benchmarks/458_position_group_action_goat.md). |
| `grape_ap_vector` | `katgpt-core/grape_ap_vector` | GRAPE-AP Vector-Similarity Path-Integral Decay Gates — content-aware extension of Wall Attention's scalar prefix-sum gates (Research 446 / arXiv:2512.07805 §5). `ψ_h(t,ℓ) = α·g(⟨p_t, R_ℓ·p_ℓ⟩/d)` with `g=log_sigmoid` — tokens whose positional embedding matches the query's decay slower. Maintains per-head prefix sum along causal path. Wall is the scalar special case (endpoint-independent embeddings). **Opt-in** — G2 latency overhead `< 1.5×` Wall's scalar path; G4 alloc-free after scratch init. See [Benchmark 459](../../.benchmarks/459_grape_ap_vector_goat.md). |
| `grape_joint_lift` | `katgpt-core/grape_joint_lift`, `grapem_rodrigues` | GRAPE Joint Lift — `GL(d+2)` block-diagonal composition of rotary (GRAPE-M) + additive (GRAPE-A, paper §4.1) into a single group action per Appendix E of arXiv:2512.07805. One-pass `score_into`: `q^T·exp(m·ω_rot·L)·k/√d + m·ω_add·(softplus(v·q/√d) + softplus(u·k/√d))`. Closes the GRAPE composition story: today Wall *replaces* RoPE; this primitive proves they *compose* into a single one-parameter subgroup of `GL(d+2)` while preserving the exact relative law. Decoupled `omega_rot`/`omega_add` is a strict generalization of the paper's shared `ω`. **Opt-in** — G1 bit-identical to manual composition + relativity; G2 latency smoke; G4 alloc-free after `new`. See [Benchmark 460](../../.benchmarks/460_grape_joint_lift_goat.md). |
| `causal_identification` | `katgpt-core/causal_identification` | Causal-ID — Algorithmic Syntactic Causal Identification (Plan 457, Research 450, arXiv:2403.09580 Cakiqi & Little 2024). Pure modelless graph rewriting on ADMGs with bidirected confounders: `identify(Y, do(A))` returns the interventional signature backbone `Y⋆ = An(Y in G[V\A])` via the recursive Shpitser-Pearl ID algorithm. Four submodules: `types` (`NodeId` BLAKE3 `[u8;32]`, `Admg`, `AdmgSignature` inline-ArrayVec<32>→heap fallback, `IdentificationError` with hedge pair), `fixing` (`districts`, `ancestors`, `fix_node`, `try_fixseq` greedy fixing sequence), `identify` (the 6-step recursive ID), `subgraph` (bounded-BFS to keep the `O(k²)`–`O(k³)` algorithm on a ≤32-node subgraph). **DEFAULT-ON** (Plan 457 Phase 5 promotion, 2026-07-18): Phase 2 GOAT gate G1+G2+G3+G4 ALL PASS — G4 closed by Issue 183 (Scratch refactor cut per-call allocs 284→198, -30%, latency 8.26→6.07µs, -27%) + Issue 184 / Benchmark 466 (`districts()` + `try_fixseq` graph-construction allocators eliminated via callback-based `for_each_district_with_buffers` + workspace-based `try_fixseq_into`; allocs further reduced 198→133/call, -33% more, -53% cumulative from 284). Phase 4 T4.5 synthetic Consumer A bench cleared T4.7 promotion gate (71.7% non-trivial Ok rate, 43/60 queries on a 100-node game-world KG with 3 faction confounder cliques). Promotion follows the codebase pattern (manifold_bandit P370, set_attention P354, poincare_navigator P449). Offline-only (24µs on 13 nodes is well outside the 20Hz tick). Pure modelless — no training, no learned params. `blake3` + `arrayvec` already non-optional. Zero runtime cost unless invoked. See [`.benchmarks/465_causal_id_alloc_free_scratch.md`](../../.benchmarks/465_causal_id_alloc_free_scratch.md) + [`.benchmarks/466_causal_id_p4_zero_alloc.md`](../../.benchmarks/466_causal_id_p4_zero_alloc.md) + [`.benchmarks/464_causal_id_consumer_a_synthetic.md`](../../.benchmarks/464_causal_id_consumer_a_synthetic.md). |
| `conformal_predictive_intervals` | `katgpt-core/conformal_predictive_intervals` | Conformal Predictive Intervals — modelless UQ overlay wrapping any `PointForecaster` with a per-channel × per-horizon-bucket exp-recency-weighted residual ring buffer, reading empirical quantiles to produce coverage-guaranteed predictive intervals `[point+q_{α/2}, point+q_{1−α/2}]` (Plan 340, Research 322, arxiv 2605.03789 CSP + 2606.09473 "Report the Floor"). CRPS / Winkler / empirical-coverage metrics (`conformal::metrics`). The `ConformalIntervalCalibrator<SeasonalNaiveForecaster>` with m=1 is the **canonical conformal-naive floor** — every UQ-bearing primitive's GOAT gate MUST beat it (Issue 010 "Report the Floor" rule, codified in AGENTS.md Feature Flag Discipline). No training, no learned params. **DEFAULT-ON** (Plan 468 promotion, 2026-07-20): primitive-level G1–G4 GOAT PASSed (Bench 340, 2026-06-30) — coverage [0.9445, 0.9493] ∈ [0.93, 0.97], `interval_into` H=1 **642 ns** (≤ 1 µs target), 0 allocs/100 calls, bit-reproducible; runtime-consumer promotion gate satisfied by Bench 564 (MCTS collapse G3 PASS — per-NPC calibrated τ beats fixed magic number on F1) + Bench 565 (Salience Tri-Gate G3 PASS — interval-width Delegate nudge dF1=+0.3145 at 6.3× gate margin, dFP=−0.8155; Plan 513 width-definition fix vindicated bit-identically). Two consumers FAILED (Bench 562 curiosity — wider than 5×EMA; Bench 563 sleep-time — distribution-level summary loses cycle info); Cargo.toml language required only one PASS, two landed. **Consumer-level gates STAY opt-in** — `karc_conformal_width` (riir-engine, +113.9% overhead per Plan 512 — FAIL default promotion), `salience_conformal_width`, 4 probe features. The three-layer split (primitive DEFAULT-ON + consumer gates opt-in) is the canonical append-only pattern. Pure modelless (empirical-quantile calibration). Zero runtime cost unless invoked. See [`.benchmarks/340_conformal_goat.md`](../../.benchmarks/340_conformal_goat.md) + [`.docs/04_calibration/conformal_predictive_intervals.md`](../04_calibration/conformal_predictive_intervals.md). |
| `karc_forecaster` | `katgpt-core/karc_forecaster` | KARC — Kolmogorov-Arnold Reservoir Computing delay-basis ridge trajectory forecaster (Plan 308, Research 288, arXiv:2606.19984 Huang/Kurths/Tang). `KarcForecaster<D,M,K>` + sealed `KarcBasis` trait (Fourier/Chebyshev/BSpline) × closed-form ridge readout. Phase 2 ships higher-order R=2 + chunked Gram + ALS low-rank `Wout ≈ A·B` (the form that persists into a `KarcShard` in riir-neuron-db). **DEFAULT-ON** (Phase 22, 2026-07-21, Issue 186 Path D3 split-config gate): G1 NRMSE **9.43e-4** PASS at K=8/M=8/R=2 d_h=18_720 (λ=5e-2 λ-sweep recovering the underdetermined system); G1 threshold **8.16 LT** PASS at K=8/M=24/R=1 λ=5e-3 (Phase 1 + Phase 5.3 confirm). Both passing configs at the same K=8 delay length; the compound gate (both legs in ONE config) is structurally infeasible — NRMSE requires R=2, threshold requires M≥24, R=2 × M=24 → d_h ≥ 166_752 (Gram ≈ 222 GB). G2 (381 ns), G3 (0 allocs), G4 (bit-reproducible Wout) ALL PASS. Pure modelless (no training, no learned params). Zero runtime cost unless a caller constructs a KarcForecaster. See [`.benchmarks/308_karc_goat.md`](../../.benchmarks/308_karc_goat.md). |
| `karc_householder_eig` | `katgpt-core/karc_householder_eig` | Issue 186 Path B (2026-07-20) — swaps the G-path eigendecomp in `low_rank_fit_jacobi_bstep` from `karc::jacobi_eigen` (O(d_h³·n_sweeps), infeasible at d_h > ~5000) to `linalg::symmetric_eig` (Householder tridiag + implicit-shift QL, ~5-10× faster at d_h ≥ 256, feasible at d_h=18_720). The new eigensolver is always compiled as a generic `linalg` primitive; this feature gates only the wiring in `karc::large_dh`. Implies `karc_forecaster`. **Opt-in** — T1-T5 PASS (correct + 7-14× faster than Jacobi at n ≤ 512); T6 (d_h=18_720 ≤30 min wall) PASS via full-rank direct Cholesky path. The Householder+QL path itself stays opt-in because direct Cholesky is both faster and more accurate for the G1 measurement. |
| `karc_householder_eig_par` | `katgpt-core/karc_householder_eig_par` | Issue 187 (2026-07-20) — row-parallel rayon variant of `linalg::symmetric_eig` for the d_h=18_720 timing trial. The four row-parallel hot loops (Householder matvec, rank-2 update, Q accumulation, QL eigenvector rotation) parallelize across rows via `par_chunk_mut(n)`; each row's work is fully sequential so the result is bit-identical to the serial path. Implies `karc_householder_eig`. **Opt-in** — T5/T6 PASS (row-parallel bit-identity + timing feasibility). **Landed a critical QL convergence fix** for near-singular Grams (the NR-local check `|e[m]| + dd == dd` cannot deflate tiny-eigenvalue matrices; added the LAPACK `dsteqr` global-scale criterion — affects both serial and parallel paths). Stays opt-in: full-rank direct Cholesky is preferred for the G1 measurement at d_h=18_720. See [`.benchmarks/308_karc_goat.md`](../../.benchmarks/308_karc_goat.md) §Phase 5. |
| `karc_regime_gate` | `katgpt-core/karc_regime_gate` | Plan 556 Phase 1 (2026-07-20) — KARC Regime Gate. Closed-form residual-MSE mux between `KarcForecaster` (chaotic-regime specialist) and `SeasonalNaiveForecaster` (periodic-regime floor). Directly fixes the structural periodic-blindness documented in `.benchmarks/010_report_the_floor_consolidated.md` §T7 (K-sweep refuted the "K=4 too shallow" hypothesis: KARC's basis can't fit periodic data regardless of K). Two `WelfordMse` accumulators + sigmoid confidence + cold-start floor. **Revised from variance-only to MSE (variance + bias²)** after Plan 514 surfaced the failure mode where a consistently-biased forecaster has variance 0 but large error. Implies `karc_forecaster` (the gate routes to KARC) + `conformal_predictive_intervals` (the floor). **Opt-in** — primitive-level G1+G2+G3+G4 ALL PASS (37 ns median `decide()`, 0 allocs, bit-identical when gate routes to KARC); runtime-integration gain measured by riir-ai Plan 514 Phase 1: **G1 PASS (92.45% MAE reduction on mixed-regime NPC corpus)** + G2 ~at-budget (89 ns/tick). Stays opt-in pending production-corpus gain. See [`.benchmarks/556_karc_mitigations_goat.md`](../../.benchmarks/556_karc_mitigations_goat.md). |
| `karc_batched_matvec` | `katgpt-core/karc_batched_matvec` | Plan 556 Phase 2 (2026-07-20) — KARC Batched MatVec. SIMD-batched forecast across N forecasters of identical (D,M,K) shape. Crowd-scale perf primitive: amortizes memory bandwidth by laying out N `Wout` matrices contiguously and hoisting the per-output-row `simd::simd_matvec` call across the batch. `KarcBatchForecaster` + `karc_batched_matvec_into`. Implies `karc_forecaster` (operates on KarcForecaster's Wout). **Opt-in** — G1+G3+G4 PASS; **G2 PARTIAL PASS** (`.benchmarks/556_karc_mitigations_goat.md`, 2026-07-20): pure-matvec amortizes well (4.0× at N=8, 7.0× at N=32 — contiguous-layout + loop-hoisting wins); full-forecast amortization does NOT materialize (1.05× at N=8) because per-NPC `feature_expand` dominates (~75% of per-forecast cost) and is not amortized by the batched matvec. Hitting the original G2 full-forecast target requires a separate `feature_expand_batched` primitive (future work). **Plan 514 Phase 3 architecture validated:** the right consumer is cell-shared-KARC + per-NPC latent_functor deviation (ONE feature_expand per cell, batched matvec across N NPC Wouts), not per-NPC-Wout batching. Pure modelless (linear algebra only). Zero runtime cost unless constructed. Opt-in until Plan 514 Phase 3 cell-shared design demonstrates the gain. |
| `karc_lod_tier` | `katgpt-core/karc_lod_tier` | Plan 556 Phase 3 (2026-07-20) — KARC LOD Tier. Config tag + tier-promotion Wout projection. Three nested tiers (LOD0 background D=8/M=4/K=2 d_h=64 / LOD1 midground D=8/M=8/K=4 d_h=256 / LOD2 hero D=8/M=8/K=8 d_h=512) map to different `KarcForecaster` const-generic monomorphizations. Nested-subset structure (LOD0 features are a strict prefix of LOD1; LOD1 of LOD2) makes tier promotion a pure index remap — down-tier preserves surviving Wout columns bit-identically; up-tier zero-fills new columns. R=1 only in Phase 3; R=2 promotion-gate config (d_h=18_720, Issue 185/186/187) deferred. Pure modelless (matrix projection). Zero per-tick cost; tier promotion is one-time per NPC. Implies `karc_forecaster`. **Opt-in** — primitive-level GOAT G1-G4 all PASS (worst-case tier promotion 831 ns vs 10 µs target). **Runtime integration (riir-ai Plan 514 Phase 2) — honest split verdict**: G2 **PASS at 1k production scale** (14.7% savings, 5.3× headroom, re-validated 2026-07-20) but **FAIL at 10k crowd scale** (4.9% savings — dormant-Lod1 memory overhead cancels the compute savings because 10k-NPC state exceeds L3 cache, so memory bandwidth dominates). Plan 514 Phase 3/4 G2 targets revised from "10k NPCs on a single node" to "1k NPCs per shard"; the crowd-scale NPC sharding substrate landed 2026-07-25 at `riir-engine/src/npc_shard.rs` (feature `npc_shard`) — Issue 556 POC confirmed single-process sharding is ruled out (22% regression vs flat 10k — L2 is shared across the process; per-NPC tick has no intra-tile reuse) + multi-node distribution is required, see `riir-ai/.benchmarks/556_npc_shard_goat.md`. Stays opt-in until either a pure-enum redesign (breaks `forecaster()` API) or a positive gain on a smaller-scale corpus. See [`.benchmarks/556_karc_mitigations_goat.md`](../../.benchmarks/556_karc_mitigations_goat.md) + [`riir-ai .benchmarks/514_karc_lod_dispatch_goat.md`](../../../riir-ai/.benchmarks/514_karc_lod_dispatch_goat.md). |
| `poincare_navigator` | `katgpt-core/poincare_navigator` (implies `subspace_phase_gate`) | Poincaré Adapter — closed-form latent navigation distilled from SeeSE3 (Plan 449, Research 449, arXiv:2607.14228 Chen et al. DeepMind 2026). A frozen `PoincareAdapter` Pod holds `(φ, W, W†)` — given a desired movement in target space (3D pose delta / HLA affect delta), recover the latent step via `z_dest = z_src + φ⁻¹(φ(z_src) + W†·Δtarget)`. `poincare_navigate_into` is the zero-alloc hot path; `fit_poincare_adapter` is the cold closed-form ridge + PCA + thin SVD fit. **DEFAULT-ON** (Plan 449 Phase 3 promotion, 2026-07-18): G1–G7 GOAT gate ALL PASS — G1 local decodability (max |decoded delta| 0.0126, 800× under sanity bound); G3 inverse navigation Hit@0.3 = **1.000** (perfect); G4 0 allocs/100 calls; G5 `poincare_navigate_into` **809 ns/call** at d=64 (≤1µs target, 20% headroom); G6 4-step open-loop trajectory bit-identical + bounded; G7 latent-vs-raw boundary (TypeId check). G2 caveat (modelless PCA-tanh adapter R²=0.71 < linear-only ridge R²=0.93 on a coupled curved fixture) **closed by riir-train Plan 317** — trained 2-layer MLP φ reaches R²=0.9997. Promotion pattern (manifold_bandit P370 / set_attention P354 / ac_prefix P313): modelless + zero-cost-unless-invoked + GOAT-passes-on-load-bearing-axis (G3 inverse navigation, G4 zero-alloc, G5 latency) → DEFAULT-ON. Co-gates `subspace_phase_gate` for `thin_svd_into`; reuses Plan 308 `ridge_solve_direct_f32`. Pure modelless (closed-form PCA + ridge + SVD pseudoinverse). Zero runtime cost unless invoked. See [`.benchmarks/449_poincare_goat.md`](../../.benchmarks/449_poincare_goat.md) + [`.docs/05_adaptation/poincare_navigator.md`](../05_adaptation/poincare_navigator.md). |
| `chunked_content_store` | `katgpt-core/chunked_content_store` | ChunkedContentStore — Lore-distilled chunked content-addressed Merkle blob store (Plan 448, Research 262, EpicGames/lore). Bytes → `FixedSizeChunker` / `FastCdcChunker` (content-defined chunking) → BLAKE3 per chunk → dedup via `papaya` lock-free hashmap → binary Merkle root = `BlobId`. O(log n) inclusion proofs via `build_binary_merkle_proof` + light-client-friendly associated fn `verify_binary_merkle_proof` (no `&self`). `chunked_net_fetch` adds the optional `NetChunkFetcher`. **DEFAULT-ON** (Phase 19b promotion fix-up, 2026-07-18 — bench file recorded promotion but Cargo.toml entry was missed until then): G1–G7 GOAT gate ALL PASS — G1 dedup **8.47×** on 90%-shared corpus (≥5.0 target); G2 incremental push **1.35%** bytes touched (CDC) vs 52.94% (FixedSize negative control, ≤5% target); G3 inclusion prove **588 ns** + verify < 1µs (release; 2088× speedup after cached Merkle levels fix in Plan 448); G4 type-system-enforced light-client verify (associated fn, no `&self`); G5 hot-path p99 < 200 ns (release, papaya zero-alloc `.copied()`); G6 `--no-default-features` clean; G7 tamper detection **10000/10000** on 1-bit flip. Pure modelless (BLAKE3 + binary Merkle, no training, no learned params). Zero runtime cost unless a caller constructs a store. Consumed by riir-ai Plan 319 (Executable Asset Vessel + Quorum Gitflow) — `AssetStoreAdapter<S: ChunkedContentStore>` in `crates/riir-ffi/src/asset_vessel_sidecar.rs`. See [`.benchmarks/262_chunked_content_store_goat.md`](../../.benchmarks/262_chunked_content_store_goat.md) + [`.docs/03_memory/chunked_content_store.md`](../03_memory/chunked_content_store.md). |
| `smooth_min_similarity` | `katgpt-core/smooth_min_similarity` | Smooth-Min Soft Similarity — variable-length multi-token retrieval aggregator (Plan 437, Research 385, arXiv:2602.10908 SoftMatcha 2 Yoneda et al. ICML 2026). `smooth_min_similarity(cosines: &[f32], beta: f32) -> f32` interpolates between plain-min (β→∞, strictest) and plain-sum (β≈1, most lenient) — penalizes low-cosine positions more than plain mean, defeating the "distractor with 1-2 exact-match positions + several unrelated positions" failure mode. **DEFAULT-ON** (Issue 041 T6 consumer GOAT PASS, 2026-07-12): PoC GOAT G1 recall@5 **+12.0pp** (0.815 vs 0.695) on synthetic 200-item / 200-query fixture; G2 latency overhead **~0 ns** (LLVM vectorized); G3 β sensitivity all β ∈ [10¹, 10⁶] beat plain cosine. Consumer GOAT (T6 SmoothMinAligned in katgpt-attn-match): recall@5 = **1.000** vs Cosine 0.495 (+50.5pp) on position-aligned multi-token retrieval. Pure modelless (arithmetic on pre-computed cosines, no training, no weights, zero deps). Zero runtime cost unless called. |
| `octree_ctc` | `sense_composition` (alias) | OctreeCTC Reconstructive Memory Navigation (Plan 248, Research 216, arXiv:2606.06036). `octree_ctc` is an **alias feature** for `sense_composition` in katgpt-core — the standalone feature was removed from the root crate after Issue 007 Phase C moved the only consumers (`octree_ctc_demo` + recall test) to riir-engine; katgpt-core still ships the alias for direct consumers. **DEFAULT-ON** (Plan 248 Phase 5): GOAT PASS — recall ≥ 20%, **93.2 ns** < 200 ns target. Pure modelless (octree reconstruction + cosine gates). Zero runtime cost unless a caller constructs a reconstruction. |
| `sector_projection` | `katgpt-sense/sector_projection` (forwarded via `katgpt-core/sector_projection`) | SectorProjection — multi-sector spatial projection primitive (Plan 262, Research 216). `SectorProjection<N_DIR, N_SECTOR>` projects an observation onto a fixed bank of canonical sector directions — the spatial-cognition half of the Latent Physics pair (with `action_bridge`). Latent→raw bridge for NPC perception ("where am I being pushed from?"). **DEFAULT-ON** since Plan 262 Phase 2 GOAT gate. Pure modelless (closed-form dot products). Zero runtime cost unless constructed. |
| `spectral_differentiation` | `katgpt-core/spectral_differentiation` (implies `dep:rustfft`) | Spectral Differentiation — standalone FFT-based spectral differentiation for periodic uniform 1D grids (Plan 325, Research 307 §3 candidate #2, arXiv:2511.05963 *Fourier Neural Operators Explained* §2.1). The specialized case where DEC's general `exterior_derivative` (cell-complex machinery) is overkill — closed-form FFT + frequency-domain multiplier `(iω)^m`. **DEFAULT-ON** (Plan 325 Phase 3, 2026-06-25): G1 order-1 err **5.4e-7**<1e-4 + order-2 err 1.3e-6<1e-3 + spectral-vs-FD **290×** ≥100×; G2 N=1024 **3.82µs**<50µs (13× under); G3 order=0 identity bit-identical (err 2.4e-7<1e-5); G4 0 allocs/100 calls (cached `Arc<Fft>` plans + `process_with_scratch`). Pure modelless closed-form FFT. |
| `arg_protocol` | `katgpt-core/arg_protocol` | ARG Protocol Primitives — generic protocol vocabulary distilled from the ARG Standard (Plan 327, Research 309, Iris Technologies 2026). Ships: `PolicyEnvelope` + `TaxonomyValidator` (264-node) + `LifecycleState` + `RedirectTable` + `TypedOfflineCandidate` + `OfflineCandidateScorer` + `InfoRegistry`. **DEFAULT-ON** (Plan 327 Phase 4, 2026-06-25): G1 61 tests; G2a PolicyEnvelope ~0.4ns<50ns; G2b TaxonomyValidator ~170ns<200ns; G3 all-features/default/no-default clean; G4 0 allocs/100 calls (fixed via scratch + clone-instead-of-mem::take); G5 silence-bias strict inequalities. Pure modelless protocol vocabulary — no game/chain/shard IP. Composes with `non_interference_branches` `LifecycleState` when both features on. Private runtime wiring: riir-ai Plan 337 / Guide 160. |
| `phase_rotation_coupling` | `katgpt-core/phase_rotation_coupling` | Phase-Modulated Coupling — norm-preserving subspace rotation gate (Plan 322, Research 305, arXiv:2605.12700 UFO Qiao/Karniadakis/Munirazzaman May 2026). `cos α ⊙ a + sin α ⊙ b` where α comes from a sigmoid projection — the open math hook for norm-preserving NPC affect rotation / crowd-coherent mode transition / chain-committed phase for deterministic replay. **DEFAULT-ON** (Plan 322 Phase 2, 2026-06-25): G1 per-channel Pythagorean drift **5.96e-8**<1e-4 (1677× headroom); G2 0 reversals/100-step sweep (monotone interpolation); G3 D=8 scalar+mix **18.9ns**<50ns + D=8 mix-only **5.0ns**<20ns + D=64 per-channel+mix 355.7ns<1500ns; G4 0 allocs; G6 sigmoid(0)=0.5→cos=sin=1/√2 (softmax would give 1.0). Design pivot: independent Padé cos/sin drifts in cos²+sin²=1 by ~5e-3 (50× G1 budget) — replaced with `phase_safe_cos_sin` (libm sin + Pythagorean sqrt(1−sin²) recovery). Pure modelless. Private fusion guides deferred to riir-ai (HLA runtime) / riir-chain (LatCal committed phase) / katgpt-rs (DEC Hodge mixer). |
| `phase_separation` | `katgpt-core/phase_separation` | Phase Separation Probe — per-entity minimum circular distance on a phase circle, distilled from the Lonely Runner Conjecture (Plan 571, Research 470, arXiv:0710.4495 Barajas & Serra 2007). `phase_separation(i) = min_{j≠i} ‖φ_i − φ_j‖ mod 1 ∈ [0, 0.5]` via three paths: O(N) single entity, O(N²) all-pairs, O(N log N) sort+scan (`phase_separation_sorted`, production). The LRC guarantees (proven N≤7, conjectured beyond) every entity cycles through `phase_separation ≥ 1/N` — a coverage guarantee no existing primitive provides. Two bridge helpers: `from_speeds_and_tick` (raw time-phase, sync-safe) + `from_latent_projection` (latent σ(d·z) bridge, local-only). **DEFAULT-ON** (Plan 571 Phase 1, 2026-08-07): G1–G4 ALL PASS (bench_571) — G1 8 lib tests incl. LRC N≤7 bound confirmation (orbit k=0..420); G2 7947ns @ N=1000 (<10µs) + O(N log N) 12.49× scaling (<20×); G3 1845→1853 (+8, 0 regressions); G4 0 allocs/1000 calls. Think-brain primitive for zone-attention routing / curiosity / coverage scoring. Pure modelless (closed-form modular arithmetic + sigmoid + dot-product). Zero runtime cost unless invoked. See [`.benchmarks/571_phase_separation_goat.md`](../../.benchmarks/571_phase_separation_goat.md). |
| `non_interference_branches` | `katgpt-core/non_interference_branches` | Non-Interference Memory Branches — continual adaptation primitive distilled from RIZZ (Plan 329, Research 310, arXiv:2606.20638 Goel et al. Oxford Jun 2026). Five generic primitives: `BranchBank` + `BranchRouter` + `VerifierGate` + `NonInterferenceProjection` + `BudgetCompiler`. The Super-GOAT fusion of BAKE × CLR × MCGS × Engram × ARG × closure-instrument × Salience into per-NPC continual adaptation without catastrophic forgetting. **DEFAULT-ON** (Plan 329 Phase 3, 2026-06-26): G1 8 orthogonal directions in D=8 (pairwise interference 0.00e0<1e-6; write to b_i does not contaminate b_j stores; 9th direction correctly rejected at 0.3536≥1/√8); G2 route **301.5ns**<1µs (64-branch bank, 3.3× margin); G3 all-feature combos clean; G4 0 allocs/100 calls; G5 `[]` deps. 101/101 unit tests. Composes with `arg_protocol` LifecycleState when both features on. Pure modelless (structural geometric orthogonality, not learned). Private runtime wiring deferred to riir-ai Plan 338. |
| `best_belief` | `katgpt-core/best_belief` | Best-Belief Beta Selector — ε-quantile Beta lower bound for conservative selection (Plan 336, Research 320, RQGM arXiv:2606.26294 Prop. 4). Complements `sample_beta` (Thompson sampling for EXPLORATION) with a conservative EXPLOITATION/SELECTION counterpart. **DEFAULT-ON** (Plan 336 Phase 2 G2-unblock, 2026-06-28): LUT hot path **3.38ns**, G1 3.099e-5<1e-4 vs statrs, G4 0 allocs. **Issue 010 T5 "Report the Floor" comparison: BEATS the MLE floor** in the heteroscedastic regime (variable observation counts — the real-world use case for frozen snapshots/archetype shards with different deployment durations → 15–30% selection regret ↓); ties at uniform n (the monotonicity theorem). Confirms DEFAULT-ON promotion. Pure modelless (closed-form Beta inverse-CDF via LUT). |
| `cognitive_architecture_root` | `katgpt-core/cognitive_architecture_root` (implies `engram`) | Cognitive Architecture Root — whole-architecture BLAKE3 commitment `CognitiveArchitectureRoot([u8; 32])` (Issue 039, 2026-07-04). The anti-cheat / quorum-attested personality freeze-thaw / on-chain NPC avatar portability primitive. Implies `engram` (so `engram` is transitively default-on via this feature — the Plan 299 "default-off" label predates this promotion, see Plan 360 status sync note). **DEFAULT-ON** (Issue 039, 2026-07-04): G1 spec-match 13/13 + bit-flip every input; G1-avalanche min 120/256 avg 126/256 (BLAKE3 ~128, floor 96); G2 `from_parts` 208ns + `verify` 208ns (<500ns); G2-alloc 0/1000; G3 `--all-features` + `--no-default` clean; G4 `size_of == 32`. Pure modelless. Zero runtime cost unless a caller constructs/verifies a root. |
| `ptg_functor_edges` | `katgpt-core/ptg_functor_edges` (implies `closure_instrument`) | PTG × latent_functor Edge composition (Issue 040, 2026-07-04). Adds `FunctorPtg` composite (wraps an unchanged `PrimitiveTransitionGraph` with a parallel `Vec<Option<FunctorEdgeParams>>`) + `apply_functor_edge_into` (zero-alloc sigmoid-gated cosine·direction apply path) + `functor_edge_gate` (diagnostic gate query). Wire-format safe: the inner PTG is byte-identical to a bare PTG (T1 audit found postcard `#[serde(default)]` does NOT work for missing trailing fields — "Hit end of buffer" — so the composite approach is mandatory). Implies `closure_instrument`. **DEFAULT-ON** (Issue 040 T7, 2026-07-04): G1 6/6 sub-checks (high-coherence ≈ state+dir, low-coherence ≈ identity, determinism, threshold gate=0.5, FunctorPtg preserves inner commitment, wire-format byte-identical) + 17 unit tests; G2 `apply_functor_edge_into` **28.5ns** at D=64 (target <200ns, 7× headroom); G2-alloc 0/1000; G3 default + `--all-features` + `--no-default` clean; G4 `size_of::<FunctorEdgeParams> == 44` bytes (no heap indirection); G5/G6 pure modelless (closed-form cosine + sigmoid + SAXPY). |
| `heal_validation` | `katgpt-core/heal_validation` | Heal-Validation Conflict Detector — `HealConflictDetector` trait for healed-state semantic validation (Issue 133, 2026-07-12). The heal-path analog of LDT's `ConflictDetector` (Plan 088): where `ConflictDetector` checks token candidate sets for satisfiability (signature: `marginals`, `pruned_count`, `total_candidates`), this checks healed flat `&[f32]` state (style_weights for shards, emotion axes for HLA) for semantic impossibility (NaN, degenerate blend, anger+calm both >0.7, etc.). The signature is intentionally different — forcing the token-specific signature onto heal validation would abuse its parameters (Interface Segregation Principle). **Passive trait** — zero behavior change unless consumers implement it. **DEFAULT-ON** (Issue 133, 2026-07-12): G1–G6 ALL PASS. Two consumer impls pass GOAT: `ShardConflictDetector` (riir-neuron-db, **30ns**) and `HlaConflictDetector` (riir-games, **2ns**), both <50ns target. Pure modelless (threshold checks). |
| `latent_confounder_audit` | `katgpt-core/latent_confounder_audit` | LatentConfounderAudit — three CD-LAM §III-B diagnostics (zero-transition response `R₀`, shift-invariance response `R_shift`, shortcut leakage `L`) for pre-deployment confounder-purity audit of direction vectors (Issue 194, Research 460, arXiv:2607.09185 CD-LAM). `audit_confounders_into` takes pre-allocated `AuditScratch`; encoder API is `Fn(&[f32], &[f32], &mut [f32])`. **Opt-in (diagnostic)** — G1-G4 ALL PASS modellessly (Bench 194, 2026-07-28): G1 12 unit tests + monotone in confounder coefficient c; G2 **292 ns/call** at HLA d=8 (3.4× under 1µs); G3 default-feature count unchanged (1814 → 1814, +12 with feature on); G4 0 allocs/100 audit calls (TrackingAllocator sentinel-verified). Stays opt-in — promotion requires a concrete consumer (MAG/TILR/Steering/Blend) benchmarking a real-bug-caught gain. The CD-LAM training recipe routes to riir-train. Pure modelless (norm + cosine). Zero runtime cost unless invoked. |
| `transformer_inversion` | `katgpt-core/transformer_inversion` (parent; `grad_policy` adds gradient-guided driver) | SipIt Transformer Inversion — `invert_sequence` recovers discrete input tokens from observed transformer hidden states (Plan 561, Research cross-refs 158/232/244, arXiv:2510.15511 Nikolaou et al. ICLR 2026). Two policies: `RandomPolicy` (uniform-without-replacement) + `GradientGuidedPolicy` (paper Alg 3 — proxy hidden state + finite-difference gradient + periodic vocab projection + random fallback). **Opt-in (research infrastructure)** — Phases 1-4 DONE (2026-07-26), Phase 5 awaiting consumer: G1 3/3 sub-tests + 20 unit tests on toy 2-layer GELU transformer (d=16, \|V\|=32, T=8); G2 random 37→130→1375µs/pos for \|V\| 32→128→512 (linear) + grad-guided 317 vs random 1075 acceptance tests (70.5% reduction, 3.4×); G3 default-feature count unchanged (1814 → 1814); G4 per-call 2/5 allocs, steady-state 10× per-call = no per-trial leak; Phase 4 Theorem 3.2 perturbation guarantee verified (recovery holds below `Δ_π/2` noise, degrades above). Toy cannot validate paper's \|V\|≥32K regime. No consumer wired (grep verified across all 7 repos). Re-evaluate 2026-10-26. Pure modelless (gradient on continuous proxy, no backprop through weights). |
| `canon` · `canon_subspace` · `canon_mask` | `katgpt-canon` crate (depends on katgpt-core + katgpt-spectral) | katgpt-canon — Canonical Intent Space adapter substrate (Proposal 009, Research 459, arXiv:2209.04836 Git Re-Basin + arXiv:2512.05117). `CanonicalIntent { tag, direction }` + `ModelAdapter` trait + 3 adapters: `ProcrustesAdapter` (orthogonal rotation), `SubspaceAdapter` (joint SVD), `MaskAdapter` (lottery-ticket apply). **Opt-in (cross-arch Super-GOAT PERMANENTLY DEMOTED)** — G1/G2/G4 ALL PASS 17/17 (Bench 562, 2026-07-28): Procrustes d=256 **16.17µs** ≤ 50µs post-SIMD (was 29µs; d=2304 diagnostic 1.328ms, NOT gated — O(d²) scaling, setup-time use only); Subspace k=4 d_b=1536 **417ns** ≤ 50µs (carries Bench 423 G5 GO at k∈{2,4}: mean cos +0.87/+0.75 Gemma↔MiniCPM real weights); Mask d=2304 **1.38µs** ≤ 50µs; all three BLAKE3-deterministic + 0 allocs hot path. Cross-arch Super-GOAT headline permanently demoted (Bench 424/425/426/427 — 4 hidden-state methods all failed G6 ≥0.5 cross-arch agreement; modelless path declared exhausted, reopens only on non-hidden-state construction per Proposal 010 draft). Pure modelless (linear algebra + SVD). |
| `event_log_query` | `katgpt-pruners/event_log_query` (implies `event_log`) | EventLog Query Combinator — programmatic-search axis over `EventLog<A>` (Plan 562, Research 461, arxiv 2607.20064 PRO-LONG). `Predicate<A>` enum (EventTypeIs / EventTypeIn / IdRange / IdRangeFrom / And / Or / Not / All / None_ / Custom) + `filter` / `query_window` / `count_where` / `first_where` / `last_where` over the existing Plan 124 `EventLog<A>` — the deterministic, LLM-free analog of "coding agent greps the log." **Opt-in (ship-quality gate met, pending downstream consumer)** — G1–G4 ALL PASS (Bench 564, 2026-07-29): G1 13/13 predicate combinations (including composed And/Or/Not + Custom escape hatch); G2 `filter` **4.99 ns/result-event** (200× under 1µs target), `query_window` **0.46 ns/call** (217× under 100ns target), `first_where`/`last_where` early-exit **4–6 ns/call**; G3 feature-off build clean + existing Plan 124 API unchanged; G4 zero steady-state allocation (lazy iterators, capacity-stability proxy 512→512). Pure modelless (predicate enum + slice iterators, no LLM, no embedding, no training). Zero runtime cost unless invoked. Promotion deferred to Phase 3 — requires a downstream consumer (riir-engine CLR/KARC trajectory search, riir-neuron-db Raven/δ-Mem consolidation, katgpt-pruners MCTS planner) to prove a measurable gain over the no-query baseline. |
| `ane_roofline` | — | ANE-aware roofline cost model — Bryngelson arXiv:2606.22283 (Plan 379, Research 377). Extends `roofline_cost` with ANE-specific axes (2MB working-set cliff, 0.23ms dispatch floor, family-floor capability gate) and per-chip peaks for M1-M5. **DEFAULT-ON** (Plan 379 Phase 2 GOAT, 2026-07-04): G1–G5 ALL PASS — G1 5/5 routing verdicts match Bryngelson ch.11 + cross-chip + family roundtrip, G2 <1µs, G2-alloc 0, G4 struct sizes 48/40/32B. Pure modelless arithmetic. Zero runtime cost unless `ane_estimate` is invoked. |
| `bom_sampling` | `micro_belief`, `simd_sigmoid` | BoMSampler — K-hypothesis single-pass belief sampling (Plan 281, Research 248). Auto-enables `simd_sigmoid` (G3 PASS: K=8 at 1.87× step). **DEFAULT-ON** (Plan 281 T2.4 full, 2026-06-17): G2 PASS +31.49pp proven in riir-ai Plan 314 (MultiThreatArena + MultiHypothesisBoMMinimaxPlanner vs deterministic). Pure modelless. Opt-out via `--no-default-features` if a consumer needs to gate it explicitly. |
| `cce_moderator` | — | CCE — Coarse Correlated Equilibria moderator primitives (Plan 295 + Plan 300, Research 274). **DEFAULT-ON** (Phase 10 absorption). Pure modelless protocol primitives. Zero runtime cost unless invoked. |
| `cross_resolution_transport` | `funcattn` | Cross-Resolution Spectral Transport — asymmetric-basis FUNCATTN for `d_src ≠ d_dst`, enabling train-on-small-deploy-on-large latent transfer (Plan 310, Research 291, arXiv:2605.31559). Implies `funcattn` for solver/C-operator reuse. **DEFAULT-ON** (Plan 310 Phase 4, 2026-06-23): G1 mean cos 0.8944≥0.85, G2-A rank preservation mean cos 0.9300≥0.85 (Super-GOAT headline holds), G3 elbow at k=8, G4 0 allocs/1000 transports. Pure modelless. Zero runtime cost unless invoked. |
| `fourier_continuation` | — | Fourier Continuation for non-periodic latent fields — closed-form polynomial periodic extension so the FFT does not produce Gibbs ringing at boundaries (Plan 323, Research 307 §3 candidate #1). The one modelless FNO primitive the codebase genuinely lacked. **DEFAULT-ON** (Plan 323 Phase 3, 2026-06-25): G1 wrap discontinuity <50% of naive + interior join C1-smooth, G2 8.9µs<50µs, G3 passthrough bit-identical, G4 0 allocs. Pure modelless. |
| `funcattn_structured_basis` | `funcattn` | Principled multi-scale basis constructors for FUNCATTN — DCT-log + Haar-packet (Plan 332, Research 307). **DEFAULT-ON** (Plan 332 Phase 4, 2026-06-26): per-basis GOAT gate PASS on realistic broadband PDE-like signal — DCT-log beats random by +0.3409 cos (captures 200.6%), Haar-packet by +0.1615 cos (95.0%), both clearing G1 (≥+0.05) + G2 (≥50%) thresholds. Pure modelless (deterministic fixed bases, no training). |
| `geometric_product` | — | Channel-wise Clifford Geometric Product — per-point coherence (dot) + structure (wedge) latent interaction (Plan 319, Research 299, arXiv:2601.06793). Hadamard + cyclic shift + subtract, O(D·\|S\|), zero-alloc. **DEFAULT-ON** (2026-06-25, Issue 003 RESOLVED): polynomial Padé [4/4] SiLU eliminates the exp() floor (2.06× speedup at D=64); G1 non-redundancy +17.6/+7.9pp, G2 rotational recovery r=0.90/0.96, G3 0 allocs, G4 D=8 118ns/D=64 525ns. Pure modelless. |
| `hebbian_kernel_memory` | — | Hebbian Kernel Memory — Closed-Form Fact-Storing MLP Construction + MLP Swap (Plan 559, Research 455, arXiv:2607.10034 Garcia et al., Stanford/UB 2026-07-10). Bilinear sketched-K₂ feature map + ridge-whitened readout achieves information-theoretic optimal fact-storage capacity W=Θ(F·log F). **DEFAULT-ON** (Plan 559 Phase 3, 2026-07-25): G1+G2+G3+G4 GOAT gate ALL PASS (bench_559 — γ_min=25.11>0 + 18 unit tests + interpolation err 8.33e-5<1e-3; HLA 97ns/query; 0 allocs/100 calls); G5 Super-GOAT quality axis PASS (Bench 462 riir-neuron-db: Constructed=GD=1.000 edit_score at 2/5/10% edits vs Frozen 0.000). Pure modelless (closed-form Gaussian features + ridge-whitened least squares). Zero runtime cost unless constructed. |
| `hope_capacity` | — | HOPE — Hilbert-Schmidt Capacity Kernel + Optimal Rank-1 Parent (Plan 469, Research 454, arXiv:2607.21366 Mobahi & Bartlett, Google DeepMind 2026-07-24). Closed-form math for scale-invariant capacity metric + optimal rank-1 parent merge via principal eigenvector of rank-2 AᵀA + Dantzig greedy distortion-rate selection. **DEFAULT-ON** (Plan 469 Phase 4 T4.6, 2026-07-24): G1+G2+G3+G4 GOAT gate ALL PASS (bench_469 — 4 correctness invariants; 9 kernels at D=8 mean over 10000×256 calls, all under target by 10-160×; 0 allocs). Pure modelless (erf approximation + rank-2 closed-form eigendecomp). Zero runtime cost unless invoked. |
| `indicator_probe_bank` | — | IndicatorProbeBank — structured N-direction OR-fused cascade (Plan 320, Research 301, arXiv:2606.24251 Zhou et al.). Sigmoid-gated dot-projection onto pre-computed, BLAKE3-committed, freeze/thaw-versioned direction vectors; argmax OR-fusion into one firing label. **DEFAULT-ON** (Plan 320 Phase 5, 2026-06-25): G1 per-indicator AU-ROC 1.000 (all 8 ≥0.85), G2 OR-fusion TPR 1.000/FPR 0.041, G3 cascade 100× FPR reduction at 0pp TPR cost, G4 53.9ns/call (<200ns) + 0 allocs, G5 cluster ARI 1.000 (≥0.9), G6 feature-off clean, G7 wire tamper-evident. Pure modelless read-side primitive. |
| `indicator_similarity` | `indicator_probe_bank` | IndicatorSimilarityMatrix — pairwise cosine structure + greedy block recovery of an IndicatorProbeBank's directions (Plan 320 Phase 2, Research 301 Fig. 6). O(N²·D) construction, O(1) lookup, complete-linkage cluster recovering within-category blocks. **DEFAULT-ON** (Plan 320 Phase 5, 2026-06-25): G5 block recovery ARI 1.000 (≥0.9) on planted 4×2 synthetic bank. Pure modelless read-side primitive. |
| `latent_field_steering` | — | Latent Field Steering — top-down direction-vector injection into mutable latent state (Plan 309, Research 290, CAA + functional emotions). Zero-alloc SIMD SAXPY + sigmoid-falloff localized support (radius/zone). **DEFAULT-ON** (Plan 309 Phase 4, 2026-06-23): G1–G5 ALL PASS — G1 fear-axis 1.50×≥1.30, G2 mean cos 0.9958≥0.95 min cos 0.9667≥0.90, G3 leakage 4.5e-5<0.01, G4 19.2µs<1ms, G5 0 allocs. Caveat: 8% argmax flip at α=0.3 — keep α≤0.3 for hot-path. Pure modelless. |
| `linking_fold_fold` | — | Linking-Fold hot-path correction ONLY — coordinate-wise \|x−c\| fold (paper Eq. 1: \|x\|=x+2·ReLU(−x)) as the deterministic modelless unlinking correction (Plan 410, Research 391, arXiv:2606.31856 Ren & Lim ICML 2026). `fold_projection_into` / `fold_gelu_into`. Closes a gap every sigmoid projection has: monotonicity → provably cannot linearly separate linked manifolds. **DEFAULT-ON** (Plan 410 T4.4 Option C, 2026-07-07): G1–G5 ALL PASS — G1 fold unlinks Hopf link, G2 12.5ns@D=8 Abs / 16.1ns Gelu / 16.8ns@D=64 (all under 50ns/500ns budgets), G3 clean, G4 0 allocs/1000 calls ×4, G5 bit-identical. Hot-path, zero-alloc, `#[inline]`, pure stdlib. Zero runtime cost unless invoked. |
| `local_branch_routing` | — | Post-Candidate Branch Router — distilled from Local Branch Routing (arXiv:2606.25354 Yin et al. June 2026, Plan 377, Research 376). Forward K candidate next-tokens, score each post-candidate hidden state by dot-product onto a frozen direction, commit the argmax (or perturbed-argmax sample with Logistic noise — the sigmoid analog of Gumbel-max). **DEFAULT-ON** (Plan 377 Phase 3, 2026-07-04): G1 22/22 unit tests, G2 route_argmax 51.1ns + route_sampled 69.1ns at K=3 D=64 (<1µs target, 14-20× headroom), G3 K=1 bit-identical, G4 0 allocs/100 calls, G5 modelless ([] deps), G6 sigmoid-not-softmax. PoC +9pp to +26pp quality gain across 5 noise cells. Pure modelless. |
| `mag_mining` | — | MAG — Mining via Activation Geometry (Plan 418, Research 397, arXiv:2607.04222 LeVi/David/Fomin ICML 2026 FAGEN). Unsupervised direction mining from prefix-induced activation shifts using the model's own verdict y_M as the label (no human labels, no GD). The missing acquisition step for the direction-vector ecosystem: today every direction is designer-authored or supervised-extracted; MAG mines them unsupervised. **DEFAULT-ON** (Phase 2 GOAT G1–G6 ALL PASS, 2026-07-09): G1 mine_direction cos=1.000/mine_contrast cos=0.985, G2 LOO acc 0.925@σ=1.5 + 0.810@σ=3.0 (headline kill-it gate PASS), G3 ϵ_Q 0.0/1.0/4.0, G4 MAG Top-1 0.720 vs raw cosine 0.220 (3.3×), G5 zero-alloc (_into variants), G6 mine_direction 10.13µs/transfer_score 0.52µs/recon_error 4.41µs. Pure modelless (mean-difference + cosine + BLAKE3). |
| `manifold_bandit` | — | Manifold Bandits — Latent Task Tree + Hierarchical Thompson Sampler + BayesianFilterArm (Plan 370, Research 370, arXiv:2606.19750 McKenzie et al. UCSD 2026). Modelless inference-time routing primitive distilled from the paper's Bayesian Manifold Curriculum (BMC training loop → riir-train). Three composable parts: (1) LatentTaskTree — frozen, BLAKE3-committable hierarchical clustering; (2) top-down Beta posterior Thompson descent with bottom-up EVIDENCE-pooling Empirical Bayes; (3) BayesianFilterArm — per-arm non-stationary belief via predict-update drift filter. **DEFAULT-ON** (Plan 370 Phase 2, 2026-07-03): G1 structural advantage ratio 0.723≤0.8, G3 non-stationarity recovery ratio 0.350≤0.5, G4 sample 408ns<500ns + observe 26ns<300ns + 0 allocs, G5 bit-reproducible. G2 FAIL is plan-level expectation error (hierarchical correctly exploits more: +10.5% reward; diversity is curriculum-learning-specific). Pure modelless (closed-form Beta posteriors + deterministic Empirical Bayes). |
| `manifold_erasure` | — | MANCE — Manifold-Aware Concept Erasure (Plan 426, Research 409, arXiv:2607.03973). Local tangent + spectral weighting + trust-bounded erasure: k-NN natural neighbors → local tangent SVD → σ^α-weighted direction projection → ε·r_i trust region. Pure modelless linear algebra. Consumes a pre-computed erasure direction (probe is a consumer concern — MAG/CNA/EmotionDirections). Gated on `subspace_phase_gate` for Plan 301 `thin_svd_into`. **DEFAULT-ON** (2026-07-11): G1–G6 GOAT gate ALL PASS. See `.benchmarks/426_manifold_erasure_goat.md`. Pure modelless. Zero runtime cost unless invoked. |
| `mean_field_regime` | — | Mean-Field Crowd Oscillation Regime Classifier — (κ, κ_a, Q) order-parameter aggregator + Hopf boundary detector + four-way regime taxonomy (Plan 371, Research 371, arXiv:2606.30366 Zheng/Miller/Fiete MIT Jun 2026). Three modelless primitives: MeanFieldOverlap, hopf_boundary/static_boundary/saddle_strength (closed-form 2×2 Jacobian eigenvalue check extending Plan 301 to complex-eigenvalue Hopf phase transitions), RegimeClassifier. **DEFAULT-ON** (Plan 371 Phase 6, 2026-07-03): G1 25/25 (100%) + 4/4 distinct regimes, G2 9.79µs<15µs + 0ns hopf/classify, G3 710/710+682/682 clean, G4 0 allocs, G5 bit-identical. Pure modelless (closed-form f32: Jacobian eigenvalues + sigmoid gates). |
| `product_key_memory` | — | Product Key Memory — O(√N) factored retrieval open primitive (Plan 408, Research 387, arXiv:2601.00671 Lample et al. 2019 §2.2 / Zhao & Jones 2026 distillation). Split query → score two √N codebooks → top-k of k² Cartesian product. Retrieval stack: Raven O(1) / Engram O(1)-hash / δ-Mem O(r) / PKM O(√N) — four distinct complexity classes. Leaf-clean: zero deps, pure stdlib. Modelless: FwPKM's GD half forbidden (constraint #1), replaced by shipped δ-rule (Plan 053). **DEFAULT-ON** (Plan 408 Phase 3 GOAT, 2026-07-07): G1 latency 1670× speedup (PKM p50 17.5µs vs O(N) brute-force p50 29.2ms at SQRT_N=1000/N=10⁶), G2 top-k Jaccard 1.0000 vs brute-force, G3 IDW centroid-ness PASS, G4 0 allocs/1000 steady-state `query_into` calls. Pure modelless. Zero runtime cost unless a caller constructs `ProductKeyMemory`. |
| `qmc_sampling` | — | Quasi-Monte Carlo uniform sources (Lattice/Stratified/Sobol) for correlated-but-marginally-exact parallel sampling (Plan 367, Research 367, arXiv:2607.01179 QuasiMoTTo). Drop-in replacement for i.i.d. `rng.uniform()` in K-rollout paths: each rollout is marginally Unif[0,1) exact, but the batch covers output space more evenly → 25–47% fewer rollouts for matched pass@k. Zero-dep (Sobol direction numbers computed at construction from GF(2) primitive polynomials; Owen scramble via digital XOR shift). **DEFAULT-ON** (Plan 367 Phase 5, 2026-07-03): G1 chi-square marginal exactness PASS all 3 sources, G2 pass@k Lattice 50% sample reduction (K_qmc=8 vs K_iid=16), G3 K=1 bit-identical to i.i.d. (0/10000 mismatches), G4 0 allocs/100 calls, G5 per-rollout 25-34ns (<1000ns), G6 all-features/no-default clean. Pure modelless (closed-form arithmetic coding + QMC points). |
| `region_subspace_steering` | `subspace_steering` | Region-Conditioned Subspace Field — MFA local-geometry steering (Plan 416, Research 396, arXiv:2602.02464 Shafran et al. "From Directions to Regions"). The region-conditioned generalization of Plan 412: K regions, each with a centroid μ_k and a local R-dim factor-analyzer subspace W_k. Two-mode steering: centroid interpolation + local subspace offset. Per-region sigmoid membership gates (reformulated from softmax responsibilities per AGENTS.md). At degenerate K=1,μ=0,W=I bit-identical to Plan 412. Implies `subspace_steering` (K=1 parity gate reference). **DEFAULT-ON** (Plan 416 Phase 4, 2026-07-09): G1+G2+G3+G4+G5 ALL PASS (0 mismatches / 800 comparisons at degenerate limit). Pure modelless (closed-form: sigmoid + SAXPY + R×R Gauss-Jordan inverse). |
| `renoise_ce` | — | Renoise-CE Self-Verifier — perturb completed state + re-resolve through same operator + measure drift as verifier-free correctness score (Plan 406, Research 369, arxiv 2606.29150). Third orthogonal self-eval signal alongside CLR (claim-vote) and CoE (trajectory-shape). Operator-agnostic trait over any state→state map. **DEFAULT-ON** (2026-07-06): G1 @ 50% coverage renoise-CE=1.000 vs plurality=0.000 (100pp gap); G2 @ 70% coverage fusion=1.000 vs clr-alone=0.695 (+30.5pp, 6× the +5pp target); G3/G6 default 1296/1296 + all-features + no-default clean; G4 0 allocs with fixed-array State; G5 36µs<100µs at D=8 k=8 (2.7× headroom). Pure stdlib + fastrand. NOT a UQ primitive (raw ranking signal). |
| `salience_tri_gate` | — | Salience Tri-Gate Primitive — 3-way per-tick emit gate (Plan 303, Research 281). **DEFAULT-ON** (Phase 10 absorption, 2026-07-04). Pure modelless. Zero runtime cost unless invoked. |
| `set_attention` | — | Cross-Datapoint Set Attention — sigmoid-gated, permutation-equivariant cross-entity refinement kernel (Plan 354, Research 354, arXiv:2106.02584 Kossen et al. NeurIPS 2021). Distilled from Non-Parametric Transformers: the inference-time operator only (training of Q/K/V via BERT-style masking stays in riir-train). Substrate-agnostic: operates on `&[f32]`, produces `&mut [f32]`. Sigmoid gates (NEVER softmax per AGENTS.md §2) — each pair independently ∈ (0,1). **DEFAULT-ON** (Plan 354 Phase 2 + Plan 355 G6/G7/G9, 2026-07-01): G1 permutation equivariance bit-exact (1e-6 tol), G2 identity-floor meaningfulness on 2-cluster, G3 latency 21.96µs at N=64 (2200× headroom under 50ms tick), G4 0 allocs/100 calls, G5 sigmoid-not-softmax lonely-query correctness. riir-ai runtime: G6 fusion cosine sim <0.95, G7 crowd stability <5% drift over 100×2000 ticks, G9 production latency 75.7µs mean/tick at 100 NPCs. **G8 collective inference CLOSED by `clr_weighted_set_attention` (Plan 570, 2026-08-06)** — the CLR-weighted sibling uses per-entity reliability weights to convert averaging into amplification (+8.7pp identification, 3.88× aggregate amplification over plain SA). Validated selling point: crowd coherence + CLR-amplified collective inference. Pure modelless (closed-form sigmoid gates). |
| `clr_weighted_set_attention` | `set_attention` | CLR-Amplified Set Attention (Plan 570, Research 469, Issue 575). Reliability-weighted sibling of `set_sigmoid_attention_into`: `output_i = h_i + (γ/Σ r_j)·Σ_j α_ij·r_j·(v_j−h_i)`. Uniform `r_j=1` reduces bit-identically to plain SA (G1 special case). The companion `clr_reliability_scores` computes `r_j = (mean_m sigmoid(h_j·dir_m))^M` — the CLR headline formula (Plan 284), distilled from Wang/Plotkin PNAS 2025 feedback-payoff theory (Research 469). The reliability weighting concentrates the aggregate toward high-reliability entities, converting plain averaging (which dilutes signal — the original G8 failure) into amplification. **DEFAULT-ON** (Plan 570 Phase 3, 2026-08-06): G1 uniform≡plain SA bit-identical (dense+topk), G2 latency ratio 1.00× (≤2× target — one extra multiply per peer), G4 0 allocs/100 calls, G8a CLR reliability identification 17.6% vs plain SA 8.9% (Δ+8.7pp ≥5pp target), G8b CLR-weighted aggregate amplification 3.88× (≥2× target) on the Issue 575 N=64 crowd threat-detection fixture. Pure modelless. Zero runtime cost unless invoked. |
| `spherical_steering` | — | Spherical Steering — single-target geodesic Slerp rotation `sin((1−t)θ)/sin θ · ĥ + sin(tθ)/sin θ · μ_T` of a latent vector toward a unit-norm target direction on S^{d-1}, with sigmoid-translated vMF confidence gate (Plan 405, Research 382, arXiv:2602.08169 You/Deng/Chen ICML 2026). Sibling to Plan 322's 2-subspace phase rotation — same norm-preservation thesis, different parameterization. Use case: pull a drifted vector back toward a committed archetype direction. §3.5 modelless Path 3 (closed-form trig + sigmoid; no training). **DEFAULT-ON** (Plan 405 Phase 2, 2026-07-06): G1–G6 ALL PASS — G1 norm-preservation max rel drift D=8 8.22e-7/D=64 5.04e-7 (budget 1e-4) + Slerp unit-modulus identity drift 8.35e-7, G2 0 NaN/0 OOB across 16000 vmf-gate combos + antipodal returns AntipodalDegenerate cleanly, G3 D=8 full 37.6ns<100ns + D=64 full 58.9ns<1500ns (25× headroom), G4 0 allocs, G5 double-Slerp drift correction monotone, G6 sigmoid fingerprint exact. Pure modelless (closed-form trig + sigmoid). Zero runtime cost unless invoked. |
| `ssmax_temperature` | — | SSMax — length-aware log-N attention temperature (Plan 411, Research 392, arxiv 2607.01538 Gollapudi et al. *Drowning in Documents at Million Token Scale*). Multiplicative pre-attention logit rescaling `s̃ = s_L · log(N) · s` that cancels the (N−1) denominator growth in the attention dilution bound. Default `s_L = 1.0` is truly modelless (zero training, zero new params). Composes with sigmoid parallax + standard SDPA; does NOT apply to `funcattn`. **DEFAULT-ON** (Plan 411 Phase 5, 2026-07-07): G1+G2+G3+G4+G5 ALL PASS. Zero runtime cost unless invoked (`ParallaxConfig.ssmax` defaults `None`; `ssmax_none_is_bit_identical_to_base` test verifies zero default-behavior change). Pure modelless (closed-form `s_L.log(N).s` rescale). |
| `subspace_steering` | `latent_field_steering` | Subspace Steering Field — k-dim manifold steering (Plan 412, Research 393, arXiv:2606.25234 Goodfire BSF). The k-dim generalization of Plan 309: an orthonormal block `{u_1..u_k}` + per-axis strengths `{α_1..α_k}`, math `s' = s + Σ_j α_j · u_j`. At K=1 bit-identical to Plan 309; at K≥2 enables manifold walking. Pure modelless consumer of pre-discovered blocks (Gram-Schmidt orthonormalization, NOT Newton-Schulz which diverges on non-square K<D). Implies `latent_field_steering` (K=1 parity gate reference). **DEFAULT-ON** (Plan 412 Phase 5, 2026-07-08): G1+G3+G4+G5 ALL PASS. At K=1 bit-identical to Plan 309 (0 mismatches / 800 comparisons). Pure modelless (SAXPY + Gram-Schmidt). |
| `temp_loss_fingerprint` | — | TEMP — Perturbed-Loss-Vector Diversity Fingerprint (Plan 341, Research 323, arXiv:2606.26797 Jin et al. ICML 2026). Modelless diversity selector: given two committed snapshots S_0, S_1, extrapolate K checkpoints along v = S_1 − S_0, compute per-candidate short-prefix loss vectors, and select the K-subset with maximal Lipschitz-bound spread — gradient-diversity ranking without gradients. Composes with `ac_prefix::ConditionalLogprob`, HLA surprise, RavenSlotLossKernel. **DEFAULT-ON** (Plan 341 Phase 2, 2026-06-29): G1 15.44×≥2× (MIN pairwise bound metric, Gaussian mixture fixture), G2 Kendall tau 0.9839≥0.85 at N=32 vs N=256 (paper Fig. 6 modelless analog), G3 perturbed_loss_vector 2.46µs<5µs + select_diverse_subset 130µs<1ms + 0 allocs hot path, G4 bit-identical determinism, G5 134/134 each-feature clean. riir-neuron-db Plan 005 G2' integration gain +0.1672 (modelless). Pure modelless. |
| `tilr_invariant_subspace` | — | TILR — Trajectory-Invariant Latent Refinement (Plan 425, Research 408, arXiv:2606.29164). Alignment-gated subspace-projected correction: projects a contrastive direction onto a frozen SVD basis, modulates step size by γ = ‖Πd‖/‖d‖ so γ→0 bit-recovers the input (strict no-harm). Pure linear algebra (flat `&[f32]` + SIMD dot), zero crate deps. Consumes a pre-computed SVD basis (Plan 301); does not compute it. **DEFAULT-ON**: G1–G4 GOAT gate ALL PASS. See `.benchmarks/425_tilr_goat.md`. Pure modelless. Zero runtime cost unless invoked. |
| `tropical_algebra` | `dec_operators` | Tropical (max, +) semiring primitive + DEC wrappers (Plan 337, Research 321, Smets Ch. 3 §3.5). `tropical_matvec` (max-plus analog of `simd_matvec`) + `tropical_exterior_derivative` / `tropical_codifferential` / `tropical_line_integral` (max-plus analogs of shipped DEC d/δ/line_integral). **DEFAULT-ON** (Plan 337 Phase 3, 2026-06-28): G1 non-redundancy 3/3 PASS (S1 DEC cochain top-3 sym-diff=2, S2 HLA pairs rho=+0.3468, S3 path rho=+0.6991) → Super-GOAT quality tier. G2 perf PASS at gate dims after NEON specialization (D=64 0.96×, D=128 1.03× vs simd_matvec). G3 clean, G4 0 allocs, G5 pure modelless (max +). Result noisy at D=64 boundary. |
| `tucker_factorization` | `subspace_phase_gate` | Tucker / HOSVD N-mode tensor factorization (Plan 326, Research 307 §3 candidate #3, arXiv:2511.05963 §6.1) — the N-mode generalization of `thin_svd_into`: decomposes X ∈ R^(I₀×…×I_{N-1}) into S ×₀ A^(0) ×₁ A^(1) × … via N SVDs of the mode-n unfoldings. Deterministic, modelless (closed-form, no training). N≤4 modes; `SVD_MAX_RANK=16` limits per-mode unfolding min-dim. **DEFAULT-ON** (Plan 326 Phase 3, 2026-06-25): G1 rank-(2,2,2) recovery rel err 4.1e-8<1e-4, G2 (8,8,8) mean 71µs<500µs, G3 full-rank max err 1.0e-6<1e-4, G4 0 allocs/100 calls. Pure modelless (closed-form: N SVDs + tensor-times-matrix contractions). |
| `velocity_field_ensemble` | — | Velocity-Field Ensemble — algebraic combination of P frozen pre-trained velocity fields into a regression-optimal combined drift b̂(x) = Σ η_i b_i(x), η solved once from N data pairs via `linalg::ridge_solve` (Plan 376, Research 375, arXiv:2602.20070 Coeurdoux et al. ICML 2026 SPIGM). Same P×P Cholesky math as KARC; the contribution is the basis construction (P model outputs as features). Includes the optimal-diffusion SDE integrator (paper Algorithm 1) as a decoupled utility. **DEFAULT-ON** (Plan 376 Phase 3, 2026-07-04): G1 9/9 unit tests (η recovery <1e-4), G2 3/3 cross-domain metrics (3.5× MSE reduction vs single-best in related-sources regime; honest PoC with unrelated-sources null), G3 0 allocs + all-features/no-default clean, G4 fit_into 6.27µs<50µs + eval_into 21ns<200ns + eval_batch(1000) 20µs<5ms. Pure modelless (closed-form ridge solve, no training, no softmax). |
| `zone_density_routing` | `dep:papaya` | Zone Density Routing — modelless per-zone physical compute scheduler (Plan 351, Research 350 — Treuille Continuum Crowds + Fokker-Planck-on-cochains). `zone_density_classify` (mobility = fast_sigmoid(−β·(ρ−ρ₀)) → tier + cache_key) + `schedule_outer_first` (ascending-density sort) + `ZoneDensityCache<V>` (papaya LRU with tier/drift/TTL invalidation). Sibling to Plan 305 cognitive gating, NOT replacement. No UQ claim (mobility is deterministic weight, not probability). **DEFAULT-ON** (Plan 351 Phase 3, 2026-06-29): G5a +19.4% entropy gain, G5b 99.1% compute saved, G5c 0 stale reads. Pure modelless. |
| `lattice_operad` | `katgpt-pruners/lattice_operad` | LatticeOperad — canonical AND/OR pruner expression composition (Plan 252 Phase 2, Research 220). `PrunerExpr` enum (Atom/And/Or) + `canonicalize()` + `eval()` + operadic composition of pruner expressions + distributive lattice word problem solver. The composition half of the Cubical Category family (with `interval_pruner` Phase 1 + `cubical_nerve` Phase 3, both opt-in). **DEFAULT-ON** (Phase 12 absorption, 2026-07-04 — module moved to katgpt-pruners; root forwards): GOAT T25-T27 validated (Plan 252 Phase 4 / Plan 278 cross-validation) — pruner composition overhead acceptable for 2/4/8 pruners vs ad-hoc AND. Pure modelless (lattice algebra, no training). Zero runtime cost unless a caller constructs a `PrunerExpr`. |
| `cp_hopfield` | `katgpt-core/cp_hopfield` | CP^(d-1) Symmetric-Space Hopfield — Top-Eigenvector Recall (Plan 567, Research 466, Galitski *High-Capacity Generalized Hopfield Networks* alphaXiv 2607.hopfield-networks 2026-07-31). Associative-memory recall on CP^(d-1) = SU(d)/U(d-1) instead of the sphere: memory kernel `K_i` is a d×d Hermitian spiked random matrix, recall = top eigenvector (BBP-protected). α_c grows with d (0.05→0.62→2.41 for d=2→3→4). **Opt-in — STAYS OPT-IN** (Bench 567): G1+G2+G3+G4+G7 PASS; **G5 PASS narrowly** (Plan 276 unblock with task-aligned memories: flips 347→3, tracking 0.000→1.000 — but snap hyperparameter has no principled setting); **G6 FAIL** (CP² recall worse than cosine ANN on KG capacity — associative recall destroys angular precision; projected-cosine diagnostic helps 3–9× as denoising but not actionable on production retrieve paths). Promotion requires G5+G6+G7 all PASS. Pure modelless (closed-form Lie-algebraic construction + Rayleigh-quotient ascent; memories load from frozen snapshot, freeze/thaw Path 1). Zero runtime cost unless constructed. |
| `gemma4_inference` | `katgpt-core/gemma4_inference` (forwards `katgpt-types/gemma4_inference`) | Gemma 4 inference config — `Gemma4LayerType` enum + `Config::gemma4_12b()` preset + `ModelArchitecture::Gemma4` + `MultiLayerKVCache::new_with_per_layer_kv_dim` (Issue 577). Infrastructure — gates the loader + forward path in riir-engine (Plan 318 Phase F 4B/2B MLA-MoE student needs a real 12B baseline). **Opt-in** — leaf-clean type addition; no own GOAT gate (infrastructure). Zero runtime cost unless a consumer constructs the config. |
| `kimi_k3_backward` | `kimi_k3_loader`, `katgpt-attn/mla_backward`, `katgpt-attn/kda_backward`, `katgpt-transformer/moe_backward` | Kimi-K3 full-model analytic backward (BPTT) — training-time reference consumed by riir-train, NOT a production inference path (modelless-by-mandate exception). Composes MLA + MoE + KDA per-primitive backward + model-level composition (attn-res + RMSNorm + dense SiTU FFN + LM head). Gradient checks: single-layer PASS (0.26%), no-attn-res PASS (1.38%), all-KDA PASS (8.34%), full-model with MLA FAIL at 36% (MLA backward validated in isolation at 1.77% — composition issue under investigation). Also includes gradient checkpointing + training-suitable weight init. **Opt-in** — heavy dep chain + training-only. |
| `ica_lens` | `katgpt-spectral/ica_lens` (implies `hla_eigenbasis_recovery`) | ICA Lens — FastICA non-Gaussian direction mining + ERF diagnostic (Plan 475, arxiv 2606.11722). The missing third corner of the direction-acquisition triangle: designer-authored (R290) / verdict-supervised (MAG R397) / **unsupervised-statistical (this)**. Hyvärinen 1999 deflationary FastICA with 3 stability recipes (row-normalization, p95-LIM acceptance, adaptive refit). **DEFAULT-ON** (2026-08-11): all 5 GOAT gates PASS — G1 439µs ≤ 1ms (offline corpus fit), G2(a) 4.33× + G2(b) 6.26× kurtosis ratio vs PCA, G3 85→107 tests, G4 0 bytes steady-state, G5 bit-identical determinism. Pure modelless (linear algebra + fixed-point iteration; no SAE dictionary learning). Zero runtime cost unless invoked. See [`.benchmarks/475_ica_lens_fastica_goat.md`](../../.benchmarks/475_ica_lens_fastica_goat.md). |
| `similarity_inference` | `katgpt-core/similarity_inference` | Similarity Inference — endogenous correlation device from joint-action history (Plan 526, Research 471, arxiv 2608.03958). Infers `ω ∈ (0,1)` endogenously (Bayesian posterior) + switches Nash→CCE when `ω` crosses a payoff-derived threshold. The missing acquisition step for the shipped `CceLp<N,A>` (Plan 295) which uses an *exogenous* designer-set correlation device. Includes the indirect-inference subset (Phase 7 scoped Super-GOAT-capability: zero-shot cooperation from third-party observation). **OPT-IN** (demoted from DEFAULT-ON 2026-09-04 per the quarterly goat-audit, riir-ai Issue 867 T1.3 — 24 days default-on with zero consumers workspace-wide; Plan 526 Phase 6 promotion 2026-08-11): G1–G8 ALL PASS (Bench 579) — G1 rel_err <1e-5, G2 shared 100% vs random 0% cooperation, G5 indirect shared 100% vs random 0%, G6 24 ns/update @ 1000 entities, G7 Brier 0.0012 vs floor 0.146 (119× better), G8 PD threshold 0.500±0.001. Verdict GOAT (not Super-GOAT — equilibrium concept covered by CCE; novelty is the mechanism). Pure modelless (Bayesian posterior + sigmoid + best-response; no training). Zero runtime cost unless invoked. See [`.benchmarks/579_similarity_inference_goat.md`](../../.benchmarks/579_similarity_inference_goat.md). |
| `channel_simd_align` | — | Channel SIMD Alignment — cache-line-padded weight storage for vectorized matvec (Plan 227 Phase 5, Gemma 4 QAT). The 6th and final phase of QAT Infusion: contiguous per-channel weights so SIMD load paths get aligned data. **DEFAULT-ON** (2026-08-11): all 5 GOAT gates PASS (Bench 580) — G1 aligned=unaligned within 1e-3, G2 contiguous + 0% padding, G3 8/8 tests, G4 alloc-free (`matvec_into`), G5 **84.9% / 86.7%** throughput improvement in release mode (far exceeds ≥5% gate). Debug-mode showed only 1.02× — uninitialized-SIMD noise (debug doesn't auto-vectorize); the test is `#[cfg(debug_assertions)]` structure-verify + `#[cfg(not(debug_assertions))]` throughput-enforce. Pure modelless (memory layout optimization). Zero runtime cost unless the aligned path is used. See [`.benchmarks/580_channel_simd_align_release_goat.md`](../../.benchmarks/580_channel_simd_align_release_goat.md). |
| `plot` | `dep:plotters` | Benchmark SVG plot output (Issue 355 Phase 2a). Only `src/plot.rs` + `src/main.rs` benchmark runner use plotters. **DEFAULT-ON** to preserve historical behavior; downstream consumers (riir-ai) set `default-features = false` to drop the plotters dep. Pure utility (no inference logic). |
| `flashmemory_sparse` | `katgpt-attn/flashmemory_sparse` (implies `mla_attention` + `dash_attn`) | FlashMemory periodic sigmoid-threshold sparse attention for MLA (Issue 584, Research 436, arXiv:2606.09079). τ-step refresh + `sigmoid ≥ 0.5` block selection + centroids from `MlaKVCache` latent KV, on the VortexFlow substrate. **Opt-in** — G1 PASS on real Kimi-K3-0.40B weights (cos ≥ 0.9566, ~74% KV reduction, bench_021), G2 PASS 1.8× decode at 64K on 4090 + 256K measured (Bench 671, riir-ai, 2026-08-14), G3 PASS (169 tests), G4 PASS (0 allocs, bench_022), G5 PARTIAL ~50% synthetic at 4K-256K / 74% real ≤4K (data-dependent; the paper's 90% needs real long-context attention patterns). Honest NIAH negative: single-layer retrieval not testable (bench_023). Stays opt-in: serving-regime-specific (256K Bonsai on 4090) + promotion waits for a trained indexer (riir-train Plan 337). Zero runtime cost unless invoked. |
| `trained_indexer` | `katgpt-attn/trained_indexer` (implies `flashmemory_sparse`) | FlashMemory trained dual-encoder indexer — two tiny MLPs (Q-Indexer + K-Indexer, 2114 params @ d_h=64) replacing the modelless centroid-dot scorer (riir-train Plan 337). **Opt-in, NEVER default-on** (requires training — modelless-first mandate; the modelless `FlashMemorySelector` is the production default). Bench 025: pipeline works end-to-end, Kimi-K3-0.40B attention too uniform for meaningful quality (needs Bonsai-27B on 4090). Bench 026: synthetic convergence proof (Adam + bias init → 100% accuracy, +50pp over modelless, gradient check 0.000075) — the non-convergence is a data-quality issue, not an algorithm bug. |
| `full` | all above (excludes `stepcode`, `sp_kv`, `shard_kv`, `peira_distill`, `dirichlet_energy`, `data_probe`, `rmsd_distill`, `safe_bandit`, `stiff_anomaly`, `state_source`, `nexus_elo`, `skill_opt`, `proof_cert`, `mech_attribution`, `ega_attn`, `event_log`, `event_log_query`, `spec_cost_model`, `spechop`, `rt_turbo`, `tf_loop`, `plasma_path`, `parallel_probe`, `parallax_attn`, `sigmoid_margin`, `moa_inference`, `dual_gram_pca`, `roofline_cost`, `leo_all_goals`, `dual_leo`, `stability_metrics`, `asymmetric_kv`, `kog_cpu_fusion`, `caddtree_budget`, `sense_composition`, `bake_precision`, `induced_cwm`, `induced_cwm_ismcts`, `induced_cwm_tournament`, `interpolation_geometry`, `grapem_rodrigues`, `position_group_action`, `grape_ap_vector`, `grape_joint_lift`) | Enable all features |
| `cluster_lm_head` | `katgpt-forward/cluster_lm_head` | Clustered LM head — two-stage vocab head: stage 1 scores k≈V/128 clusters (❈h, centroid❉), stage 2 runs the exact head over the admissible set only (Plan 574 + Issues 657/658/661/666). `ClusterStop::Admissible` (Cauchy–Schwarz radius bound) + `ClusterStop::TopK` budget stops; D² seeding (the degenerate-strided-init fix, Bench 658) is the construction default. **Opt-in** — G2b recall 0.675 → **1.0000** (Bench 658); **8.3–9.2×** on structured heads after the cluster-contiguous row permutation (Issue 666: 2.2× → 8.3×); wave-parallel stage 2 measured a WASH (Issue 661 — the bottleneck is locality, not work); G3 honestly FAILS on the uniform-random control (0.08×) — cost is regime-dependent, promotion measured **PERMANENT NEGATIVE** 2026-08-17 ([Bench 688](../../../riir-ai/.benchmarks/688_clustered_lm_head_real_checkpoint_harness.md): Gemma 2 2B tied wte, 123 real `after_final_norm` probes — admissible active 99.95%, the uniform-random regime; packed 0.44×; recall 1.0000 exactness held — the bound is sound, real geometry is simply inclusive). Pure modelless (k-means + bounds). |
| `bigram_markov` | `katgpt-speculative/bigram_markov` | Bigram Markov head — modelless sequential drafter primitive (Issue 659, Research 316 §3.5 path 2): deterministic CSR top-m successor table built from corpus bigram counts (packed-u64 sort + two-pointer, `(count desc, next asc)` top-m, bit-identical rebuilds), zero-alloc marginal emission (`BigramMarginalBuffer`, O(steps × top_m) touched-reset sparse writes), greedy-chain conditioning, zero-row fallback for unseen prevs (the seam skips prob ≤ 0 — unseen proposes nothing). Drops into `build_dd_tree(marginals, config)`. The Metal-viable alternative drafter for Bonsai where the 6-layer DSpark forward does not amortize at batch-1 (Bench 656 mode 2). **Opt-in** — primitive gate PASS (Bench 663): **181 ns/call (23 ns/step)** at Bonsai scale on M3 release (~5,600× under a 6-layer drafter forward per step), 17 MB worst-case table vs 268 MB low-rank, G1 bit-identical rebuilds + brute-force-pinned, G4 alloc-free steady state. Consumer gate (acceptance rate at equal draft depth + wall-clock on Metal AND 4090 vs the Bonsai target) deferred to the riir-ai Bonsai consumer (Plan 528). |
| `switch_cost` | `katgpt-core/switch_cost` | SwitchCostTable — directed pairwise skill-entropy switch-cost primitive (Issue 663, Research 484, TTT-Discover): `ske(a,b) = ln(Z(a∪b)/Z(a))` over per-skill Bernoulli success counters — asymmetric by construction (the "how much does knowing/doing a cost b" question), u32 counters commute exactly (record-order independent, replay bit-identical), zero allocs. For task-switch scheduling / router curriculum ordering. **Opt-in** — GOAT PASS (Bench 660): G1 formula hand-computed fixture 3.0/0.667 (tol 1e-6), directionality pinned (gap >1.0), determinism forward-vs-reverse replay bit-identical `to_bits()`. Pure modelless (log of exact integer sums). |
| `freedom_selection` | `katgpt-core/freedom_selection` (implies `renoise_ce`) | Freedom Selection — extension-count (freedom-of-function) best-of-K criterion (Issue 665, Bennett arXiv:2608.05423 / Research 486): among candidates within a loss gate of the winner, prefer the one opening an unoccupied output region — freedom of function provably orders generalization. Ships `log_freedom` (Σ log(2^a − 1) over a declared finite partition), `freedom_gain`, `LossGate` (absolute/relative tolerance), `ExtensionOccupancy` (O(1) update) + the renoise-CE selection sibling `best_of_n_freedom`. **Opt-in** — PoC gate PASS (Research 486 §PoC Addendum): parent-hit **0.7075** vs 0.4453 (min-loss) vs 0.5156 (random-near-best confound control), 64/64 seeds; **73% of the gain = the freedom signal**. Stays opt-in until a production consumer A/B (the `switch_cost` / Issue-663 precedent). |
| `effective_degree` | `katgpt-core/effective_degree` (implies `karc_forecaster`) | Effective Degree — modelless function-space simplicity metric (Issue 668): ED = Σ|c_k|·k over Chebyshev coefficients fitted along data-pair interpolation polynomials. **Opt-in, diagnostic-only** — consumer verdict SCOPE-LIMITED (riir-neuron-db Issue 602 / Bench 484): out-correlates `output_flatness` **12.6×** pooled (0.598 vs 0.047, control 0.032) and beat it 4/4 scenarios, but the **sign inverts between grains** (pooled +0.598, within-regime all four negative — a Simpson reversal on 3 disjoint seed sets), so no threshold wires it as a one-sided freeze gate. Surviving promotion axis: cross-regime triage (the KARC regime-mismatch probe, Research 488 §4). Not UQ-bearing (the Issue 010 floor rule does not apply). |
| `ignition_schedule` | `katgpt-core/ignition_schedule` | IgnitionSchedule — closed-form logistic ignition curves `z(t) = K·σ(ζt − ln((K−z₀)/z₀))`, the patience law `t* = ln(1/ε)/ζ` (Thm 8, capacity-free), per-curve inverse `time_to_reach`, and ζ-descending ignition ordering `order_by_ignition_into` (Issue 459 T5, arXiv:2608.13335 Thms 5–8 / riir-train Research 422 §3.5 — the Path-0 modelless half). One `exp`, no iteration; RK4-ODE-anchored to the GLV dynamics `ż = ζ·z·(1 − z/K)`. **Opt-in** — GOAT G1–G4 ALL PASS (Bench 666): G1 monotone ranking (t* strictly decreasing over ζ ∈ [0.1, 4.0] at ε ∈ {1e-2..1e-4}; ordering == observed threshold-crossing order), G2 **3.88 ns/call** release (12.9× headroom), G3 default 1897/0/6 exact + feature-on 1911/0/6, G4 0 allocs. Promotion to default requires the consumer pilot win (riir-clippy selection patience scaled by `ignition_time` vs fixed — the Issue 026 starved-pool negative is the measured anchor). |
| `spectral_pencil` | `katgpt-core/spectral_pencil` (implies `hebbian_kernel_memory` + `karc_forecaster`) | The affine matrix pencil scalar gate `f(x) = λk(A₀ + Σ xᵢAᵢ)` (arXiv:2608.08003 "The Spectral Neuron", Issue 676 / Research 495) — shape-by-construction (k=1 concave, k=d convex, PSD ⇒ Loewner-monotone per feature), Weyl global influence bounds, exact Hellmann–Feynman attribution `∂f/∂xᵢ = vᵀAᵢv`, canonical gauge commitment bytes, invertible monotone warp, + the γk ≥ ½ eigengap-guaranteed init (Lemma 2). **Opt-in** — GOAT G1–G4 PASS ([Bench 671](../../.benchmarks/671_spectral_pencil_goat.md)): tridiag eval 748 ns–3.71 µs (d 8–32, the per-tick path at 10k NPC × 20 Hz), dense 3.95–166.7 µs (spawn/GM path), `count_below` **51 ns** exact. Consumers: `riir_game_sdk::spectral` facade (riir-ai Issue 736 B2) + the mmorpg spectral fear-gate / `FusionArm::Spectral` (mmorpg Bench 028). Full narrative: [`.docs/02_inference/spectral_pencil.md`](../02_inference/spectral_pencil.md) |
| `signed_coupling_dynamics` | `katgpt-core/signed_coupling_dynamics` | Signed-coupling opinion dynamics — Glauber update on a **signed** social graph (`h_i = β⁺ΣJ⁺s + β⁻ΣJ⁻s + β₀Σ|J|s + g_i`, `P(s_i=+1) = σ(h_i)`) plus the three crowd order parameters (arXiv:2608.16578 "Physics of Agents", Issue 680 / [Research 497](../../.research/497_Signed_Coupling_Opinion_Phase_Forecast.md)). Ships `SignedGraph` (CSR signed adjacency, symmetric + directed), `Couplings::at_social_temperature` (the one-scalar crowd dial: high T = apathy, low T = mob), the 5-coupling truth-asymmetry sibling kernel, and the reducers `net_opinion` / `crowd_conviction` (**mean(s²) — nothing in the stack shipped a mean-square crowd reducer**) / `SusceptibilityAccumulator` (`χ = N·Var_t(|n|)`, whose sweep peak locates the critical social temperature — offline). **Opt-in** — GOAT G1–G4 ALL PASS ([Bench 672](../../.benchmarks/672_signed_coupling_goat.md)): G1a/G1b indifference/polarization/consensus on 3 graph families, deterministic **and** seeded-stochastic; G1c the paper's β⁺>β⁻ consensus bias as a mechanism; G1d interior χ peak over the paper's 41-point sweep; G2 **~1.8 ns/edge** flat N=32→1024 at 0.97–1.02× the naive three-accumulator form (median pairwise ratio, 9 interleaved rounds); G4 0 allocs. Promotion waits on a production consumer (the CLR precedent; Research 497 §7 lists swarm emotions as the natural first). Not UQ-bearing as shipped — `σ(h_i)` is a dynamics rule, and any prediction-quality claim owes the Issue 010 conformal floor. **Grep warning:** `crowd_conviction` (order parameter, `mean(s²)`) ≠ Sheaf-ADMM `conviction` (per-agent consensus resistance). **`verdict_margin`** (added 2026-08-23, Plan 545 T1) — the one-snapshot crowd-manipulability forecast on this substrate: CLR-reliability-weighted verdict margin over binary verdictification, measured ρ(margin₀, verdict-flip-frac) = −0.65 on N=200 signed ring crowds (riir-ai Issue 745 / [Research 499](../../.research/499_Jagged_Judges_Epistemic_Stability.md)); reachable as `signed_coupling::verdict_margin`, consumed by riir-games `social_pressure` (riir-ai Plan 545). |
| `gaussianity_probe` | `katgpt-core/gaussianity_probe` | Sketched Gaussianity Probe — multi-direction projection-normality for embedding populations (Issue 681 / [Research 498](../../.research/498_LeVLJEPA_NonContrastive_CrossModal_SIGReg.md); LeVLJEPA arXiv:2607.00784 SIGReg distilled from training loss to inference-time diagnostic). Cramér–Wold sketch: project n×d onto 16 fixed directions (4 coordinate-axis anchors + 12 BLAKE3-derived Rademacher), KS-vs-fitted-Gaussian per direction, sigmoid aggregate over the n-aware Kolmogorov min-p. Catches the bimodal / heavy-tail / discrete marginals that pass every second-moment metric — on a μ=3σ bimodal fixture the probe scores 2.4e-23 while `effective_rank` reads 83% "healthy" (the non-redundancy pin, [Bench 673](../../.benchmarks/673_sketched_gaussianity_goat.md); also 4.2× faster than erank). Consumers: band_conditioner Fisher-z guard, riir-ai #743 edge_lora monitor, riir-neuron-db freeze-gate advisory. Zero-alloc after `GaussianityScratch::new`. Opt-in — diagnostic; promotion waits on a consumer. |
| `risk_control_exit` | `katgpt-core/risk_control_exit` | Modelless dual-threshold compute-exit — "Conformal Thinking" (Plan 575 / [Research 494](../../.research/494_Conformal_Thinking_Dual_Threshold_Risk_Control_Exit.md), arXiv:2602.03814): upper stop-when-confident threshold + squeezed-sigmoid lower stop-when-not-progressing schedule `λ−(t) = σ(c(ωt − sB), l, u)`, the four bounded losses (Eq. 8–11), offline UCB/Hoeffding calibration (`Risk̂ + sqrt(ln(1/δ)/2n) ≤ ε`), two-step decoupled selection with efficiency-loss argmin, App. C `p_i ≥ p_c` disarm tripwire. **Opt-in** — GOAT ALL PASS ([Bench 681](../../.benchmarks/681_risk_control_exit_goat.md)): G1 realized exit-FP-risk ≤ ε on **40/40** resplits while naive no-correction violates **7/40** at n=40 (the paper's Fig. 4 shape); G2 dual compute **0.417** vs single-threshold 0.609 vs fixed-budget 1.000 at matched realized risk (Fig. 6 shape — the gap grows with stuck share); G4 0 allocs, **~4–5 ns/exit** (3.90/5.18 across two release runs). UQ-bearing — the Report-the-Floor rule instantiated as the naive-calibration contrast + exit-floor family (CRPS/Winkler are undefined for a decision rule). Promotion waits on Phase 3 consumers (MCTS termination, Plan 304 fusion `GainCostLoopHalter`, Bebop Issue 023 re-gate). |
| `distributional_steering` | `katgpt-core/distributional_steering` | Mean-field population steering toward a **measure-defined target** `μ* ∝ e^{λΨ} p` via Feynman-Kac weights (Plan 577 / [Research 505](../../.research/505_Mean_Field_Distributional_Steering.md), arXiv:2608.08770): first-variation reward table (`LinearReward`/`MomentReward`/`MmdReward` — Research 505's Table-2 MMD **sign slip corrected**, finite-difference-pinned), FK stepper with damped-Picard `Ψ̇` correction (kernel matrix built once per step), the weighted empirical measure `μ̂ = Σ wᵢ δ_{Xᵢ}` as the converging object, residual/systematic resampling (documented NOT for persistent agents), BoM static-tilt adapter (`bom_sampling` + `distributional_steering`). **Opt-in — G1 FAIL-partial** ([Bench 682](../../.benchmarks/682_distributional_steering_goat.md)): λ\*=5 targeting reproduced on both noise schedules (clean V-curves), λ\*=10 on 1/2, and the FK-vs-gradient **separation claim NOT reproduced** in the 1-D Langevin regime; G2 9.0 µs/particle/step @ N=1000 (exact O(N²) MMD is structurally above a 1 µs gate; FK/grad 3.91×). Consumer-critical: Picard damping must scale `α = min(1, 2/λ)`; weights-only degenerates to ESS→1 by λ≈7.5 without resampling. Reopen paths: a diffusion-sampler-shaped harness (prerequisite for the riir-ai crowd plan) + approximate kernel features. |
| `prover_selection` | `katgpt-core` (default) | Prover-selection statistics (Issue 692 / [Research 509](../../.research/509_Rewarding_Progress_PAV_Prover_Advantage.md); "Rewarding Progress" arXiv:2410.08146 distilled): the D/Al complementarity selector — `distinguishability` + `alignment` estimators + Theorem 3.1's predicted-gain bound `γ·(D+Al)` with sigmoid-gated exposure — the `first_pit` changepoint kernel (consumed by riir-clippy's PAV data curation via the riir-train Plan 356 A1 twin), and the K⭻ interior-optimum law (`k_star` + `bok_advantage`). **DEFAULT-ON since 2026-08-27** ([Bench 684](../../.benchmarks/684_prover_selection_goat.md), GOAT G1–G5 PASS — the `rating` precedent: pure modelless arithmetic, no dep surface, zero-cost-unless-invoked; the Cargo feature remains as an inert alias). Ranking by the complementarity bound beats strength-ranking on the controlled PAV harness at every paper α (16 seeds); default lib suite 1978/0/7 with the module's 27 tests default-included. |

Default features: **73** in katgpt-core (`crates/katgpt-core/Cargo.toml`) + **135** in the root crate (`Cargo.toml`). The authoritative source for the exact list is the `default = [...]` array in each `Cargo.toml` (hand-maintaining a prose enumeration of 180+ features proved unmaintainable — at the time of replacement the list was missing 41 katgpt-core defaults + 61 root defaults, and had 21 stale entries including function names that were never feature flags). See `opt_in_features.md` §69 for the DEFAULT-ON cross-reference of substrate/utility primitives not documented elsewhere in the catalog. All defaults are GOAT-proved modelless primitives (Plans 051–575). Production best perf + accuracy.

## Quick Start

```bash
cargo test --quiet --workspace --all-features   # Run all 740+ tests
cargo run --release                             # Run benchmark suite (includes Leviathan verification)
cargo run --example sudoku_01_9x9 --features sudoku           # Sudoku streaming solver
cargo run --example sudoku_02_speculative --features sudoku   # DDTree pruning demo
cargo run --example sudoku_03_tui --features sudoku           # TUI visualization
cargo run --example core_01_validator --features validator     # SynPruner + DDTree pipeline
cargo run --example core_02_raven                             # Raven RSM demo
cargo run --example core_03_ppot --features ppot              # PPoT resampling demo
cargo run --example core_04_prefill                           # PFlash prefill demo
cargo run --example bandit_01_basic --features bandit         # Bandit basics
cargo run --example bomber_01_arena --features bomber         # Bomberman arena
cargo run --example bomber_09_rubric_tournament --features ropd_rubric,g_zero,bomber  # Bomber rubric tournament (Plan 076)
cargo run --example monopoly_01_arena --features monopoly     # Monopoly arena
cargo run --example fft_01_arena --features fft               # FFT Tactics arena
cargo run --example fft_02_rubric_tournament --features ropd_rubric,g_zero,fft  # FFT rubric tournament (Plan 076)
cargo run --example go_06_bench --features go --release       # Go benchmark suite
```

## Config Presets

| Config | vocab | embd | heads | layers | mlp | Purpose |
|--------|-------|------|-------|--------|-----|---------|
| `micro` | 27 | 16 | 4 | 1 | 64 | Default benchmark target |
| `micro_lora` | 27 | 16 | 4 | 1 | 64 | Micro + LoRA adapter support |
| `draft` | 27 | 4 | 2 | 1 | 16 | Tiny draft model |
| `game` | 27 | 16 | 4 | 1 | 64 | Game domain preset (domain_latent) |
| `bpe` | 4096 | 32 | 4 | 1 | 128 | BPE Rust code model |
| `bpe_draft` | 4096 | 8 | 2 | 1 | 32 | BPE draft model |
| `small_target` | 4096 | 64 | 4 | 4 | 256 | Multi-layer target |
| `gqa_draft` | 4096 | 64 | 8 | 4 | 256 | GQA draft (n_kv_head=2) |
| `micro_dllm` | 27 | 16 | 4 | 1 | 64 | D2F discrete diffusion (bidirectional) |
| `game_go` | 85 | 32 | 4 | 1 | 128 | Go board 9×9 + action (~16K params) |
| `qwen_deltanet` | 151936 | 2048 | 16 | 4 | 8192 | QwenDeltaNet hybrid DeltaNet/Attention (kv_heads=8, head_dim=128, Plan 182) |
| `gemma2_2b` | 256000 | 2304 | 8 | 26 | 9216 | Gemma 2 2B architecture (kv_heads=4, head_dim=256) |

### ManifoldPruner Code Example (Plan 234, opt-in)

```rust
// Before: Binary pruning (misses boundary tokens)
if pruner.is_valid(depth, token, prefix) {
    tree.expand(token);
}

// After: ManifoldPruner captures boundary tokens
if pruner.manifold_score(depth, token, prefix) > threshold {
    tree.expand(token); // threshold < 0.5 captures boundary tokens
}
```

> **Note:** G1 FAIL — `sigmoid(x) > 0.5 ⟺ x > 0`, so at default 0.5 cutoff this is identical to binary pruning. The Gaussian kernel (G2 PASS) remains valuable for ranking.

## Key Design Principles

1. **Zero allocations on hot paths** — all buffers pre-allocated in `SpeculativeContext` and `ForwardContext`
2. **Feature-gated modularity** — domain code (sudoku, validator) never pollutes core
3. **Trait-based strategy** — `ConstraintPruner`, `SpeculativeVerifier`, `PrefillScorer`, `ScreeningPruner` for swappable behavior
4. **SOLID module decomposition** — each file < 1024 lines, single responsibility
5. **`mod.rs` for index only**, minimal `main.rs`/`lib.rs`
6. **Unsafe only in verified hot-path kernels** with `get_unchecked` + `#[inline(always)]` + SIMD intrinsics (`core::arch` NEON/AVX2)

## Related Documentation

Docs are grouped into topic folders (no number prefix) — see [`.docs/README.md`](../README.md)
for the full index with fusion maps. Quick map:

| Group | Docs | Topic |
|---|---|---|
| [`orientation/`](../01_orientation/) | `overview.md` | Overview & reference card (this file) |
| [`orientation/`](../01_orientation/) | `architecture.md` | Architecture details (forward pass, routers, LoRA) |
| [`orientation/`](../01_orientation/) | `paper_feature_comparison.md` | Paper feature comparison |
| [`inference/`](../02_inference/) | `speculative_decoding.md` | Speculative decoding deep-dive |
| [`inference/`](../02_inference/) | `spechop.md` | SpecHop architecture |
| [`inference/`](../02_inference/) | `kv_compression.md` | KV compression alternatives |
| [`inference/`](../02_inference/) | `mtp_threshold.md` | MTP threshold guide (Plan 055) |
| [`inference/`](../02_inference/) | `progressive_mcgs.md` | Progressive MCGS graph search |
| [`memory/`](../03_memory/) | `raven_rsm.md` · `product_key_memory.md` · `engram.md` · `micro_belief.md` · `sense_composition.md` · `sleep_consolidation.md` | Memory primitives |
| [`calibration/`](../04_calibration/) | `cce_moderator.md` · `causal_head_importance.md` · `faithfulness_probe.md` · `salience_tri_gate.md` · `universality_class_escape.md` | Calibration, probes, gates |
| [`adaptation/`](../05_adaptation/) | `model_adaptation.md` · `lucebox_techniques.md` · `peira_distillation.md` | Model adaptation & distillation |
| [`game_arenas/`](../06_game_arenas/) | `sudoku.md` · `heuristic_learning.md` · `bomber_arena.md` · `monopoly_fsm.md` · `fft_arena.md` · `go_arena.md` · `hl_arena_detail.md` · `open_ended_evolution.md` · `bomber_lora_ab.md` | HL game arenas |
| [`validator/`](../07_validator/) | `constraint_validator.md` · `percepta.md` | Constraint validator + SynPruner, transformer-VM |
| [`performance/`](../08_performance/) | `engineering.md` | Performance engineering & benchmarks |
| [`feature_catalog/`](../09_feature_catalog/) | `opt_in_features.md` · `negative_results.md` | Opt-in features, negative results |
| [`audits/`](../10_audits/) | `loser_sweep_audit.md` · `claim_rubric_audit.md` · `cross_repo_consolidation_audit.md` | One-off audits |