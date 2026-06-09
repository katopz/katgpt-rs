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
- 740+ tests passing (111 test files), zero clippy warnings, 111 examples across 24 groups
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
- Shard Embedding: Johnson-Lindenstrauss random orthogonal projection [f32;64]→[f32;8] for O(1) cosine similarity shard lookup (Plan 230)
- NFCoT FlowScore Drafter: Inference-time normalizing flow density scoring for speculative candidates, zero training (Plan 229)
- Union Bound Confidence: Additive branch confidence via Boole's inequality (Plan 231, default-ON, GOAT 6/6)
- PathwayTracker: Intrinsic pathway stability detection (Plan 231, default-ON, GOAT 7/7)
- FederationComposer: Explicit pruning with residual early termination (Plan 231, default-ON, GOAT 7/7)

## Module Structure

```
crates/
  katgpt-core/    Shared types + SIMD kernels (multi-crate reuse):
    types.rs        Config (all presets + with_overrides + validate), Rng, HlaMode, AttentionMode (Causal/Bidirectional/BlockCausal/SpKv/SpKvQuant/DashAttn), ModelArchitecture (Generic/Gemma2/Llama/QwenDeltaNet), WeightDtype (F32/F16/BF16), InferenceOverrides, InferenceResult, DashAttnConfig, DeltaRoutingConfig, DeltaRoutingMode, ConvergenceSelector, LoopMode, HybridPattern, SdpaOutputGate, ResidualGate, PlanningDecision, ConfiguratorContext, DataGate, GateDecision, ProposerTask, TaskType, kv_dim, softmax, softmax_scaled, rmsnorm, rmsnorm_with_gamma, rmsnorm_with_gamma_eps, gegelu, gegelu_tanh, matmul, matmul_relu, sparse_matmul, sample_token, LoraAdapter, LoraPair, DomainLatent
    simd.rs         SimdLevel (Scalar/Neon/Avx2), simd_level(), simd_dot_f32, simd_dot_f16_f32, simd_fma_row, simd_outer_product_acc, simd_matvec, simd_matmul_rows, simd_matmul_rows_parallel, simd_matmul_relu_rows, simd_matmul_f16_f32_rows, simd_matmul_f16_f32_rows_parallel, simd_sparse_dot_f32, simd_sparse_matmul_rows, simd_scale_inplace, simd_fused_decay_write, simd_scale_mul_inplace, simd_exp_inplace, maxsim_score, maxsim_score_packed
    lib.rs          Feature gates: tiled_attention, coda_fusion, parallax_attn, leo_all_goals, dual_leo, questbench, tf_loop, plasma_path, peira_distill, dirichlet_energy, spectral_hierarchy, sigmoid_margin, dual_gram_pca, roofline_cost, domain_latent, sr2am_configurator, data_gate, sparse_mlp, modal_spec, mux_pruner, and_or_dtree, rim_slots, wall_attention
    traits.rs       ConstraintPruner, ScreeningPruner, GameState, StateHeuristic, RolloutPolicy, SpeculativeGenerator, NoPruner, NoScreeningPruner, BinaryScreeningPruner, RandomRolloutPolicy, ActionSpaceLog (Plan 107 Phase 0, consolidated from both crates)
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
    shard_embedding.rs  JL random orthogonal projection [f32;64]→[f32;8] — ShardEmbedding with cosine similarity (Plan 230)
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
  ⚛ behind --features peira_distill   (opt-in)
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
| `kog_cpu_fusion` | — | Kog AI monokernel CPU fusion — RMSNorm gamma folding + QKV interleaving (Plan 160, Research 139, opt-in) |
| `flashar_anchor` | `dllm` | FlashAR strided anchor-then-fill D2F decoding (Plan 166 T11, opt-in) |
| `flashar_consensus` | `tri_mode`, `plasma_path` | FlashAR consensus tri-mode with ternary thermal paths (Plan 166, default-on) |
| `budget_adaptation` | — | Compression-adaptive decode budget (Plan 167, default-on) |
| `ilc_distill` | — | ILC iterative latent clustering distillation — synonym-aware DDTree pruning (opt-in) |
| `hydra_budget` | — | Hydra-aware adaptive layer budget — emergent self-repair layer skipping (Plan 165, default-on) |
| `gepa_reflective` | `bandit` | GEPA-D reflective config evolution — Pareto bandit config evolution (Plan 164, default-on) |
| `phrase_boost` | — | PhraseBoost context trie phrase boosting for DDTree (Plan 164, default-on) |
| `shard_kv` | `spectral_quant`, `turboquant` | ShardKV asymmetric K/V compression — undo RoPE + PCA K path, Hadamard + K-means V path (Plan 147, opt-in) |
| `sleep_consolidation` | `lt2_looped`, `gdn2_attention` | Sleep Consolidation — offline recursive memory consolidation at KV eviction into GDN2 fast weights (Plan 154, default-on, GOAT 14/14) |
| `spectral_hierarchy` | `katgpt-core/spectral_hierarchy` | Spectral hierarchy diagnostic — eigenspace alignment, Haar wavelets, Cauchy interlacing for KG extraction validation (Plan 156, default-on, GOAT) |
| `dual_gram_pca` | `katgpt-core/dual_gram_pca` | Dual-Gram PCA routing for short-sequence calibration (Plan 159, default-on, GOAT) |
| `roofline_cost` | `katgpt-core/roofline_cost` | Roofline cost model for GPU operator runtime prediction — compute/memory/launch bottleneck estimation (Plan 159, default-on, GOAT) |
| `peira_distill` | `katgpt-core/peira_distill`, `bandit` | PEIRA inter-view regressor alignment — collapse-free modelless distillation via EMA covariance (Plan 153, opt-in) |
| `parallax_attn` | `tiled_attention`, `newton_schulz`, `katgpt-core/parallax_attn` | Parallax parameterized local linear attention — streaming covariance correction (Plan 135, opt-in) |
| `freq_bandit` | `bandit` | FreqBandit — oscillatory spectral bandit for cyclic pattern detection to adaptive speculative decode (Plan 189, default-on, GOAT 7/7 G189=GAIN) |
| `belief_drafter` | `katgpt-core/belief_drafter`, `papaya` | NextLat Belief-State Speculative Drafter — lightweight 3-layer residual MLP recursive hidden-state prediction for variable-length self-speculative decoding + LatentTransitionCache + BeliefRankPruner (Plan 217, default-on, GOAT 43 tests + 7 benchmarks) |
| `bfcf_lfu_shard` | `bfcf_tree`, `papaya` | BFCF × LFU × Sharding — region-level LFU cache with frequency-aware sharding, batch processing, NeuronShard compound keys, emotion-aware eviction, KG triple transitions (Plan 218, default-on, GOAT 44 tests + 10 benchmarks) |
| `caddtree_budget` | `spec_cost_model` | CaDDTree — Cost-Aware Adaptive DDTree Budget Selection (Plan 219, 7 GOAT tests, opt-in) |
| `manifold_pruner` | — | ManifoldPruner — ManifoldE point-to-manifold soft validity scoring + kernel-tricked relevance for ScreeningPruner (Plan 234, opt-in, GOAT G1 FAIL) |
| `sense_composition` | `katgpt-core/sense_composition` | KG Latent Octree NPC sense modules — ternary bit-plane projection, GM override, hot-swap, bandit feedback (Plan 221, opt-in) |
| `shard_embedding` | — | (always-on) JL random orthogonal projection [f32;64]→[f32;8] for O(1) cosine similarity shard lookup (Plan 230) |
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
| `substrate_gate` | `katgpt-core/substrate_gate` | SubstrateGate — inference-time routing via substrate conditions (Plan 216, opt-in) |
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
| `full` | all above (excludes `stepcode`, `sp_kv`, `shard_kv`, `peira_distill`, `dirichlet_energy`, `data_probe`, `rmsd_distill`, `safe_bandit`, `stiff_anomaly`, `state_source`, `nexus_elo`, `skill_opt`, `proof_cert`, `mech_attribution`, `ega_attn`, `event_log`, `spec_cost_model`, `spechop`, `rt_turbo`, `tf_loop`, `plasma_path`, `parallel_probe`, `parallax_attn`, `sigmoid_margin`, `moa_inference`, `dual_gram_pca`, `roofline_cost`, `leo_all_goals`, `dual_leo`, `stability_metrics`, `asymmetric_kv`, `kog_cpu_fusion`, `caddtree_budget`, `sense_composition`, `bake_precision`) | Enable all features |

Default features: `sparse_mlp`, `domain_latent`, `ppot`, `bandit`, `bandit_top_p`, `bt_rank`, `spectral_quant`, `hybrid_oct_pq`, `elf_sde`, `cna_steering`, `deep_manifold`, `federation`, `tes_loop`, `lattice_deduction`, `delta_routing`, `stability_metrics`, `mls_aggregate`, `gdn2_attention`, `dash_attn`, `dreamer`, `lt2_looped`, `dmax_spd`, `eqr_convergence`, `subterranean`, `sr2am_configurator`, `data_gate`, `plasma_path`, `parallel_probe`, `tf_loop`, `leo_all_goals`, `dual_leo`, `sigmoid_margin`, `moa_inference`, `sleep_consolidation`, `spectral_hierarchy`, `dual_gram_pca`, `roofline_cost`, `newton_schulz`, `river_valley`, `peira_distill`, `kog_cpu_fusion`, `gepa_reflective`, `phrase_boost`, `hydra_budget`, `flashar_consensus`, `budget_adaptation`, `ilc_distill`, `thinking_prune`, `rim_slots`, `thinking_cot`, `freq_bandit`, `spec_reconciliation`, `trust_region_spec`, `curvature_alloc`, `directional_credit`, `kv_share`, `nds_proxy`, `wealth_pruner`, `speculative_generator`, `kvarn`, `and_or_dtree`, `belief_drafter`, `bfcf_lfu_shard`, `slod`, `schema_centroid`, `union_bound_confidence`, `pathway_tracker`, `federation_composer`, `llmexec_guard`, `outlier_guard`, `segment_checkpoint`, `self_distilling_bandit`, `precision_aware_draft`, `static_cal_tables`, `targeted_precision`, `egcs`, `reward_mem`, `symbolic_distill`, `concept_grounding`, `reward_calibrator`, `decision_explain`, `collapse_aware_thinking`, `ilc_distill`, `kog_cpu_fusion` (~80+ default features — production best perf + accuracy, all GOAT-proved, Plans 051–237).

## Quick Start

```bash
cargo test --quiet --workspace --all-features   # Run all 740+ tests
cargo run --release                             # Run benchmark suite (includes Leviathan verification)
cargo run --example hello_py2rs                                # BPE + bidirectional prefill demo
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

| # | Document | Topic |
|---|----------|-------|
| 01 | `01_overview.md` | Overview & reference card (this file) |
| 02 | `02_architecture.md` | Architecture details (forward pass, routers, LoRA) |
| 03 | `03_speculative_decoding.md` | Speculative decoding deep-dive |
| 04 | `04_performance.md` | Performance engineering & benchmarks |
| 05 | `05_sudoku.md` | Sudoku solvers |
| 06 | `06_validator.md` | Constraint validator + SynPruner |
| 07 | `07_adaptation.md` | Model adaptation (bidirectional prefill, LoRA switching, sparse MLP, domain latent) |
| 08 | `08_lucebox_techniques.md` | LuceBox techniques |
| 09 | `09_heuristic-learning.md` | Heuristic learning, bandit, HL arena |
| 10 | `10_bomber_arena.md` | Bomberman HL arena (Plan 033) |
| 11 | `11_monopoly_fsm.md` | Monopoly FSM arena (Plan 035) |
| 12 | `12_fft_arena.md` | FFT Tactics Arena (Plan 053) |
| 13 | `13_mtp_threshold_guide.md` | MTP threshold guide (Plan 055) |
| 14 | `14_go_arena.md` | Go Arena (Plan 065) |
| 15 | `15_paper_feature_comparison.md` | Paper feature comparison |
| 16 | `16_spechop_architecture.md` | SpecHop architecture |
| 17 | `17_peira_distillation.md` | PEIRA distillation |
| 18 | `18_sleep_consolidation.md` | Sleep consolidation |
| 19 | `19_kv_compression.md` | KV compression alternatives |
| 20 | `20_negative_results.md` | Negative results |
| 21 | `21_opt_in_features.md` | Opt-in features |
| 22 | `22_percepta.md` | Percepta full detail |
| 23 | `23_hl_arena_detail.md` | HL & Arena detail |
| 24 | `24_sense_composition.md` | **NPC Sense Composition** (Plans 221/230/235/236/237) |