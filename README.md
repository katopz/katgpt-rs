# KatGPT-RS

A **GOAT-proved** neuro-symbolic micro-Transformer with speculative decoding, constraint pruning, and **310+ feature flags (130+ default-on, all GOAT-proved)** — built in Rust. Pure algorithms, zero side effects, MIT licensed.

Inspired by [Andrej Karpathy's microgpt](https://karpathy.github.io/2026/02/12/microgpt/).

<img width="580" height="385" alt="tactical_09_fog_tui" src="https://github.com/user-attachments/assets/57bdc3e1-1c3e-4843-b428-a43070f8ac36" />

## 🚀 Key Results

| Result | Number | Feature |
|--------|--------|---------|
| **TTFT Speedup** | **29×** (X16 compression) | MUX-Latent zero-training context compression |
| **KV Memory Reduction** | **93.8%** | MUX superposition fusion |
| **Prefill Seq Reduction** | **21×**, 100% NIAH retrieval | PFlash block-sparse prefill |
| **KV Rotation FMAs** | **64× fewer**, best MSE | Hybrid OCT+PQ codec |
| **RMSNorm Speedup** | **2.4×** | Kog CPU fusion kernel |
| **Sudoku Compression** | **7,079×** on Inkala's Hardest | Path-aware ConstraintPruner |
| **Bomber HL Score** | **+177** vs Random −55 | Adaptive intelligence arena proof |
| **NFSP/MCTS Duality** | **75%** vs MCTS 8% | Bandit-guided backward→forward search |
| **BoM Belief Sampling** | **+31.49pp** arena win rate (K=8 @ 1.87× step) | Single-pass K-hypothesis belief sampling |
| **Self-Advantage Gate** | **18× forward-pass reduction** (paper claim) | Dead-compute detector via pre/post log-ratio |
| **Temporal Derivative** | **4/4 fusion gates PASS** (HLA, δ-Mem, collapse, curiosity) | Dual fast/slow EMA surprise signal |
| **Triggered Injection** | **50% skips @ 0.63% quality delta** | Sigmoid-thresholded inject/skip hot-path gate |

## 🏗️ Architecture

Matching the talos-vs-macbook reference model:

| Parameter | Value |
|-----------|-------|
| `vocab_size` | 27 (a–z + BOS) |
| `block_size` | 16 |
| `n_embd` | 16 |
| `n_head` | 4 |
| `mlp_hidden` | 64 (4×) |
| `n_layer` | 1 |
| `temperature` | 0.5 |
| `ModelArchitecture` | `NanoGpt`, `QwenDeltaNet` |
| `AttentionMode` | `Standard`, `SpKvQuant`, `DashAttn` |
| `WeightDtype` | `F32`, `F16`, `BF16` |

### Core Pipeline

```
LLM drafts logits → ConstraintPruner filters invalid → DDTree builds valid-only tree → Target verifies
```

### Key Traits

```rust
// From katgpt-core/src/traits.rs (signatures abbreviated)
pub trait ConstraintPruner: Send + Sync {
    fn is_valid(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool;
    fn batch_is_valid(&self, depth: usize, tokens: &[usize], parent_tokens: &[usize], out: &mut [bool]);
    fn propagate(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) { }
    fn manifold_score(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 { 0.0 }
    fn constraint_vector(&self, depth: usize, parent_tokens: &[usize]) -> Vec<f32> { vec![] }
}

pub trait ScreeningPruner: Send + Sync {
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32;
}

pub trait SpeculativeGenerator {
    type Condition;
    type Output;
    type Error;
    fn generate(&mut self, condition: &Self::Condition, rng: &mut fastrand::Rng) -> Result<Vec<Self::Output>, Self::Error>;
    fn generate_batch(&mut self, conditions: &[Self::Condition], rng: &mut fastrand::Rng) -> Result<Vec<Vec<Self::Output>>, Self::Error>;
}
```

Additional core traits in `katgpt-core/src/traits.rs`: `DominoPruner`, `CompletionHorizon`, `CollapseDetector`, `GameState`, `StateHeuristic`, `RolloutPolicy`, `LeoHead`, `AllGoalsUpdate`, `DualLeoMixer`, `AutocurriculumSampler`, `GenerativeConstraintPruner`, `QGradientOracle`, `PartialScorer`, `ProblemMutator`, `BestBuddyAligner`. Plus `DataGate` in `types.rs`. See [`crates/katgpt-core/src/traits.rs`](crates/katgpt-core/src/traits.rs) for full signatures.

### Routing & Conditioning

- **Prompt Router** — `KeywordRouter` scores prompt against domain keywords, `ExpertRegistry` selects `ScreeningPruner` + LoRA. `InferenceBackend` trait + `CpuBackend` for backend abstraction.
- **TriggerGate** — Adaptive tier promotion: CPU → GPU → ANE based on workload complexity.
- **Embedding Router** — Three-tier fallback: embedding search → domain classify → keyword (local).
- **Bidirectional Prefill** — Prompt tokens attend to ALL other prompt tokens (no causal mask during prefill).
- **Modality LoRA Switching** — `reader_lora` active during prefill, `writer_lora` active during decode. Reference swap, zero data movement.
- **PPoT** — Logit-parameterized CPU resampling on failure. Zero overhead on success path.

## 🔄 E2E Inference Flow — Default GOAT Stack

The default production stack has **130+ GOAT-proved default-on features** (310+ total flags), but they don't all run on every token. The architecture uses **layered gating** — most features are bandit-driven, Option-gated, or compile-time-only.

```mermaid
flowchart TD
    subgraph HOT["🔴 Always-On Hot Path — 12 features per token"]
        KOG["kog_cpu_fusion\nFused RMSNorm+QKV kernel"]
        SPARSE["sparse_mlp\nTwELL sparse matmul"]
        DELTA["delta_routing\nBlock-boundary delta accumulate"]
        MLS["mls_aggregate\nMulti-layer residual sum"]
        DOMAIN["domain_latent\nMid-layer K/V inject"]
        PPOT["ppot\nCPU resampling"]
        SPECTRAL["spectral_quant + hybrid_oct_pq\nKV cache storage format"]
        KVARNS["kvarn + kv_share\nVariance-norm KV + Q-K=V sharing"]
        ATTNS["gdn2_attention + lt2_looped\nO(1) decode recurrent attention"]
        ELF["elf_sde\nDDTree noise injection"]
    end

    subgraph GATED["🟡 Conditional — ~30 features, 1 check each"]
        BANDIT["Bandit-driven arm select\nbandit, bandit_top_p, freq_bandit\nsr2am, curvature_alloc, wealth_pruner\nrosetta, directional_credit, self_distilling"]
        OPTION["Option-gated\nhydra_budget, cna_steering\nkurtosis_gate, domino_correction"]
        THINK["Thinking mode only\nthinking_cot, chain_fold\nthinking_prune, parallel_probe"]
        SPEC["Speculative pipeline\nbt_rank, lodestar, best_buddies\ntrust_region_spec, corr_budget\nbelief_drafter, bfcf_tree"]
    end

    subgraph OFFLINE["🔵 Offline — ~8 features, not in forward pass"]
        DIAG["Training/diagnostics\nnewton_schulz, river_valley\nspectral_hierarchy, roofline_cost\nsigmoid_margin, stability_metrics"]
        BG["Background\nsleep_consolidation\ndreamer"]
    end

    HOT --> GATED
    HOT -.->|"post-token"| BG
    GATED -.->|"offline"| DIAG
    GATED -.->|"between sessions"| BG
```

### 🔴 Always-On Hot Path (12 Features)

These execute unconditionally on every token — they replace kernels, formats, or accumulate state:

| Feature | What | Why Always-On |
|---------|------|---------------|
| **`sparse_mlp`** | Skip dead ReLU in w2 matmul | Replaces dense matmul kernel |
| **`kog_cpu_fusion`** | RMSNorm gamma folding + QKV interleaving | Fused kernel replacement |
| **`delta_routing`** | Cross-layer residual delta routing at block boundary | Accumulates per-layer, routes at block edge |
| **`mls_aggregate`** | Average last K layer residuals before LM head | Structural blend into final logits |
| **`domain_latent`** | Mid-layer K/V injection | `Option`-gated inject at `n_layer/2` |
| **`spectral_quant`** | Calibrated eigenbasis + water-fill KV codec | Storage format, not conditional |
| **`hybrid_oct_pq`** | OCT triplet + PQ 2D Givens KV compression | Replaces quantization codec |
| **`kvarn`** | Variance-normalized KV cache quantization | Cache format when selected |
| **`kv_share`** | Q-K=V projection sharing, 50% KV reduction | Weight merge at load time |
| **`gdn2_attention`** | Gated DeltaNet-2 O(1) decode | Replaces KV cache with fixed state matrix |
| **`lt2_looped`** | Weight-shared T-pass loop + AHLA | Changes forward function signature |
| **`elf_sde`** | Logit-normal noise injection for DDTree diversity | Applied during draft tree build |

### Simplified Inference Flow

```mermaid
graph LR
    subgraph Input
        A[Tokenizer] --> B[PFlash/DashAttn Prefill]
    end
    subgraph Model
        B --> C[Transformer Forward]
        C --> D[Delta Routing]
        C --> E[Hybrid OCT+PQ KV]
        C --> F[Sparse MLP]
        C --> G[MLS Aggregate]
    end
    subgraph Decode
        C --> H[DDTree Search]
        H --> I[BT Rank]
        I --> J[Leviathan Verify]
    end
    subgraph Adapt
        K[SR2AM Config] --> H
        L[BanditPruner] --> H
        M[CNA Steering] --> C
    end
```

### Input Layer

| Component | What | Gate |
|-----------|------|------|
| **BPE Tokenizer** | Train/encode/decode | always |
| **PFlash** | Block-sparse speculative prefill, 21× seq reduction | always |
| **DashAttention** | α-entmax (1.5) adaptive routing replaces fixed top-k | `dash_attn` |
| **RTPurbo** | Head-wise retrieval/local classification, dynamic top-p | `rt_turbo` |
| **Budget Adaptation** | Compression-adaptive DDTree budget [0.5×, 2.0×] | `budget_adaptation` |

### Model Layer

| Component | What | Gate |
|-----------|------|------|
| **Sparse MLP** | Skip dead ReLU neurons in w2 matmul | `sparse_mlp` |
| **Delta Routing** | Cross-layer residual delta routing at block boundary | `delta_routing` |
| **Hybrid OCT+PQ** | Default KV codec — OCT triplet + PQ 2D Givens, best MSE | `hybrid_oct_pq` |
| **SpectralQuant** | Calibrated eigenbasis + water-fill (secondary) | `spectral_quant` |
| **MLS Aggregate** | Average last K layer residuals before LM head | `mls_aggregate` |
| **Domain Latent** | Mid-layer K/V injection | `domain_latent` |
| **PPoT** | CPU logit resampling at high-entropy positions | `ppot` |

### Attention (O(1) alternatives)

> **Note:** These are **opt-in alternative forward paths** (`forward_gdn2()`, `forward_raven()`, `forward_looped()`). The default `forward()` → `forward_base()` uses standard O(N) softmax attention.

| Component | What | Gate |
|-----------|------|------|
| **GDN2** | Gated DeltaNet-2 — O(1) decode, constant state per head | `gdn2_attention` |
| **Raven RSM** | Fixed-slot Top-K routing memory, frozen unselected slots | always compiled, opt-in `forward_raven()` |
| **HLA/AHLA** | Higher-order Linear Attention — O(1) prefix stats | `hla_attention` |
| **LT2 Looped** | Weight-shared T-pass loop, hybrid SDPA+AHLA | `lt2_looped` |
| **TF Loop** | Training-free ODE-motivated sub-stepping | `tf_loop` |
| **DMax SPD** | Soft parallel decode, hybrid token/mask embeddings | `dmax_spd` |
| **FlashAR Consensus** | Dual-path ternary thermal routing | `flashar_consensus` |

### Decode Layer

| Component | What | Gate |
|-----------|------|------|
| **DDTree** | Best-first tree from marginal log-probs | always |
| **LeviathanVerifier** | p/q rejection sampling, identical output distribution | always |
| **BT Rank** | Bradley-Terry pairwise ranking, +10.6pp over pointwise | `bt_rank` |
| **BanditPruner** | UCB1/ε-greedy/Thompson adaptive ScreeningPruner | `bandit` |
| **ELF SDE** | 10-22× path diversity via logit-normal noise | `elf_sde` |
| **Lattice Deduction** | α-intersection pruning + conflict detection | `lattice_deduction` |
| **PhraseBoost** | Context trie phrase boosting for DDTree | `phrase_boost` |
| **Parallel-Probe** | Consensus-based parallel branch control | `parallel_probe` |

### Infrastructure

| Component | What | Gate |
|-----------|------|------|
| **SR²AM Configurator** | Per-turn planning regulation (PlanNew/Extend/Skip) | `sr2am_configurator` |
| **Data Gate** | Task-level filtering before solver | `data_gate` |
| **CNA Steering** | Contrastive Neuron Attribution + runtime modulation | `cna_steering` |
| **Deep Manifold** | L2/KL fixed-point residual scoring | `deep_manifold` |
| **Federation** | Symmetric KL coupling between domain experts | `federation` |
| **SimpleTES** | RPUCG graph-based bandit loop | `tes_loop` |
| **Stability Metrics** | P50/P99/CV per-step latency instrumentation | `stability_metrics` |
| **PlasmaPath** | Bit-plane ternary SIMD matvec, 1.58 bits/weight | `plasma_path` |
| **MoA Inference** | Token-adaptive Mixture-of-Activations SwiGLU | `moa_inference` |
| **Newton-Schulz** | Cubic fixed-point orthogonalization + Muon momentum | `newton_schulz` |
| **Spectral Hierarchy** | Eigenspace alignment, Haar wavelets, Cauchy interlacing | `spectral_hierarchy` |
| **Roofline Cost** | GPU operator runtime prediction (~5µs CPU) | `roofline_cost` |
| **Kog CPU Fusion** | RMSNorm gamma folding + QKV interleaving | `kog_cpu_fusion` |
| **PEIRA Distill** | Collapse-free inter-view regressor alignment | `peira_distill` |
| **ILC Distill** | Synonym-aware DDTree pruning via offline k-means | `ilc_distill` |
| **Hydra Budget** | Emergent self-repair layer skipping | `hydra_budget` |
| **Trigger Gate** | CPU/GPU/ANE tier promotion via QPS/latency/queue monitoring | `inference_router` |
| **FreqBandit** | Oscillatory spectral bandit — cyclic pattern detection → adaptive speculative decode | `freq_bandit` |

📖 **Full GOAT audit table** with research source, real gain, and replaced feature: See [`.docs/01_overview.md`](.docs/01_overview.md).

### GOAT-Proved Additions (Plans 225–294+)

| Feature | Plan | GOAT | Key Gain |
|---------|------|------|----------|
| **Posterior-Guided Pruner Evolution** (`posterior_evolution`) | 239 | 8/8 ✅ | Bayesian precision-gated lifecycle actions (Patch/Split/Compress/Retire), 258ns overhead |
| **Spectral Irrep Pruner** (`spectral_pruner`) | 246 | ✅ | Spectral flatness detection for converged logit distributions, +3.6% overhead only |
| **Spectral Budget Router** (`spectral_budget`) | 254 | 19/19 ✅ | Layer-adaptive NS depth + rank-p spectral truncation (opt-in — GOAT-gated, not in default)
| **Regime Transition** (`regime_transition`) | 215 | 8/8+4/4 ✅ | Self-revising discovery, -0.3% overhead vs real decode |
| **SubstrateGate** (`substrate_gate`) | 216 | ✅ | Inference-time capability substrate routing via MLP masks |
| **Critical Interval Gate** (`critical_interval_gate`) | 222 | ✅ | Entropy-triggered solver switch, zero cost (entropy already computed) |
| **LLMExecGuard** (`llmexec_guard`) | 223 | ✅ | Entropy-driven verification budgeting, zero cost when guard holds |
| **Outlier-Aware Quant Guard** (`outlier_guard`) | 224 | ✅ | KS-test outlier detection for weight matrices |
| **EGCS** (`egcs`) | 206 | ✅ | Episode-guided constraint synthesis from successful translations |
| **Three-Mode Router** (`three_mode_router`) | 211 | ✅ | Neuro-symbolic bandit: Direct/CoT/Symbolic per-query routing |
| **Breakeven Routing** (`breakeven_routing`) | 250 | 7/7 ✅ | 49% wallclock savings on long sequences, ~9ns overhead |
| **DEC Operators** (`dec_operators`) | 251 | Foundational ✅ | Discrete Exterior Calculus on cell complexes, conservation-guaranteed |
| **Cubical Topology** (`lattice_operad`) | 252 | Foundational ✅ | IntervalPruner + CubicalNerve + LatticeOpernad composition |
| **Segment Checkpoint** (`segment_checkpoint`) | 226 | ✅ | Cached KV segment checkpoints at segment boundaries |
| **RCD Residual** (`rcd_residual`) | 258 | ✅ | Entropy-weighted residual context injection for D2F |
| **Spec Pruner** (`spec_pruner`) | 259 | ✅ | Modelless spec-to-constraint O(1) RoaringBitmap compilation |
| **Epiplexity Bandit** (`epiplexity_bandit`) | — | ✅ | Epistemic perplexity bandit for domain-aware routing |
| **CADDTree Budget** (`caddtree_budget`) | 219 | ✅ | Compositional adaptive DDTree budget allocation |
| **Static Cal Tables** (`static_cal_tables`) | 227 | ✅ | Pre-computed quantization calibration, zero inference cost |
| **Targeted Precision** (`targeted_precision`) | 227 | ✅ | Per-head bit allocation from weight statistics |
| **Modality Pruned Load** (`modality_pruned_load`) | 227 | ✅ | Pipeline pruning for modality-specific context loading |
| **Precision Aware Draft** (`precision_aware_draft`) | 227 | ✅ | Quantization-aware speculative draft scoring |
| **Async QDQ Overlap** (`async_qdq_overlap`) | 227 | ✅ | Overlapped quantize-dequantize with compute |
| **Sparse Off-Principal Task Vector** (`sparse_task_vector`) | 264 | G1–G2 ✅ | OPD-grounded sparse delta format, 2.9–5.7× storage reduction vs dense LoRA |
| **Off-Principal Retrieval** (`off_principal_retrieval`) | 264 | G3–G4 ✅ | ≥99% principal energy removed, off-principal beats cosine top-1 |
| **Spectral-Concentration Adaptive Rank** (`spectral_rank`) | 264 | G5–G6 ✅ | ≥30% avg rank reduction via OPD spectrum concentration |
| **Module-Energy Compute Routing** (`module_energy_route`) | 264 | G7–G8 ✅ | Paper FFN profile match (Plasma/GPU/ANE/SIMD), monotone QPS routing |
| **Gauge-Invariant Adapter Composition** (`gauge_invariant`) | 270 | 17/17 ✅ | LoRA-Muon NS inv-sqrt + gauge rebalance + compose, 4609%→0% error |
| **CHIAR Chiaroscuro Attention** (`chiaroscuro`) | 269 | 9/9 ✅ | Per-token DCT spectral entropy KV strategy (3.03× compression), operator routing, collapse discovery |
| **Attention Matching** (`attn_match`) | 271 | 9/9 ✅ | Modelless KV compaction `(K,V)→(Ck,β,Cv)`: β-recovery 1e-6, Cv Frobenius 0.0, 3.01× SIMD, blocked Cholesky (32×32), adaptive router (scalar/SIMD/rayon/GPU/ANE) |
| **Manifold Power Iteration MoE Router** (`manifold_power_iter_router`) | 279 | 9/9 ✅ | One-shot router-row conditioning at snapshot swap, sub-ms swap (0.076ms N=8 D=256), byte-identical determinism |
| **Temporal Derivative Kernel** (`temporal_deriv`) | 277 | 4/4 fusions ✅ | Dual fast/slow EMA surprise signal — state-vector companion, surprise-gated writes, collapse detection, curiosity signal |
| **Triggered Injection Gate** (`triggered_injection`) | 278 | G1/G2/G3/G8 ✅ | Sigmoid-thresholded inject/skip gate — 50% skips w/ 0.63% quality parity in saturated regime |
| **FaithfulnessProbe** (`faithfulness_probe`) | 278 | G1/G2/G8 ✅ | Causal intervention diagnostic — 100%/100% detection, IG surrogate Spearman ρ=1.0, audit cadence |
| **CS-KV-Importance Probe** (`cs_kv_probe`) | 280 | G1/G2/G3 ✅ | Compressed-sensing KV-group importance probe + density-budget interpolator, sigmoid-compatible |
| **BoMSampler** (`bom_sampling`) | 281 | G1/G2/G3 ✅ | K-hypothesis single-pass belief sampling — K=8 at 1.87× step, **+31.49pp** arena win in riir-ai Plan 314 |
| **Self-Advantage Gate** (`self_advantage_gate`) | 283 | 4/4 ✅ | Dead-compute detector via `log π+(a) − log π̂(a)` — paper 18× forward-pass reduction, vocab ≤ 128 |
| **CLR Claim-Level Reliability** (`clr`) | 284 | ✅ | Runtime CLR — sigmoid projection vote over claim embeddings, self-adaptive test-time scaling |
| **Sink-Aware Attention** (`sink_aware_attn`) | 287 | G1/G2 cached ✅ | NOP/Broadcast classifier + dual-policy sigmoid gate — cache cadence=16 ≤5% steady-state |
| **ICT Branching Detector** (`ict_branching`) | 294 | G1/G3/G4/G5/G6/G10 ✅ | `collision_purity β(π) = Σ π²`, JS-divergence novelty, BranchingDetector — ρ(H₁,JS)=0.065 (Super-GOAT proceeds) |
| **CCE Moderator** (`cce_moderator`) | 295 | G1/G2/G3 ✅ | LP-CCE solver + Bregman primal-dual iterator (arxiv 2606.20062). Chicken CCE welfare +37.5% over Nash; designer steering demo shows two Γ₀ → two different CCEs. Default-off pending riir-ai Plan 325 runtime integration (G4 latency + G5 LatCal). |
| **MicroRecurrentBeliefState** (`micro_belief`) | 276 | G1.1–G1.4 ✅ | BeliefKernel trait unifying attractor + leaky-integrator families — G2 (attractor coherence) deferred |
| **Forensic Watermark** | Moved to riir-ai | Recipe impl relocated to Plan 322 (honeypot OPSEC) |

## 🎮 Arena Proofs — HL Thesis Validated

Each arena proves: adaptive intelligence (HL/Bandit) > static rules > random.

| Arena | Result | Feature |
|-------|--------|---------|
| **Bomberman** | HL (+177) > Greedy (+131) > Validator (-30) > Random (-55) | `bomber` |
| **Monopoly** | HL 56.5% win rate, +41.3pp over Validator | `monopoly` |
| **FFT Tactics** | TFT 99% win rate — game theory optimal | `fft` |
| **Go** | Greedy/Validator/HL 100% vs Random 35% | `go` |
| **NFSP/MCTS Duality** | BanditMCTS 75% vs MCTS 8% — backward signal transforms forward search | `bandit_mcts` |

📖 **Full benchmarks, architecture, API:** [`.docs/23_hl_arena_detail.md`](.docs/23_hl_arena_detail.md).

## 🧠 Deterministic Validator

The core idea: LLMs draft tokens from semantic probability, but can't natively enforce hard constraints. A deterministic rules engine sits between draft and verification:

```
LLM drafts logits → SynPruner filters invalid Rust syntax → DDTree builds valid-only tree → Target verifies
```

**Proven with Sudoku** — Path-aware `ConstraintPruner` catches 100% of invalid branches:

```
Unpruned:    100 nodes,  46 accumulated-valid (46.0%)
Static-Only: 100 nodes,  84 accumulated-valid (84.0%)
Path-Aware:  100 nodes, 100 accumulated-valid (100.0%)
```

**Arto Inkala "World's Hardest Sudoku"**: 49,559 steps, 7 hull vertices, 7,079.9× compression.

📖 See [`.docs/05_sudoku.md`](.docs/05_sudoku.md) and [`.docs/06_validator.md`](.docs/06_validator.md).

## 🪦 What Didn't Work

| Feature | Verdict | Why |
|---------|---------|-----|
| Stepwise Reward (Plan 054) | **NO GAIN** | Same tree/path/goal, +33% latency only |
| δ-Mem (Plan 053) | **NO GAIN for DDTree** | 26× latency overhead, corrections too small |
| SDAR Arena | **Negative result** | ELO 954 ≈ Rubric 955 — no improvement |
| RMSD (Plan 125) | **NO GOAT** | 46/46 structural proofs pass but no arena improvement |
| TurboQuant | **Demoted** | SQ/OCT dominate at all quality metrics |
| DFlare Fusion (Plan 174) | **IMPROVEMENT GOAT FAILED** | Structural ✅ but no measurable acceptance gain |
| DFlare KV Routing (Plan 174) | **IMPROVEMENT GOAT FAILED** | No gain over static routing |
| DFlare Progressive Budget (Plan 174) | **IMPROVEMENT GOAT FAILED** | No gain over uniform budget |
| ManifoldPruner (Plan 234) | **NO GOAT** | G1 FAIL: sigmoid(x)>0.5 ⟺ x>0, identical to binary at 0.5 cutoff |
| FuncAttn (Plan 286) | **G6 FAIL** | 0.969 < SDPA 1.000 on masked-token LM prediction at 600 FD-SGD steps — stays opt-in |
| CompressionDrafter (Plan 285) | **GOAT FAILED (2 runs)** | G1 1.50× (<3× target), G2 1077× (>2× target). Beam search structurally loses to template selection at Hot-tier |

📖 **Full negative result detail + replaced feature audit:** [`.docs/20_negative_results.md`](.docs/20_negative_results.md).

## 🔀 Feature Showcase

### 🧠 Attention Matching: Modelless KV Compaction (Plan 271, arxiv 2602.16284)

Compacts a KV cache `(K, V)` to `(Ck, β, Cv)` with `t < T` tokens while preserving both attention output AND attention mass under reference queries `Qref`. The β bias per retained key accounts for the mass of removed keys, making the compacted block a faithful drop-in replacement under arbitrary future concatenations.

**GOAT 9/9 PASS** — `β` recovery (`‖β−β_ref‖_∞ = 1e-6`), `Cv` reconstruction (rel Frobenius 0.0), OMP residual (0.0%), reconstruction quality (0.71% rel error), router determinism, zero alloc in hot loop, SIMD speedup (3.01× release on Apple NEON).

```mermaid
flowchart LR
    subgraph Input["Input KV cache"]
        K["K (T, d)"]
        V["V (T, d)"]
        Q["Qref (n, d)"]
    end
    subgraph Stage1["Stage 1 — Key Selection"]
        HA["HighestAttn keys
(top-t by RMS score)"]
        OMP["OMP keys
(greedy mass pursuit)"]
    end
    subgraph Stage2["Stage 2 — β NNLS"]
        BETA["Per-token bias β
(projected GD, bounded w = e^β)"]
    end
    subgraph Stage3["Stage 3 — Cv Fit"]
        CV["Least squares Cv
(blocked Cholesky, jitter fallback)"]
    end
    K --> HA
    K --> OMP
    Q --> HA
    Q --> OMP
    HA --> BETA
    OMP --> BETA
    BETA --> CV
    V --> CV
    CV --> OUT["(Ck, β, Cv) — t tokens"]
```

**Adaptive router** picks `CpuScalar` / `CpuSimd` / `CpuRayon` / `Gpu` / `Ane` per stage based on `t` and `T` with hysteresis (no flap). Blocked Cholesky (32×32 L2-resident) activates automatically for `t ≥ 32`. GPU dispatch stub wired (T2.8) — falls back to rayon when no shader bundled.

| Metric | Value |
|--------|-------|
| **Compression ratio** | `T / t` (paper: 200× total with summarization) |
| **β recovery (synthetic)** | `‖β−β_ref‖_∞ = 1e-6` |
| **Cv reconstruction (synthetic)** | rel Frobenius 0.0 |
| **Router decision time** | 1.59 ns/call, zero alloc |
| **SIMD speedup (release, NEON)** | 3.01× scalar (≥1.5× threshold) |

Feature gate: `attn_match` (**default-ON** since Plan 271 Phase 7 GOAT 9/9). Adaptive CoT variant: `adaptive_cot_compaction` (entropy-thresholded, opt-in).

📖 Plan: [`.plans/271_attention_matching_compaction.md`](.plans/271_attention_matching_compaction.md). Research: [`.research/233_Attention_Matching_KV_Compaction.md`](.research/233_Attention_Matching_KV_Compaction.md). Paper: [arxiv 2602.16284](https://arxiv.org/abs/2602.16284).

### 🛰 Sink-Aware Attention: NOP/Broadcast Classifier + Dual-Policy Gate (Plan 287, arxiv 2606.08105)

Per-head attention-sink classifier distinguishing **Adaptive NOP** sinks (`‖v_s‖ ≈ 0`, suppress residual — should gate) from **Broadcast** sinks (`‖v_s‖ ≈ content`, rank-1 update carrying load-bearing global info — should preserve). Builds on Fesser et al. *A Unifying View of Attention Sinks: Two Algorithms, Two Solutions*.

Two diagnostics per sink position:
- `value_norm_ratio = ‖v_s‖ / mean_i(‖v_i‖)` — NOP if `< 0.2`, Broadcast if `≈ 1`.
- `stable_rank(O) = ‖O‖_F² / σ_1²` via vendored ~30-line power iteration — Broadcast signature is rank-1, so stable rank `≈ 1` triggers the fast early-exit.

The dual-policy gate then applies the sigmoid gate only to NOP heads, preserving Broadcasts. Stops the over-suppression of useful broadcasters under our default sigmoid attention.

**Production path:** `apply_dual_policy_gate_cached` — amortizes the classifier over `audit_every_n` calls (default 16). Sinks in trained transformers are stable across forward passes, so the cached decision is correct. Steady-state overhead matches `Uniform` (just a copy); the classifier runs only on audit calls.

**Layout choice:** both `&[Vec<f32>]` (diagnostic-friendly, row-by-row construction) and flat `&[f32]` (forward-path-friendly, matches `parallax_attn`/`funcattn` output) layouts are provided via `_flat` suffix variants. **Flat variants are 1.8×–5.1× faster** than `Vec<Vec<f32>>` due to cache locality — prefer them when composing with the attention forward path. See [Plan 288](.plans/288_sink_aware_flat_layout.md).

```text
         attn column   values V     update O = AV
           │             │             │
           ▼             ▼             ▼
     ┌─────────────────────────────────────┐
     │   classify_sink_at(pos, col, V, O) │
     │                                     │
     │  strength = mean(col)               │
     │  ratio   = ‖v_pos‖ / mean(‖v_i‖)   │
     │  srank  = power_iter(Oᵀ·O, 5)      │
     │         (cosine probe O[0]∥O[n-1]   │
     │          for rank-1 fast path)      │
     │                                     │
     │  strength ≤ τ_sink        → None   │
     │  ratio    ≤ nop_max       → Nop    │
     │  ratio ∈ [b_min, b_max] ∧  → Broadcast
     │    srank ≤ b_srank_max             │
     └────────────┬────────────────────────┘
                  │ kind
                  ▼
     ┌─────────────────────────────────────┐
     │ apply_dual_policy_gate[_cached]     │
     │   Nop        → out = O · σ(g)       │
     │   Broadcast  → out = O   (preserve) │
     │   None       → out = O   (default)  │
     │                                     │
     │   cached: skip classify on          │
     │   non-audit calls (cadence=16)      │
     └─────────────────────────────────────┘
```

| Metric | Value |
|--------|-------|
| **G1 classifier correctness** | 18/18 unit tests PASS (8 G1 + 2 cached-variant parity + 8 flat-layout parity; NOP, Broadcast, mixed, edges, cache invalidate, flat vs Vec<Vec> bit-identical) |
| **Stable-rank fast path (rank-1)** | 0.625 µs for n=128, d_h=64 (was 3.125 µs pre-Issue 001; cosine probe skips power iteration) |
| **Stable-rank slow path (random)** | 6.583 µs for n=128, d_h=64 (target was <1µs — documented G2.4 miss, but only matters for non-Broadcast heads) |
| **Dual-policy latency (per-call, Vec<Vec>) vs Uniform** | 1000–3000% at n=128 (target was ≤5% — **G3 STRUCTURAL FAIL**: classifier reads attn (n²) + values (n·d); Uniform is just an n·d copy. Memory-bandwidth bound.) |
| **Dual-policy latency (per-call, flat &[f32]) vs Uniform** | 390–1700% at n=128 — **1.8×–5.1× faster than Vec<Vec<f32>>** (Plan 288). Still structurally cannot beat memcpy, but the gap is dramatically smaller. |
| **Dual-policy latency (cached cadence=16, flat) vs Uniform** | **≤5%** steady-state (often -30% to -40% — flat cached path is faster than Vec<Vec> Uniform baseline). Production path. |
| **Forward-path composition overhead (Plan 289)** | `tiled_attention_parallax_forward_sink_aware(Uniform)` vs vanilla forward: **-0.3% / 0.0% / +0.6%** at n ∈ {64, 128, 256}. Zero-cost abstraction contract verified. DualPolicy adds 2.1%–11.0% (matches per-call cost); cached brings it to ≤3%. |
| **Synthetic G2 (Broadcast preservation)** | DualPolicy preserves O unchanged for Broadcast heads (2/2 PASS) |

**Scope reductions** (documented in [`.benchmarks/059_sink_aware_goat.md`](.benchmarks/059_sink_aware_goat.md)):
- ~~Plan T3.1–T3.3 direct wiring into `parallax_attn.rs` / `funcattn.rs` forward paths is **deferred**~~ → **RESOLVED for parallax** (Plan 289): `tiled_attention_parallax_forward_sink_aware` ships as a separate entry point (not a `ParallaxConfig` field), preserving `Default::default()` backwards-compat. **FuncAttn wiring closed as not-applicable** — see [Research 261](.research/261_FuncAttn_Sink_Semantics_Verdict.md): FuncAttn's `Φ · C · Ṽ` structure has no `n×n` attention matrix for the sink classifier to scan (basis modes are partition-of-unity by design, so the NOP/Broadcast discrimination collapses into a column-norm check).
- Real-ViT `effective_rank` G2 gate is **DEFERRED** — needs a frozen model. Synthetic G2 substitute in `tests/sink_aware_g2_synthetic.rs` (and now in `parallax_attn::sink_aware_tests` via the forward path).

Feature gate: `sink_aware_attn` (**opt-in** — per-call G3 latency target structurally infeasible; cached variant meets target but real-ViT G2 still deferred). Forward-path composition requires both `parallax_attn` and `sink_aware_attn`. Issue: [`.issues/001_sink_aware_g3_latency.md`](.issues/001_sink_aware_g3_latency.md). Flat-layout variants: [Plan 288](.plans/288_sink_aware_flat_layout.md). Forward-path wiring: [Plan 289](.plans/289_sink_aware_forward_path_wiring.md).

📖 Plan: [`.plans/287_sink_aware_attention.md`](.plans/287_sink_aware_attention.md) + [`.plans/288_sink_aware_flat_layout.md`](.plans/288_sink_aware_flat_layout.md) + [`.plans/289_sink_aware_forward_path_wiring.md`](.plans/289_sink_aware_forward_path_wiring.md). Research: [`.research/258_Attention_Sink_Dual_Mechanism_NOP_Broadcast.md`](.research/258_Attention_Sink_Dual_Mechanism_NOP_Broadcast.md). Paper: [arxiv 2606.08105](https://arxiv.org/abs/2606.08105).

### 🔀 MUX-Latent: Zero-Training Context Compression (Plan 238)

Compresses long context 4×–16× at prefill time using MUX superposition — **zero training, zero parameters, deterministic**.

```mermaid
flowchart LR
    subgraph Encode["ENCODER — zero training"]
        T["[t1..t8] span"] --> MUX["MUX Superpose\nΣ decay^j × onehot(t_j)"]
        MUX --> Z["z_i (1 latent slot)"]
    end
    subgraph Wire["WIRE — latent-to-latent"]
        Z -->|"f32 vector, BLAKE3 committed"| STREAM["Stream / Patch\nno decompress needed"]
    end
    subgraph Decode["DECODER — domain_latent inject"]
        STREAM --> INJ["Mid-layer K/V\n1 KV entry (not 8)"]
        INJ --> GEN["Generate tokens"]
        GEN -.->|"on demand"| EXPAND["EXPAND(i)\nO(1) lossless recovery"]
    end
```

| Metric | X4 | X8 | X16 |
|--------|-----|-----|------|
| **TTFT Speedup** | 6.6× | 14.0× | **29.0×** |
| **KV Memory Reduction** | 75% | 87.5% | **93.8%** |
| **Logit Cosine Sim** | 0.597 | 0.617 | 0.552 |

Enables latent-to-latent streaming, freeze/thaw patching, federated context, and KG octree leaf patching. Feature gate: `mux_latent_context` (**default-ON**, GOAT 5/5 PASS).

📖 Plan: [`.plans/238_mux_latent_superposition_fusion.md`](.plans/238_mux_latent_superposition_fusion.md).

#### MUX-Latent Wire Patch (Plan 243)

Latent-to-latent patching over the wire — no decompress/recompress round-trip. Patches MUX latent slots as KG octree leaf nodes. 68-byte wire format (4B segment_id + 32B weights + 32B BLAKE3). SIMD batch at ≥100K/sec. Feature gate: `mux_latent_wire`.

```
Client (Plasma/Hot)           Wire (Fourier Shell)         Server (Warm/Cold)
─────────────────────         ────────────────────         ──────────────────
MUX encode 256 tokens → 32 slots
    │
    ├─ Dirty check → 3 slots changed
    │
    └─ LatentPatchBatch ──────► Fourier shell encodes ──────► SIMD 4-wide BLAKE3 verify
       {patches: [(sid, δ, blake3)×3]}                       │
                                                              ├─ Patch CompressedContext
                                                              ├─ Reinject via DomainLatent
                                                              │
                                    ◄── PatchReceipt ─────────┘
                                        {committed: [sid×3]}
```

| Metric | Target |
|--------|--------|
| Single patch encode | ≤ 50ns |
| SIMD batch 256 verify | ≤ 10μs |
| E2E round-trip | ≤ 500μs |
| Throughput | ≥ 100K patches/sec |

**Security:** BLAKE3 commitment + scalar projections only on wire (no 64-dim HLA). Fourier shell on write path. Chain-layer: full validation (mod 1).

```sh
cargo run --example mux_latent_wire_patch --features mux_latent_wire
cargo run --example mux_latent_octree_bridge --features mux_latent_wire
cargo test --features mux_latent_wire --test bench_243_mux_latent_wire_goat -- --nocapture
```

📖 Plan: [`.plans/243_mux_latent_wire_patch.md`](.plans/243_mux_latent_wire_patch.md).

### 🧵 ThoughtFold: Inference-Time Chain Folding (Plan 195)

Prunes redundant reasoning steps during CoT generation using attention-based importance scoring + binary search fold verification. No LLM training — pure inference-time optimization.

```text
ThinkingController (Plan 194)
    │
    ├── Direct mode → no folding (zero cost)
    │
    └── Latent/CpuResample mode
            │
            ├── StepBoundaryTracker — detects \n\n, think-tags
            ├── ChainFolder (ScreeningPruner) — attention importance + binary search
            ├── FoldBandit — 5-arm Thompson sampling for fold budget
            └── FoldCache — KV cache truncation/replay planning
```

| Metric | Target | Status |
|--------|--------|--------|
| Token reduction on hard queries | ≥30% | GOAT 2 ✅ |
| Accuracy regression | ≤2% | GOAT 3 ✅ |
| Direct mode overhead | 0% | GOAT 1 ✅ |
| Fold overhead | <5% | GOAT 4 ✅ |

Feature gate: `chain_fold` (depends on `thinking_cot`, default-OFF until GOAT proof on real model).

### 🛑 Collapse-Aware Adaptive Thinking (Plan 212)

Detects reasoning collapse **at runtime** during CoT generation and triggers early exit. Three-layer stack composes with existing infrastructure:

1. **Pre-Decide** — SelectivityRouter kurtosis → Direct vs CoT (Plan 204)
2. **Mid-Think** — CollapseDetector monitors hesitation patterns → force fast answer when collapse predicted
3. **Post-Verify** — T2M option stripping prevents option-matching shortcut

| Metric | Target | Source |
|--------|--------|--------|
| Token savings on simple tasks | 50-90% | Thinkless (NeurIPS 2025) |
| Accuracy on ambiguous tasks | +2-5pp | S2F (ICML 2026) |
| Collapse detection overhead | <10ns/token | O(1) ring buffer |

Feature gate: `collapse_aware_thinking` (**default-ON**). 📖 Research: [`.research/187_S2F_Slow_to_Fast_Adaptive_Reasoning.md`](.research/187_S2F_Slow_to_Fast_Adaptive_Reasoning.md).

### 🔄 SwiR Switch-Thinking: Explicit↔Latent Mode Controller (Plan 275)

Distills SwiReasoning (ICLR 2026, [arXiv:2510.05069](https://arxiv.org/abs/2510.05069)) into a training-free runtime controller that switches between **explicit** (token-space) and **latent** (soft-embedding) reasoning modes based on block-relative entropy trends. Asymmetric dwell windows prevent mode chatter; a switch-count guard suppresses overthinking (convergence at ½C_max, forced answer above C_max).

Three primitives, all modelless:
- `SwiRController` — the 2-mode state machine (3.1 ns/step, zero-alloc).
- `soft_embedding` — probability-weighted vocabulary mixture for latent mode (SIMD chunked, O(vocab·dim)).
- `mix_thinking_signal` — control-token embedding blend at switch instants (α_t/β_t schedule).

Integrates into `thinking_cot` (Plan 194) as a `ThinkingStrategy`. Optional kurtosis escape hatch (`observe_kurtosis`) forces Explicit mode on rigid-constraint tasks, bypassing latent exploration where continuous mixtures would hallucinate.

| Gate | Target | Result |
|------|--------|--------|
| G3 step() perf | < 200 ns/call | **3.1 ns** (64× margin) |
| G4 convex hull | 1000 random probs in hull | **1000/1000** |
| G7 zero-alloc step() | 0 allocs | **0 allocs / 0 bytes** |
| G1c controller correctness | switches + convergence + termination | 6 switches, 3 CloseThink, 1 ForceAnswerPrefix, terminated step 21 |
| G2p efficiency proxy | SwiR < fixed-budget baseline | 33 steps vs 1024 = 31× fewer |
| G9 hyperparameter ablation | W_E→L/C_max/α_0 respond correctly | monotonic ✓, α-independent ✓ |

**G1/G2 real-model validation (riir-ai Plan 313, 2026-06-19):** ran on Gemma 2 2B IT + MATH-500 (CPU M1 Pro). **G2 = 1.37× (GATE PASS, target ≥ 1.3×)** at the tuned config `w_e_to_l=32, c_max=64` (n=5; 1.43× at n=10 partial) — non-monotonic Pareto curve peaks at c_max=64. **G1 = 0%** — blocked purely by Gemma 2 2B capability (T4.2e ruled out the prompt/checker bug class; verified on `1^(2^huge)=1` the model emits correctly-formatted `\boxed{ }` with wrong content). Definitive G1 gate pass requires Qwen3-4B/8B. **Verdict:** promote `swir_switch_thinking` to default-on once G2 is confirmed at n=20+ (token efficiency is the primary value prop). katgpt-rs is modelless (no model loader); the algorithmic invariants above are necessary preconditions.

Feature gate: `swir_switch_thinking` (depends on `thinking_cot`, **opt-in** until G1/G2 pass on a real model). 📖 Plan: [`.plans/275_swir_switch_thinking.md`](.plans/275_swir_switch_thinking.md). Research: [`.research/241_SwiReasoning_Explicit_Latent_Switch.md`](.research/241_SwiReasoning_Explicit_Latent_Switch.md). Benchmark: [`.benchmarks/275_swir_switch_thinking_goat.md`](.benchmarks/275_swir_switch_thinking_goat.md).

### 🧠 NextLat Belief-State Speculative Drafter (Plan 217)

Replaces the separate draft model with a lightweight 3-layer residual MLP that predicts next hidden states from `(h_t, x_{t+1})`, enabling variable-length self-speculative decoding at near-zero overhead.

| Gate | Result |
|------|--------|
| Belief vs MTP overhead | 2.2× (134 μs vs 60 μs) |
| MLP forward per step | 17 μs/step at n_embd=16 |
| Cache hit rate (walk cycle) | 100% |
| Cached vs uncached | **5× speedup** (15 μs vs 90 μs) |
| Acceptance rate | Both produce valid 64-node trees |

**43 tests + 7 benchmarks**, GOAT all pass. Feature gate: `belief_drafter` (**default-ON**).

📖 Plan: [`.plans/217_nextlat_belief_state_drafter.md`](.plans/217_nextlat_belief_state_drafter.md).

### 🗂️ BFCF × LFU × Sharding (Plan 218)

Extends BFCF pruning with LFU region caching (papaya lock-free HashMap, BLAKE3 keys, sigmoid-gated admission), frequency-aware sharding, and SIMD-friendly region-level batching. **44 tests + 10 benchmarks, GOAT all pass.** Cache hit rate: 95% on cyclic workload.

Feature gate: `bfcf_lfu_shard` (**default-ON**). 📖 Plan: [`.plans/218_bfcf_lfu_shard.md`](.plans/218_bfcf_lfu_shard.md).

### 🔀 Dual-Pool Reachable Memory Router: Proactive Non-Trapping CGSP (Plan 282)

Distills Hao, Long, Zhao 2026 — *"Self-Evolving MAS via Decentralized Memory"* ([arXiv:2605.22721](https://arxiv.org/abs/2605.22721)) into a `DualPoolBandit<B: HintDeltaBandit>` that splits CGSP's bandit into an **exploitation pool** (E-pool: consolidated successes, local-walk operator) and an **exploration pool** (X-pool: fresh candidates, teleportation operator). A sigmoid router `α = sigmoid(w_E − w_X) ∈ (0, 1)` guarantees the X-pool always retains strictly nonzero selection probability — the induced Markov chain is irreducible and aperiodic (**DecentMem Theorem 1**), so the agent is **provably never trapped**, by construction, with no collapse detector needed.

```mermaid
flowchart TB
    BC["begin_cycle
α = sigmoid(w_E − w_X)"] --> SEL{"u < α ?"}
    SEL -->|"yes (α)"| E["E-pool
consolidated successes
local-walk operator"]
    SEL -->|"no (1−α) > 0"| X["X-pool
fresh candidates
teleportation operator"]
    E --> CYCLE["CgspLoop::cycle
operates on active pool"]
    X --> CYCLE
    CYCLE --> EC["end_cycle"]
    EC --> RU["route_update
DecentMem Eq. 6/7"]
    RU --> CON["consolidate
DecentMem Eq. 8"]
    CON --> BL["blend
Phase 1: priority-blend"]
    CON --> GR["grow
Phase 4: push_arm"]
    GR --> GATE["gate(arm)?
FaithfulnessProbe
(Plan 278)"]
    GATE -->|"live"| PROMOTE["promote X→E"]
    GATE -->|"dead"| REJECT["reject"]
```

**GOAT G1–G4 PASS (G5 deferred to riir-ai). Feature stays opt-in until personality divergence validated.**

| Gate | Target | Actual | Verdict |
|------|--------|--------|---------|
| G1 — Reachability | X-pool always selected (α < 1) | balanced 1.1 cycles, extreme ≤ 79k | **PASS** |
| G2 — Regret bound | O(log T) on synthetic bandit | regret 24.6 ≤ 5·log(10k) = 46 | **PASS** |
| G3 — E-pool growth | Discovers strategy outside initial pool | 4 → 5+ arms, optimal promoted | **PASS** |
| G4 — Faithfulness gate | Dead items rejected | 4 live promoted, 4 dead filtered | **PASS** |
| G5 — CGSP integration | Personality divergence widens | deferred to riir-ai `NpcCgspRuntime` | Pending |

Key findings:
- **Proactive vs reactive:** Dual-pool pays 0.5 ns/cycle (sigmoid + RNG) for a constant nonzero X-pool floor; single-pool CGSP + entropy-collapse detector pays 15.1 ns/cycle and only recovers **after** entropy degenerates. Dual-pool is **30× cheaper per cycle** and never traps. Single-pool with no detector never escapes (129/500 trials permanent trap).
- **Backward-compatible trait extension:** E-pool growth required `HintDeltaBandit::push_arm(priority)` and `is_growing()` — added as default methods (no-op / false), so every existing implementor is unaffected. `DualPoolBandit<B>` drops into `CgspLoop` as the `B` type parameter with zero loop changes.
- **Sigmoid (not ratio):** Per AGENTS.md, `α = sigmoid(w_E − w_X)` replaces the paper's `w_E/(w_E+w_X)`. Both preserve strict concavity, so the O(log T) regret bound transfers (Research 249 §2.3). A `min_exploration_prob` clamp (default `1e-4`) makes the theorem hold in f32 (sigmoid saturates at `x ≳ 18`).
- **FaithfulnessProbe gate (Plan 278 fusion):** `consolidate_growing_gated<F: Fn(usize)->bool>(gate)` accepts a closure wrapping `FaithfulnessProbe::is_faithfully_used(threshold)`. Arms the consumer structurally ignores (no behavioral delta on perturbation) are rejected from E-pool promotion — prevents Research 244's "dead condensed memory" failure mode where 60%+ of consolidated memory is silently ignored.
- **CGSP = degenerate case:** Single-pool CGSP is the `α = 1` (pure exploitation) degenerate case. Dual-pool strictly generalizes it.

Feature gate: `cgsp_dual_pool` (opt-in, requires `cgsp`). 📖 Plan: [`.plans/282_dualpool_reachable_router.md`](.plans/282_dualpool_reachable_router.md). Research: [`.research/249_DecentMem_DualPool_Reachable_Router.md`](.research/249_DecentMem_DualPool_Reachable_Router.md). Paper: [arXiv:2605.22721](https://arxiv.org/abs/2605.22721).

### 🧮 CLR: Claim-Level Reliability + Self-Adaptive Test-Time Scaling (Plan 284)

Distills Xu et al. 2026 — *"VibeThinker-3B"* ([arXiv:2606.16140](https://arxiv.org/abs/2606.16140), Sina Weibo Inc.) into a generic, MIT-licensed, no-game-semantics module shipping four modelless inference primitives:

1. **`clr_vote()`** — the headline nonlinear reliability gate. Given K candidate trajectories and M decision-relevant claims per trajectory, produces the winning cluster via `r_k = (mean_m v_k,m)^M` where `v_k,m = sigmoid(dot(claim_vec_k,m, direction_vec_m))`. Dot-product + **sigmoid, never softmax** (per `AGENTS.md`). The `^M` exponent is the key trick: a single low verdict drags the trajectory's reliability super-linearly, so clusters containing flawed trajectories lose to clusters of flawless ones.
2. **`ClaimExtractor` / `ClaimVerifier` traits** — open extension points. Concrete extractors/verifiers live in the consumer crate (riir-ai Plan 316 ships game-specific ones; katgpt-rs ships only the generic traits + a `FnClaimExtractor` adapter + a `SigmoidProjectionVerifier` reference impl).
3. **`brevity_tiebreak()`** — the Long2Short zero-sum tiebreak. Among clusters tied on Σ r_k within `ε`, pick the one whose representative trajectory has the shortest length. Pure algorithm, no quality change.
4. **`learning_potential()` + `mgpo_sampling_weight()`** — the curiosity feedback signals. `S_LP(y) = -(1/|y|) Σ log π(y_t|...)` ("how surprising was this under the frozen brain?"). `w(p) = exp(-γ|2p-1|)` (peaks at p=0.5, the calibration boundary). Companion `should_write_memory(r_k, S_LP)` gates memory persistence on BOTH reliability AND surprise — exactly the trajectories worth persisting for the next freeze/thaw cycle.

```mermaid
flowchart TB
    K["K trajectories
M claims each"] --> EXTRACT["extractor.extract
per-trajectory claims"]
    EXTRACT --> VERIFY["verifier.verify
v_k,m = sigmoid(dot(emb, dir_m))"]
    VERIFY --> GATE["nonlinear gate
r_k = (mean_m v_k,m)^M"]
    GATE --> CLUSTER["cluster by outcome_eq
Σ r_k per cluster"]
    CLUSTER --> TIE["brevity_tiebreak
shortest rep wins ties"]
    TIE --> WIN["winner cluster"]
    GATE -.-> LP["learning_potential
S_LP = -(1/|y|) Σ log π"]
    LP -.-> WRITE{"should_write_memory?
r_k > τ_reliable ∧ S_LP > τ_curiosity"}
    WRITE -->|yes| PERSIST["persist for freeze/thaw"]
    WRITE -->|no| DROP["skip"]
```

**GOAT G1–G5 PASS — promoted to default-on (Phase 5 T5.6).**

| Gate | Target | Actual | Verdict |
|------|--------|--------|---------|
| G1 — CLR beats majority | Δ ≥ 3pp | **+78.0pp** (CLR 100% vs majority 22%) | ✅ |
| G2 — Verifier ECE | ≤ 0.10 | **0.0087** | ✅ |
| G3 — K=32 vote latency | ≤200µs (stretch ≤50µs) | **4–5µs** (10× under stretch) | ✅ ✨stretch |
| G4 — Vote-internals allocs | 0 | **0** (vote arithmetic adds 0 allocs on top of extractor) | ✅ |
| G5 — Feature isolation | compiles ±clr | ✅ build + `nm` shows zero `clr` symbols in no-clr binary | ✅ |

Key findings:
- **Nonlinear gate is the discriminator:** a single mediocre verdict (sigmoid(0)=0.5 from an orthogonal claim) drops `r_k` from ~0.22 (clean) to ~0.14 — a 36% penalty. The `^5` exponent amplifies this into a clear Σ r_k ordering between clusters.
- **Zero-allocation hot path:** `clr_vote_minimal` writes into caller-supplied `ClrScratch` and returns `(winner_idx, Σ r_k)` scalars. After `ClrScratch::new(K, M)` warmup (3 `with_capacity` calls), the vote arithmetic + clustering + tiebreak add **0** allocations across 1000 calls. The only per-call allocations are inside `ClaimExtractor::extract()` (caller-domain — a future pre-extracted variant would eliminate these).
- **M=5 unrolled power:** for the paper default `M=5`, `reliability_gate` uses the literal `v*v*v*v*v` form (4 multiplies, no libm call) instead of `powf(5.0)`. All other M fall back to the general `powf` path.
- **Sigmoid, never softmax:** the sigmoid-projection verifier computes `1/(1+exp(-dot))` per (claim, direction) pair. Two directions on the same claim can BOTH return > 0.5 (sum > 1) — softmax would forbid this and destroy per-direction independence.
- **Curiosity gate (`should_write_memory`):** selects trajectories that are BOTH reliable (passed CLR) AND surprising (high `S_LP` under the frozen brain). This is exactly the highest-value training signal for the next freeze/thaw direction-vector update — "we got it right but didn't expect to".

Feature gate: `clr` (**default-on** since Plan 284 Phase 5 GOAT G1–G5 all pass). 📖 Plan: [`.plans/284_runtime_clr_self_adaptive_loop.md`](.plans/284_runtime_clr_self_adaptive_loop.md). Research: [`.research/255_VibeThinker_CLR_Test_Time_Reliability.md`](.research/255_VibeThinker_CLR_Test_Time_Reliability.md). Paper: [arXiv:2606.16140](https://arxiv.org/abs/2606.16140). Scorecard: [`.benchmarks/284_clr_goat.md`](.benchmarks/284_clr_goat.md). Examples: [`clr_minimal`](examples/clr_minimal.rs), [`clr_brevity_tiebreak`](examples/clr_brevity_tiebreak.rs), [`clr_learning_potential`](examples/clr_learning_potential.rs).

### 🌊 VortexFlow: Composable Sparse KV Routing (Plan 196)

Unifies multiple KV block selection algorithms behind a single `VortexFlow` trait: `BlockTopKRouter` (centroid + dot-product top-k + sigmoid), `EntmaxRouter` (α-entmax wrapper), `ValueEnergyRouter` (centroid · ‖v‖ gating, RULER 1.00). Feature gate: `vortex_flow` (default-OFF).

#### MSA Sparse Attention Family (Plan 256 — Opt-In, GOAT FAILED)

Distills MSA-style blockwise sparse scoring into VortexFlow routers. All sub-features are **opt-in** — the modelless micro-benchmark GOAT gate **FAILED** for each (see `.plans/256_msa_blockwise_sparse_distillation.md`):

| Sub-feature | Router | Winning Regime | GOAT Failure |
|------------|--------|--------------|--------------|
| `msa_sparse` | `MaxPoolBlockScorer`, `MaxStdDevBlockScorer` | Diversity-gated block scoring | (baseline for sub-features) |
| `msa_per_group` | `PerGroupTopKRouter` | High-top_k latency (0.40–0.52× vs shared) | Coverage saturated at 1.003× (need ≥1.5×) |
| `msa_kv_outer` | `KvOuterPrefill` | Short context with high block sharing (2.02× at 32K) | Block sharing drops at long context (0.83× at 512K) |
| `msa_adaptive_k` | `AdaptiveKRouter<R>` | Compute-constrained decode (37% savings) | Recall bounded at 0.629 (need ≥0.90) |

📖 Plan: [`.plans/256_msa_blockwise_sparse_distillation.md`](.plans/256_msa_blockwise_sparse_distillation.md). Full RULER arena deferred to [Issue 014](.issues/014_msa_arena_ruler_benchmark_infrastructure.md).

### 🦅 Raven RSM: O(1) Routing Slot Memory

Fixed-size slot memory with sparse Top-K routing. Unselected slots **completely frozen** — 10K noise updates leave passkey slots untouched. **2.98× faster** than flat attention at pos=8 (62,653 tok/s vs 21,019 tok/s). Opt-in alternative forward path (`forward_raven()`), not in default hot path.

📖 [`.docs/25_raven_rsm.md`](.docs/25_raven_rsm.md).

### 🔬 Percepta: Transformer-VM in Rust

Rust port of [Percepta's transformer-vm](https://github.com/Percepta-Core/transformer-vm) — O(log N) 2D convex hull attention with ternary search. **~9K lines Python+C++ → idiomatic Rust.** Apache-2.0.

Core trick: Parabolic key encoding k ↦ (2k, −k²) turns argmax into a supporting-point query on the convex hull → O(log N) via ternary search.

📖 [`.docs/22_percepta.md`](.docs/22_percepta.md).

### 🧠 Heuristic Learning Infrastructure

HL = software systems evolve through **code updates** not weight updates.

```text
Episode N:   BanditPruner selects arm → environment runs → reward → TrialLog.append()
Episode N+k: AbsorbCompress promotes stable low-Q arms to hard blocks
```

Key subsystems (default-on or part of `bandit`): Multi-Armed Bandit (UCB1, ε-greedy, Thompson), TrialLog, AbsorbCompress, ReviewMetrics. The runtime hot-swap, mid-layer emotion projection, and session-level OOD wiring live in `riir-ai`.

📖 [`.docs/09_heuristic-learning.md`](.docs/09_heuristic-learning.md).

### 🎯 G-Zero: Verifier-Free Self-Play

Modelless HL Phase 1 — Hint-δ intrinsic reward drives `AbsorbCompress` + `BanditPruner` without an external verifier:

```text
δ(q, h, a_hard) = (1/T) Σ [log πG(at | q, h, a<t) − log πG(at | q, a<t)]
```

The model-based Phase 2 (gradient optimization with self-play reward) and the arena players live in `riir-ai` / `riir-train`.

📖 [`.docs/23_hl_arena_detail.md`](.docs/23_hl_arena_detail.md) §11.

### 🧮 Deep Manifold: Fixed-Point Boundary Conditions

GOAT 6/6 proved, default-on. Mathematical foundation from [Deep Manifold Part 2](https://arxiv.org/pdf/2512.06563):

| Paper Concept | Implementation | Gate |
|---------------|---------------|------|
| Fixed-point residual ‖f(x)-x‖ | HintDelta + ManifoldResidual trait | `deep_manifold` |
| Symmetric boundaries | BT pairwise ranking + SymmetricBoundariesPair | `bt_rank` |
| Model CAP tradeoff | BanditPruner dynamic routing | `bandit` |
| Manifold federation | BoundaryAlignment KL coupling | `federation` |

**Plan 231 sub-features** (all default-ON, GOAT-proven):

| Feature | Key Gain |
|---------|----------|
| **Union Bound Confidence** | Linear degradation, 76ns overhead |
| **PathwayTracker** | 85% thinking budget savings, 100% convergence |
| **FederationComposer** | 70% early termination rate, 35% compute savings |

📖 [`.research/051_Deep_Manifold_Fixed_Point_Boundary_Conditions.md`](.research/051_Deep_Manifold_Fixed_Point_Boundary_Conditions.md).

### 🧬 Posterior-Guided Pruner Evolution (Plan 239)

Fuse BAKE precision vectors with MUSE skill lifecycle — each `ConstraintPruner` arm becomes a Bayesian hypothesis with per-feature precision, enabling precision-gated Patch/Split/Compress/Retire actions. **GOAT 8/8 PASS**, promoted to default-ON.

| Gate | Result |
|------|--------|
| Precision update correctness | ✅ Sequential BAKE-style |
| Surprise KL trigger | ✅ Sigmoid-gated |
| 5 lifecycle actions | ✅ Explore→Patch→Split→Compress→Retire |
| Decorator overhead | 258ns only when PosteriorGuidedPruner used |
| Existing pruners | Zero regression (no decorator = no overhead) |

Feature gate: `posterior_evolution` (**default-ON**). 📖 Plan: [`.plans/239_posterior_guided_pruner_evolution.md`](.plans/239_posterior_guided_pruner_evolution.md).

### 🔭 Spectral Budget Router (Plan 254)

Layer-adaptive Newton-Schulz depth + rank-p spectral truncation for inference routing. Pre-computed NS config matches empirical quantile thresholds. **GOAT 19/19 PASS**.

Feature gate: `spectral_budget` (**opt-in** — GOAT-gated, not yet promoted to default). 📖 Plan: [`.plans/254_spectral_budget_router.md`](.plans/254_spectral_budget_router.md).

### 🏛️ DEC Operators + Cubical Topology (Plans 251–252)

Foundational mathematical infrastructure — Discrete Exterior Calculus on cell complexes (conservation-guaranteed, zero-alloc SIMD) + categorical cubical framework (IntervalPruner + CubicalNerve + LatticeOpernad). Both default-ON, no GOAT gate needed (foundational).

Feature gates: `dec_operators`, `lattice_operad` (**both default-ON**). 📖 Plans: [`.plans/251_dec_operators_cell_complex.md`](.plans/251_dec_operators_cell_complex.md), [`.plans/252_cubical_category_interval_topology.md`](.plans/252_cubical_category_interval_topology.md).

### ⚖️ Breakeven Complexity Routing (Plan 250)

Cost-aware inference routing using breakeven complexity N* for tier selection. **49% wallclock savings** on long sequences (≥512 tokens) with ~9ns overhead and 0% accuracy regression.

Feature gate: `breakeven_routing` (**default-ON**, GOAT 7/7). 📖 Plan: [`.plans/250_breakeven_inference_routing.md`](.plans/250_breakeven_inference_routing.md).

### 🔄 Regime-Transition Inference (Plan 215)

Self-revising discovery with regime-aware inference. Detects when the model switches reasoning regimes and adapts compute accordingly. **-0.3% overhead** vs real decode, 8/8 mock + 4/4 real GOAT tests.

Feature gate: `regime_transition` (**default-ON**). 📖 Plan: [`.plans/215_regime_transition_inference.md`](.plans/215_regime_transition_inference.md).

### 🛡️ SubstrateGate — Capability Substrate Routing (Plan 216)

Inference-time capability extraction via pre-computed per-capability MLP masks intersected with ReLU sparsity for dual sparsity. DDTree branches routed through different substrates. **25/25 tasks done**, wired into `forward_pass`.

Feature gate: `substrate_gate` (**default-ON**). 📖 Plan: [`.plans/216_substrate_gate_capability_routing.md`](.plans/216_substrate_gate_capability_routing.md).

### 🧮 Sparse Off-Principal Task Vector — OPD-Grounded Sparse LoRA (Plan 264)

Distillation of Dense Supervision, Sparse Updates (arXiv:2606.13657). Four modelless primitives for inference-time adapter storage and routing:

1. **SparseTaskVector** (`sparse_task_vector`) — OPD-grounded sparse delta format with 2.9–5.7× storage reduction vs dense LoRA at paper densities (17.5%, 10.5%).
2. **Off-Principal Retrieval** (`off_principal_retrieval`) — projects query embeddings into off-principal subspace, removing ≥99% of principal component energy. Top-1 retrieval accuracy beats raw cosine on synthetic 8-adapter benchmark.
3. **Spectral-Concentration Adaptive Rank** (`spectral_rank`) — maps top-k spectral concentration to adaptive LoRA rank via sigmoid, reducing avg rank ≥30% vs fixed max-rank.
4. **Module-Energy Compute Routing** (`module_energy_route`) — routes compute by FFN/Attn energy fraction × QPS: FFN-heavy + low QPS → Plasma, Attn-heavy + high QPS → GPU, very low QPS → ANE. Matches paper's OPD/RLVR module profile (FFN=0.78).

**GOAT:** G1–G10 all pass (66 tests). Zero-alloc hot paths, sigmoid not softmax.

Feature gates: all four **default-ON** (GOAT-proven). 📖 Plan: [`.plans/264_sparse_off_principal_task_vector_modelless.md`](.plans/264_sparse_off_principal_task_vector_modelless.md), Research: [`.research/231_Sparse_Off_Principal_Task_Vector_OPD.md`](.research/231_Sparse_Off_Principal_Task_Vector_OPD.md).

### ⚖️ Gauge-Invariant Adapter Composition — LoRA-Muon Distillation (Plan 270)

Distillation of LoRA-Muon (arXiv:2606.12921). Three modelless primitives for gauge-invariant adapter composition:

1. **`ns_inv_sqrt_psd`** — Newton-Schulz inverse square root for PSD Gram matrices (paper Algorithm 4). Extends `src/newton_schulz.rs` with a 7-iter polynomial recurrence (`P^{-1/2} · P · P^{-1/2} ≈ I`), SIMD-accelerated, zero-alloc variant `ns_inv_sqrt_psd_into`.
2. **`gauge_rebalance`** — scalar factor-pair rebalancing (paper Algorithm 2). Computes `c = (σ_max(B)/σ_max(A))^{α/2}` via 5-step power iteration, then `A ← c·A`, `B ← B/c`. Preserves `‖AB^T‖_F` exactly.
3. **`gauge_invariant_compose`** — weighted sum of `(η_i, A_i, B_i)` pairs. Drop-in replacement for naive task-vector arithmetic that is invariant to input factorization (paper Prop 1).

**Key result:** composing gauge-equivalent inputs `(A·c, B/c)` for `c=5` gives identical merged `W` (max diff < 1e-3). Naive sum produces 4609% error; gauge-invariant compose produces 0.0000% error.

Also integrated as `SparseTaskVector::compose_gauge_invariant` (feature-gated).

**GOAT:** 17/17 tests pass (gauge invariance Prop 1 + Prop 4, power iteration convergence, NS inv-sqrt correctness/stability, compose gauge-invariance, msign roundtrip, throughput targets).

Feature gate: `gauge_invariant` (**default-ON**, GOAT 17/17). 📖 Plan: [`.plans/270_gauge_invariant_adapter_composition.md`](.plans/270_gauge_invariant_adapter_composition.md), Research: [`.research/238_LoRA_Muon_Spectral_Low_Rank_Manifold.md`](.research/238_LoRA_Muon_Spectral_Low_Rank_Manifold.md).

### 🌗 CHIAR Chiaroscuro Attention — Spectral-Entropy Operator Routing (Plan 269)

Distillation of CHIAR-Former (arXiv:2606.08327). Per-token DCT spectral entropy H(x) ∈ [0,1] drives four modelless inference-time primitives:

1. **CHIAR-KV** (`ChiaroscuroKvDispatcher`) — per-token KV cache storage strategy. H(x)<τ_lo → DCT-truncated (3.03× compression), H(x)<τ_hi → Quantized, else → Full f16. Streaming τ calibration converges to paper's [0.856, 0.864] within 1024 tokens.
2. **ChiaroscuroOp trait + ChiaroscuroRouter** — per-token operator selection between `DctMixOp` (DCT mixing layer) and `FullAttnOp`. Hard threshold gate (no STE — modelless).
3. **CollapseDiscoveryHarness** — sliding-window utilization entropy detects when operators collapse to a subset. Auto-generates `OpPromotion` recommendations.
4. **ChiarRegimeGate** — naturalistic vs synthetic prompt gate. Long + high-variance → apply CHIAR; short/flat → skip.

**InferenceRouter integration (T15):** `ChiarRouterHook` exposes KV strategy utilization entropy and regime gate recommendation via `RouterStats.chiar_stats`. Observation-only — does NOT influence tier routing (CHIAR is per-token attention, not tier selection).

**GOAT:** G1-G9 all pass — 2.48× KV compression, 12 dB SNR on smooth tokens, 0.0 reconstruction error (Theorem 1), DCT overhead 0.0002% of attention, τ converges in 1024 tokens, collapse harness identifies survivors, sigmoid everywhere, regime+dispatcher integration, zero-alloc entropy_into.

Feature gate: `chiaroscuro` (**default-ON**, GOAT 9/9). 📖 Plan: [`.plans/269_chiaroscuro_spectral_entropy_operator_routing.md`](.plans/269_chiaroscuro_spectral_entropy_operator_routing.md).

### 🕸️ DenseMesh — Latent Node Network for Modelless Inference (Plan 266)

Distillation of LMNet (arXiv:2505.12741, ICML 2026). Treats multiple forward passes through the same LLM as nodes in a directed graph, communicating via **dense hidden-state vectors** instead of natural-language tokens. Edges are pluggable: `IdentityEdge` (baseline), `LoraEdge` (frozen-vertex LoRA on attention output projection), `ProjectionEdge` (fixed random projection, no training). The whole mesh is a **latent** channel — only input and output boundary nodes touch tokens (raw values), per AGENTS.md latent/raw rules.

Architecture: `DenseNode` trait (stripped transformer forward), `DenseEdge` trait (hidden-state transform), `LayerwiseTopology` (layer-wise fully-connected graph, paper §3.1.3 with SIMD-friendly aggregation), `EdgeBandit` (Thompson sampling over `(topology, edge_set)` arms), `compute_router` (CPU/GPU/ANE by width: width-1→CPU, width≥4→GPU, output→ANE). Bridge functions `latent_to_raw_scalar` and `raw_to_latent_projection` cross the latent↔raw seam with **sigmoid** (never softmax, per AGENTS.md).

**GOAT status:** Gate 1 (correctness) ✅, Gate 3 (easy overhead — 0.997× at production scale) ✅, Gate 5 (bandit convergence) ✅. **Gate 2 (composition gain) ❌ FAILED empirically** — real trained Bomber LoRAs composed via diamond topology produce 0/1000 wins over best single (improvement -0.00%). Untrained LoRA composition is a no-op ensemble. Gate 4 (hard bound) ⚠️ measured 9.27× single-thread vs paper bound 2.5× — requires vertex parallelism (Issue 020). **Demoted to experimental.** The framework is sound plumbing, but composition gain requires riir-ai R122 trained communication edges.

Feature gate: `dense_mesh` (**opt-in, experimental** — gate 2 failed empirically). 📖 Plan: [`.plans/266_densemesh_latent_node_network.md`](.plans/266_densemesh_latent_node_network.md), Research: [`.research/234_DenseMesh_Latent_Node_Network.md`](.research/234_DenseMesh_Latent_Node_Network.md), Benchmark: [`.benchmarks/266_densemesh_goat.md`](.benchmarks/266_densemesh_goat.md).

> **Commercial bound:** the public MIT framework ships here. Trained-edge LoRA composition recipes stay in riir-ai (R122, private).

### 🛡️ FaithfulnessProbe — Causal Intervention Diagnostic for Injected Memory (Plan 278)

Distillation of Zhao et al. 2026 (arXiv:2601.22436, ICML). Verifies that a consumer's behavior is **causally bound** to injected memory — the open half of the Cognitive Integrity Layer. Three modelless primitives, all zero-training, all zero-backprop:

- **`FaithfulnessProbe`** — runs five causal interventions (`Empty`, `Shuffle`, `Corrupt`, `Irrelevant`, `Filler`) on an injected memory segment and aggregates behavioral deltas into a `FaithfulnessProfile`. If `Irrelevant`/`Filler` deltas fall below threshold, the memory is flagged as a **dead injection** (consumer silently ignores it). Runs at **audit cadence** (every N ticks), not per-tick.
- **`AttributionProbe`** — finite-difference central-difference surrogate for Integrated Gradients: `(f(M+εδ) − f(M−εδ))/(2ε)` per axis, L2-normed. No gradient graph needed. Validated against exact IG on a non-linear consumer with Spearman ρ = 1.0000 across 64 segments (G2).
- **`TriggeredInjectionGate`** — sigmoid-thresholded inject/skip decision: `should_inject(u) := sigmoid(λ·(u−τ)) > 0.5`. Collapses to `u > τ` for the boolean case (0.132 ns/call — one compare, no `exp()`). The full sigmoid value is preserved for opt-in soft-gating. **Sigmoid, never softmax** (AGENTS.md hard constraint).

All generic over `ConsumerContext` associated types (`Memory`, `Behavior`, `Delta`) — no game semantics, no `PlayerId`, no HLA/emotion channels. Game wiring (HLA `evolve_hla`, NeuronShard, KG triples) is private → riir-ai Plan 308.

**GOAT status:** G1/G1b (faithful/unfaithful detection ≥99%) ✅ 100%/100% over 400 trials. G2 (IG surrogate Spearman ρ ≥0.8) ✅ ρ=1.0000. G3 (triggered injection skips ≥50% w/ ±2% quality parity) ✅ 50.0% skips, 0.63% quality delta. G8 (zero-overhead off) ✅ 0 symbols in default build. **Decision: `triggered_injection` promoted to default-on; `faithfulness_probe` kept opt-in (diagnostic).**

Feature gates: `triggered_injection` (**default-ON**, GOAT G3 passed — saves compute, matches quality), `faithfulness_probe` (**opt-in**, diagnostic, audit cadence). 📖 Plan: [`.plans/278_faithfulness_probe_modelless.md`](.plans/278_faithfulness_probe_modelless.md), Research: [`.research/244_Self_Evolver_Faithfulness_Cognitive_Integrity.md`](.research/244_Self_Evolver_Faithfulness_Cognitive_Integrity.md), Benchmark: [`.benchmarks/278_faithfulness_probe_goat.md`](.benchmarks/278_faithfulness_probe_goat.md), Docs: [`.docs/faithfulness_probe.md`](.docs/faithfulness_probe.md).

> **Unblocks:** riir-ai Plan 308 (Cognitive Integrity Layer runtime integration — HLA `evolve_hla`, NeuronShard, KG Octree, dMoE). The bidirectional fusion with Plan 054 path-hacking stays private in riir-ai.

### 🌀 Manifold Power Iteration MoE Router (Plan 279)

Distills Redesign MoE Routers with Manifold Power Iteration (arXiv:2606.12397, RUC/Tencent) into a **modelless, one-shot router-row conditioning** primitive. Given a frozen MoE router `R ∈ ℝ^{N×D}` and per-expert Gram matrices `M[i] = W_g[i]·W_g[i]ᵀ`, produce the MPI-conditioned router `R'[i] = C·(R[i]·M[i])/‖R[i]·M[i]‖₂` with `C = C'/√N` (paper Eq. 4–5). **Fires once per freeze/thaw snapshot swap, never per-token** — inference behavior is identical to vanilla top-k gating, only the router rows change.

- **`power_iter_retract`** (shared helper in `spectral_retract.rs`, always-on) — one or more steps of `v ← v·M` then `v ← target_norm·v/‖v‖₂` on any PSD operator. Zero-alloc, caller-owned scratch. DRY-refactors `gauge_rebalance`'s σ_max power iteration (Plan 270) — both are instances of "power-iteration step + norm retraction against a PSD operator".
- **`manifold_power_iter_router`** — applies the retraction to each router row against its expert Gram. Returns `MpiRouterResult` with `lambda_alignment` (paper Eq. 11) and `maxvio` diagnostics.
- **`gate_sigmoid_topk`** — **independent per-expert sigmoid** `σ(β·x·R'[i]ᵀ)`, then TopK. **Never softmax** (AGENTS.md constraint, G7 enforces).
- **`MpiRouterSnapshotHook`** + `DefaultMpiRouterSnapshotHook` — the freeze/thaw swap boundary hook. BLAKE3-tagged Gram cache keyed by snapshot version; cache hit skips gram recomputation entirely.

**GOAT gate:** G1 (λ alignment gain, `λ(R') ≥ 0.5·λ(R_optimal)`) ✅, G2 (MaxVio reduction `≤ 0.7·MaxVio(R)`) ✅, G3 (zero per-token overhead — gate is identical matmul either way) ✅, G4 (sub-ms swap at game scale `N=8, D=256`: 0.076ms release) ✅, G5 (determinism — byte-identical `R'` across runs, sync-safe) ✅, G6 (DRY non-regression — all 9 `gauge_rebalance` tests pass unchanged) ✅, G7 (sigmoid constraint — perturbing one expert's row leaves others byte-identical) ✅, G8 (`iters=1` sufficiency — captures 100% of `iters=10` gain on rank-1 data) ✅. **9/9 green** (release-build GOAT bench, commit `306cc047`). **Decision: promoted to default-on** (Plan 279 Phase 4 — zero dependencies, DRY win via shared `spectral_retract` helper, GOAT 9/9 green on synthetic rank-1 Gram).

Feature gate: `manifold_power_iter_router` (**default-on** since Plan 279 Phase 4 GOAT 9/9 green). 📖 Plan: [`.plans/279_manifold_power_iter_router.md`](.plans/279_manifold_power_iter_router.md), Research: [`.research/246_Manifold_Power_Iteration_MoE_Router.md`](.research/246_Manifold_Power_Iteration_MoE_Router.md).

### 📡 CS-KV-Importance Probe + Density-Budget Interpolator (Plan 280)

Distills Chen et al. 2026 (arXiv:2606.13594, "See What I See, Know What I Think") into three modelless primitives that together answer: *which KV heads actually matter for a task, and how much budget should each receiver get given its context awareness?* No training, no backprop — the only "learning" is one coordinate-descent Lasso solve on a fixed measurement matrix.

- **`CsKvProbe`** — compressed-sensing KV-group importance probe. Ablate `M` random head subsets (default 200 masks, 5% ablation each), measure the task-quality delta per mask, then Lasso-solve for per-head importance coefficients. Returns a `KvGroupRanking` sorted by importance. On synthetic signal `{3, 17, 42}` the probe recovers all three as top-3 with 0.99/0.96/0.94 scores vs 0.13 for noise heads (G1).
- **`DensityBudget`** — the `K(ca)` interpolator. Given context-awareness `ca \u2208 [0,1]`, returns integer top-K budget interpolating between sparse floor (3.5% of D) and dense ceiling (87% of D). Monotone, bounded, branchless (G3).
- **`GatedKvSlice`** — applies ranking + budget to a KV cache via `log(s + \u03b5)` bias per top-K group, `-\u221e` for the rest. Sigmoid-compatible, never softmax. Zero-allocation apply path (`&mut [f32]` out, verified by T3.5).

**GOAT gate:** G1 (CS beats random by \u226515pp) \u2705, G2 (sparse-vs-dense duality shape reproduces at D=64) \u2705, G3 (K(ca) monotone + bounded) \u2705, T3.4 (zero-overhead when feature off) \u2705, T3.5 (zero-alloc in apply) \u2705. **Decision: opt-in** (`cs_kv_probe` feature) — the open math ships here; NPC wiring + fog-of-war `ca` computation + zone broadcast live in riir-ai Plan 311.

Feature gate: `cs_kv_probe` (**opt-in**). 📖 Plan: [`.plans/280_cs_kv_importance_probe.md`](.plans/280_cs_kv_importance_probe.md), Research: [`.research/247_Dense_Latent_Heterogeneous_Communication_CS_Probe.md`](.research/247_Dense_Latent_Heterogeneous_Communication_CS_Probe.md).

### 🔬 Closure-Expansion Instrument: PTG + Motif Mining + PRI/CDG/TaR (Plan 290, arxiv 2606.15386)

Ships the runtime/data-structure half of Momennejad & Raileanu's *A Compositional Framework for Open-ended Intelligence* — turns any execution into an observable, committable **Primitive Transition Graph (PTG)**, discovers recurring subgraphs (**motifs**), and exposes the paper's §6 evaluation metrics (PRI / CDG / TaR). Measurement layer, not a new capability class.

```mermaid
flowchart LR
    A[Wake phase:<br/>PtgTracedPruner] -->|finish_episode| B[MotifMiner<br/>ring buffer]
    B -->|sleep-cycle boundary| C[mine_motifs_at_sleep_cycle<br/>+ compute_pri + CDG fold]
    C -->|MDL gate| D{MotifAdmitter}
    D -->|admit| E[Register Composite<br/>primitive id]
    D -->|reject| F[Drop]
    E -.->|next wake phase<br/>emits compressed node| A
```

- **`PtgTracedPruner<P: ScreeningPruner>`** — zero-cost decorator that auto-instruments any pruner exposing `AbsorbCompress`. Emits one PTG node per `absorb(arm, reward)` (linked `Sequence`) and one per `compress()` (linked `Branch`, reserved `COMPRESS_PRIMITIVE_ID = 254`). Bandit `update(arm, reward)` traced via explicit `trace()` API. The decode hot path (`relevance()`) is strictly pass-through.
- **`MotifMiner`** — lock-free `papaya`-backed index + 1024-PTG ring buffer. `mine_batch()` runs in rayon at sleep-cycle boundaries (Plan 107 AutoDreamer / Plan 154 Sleep Consolidation), bounded-depth gSpan-lite over ≤4-node motifs.
- **`MotifAdmitter`** — wraps Plan 215's MDL admission gate. Accepts iff `PRI ≥ 0.1` AND `occurrence_count ≥ 3` AND `dl_old_bits > admission_cost`. Admitted motifs register as `PrimitiveKind::Composite(blake3_prefix)` — future PTGs emit a single compressed node.
- **`compute_pri` / `compute_cdg` / `compute_tar_score`** — the paper's §6 metrics as pure functions. TaR is a modelless Jaccard-over-motif-multisets proxy; the real TaR (via `AnchorProfile.translate_priorities()`) lives in riir-ai private IP.
- **Latent bridges** — `ptg_to_motif_embedding` (raw→latent, dot-product + **sigmoid, never softmax**) and `motif_embedding_to_tar_score` (latent→raw scalar, clamped [0,1]). SIMD-friendly via `simd_dot_f32`.

**GOAT gate (G1–G4 must ALL pass for default-on; G5 is demotion):**

| Gate | Target | Measured | Verdict |
|------|--------|----------|---------|
| G1 | PRI < 100µs / 1K traces (hot-tier) | 22–26µs | ✅ PASS (bit matrix + ahash, Issue 035; was 4507µs) |
| G2 | Motif mining < 5% of admission path | 2.07ms mine / 250ns admit | ✅ PASS |
| G3 | TaR correlates with real transfer ≥0.5 | synthetic proxy 1.0/0.0 | ✅ PASS (proxy — real correlation needs riir-ai) |
| G4 | 10K-trace snapshot < 1MB | 1.774MB | ⚠️ PARTIAL (32B per-node blake3 is structural, locked in Phase 0; sole remaining blocker) |
| G5 | Demotion if no quality correlation | N/A | DEFERRED (needs riir-ai transfer traces) |

**Decision: `closure_instrument` stays opt-in.** Per the plan's T4.7 promotion rule, G1–G4 must ALL pass. **G1 is now green** (Issue 035, 2026-06-19: replaced the nested `HashMap<PrimitiveKind, HashSet<u32>>` with a primitive×family bit matrix + ahash + rolling-tag dedup — 22–26µs / 1K traces, was 4507µs, ~180× speedup). **G4 is the sole remaining blocker** and is structural — it stems from the locked Phase 0 data model (`blake3_in: [u8; 32]` per node), not from an implementation bug. All 9 GOAT tests + 9 metrics unit tests + 6 integration tests pass; the wake→sleep→admit loop is proven end-to-end on real `AbsorbCompressLayer<NoScreeningPruner>`.

Feature gate: `closure_instrument` (**opt-in**, requires `katgpt-core/closure_instrument`; auto-tracing of `AbsorbCompress` additionally needs `bandit`). 📖 Plan: [`.plans/290_closure_expansion_instrument.md`](.plans/290_closure_expansion_instrument.md), Research: [`.research/264_Compositional_Open_Ended_Intelligence_Framework.md`](.research/264_Compositional_Open_Ended_Intelligence_Framework.md), Benchmark: [`.benchmarks/290_closure_instrument_goat.md`](.benchmarks/290_closure_instrument_goat.md), Paper: [arxiv 2606.15386](https://arxiv.org/abs/2606.15386).

### 🌿 ICT Distributional Branching-Point Detector (Plan 294, arxiv 2606.19771)

Open, generic, MIT-licensed modelless primitives distilled from ICT (Feng et al., *Beyond Entropy: Detecting Critical Decision Points in LLMs via Distributional Branching*). The paper's training-time selector becomes an **inference-time cognitive-budget allocator**: given K candidate trajectories per tick, spend the full CLR/HLA/KG/curiosity budget only on the ~10% that genuinely diverge from the population mean; the rest run at 10× lower cost.

Three core primitives:
- **`collision_purity(π) = Σ π² = exp(−H₂)`** — ICT §A.2.5 proves ∂β/∂π(a) = 2π(a) > 0 unconditionally. Shannon entropy H₁ only has the right gradient for π(a) > e⁻¹ ≈ 0.37 — β is the correct concentration signal for the long tail.
- **`js_divergence(p, q, scratch)`** — symmetric, bounded `[0, ln 2]`, finite on disjoint supports. ICT §A.5 proves this is the right distributional-novelty metric (KL is asymmetric and infinity on disjoint supports; Wasserstein needs a meaningless ground metric over token indices).
- **`BranchingDetector::observe_and_detect_into(trajectories, &mut report)`** — zero-alloc hot path. Population mean P̄ → per-trajectory `u_k = JS(π_k, P̄)` → top-k% mask → per-step β EMA. Returns a `BranchingReport { mask, beta_per_step, uniqueness_scores }`.

**GOAT gate results (Plan 294 Phases 2–6):**

| Gate | Target | Measured | Verdict |
|------|--------|----------|---------|
| G1 | β distinguishes where H₁ cannot (paper Fig 1a) | ΔH₁ = 1.2e-7, Δβ = 0.12 | ✅ PASS |
| G2 | Median inflection ∈ [5%, 20%] (paper §A.4.1 ~10%) | median 37.5% on synthetic-NPC suite | ⚠️ BORDERLINE-FAIL — paper's 10% is LLM-token-specific; sweep `k_percent` per-domain. [Issue 033](.issues/033_ict_g2_inflection_37_percent_npc_domain.md). Does NOT block G3. |
| G3 ⭐ | Spearman ρ(H₁, JS-uniqueness) < 0.5 (**MAKE-OR-BREAK**) | ρ = 0.0652, 95% CI [-0.017, 0.150] | ✅ PASS — JS captures structurally-different information from H₁. Super-GOAT proceeds. |
| G4 | ≤ 50µs per `observe_and_detect_into` call (K=8, action_dim=32) | mean 1.96µs, p99 2.00µs | ✅ PASS (25× headroom) |
| G5 | 0 allocs/call after warmup | 0 across 1000 calls | ✅ PASS |
| G6 | Feature isolation via cargo + nm | all 3 sub-tests pass | ✅ PASS |
| G10 | H₂ forecast beats H₁ on long-tail regime | MAE 0.402 vs 0.423 (long-tail) | ✅ PASS — Bebop R243 Issue 023 should adopt the H₁→H₂ upgrade |

**Promotion decision (T8.4): `ict_branching` stays opt-in.** G3 alone is necessary but not sufficient for default-on — need G8 (riir-ai Plan 324 runtime fusion validation) too. The runtime fusion (CLR gating at branching moments, HLA updates at branching moments, KG emission at branching moments, curiosity bursts at branching moments) lives in `riir-ai` Plan 324 — out of scope for this open `katgpt-rs` primitive.

**What ships regardless of promotion:**
- The math primitives (`collision_purity`, `renyi_h2`, `shannon_h1`, `js_divergence`) — useful anywhere we currently reach for entropy as a concentration signal.
- `AcceptanceForecastH2` — the Bebop H₁→H₂ drop-in upgrade (G10 PASS). Independent of the runtime fusion, this is the broadly-valuable piece.
- The Curiosity Pulse (R041) H₁→β drop-in spec (reference doc only — implementation in riir-ai Plan 274).

**Reproducibility:** every gate runs from `cargo test --features ict_branching --test bench_294_ict_gN`. Synthetic LCG seeds are fixed for byte-identical reruns.

Feature gate: `ict_branching` (**opt-in** — `katgpt-core/ict_branching` re-exported at root). 📖 Plan: [`.plans/294_ict_branching_detector.md`](.plans/294_ict_branching_detector.md), Research: [`.research/270_Beyond_Entropy_ICT_Distributional_Branching_Detector.md`](.research/270_Beyond_Entropy_ICT_Distributional_Branching_Detector.md), Benchmarks: [G1](.benchmarks/294_ict_g1.md) · [G2](.benchmarks/294_ict_g2.md) · [G3](.benchmarks/294_ict_g3.md) · [G4–G6](.benchmarks/294_ict_goat_gates.md) · [G10](.benchmarks/294_ict_g10.md), Issue: [033](.issues/033_ict_g2_inflection_37_percent_npc_domain.md), Paper: [arxiv 2606.19771](https://arxiv.org/abs/2606.19771).

### 🧠 MicroRecurrentBeliefState — Attractor/Leaky Belief Kernel (Plan 276, arxiv 2604.17121)

Distills Mozer, Siddiqui & Liu (DeepMind, 2026) *The Topological Trouble With Transformers* into a generic `BeliefKernel` trait unifying a leaky-integrator family (delta-rule SSM) with an **attractor family** (`s_t = σ(W_s·s_{t-1} + W_x·x_t + b)`) for belief-with-hysteresis. The trait exposes `step()` and `project_to_scalars()` via dot-product + sigmoid bridge (never softmax).

**Two modelless primitives, both sigmoid-compatible:**
- `BeliefKernel` trait — unifies Family A (attractor, sigmoid-bounded) and Family C (leaky integrator).
- `AttractorKernel` — the GOAT candidate. σ-bounded step prevents long-horizon flip-flop.

**Verdict:** revised Super-GOAT → GOAT after prior-art check. **G1.1–G1.4 PASS** (determinism, boundedness, bridge ranking, latency). **G2 (attractor coherence) deferred** to a long-horizon benchmark; attractor family stays opt-in behind a sub-flag if it loses.

Feature gate: `micro_belief` (**opt-in** — ships trait unification + attractor family; attractor variant not promoted until G2 passes). Snapshot/hot-swap integration lives in `riir-ai`. 📖 Plan: [`.plans/276_micro_recurrent_belief_state.md`](.plans/276_micro_recurrent_belief_state.md), Research: [`.research/242_Topological_State_Tracking_Recurrent_Belief.md`](.research/242_Topological_State_Tracking_Recurrent_Belief.md), Paper: [arxiv 2604.17121](https://arxiv.org/abs/2604.17121).

### 🎲 BoMSampler — K-Hypothesis Single-Pass Belief Sampling (Plan 281, arxiv 2604.04913)

Distills Kerssies et al. *A Frame is Worth One Token: Efficient Generative World Modeling with Delta Tokens* (Apr 2026) into a single novel inference primitive — **K diverse next-belief-states per tick in one batched kernel evaluation**, by injecting K Gaussian noise queries at the kernel input site. `BoMSampler` trait extends `MicroRecurrentBeliefState` (Plan 276); the deterministic `step()` path is unchanged.

```text
Inputs:  s_prev ∈ ℝ^D, x ∈ ℝ^D, queries[0..K-1] ∈ ℝ^D_q
                │
                ▼
  ┌─────────────────────────────────────────┐
  │ act[i] = W_s[i]·s_prev + W_x[i]·x + b[i] │   1 matvec (D dots)
  └───────────────────┬─────────────────────┘
                      │
                      ▼  add queries, sigmoid K×
  ┌─────────────────────────────────────────┐
  │ for k in 0..K:                            │
  │   out[k] = σ(act + W_q·queries[k])       │  K× (D adds + D sigmoids)
  └───────────────────┬─────────────────────┘
                      │
                      ▼
  K diverse next-belief-states (single kernel eval)
```

**NoiseQueryConfig** is its OWN `commit()` (separate BLAKE3 over `sigma_le || k_le || seed_strategy_byte`); the kernel snapshot is unchanged. Paper trains K=256, evals K=20; we default **K=8 (plasma-tier budget)**.

| Gate | Target | Measured | Verdict |
|------|--------|----------|--------|
| **G1.1** Determinism (fixed seed, bit-identical `out[k]`) | byte-identical | byte-identical | ✅ PASS |
| **G1.2** K-distribution spread | σ(K unique vectors) > 0 | true for σ > 0 | ✅ PASS |
| **G1.3** SIMD speedup vs scalar | K=8 ≥ 1.5× | **1.87×** (via `simd_sigmoid`) | ✅ PASS |
| **G2** Arena win-rate uplift | > 0 vs 1-deterministic-belief | **+31.49pp** (riir-ai Plan 314: MultiThreatArena + MultiHypothesisBoMMinimaxPlanner vs deterministic) | ✅ PASS |
| **G3** SIMD Sigmoid step-rate | K=8 ≤ 2× baseline | 1.87× (Issues 024/025 closed) | ✅ PASS |

**Verdict: Gain** (not GOAT, not Super-GOAT — see Research 248 §3). The G2 arena win is the deciding result. **Promoted to default-on** in `katgpt-core` (T2.4 full, 2026-06-17). Stays **opt-in at `katgpt-rs` root** until T2.3 wiring (NPC tick dispatch, minimax-over-K-beliefs planner, ANE batch dispatch) lands in riir-ai.

Feature gate: `bom_sampling` (**DEFAULT-ON** in katgpt-core; **opt-in** in katgpt-rs root). Auto-enables `simd_sigmoid` (G3 PASS). 📖 Plan: [`.plans/281_bom_single_pass_diverse_sampling.md`](.plans/281_bom_single_pass_diverse_sampling.md), Research: [`.research/248_DeltaTok_DeltaWorld_BoM_Single_Pass_Diverse_Sampling.md`](.research/248_DeltaTok_DeltaWorld_BoM_Single_Pass_Diverse_Sampling.md), Paper: [arxiv 2604.04913](https://arxiv.org/abs/2604.04913).

### ⚡ Temporal Derivative Kernel — Dual Fast/Slow Surprise Signal (Plan 277, arxiv 2606.08720)

Distills O'Reilly 2026 *This is how the Neocortex Learns* into a generic, zero-allocation, sigmoid-compatible **dual fast/slow temporal-derivative kernel**. Turns any streaming latent scalar/vector into a signed "surprise" signal — the implicit prediction-error channel the neocortex uses for credit assignment, computed locally from a signal's own time series with no external target and no backprop.

```text
  observe(signal):
    fast = (1 - α_fast)·fast + α_fast·signal      (high-pass: tracks what's happening now)
    slow = (1 - α_slow)·slow + α_slow·signal      (low-pass: tracks what's stable)
    return fast - slow                            (band-pass: tracks how fast it's changing)

  surprise_norm = ‖fast - slow‖₂                   (0 when stable, spikes on novelty)
  curiosity_gate = sigmoid(β · surprise_norm)     (AGENTS.md sigmoid, never softmax)
```

**Composes with existing belief-state and curiosity primitives** — four fusion gates passed (per Research 243): state-vector companion, surprise-gated memory writes, derivative-augmented collapse detection, and zero-cost sigmoid curiosity signal. Consumer wiring lives in `riir-ai`.

**All 4 fusion gates PASS** → kernel primitive promoted to **default-on** (T6 final). Microbench: `observe` N=8 at 7.9ns (< 10ns target).

Feature gate: `temporal_deriv` (**DEFAULT-ON** since GOAT 4/4 fusions passed). Auto-enabled by `bom_sampling` for the sigmoid-surprise gate. 📖 Plan: [`.plans/277_temporal_derivative_kernel.md`](.plans/277_temporal_derivative_kernel.md), Research: [`.research/243_Temporal_Derivative_Kernel_Neocortical_Learning.md`](.research/243_Temporal_Derivative_Kernel_Neocortical_Learning.md), Paper: [arxiv 2606.08720](https://arxiv.org/abs/2606.08720).

### 🛡️ Self-Advantage Gate — Dead-Compute Detector via Pre/Post Log-Ratio (Plan 283, arxiv 2511.16886)

Distills Asadulaev et al. *Latent Reasoning in TRMs is Secretly a Policy Improvement Operator* (ICML 2026) into three modelless primitives. The paper proves latent recursion is a policy improvement operator in disguise; we extract the inference-time consequence — detect when a recursion step is **dead compute** and skip it.

```text
  self_advantage(pre, post, candidate) :=
      A(candidate) - E_{a∼π_w}[A(a)]
      where A(a) = log π+(a) - log π̂(a)

  AdvantageMarginGate::should_recurse(pre, post, candidate):
      return self_advantage_margin(pre, post, candidate, scratch) > 0
      // positive margin → recursion benefits this candidate → recurse
      // negative margin → dead compute → skip
```

**Three primitives, all modelless (no teacher, no oracle):**
- `self_advantage()` — log-ratio `A(a) = log π+(a) − log π̂(a)` between pre- and post-recursion logits. Zero-alloc: writes into caller-provided scratch.
- `AdvantageMarginGate` — accept recursion step iff `A(y*) > E_a[A(a)]` (paper Eq. 18). Paper claims **18× forward pass reduction**.
- `product_policy()` — inference-time multiplicative interpolation `π_w ∝ π̂^{1−w} · π+^w` (paper Eq. 16). Controllable reasoning trust weight `w`.

**GOAT 4/4 PASS** (vocab ≤ 128 operating range, Bench 056/057):

| Gate | Target | Measured | Verdict |
|------|--------|----------|--------|
| **G1** Skip detection on identical pre/post | 0% argmax change | 0% | ✅ PASS |
| **G2** Skip count on dead-compute traces | > 0 skips | significant skips | ✅ PASS |
| **G3** Step reduction at vocab ≤ 128 | ≥ 2× | met | ✅ PASS |
| **G4** Argmax match vs ungated | 100% | 100% | ✅ PASS |

**Belief-state integration (T5.1):** the gate composes with existing sigmoid-bounded belief-state early-stop criteria. GOAT 3/3 PASS → Bench 057.

Feature gate: `self_advantage_gate` (**DEFAULT-ON** since GOAT 4/4 PASS). Deep integrations T2.2/T2.3 + freeze/thaw T5.3 remain **deferred** in [Issue 028](.issues/028_self_advantage_gate_integration_followups.md). 📖 Plan: [`.plans/283_self_advantage_recursion_gate.md`](.plans/283_self_advantage_recursion_gate.md), Research: [`.research/250_Latent_Recursion_Policy_Improvement_Advantage_Margin.md`](.research/250_Latent_Recursion_Policy_Improvement_Advantage_Margin.md), Paper: [arxiv 2511.16886](https://arxiv.org/abs/2511.16886).

### 🔏 Forensic Watermark — Moved to riir-ai (Plan 322)

The forensic watermark recipe primitive (Plan 293, arxiv 2606.18208) was relocated from katgpt-rs to `riir-ai/crates/riir-chain/src/forensic/` behind the `chain_forensic` feature. Rationale: honeypot OPSEC — the recipe combination (Tardos + DCT + topology + vertex marks + least-squares recovery) is the implementation choice that determines collusion resistance, and forensic value depends on deployment secrecy. Per strategy verdict 003: "How = private." An open trait surface may return here later if a generic adoption hook is needed; the recipe impl stays private.

## 🔧 KV Compression

Default: **Hybrid OCT+PQ** (OCTOPUS triplet encoding + PlanarQuant 2D Givens rotation). Best MSE + 64× fewer rotation FMAs.

| Backend | Rotation | FMAs (d=128) | MSE (3-bit) | Calibration |
|---------|----------|-------------|-------------|-------------|
| **Hybrid OCT+PQ** ⭐ | 2D Givens | 256 | 0.026 | 0 samples |
| OCTOPUS | WHT (full) | 16,384 | 0.026 | 0 samples |
| SpectralQuant | Eigenbasis | 16,384 | 0.038 | 256 samples |
| PlanarQuant | 2D Givens | 256 | 0.034 | 0 samples |
| TurboQuant | Random | 16,384 | 0.034 | 0 samples |

📖 **Full comparison tables, benchmarks, code examples:** [`.docs/19_kv_compression.md`](.docs/19_kv_compression.md).

## 🔀 Opt-In & Gated Features

| Feature | What | Status |
|---------|------|--------|
| **D2F / Tri-Mode** | Block-parallel denoising + AR self-speculation | Experimental decode strategy |
| **G-Zero** (`g_zero`) | Hint-δ self-play + arena players | Bench-only, does NOT touch forward() |
| **GameState** (`game_state`) | Generic MCTS, STRATEGA forward model | Arena-specific |
| **SpecHop** (`spechop`) | Hop-level speculation for multi-step agents | Awaiting GOAT proof |
| **Percepta** (full) | Transformer-VM with WASM interpreter in weights | Research-grade |
| **Sense Composition** (`sense_composition`) | Ternary bit-plane projection for sense-module context. Recurrent belief state + sigmoid-dot bridge wiring live in `riir-ai` | Opt-in — requires `plasma_path`, `domain_latent` |
| **BAKE Precision** (`bake_precision`) | Per-dimension Bayesian precision tracking for KG embeddings | GOAT 10/10, drift marginal (4.7%) |
| **NFCoT FlowScore** (`nf_flow`) | Normalizing flow density scoring for speculative candidates | GOAT ⚠️ MARGINAL, all sub-features default OFF |
| **FOL Constraints** (`fol_constraints`) | DDTree→FOL logical rule extraction | GOAT 6/6 |
| **AND-OR DDTree** (`and_or_dtree`) | Hierarchical subgoal decomposition | GOAT proven |
| **Trigger Gate** (`inference_router`) | CPU → GPU → ANE tier routing | CPU ✅, GPU/ANE blocked on hardware deps |
| **SLoD** (`slod`) | Poincaré ball hyperbolic geometry + heat diffusion tier routing | **default-ON**, GOAT G1–G6 pass |
| **Schema Centroid** (`schema_centroid`) | Per-class embedding centroids for informed KG entity init | **default-ON**, GOAT 7/7 |
| **Shard Embedding** (`shard_embedding`) | JL random orthogonal projection [f32;64]→[f32;8] | Always compiled in `katgpt-core` |
| **DFlare** (Plan 174) | Marginal fusion + KV routing + progressive budget | 🪦 GOAT FAILED on all 3 sub-features |
| **ManifoldPruner** (Plan 234) | ManifoldE point-to-manifold soft validity | 🪦 GOAT G1 FAIL |
| **MUX-Latent Wire** (`mux_latent_wire`) | Latent-to-latent patching over wire, 68B format, SIMD batch | Opt-in — GOAT 11/11, awaiting E2E integration |
| **RAT+ Bridge** (`rat_plus_bridge`) | GDN2 recurrent state as dilated sparse attention bridge | Opt-in — GOAT gated, D=16 proven |
| **TRDraft** (`trd_refined_draft`) | Trajectory-refined draft: re-draft failed DDTree branches | GOAT proven, opt-in |
| **Vocab Channel Pruner** (`vocab_channel`) | ROTATE MLP weight decomposition → DDTree pruning | GOAT 6/7 conditional |
| **MSA Sparse** (`msa_sparse`) | Blockwise sparse attention distillation into VortexFlow | Opt-in — GOAT gated |
| **GPart Adapter** (`gpart_adapter`) | Isometric partition matrix, 2-100× compression vs LoRA | Opt-in — GOAT gated |
| **LinOSS Threat** (`linoss_threat`) | Oscillation dynamics for anticipatory NPC threat prediction | Opt-in — pending benchmark |
| **Fourier Flow** (`flow_field_nav`) | FFT-smoothed shared flow fields for O(1) crowd navigation | GOAT PASS 46.9%, opt-in |
| **StillKV** (`still_kv`) | Perceiver-based KV compaction with heuristic query banks | Opt-in — pending GOAT proof |
| **ECHO Predictor** (`echo_predictor`) | Inference-time prediction scoring for policy quality | Opt-in — pending GOAT proof |
| **Merkle Octree** (`merkle_octree`) | Node-tier curator consensus with BLAKE3 commitment | Opt-in — modelless verification |
| **ANE NPC Brain** (`ane_npc`) | Move NPC think-brain compute to Apple ANE batch | Opt-in — GOAT gated |
| **DendriticGate** (`dendritic_gate`) | NMDA-inspired adaptive DDTree branching via entropy+coincidence | In progress — GOAT gated |
| **Closure-Expansion Instrument** (`closure_instrument`) | PTG recorder + motif miner + PRI/CDG/TaR metrics (Momennejad & Raileanu 2026, arxiv 2606.15386). `PtgTracedPruner` wraps any `ScreeningPruner`; `mine_motifs_at_sleep_cycle()` runs at sleep-cycle boundaries. Fuses with Plan 215 MDL gate, MUSE lifecycle, AnchorProfile transfer. | Opt-in — G2/G3 PASS, G1/G4 PARTIAL (structural: std HashMap + per-node blake3); not promoted |
| **MicroRecurrentBeliefState** (`micro_belief`) | Generic `BeliefKernel` trait unifying attractor + leaky-integrator families. | Opt-in — G1.1–G1.4 PASS; G2 (attractor coherence) deferred. Auto-enabled by `bom_sampling`. |
| **BoMSampler** (`bom_sampling`) | K-hypothesis single-pass belief sampling (Plan 281, arxiv 2604.04913). `BoMSampler` extends `MicroRecurrentBeliefState`. | **DEFAULT-ON** in `katgpt-core` (G2 PASS +31.49pp). Opt-in at katgpt-rs root. Auto-enables `simd_sigmoid`. |
| **CompressionDrafter** (`compression_drafter`) | LZ4 corpus-as-model drafter (Plan 285, nathan.rs/gzip-lm) | 🪦 GOAT FAILED (2 runs) — stays opt-in, unused. `TernaryDraftModel` remains Hot-tier default. |
| **FuncAttn** (`funcattn`) | Functional Attention — closed-form Tikhonov k×k spectral transport (Plan 286, arxiv 2605.31559) | 🪦 G6 FAIL on LM prediction (0.969 < SDPA 1.000). Stays opt-in, NOT default. Gain-tier. |
| **Forensic Watermark** | Moved to `riir-ai` (Plan 322) — recipe implementation relocated to preserve honeypot value per strategy verdict 003 | — |
| **ICT Branching Detector** (`ict_branching`) | `collision_purity β(π)` + JS-divergence novelty + `BranchingDetector` (Plan 294, arxiv 2606.19771) | Opt-in — G1/G3/G4/G5/G6/G10 PASS (Super-GOAT proceeds); G8 (runtime fusion) deferred to riir-ai Plan 324. |

📖 **Full detail for ALL opt-in features + complete feature flag reference:** [`.docs/21_opt_in_features.md`](.docs/21_opt_in_features.md) and [`Cargo.toml`](Cargo.toml).

## 🛠️ Getting Started

### Prerequisites

- Rust 1.85+ (edition 2024, 1.93+ recommended)

### Build & Run

```sh
cargo build --release                              # Build with optimizations
cargo run --release                                # Run benchmark + generate plot
cargo run --release --all-features                 # Run everything
cargo test --quiet --workspace --all-features       # Run all tests (245 test files)
cargo run --example sudoku_01_9x9 --features sudoku # Sudoku solver
cargo clippy --all-targets --all-features --quiet   # Lint
```

### Feature Flags

**310+ feature flags** with **130+ default-on** (all GOAT-proved). Default features include: `sparse_mlp`, `domain_latent`, `ppot`, `bandit`, `bt_rank`, `spectral_quant`, `hybrid_oct_pq`, `elf_sde`, `cna_steering`, `deep_manifold`, `federation`, `gdn2_attention`, `dash_attn`, `lt2_looped`, `kv_share`, `kvarn`, `belief_drafter`, `bfcf_lfu_shard`, `mux_latent_context`, `collapse_aware_thinking`, `slod`, `schema_centroid`, `union_bound_confidence`, `pathway_tracker`, `federation_composer`, **`posterior_evolution`**, **`spectral_pruner`**, **`breakeven_routing`**, **`substrate_gate`**, **`regime_transition`**, `rcd_residual`, `lattice_operad`, `spec_pruner`, `caddtree_budget`, `ssd_block`, `ss_pruner`, `dendritic_gate`, `sparse_task_vector`, `off_principal_retrieval`, `spectral_rank`, `module_energy_route`, `gauge_invariant`, `chiaroscuro`, `attn_match`, **`manifold_power_iter_router`** (Plan 279 GOAT 9/9), **`triggered_injection`** (Plan 278 G3 PASS), **`temporal_deriv`** (Plan 277 4/4 fusions PASS), **`self_advantage_gate`** (Plan 283 GOAT 4/4 PASS), **`clr`** (Plan 284), and 80 more.

📖 **Full feature flag table (310+ flags):** [`.docs/21_opt_in_features.md`](.docs/21_opt_in_features.md) and [`Cargo.toml`](Cargo.toml).

## 📁 Project Structure

```
crates/katgpt-core/   Shared types + SIMD kernels + traits (consumed by katgpt-rs + riir-engine)
  types.rs            Decoupled structs (Config, Rng, LoraAdapter, DomainLatent, ShardEmbedding, DataGate, ...)
  traits.rs           Core trait definitions (18 traits + helper structs)
  simd.rs             SIMD kernel implementations (NEON/AVX2) — incl. `simd_sigmoid` (Issues 024/025 M1)
  shard_embedding.rs  JL random orthogonal projection [f32;64]→[f32;8]
  attention.rs        Tiled online-softmax flash attention
  coda.rs             CODA fused SIMD kernels
  parallax_attn.rs    Parallax parameterized local linear attention
  funcattn.rs         Functional Attention — Tikhonov k×k spectral transport operator (Plan 286)
  peira.rs            PEIRA inter-view regressor alignment
  dirichlet.rs        Dirichlet Energy structural alignment diagnostic
  spectral_hierarchy.rs  Eigenspace alignment, Haar wavelets, Cauchy interlacing
  spectral_retract.rs Shared power-iteration retraction (Plan 270 gauge + Plan 279 MoE router DRY)
  roofline.rs         Roofline cost model for GPU operator runtime prediction
  questbench.rs       QuestBench underspecification scoring
  linoss.rs           LinOSS oscillatory state-space cell + ModalSpec drafter
  irrep_pruner.rs     Spectral Irrep Pruner (spectral flatness decoding pruning)
  temporal_deriv.rs   Temporal Derivative Kernel — dual fast/slow EMA surprise signal (Plan 277)
  compression_drafter.rs LZ4 corpus-as-model drafter (Plan 285)
  merkle.rs           Merkle octree hierarchical BLAKE3 commitment
  curator.rs          Curator verification layer for Merkle octree
  dendritic_gate.rs   NMDA-inspired adaptive DDTree branching
  slod.rs             SLoD Spectral Level-of-Detail Pruner (Poincaré ball)
  sense/              Sense Composition modules (Plan 240)
  micro_belief/       BeliefKernel trait + Attractor/Leaky family + BoMSampler (Plans 276, 281)
  faithfulness/       FaithfulnessProbe + TriggeredInjectionGate (Plan 278)
  forensic/           Forensic watermark recipe primitive (Plan 293)
  ict/                ICT Distributional Branching-Point Detector (Plan 294)
  closure/            Closure-Expansion Instrument: PTG + motif mining (Plan 290)
  content_store/      ChunkedContentStore — content-addressed Merkle store (Plan 272; asset pipeline lives in riir-ai)
  and_or/             AND-OR DDTree blueprint decomposition
  mux/                MUX superposition pruning (span pruner, DDTree, BFS, bandit, freeze/thaw, demux)
  bridge/             Generic latent→raw action bridge
  cgsp/               Curiosity-Guided Self-Play triad (Solver/Conjecturer/Guide)
  dec/                Discrete Exterior Calculus operators
  flow/               Fourier-smoothed flow fields for LEO crowd navigation
  qgf/                Q-Guided Flow — test-time Q-gradient guidance
src/
  transformer.rs      Weights, KVCache (flat/paged/raven), forward/generate
  speculative/        DDTree, DFlash, Verifier, Prefill, D2F, budget, flashar
  pruners/            BanditPruner, TrialLog, HotSwap, BT Rank, CNA, G-Zero, Arena, SelfAdvantageGate (Plan 283)
  tokenizer/          BPE tokenizer
  validator/          SynPruner + PartialParser
  benchmark/          Benchmark framework (multi-category, CSV timeseries)
  gdn2/               Gated DeltaNet-2 recurrent attention
  dash_attn/          DashAttention adaptive sparse attention
  hybrid_oct_pq/      Default KV codec (OCT + PlanarQuant)
  chiaroscuro/        CHIAR per-token DCT spectral entropy operator routing
  ...                 45 additional submodules + 50 top-level modules
examples/            178 examples (see examples/README.md)
tests/               245 integration test & benchmark files
benches/             Criterion benchmarks
```

## 📖 Documentation Index

- [Architecture overview](.docs/01_overview.md)
- [Full architecture detail](.docs/02_architecture.md)
- [Speculative decoding, D2F](.docs/03_speculative_decoding.md)
- [Benchmarks, throughput tables](.docs/04_performance.md)
- [Sudoku solver detail](.docs/05_sudoku.md)
- [Validator detail](.docs/06_validator.md)
- [Adaptation strategies](.docs/07_adaptation.md)
- [PFlash techniques](.docs/08_lucebox_techniques.md)
- [HL infrastructure, FFT benchmarks](.docs/09_heuristic-learning.md)
- [Bomberman arena](.docs/10_bomber_arena.md)
- [Monopoly FSM](.docs/11_monopoly_fsm.md)
- [FFT Tactics Arena](.docs/12_fft_arena.md)
- [MTP threshold guide](.docs/13_mtp_threshold_guide.md)
- [Go arena](.docs/14_go_arena.md)
- [Paper feature comparison](.docs/15_paper_feature_comparison.md)
- [SpecHop architecture](.docs/16_spechop_architecture.md)
- [PEIRA distillation](.docs/17_peira_distillation.md)
- [Sleep consolidation](.docs/18_sleep_consolidation.md)
- [KV compression alternatives](.docs/19_kv_compression.md)
- [Negative results](.docs/20_negative_results.md)
- [Opt-in features + full feature flag reference](.docs/21_opt_in_features.md)
- [Percepta full detail](.docs/22_percepta.md)
- [HL & Arena detail](.docs/23_hl_arena_detail.md)
- [NPC Sense Composition](.docs/24_sense_composition.md)
- [Raven RSM — Opt-in O(1) routing slot memory](.docs/25_raven_rsm.md)
- [Progressive MCGS — graph search with reference edges](.docs/progressive_mcgs.md)
- [Open-ended problem evolution arena](.docs/191_open_ended_problem_evolution_arena.md)
- [178 examples grouped by category](examples/README.md)
- [DEC Operators & Cubical Topology](.plans/251_dec_operators_cell_complex.md)
- [Spectral Budget Router](.plans/254_spectral_budget_router.md)
- [Posterior-Guided Pruner Evolution](.plans/239_posterior_guided_pruner_evolution.md)
- [Regime-Transition Inference](.plans/215_regime_transition_inference.md)
- [SubstrateGate Capability Routing](.plans/216_substrate_gate_capability_routing.md)
- [Breakeven Complexity Routing](.plans/250_breakeven_inference_routing.md)

## 📜 References

- [Andrej Karpathy's microgpt](https://karpathy.github.io/2026/02/12/microgpt/)
- [microgpt-c](https://github.com/nicholasgasior/microgpt-c) — Original C implementation
- [talos-vs-macbook](https://github.com/AlexCheema/talos-vs-macbook) — Reference model
- [Percepta](https://www.percepta.ai/blog/can-llms-be-computers) — 2D convex hull attention, WASM in transformer weights
