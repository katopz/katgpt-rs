# KatGPT-RS

A neuro-symbolic micro-Transformer with speculative decoding, constraint pruning, recurrent attention, and adaptive test-time scaling — built in Rust.

Inspired by [microgpt-c](https://github.com/nicholasgasior/microgpt-c), [talos-vs-macbook](https://github.com/AlexCheema/talos-vs-macbook), and [Luce-Org/lucebox-hub](https://github.com/Luce-Org/lucebox-hub/).

## 🚀 Key Features

- **Real Transformer Inference** — Full GPT forward pass with RMSNorm, multi-head causal attention, ReLU MLP, KV cache, and temperature sampling.
- **Zero-Alloc Forward Pass** — Pre-allocated `ForwardContext` buffers eliminate heap allocations per inference step.
- **DDTree (Dynamic Draft Tree)** — Best-First Search using a `BinaryHeap` to build a candidate token tree from marginal log-probabilities.
- **ConstraintPruner** — Pluggable trait for neuro-symbolic intercept: deterministic rules engine prunes invalid branches before target verification.
- **ScreeningPruner** — Upgraded binary pruning to graded relevance (`R ∈ [0.0, 1.0]`) with blended score formula.
- **SpeculativeVerifier** — Swappable verification via trait: `SimulatedVerifier` (fast) or `LeviathanVerifier` (real p/q rejection sampling).
- **Raven RSM** — O(1) KV cache replacement with sparse Top-K routing. Unselected slots completely frozen.
- **Hybrid OCT+PQ KV Cache** — Default codec: OCTOPUS triplet encoding + PlanarQuant 2D Givens rotation. Best MSE + 64× fewer rotation FMAs (Bench 024, Plan 101).
- **PFlash Block-Sparse Prefill** — Up to 21× sequence reduction with 100% NIAH needle retrieval.
- **BPE Tokenizer** — Train/encode/decode with Config::bpe() preset for code generation.
- **Bomberman Arena** — 4-player HL proof: adaptive intelligence (+177) > greedy (+131) > static rules (-30) > random (-55).
- **G-Zero Self-Play** — Verifier-free Hint-δ intrinsic reward — no external LLM judge needed.

📖 **Deep dives:** [`.docs/`](.docs/) for architecture, speculative decoding, performance, sudoku, validator, HL, arena, and all research detail.

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

### Core Pipeline

```
LLM drafts logits → ConstraintPruner filters invalid → DDTree builds valid-only tree → Target verifies
```

### Key Traits

```rust
pub trait ConstraintPruner: Send + Sync {
    fn is_valid(&self, depth: usize, token_idx: usize, parent_token: &[usize]) -> bool;
}

pub trait ScreeningPruner: Send + Sync {
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32;
}

pub trait SpeculativeVerifier: Send + Sync {
    fn speculate(&mut self, draft_weights, draft_config, token, pos, rng) -> Vec<usize>;
}
```

### Routing & Conditioning

- **Prompt Router** — `KeywordRouter` scores prompt against domain keywords, `ExpertRegistry` selects `ScreeningPruner` + LoRA.
- **Embedding Router** — Three-tier fallback: embedding search → domain classify → keyword (local).
- **Bidirectional Prefill** — Prompt tokens attend to ALL other prompt tokens (no causal mask during prefill).
- **Modality LoRA Switching** — `reader_lora` active during prefill, `writer_lora` active during decode. Reference swap, zero data movement.
- **PPoT** — Logit-parameterized CPU resampling on failure. Zero overhead on success path.

📖 See [`.docs/02_architecture.md`](.docs/02_architecture.md) for full details.

## 🔄 E2E Inference Flow — Default GOAT Stack

The default production stack flows through these layers. Each item is default-on, GOAT-proved.

```mermaid
graph LR
    subgraph Input
        A[Tokenizer] --> B[PFlash/DashAttn Prefill]
    end
    subgraph Model
        B --> C[Transformer Forward]
        C --> D[Raven RSM]
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
| **Raven RSM** | O(1) KV cache with 16-slot Top-K routing | always |
| **Hybrid OCT+PQ** | Default KV codec — OCT triplet + PQ 2D Givens, best MSE | `hybrid_oct_pq` |
| **SpectralQuant** | Calibrated eigenbasis + water-fill (secondary) | `spectral_quant` |
| **MLS Aggregate** | Average last K layer residuals before LM head | `mls_aggregate` |
| **Domain Latent** | Mid-layer K/V injection | `domain_latent` |
| **Delta Routing** | Cross-layer residual delta routing | `delta_routing` |
| **PPoT** | CPU logit resampling at high-entropy positions | `ppot` |

### Attention (O(1) alternatives)

| Component | What | Gate |
|-----------|------|------|
| **GDN2** | Gated DeltaNet-2 — O(1) decode, constant state per head | `gdn2_attention` |
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
| **Sleep Consolidation** | Offline recursive memory consolidation at KV eviction | `sleep_consolidation` |
| **Dreamer** | Offline memory consolidation (Q-value clustering) | `dreamer` |
| **PlasmaPath** | Bit-plane ternary SIMD matvec, 1.58 bits/weight | `plasma_path` |
| **MoA Inference** | Token-adaptive Mixture-of-Activations SwiGLU | `moa_inference` |
| **Newton-Schulz** | Cubic fixed-point orthogonalization + Muon momentum | `newton_schulz` |
| **Spectral Hierarchy** | Eigenspace alignment, Haar wavelets, Cauchy interlacing | `spectral_hierarchy` |
| **Dual-Gram PCA** | Short-sequence calibration via dual-gram routing | `dual_gram_pca` |
| **Roofline Cost** | GPU operator runtime prediction (~5µs CPU) | `roofline_cost` |
| **River-Valley** | Subspace ratios, effective rank, update cosine | `river_valley` |
| **LEO All-Goals** | Vectorized Bellman all-goals Q-value framework | `leo_all_goals` |
| **Dual LEO** | Teacher/student Q-value mixing + autocurriculum | `dual_leo` |
| **Sigmoid Margin** | SigLIP softplus loss + dimension sufficiency bound | `sigmoid_margin` |
| **Kog CPU Fusion** | RMSNorm gamma folding + QKV interleaving | `kog_cpu_fusion` |
| **PEIRA Distill** | Collapse-free inter-view regressor alignment | `peira_distill` |
| **ILC Distill** | Synonym-aware DDTree pruning via offline k-means | `ilc_distill` |
| **GEPA-D Reflective** | Pareto bandit config evolution | `gepa_reflective` |
| **Hydra Budget** | Emergent self-repair layer skipping | `hydra_budget` |
| **Subterranean** | Token-rewriting procedures compiled to native code | `subterranean` |
| **EqR Convergence** | Smallest marginal-change residual selection | `eqr_convergence` |
| **Thinking Prune** | FrozenBaseGuard for intermediate steps | `thinking_prune` |

📖 **Full GOAT audit table** with research source, real gain, and replaced feature: See [`.docs/01_overview.md`](.docs/01_overview.md).

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

## 📊 Benchmark Results

📖 Raw throughput tables, GRAM width-vs-depth, and per-benchmark explanations: [`.docs/04_performance.md`](.docs/04_performance.md).

### MoE+SD Cost Model

Amdahl cost model for LeviathanVerifier speculative decoding. Feature gate: `spec_cost_model`.

| Proof | Result |
|-------|--------|
| SpecCostSnapshot construction | ✅ |
| Amdahl prediction accuracy | ✅ |
| f_sparse consistency | ✅ < 10% variance |
| Cost model error bound | ✅ < 15% |

## 🦅 Raven RSM: O(1) Routing Slot Memory

Fixed-size slot memory with sparse Top-K routing. Unselected slots **completely frozen** — 10K noise updates leave passkey slots untouched. 2.98× faster than flat attention at pos=8.

| Property | Evidence |
|----------|----------|
| Frozen slots work | 10,000 noise updates, slot 12 identical to 6 decimals |
| O(1) stays flat | Raven stays 1.0× while flat grows 1.1× from pos 16→240 |
| 2.98× faster | 62,653 tok/s (Raven) vs 21,019 tok/s (flat) |

📖 See [`.docs/08_lucebox_techniques.md`](.docs/08_lucebox_techniques.md).

## 🔬 Percepta: Transformer-VM in Rust

Rust port of [Percepta's transformer-vm](https://github.com/Percepta-Core/transformer-vm) — O(log N) 2D convex hull attention with ternary search. **~9K lines Python+C++ → idiomatic Rust.** Apache-2.0.

**Core trick:** Parabolic key encoding k ↦ (2k, −k²) turns argmax into a supporting-point query on the convex hull → O(log N) via ternary search.

Feature flags layer: `percepta` → `percepta_gates` → `percepta_graph` → `percepta_wasm` → `percepta_compile`. All 11 task groups (TG-A through TG-K) complete except TG-K (examples/docs).

📖 **Full detail:** [`.docs/22_percepta.md`](.docs/22_percepta.md) — feature flags, module structure, compiler stack, verified properties.

## 🎮 Arena Proofs — HL Thesis Validated

Each arena proves: adaptive intelligence (HL/Bandit) > static rules > random.

| Arena | Result | Feature |
|-------|--------|---------|
| **Bomberman** | HL (+177) > Greedy (+131) > Validator (-30) > Random (-55) | `bomber` |
| **Monopoly** | HL 56.5% win rate, +41.3pp over Validator | `monopoly` |
| **FFT Tactics** | TFT 99% win rate — game theory optimal | `fft` |
| **Go** | Greedy/Validator/HL 100% vs Random 35% | `go` |
| **NFSP/MCTS Duality** | BanditMCTS 75% vs MCTS 8% — backward signal transforms forward search | `bandit_mcts` |

📖 **Full benchmarks, architecture, API, and game-specific detail:** [`.docs/23_hl_arena_detail.md`](.docs/23_hl_arena_detail.md).

## 🧠 Heuristic Learning Infrastructure

HL = software systems evolve through **code updates** not weight updates.

```
Episode N:   BanditPruner selects arm → environment runs → reward → TrialLog.append()
Episode N+k: AbsorbCompress promotes stable low-Q arms to hard blocks
Round N+m:   Agent writes new validator.rs → compile .wasm → HotSwapPruner.reload() → RegressionSuite
```

Key subsystems (all default-on or part of `bandit`):
- **Multi-Armed Bandit** — UCB1, ε-greedy, Thompson Sampling strategies
- **TrialLog** — JSONL persistence of episode data
- **AbsorbCompress** — Q-value → hard block promotion
- **HotSwapPruner** — Runtime pruner reload via BLAKE3
- **ReviewMetrics** — Helpfulness/Harmfulness benefit-risk ratio
- **Emotion Vector** — O(d) mid-layer emotion projection, desperation detection
- **Entropy Anomaly** — Session-level OOD monitoring

📖 See [`.docs/09_heuristic-learning.md`](.docs/09_heuristic-learning.md).

## 🎯 G-Zero: Verifier-Free Self-Play

Makes modelless HL smarter with Hint-δ intrinsic reward — no external verifier needed:

```text
δ(q, h, a_hard) = (1/T) Σ [log πG(at | q, h, a<t) − log πG(at | q, a<t)]
```

Two phases: **Phase 1** (modelless — δ → AbsorbCompress + BanditPruner, no gradients) → **Phase 2** (model-based — GRPO + DPO in riir-gpu).

📖 **Full detail:** [`.docs/23_hl_arena_detail.md`](.docs/23_hl_arena_detail.md) §11.

## 🔀 Opt-In & Gated Features

Proven features behind feature flags — not in default set:

| Feature | What | Why Gated |
|---------|------|-----------|
| **D2F / Tri-Mode** | Block-parallel denoising + D2F+AR self-speculation | Experimental decode strategy |
| **G-Zero** (`g_zero`) | Hint-δ self-play + Bomber/FFT arena players | Bench-only, does NOT touch forward() |
| **GameState** (`game_state`) | Generic MCTS, STRATEGA forward model | Depends on bomber, arena-specific |
| **SpecHop** (`spechop`) | Hop-level speculation for multi-step agents | Requires GOAT proof before default-on |
| **SR²AM** (detail) | Adaptive PlanNew/Extend/Skip, context-aware UCB1 | Full API/benchmarks in `.docs/` |
| **FeedbackBandit** | 6-arm UCB1 extends SR²AM with harness/weight updates | Opt-in, requires sr2am_configurator |
| **Committee Boost** | Oracle-gap recovery, debiased BtRank, budget sizing | Opt-in |
| **GFlowNet** | Shortest-path flow into DDTree stack | Opt-in |
| **ROPD Rubric** | Multi-criterion rubric reward vectors | Arena-specific |
| **VPD** | EM-style co-evolutionary teacher-student | Opt-in |
| **HLA/AHLA** | O(1) attention via higher-order linear attention | Alternative attention path |
| **Percepta** (full) | Transformer-VM with WASM interpreter in weights | Research-grade |
| **SP-KV** | Self-pruned KV attention with learned utility | Requires joint training |
| **MaxSim** | Late-interaction scoring, 7.46× SIMD | Amplifies quantization error |
| **DFlare Fusion** (`dflare_fusion`) | Multi-source marginal blending (Plan 174) | Structural ✅, improvement < GOAT threshold |
| **DFlare KV Routing** (`dflare_kv_routing`) | Pruner-confidence KV routing (Plan 174) | Structural ✅, improvement < GOAT threshold |
| **DFlare Progressive Budget** (`dflare_progressive_budget`) | Position-weighted DDTree budget (Plan 174) | Structural ✅, improvement < GOAT threshold |

📖 **Full detail for ALL opt-in features:** [`.docs/21_opt_in_features.md`](.docs/21_opt_in_features.md).

## 🔧 KV Compression Alternatives

Default: **Hybrid OCT+PQ** (OCTOPUS triplet encoding + PlanarQuant 2D Givens rotation). Alternatives:

| Backend | Rotation | FMAs (d=128) | MSE (3-bit) | Calibration |
|---------|----------|-------------|-------------|-------------|
| **Hybrid OCT+PQ** ⭐ | 2D Givens | 256 | 0.026 | 0 samples |
| OCTOPUS | WHT (full) | 16,384 | 0.026 | 0 samples |
| SpectralQuant | Eigenbasis | 16,384 | 0.038 | 256 samples |
| PlanarQuant | 2D Givens | 256 | 0.034 | 0 samples |
| TurboQuant | Random | 16,384 | 0.034 | 0 samples |

📖 **Full comparison tables, benchmarks, code examples:** [`.docs/19_kv_compression.md`](.docs/19_kv_compression.md).

## 🪦 Negative Results

| Feature | Verdict | Why |
|---------|---------|-----|
| Stepwise Reward (Plan 054) | **NO GAIN** | Same tree/path/goal, +33% latency only |
| δ-Mem (Plan 053) | **NO GAIN for DDTree** | 26× latency overhead, corrections too small |
| SDAR Arena | **Negative result** | ELO 954 ≈ Rubric 955 — no improvement |
| RMSD (Plan 125) | **NO GOAT** | 46/46 structural proofs pass but no arena improvement |
| TurboQuant | **Demoted** | SQ/OCT dominate at all quality metrics |

📖 **Full negative result detail + replaced feature audit:** [`.docs/20_negative_results.md`](.docs/20_negative_results.md).

## 🔧 TileRT Execution Pipeline (Plan 102)

Three CPU-applicable insights from TileRT: execution stability metrics, contiguous weight allocation, stage-specialized decode. **GOAT 13/13.**

| Deliverable | Status | Value |
|-------------|--------|-------|
| **D1 Stability Metrics** | ✅ Production-ready | P50/P99/CV observability, +0.6% overhead |
| **D2 Contiguous Weights** | 🔧 Infrastructure | 27→1 allocation, needs ≥8 layers for speed gain |
| **D3 Stage Specialize** | 🔧 Infrastructure | Dispatch free (-0.2%), specialization pending |

## 🧮 Deep Manifold: Fixed-Point Boundary Conditions

Mathematical foundation from [Deep Manifold Part 2](https://arxiv.org/pdf/2512.06563):

| Paper Concept | Our Implementation | Gate |
|---------------|-------------------|------|
| Fixed-point residual ‖f(x)-x‖ | HintDelta + ManifoldResidual trait | `deep_manifold` |
| Symmetric boundaries | BT pairwise ranking + SymmetricBoundaryPair | `bt_rank` |
| Model CAP tradeoff | BanditPruner dynamic routing | `bandit` |
| Manifold federation | BoundaryAlignment KL coupling | `federation` |

GOAT 6/6 proved. Default-on.

📖 See [`.research/051_Deep_Manifold_Fixed_Point_Boundary_Conditions.md`](.research/051_Deep_Manifold_Fixed_Point_Boundary_Conditions.md).

## 🏭 Productions

KatGPT-RS is the **core inference library** — pure algorithms, zero side effects.

```
RAG Engine (anyrag) → Training Pipeline (riir-burner) → Service Layer (riir-ai)
```

| Layer | Repo | What | License |
|-------|------|------|---------|
| Engine | katgpt-rs | DDTree, zero-alloc, pruner traits | MIT |
| Validator | katgpt-rs | SynPruner + PartialParser | MIT |
| RAG Engine | anyrag | Plugin ingestion, episodic memory, Turso/SQLite | MIT |
| Training | riir-burner | LoRA fine-tuning (Gemma 4 E4B) | MIT |
| WASM SDK | riir-ai | Validator trait + export macro | Private |
| GPU Training | riir-ai | wgpu pipeline (26 WGSL kernels), DPO+GRPO | Private |
| Router | riir-ai | Keyword + Embedding routing, ExpertRegistry | Private |

## 🛠️ Getting Started

### Prerequisites

- Rust 1.85+ (edition 2024, 1.93+ recommended)

### Build & Run

```sh
cargo build --release                              # Build with optimizations
cargo run --release                                # Run benchmark + generate plot
cargo run --release --all-features                 # Run everything
cargo test --quiet --workspace --all-features       # Run all tests (111 files, 740+ cases)
cargo run --example sudoku_01_9x9 --features sudoku # Sudoku solver
cargo clippy --all-targets --all-features --quiet   # Lint
```

### Feature Flags

📖 **Complete feature flag table** (90+ flags with descriptions): See main README Feature Flags section → [`.docs/`](.docs/) for per-feature detail.

**Default features** (47, all GOAT-proved): `sparse_mlp`, `domain_latent`, `ppot`, `bandit`, `bt_rank`, `spectral_quant`, `hybrid_oct_pq`, `elf_sde`, `cna_steering`, `deep_manifold`, `federation`, `tes_loop`, `lattice_deduction`, `delta_routing`, `stability_metrics`, `mls_aggregate`, `gdn2_attention`, `dash_attn`, `dreamer`, `lt2_looped`, `dmax_spd`, `eqr_convergence`, `subterranean`, `sr2am_configurator`, `data_gate`, `plasma_path`, `parallel_probe`, `tf_loop`, `leo_all_goals`, `dual_leo`, `sigmoid_margin`, `moa_inference`, `sleep_consolidation`, `spectral_hierarchy`, `dual_gram_pca`, `roofline_cost`, `newton_schulz`, `river_valley`, `peira_distill`, `kog_cpu_fusion`, `gepa_reflective`, `phrase_boost`, `hydra_budget`, `flashar_consensus`, `budget_adaptation`, `ilc_distill`, `thinking_prune`, `rim_slots`.

<details>
<summary>📋 Full Feature Flag Table</summary>

| Flag | Description |
|------|-------------|
| `sudoku` | SudokuPruner constraint pruning + examples |
| `validator` | SynPruner + partial parser (BPE tokenizer, `syn` AST) |
| `sparse_mlp` | TwELL-inspired sparse MLP matmul (Plan 022) |
| `sp_kv` | SP-KV self-pruned key-value attention (Plan 070) |
| `ppot` | PPoT logit-parameterized CPU resampling (Plan 026) |
| `domain_latent` | Mid-layer domain conditioning (Plan 038) |
| `bandit` | Multi-armed bandit + HL infrastructure |
| `bomber` | Bomberman HL arena (bevy_ecs + bandit, Plan 033) |
| `bomber-wasm` | WASM bomber validator loader |
| `bomber-agent` | Coding agent validator loop |
| `game_state` | GameState forward model + generic MCTS (Plan 056) |
| `bandit_mcts` | Bandit-guided MCTS rollout — NFSP/MCTS duality (Plan 067) |
| `budget_adaptation` | Compression-adaptive decode budget (Plan 167, **default-on**) |
| `monopoly` | Monopoly FSM arena (bevy_ecs + bandit) |
| `feedback` | E2E feedback loop — REST endpoint |
| `hla_attention` | Higher-order Linear Attention — O(1) inference cache (Plan 057) |
| `percepta` | CHT hull cache, parabolic encoding, CumSum (Plan 064 TG-A) |
| `percepta_gates` | + ReGLU, stepglu, multiply, persist gates (TG-B) |
| `percepta_graph` | + Expression/Dimension DSL, ProgramGraph (TG-C) |
| `percepta_wasm` | + WASM decoder + lowering + interpreter (TG-E+F) |
| `percepta_compile` | + MILP + weights + transformer + Futamura + evaluator (TG-D+G-J) |
| `maxsim` | MaxSim late-interaction scoring (Plan 080) |
| `delta_mem` | δ-Mem associative bandit memory — no DDTree gain (Plan 053, off) |
| `g_zero` | G-Zero self-play + FFT + Bomber arena players |
| `go` | Go GameState + AutoGo API bridge + tournament (Plan 065) |
| `fft` | FFT Tactics Arena — ATB battle engine |
| `stepcode` | ⚠️ Plan 054 — NO GAIN proven. Off by default |
| `ropd_rubric` | ROPD rubric modelless distillation (Plan 071, off) |
| `sdar_gate` | SDAR sigmoid-gated distillation (Plan 072, off) |
| `vpd_em_distill` | VPD EM-style co-evolutionary distillation (off) |
| `dllm` | D2F Discrete Diffusion Forcing (Plan 066) |
| `tri_mode` | Tri-Mode — AR + Diffusion + Self-Speculation (Plan 089) |
| `flashar_anchor` | FlashAR strided anchor-then-fill (Plan 166, opt-in) |
| `flashar_consensus` | FlashAR consensus tri-mode (**default-on**) |
| `toast_tokenizer` | ToaST split-tree tokenization (Plan 122, opt-in) |
| `convex_tok` | ConvexTok LP vocabulary optimizer (Plan 127, opt-in) |
| `datrie_vocab` | Double-array trie vocab lookup (opt-in) |
| `ilc_distill` | ILC synonym-aware DDTree pruning (**default-on**) |
| `spectral_quant` | SpectralQuant calibrated eigenbasis (**default-on**) |
| `octopus` | OCTOPUS octahedral triplet codec (legacy) |
| `hybrid_oct_pq` | Default KV codec — OCT + PQ (**default-on**) |
| `planar_quant` | 2D Givens rotation KV cache (opt-in) |
| `iso_quant` | 4D quaternion rotation KV cache (opt-in) |
| `asymmetric_kv` | Asymmetric K/V benchmarks (Plan 123, requires turboquant) |
| `shard_kv` | ShardKV asymmetric compression (Plan 147, opt-in) |
| `elf_sde` | ELF SDE noise injection — 10-22× diversity (**default-on**) |
| `cna_steering` | CNA Contrastive Neuron Attribution (**default-on**) |
| `epiplexity_scoring` | Epiplexity structural information scoring (opt-in) |
| `opus_selection` | OPUS Boltzmann + redundancy selection (opt-in) |
| `committee_boost` | Committee Boost — oracle-gap recovery (opt-in) |
| `questbench` | QuestBench underspecification scoring (opt-in) |
| `tes_loop` | SimpleTES RPUCG loop (**default-on**) |
| `deep_manifold` | Deep Manifold fixed-point scoring (**default-on**) |
| `dirichlet_energy` | Dirichlet Energy structural alignment (opt-in) |
| `federation` | Federated KL coupling (**default-on**) |
| `lattice_deduction` | LDT Lattice Deduction (**default-on**) |
| `memo_reflections` | MeMo 5-step Reflection QA pipeline (off) |
| `gepa_reflective` | GEPA-D Pareto bandit config evolution (**default-on**) |
| `spec_cost_model` | Amdahl cost model for LeviathanVerifier (off) |
| `delta_routing` | Delta Block cross-layer routing (**default-on**) |
| `stability_metrics` | Per-step stability instrumentation (**default-on**) |
| `decode_specialize` | Stage-specialized decode paths (off) |
| `hydra_budget` | Hydra-Aware adaptive layer budget (**default-on**) |
| `tiled_attention` | Tiled online-softmax flash attention (opt-in) |
| `parallax_attn` | Parallax parameterized local linear attention (opt-in) |
| `coda_fusion` | CODA fused SIMD kernels (opt-in) |
| `mls_aggregate` | MLS Multi-Layer Sum (**default-on**) |
| `gdn2_attention` | GDN2 recurrent attention (**default-on**) |
| `dash_attn` | DashAttention adaptive sparse attention (**default-on**) |
| `rt_turbo` | RTPurbo retrieval head sparse decode (opt-in) |
| `dreamer` | Auto-Dreamer offline consolidation (**default-on**) |
| `lt2_looped` | LT2 looped inference (**default-on**) |
| `dmax_spd` | DMax soft parallel decode (**default-on**) |
| `plasma_path` | Bit-plane ternary SIMD matvec (**default-on**) |
| `phrase_boost` | PhraseBoost context trie (**default-on**) |
| `tf_loop` | Training-free loop (**default-on**) |
| `eqr_convergence` | EqR convergence selection (**default-on**) |
| `subterranean` | Procedure compilation (**default-on**) |
| `sr2am_configurator` | SR²AM planning regulation (**default-on**) |
| `data_gate` | Self-play stability filtering (**default-on**) |
| `spechop` | SpecHop multi-hop speculation (opt-in) |
| `thinking_prune` | FrozenBaseGuard for intermediate steps (**default-on**) |
| `event_log` | Event-sourced game traces with fork-diff (opt-in) |
| `safe_bandit` | PrudentBanker safe-phased bandit (opt-in) |
| `cache_prune` | CachePrune SAT + rolling hash (opt-in) |
| `leo_all_goals` | LEO all-goals Q-value framework (**default-on**) |
| `dual_leo` | Dual LEO teacher/student mixing (**default-on**) |
| `sigmoid_margin` | Sigmoid margin loss (**default-on**) |
| `moa_inference` | Mixture-of-Activations SwiGLU (**default-on**) |
| `sleep_consolidation` | Offline memory consolidation (**default-on**) |
| `spectral_hierarchy` | Spectral hierarchy diagnostic (**default-on**) |
| `dual_gram_pca` | Dual-Gram PCA routing (**default-on**) |
| `roofline_cost` | Roofline cost model (**default-on**) |
| `newton_schulz` | Newton-Schulz + Muon (**default-on**) |
| `river_valley` | River-valley diagnostics (**default-on**) |
| `peira_distill` | PEIRA inter-view alignment (**default-on**) |
| `kog_cpu_fusion` | Monokernel CPU fusion (**default-on**) |
| `recfm` | Recursive Cross-Scale Consistency (opt-in) |
| `dflare_fusion` | DFlare Marginal Fusion — multi-source conditioning blend (Plan 174, opt-in) |
| `dflare_kv_routing` | DFlare Pruner-Confidence KV Routing (Plan 174, opt-in) |
| `dflare_progressive_budget` | DFlare Position-Weighted DDTree Budget (Plan 174, opt-in) |
| `full` | Enable all features (excludes some opt-in) |

</details>

## 📁 Project Structure

```
crates/katgpt-core/   Shared types & SIMD kernels
src/
  lib.rs              Module index + debug tracking allocator
  main.rs             Entry point (proof → bench → plot)
  transformer.rs      Weights, KVCache (flat/paged/raven), forward/generate
  speculative/        DDTree, DFlash, Verifier, Prefill, D2F, budget, flashar
  pruners/            BanditPruner, TrialLog, HotSwap, BT Rank, CNA, G-Zero, Arena
  tokenizer/          BPE tokenizer
  validator/          SynPruner + PartialParser
  percepta/           Transformer-VM (CHT, hull, WASM interpreter, MILP)
  turboquant/         TurboQuant KV compression (legacy)
  hla/                Higher-order Linear Attention
  gdn2/               Gated DeltaNet-2 recurrent attention
  dash_attn/          DashAttention adaptive sparse attention
  hybrid_oct_pq/      Default KV codec (OCT + PlanarQuant)
  planar_quant/       2D Givens rotation
  spectralquant/      Calibrated eigenbasis compression
  sleep/              Sleep consolidation
  dllm.rs             D2F discrete diffusion
  tf_loop.rs          Training-free loop
examples/            84 examples
tests/               111 test files + 9 benchmark suites
```

📖 **Full file-level detail:** See original README Project Structure in git history.

## 📖 Documentation Index

| Document | Content |
|----------|---------|
| [`.docs/01_overview.md`](.docs/01_overview.md) | Architecture overview |
| [`.docs/02_architecture.md`](.docs/02_architecture.md) | Full architecture detail |
| [`.docs/03_speculative_decoding.md`](.docs/03_speculative_decoding.md) | Speculative decoding, D2F |
| [`.docs/04_performance.md`](.docs/04_performance.md) | Benchmarks, throughput tables |
| [`.docs/05_sudoku.md`](.docs/05_sudoku.md) | Sudoku solver detail |
| [`.docs/06_validator.md`](.docs/06_validator.md) | Validator detail |
| [`.docs/07_adaptation.md`](.docs/07_adaptation.md) | Adaptation strategies |
| [`.docs/08_lucebox_techniques.md`](.docs/08_lucebox_techniques.md) | Raven, PFlash techniques |
| [`.docs/09_heuristic-learning.md`](.docs/09_heuristic-learning.md) | HL infrastructure, FFT benchmarks |
| [`.docs/10_bomber_arena.md`](.docs/10_bomber_arena.md) | Bomberman arena |
| [`.docs/11_monopoly_fsm.md`](.docs/11_monopoly_fsm.md) | Monopoly FSM |
| [`.docs/12_fft_arena.md`](.docs/12_fft_arena.md) | FFT Tactics Arena |
| [`.docs/13_mtp_threshold_guide.md`](.docs/13_mtp_threshold_guide.md) | MTP threshold guide |
| [`.docs/14_go_arena.md`](.docs/14_go_arena.md) | Go arena |
| [`.docs/15_paper_feature_comparison.md`](.docs/15_paper_feature_comparison.md) | Paper feature comparison |
| [`.docs/16_spechop_architecture.md`](.docs/16_spechop_architecture.md) | SpecHop architecture |
| [`.docs/17_peira_distillation.md`](.docs/17_peira_distillation.md) | PEIRA distillation |
| [`.docs/18_sleep_consolidation.md`](.docs/18_sleep_consolidation.md) | Sleep consolidation |
| [`.docs/19_kv_compression.md`](.docs/19_kv_compression.md) | **KV compression alternatives** (TurboQuant, SpectralQuant, OCTOPUS, PlanarQuant, Asymmetric) |
| [`.docs/20_negative_results.md`](.docs/20_negative_results.md) | **Negative results** (StepCode, δ-Mem, SDAR, RMSD, Replaced features) |
| [`.docs/21_opt_in_features.md`](.docs/21_opt_in_features.md) | **Opt-in features** (D2F, GFlowNet, SpecHop, Committee Boost, etc.) |
| [`.docs/22_percepta.md`](.docs/22_percepta.md) | **Percepta full detail** (module structure, compiler stack, verified properties) |
| [`.docs/23_hl_arena_detail.md`](.docs/23_hl_arena_detail.md) | **HL & Arena detail** (all games, G-Zero, Freeze/Thaw, Emotion Vector, etc.) |
| [`examples/README.md`](examples/README.md) | 84 examples grouped by category |

## 📦 Related Crates

- **[riir-ai](../riir-ai/)** — Frame-sampling real-time gamestate bridge ([Plan 070](../riir-ai/.docs/17_frame_sampling_gamestate.md))

## 📜 References

- [microgpt-c](https://github.com/nicholasgasior/microgpt-c) — Original C implementation
- [talos-vs-macbook](https://github.com/AlexCheema/talos-vs-macbook) — Reference model
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/pdf/2211.17192) — Leviathan et al., 2022
- [DFlash](https://arxiv.org/abs/2602.06036) + [DDTree](https://arxiv.org/abs/2604.12989) — Block diffusion draft trees
- [Raven: Sparse Memory Routing](https://github.com/goombalab/raven) — Afzal et al., 2025
- [Percepta](https://www.percepta.ai/blog/can-llms-be-computers) — 2D convex hull attention, WASM in transformer weights
- [TurboQuant](https://arxiv.org/pdf/2504.19874) — Zandieh et al., 2025
- [G-Zero](https://arxiv.org/pdf/2605.09959) — Verifier-free self-play via Hint-δ
- [Deep Manifold Part 2](https://arxiv.org/pdf/2512.06563) — Fixed-point boundary conditions
- [Luce-Org/lucebox-hub](https://github.com/Luce-Org/lucebox-hub/) — Per-chip LLM inference
- [Learning Beyond Gradients](https://trinkle23897.github.io/learning-beyond-gradients/) — Heuristic Learning paradigm
