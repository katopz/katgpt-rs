# Research 58: GRAM — Generative Recursive Reasoning

**Paper:** GRAM: Generative Recursive Reasoning (arXiv:2605.19376)
**Authors:** Baek, Jo, Kim, Ren, Bengio, Ahn (KAIST, Mila, NYU)
**Date:** May 2026
**Distilled:** 2026-07

---

## 1. TL;DR

GRAM turns deterministic Recursive Reasoning Models (RRMs like HRM, TRM, Looped Transformers) into probabilistic generative models by injecting learnable stochastic guidance (ε ~ N(μ_θ(u_t), σ²_θ(u_t))) at each recursive transition. The key distinction from PTRM (Research 49): GRAM's noise has a **learned mean μ_θ** — the model learns WHICH direction to explore, not just how much noise to add. Combined with hierarchical (h, l) states, width-based inference scaling (N parallel trajectories), and ELBO variational training, GRAM achieves strong results: 93.96% on Sudoku-Extreme (vs TRM 90.5%), 99.7% on N-Queens 8×8 (vs TRM 66.8%), and valid unconditional Sudoku generation at 99.05%.

**Verdict: STRONG VALIDATION, MINIMAL ACTION. GRAM independently validates our existing SDE noise + DDTree width scaling + BanditPruner selection stack. Three small distillations: (1) learned-mean SDE guidance option, (2) explicit width-vs-depth benchmark, (3) KL balance coefficient for riir-gpu training. No new feature flag needed — `elf_sde` + `bandit` + `bt_rank` already cover everything.**

---

## 2. What GRAM Actually Does

### 2.1 The Core Loop

```text
┌──────────────────────────────────────────────────────────┐
│  Input x                                                  │
│    ↓                                                      │
│  h₀ = encoder(x)                                         │
│    ↓                                                      │
│  for high-level step t = 1..T:                            │
│    u_t = deterministic_transition(h_{t-1})                │
│    μ_t = μ_θ(u_t)          ← learned guidance direction  │
│    σ_t = σ_θ(u_t)          ← learned noise scale         │
│    ε_t ~ N(μ_t, σ_t²)      ← STOCHASTIC latent          │
│    h_t = u_t + ε_t                                        │
│    ↓                                                      │
│    for low-level refinement k = 1..K:                     │
│      l_k = low_level(l_{k-1}, h_t)                       │
│    ↓                                                      │
│  output = decoder(l_K)                                    │
│                                                           │
│  Width scaling: N parallel trajectories, select best       │
│    → majority vote OR Latent Process Reward Model (LPRM)  │
└──────────────────────────────────────────────────────────┘
```

1. **Deterministic backbone**: A standard RRM (HRM/TRM/Looped Transformer) produces a deterministic transition u_t at each recursive step.
2. **Stochastic guidance**: Instead of pure zero-mean noise, GRAM learns a direction μ_θ(u_t) and scale σ_θ(u_t). The latent becomes h_t = u_t + ε_t where ε_t ~ N(μ_θ(u_t), σ²_θ(u_t)).
3. **Hierarchical decomposition**: (h, l) states — high-level h_t updated K times per step, with K low-level refinements l_k per h update.
4. **Width-based inference**: Sample N parallel trajectories, select via majority voting or LPRM.
5. **ELBO training**: Variational lower bound with truncated gradients (surrogate objective per supervision step).
6. **ACT halting**: Q-learning-based halt head for adaptive computation time.
7. **Unconditional generation**: Replace input with empty embedding to generate from scratch.

### 2.2 Why Learned Guidance Works (Intuition)

PTRM (Research 49) showed that zero-mean Gaussian noise breaks deterministic collapse. GRAM goes further: the model learns **which direction** to perturb. This is the difference between:
- **PTRM**: "Explore randomly, let Q-head select" — zero-mean noise + selection
- **GRAM**: "Explore intelligently, then select" — learned-mean noise + selection

The learned mean μ_θ is conditioned on the current deterministic state u_t, so it's a form of **state-dependent exploration** — similar to a policy gradient method where the policy is the noise distribution.

However, GRAM's own ablation reveals nuance:
- Removing guidance (zero-mean) on **Sudoku**: 94.88% vs 93.96% with guidance (slightly better without!)
- Removing guidance on **N-Queens**: 50.27% vs 99.7% with guidance (catastrophic drop)

This means guidance helps when the exploration space has a clear directional structure (N-Queens constraints), but hurts when the space is more uniform (Sudoku digits). For our game domains (Go, Bomber, FFT), the truth likely lies in between.

### 2.3 Key Innovation vs PTRM

| Aspect | PTRM (Research 49) | GRAM (This) |
|---|---|---|
| Noise distribution | N(0, σ²) — zero-mean | N(μ_θ(u_t), σ²_θ(u_t)) — learned mean |
| Guidance | None (pure exploration) | State-dependent direction |
| Training | Q-head (discriminative) | ELBO variational (generative) |
| Hierarchy | None | (h, l) with K low-level per high-level |
| Selection | Q-head AUROC ~0.94 | LPRM or majority vote |
| Unconditional gen | No | Yes (empty input) |
| Noise location | Logits/embeddings | Latent state transitions |
| Posterior collapse | N/A | KL balance 0.8 prevents it |

---

## 3. Key Results

### 3.1 Benchmark Performance

| Benchmark | GRAM | TRM (baseline) | HRM | AR (autoregressive) |
|---|---|---|---|---|
| Sudoku-Extreme | **93.96%** | 90.5% | 82.5% | — |
| N-Queens 8×8 accuracy | **99.7%** | 66.8% | — | — |
| N-Queens 8×8 coverage | **90.3%** | 36.1% | — | — |
| Graph Coloring 8v (conflict edges) | **2.7** | — | — | 19.0 |
| MNIST unconditional IS | **2.04** | — | — | — |
| MNIST unconditional FID | **73.34** | — | — | — |
| Valid Sudoku boards (unconditional) | **99.05%** | — | — | — |

### 3.2 Width Scaling: The Money Plot

GRAM's strongest result: N=20 samples at 16 iterations beats ALL deterministic baselines at 320 iterations.

| Configuration | Sudoku-Extreme | Compute |
|---|---|---|
| Deterministic, 16 iters | ~88% | 1× |
| Deterministic, 320 iters | ~92% | 20× |
| **GRAM, N=20, 16 iters** | **93.96%** | 20× (parallel) |

This directly validates PTRM's finding (Research 49): width >> depth. 20 parallel noisy trajectories at shallow depth > 1 deterministic trajectory at 20× depth.

### 3.3 Critical Ablations

| Ablation | N-Queens 8×8 | Interpretation |
|---|---|---|
| Full GRAM (stochastic + guidance) | **99.7%** | Full system |
| Remove guidance (zero-mean only) | 50.27% | Guidance essential for N-Queens |
| Remove stochasticity (deterministic) | 0.0% | **Noise is non-negotiable** |
| Naive stochasticity (random init) | ~66.8% (same as TRM) | Random init ≠ variational noise |
| Naive stochasticity (stochastic decode) | ~66.8% (same as TRM) | Temperature noise ≠ learned noise |
| Low-level noise only | No improvement | Only high-level noise helps |
| KL balance 0.0 (no KL) | Posterior collapse | KL needed to prevent collapse |
| KL balance 0.8 (optimal) | 99.7% | Sweet spot |
| KL balance 1.0 (standard) | Suboptimal | Over-regularization |

**Key insight**: The gains come from the **variational framework**, not mere randomness. Naive stochasticity (random init, stochastic decode) gives zero improvement. The ELBO training objective + learnable noise distribution together create the benefit.

---

## 4. Mapping to Our Architecture

### 4.1 Modelless Path (microgpt-rs, Primary)

| GRAM Concept | Our Equivalent | Location | Status |
|---|---|---|---|
| Stochastic latent transition h_t = u_t + ε_t | `inject_sde_noise()` on marginals | `src/speculative/dd_tree.rs:69-110` | ✅ Implemented |
| Zero-mean noise N(0, σ²) | `SdeConfig.gamma` × N(0,1) | `src/speculative/types.rs:492-499` | ✅ ELF default γ=1.0 |
| **Learned mean** N(μ_θ(u_t), σ²) | **Not yet** (zero-mean only) | — | 🟡 Small gain (see §7.1) |
| Noise scale σ_θ(u_t) | `SdeConfig.gamma` (fixed) | `src/speculative/types.rs:492-499` | ✅ Configurable |
| Preserve top-1 token | `SdeConfig.preserve_top1` | `src/speculative/types.rs:495` | ✅ Implemented |
| Confidence floor | `SdeConfig.confidence_floor` | `src/speculative/types.rs:496` | ✅ Implemented |
| N parallel trajectories | `DDTreeBranchCache` with K branches | `src/speculative/types.rs:301-305` | ✅ Implemented |
| Branch forking | `DDTreeBranchCache::fork_branch()` | `src/speculative/types.rs:320+` | ✅ Copy-on-write KV |
| Width scaling (N samples) | `max_branches` config | `src/speculative/types.rs:307-317` | ✅ Configurable |
| Trajectory selection (LPRM) | `BanditPruner<P>` Q-values | `src/pruners/bandit.rs:289+` | ✅ Online Q-learning |
| Majority voting | `extract_best_path()` | `src/speculative/dd_tree.rs` | ✅ Best path extraction |
| Richer pairwise selection | `BtRank` (Bradley-Terry) | `src/pruners/bt_rank.rs` | ✅ Feature `bt_rank` |
| Flow-based exploration | `FlowPruner<P>` (GFlowNet) | `src/speculative/flow_pruner.rs:43-52` | ✅ GRAM doesn't have this |
| Sigmoid-gated selection | `SdarBanditPruner<P>` | `src/pruners/sdar/sdar_bandit.rs:187-196` | ✅ SDAR gate |
| ACT early exit | `domain_latent` feature | Feature flag | ✅ Implemented |
| Deep supervision (N_sup steps) | Plan 049 deep supervision | `.plans/` | ✅ Planned |
| Feature flag | `elf_sde`, `bandit`, `bt_rank` | `Cargo.toml` features | ✅ All default-on |

### 4.2 Model-Based Path (riir-ai, Opt-In)

| GRAM Concept | Our Equivalent | Location | Status |
|---|---|---|---|
| ELBO variational training | DPO loss | riir-gpu (Plan 059) | 🟡 Planned |
| Posterior q_φ(τ\|x,y) | GRPO proposer | riir-gpu (Plan 059) | 🟡 Planned |
| KL balance coefficient (0.8) | DPO β parameter | riir-gpu config | 🟡 See §7.3 |
| Model weight updates | LoRA training | riir-gpu | 🟡 Planned |
| Learned mean μ_θ | Logit-shift in LoRA | riir-gpu | 🟡 Small gain |

### 4.3 Structural Equivalence

GRAM's loop is:

```text
for each trajectory n in N:
    for each high-level step t in T:
        u_t = RRM_transition(h_{t-1})
        μ_t, σ_t = guidance_network(u_t)
        ε_t ~ N(μ_t, σ_t²)
        h_t = u_t + ε_t
        for each low-level k in K:
            l_k = refine(l_{k-1}, h_t)
    output_n = decode(l_K)

select best from {output_1, ..., output_N} via LPRM
```

Our loop is:

```text
DDTreeBranchCache::new(config, max_branches=K)     // K parallel branches
marginals = [model.forward(token, pos) for pos in T] // T depth steps
noisy = inject_sde_noise(&marginals, &sde_config, rng) // ε injection (zero-mean)
tree = build_dd_tree_screened(&noisy, config, screener)  // branch + prune
best = extract_best_path(&tree)                          // trajectory selection
```

The mapping is near-1:1 with one difference: our noise is zero-mean (N(0, γ²)) while GRAM uses learned-mean (N(μ_θ, σ²)). This is addressed in §7.1.

---

## 5. What We Already Have (GRAM Validates Our Design)

### 5.1 SDE Noise Injection = GRAM's Stochastic Guidance ε_t ✅

GRAM's core innovation is injecting stochastic transitions at each recursive step. Our `inject_sde_noise` does exactly this:

```src/speculative/dd_tree.rs#L69-79
pub fn inject_sde_noise(
    marginals: &[&[f32]],
    sde_config: &SdeConfig,
    rng: &mut Rng,
) -> Vec<Vec<f32>> {
    if !sde_config.is_enabled() {
        return marginals.iter().map(|m| m.to_vec()).collect();
    }
    // ...
}
```

GRAM validates: noise injection is non-negotiable. Their ablation shows 0% N-Queens without stochasticity. Our `elf_sde` feature being default-on is confirmed correct.

### 5.2 DDTreeBranchCache K Branches = GRAM's Width Scaling N ✅

GRAM proves N=20 at 16 iters beats deterministic at 320 iters. Our `DDTreeBranchCache` with `max_branches=K` is the same mechanism:

```src/speculative/types.rs#L301-317
pub struct DDTreeBranchCache {
    paged: PagedKVCache,
    branch_count: usize,
    max_branches: usize,  // This is GRAM's N
}
```

More branches >> deeper lookahead. Research 49 (PTRM) showed +28.6pp from K=64 vs +3.1pp from T=64. GRAM independently confirms this from a completely different research group.

### 5.3 BanditPruner Q-values = GRAM's LPRM Value Prediction ✅

GRAM uses LPRM (Latent Process Reward Model) to score trajectories. Our `BanditPruner<P>` does the same with online Q-learning:

```src/pruners/bandit.rs#L289-293
pub struct BanditPruner<P: ScreeningPruner> {
    inner: P,
    strategy: BanditStrategy,
    stats: BanditStats,
    // ...
}
```

Key advantage: BanditPruner learns online (no separate training phase), adapts to distribution shift, and supports multiple strategies (UCB1, ε-greedy). This is richer than GRAM's static LPRM.

### 5.4 BtRank Pairwise = Richer Than GRAM's Best-of-N ✅

GRAM selects trajectories via LPRM scoring or majority vote — both pointwise methods. Our `BtRank` (Bradley-Terry) uses pairwise comparisons that internalize relative strength:

```src/pruners/bt_rank.rs
// BtRank: pairwise ranking with Bradley-Terry model
// Each comparison updates both candidates' ratings
// More robust than pointwise scoring
```

Where GRAM says "pick the highest LPRM score," we say "compare all pairs, rank by transitive strength." This is strictly more informative.

### 5.5 FlowPruner GFlowNet = GRAM Doesn't Have This ✅

GRAM explores via parallel noisy trajectories. Our `FlowPruner<P>` adds GFlowNet flow-based exploration:

```src/speculative/flow_pruner.rs#L43-52
pub struct FlowPruner<P: ScreeningPruner> {
    inner: P,
    lambda: f32,  // Flow regularization
    // ...
}
```

This provides a principled way to explore diverse high-reward trajectories — something GRAM achieves only through brute-force N parallel samples.

### 5.6 ACT-Style Early Exit = GRAM's Q-Learning Halt Head ✅

GRAM uses a Q-learning halt head for adaptive computation. Our `domain_latent` feature provides ACT-style early exit:

```toml
# Cargo.toml
default = ["sparse_mlp", "domain_latent", "ppot", "bandit", "bt_rank", "spectral_quant", "elf_sde"]
```

Both mechanisms allow the model to stop early when confident, saving compute.

### 5.7 Deep Supervision = GRAM's N_sup Supervision Steps ✅

GRAM uses N_sup supervision steps in its ELBO training. Plan 049 (deep supervision) brings the same concept: supervise intermediate recursive states, not just the final output. This is already in our roadmap.

---

## 6. What We Don't Need

### 6.1 GRAM's Hierarchical (h, l) Decomposition

GRAM uses a specific (h, l) state hierarchy: K low-level refinements per high-level update. Our HLA (Hierarchical Latent Attention) handles hierarchical attention differently — we don't need GRAM's specific decomposition. Our speculative decoding already captures multi-scale reasoning through `draft_lookahead` depth.

### 6.2 GRAM's ELBO Training Objective

GRAM's ELBO variational training is for training foundation models from scratch. Our modelless path doesn't need variational inference — `inject_sde_noise` is a pure inference-time transformation. For our model-based path (riir-gpu), DPO loss serves the same role (learning to generate good trajectories) but is better suited for our LoRA fine-tuning setup.

### 6.3 GRAM's LPRM

GRAM trains a separate Latent Process Reward Model for trajectory scoring. Our `BanditPruner` already does this better — online learning, no separate training, multiple strategies. We don't need a dedicated reward model.

### 6.4 GRAM's Architecture (HRM/TRM Backbone)

GRAM builds on HRM/TRM/Looped Transformer architectures. We don't train foundation models. Our draft/target speculative decoding is the correct architecture for our use case.

### 6.5 Reparameterization Trick

GRAM uses the reparameterization trick for differentiable sampling through the stochastic layer. Our modelless path doesn't need gradients through noise — we're not training, just sampling. The reparameterization trick is only relevant for riir-gpu training, where standard backpropagation through LoRA already handles it.

### 6.6 Unconditional Generation for Games

GRAM generates valid Sudoku boards from empty input — impressive but not new for us. Our D2F (`dllm` feature) already does block-wise unconditional generation for text. For games, self-play from empty board is already implemented in g_zero (Research 21). We don't need GRAM's specific empty-embedding approach.

---

## 7. What IS Worth Exploring

### 7.1 Learned-Mean SDE Guidance Option (Small, Actionable)

This is the one genuinely new idea from GRAM that we don't have. Our `inject_sde_noise` uses zero-mean N(0, γ²). GRAM uses learned-mean N(μ_θ(u_t), σ²_θ(u_t)). The learned mean shifts exploration toward promising directions.

However, GRAM's own ablation shows this is **domain-dependent**:
- **Sudoku**: Zero-mean (94.88%) > Learned-mean (93.96%) — guidance hurts
- **N-Queens**: Learned-mean (99.7%) >> Zero-mean (50.27%) — guidance essential

For our architecture, we could add a `guided` flag to `SdeConfig`:

```/dev/null/sde_guided.rs#L1-50
/// Guided SDE noise injection using logit-shift as learned mean.
///
/// GRAM (arXiv:2605.19376) shows that learned-mean noise N(μ_θ, σ²)
/// helps on structured constraint tasks (N-Queens) but hurts on
/// uniform tasks (Sudoku).
///
/// In our modelless path, we use the current marginal's deviation
/// from uniform as a simple proxy for μ_θ. This is "free guidance" —
/// no training needed, just shift noise toward high-probability tokens.
///
/// Feature gate: `#[cfg(feature = "elf_sde")]`
///
/// # When to use
/// - `guided: true` → structured constraint domains (puzzles, planning)
/// - `guided: false` → uniform domains (text, open-ended generation)
pub struct SdeConfig {
    pub gamma: f32,
    pub preserve_top1: bool,
    pub confidence_floor: f32,
    /// NEW: Use logit-shift as learned mean for guided exploration.
    /// GRAM's μ_θ(u_t) approximated by marginal deviation from uniform.
    /// Default: false (zero-mean, PTRM-style).
    pub guided: bool,
}

impl SdeConfig {
    /// Inject SDE noise with optional learned-mean guidance.
    ///
    /// When `guided: true`:
    ///   noisy[i] = marginal[i] + gamma * N(marginal[i] - uniform, 1.0)
    ///   (shift toward current high-probability tokens)
    ///
    /// When `guided: false` (default):
    ///   noisy[i] = marginal[i] + gamma * N(0, 1.0)
    ///   (standard zero-mean perturbation)
    pub fn inject_with_guidance(
        &self,
        marginals: &[&[f32]],
        rng: &mut Rng,
    ) -> Vec<Vec<f32>> {
        marginals
            .iter()
            .map(|marginal| {
                let uniform = 1.0 / marginal.len() as f32;
                marginal
                    .iter()
                    .map(|&p| {
                        let mean = if self.guided { p - uniform } else { 0.0 };
                        let noise = rng.normal() * self.gamma;
                        p + mean + noise
                    })
                    .collect()
            })
            .collect()
    }
}
```

**Verdict: SMALL GAIN.** For our game domains (Go, Bomber, FFT), the exploration space is structured enough that zero-mean noise + bandit selection already works well. The `guided` flag is worth implementing as an option but should NOT be default. GRAM's own data shows it can hurt.

**Priority: LOW.** Add as optional `SdeConfig.guided` field, default `false`. No new feature flag.

### 7.2 Explicit Width-vs-Depth Benchmark (Validation)

GRAM proves N=20 at 16 iters beats all deterministic at 320 iters. PTRM (Research 49) proves K=64 gives +28.6pp vs T=64 giving +3.1pp. Two independent papers, same conclusion: width >> depth.

We should benchmark this explicitly on our arenas to get concrete numbers for our specific domains:

```/dev/null/bench_gram_width.rs#L1-35
/// Benchmark: GRAM-style width vs depth scaling on our domains.
///
/// Measures how much gain comes from:
/// - Width: K=1, 2, 4, 8, 16, 32, 64 branches (noise seeds)
/// - Depth: T=1, 2, 4, 8 draft_lookahead steps
///
/// Feature gate: `#[cfg(all(feature = "elf_sde", feature = "bandit"))]`
///
/// Expected result (from GRAM + PTRM):
/// - Width K=1→20: +15-25pp on constrained tasks
/// - Depth T=1→16: +3-8pp
/// - Width-dominant across all tasks
///
/// Domains to test:
/// 1. Go (19×19): win rate vs fixed opponent
/// 2. Bomber: score on procedurally generated levels
/// 3. FFT puzzles: solve rate
///
/// Run: cargo bench --features "elf_sde bandit" --bench gram_width_scaling
```

**Priority: MEDIUM.** Validates our SDE infrastructure against two independent papers. Provides concrete numbers for our specific game domains.

### 7.3 KL Balance Coefficient for riir-gpu DPO Training (Model-Based Path)

GRAM uses KL balance coefficient 0.8 to prevent posterior collapse in ELBO training. This is directly relevant to our DPO training in riir-gpu (Plan 059):

- GRAM's KL balance 0.0 → posterior collapse (N-Queens: failure)
- GRAM's KL balance 0.8 → optimal (N-Queens: 99.7%)
- GRAM's KL balance 1.0 → over-regularization (suboptimal)

Our DPO loss has a β parameter that controls KL penalty. GRAM suggests **β < 1.0** (under-regularize slightly) prevents mode collapse. Current best practice is β=0.1 for DPO; GRAM's 0.8 is for ELBO, not directly comparable, but the principle holds: too much KL penalty kills diversity.

**Priority: LOW (model-based path only).** Note in Plan 059 that β should be tuned with posterior collapse in mind. No action needed until riir-gpu training is implemented.

### 7.4 NOT Worth Exploring

The following might seem tempting based on GRAM but are explicitly not worth pursuing:

1. **Full ELBO variational training**: Overkill for our modelless path. `inject_sde_noise` achieves stochasticity without variational inference.
2. **Hierarchical (h, l) decomposition**: Our HLA handles hierarchy differently. No need for GRAM's specific split.
3. **Separate LPRM training**: `BanditPruner` online learning is superior.
4. **Reparameterization trick**: Not needed in modelless path.
5. **Q-learning halt head**: `domain_latent` ACT-style exit already covers this.
6. **Majority voting**: `BtRank` pairwise is strictly better than majority vote.
7. **Naive stochasticity (random init / stochastic decode)**: GRAM's own ablation shows these give zero improvement. The gain comes from the variational framework.

---

## 8. Verdict and Priority

### 8.1 Verdict: STRONG VALIDATION, MINIMAL ACTION

GRAM independently validates our existing design from a completely different angle than PTRM:

| Our Design | GRAM Finding | Validation |
|---|---|---|
| `inject_sde_noise` with γ=1.0 | Stochastic transitions essential (0% without) | ✅ Core mechanism confirmed |
| `DDTreeBranchCache` with K branches | N=20 at shallow depth beats deep deterministic | ✅ Width >> depth confirmed |
| `BanditPruner` Q-values | LPRM predicts trajectory quality | ✅ Selection mechanism confirmed |
| `BtRank` pairwise comparison | — | ✅ We go beyond GRAM |
| `FlowPruner` GFlowNet exploration | — | ✅ We go beyond GRAM |
| `SdeConfig.preserve_top1` | Keep high-confidence tokens stable | ✅ Engineering practice confirmed |
| `elf_sde` default-on | Stochasticity should be default | ✅ Deployment confirmed |
| Online bandit learning | Offline LPRM training | ✅ Our approach is better |
| Zero-mean noise (N(0, γ²)) | Learned-mean helps N-Queens, hurts Sudoku | ✅ Domain-dependent (keep default) |

### 8.2 Action Items

| Item | Effort | Impact | Priority | Target |
|---|---|---|---|---|
| 7.1 `SdeConfig.guided` flag | Small | Low | LOW | `src/speculative/types.rs` |
| 7.2 Width-vs-depth benchmark | Medium | High | MEDIUM | `tests/bench_gram_width.rs` |
| 7.3 KL balance note for DPO | Trivial | Low | LOW | Plan 059 update |

### 8.3 What NOT To Do

- Do NOT add ELBO variational training to modelless path
- Do NOT implement GRAM's (h, l) hierarchical decomposition
- Do NOT train a separate LPRM — `BanditPruner` is better
- Do NOT make guided noise the default — it hurts on some domains
- Do NOT add new feature flags — `elf_sde` + `bandit` + `bt_rank` cover everything
- Do NOT replace zero-mean noise with learned-mean noise — keep zero-mean as default
- Do NOT implement reparameterization trick for modelless path

### 8.4 Cross-Reference Summary

| Research | Connection to GRAM |
|---|---|
| Research 48 (HRM-Text) | GRAM builds on HRM architecture, adds stochasticity. HRM-Text is the deterministic backbone; GRAM makes it generative. |
| Research 49 (PTRM) | PTRM proves width>depth with Gaussian noise; GRAM validates same principle with learned guidance. Two independent confirmations of our design. |
| Research 35 (Attractor Models) | GRAM's stochastic transitions prevent fixed-point collapse — the exact problem attractor models face. Noise breaks attractor basins. |
| Research 21 (G-Zero) | Self-play loop structure similar to GRAM's multi-trajectory exploration. g_zero already does unconditional generation via self-play. |
| Research 44 (ELF/SDE) | Our SDE noise injection IS GRAM's stochastic guidance. ELF was our original distillation; GRAM validates from a different angle. |
| Research 37 (REAP) | Model-based/modelless duality: GRAM's ELBO training is model-based, our `inject_sde_noise` is modelless. Same effect, different paths. |
| Research 40 (Bradley-Terry) | BtRank pairwise comparison is richer than GRAM's best-of-N selection. We go beyond. |
| Research 34 (D2F) | Block-wise diffusion similar to GRAM's supervision steps. Both exploit multi-scale refinement. |
| Research 38 (SDAR) | Sigmoid-gated bandit selection is a richer trajectory selector than GRAM's majority vote. |

---

## 9. References

1. **GRAM** — arXiv:2605.19376 — Generative Recursive Reasoning (Baek, Jo, Kim, Ren, Bengio, Ahn)
2. **PTRM** — arXiv:2605.19943 — Probabilistic Tiny Recursive Model (Research 49)
3. **HRM-Text** — Sapient Inc, 2025 — Hierarchical Recurrent Pretraining (Research 48)
4. **ELF** — arXiv:2605.10938 — Embedded Language Flows (Research 44, Plan 079)
5. **Attractor Models** — arXiv:2605.12466 — Solve the Loop (Research 35)
6. **REAP** — arXiv:2510.13999 — REAP the Experts (Research 37)
7. **D2F** — Discrete Diffusion Forcing (Research 34, Plan 066)
8. **Bradley-Terry** — OpenDeepThink arXiv:2605.15177 (Research 40, Plan 079 bt_rank)
9. **SDAR** — Self-Distilled Agentic RL (Research 38, Plan 072 sdar_gate)
10. **GFlowNet** — Shortest Paths (Research 23, FlowPruner)
11. **G-Zero** — Self-Play Open-Ended Generation (Research 21)

### Key File References

| File | Role |
|---|---|
| `src/speculative/dd_tree.rs` | `inject_sde_noise`, `build_dd_tree_sde`, `extract_best_path` |
| `src/speculative/types.rs` | `SdeConfig`, `DDTreeBranchCache`, `ScreeningPruner`, `ConstraintPruner` |
| `src/speculative/verifier.rs` | `SpeculativeVerifier` trait |
| `src/pruners/bandit.rs` | `BanditPruner<P>` with Q-values and strategies |
| `src/pruners/bt_rank.rs` | `BtRank` Bradley-Terry pairwise ranking |
| `src/speculative/flow_pruner.rs` | `FlowPruner<P>` GFlowNet flow bonus |
| `src/pruners/sdar/sdar_bandit.rs` | `SdarBanditPruner<P>` sigmoid-gated bandit |
| `src/pruners/sdar/sdar_absorb.rs` | `SdarGatedAbsorbCompress<P>` sigmoid-gated absorb-compress |
| `tests/bench_elf_modelless.rs` | SDE noise benchmarks (diversity + overhead) |
| `examples/bandit_02_ddtree.rs` | BanditPruner + DDTree integration example |
| `examples/bandit_03_slot.rs` | BanditPruner proof-of-value example |