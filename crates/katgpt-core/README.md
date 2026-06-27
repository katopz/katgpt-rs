# katgpt-core

[![crates.io](https://img.shields.io/crates/v/katgpt-core.svg)](https://crates.io/crates/katgpt-core)
[![Documentation](https://docs.rs/katgpt-core/badge.svg)](https://docs.rs/katgpt-core)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Modelless inference primitives: shared types, SIMD kernels, attention
variants, spectral/manifold operators, belief kernels, and latent-space ops.
The core library of the [`katgpt-rs`](https://github.com/katopz/katgpt-rs)
inference framework.

> **Modelless-first.** Every primitive here is inference-only: no training,
> no backprop, no gradient descent. Runtime weight mutations are limited to
> freeze/thaw, deterministic raw/lora hot-swap, and latent-space projections.
> Each primitive ships behind a feature flag and is only promoted to default-on
> after passing a [GOAT gate](#goat-gate) (correctness + perf + no-regression).

## Usage

```toml
[dependencies]
katgpt-core = "0.2"
```

Enable specific primitives via features (the `default` feature enables the
GOAT-validated set):

```toml
[dependencies]
katgpt-core = { version = "0.2", features = ["viable_manifold_graph", "ac_prefix"] }
```

## What's inside

### Always-on core (no feature flag)

| Module | What it provides |
|---|---|
| `types` | `Config`, `Rng`, math utilities, `LoraAdapter`, `DomainLatent`, `ShardEmbedding`, `DataGate` |
| `traits` | 18+ shared traits for game AI and speculative decoding (`ConstraintPruner`, `ScreeningPruner`, `SpeculativeGenerator`, `GameState`, `RolloutPolicy`, ...) |
| `simd` | NEON / AVX2 accelerated linear-algebra kernels (incl. `simd_sigmoid`) |
| `shard_embedding` | JL random orthogonal projection `[f32;64] → [f32;8]` |
| `leaky_core` | Leaky integrator baseline kernel |

### Attention variants (feature-gated)

- **`tiled_attention`** — Tiled online-softmax flash attention for CPU SIMD
- **`parallax_attn`** — Parameterized local linear attention (R projection + covariance branch)
- **`coda_fusion`** — CODA fused SIMD kernels (matmul + residual + rmsnorm + activation)
- **`funcattn_structured_basis`** — Functional Attention: Tikhonov k×k spectral transport operator (default-on)

### Spectral / manifold operators

- **`dec_operators`** — Discrete Exterior Calculus: exterior derivative, codifferential, Hodge Laplacian, Hodge decomposition (Helmholtz)
- **`spectral_hierarchy`** — Eigenspace alignment, Haar wavelets, Cauchy interlacing (default-on)
- **`viable_manifold_graph`** — Safe-manifold navigation: pullback volume + CSR graph + A*/random walk (default-on)
- **`geometric_product`** — Geometric algebra product with Padé [4/4] SiLU (default-on)
- **`fourier_continuation`** / **`spectral_differentiation`** — Spectral FNO primitives (default-on)
- **`tucker_factorization`** — Modelless HOSVD (default-on)

### Belief, sampling & decision primitives

- **`micro_belief`** — `MicroRecurrentBeliefState`: attractor + leaky families, BLAKE3-committed snapshots
- **`bom_sampling`** — K-hypothesis single-pass belief sampling (Bag of Marginals) (default-on)
- **`ict`** — Distributional branching-point detector
- **`cgsp`** / **`cgsp_dual_pool`** — Curiosity-Guided Self-Play triad (Solver / Conjecturer / Guide)
- **`closure_instrument`** — PTG + motif mining + PRI/CDG/TaR (default-on)
- **`arg_protocol`** — ARG Standard protocol primitives (PolicyEnvelope, TaxonomyValidator, LifecycleState) (default-on)
- **`indicator_probe_bank`** / **`indicator_similarity`** — Tamper-evident indicator probes (default-on)

### Latent-space ops

- **`latent_field_steering`** — Top-down direction-vector injection (default-on)
- **`cross_resolution_transport`** — Train-small-deploy-large asymmetric basis (default-on)
- **`depth_invariance`** — Depth-invariance diagnostic + magnitude-regularized residual (default-on)
- **`subspace_phase_gate`** — Participation ratio + numerical rank + Jacobian SVD gate
- **`phase_rotation_coupling`** — Pythagorean-safe phase rotation (default-on)

### Memory & commitment

- **`merkle_octree`** — Hierarchical BLAKE3 commitment + curator verification
- **`content_store`** — Content-addressed chunked Merkle store
- **`rtdc`** — Resolution-Tiered Deterministic Commitment (multi-depth Merkle roots)
- **`committed_field_blend`** — Sampling-invariant per-entity MoE (FAME, arXiv:2510.00621)

### Game-AI & search

- **`leo_all_goals`** / **`dual_leo`** — LEO all-goals Q-value framework + teacher-student mixer (default-on)
- **`questbench`** — Underspecification scoring
- **`roofline_cost`** — GPU operator runtime prediction (default-on)
- **`karc`** — Kolmogorov-Arnold reservoir computing delay-basis ridge forecaster
- **`qgf`** — Q-Guided Flow: test-time Q-gradient guidance

See the [full feature showcase](https://github.com/katopz/katgpt-rs#-feature-showcase)
in the main repository for the complete list (50+ additional modules) and the
research papers each primitive is distilled from.

## GOAT gate

Every primitive must pass the **G**reatest-**O**f-**A**ll-**T**ime gate before
promotion to default-on:

| Gate | Requirement |
|---|---|
| **G1** Correctness | Matches or beats the reference implementation to within tolerance |
| **G2** Performance | Hits the latency/throughput target for its hot path |
| **G3** No-regression | Default + all-features + no-default builds all stay clean |
| **G4** Allocation-free | Zero heap allocations in hot loops (or documented budget) |
| **G5** Feature isolation | Enabling the feature doesn't break unrelated features |

A perf gain on a biased/incorrect answer is **not** a modelless gain — the
quality gate (G1) must pass modellessly for the GOAT to hold.

## Crates.io / docs.rs

- **crates.io:** https://crates.io/crates/katgpt-core
- **docs.rs:** https://docs.rs/katgpt-core
- **Repository:** https://github.com/katopz/katgpt-rs

## License

MIT ([LICENSE](https://github.com/katopz/katgpt-rs/blob/develop/LICENSE)).
