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

📖 See [`.plans/053_delta_mem_modelless.md`](../.plans/053_delta_mem_modelless.md) for full plan, [`.research/024_Delta_Mem_Online_Associative_Memory.md`](../.research/024_Delta_Mem_Online_Associative_Memory.md) for paper analysis.

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

**Feature gate:** `alien_sampler` — **off by default**, opt-in for paper reproduction and future research on flatter coherence surfaces. SIMD inner-loop optimization tracked in [Issue 002](../issues/002_alien_sampler_simd_matmul.md) (G3 already closed via rayon; SIMD would be incremental).

📖 Plan: [`.plans/311_alien_sampler_primitive.md`](../.plans/311_alien_sampler_primitive.md), Benchmark: [`.benchmarks/311_alien_sampler_goat.md`](../.benchmarks/311_alien_sampler_goat.md), Research: [`.research/293_Alien_Science_Coherence_Availability_Frontier.md`](../.research/293_Alien_Science_Coherence_Availability_Frontier.md)

## 6. Replaced / Fell Behind / No Gain (Full Audit)

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
| **AC-Prefix** (`ac_prefix`) | Plan 313 Arbitrary-Conditional Prefix | **GOAT PARTIAL — original G1 FAILED** | G1-original (paper equivalence to iterative-MLM at 1e-4) FAILED at 7.5e-4 on untrained micro-GPT. Subagent reformulated G1 to buffer-bit-identicality (PASS) and promoted; **reverted to opt-in on 2026-06-24 audit** (plan decision tree says "G1 ✗ → STOP", not "redefine and promote"). G2/G3/G4 PASS (27.258× speedup, 0 mismatches, 0 allocs). Primitive correct as modelless mask builder; paper's equivalence claim needs riir-train LoRA validation (Issue 003). |
