# Plan 184: Direction-Adaptive Credit — Entropy-Bifurcated Pruning (Modelless)

> **Research:** 164 (DASD + RLRT)
> **Feature gate:** `directional_credit` (default on)
> **Dependencies:** Plan 171 (FrozenBaseGuard), Plan 112 (SR²AM Bandit), Plan 194 (Adaptive CoT)
> **Status:** Planning

---

## Summary

Apply DASD's entropy-routed directional insight to inference-time pruning. High-entropy "forking" tokens get relaxed screening (preserve exploration); low-entropy "scaffolding" tokens get tight screening (stabilize execution). Zero training cost — entropy is already computed in softmax. Feature-gated as `directional_credit`, default on because both papers prove strict improvement at every scale.

---

## Tasks

- [x] **T1: `EntropyBifurcatedPruner<P>` struct** — `src/pruners/entropy_bifurcated.rs`
  - Wrap any `ScreeningPruner` with entropy-aware routing
  - `top1_threshold: f32` (default 0.5) — below = "fork"
  - `relax_factor: f32` (default 0.3) — scale relevance at forks
  - `relevance()` checks top-1 prob from marginals
  - Zero extra forward pass (top-1 already available from DDTree)

- [x] **T2: `EntropyRoutedSchedule` variant** — extend `PrunerSchedule` enum
  - Add `EntropyRouted { threshold: f32 }` to `configurator_bandit.rs`
  - Route by per-token entropy instead of hop position
  - Keep `FrozenBaseGuard` as fallback option

- [x] **T3: ThinkingController entropy bias** — extend mode selection
  - Query entropy > threshold → bias toward Latent mode
  - Query entropy ≤ threshold → bias toward Direct mode
  - Use same `tanh` routing principle from DASD

- [x] **T4: SelfDrivenTokenTracker** — inference-time RLRT signal
  - Track DDTree top-1 changes from parent → child nodes
  - Feed "self-driven" signal into `BanditPruner<P>` as context
  - Self-driven tokens get exploration bonus

- [x] **T5: GOAT proof tests** — `tests/directional_credit_goat.rs` (6/6 passing)
  - G1: EntropyBifurcatedPruner returns different relevance for low-H vs high-H
  - G2: BanditPruner with SDTA context has better arm selection
  - G3: Top-1 mass correlates with actual entropy (> 0.8)
  - G4: EntropyRoutedSchedule beats FrozenBaseGuard in arena
  - G5: Zero overhead profile (entropy routing ≤ FrozenBaseGuard time)

- [x] **T6: Example** — `examples/directional_credit_demo.rs`
  - Before: uniform screening (all tokens treated equally)
  - After: entropy-bifurcated screening (low-H tight, high-H relaxed)
  - Show expected: more exploration at forks, more stability at scaffolding
  - `--features directional_cot` flag

- [x] **T7: Feature gate** — `Cargo.toml` + `mod.rs` wiring
  - `directional_credit` feature, default on
  - Module under `src/pruners/entropy_bifurcated.rs`
  - Re-exports in `mod.rs`

- [x] **T8: Benchmark** — `.benchmarks/052_directional_credit_goat.md`
  - Compare: Uniform > FrozenBaseGuard > EntropyRouted
  - Measure: screening precision/recall, exploration quality, execution stability
  - CPU/GPU route: verify zero overhead on both paths

---

## Expected Results

Based on DASD/RLRT papers:
- **Exploration quality**: +50-60% at high-entropy forks (relaxed screening)
- **Execution stability**: +18% StepAcc on low-entropy scaffolding (tight screening)
- **Overall**: strict improvement over uniform screening, zero degradation
- **Overhead**: zero (entropy is softmax byproduct)

---

## References

- DASD: arXiv:2605.22263
- RLRT: arXiv:2605.10781
- Research 164: `.research/164_DASD_RLRT_Direction_Adaptive_Self_Distillation.md`
- FrozenBaseGuard: Plan 171
- SR²AM Bandit: Plan 112
- Adaptive CoT: Plan 194
