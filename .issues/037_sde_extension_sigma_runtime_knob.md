# Issue 037: SDE Extension σ as Runtime Determinism/Exploration Knob

**Opened:** 2026-06-20
**Origin:** Research 271 §5 (MIT 6.S184 textbook vocabulary crosswalk)
**Status:** TBD fusion candidate — **novelty NOT yet verified, needs Q1–Q4 gate before any verdict**
**Parent skill rule:** "If you are NOT confident enough to commit all 4 YES right now, do not write 'Super-GOAT candidate'. Write 'fusion idea — novelty TBD, needs Q1–Q4 check before verdict' and create an issue."

---

## The fusion idea

MIT 6.S184 Theorem 17 (SDE Extension Trick): given a trained flow model with vector field `u_θ_t(x)`, you can sample via either

```
ODE (deterministic):   dX = u_θ_t(X) dt
SDE (stochastic):      dX = [u_θ_t(X) + (σ²_t/2) ∇ log p_t(X)] dt + σ_t dW_t
```

for **any** `σ_t ≥ 0`, chosen **at inference time**, no retraining.

**Proposed fusion:** expose `σ_t` as a per-NPC, per-zone, or per-context runtime knob that switches between:
- **σ = 0** (sync-critical NPCs, quorum commit, deterministic replay, anti-cheat)
- **σ > 0** (exploring NPCs, curiosity-driven, generates novel latent trajectories)

This connects:
- **Freeze/thaw over fine-tuning** (don't retrain to add stochasticity, just turn the σ knob)
- **Sync boundary** (raw values still commit deterministically; only latent exploration is stochastic)
- **Curiosity/exploration** (`cgsp_runtime` — currently uses decayed-absorb bandits, NOT SDE noise)
- **Per-NPC HLA divergence** (different σ per NPC → emergent behavioral diversity)

## Why it might be novel

Reading `cgsp_runtime/runtime.rs`: curiosity is currently modeled as a **decayed-absorb priority bandit** (`p ← p·decay + reward`, decay=0.7). This is **NOT** Langevin dynamics or SDE-driven exploration. The textbook's `σ_t` is a genuinely different mechanism for the same goal (intrinsic exploration).

`bench_elf_modelless.rs::bench_sde_noise_injection_overhead` benchmarks SDE noise injection **cost** but does not expose it as a runtime determinism/exploration knob.

## Why it might NOT be novel (Q1–Q4 to run before any verdict)

- **Q1 (no prior art?):** Must grep `katgpt-rs/crates/`, `riir-ai/crates/`, `riir-armageddon/crates/` for `sigma`, `noise_inject`, `langevin`, `stochastic_explore`, `brownian`, `sde_extension`, AND codebase-vocabulary alternatives (`rng_inject`, `explore_noise`, `curiosity_noise`, `tick_jitter`). The MIT 6.S184 grep (Research 271) only confirmed it's not in the **diffusion-inference** path; it might be in `cgsp_runtime`, `npc/`, or `plasma/` under a different name.
- **Q2 (new capability class?):** "Per-NPC stochastic exploration via SDE noise injection on latent state, gated by sync-tier" — does this enable behavior no incumbent (bandits, MCTS collapse bridge) can?
- **Q3 (product selling point?):** "Our NPCs have intrinsic curiosity-driven stochasticity that's bit-identical reproducible when needed and exploratory when wanted — all from one runtime knob, no retraining." Finish this sentence or downgrade.
- **Q4 (force multiplier ≥2 pillars?):** Connects to freeze/thaw, sync boundary, cgsp_runtime, HLA divergence. Plausible ≥2. Needs explicit verification.

## Action required

Do NOT promote this to a plan or Super-GOAT candidate until Q1–Q4 pass. When working on it:

1. **Vocabulary translation** (per workflow §1 step 2): list 5+ codebase-equivalent terms for "σ", "Brownian motion", "Langevin", "stochastic exploration", "noise injection", then grep BOTH paper vocabulary AND codebase vocabulary across BOTH repos, BOTH layers (notes + code).
2. **Latent-space reframing** (per workflow §1 step 3): does σ injection belong in HLA update, latent_functor arithmetic, or cgsp cycle? Which Super-GOAT factory module owns it?
3. **Sync-boundary respect**: σ_t only affects latent exploration. Raw values MUST stay bit-identical (MapPos, HP, wallet balance, etc.). Bridge is one-way: latent σ noise → scalar clamp at sync boundary, never the reverse.
4. **Latent vs raw classification**: emotion/mood/curiosity/strategy = latent (σ eligible). Position/velocity/HP/wallet = raw (σ = 0 always).

## Reading list

- MIT 6.S184 lecture notes §2.2 (Diffusion Models), §4.2 (Theorem 17 SDE Extension Trick), §4.3 Remark 20 (Langevin dynamics)
- `katgpt-rs/.research/271_MIT_6S184_Diffusion_Flow_Textbook_Vocabulary_Crosswalk.md` (vocabulary crosswalk)
- `riir-ai/crates/riir-engine/src/cgsp_runtime/runtime.rs` (current curiosity implementation — bandits, NOT SDE)
- `katgpt-rs/tests/bench_elf_modelless.rs::bench_sde_noise_injection_overhead` (existing cost benchmark)
- `katgpt-rs/.research/215_ECHO_Environment_Prediction_Inference_Time.md` (related inference-time prediction)
- `katgpt-rs/.research/236_QGF_Test_Time_Q_Guided_Flow.md` (test-time gradient guidance, adjacent framing)
