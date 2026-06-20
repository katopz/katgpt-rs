# Issue 038: GLASS Flows (Remark 21) for MCTS Collapse Bridge

**Opened:** 2026-06-20
**Origin:** Research 271 §5 (MIT 6.S184 textbook vocabulary crosswalk)
**Status:** TBD fusion candidate — **novelty NOT yet verified, needs Q1–Q4 gate before any verdict**
**Parent skill rule:** "If you are NOT confident enough to commit all 4 YES right now, do not write 'Super-GOAT candidate'. Write 'fusion idea — novelty TBD, needs Q1–Q4 check before verdict' and create an issue."

---

## The fusion idea

MIT 6.S184 Remark 21 (GLASS Flows, citing Holderrieth et al. 2025, arxiv 2509.25170): stochastic-looking transition dynamics can be implemented **purely via ODEs** using a sampling trick. This allows search algorithms (MCTS, beam search) over stochastic-looking dynamics while keeping the efficiency and determinism of ODE simulation.

**Proposed fusion:** apply GLASS Flows to `riir-ai/crates/riir-engine/src/cgsp_runtime/mcts_collapse_bridge.rs`. The collapse bridge currently switches between MCTS (stochastic search) and direct flow (deterministic). GLASS Flows could provide a **unified ODE-based formulation** that:
- Looks stochastic to the search algorithm (enables branching / curiosity)
- Is actually deterministic underneath (bit-identical replay, anti-cheat safe)
- Avoids the bridge-switching cost

This connects:
- **MCTS collapse recovery** (existing `mcts_collapse_bridge.rs`)
- **Deterministic replay** (raw sync requirement)
- **Curiosity-driven search** (cgsp_runtime)
- **Freeze/thaw** (no retraining — just a different ODE formulation at runtime)

## Why it might be novel

GLASS Flows (Holderrieth et al. 2025) is recent. A quick codebase grep did not surface "glass_flow", "glass_flow", "ode_stochastic", or "deterministic_stochastic" as shipped primitives. The mcts_collapse_bridge uses an explicit branch-and-switch pattern, not a unified ODE.

## Why it might NOT be novel (Q1–Q4 to run before any verdict)

- **Q1 (no prior art?):** Must grep for: paper vocabulary (`glass_flow`, `transition_sampling`, `alignment_flow`) AND codebase vocabulary (`ode_search`, `deterministic_branch`, `replay_safe_search`, `mcts_ode_unified`). The collapse bridge might already implement the GLASS pattern under a different name.
- **Q2 (new capability class?):** Does unified ODE-based stochastic-looking search enable behavior the current switch-bridge cannot? Or is it just a cleaner implementation of the same capability?
- **Q3 (product selling point?):** "Our NPCs run MCTS-quality strategic search with deterministic-replay safety, no bridge switching." Finish this sentence or downgrade.
- **Q4 (force multiplier ≥2 pillars?):** Connects to MCTS, sync boundary, cgsp, replay verification. Plausible ≥2.

## Caveats

- The GLASS Flows paper (arxiv 2509.25170) is cited only in a Remark in the lecture notes — the actual paper needs to be fetched and read before any verdict. The lecture-note description is too thin to claim novelty from.
- "Search algorithms over stochastic-looking ODE dynamics" is a broad claim. The actual paper may be narrower (e.g., specific to alignment, not general MCTS).

## Action required

Do NOT promote this to a plan or Super-GOAT candidate until:

1. Fetch and read the actual GLASS Flows paper (arxiv 2509.25170) via `https://r.jina.ai/https://arxiv.org/pdf/2509.25170`
2. Verify the mechanism is what the lecture-note Remark claims (general ODE-based stochastic-looking transitions, not alignment-specific)
3. Run the Q1–Q4 novelty gate with full vocabulary translation across both repos, both layers
4. Latent-space reframing: which Super-GOAT factory module owns it? (cgsp_runtime and latent_functor are the candidates)

## Reading list

- MIT 6.S184 lecture notes §4.2 Remark 21 (GLASS Flows)
- GLASS Flows paper: [arxiv 2509.25170](https://arxiv.org/abs/2509.25170) — **must read before any verdict**
- `katgpt-rs/.research/271_MIT_6S184_Diffusion_Flow_Textbook_Vocabulary_Crosswalk.md` (vocabulary crosswalk)
- `riir-ai/crates/riir-engine/src/cgsp_runtime/mcts_collapse_bridge.rs` (existing collapse bridge)
- `katgpt-rs/.research/215_ECHO_Environment_Prediction_Inference_Time.md` (related inference-time prediction)
