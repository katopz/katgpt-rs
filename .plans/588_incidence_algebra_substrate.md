# Plan 588: Incidence-mask algebra substrate (`katgpt-core/incidence_algebra`)

**Status:** COMPLETE — T1 landed + validated (GOAT row done); consumers landed in riir-ai (Issue 874 T2) + riir-game-sdk (stealth) the same pass

**Source:** [../riir-ai/.research/364_ThoughtComm_Agreement_Tiered_Thought_Routing.md](../riir-ai/.research/364_ThoughtComm_Agreement_Tiered_Thought_Routing.md) §Fusion item 2 ("the missing algebra") + [../riir-ai/.issues/874_thoughtcomm_agreement_tiered_contagion.md](../riir-ai/.issues/874_thoughtcomm_agreement_tiered_contagion.md) T1.

## Why

The stack builds thought × agent incidence masks everywhere by construction (CLR observer sets, sheaf restriction maps, npc_comms slices, healer fan-out hits) and computes none of the paper's algebra over them. Substrate-first verified: the closest shipped analogs (`agreement_counts` in katgpt-speculative ppot rank — per-variant decode consistency; `support_count` in skill_opt — edit confidence) consume different signals entirely; no cross-agent incidence object exists. The paper's Thm 3 licenses treating the mask as the first-class identifiable artifact; this plan ships the zero-alloc integer/set algebra over it.

## Tasks

- [x] `src/incidence.rs` behind opt-in feature `incidence_algebra` (zero deps, zero-alloc hot path):
  - `agreement_counts_into` — per-thought `α_j = Σ_k 1[j ∈ support(k)]` (agent-major mask layout)
  - `support_sizes_into` — per-agent support size (the T3 audit half)
  - `agreement_score(α, κ) = σ(κ·(α−1))` — the raw monotone σ-saturated curve, α≤1 anchored to EXACTLY 0.5
  - `routing_weight = 0.5 + score` — α=1 is EXACTLY 1.0 (bit-identical to the unweighted path; soft bias, never a hard gate — the Bench-013 lesson)
  - `contagion_strength = 2·score − 1` — α=1 is EXACTLY 0.0 (a single witness broadcasts at zero crowd strength; the Plan-019 CLR fix); κ=0 is the kill-switch (c≡0, w≡1)
  - `shared_private_counts` — shared (α≥2) / private (α==1) split (paper Thms 1+2)
  - `private_fractions_into` — the consensus≠accuracy guard: `|S_private(i)|/|support(i)|`; empty support → 1.0
  - `HopcroftKarpScratch` + `hall_max_matching_into` (+ allocating convenience) — Hall feasibility, deterministic ascending-neighbor order
  - `MaskAudit` + `audit_mask` + `DENSITY_ALERT = 0.5` — density alert is a WARNING, never a gate
  - `rank_by_agreement[_into]` — α desc, index asc (deterministic total order before any truncation — the Issue-849 lesson)
- [x] Feature declaration in `Cargo.toml` + gated module in `lib.rs` (the `evidence_tripwire` pattern)
- [x] Tests: planted-mask exactness; all-shared (α=N) / all-private (α=1) non-vacuity controls; degeneration contracts pinned at `to_bits` (w(1)≡1.0, c(1)≡0.0, κ=0 kill-switch); monotonicity + saturation tables; private-retention collapse (frac→0) vs diverse (frac=1) separation; Hall cross-checked against exhaustive brute force over 300 seeded random masks + planted violation; audit dense-alert / sparse-control; permutation-invariance property (40 seeded worlds: α invariant, support sizes + private fractions equivariant, audit invariant); zero-alloc hot path via `crate::alloc` counters (`debug_assertions`, the crate's own convention)
- [x] GOAT: `cargo clippy` 0 findings in touched files (the 7 lib warnings are the pre-existing judgement-class survivors in dec_freeze/ssmax/traits/hebbian_kernel_memory/trigger_gate — not this module) + `cargo test -p katgpt-core --lib --features incidence_algebra` green (1985 passed / 0 failed; 11 new incidence tests) + default-feature lib count UNMOVED (1974 = 1974; the module compiles to nothing without the feature)

## Non-goals

- No consumer wiring in this repo (consumers are riir-ai `tick_swarm_emotions_collective*` + riir-stealth `tick_alarm` — they land with riir-ai Issue 874 T2).
- No sync surface (masks/α/weights are think-brain local; raw scalars only cross sync).
- No training-track surface (the AE + prefix adapter are riir-train Research 442, deferred with triggers).

## Validation commands

```bash
cargo clippy -p katgpt-core --features incidence_algebra --all-targets
cargo test  -p katgpt-core --lib --features incidence_algebra
cargo test  -p katgpt-core --lib                     # count must be UNMOVED
```
