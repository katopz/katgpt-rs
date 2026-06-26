# AGENTS.md — katgpt-rs

The global `~/.agents/` rules apply; this file documents repo-local context
that supplements them.

## Modelless-first mandate (the core principle)

**This repo ships modelless inference primitives.** No training, no backprop,
no gradient descent. The only weight mutations allowed at runtime are:

1. **Freeze/thaw** — swapping a frozen snapshot (atomic, versioned, BLAKE3-checked).
2. **Raw/lora hot-swap** — applying a **deterministically constructed** (not
   trained) LoRA overlay via `LoraPair { reader, writer }` (Plan 025).
3. **Latent-space updates** — direction-vector projections, sigmoid gates,
   routing tables. These update latent state, NOT base weights.

### MANDATORY: exhaust modelless paths before deferring to riir-train

Before deferring ANY gate, mechanism, or plan task to riir-train ("this needs
training"), you MUST check whether the three modelless paths above can fix it.
See the research skill §3.5 (`.agents/skills/research/SKILL.md`) for the full
decision protocol.

**Systematic, characterizable biases are modelless-correctable candidates,
NOT automatic riir-train dependencies.** If a gate fails because of a known,
named bias (e.g., "signal doubled", "position offset", "attention asymmetry"),
check whether a deterministically constructed reader-LoRA or freeze-state
correction can fix it before concluding "needs gradient descent."

**Canonical failure — AC-Prefix G1 (Plan 313, 2026-06-24):** G1 was prematurely
deferred to riir-train without checking whether the doubled-signal bias could
be corrected modellessly via a deterministic reader-LoRA. The bias was
systematic and characterizable — exactly the case where raw/lora hot-swap
might work. The deferral was premature and has been reverted; the modelless
investigation (Issue 003, resolved-and-removed in commit `552b4632`) is
captured in `.benchmarks/313_ac_prefix_modelless.md` (Path 2: `attends_dedup`
eliminates the bias bit-identically to iterative-MLM on single-layer
micro-GPT, 0.0 diff). `ac_prefix` re-promoted to DEFAULT-ON on that
modelless pass; multi-layer equivalence remains a non-blocking riir-train
follow-up.

## Build Commands

```bash
# Default features (the GOAT-validated, promoted primitives)
cargo check
cargo test -p katgpt-core --lib

# Single feature
cargo check --features <feature_name>

# All features
cargo check --all-features

# Specific feature's tests
cargo test -p katgpt-core --features <feature_name> --lib
```

## Feature Flag Discipline

Every new primitive ships behind a feature flag (opt-in). Promotion to
default-on requires the GOAT gate to pass:

1. Implement behind `feature_name = []` (opt-in).
2. Write a benchmark proving the gain (latency, quality, or security).
3. Run the GOAT gate (G1 correctness, G2 perf, G3 no-regression, G4 alloc-free
   or equivalent).
4. If all gates pass AND the gain is **modelless** → promote to `default`.
5. If the gain requires riir-train (training) → keep opt-in, note the
   dependency, do NOT promote to default.

**Promotion requires modelless gain.** A perf gain on a biased/incorrect answer
is NOT a modelless gain — it's a speedup of a wrong result. The quality gate
(G1 or equivalent) must pass modellessly for the GOAT to hold.

## Research Workflow

See `.agents/skills/research/SKILL.md` for the full research workflow:
paper classification, 5-repo routing, fusion-first distillation, novelty gate,
GOAT gate, and the mandatory modelless-unblock protocol (§3.5).

## Branch

`develop` is the working branch. Don't create feature branches; commit
directly on `develop` per the global rule.
