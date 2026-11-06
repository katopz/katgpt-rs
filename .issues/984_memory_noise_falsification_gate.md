# Issue 984 — Noise/shuffled/empty store falsification arms as a standing GOAT-gate pattern for memory-input primitives

**Status:** OPEN — poc/proof task filed from Research 656 (arXiv:2610.21375 Memorilla distillation, delegated-owner AGREE 2026-11-06). Claimed: nobody yet.

## Motivation

Memorilla's most transferable scientific instrument is not its architecture — it is the **noise-store falsification arm** and its calibration proof: under a Gaussian-noise store, Memorilla loses −5.93 on FactKG, the Q-Former −5.11, but the documents-only variant moves **+0.04** — i.e. a read that does not condition on evidence is noise-INVARIANT. The noise delta therefore measures evidence-dependence in both directions, and the paper's sibling control (matched-parameter LoRA WITHOUT documents loses on every primary metric) shows the gains came from the store, not the adaptation budget.

Our stack has memory-input primitives whose gates check determinism, cost and abstention behavior — but no standing arm that asks **"does this primitive actually consume its store?"** A memory consumer whose input is garbage must fail loudly (same discipline as the timed-region loud-zero defense: absent work satisfies bars; a noise-invariant read is absent work wearing numbers).

## Existing partial coverage (credit where due — do not re-describe as greenfield)

`crates/katgpt-core/benches/bench_845_distance_abstain_goat.rs` (cited by FULL PATH — katgpt-attn carries an unrelated `bench_845_channel_aware_dot_dispatch.rs`; always spell the target out) already ships an OOD arm and a **W2 negative-control world** that falsifies a rigged fixture. That is real falsification coverage for the GATE's fixture. What it does NOT have: store-level corruption arms. The three arms below are ADDITIONAL controls alongside W2, not a replacement.

## The four store arms (preregistered)

Run the frozen eval under four store conditions + one budget control:

1. **real** — the store as shipped.
2. **noise** — BLAKE3-seeded Gaussian matched to the store's per-dimension norm statistics (no RNG drift; determinism gate applies).
3. **shuffled** — permuted embedding↔provenance binding (does the BINDING matter, not just the geometry).
4. **empty** — no primitive / no store (the no-memory floor; the paper's 0.153-vs-0.186 lesson says a bad memory arm can sit BELOW this).
5. *(when cheap)* **adaptation-budget control** — feature-off vs budget-matched no-store arm.

Preregistered verdict inequalities (any violation = the primitive is injecting phantom structure or ignoring evidence):

- `real ≥ noise + margin` (evidence-dependence, positive direction)
- `noise ≤ empty` (structure-hallucination check — noise beating no-primitive is a red)
- `shuffled ≤ real` (binding matters)

## Proof path

- **Fixture:** the Bench 845 synthetic clustered fixture (K=128, D=64, 4 clusters — a synthetic fixture, not a corpus) + the existing W1/W2 worlds; noise/shuffled arms are pure harness code over the same fixture, zero new deps.
- **Falsification arm:** the noise arm IS the falsification arm (a deliberately corrupted store must degrade the real arm's score and leave the verdict inequalities checkable).
- **Pre-registered gate:** the three inequalities above, printed as a verdict table with the noise delta per primitive (the +0.04-style instrument readout), exit 1 on any violated inequality.

## Tasks

- [ ] T1: harness module with the four store arms over the Bench 845 fixture (BLAKE3-seeded noise matched to per-dim norm stats; provenance-shuffle keyed the same way)
- [ ] T2: apply to `CorpusDistanceGate` (bench_845) first — expected result: real ≥ noise (the gate reads max-sim over real exemplars, so a noise store must collapse confidence toward the empty floor); print the noise delta as the instrument readout
- [ ] T3: verdict table + preregistered inequalities + exit 1 on violation; wire as a named row in the relevant gate script (test_gate/suite that already runs bench_845)
- [ ] T4: apply the same arms to one game-side consumer as the cross-repo proof (episodic ring read via `riir-games-shared` bench 621 harness, feature-gated) — defer-able if the harness seam is heavier than expected; mark `- [-]` with reason, do not silently drop
- [ ] T5: document the pattern as the standing convention for NEW memory-input primitives (one paragraph in the bench header comment + a line in the research note); every future memory primitive's GOAT gate inherits the arms

## Acceptance

G1: arms deterministic across runs (BLAKE3-seeded). G2: harness cost ≤ 2× the base bench run. G-noise: the inequalities hold for CorpusDistanceGate and fire (exit 1) on a deliberately noise-invariant stub read (the canary proves the gate is live).
