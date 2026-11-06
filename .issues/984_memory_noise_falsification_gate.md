# Issue 984 — Noise/shuffled/empty store falsification arms as a standing GOAT-gate pattern for memory-input primitives

**Status:** CLOSED — landed 2026-11-06, commit `<this>` (T1–T3 + T5 done; T4 deferred `- [-]` with reason).
**Lease closed:** zed-m3-agent (katgpt-rs T1–T5), 1793928266, expires 2026-11-08T01:24:26Z — completed within lease.

## Motivation

Memorilla's most transferable scientific instrument is not its architecture — it is the **noise-store falsification arm** and its calibration proof: under a Gaussian-noise store, Memorilla loses −5.93 on FactKG, the Q-Former −5.11, but the documents-only variant moves **+0.04** — i.e. a read that does not condition on evidence is noise-INVARIANT. The noise delta therefore measures evidence-dependence in both directions, and the paper's sibling control (matched-parameter LoRA WITHOUT documents loses on every primary metric) shows the gains came from the store, not the adaptation budget.

Our stack has memory-input primitives whose gates check determinism, cost and abstention behavior — but no standing arm that asks **"does this primitive actually consume its store?"** A memory consumer whose input is garbage must fail loudly (same discipline as the timed-region loud-zero defense: absent work satisfies bars; a noise-invariant read is absent work wearing numbers).

## Existing partial coverage (credit where due — do not re-describe as greenfield)

`crates/katgpt-core/benches/bench_845_distance_abstain_goat.rs` (cited by FULL PATH — katgpt-attn carries an unrelated `bench_845_channel_aware_dot_dispatch.rs`; always spell the target out) already ships an OOD arm and a **W2 negative-control world** that falsifies a rigged fixture. That is real falsification coverage for the GATE's fixture. What it does NOT have: store-level corruption arms. The three arms below are ADDITIONAL controls alongside W2, not a replacement.

## The four store arms (preregistered, as amended)

Run the frozen eval under four store conditions + one budget control:

1. **real** — the store as shipped.
2. **noise** — BLAKE3-seeded iid Gaussian matched to the store's GLOBAL entry scale, rows unit-normalized. *(AMENDED at first measurement — see the lesson below; the original "per-dimension norm statistics" matching preserved the between-cluster subspace and a max-sim read recovered structure right through it.)*
3. **shuffled** — one fixed BLAKE3-seeded coordinate permutation applied to every row (the honest instantiation of "permuted embedding↔provenance binding" for a set-max read: internal geometry conserved exactly — an orthogonal permutation conserves all pairwise cosines — the query↔store basis channel destroyed; a ROW permutation would be the identity on the read).
4. **empty** — no primitive / no store (the no-memory floor; for Bench 845, the score-only signal). The paper's 0.153-vs-0.186 lesson says a bad memory arm can sit BELOW this.
5. *(when cheap)* **adaptation-budget control** — feature-off vs budget-matched no-store arm — not applicable here: `CorpusDistanceGate` has no adaptation budget (no learned state; "the corpus rows ARE the input").

Preregistered verdict inequalities (any violation = the primitive is injecting phantom structure or ignoring evidence; AURC lower-is-better):

- **I1 evidence-dependence** — `AURC(noise) ≥ AURC(real) + margin` (margin 0.005)
- **I2 structure-hallucination** — `AURC(noise) ≥ AURC(empty) − tol` (tol 0.001)
- **I3 binding matters** — `AURC(shuffled) ≥ AURC(real) + margin` (margin 0.005)

## The measured lesson (the arms' first catch — the arms caught THEMSELVES)

The first noise arm matched **per-dimension** mean/std. At Bench 845 full scale (K=128, D=64) that arm FAILED I1: noise AURC 0.2701 vs real 0.2704 — noise delta −0.0003, i.e. the noise store served the read as well as the real store. Root cause: per-dim std is dominated by BETWEEN-CLUSTER variance, so the noise rows concentrate near the cluster-center subspace, and a max-sim read recovers structure right through that bias. The instrument did exactly its job — it flagged an under-corrupted ARM, not a bad primitive. The noise arm was amended to global-energy-matched iid (structure-free by design), whereupon every inequality holds with a real delta. **Standing rule for the pattern: the noise arm must match GLOBAL energy only; per-dim marginal matching is NOT structure destruction for max-similarity reads.**

## Landing record (measured)

- Harness: `crates/katgpt-core/tests/common/store_falsification.rs` — the reusable pattern module (generic over a boxed `StoreReader`, std + blake3 dev-dep only, feature-neutral). `aurc`/`selective_accuracy_at` moved here from bench_845 (one metric implementation; bench behavior byte-identical — G1–G5 re-run green).
- Bench: `bench_845_distance_abstain_goat.rs` gains **G6** (full scale, K=128, D=64, 8 reps, W1 worlds):
  - real 0.2704 AURC · noise 0.2951 · shuffled 0.2907 · empty 0.2289 (score-only floor)
  - **noise deltas: +0.0246 AURC, +0.0479 sel30** (the Memorilla +0.04-style readout — a nice echo)
  - I1/I2/I3 all OK; canary store-blind stub: identical metrics under every store (noise delta exactly +0.0000) → caught by I1+I3 → polarity OK
  - GATES: G1–G6 all true.
- Named gate row: `scripts/test_gate.sh` PERF_ROWS gains `katgpt-core:4:memory_store_falsification_gate:distance_abstain` (release, single-threaded; the integration target `tests/memory_store_falsification_gate.rs` runs the arms + canary + transform determinism/geometry tests at test scale, floor 4 passed).
- G1 determinism: BLAKE3 store-seed domain-separated by label + store bits; `noise_store` byte-identical across calls (test), a flipped bit reseeds (test).
- G2 harness cost: warm release run base 0.507 s (worktree at HEAD, no G6) vs +G6 0.698 s = **1.38×** ≤ 2× bar. (Both runs warm, same box, AC power; cost ratio, not a latency claim.)
- G-noise liveness: the stub canary fails the arms (I1+I3) and `canary_holds` asserts the polarity — a stub that PASSES is itself a gate failure ("gate blind").

## Tasks

- [x] T1: harness module with the four store arms over the Bench 845 fixture (BLAKE3-seeded noise matched to per-dim norm stats; provenance-shuffle keyed the same way) — done, with the noise construction amended to global-energy matching per the measured lesson
- [x] T2: apply to `CorpusDistanceGate` (bench_845) — expected result CONFIRMED after the arm amendment: real ≥ noise with delta +0.0246 AURC / +0.0479 sel30; the noise delta printed as the instrument readout
- [x] T3: verdict table + preregistered inequalities + exit 1 on violation (bench G6 asserts in the GATES conjunction); wired as named row `katgpt-core:4:memory_store_falsification_gate:distance_abstain` in `scripts/test_gate.sh` PERF_ROWS
- [-] T4: apply the same arms to one game-side consumer (episodic ring read via `riir-games-shared` bench 621 harness, feature-gated) — DEFERRED: the seam lives in ../riir-ai (episodic_sighting opt-in feature, sibling bench harness); cross-repo claim + integration is a separate session's work and riir-ai benches are an active sibling-agent lane. The harness module is feature-neutral and generic (`StoreReader` boxed reader + `StoreWorld`), so the game-side port is a thin adapter — file there when picked up.
- [x] T5: pattern documented as the standing convention: the harness module doc (vocabulary translation for set-max reads + the per-dim lesson), the bench G6 header paragraph, and the Research 656 note line. Every future memory-input primitive's GOAT gate inherits the arms.

## Acceptance

G1: arms deterministic across runs (BLAKE3-seeded). G2: harness cost ≤ 2× the base bench run — measured 1.38×. G-noise: the inequalities hold for CorpusDistanceGate and fire (exit 1) on a deliberately noise-invariant stub read (the canary proves the gate is live) — verified in both the bench G6 and the named test row.
