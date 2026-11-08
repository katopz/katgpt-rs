# Plan 633 — Adoption/retention regression gate (AGCL form; issue 987 priority-1, fixture-first)

**Status:** ACTIVE — T0 LANDED (fixture + primitive; landing commit recorded below). T1–T4 stay unchecked pending their preconditions (ungoat discipline: the fixture IS the proof path; the real-store GOAT cell opens only after a consumer names its store).
**Claimed:** zed-86011726@m3, 1794136639, expires 2026-11-10T11:20:00+00:00

Source: `.issues/987_cactus_neurips2026_follow_ups.md` priority-1 lead — [P] gate form from AGCL (Research 657 §A5) + [ours] consumers + judge-noise handling. **Substrate-first verdict (2026-11-08, vocabulary translation run):** NO existing outcome-retention gate substrate in katgpt-rs — every `retention` hit is KV/attention-side (`dash_attn` eviction windows, needle retention), every `dedup` hit is AC-prefix attention masking or ARG ontology `RegistryDedup`; `RingRow<T,32>` lives in riir-neuron-db, which this upstream repo cannot dep (BOUNDARY law). A new primitive is justified; it mirrors the ring shape locally.

## What ships (T0 — the runnable fixture + primitive) — LANDED

`katgpt-core/src/adoption_gate.rs` behind the opt-in feature `adoption_gate` (modelless, zero new deps — blake3 is already non-optional):

- `OutcomeDigest` — newtype over `[u8; 32]` (blake3 of the outcome bytes).
- `DedupRing` — fixed 32-slot FIFO membership set over digests: `insert` returns NEW-or-seen (evicts oldest at capacity), `contains` scans 32 slots, zero alloc, `size_of` pinned by test.
- `Decontaminator` — the never-double-file splitter: every `stride`-th ACCEPTED outcome → held-out audit set A, the rest → gate set T; index-deterministic (`DEFAULT_DECONTAMINATION_STRIDE = 5`); an outcome filed to A can never also file to T (by construction + test).
- `score_adoption` — PURE scorer: score `= (1−α)·retained + α·(1−repeated)` where `retained = |candidate accepted ∩ prev T-accepted| / max(1, |prev T-accepted|)` and `repeated = |candidate outputs ∩ prev T-rejected| / max(1, |candidate outputs|)`; f64 sequential folds; ONE fixed tie rule: **ties → challenger**; `DEFAULT_ADOPTION_ALPHA = 0.5`. Consumes `(rings, slices, α)` — never incumbent internals (always-from-base, purity-pinned).
- `AdoptionGate::decide` — gate wrapper: NO-EVIDENCE (both T sets empty) → keep incumbent (decline is the default); otherwise adopt iff candidate score ≥ incumbent score on the same T window.

Fixture tests (the proof path — synthetic outcome streams, LABELED synthetic gate-mechanics, never quotable as serving quality — the 976-corpus lesson mirrored):

1. degraded-challenger red: drops previously-accepted outcomes → scores below incumbent → gate KEEPS incumbent (the AGCL failure class).
2. genuine-improvement green: retains all accepted, stops repeating a rejected output → wins; tie → challenger.
3. never-double-file: ≥500-outcome stream at stride 5 → no digest in both T and A.
4. ring bound: >32 distinct digests → oldest evicted, membership correct.
5. purity/determinism: same inputs → identical verdict bits, replayed ×2.

**T0 landing (2026-11-08):** module 715 lines + 9 tests (the five fixture gates + size/dedup/no-evidence/incumbent-equivalence pins); clippy `-D warnings` green at BOTH feature postures; rustfmt per-file clean. Pinned `size_of::<DedupRing>()` = 1064 B (parallel-arrays layout — `Option<[u8;32]>` has no niche, parallel `occupied: [bool; 32]` + `slots: [[u8; 32]; 32]` beats it). Documented deviations, all test-pinned: `Decontaminator::file` is content-free (returns the destination enum; sinks consume digests — never-double-file unrepresentable); cross-ring ingest dedup keeps accepted/rejected T-rings disjoint by construction; incumbent score derived from the rings alone (`retained ≡ 1` over its own window, `repeated = |rej|/(|acc|+|rej|)`) with bit-for-bit equivalence pinned vs `score_adoption` over ring-derived slices; `TIE_GOES_TO_CHALLENGER` is a real `pub const` read by `decide` (flip demotes ties to Keep); `NoEvidence` verdict carries informational score (== α). Lint path: cargo-refine's auto-oracle sees cfg-gated code as nothing at default features (0 findings, 0 edits) → the two armed-posture mechanicals (`filter_map_bool_then`, `manual_is_multiple_of` — the latter on the full-gate `-D` list) fixed manually, the documented fallback. Docs: feature census 729→730 (README ×4 sites + examples/README + catalog §164).

## Gates after T0 (leave UNCHECKED until their precondition lands)

- [ ] T1 — consumer wiring: riir-refine active-set demote/readmit + `seat_rerank` head swaps consume the gate (file locally in riir-refine citing 987 + this plan; the 632 `elo_k_eff` precedent shape).
- [ ] T2 — instinct LaneSlot epoch-swap consumer (file in riir-instinct likewise).
- [ ] T3 — GOAT cell: gate-on vs gate-off on the REAL store (fix trajectories + oracle verdicts); α under judge noise measured (the issue's open caveat — the fixture's perturbation arm is the harness, the real-store number is the verdict); may fail honestly — a verdict, not a bug. Pre-registered bars at T3 open time.
- [ ] T4 — adoption-proof sidecar on the freeze envelope (987's second lead) — only after T3 produces evidence worth sealing.

## Constraints

- Modelless; linear α mix (no softmax anywhere); fixed `[T; 32]` rings not Vec; f64 sequential folds (Issue 940); tie rule pinned in-code; decline-first on no evidence.
- Rust house rules: clippy `-D warnings` clean at BOTH default and `--features adoption_gate`; `CARGO_TARGET_DIR=/tmp/plan633`; `cargo refine fix` before manual lint fixes; `rustfmt <file>` per touched file (never `cargo fmt -p`); `.rs` ≤ ~2048 lines n/a (new file, keep modular).

Refs: issue 987 · Research 657 §A5 · riir-neuron-db `RingRow` (the mirrored shape, cannot dep) · katgpt-core `rating`/`step_noise` (decision-machinery neighbors) · riir-refine 632 `elo_k_eff` (consumer precedent) · AGENTS.md determinism law (Issue 940).
