# Plan 576: Hint-Regret VoI Primitive (katgpt-core)

> **Source:** Research 496 (SPADE distillation) · Research 500 (EnvHarness wrap-refinement, 2026-08-22) · Guide 340 (riir-ai)
> **Date:** 2026-08-21 (folded 2026-08-23 — owner decision, perf+sec)
> **Repo:** katgpt-rs (public, MIT) — generic math, no game semantics
> **Feature:** `hint_regret` (opt-in; promotion only after G1–G4+G8)

## Goal

Ship the modelless hint-regret primitive SPADE (arXiv:2608.19197) distills to: a paired-rollout value-of-information estimator + sigmoid band-pass difficulty gate + three-regime triage + Beta-LCB frontier ordering. This is the open half of the Super-GOAT; the game-side guide is riir-ai `.research/340`.

## Design constraints

- Zero-alloc hot path; scratch buffers; fixed-size where bounded.
- Sigmoid, never softmax (band gate = product of two sigmoids).
- UQ-bearing (the estimator claims a gap estimate with a confidence bound) → **"Report the Floor" rule**: benchmark the estimator's decision quality against a naive floor (single-arm win-rate banding without the hint arm) — the paired estimator must beat the floor on triage accuracy at matched rollout budget, or the gate FAILS.
- Deterministic under seed (CRN pairing is load-bearing for G2 variance reduction).
- **Composition discipline (Research 500 fold, 2026-08-23):** the game-side difficulty LEVER is modifier composition (Setup/Rule/Link over the frozen quest table — Guide 340 §"Wrapper composition"), never a scalar knob. The primitives below COMPOSE with that wrapper algebra: the band gate consumes the per-composition Beta posterior mean as `w`; the bandit arms ARE modifier compositions. The wrapper algebra itself stays game-side (riir-ai) — promote to an open primitive only if PoC 677 vindicates generality (Research 500 P3).

## Phase 1 — Estimator core (`crates/katgpt-core/src/hint_regret.rs`)

- [x] `HintRegretEstimator` struct: paired arms, CRN shared seeds, rolling K with Hoeffding `K(ε,δ) = ⌈(b−a)²/(2ε²)·ln(2/δ)⌉`, sequential stopping on CI half-width. *(Landed as `src/hint_regret/mod.rs` — the plan's single file became a module dir at Phase 2.)*
- [x] `estimate() -> RegretEstimate { r_hat, ci_half_width, n_pairs, arm_means }` — pure arithmetic over recorded pairs, alloc-free.
- [x] Analytic oracles as test fixtures: reveal-the-arm bandit (hint exposes μ*; `r = μ* − max_j μ̂_j`), hinted shortest path (β→∞ returns demo return bit-exactly).
- [x] G1 gate: estimator within 2× Hoeffding bound at prescribed K across 10³ seeds; coverage of the δ-guarantee ≥ nominal. *(Bench 576: max_err 0.0347 < 2×h(K)=0.10, coverage 1.000.)*

## Phase 2 — Band gate + triage (`hint_regret/gate.rs`)

- [x] `learnable_band_gate(w, w_lo, w_hi, kappa) -> f32` — `σ(κ(w−w_lo))·σ(κ(w_hi−w))`; monotone ↑↓, strict (0,1), peaks at band center. Property tests + one-line Lean extension of the shipped `sigmoid_bounded` family (`.proofs/KatgptProof`). **`w` = per-composition Beta posterior mean** (Research 500 row 11) — the gate ranks modifier compositions, and when `w` leaves the band the consumer swaps composition, not a scalar. *(Lean: `KatgptProof.HintRegret.bandGate_mem_Ioo` + SpecTests + 2 negative perturbations, 39-theorem audit green. The strict-(0,1) claim is the ideal ℝ contract; the f32 gate's two saturation surfaces — ±40 early-exit → 0, rounding → 1 at args ≳17 — are documented in the Rust doc + pinned by the property test.)*
- [x] `Regime` enum { Frontier, Mastered, Intractable } + `triage(r_hat, r_floor, unhinted_return, tau_r, tau_R) -> Regime` — 2-threshold 2D partition; property test: exactly one cell per input, boundaries pinned. *(Deviation, documented in gate.rs: the plan draft carried a fifth `r_floor` arg the guide's partition (and the landed consumer) does not use — shipped the 4-arg canonical Guide 340 form `triage(r_hat, unhinted_return, tau_r, tau_ret)`; noise-floor handling lives in the estimator's sequential stopping.)*
- [x] Wilson CI on the learnable-share statistic (UQ honesty on the signature metric). *(Byte-identical to the qmc twin; free fn `wilson_score_ci`.)*

## Phase 3 — Frontier ordering + memory seam

- [x] `beta_lcb_order(scores: &[(successes, fails)]) -> Vec<usize>` — ε-quantile of Beta(1+S, 1+F) descending (mirror of shipped `SelectionMode::BetaPosterior`; extract or re-implement leaf-clean — check katgpt-core first, DRY). **Expose the per-entry quantile (`beta_lcb`) so consumers can order BOTH directions** — descending for frontier ordering, ascending for weakness-slice diagnosis (Research 500 row 4: weakest slice = lowest LCB; Guide 340 diagnosis-first loop). One primitive, two consumers. *(Landed as `beta_lcb_order_into` — DRY over `best_belief_score`'s Beta-quantile, zero-alloc scratch form; per-entry `beta_lcb` exposed.)*
- [x] `RegretMemoryEntry { content_hash, r_hat, ci, skill_tag_bits, last_seen_tick }` + salience `r_hat · σ(−λ·Δt)` (staleness decay — same family as `decay_confidence`).
- [x] Oldest-first eviction at capacity; absorbing-eviction for Intractable-classified entries. *(Bounded tombstone ring.)*

## Phase 4 — GOAT gate + bench

- [x] `benches/bench_576_hint_regret_goat.rs`:
  - G1: oracle calibration (Phase 1) + triage partition properties.
  - G2: CRN variance ratio ≥ 2× vs independent-seed estimation (1000 reps); per-pair cost sub-µs alloc-free. *(2.76× / 24.7 ns — fixture calibration + noise-stream fix documented in Bench 576 §Honest findings.)*
  - G3: default feature set untouched (gate compiles out); cgsp suite count-identical with `hint_regret` off. *(Default lib 1904 = clean-HEAD 1904, worktree-verified.)*
  - G4: counting-allocator zero steady-state allocs over 10⁴ pairs.
  - G-Floor (Report the Floor): triage accuracy vs single-arm banding floor at matched budget — paired arm must win or FAIL. *(0.997 vs 0.693.)*
  - G8 (simulated): learnable-share rises under regret-gated selection vs uniform on a synthetic curriculum (the 0.16→0.31 signature, modelless). *(1.000 vs 0.273, 8 seeds.)*
- [x] `.benchmarks/576_hint_regret_goat.md` record. *(Landed as `tests/` integration binaries, not `benches/` — gate tests, not criterion benches; same convention as the sibling GOAT gates.)*
- [x] Promotion decision: default-on only if G-Floor and G8 pass modellessly; else stays opt-in with the verdict recorded. *(**Stays opt-in** — G-Floor + G8 pass modellessly (criterion met, necessary-not-sufficient); promotion deferred per the no-default-consumer rule until the Phase 5 consumer migrates onto `katgpt_core::hint_regret` — the consumer migration LANDED 2026-08-24 (`bae567c`, see Phase 5), so the default-on flip is now unblocked and remains the owner's call. Verdict in Bench 576 §Promotion verdict.)*

## Phase 5 — Consumer wiring (defer-marked, owner-gated)

**Owner authorization 2026-09-11:** the consumer migration is AUTHORIZED (DRY — removes duplicate substrate; GOAT G-Floor 0.997 vs 0.693, G8 1.000; consumer exists: riir-mmorpg-examples `demo_coverage_curiosity`). Evidence note: the core migration itself already LANDED 2026-08-24 (row 1 below, `bae567c`) — this verdict ratifies it; the remaining rows (modifier-composition variant, CGSP conflation evaluation, edge_lora) and any further implementation stay DEFERRED.

- [-] riir-ai quest-center frontier weighting — **CORE LANDED 2026-08-24** (riir-mmorpg-examples `2c17f08`, behind the existing opt-in `demo_coverage_curiosity`: the regime triage is the hunt scorer's PRIMARY term, lexicographic over counter quality — Issue 677's DEFEND policy shipped; 8 new tests, all suites green). **CONSUMER MIGRATED ONTO THE PRIMITIVE 2026-08-24** (riir-mmorpg-examples `bae567c`): the local `FrontierRegime` collapse deleted; `frontier_regime_of` now delegates the partition to `katgpt_core::hint_regret::triage` (the domain probe stays, per the primitive's caller-owns-rollouts contract); outputs bit-identical — the exhaustive partition test (32 masks × 7 ranks, inline-re-derived expectation) green, feature-on lib 532/0 = pre-migration baseline, all 5 CI lib pins exact. `demo_coverage_curiosity` now enables `katgpt-core/hint_regret` (dep moved to the cross-target section — already in the wasm32 graph transitively). **The Setup/Rule/Link modifier-composition variant over the frozen `QuestTemplateRow` table remains deferred** — it was never PoC-validated (Issue 677's arms were regret/uniform/aggregate; the Research 500 wrap arm + transfer-back evaluation did NOT run); the verifier BLAKE3-pinned invariant across wraps stays its gate condition. The landed core needs no verifier concern: pure ordering, content never transformed.
- [-] CGSP conflation fix evaluation: whether `(1−solve_rate)·guide_score` gains an intractable-separation term behind the feature (behavior change to shipped substrate — needs its own gate run). *(Partially answered by the P2 landing: the quest-center seam now preferentially offers frontier + sinks intractable content at the CONSUMER level — the aggregate arm's measured 24 wasted intractable encounters/seed is the behavior the core wiring removes — but the shipped CGSP reward itself is unchanged; the substrate-side term still needs its own gate run.)*
- [-] edge_lora arena opponent selection (riir-train Plan 346 dependency).

## Non-goals

- The wrapper algebra itself (Setup/Rule/Link combinators, verifier-hash contract, EnvRigger-style diagnosis loop) — game-side (Guide 340 §"Wrapper composition", riir-ai); promote to an open primitive only if PoC 677 vindicates generality (Research 500 P3).
- Training a neural environment designer (SPADE proper) — that is riir-train Plan 346's POC-only item.
- Executable-environment generation (code-as-env) — our designer stays the parameterized quest-grammar drafter; the invisible-leash boundary is documented in Research 496 §Honest caveats.
