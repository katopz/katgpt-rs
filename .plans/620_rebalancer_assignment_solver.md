# Plan 620: katgpt-assign — Spec-Driven Constrained Assignment Solver (Rebalancer distillation)

> **Status:** Active — Phase 1 LANDED 2026-10-07 (`feat:` + `docs:` commits; G1/G2/G4 green, feature OPT-IN per the verdict protocol; Phase 2/3 pending consumers)
> **Claimed:** katgpt-rs-idle-620, 1792246201, expires 2026-10-19T14:10:01Z — Phase 2 (specs/moves/explain/policy)
> **Source:** Research 607 (OSDI'24 Rebalancer, facebook/rebalancer @ e4c35517980d893849275f71b3cb90f81164b6d5)
> **Consumers:** riir-chain Issue 164 (primary — shard_assignment solver); riir-rethink Issue 029 (candidate; closed 2026-10-07 — record: riir-rethink HISTORY.md, the promote triggers folded there)

## Phase 1 — the crate + core solver (this plan's whole scope)

New leaf member crate **`katgpt-assign`** — std-only, alloc allowed, **zero external deps, zero workspace deps** (sits upstream of `katgpt-core`; core takes it as an optional path dep and re-exports `katgpt_core::assign` behind the opt-in `assignment` feature — the `katgpt-dec` → `katgpt_core::dec` precedent). Run the `boundary-guard` skill BEFORE the crate lands. Integer determinism is a G1 requirement from the first commit — the API never carries float accumulation, so it can't be retrofitted.

- [x] Run `boundary-guard` (allowlist + domain read for a new member crate; confirm zero-dep posture) — domain read as written (modelless, zero deps, upstream); Owns bullet added to BOUNDARY.md (the katgpt-device-verify widening precedent); scoped contract run post-landing
- [x] Scaffold `crates/katgpt-assign` (workspace member; `types.rs` for decoupled structs per repo convention)
- [x] Types: `Problem` (objects/containers as dense `u32` indices; `Dimension` values `i64`; flat scopes; `Group` reserved for Phase 2), `Assignment` (dense `Vec<u32>` object→container), `Seed` (u64)
- [x] Expression DAG: `Lookup` leaves (shared per (dimension, container)) + `Sum` + `Max` internal nodes; per-node cached `i64` value; leaf-affectance maps `M_o` (object→leaves), `M_b` (container→leaves); evaluate-const / apply-mutate split — plus `Affine{k,c}` and `LeafMoved` (the minimal faithful encoding of MinimizeMovement); parents CSR for upward delta propagation
- [x] Specs (constraint-or-goal duality; broken constraint → fix-it goal + never-worse guard, paper defaults 100/10000): `Capacity` (scope-item util ≤ limit) · `Balance` (minimize util spread across scope items) · `MinimizeMovement` (penalize moved objects vs initial) — DEVIATIONS RECORDED: (a) a secondary sum-of-rows repair term rides the goal side (identically 0 at feasibility; measured — the Max fold alone stalls on tied-at-worst plateaus, perfect_balance froze at violation 10400); (b) sideways (δ=0) moves are DEFAULT-ON with a 25k budget + heat>0-source + immediate-reversal guards — the plan's `obj_δ ≤ 0` window is load-bearing for Max-fold ties
- [x] Fold all per-container constraint rows into one root via `Max` (linear graph size)
- [x] Local search: strict improvement (`obj_δ ≤ 0` signed integer folding violation + objective delta), **hot-container ordering** from node potentials (leaf-side heat approximation — violation mass + balance deviation; the paper's measured win kept), move types `Single` + `Swap`, deterministic seeded tie-breaks, time + move limits (time advisory-only: breaks cross-machine byte-determinism, documented)
- [x] Determinism test (G1): same input + same seed ⇒ byte-identical `Assignment` across runs AND across `Vec` iteration-order perturbations (spec-order rotation); no HashMap probe-order dependence anywhere in move generation
- [x] G1 correctness fixtures: feasibility under Capacity; objective ≤ initial on every fixture; improvement-vs-greedy-init sanity; constraint-violation fallback behavior — PLUS the delta-evaluation property test (incremental root == full recompute, the exactness claim replacing Meta's segment tree)
- [x] G2 outside-baseline bench (`--release`, box-state provenance line per the latency-claim rule):
  - [x] vs greedy / first-fit-decreasing on the same instances (quality + wall time) — workspace precedent `riir-rag/src/packer.rs` — **local dominates FFD on every fixture (3–47× better folded objective; exact optima on both known-optimum families)**
  - [x] optimality gap vs brute force, n ≤ 12 (exhaustive) — tiny-instance gap 0 (one-fixture tolerance documented in-test)
  - [x] mid-size lower-bound gap, n ≈ 100–1000 (capacity-relaxation bound, or Hungarian on the pure-assignment subset) — pigeonhole LB on max-util: solver sits on/above the bound (sanity) and ≤ FFD's max-util; gap-to-bound reported in Bench 924 (bound not tight for this family — reported, not asserted)
  - [x] NO-GO clause: NOT TRIGGERED (greedy loses 3–47× at target sizes) — primitive stays opt-in pending the consumer per the verdict protocol; delta-vs-full-recompute recorded as diagnostic (delta-property test pins bit-identity)
- [x] G4: hot loop alloc-free under the repo's counting allocator (`debug_assertions`-gated per Issue 856 — `any(debug_assertions, feature = "alloc_tracking")`, the Issue-741 profile-free posture)
- [x] `cargo clippy` clean (healer-first for mechanical findings — `cargo refine` applied 3, manual 4); feature-gated re-export compiles at BOTH postures (`assignment` on/off verified via cargo check + cargo tree)
- [x] README block in the crate + one line in the workspace members list (+ README crate-count/flag-count sites, count_features gate green)
- [x] Bench-target disposition (the `suite_membership_audit` open note): `bench_924_assignment_g2_baselines` (`harness = false`) is named by no suite — CORRECT BY DESIGN, no suite row added. The audit's scope is TEST targets only; benches are the Issue-834 standing skip class (one-time measurement records — Bench 924 owns the numbers; a re-gate re-runs the bench by name in its command line).

## Phase 2 — only after G1/G2 pass vs outside baselines

- [ ] Specs: `GroupCount` (spread), `AvoidMoving`, `AssignmentAffinities`
- [ ] Moves: triple-loop, KL-search; sampled/greedy variants
- [ ] Equivalence-class collapsing (paper Alg 2)
- [ ] Parallel candidate evaluation (rayon is core-external — keep katgpt-assign dep-free; thread via core's existing pattern or leave single-threaded)
- [ ] `explain` surface: why-was-this-move-rejected + assignment diff (the Explorer role)
- [ ] Constraint-policy variants beyond DEFAULT

## Phase 3 — consumer wiring (each is its own consumer-repo issue, not this plan)

- [x] riir-chain Issue 164: shard→node assignment for `shard_assignment.rs` (solver-run location choice recorded there: leader-computes-commits-raw leaning; byte-equality test REQUIRED before any consensus-replayable use) — **CLOSED 2026-10-07, all four items**: leader-computes + commit-raw (owner call on Claude verdict) + follower feasibility check + audit header; opt-in `shard_solver` wiring `992cbf0b` over main `6a9d4d5a`; byte-identity + drained-node rebalance tests shipped
- [ ] Game runtime follow-ups per Research 607 §3 (spawn placement, migration balancing) — respect brain+FSM law
- [ ] riir-rethink Issue 029 candidate tracker: revisit only if escalate rows return to serve

## GOAT gate verdict protocol

Feature stays `assignment` opt-in. Promotion to default requires: G1+G2 pass vs outside baselines AND a real consumer wired (riir-chain Issue 164 minimum). Otherwise demote/keep opt-in and record the verdict here. NO-GO outcome ⇒ NEGATIVE record in Research 607 + this plan closed.

**Phase 1 verdict (2026-10-07): G1 + G2 + G4 PASS, NO-GO not triggered — feature STAYS OPT-IN pending the consumer.** G1: byte-identical determinism across runs + spec-order perturbations; delta evaluation bit-identical to full recompute (1,200-move property test); feasibility + never-worse + infeasible-fallback fixtures green. G2 (Bench 924): local dominates FFD 3–47× on the random family, exact optima on both known-optimum families, pigeonhole-LB sanity green, tiny-instance brute-force gap 0 — the NO-GO clause (greedy ≈ local search) is refuted at every target size. G4: 0 allocations in `solve()` verified against optimised code (`--release --features alloc_tracking`). Consumer leg: riir-chain Issue 164 acceptance items 1–2 closed (Phase 1 + the cluster-shape bench — spread 3671→3 at 21% shard relocation); items 3–4 open (owner call + the katgpt-core main-promote prerequisite recorded there). Promotion to default re-arms when that issue's wiring lands.

**Promotion re-arm adjudicated (2026-10-07, same day): stays OPT-IN.** The consumer landed — riir-chain Issue 164 CLOSED (all four items): the develop→main promote went in (main `082357375`→`6a9d4d5ae`, clean FF, owner-directed), the lock bumped, and the wiring shipped as riir-chain's opt-in `shard_solver` (`992cbf0b`: propose/verify-feasible/audit-header, byte-identity + follower-check tests, `+shard_solver` gate row floor 405). But the promotion criterion's spirit is a DEFAULT consumer: riir-chain's `shard_solver` is opt-in there (per its own repo's discipline), so promoting `assignment` to katgpt-core DEFAULT would add the crate's compile cost to every downstream build with zero default-path gain — the no-default-consumer rule. Re-arm again when a consumer promotes `shard_solver` (or equivalent) to its own default.
