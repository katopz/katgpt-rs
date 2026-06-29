# Issue 012 — Cross-repo Lean 4 FV rollout coordinator

> **Status:** 🟡 OPEN — coordination/tracking task across the 5-repo quintet
> **Type:** Formal verification (Lean 4) — cross-repo strategy
> **Origin:** Discussion following `katgpt-rs/.proofs/KatgptProof` (Plan 293)
> audit (2026-06-29). Question: "we proved katgpt-rs which is prod — should we
> prove riir-* which is also prod?" Answer: **yes, with priorities.**
> **Blocks:** 4 sibling issues. **Blocked by:** Nothing.
> **Priority:** P0 (coordination) — the sibling P0 theorems unblock on this
> issue's conventions being agreed.
> **Cross-repo siblings:** `riir-neuron-db/.issues/004_*` (P0),
> `riir-chain/.issues/001_*` (P0), `riir-ai/.issues/348_*` (P1),
> `riir-train/.issues/308_*` (EXCLUDED).

---

## 1. The thesis

`katgpt-rs/.proofs/KatgptProof` proves a public primitive (sigmoid ranking
preservation). That's 1 of 5 repos. The other 4 are also production code,
and three of them carry invariant-shaped properties currently enforced only
by empirical tests — the same shape as past bugs (`merkle_root`, `can_freeze`,
AC-Prefix G1).

**A Lean theorem is the ultimate modelless correctness guarantee: zero
runtime cost, forever-verified, refactor-immune.** It's strategically
aligned with:
- The modelless mandate (`katgpt-rs/AGENTS.md`) — proofs cost nothing at
  runtime.
- The sync-boundary rule (global `AGENTS.md`) — "must be deterministic" is a
  theorem, not an aspiration.
- The lessons-learned bug class — every past bug was an invariant violation
  we asserted but didn't prove.

## 2. Current state (2026-06-29)

| Repo | `.proofs/` exists? | Theorems shipped | Sibling issue |
|---|---|---|---|
| `katgpt-rs` (public) | ✅ `KatgptProof` (Plan 293) | `action_bridge_ranking_preserved`, `action_bridge_argmax_preserved` | this issue (coordinator) |
| `riir-chain` (private) | ✅ `RiirChainProof` (Plan 004) | LatCal fixed-point round-trip | `riir-chain/.issues/001_*` (extend) |
| `riir-neuron-db` (private) | ❌ none | — | `riir-neuron-db/.issues/004_*` (P0, **start here**) |
| `riir-ai` (private) | ❌ none | — | `riir-ai/.issues/348_*` (P1) |
| `riir-train` (private) | ❌ none | — | `riir-train/.issues/308_*` (**EXCLUDED**) |

## 3. Recommended sequencing

```
Phase 1 (P0): riir-neuron-db/.proofs/      ← START HERE
              ├─ shard_layout_determinism.lean     (merkle_root lesson)
              └─ can_freeze_consistency.lean       (Plan 002 lesson)
              Rationale: highest ROI (two bug-shaped invariants), most
              tractable (pure layout/algebra), leaf crate (no chain dep,
              clean Mathlib-free Lean possible).

Phase 2 (P0): riir-chain/.proofs/          ← extend existing RiirChainProof
              ├─ quorum_commit_determinism.lean
              └─ shard_merkle_root_init.lean  (coordinate with Phase 1)
              Rationale: sync-boundary criticality; builds on LatCal lemma.

Phase 3 (P1): riir-neuron-db + riir-chain fill-ins
              ├─ Merkle proof soundness (neuron-db)
              ├─ Split-key security (chain)
              └─ Slashing monotonicity (chain)

Phase 4 (P1): riir-ai/.proofs/             ← new instance
              ├─ hla_scalar_boundedness.lean   (cheap, extends KatgptProof)
              └─ freeze_thaw_reader_invariant.lean  (hard — memory model)

Phase 5 (P2/P3): riir-ai extensions
              └─ bridge_ordering_learned_directions.lean
```

`riir-train` is **excluded** (`riir-train/.issues/308_*`) — training
properties are probabilistic/behavioral, Lean is the wrong tool.

## 4. Cross-repo conventions to lock in BEFORE Phase 1 starts

These must be agreed once and applied uniformly:

- [ ] **C1 Toolchain pin policy.** Each `.proofs/` pins its own
      `lean-toolchain`. `RiirChainProof` uses `v4.31.0` (Mathlib-free, `omega`).
      `KatgptProof` uses `v4.32.0-rc1` (Mathlib required for transcendental
      analysis). Rule: pin the lowest version that compiles the theorem.
      Don't force Mathlib where `omega`/`ring` suffice.
- [ ] **C2 Axiom policy.** Target axioms = `{propext, Classical.choice,
      Quot.sound}` only (Mathlib's standard foundation). No `sorry`, ever.
      Verified by `#print axioms` in CI.
- [ ] **C3 Spec-match test convention.** Every Lean theorem has a paired
      Rust spec-match test (pattern: `katgpt-rs/tests/bridge_spec_match.rs`).
      Lean proves the math; Rust test catches spec drift. Two-way gate, both
      must pass for the proof to be valid.
- [ ] **C4 Private proofs stay private.** Lean files in `riir-*/.proofs/`
      are internal-only. The open/private FV split mirrors the open/private
      code split (Research 003 §322-325): `katgpt-rs/.proofs/` proves generic
      math; `riir-*/.proofs/` proves the HOW — fine because the repo is
      private. **Do not cross-port private proofs into the public repo, even
      as "reference".**
- [ ] **C5 Build isolation.** `lake build` artifacts (`.lake/`) must not
      pollute Cargo `target/`. Add `.lake/` to each repo's `.gitignore`.
      CI script invokes `lake build` separately from `cargo test`.
- [ ] **C6 README discipline.** Each `.proofs/README.md` documents: theorem
      list, axiom inventory, Mathlib-dependency rationale, spec-match test
      path, regeneration protocol (what to do when the Rust side changes).

## 5. Tasks (coordinator-level)

- [ ] **T1** Confirm C1-C6 conventions with the team (this issue's §4).
- [ ] **T2** Track Phase 1 (`riir-neuron-db/.issues/004_*`) to P0 theorem
      completion. This is the rollout's first concrete deliverable.
- [ ] **T3** Track Phase 2 (`riir-chain/.issues/001_*`) — coordinate the
      shared `merkle_root` audit between `riir-neuron-db` (shard constructors)
      and `riir-chain` (chain-side shard wrappers). Same bug class, two repos,
      must be consistent.
- [ ] **T4** Update Research 003 §167 ("9 GOAT proofs") to reference the FV
      rollout — the public capability claim should cite the actual theorems
      once they exist, not just empirical gates.
- [ ] **T5** Once Phase 1 ships, write a `.research/` note in `katgpt-rs`
      distilling the cross-repo FV pattern (open primitive + private guides +
      spec-match tests) as a reusable Super-GOAT capture protocol. This is
      process IP worth capturing.

## 6. Tractability summary (honest cost forecast)

| Repo | Hardest theorem | Effort estimate | Risk |
|---|---|---|---|
| riir-neuron-db | shard layout consistency | 1-2 days | Low (algebra) |
| riir-chain | quorum commit determinism | 3-5 days | Medium (needs LatCal lemma composition) |
| riir-ai | freeze/thaw reader invariant | 1-2 weeks | **High** (memory model — see `riir-ai/.issues/348_*` §5) |
| riir-train | (excluded) | — | — |

The riir-ai freeze/thaw theorem is the long pole. Plan accordingly: ship
Phases 1-3 first (high-confidence wins), then attempt Phase 4 with the
stronger-SC + stress-test-fallback approach (`riir-ai/.issues/348_*` §5
option C).

## 7. Cross-references

- Existing public instance: `katgpt-rs/.proofs/KatgptProof` (Plan 293)
- Existing private instance: `riir-chain/.proofs/RiirChainProof` (Plan 004)
- Strategy doc: `riir-ai/.research/003_Commercial_Open_Source_Strategy_Verdict.md`
- Sibling issues: `riir-neuron-db/.issues/004_*`, `riir-chain/.issues/001_*`,
  `riir-ai/.issues/348_*`, `riir-train/.issues/308_*`
- Past bugs being prevented: `merkle_root` (riir-neuron-db AGENTS.md),
  `can_freeze` (Plan 002 Phase 5), AC-Prefix G1 (Plan 313)

## TL;DR

`katgpt-rs/.proofs/` proved a public primitive. The other 4 production repos
deserve the same treatment — **except `riir-train`** (excluded: training is
probabilistic, not invariant-shaped). Priority order: `riir-neuron-db` (P0,
start here — two bug-shaped invariants, most tractable) → `riir-chain`
extend (P0) → fill-ins (P1) → `riir-ai` (P1, freeze/thaw is the hard long
pole). Lock C1-C6 conventions before Phase 1 starts. This issue coordinates
the rollout; sibling issues own each repo's concrete theorems.
