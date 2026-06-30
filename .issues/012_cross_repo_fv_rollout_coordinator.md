# Issue 012 — Cross-repo Lean 4 FV rollout coordinator

> **Status:** ✅ READY TO CLOSE — all phases resolved (1–4 COMPLETE, Phase 5
> CLOSED as "covered by specialization"). All coordination tasks T1–T5 DONE.
> 79 Lean 4 theorems across 4 instances. The cross-repo FV rollout is
> complete; this issue can be closed.
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

## 2. Current state (2026-06-30)

| Repo | `.proofs/` exists? | Theorems shipped | Sibling issue |
|---|---|---|---|
| `katgpt-rs` (public) | ✅ `KatgptProof` (Plan 293) | `action_bridge_ranking_preserved` (+ `'` variant), `action_bridge_argmax_preserved` (3 thms) | this issue (coordinator) |
| `riir-chain` (private) | ✅ `RiirChainProof` (Plans 004 + 008 + 009) | LatCal round-trip (3), quorum determinism (5) + tier/block root funcs (2), chain-side `merkle_root` (4), **slashing monotonicity (8)**, **split-key security (10)** — 32 thms total | `riir-chain/.issues/001_*` (**CLOSED** — T1–T9 all done) |
| `riir-neuron-db` (private) | ✅ `NeuronDbProof` (Plans 007 + 008) | `Shard/Layout` (16 thms), `Consolidation/FreezeGate` (8 thms), `Merkle/Soundness` (4 thms) — 28 thms total | `riir-neuron-db/.issues/004_*` (Phase 1 + P1 done; P2 pending) |
| `riir-ai` (private) | ✅ `RiirAiProof` (Plan 353 + Issue 348 T2) | HLA boundedness (14 thms: sigmoid open-interval `(0,1)` + clamp closed-interval `[0,1]` + composite `curiosity_drive_bounded`) + freeze/thaw reader invariant (2 thms: `read_snapshot_consistent` MAIN + `read_snapshot_no_torn_ab_pair`) — 16 thms total | `riir-ai/.issues/348_*` (**T1–T8 all done** — Phase 4 COMPLETE) |
| `riir-train` (private) | ❌ none | — | `riir-train/.issues/308_*` (**EXCLUDED**) |

### Phase progress

- ✅ **Phase 1 (P0): riir-neuron-db** — DONE (Plan 007, commit `179b336`).
  Shipped `Shard/Layout.lean` (16 axiom-free thms: gap-free layout, monotone
  offsets, every constructor sets `merkle_root` to `zeros32`) and
  `Consolidation/FreezeGate.lean` (8 thms: `can_freeze = input ∧ output`,
  implications, sufficiency, `WellFormed` contract).
- ✅ **Phase 2 (P0): riir-chain** — DONE (Plan 008). Shipped
  `Consensus/QuorumDeterminism.lean` (4 axiom-free thms: `compute_congr`,
  `compute_refl`, `compute_congr_tier_roots`, `compute_congr_block_root` —
  the sync-boundary determinism rule as a theorem) and
  `Shard/MerkleRoot.lean` (4 thms: `commit_stamp_all` axiom-free, plus
  length-preservation and per-element corollaries). Coordinated with Phase 1:
  constructors init `merkle_root = zeros32`, commit functions override to
  batch root — together they close the bug class across both repos.
- 🟡 **Phase 3 (P1): neuron-db + chain fill-ins** — Merkle proof soundness,
  split-key security, slashing monotonicity.
  - ✅ **neuron-db Merkle soundness** — DONE (Plan 008, 2026-06-30). Shipped
    `Merkle/Soundness.lean` (4 axiom-free thms: `computeRootFromProof_empty`,
    `computeRootFromProof_injective_in_leaf` — the main tamper-evidence theorem,
    parameterized over any injective `hashFn` so the cryptographic assumption is
    a *hypothesis* not an axiom; `verifyProof_tamper_evident` — applied form;
    `computeRootFromProof_deterministic`). Spec-match test
    `tests/merkle_soundness_spec_match.rs` (7/7, incl. 1000-trial tamper stress
    + 16K single-byte tampers) validates BLAKE3 satisfies the `h_inj`
    hypothesis empirically.
  - ✅ **chain split-key security + slashing monotonicity** — DONE (Plan 009,
    2026-06-30). Shipped `Economics/SlashingMonotone.lean` (8 axiom-free thms:
    `slashed_is_absorbing` — once slashed, no sequence of slashes can un-slash;
    `slash_evidence_first_writer_wins` — idempotency for the Plan 212 penalty
    tracker; + 6 helpers/corollaries) and `Crypto/SplitKey.lean` (10 thms:
    `wire_safe_only_commitment` — only `txCommitment` is wire-safe;
    `combine_not_const_in_alpha` / `combine_not_const_in_beta` — under
    `CombineNonDegenerate`, `combine` depends on each argument). Parameterized
    over `combine` (Approach B, matching neuron-db Plan 008) so the crypto
    assumption is a hypothesis. Spec-match tests: 8/8 slashing (incl. 1000-trial
    × 16-reslash stress) + 7/7 split-key (incl. 10K-input BLAKE3
    non-degeneracy corpus). Closes `riir-chain/.issues/001_*` (T1–T9 all ✅).

  **Phase 3 COMPLETE.**
- ✅ **Phase 4 (P1): riir-ai** — `hla_scalar_boundedness` + freeze/thaw
  reader invariant. **COMPLETE (2026-06-30).**
  - ✅ **HLA scalar boundedness** — DONE (Plan 353, 2026-06-30). Bootstrapped
    `riir-ai/.proofs/RiirAiProof` (fourth FV instance, Mathlib-required,
    toolchain v4.32.0-rc1). Shipped `Hla/Basic.lean` (spec: `dot`, `clamp01`,
    `NpcEmotionScalars`, `clamped`, `curiosity_drive`) and `Hla/Bounded.lean`
    (14 theorems across 2 classes: Class A — sigmoid open-interval `(0,1)`, 4
    thms, extends `KatgptProof` from ranking to boundedness; Class B — clamp
    closed-interval `[0,1]`, 9 thms, the actual sync invariant + composite
    `curiosity_drive_bounded`). All within `{propext, Classical.choice, Quot.sound}`
    axiom budget. Spec-match test 6/6 green (incl. f32 saturation caveat +
    NaN/Inf edge cases).
  - ✅ **Freeze/thaw reader invariant** — DONE (Issue 348 T2, 2026-06-30). The
    path was the canonical FV-as-bug-finder story: scoping T2 revealed the
    `LoRAWeightVersion` code violated the invariant (Issue 354 — torn-read
    hazard from three independent atomics). The Issue 354 fix (single-
    `ArcSwap<Inner>` packing) made the invariant hold by construction. The
    Lean proof then reduced to a definitional unfold. Shipped
    `Runtime/Basic.lean` (spec) + `Runtime/FreezeThaw.lean` (2 theorems:
    `read_snapshot_consistent` MAIN + `read_snapshot_no_torn_ab_pair`
    corollary). **Notably: the T2 theorems depend ONLY on `[propext]`** — not
    even Classical.choice or Quot.sound, because the SC atomicity is
    structural in the single-field `ArcSwap` model. The
    `arcswap_store_atomicity` axiom is documentation-only. Spec-match stress
    test `concurrent_lora_no_torn_read` (100K iterations, hard-fails on any
    torn read) shipped with the Issue 354 fix.
- ✅ **Phase 5 (P2/P3): riir-ai** — bridge ordering over learned directions.
  **CLOSED (2026-06-30) — covered by specialization.** The public
  `action_bridge_ranking_preserved` theorem is fully parameterized over
  arbitrary direction vectors (`q d₁ d₂ : ι → ℝ`, universally quantified),
  so riir-ai's learned/tuned direction vectors are covered by instantiation.
  Shipped `.proofs/RiirAiProof/Phase5Specialization.lean` (documentation-only
  Lean module, no new theorems) as the formal record — it explains why a
  one-line wrapper theorem would add no mathematical content and enumerates
  the three distinct direction-shaped objects in riir-ai (canonical HLA basis,
  committed-blend archetype fields as operators, freeze/thaw-swapped trained
  tables). Commit `adcc3bee`. `lake build` passes (2237 jobs); axiom inventory
  unchanged.

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
              ├─ Merkle proof soundness (neuron-db)        ← neuron-db half DONE (Plan 008)
              ├─ Split-key security (chain)                ← open
              └─ Slashing monotonicity (chain)             ← open

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

- [x] **T1** Confirm C1-C6 conventions with the team (this issue's §4).
      ✅ DONE (2026-06-30). Applied empirically in Phases 1–4 and formally
      distilled in `katgpt-rs/.research/351_*` (T5). All 4 FV instances
      (KatgptProof, RiirChainProof, NeuronDbProof, RiirAiProof) follow C1–C6
      with zero exceptions across 79 theorems. The conventions are
      load-bearing, not aspirational — the rollout never needed an escape
      hatch. Per-toolchain pins (C1: 3/4 on v4.31.0, riir-ai on v4.32.0-rc1
      for Mathlib), axiom budget (C2: all 79 within {propext, Classical.choice,
      Quot.sound}, with the freeze/thaw thms under-shooting to propext-only),
      spec-match tests (C3: every theorem has a paired Rust test), private
      proofs stay private (C4), build isolation (C5: .lake/ gitignored, lake
      build separate CI layer), README discipline (C6: all 4 instances have
      regeneration protocols).
- [x] **T2** Track Phase 1 (`riir-neuron-db/.issues/004_*`) to P0 theorem
      completion. This is the rollout's first concrete deliverable.
      ✅ DONE (Plan 007, commit `179b336`).
- [x] **T3** Track Phase 2 (`riir-chain/.issues/001_*`) — coordinate the
      shared `merkle_root` audit between `riir-neuron-db` (shard constructors)
      and `riir-chain` (chain-side shard wrappers). Same bug class, two repos,
      must be consistent.
      ✅ DONE (Plan 008). The two-repo audit is closed: leaf-crate Plan 007
      proves constructors init `merkle_root = zeros32`; chain Plan 008 proves
      commit functions override to batch root. Coordinated spec-match test
      (`shard_merkle_root_spec_match.rs::spec_commit_overrides_constructor_default`)
      cross-references both invariants.
- [x] **T4** Update Research 003 §167 ("9 GOAT proofs") to reference the FV
      rollout — the public capability claim should cite the actual theorems
      once they exist, not just empirical gates.
      ✅ DONE (2026-06-30, commit `f30daf00` in riir-ai). Updated
      `riir-ai/.research/003_Commercial_Open_Source_Strategy_Verdict.md`
      (the internal full strategy doc; the public katgpt-rs `.research/003_*`
      is the trimmed routing-rules subset and needs no FV citations).
      Changes: (a) riir-ai capability section — freeze/thaw "readers never
      see a torn snapshot" now cites `read_snapshot_consistent` (2 thms,
      propext-only); sigmoid-gated scalar boundedness now cites the 14
      Hla/Bounded.lean theorems. (b) riir-chain capability — replaced the
      outdated "9 GOAT proofs" line with the 32 Lean theorems (LatCal,
      quorum determinism, merkle_root, slashing, split-key). (c) riir-neuron-db
      capability — layout consistency (16 thms), tamper-evidence (4 thms),
      freeze gate contract (8 thms) all cited. (d) All three "Why Hard to
      Replicate" moat sections updated to cite theorems as moat (correct
      implementation *defined* by proof, not just tested). (e) New dedicated
      section "Formal Verification — the Modelless Correctness Guarantee"
      with full theorem-count table, axiom discipline notes, spec-match
      pattern, and the Issue 354 bug-finding case study. (f) Related table
      updated to cite coordinator (012) + T5 note (351).
- [x] **T5** Once Phase 1 ships, write a `.research/` note in `katgpt-rs`
      distilling the cross-repo FV pattern (open primitive + private guides +
      spec-match tests) as a reusable Super-GOAT capture protocol. This is
      process IP worth capturing.
      ✅ DONE (2026-06-30, commit `fc0c0b80` in katgpt-rs). Shipped
      `katgpt-rs/.research/351_cross_repo_lean4_fv_pattern.md` (184 lines).
      Covers: (a) the four instances (3 + 32 + 28 + 16 = 79 theorems, full
      breakdown); (b) the C1–C6 conventions, empirically validated across
      all 4 instances; (c) the spec-match test pattern (3 concrete examples:
      what Lean catches vs what Rust catches, neither sufficient alone);
      (d) the bug-finding payoff — the Issue 354 torn-read case study; (e)
      the "fix the code, not the proof" move (collapsing the 1–2 week long
      pole into a definitional unfold); (f) when to apply vs not apply the
      pattern (static + invariant-shaped vs probabilistic/behavioral);
      (g) a 9-step reusable scaffold for the next FV effort; (h) open
      questions (Phase 5 redundancy, UQ floor gate, cross-instance lemma
      sharing, richer memory models). Classification: Public (process note).
      This is the process-IP counterpart to T4's capability-face update.

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

**Rollout status (2026-06-30):** ✅ **ALL PHASES COMPLETE.** Four FV
instances shipped (KatgptProof, RiirChainProof, NeuronDbProof, RiirAiProof)
with **79 theorems total** across the quintet (3 + 32 + 28 + 16). Phase 4
(riir-ai) shipped both the HLA boundedness (Plan 353, 14 thms) AND the
freeze/thaw reader invariant (T2, 2 thms depending only on `propext`) — the
long pole collapsed after the Issue 354 fix made the invariant hold by
construction. The freeze/thaw proof effort also surfaced a real torn-read
bug (Issue 354), validating FV as a bug-finding tool. **All coordination
tasks T1–T5 are DONE** — C1–C6 conventions empirically validated (T1), all
phases tracked (T2/T3), the capability doc cites all 4 instances (T4,
commit `f30daf00`), and the cross-repo FV pattern is distilled as a reusable
Super-GOAT capture protocol (T5, `katgpt-rs/.research/351_*`, commit
`fc0c0b80`). **Phase 5 (bridge ordering) CLOSED** as "covered by
specialization" — the public theorem is fully parameterized over arbitrary
direction vectors, so riir-ai's learned directions are covered by
instantiation (commit `adcc3bee`, `Phase5Specialization.lean`). The
cross-repo FV rollout is complete; this issue can be closed.
