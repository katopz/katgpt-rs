# Research 351: Cross-Repo Lean 4 Formal Verification Pattern — The Super-GOAT Capture Protocol

> **Cross-reference note.** Coordinator: `katgpt-rs/.issues/012_cross_repo_fv_lean4_rollout_coordinator.md` (Phases 1–4 COMPLETE).
> Capability-face doc: `riir-ai/.research/003_Commercial_Open_Source_Strategy_Verdict.md` §"Formal Verification".
> Bug-finding case study: `riir-ai/.issues/354_lora_weight_version_torn_read.md` (riir-ai issue, closed + removed; all tasks DONE).
> Predecessor (gap analysis): `katgpt-rs/.research/292_Bridge_Neuro_Symbolic_Formal_Verification_Gap.md` (written when zero proofs existed).
> **Date:** 2026-06-30
> **Status:** Active — Super-GOAT process IP (the FV rollout pattern itself is the reusable capture).
> **Classification:** Public (katgpt-rs) — this is the *process* note; private proof internals stay in `riir-*/.proofs/`.

---

## TL;DR

Research 292 (2026-06-23) documented the FV gap: "~40 empirical `#[test]` FV gates across the quintet, **zero machine-checked**." Seven days later, the gap is closed for four of five repos: **79 Lean 4 theorems across four `.proofs/` instances**, all within Mathlib's standard axiom budget. This note distills the pattern that made the rollout work, so the next FV effort (a new repo, a new invariant class) can reapply it without re-deriving the conventions.

The pattern is a **Super-GOAT capture protocol** in the sense of Research 003 §"Super-GOAT Capture Protocol": the open primitive (the proof technique + spec-match test scaffold) is public here; the private guides (the actual theorems over private code) stay in `riir-*/.proofs/`. A competitor reading this note gets the *process* but not the *theorems* (convention C4).

---

## 1. The four instances (as of 2026-06-30)

| Instance | Repo | Toolchain | Theorems | Axioms | What's proven |
|---|---|---|---:|---:|---|
| `KatgptProof` | `katgpt-rs` (public) | `v4.31.0` | 3 | 0 | Sigmoid ranking preservation (`action_bridge_ranking_preserved` + `'` variant + `action_bridge_argmax_preserved`). The public adoption hook. |
| `RiirChainProof` | `riir-chain` (private) | `v4.31.0` | 32 | 0 | LatCal round-trip (3), quorum determinism (5) + tier/block root funcs (2), chain-side `merkle_root` init (4), slashing monotonicity (8), split-key security (10). |
| `NeuronDbProof` | `riir-neuron-db` (private) | `v4.31.0` | 28 | 0 | Shard layout consistency + `merkle_root` init (16), freeze gate contract (8), Merkle tamper-evidence (4, parameterized over injective `hashFn`). |
| `RiirAiProof` | `riir-ai` (private) | `v4.32.0-rc1` | 16 | 1† | HLA scalar boundedness (14), freeze/thaw reader invariant (2, `propext`-only). |
| **Total** | | | **79** | **1†** | |

† `arcswap_store_atomicity` in `RiirAiProof/Runtime/Basic.lean` is **documentation-only** — the actual theorems depend only on `[propext]`. See §3.2.

`riir-train` is **excluded** from FV: training properties (convergence, no-NaN, quality gain) are probabilistic/behavioral, not invariant-shaped. Lean is the wrong tool for "this adapter plays 12% better" — that's an empirical GOAT gate, not a theorem.

---

## 2. The six conventions (C1–C6) — empirically validated

These were proposed in `katgpt-rs/.issues/012_*` §4 *before* Phase 1 started, as conventions to "lock in BEFORE Phase 1 starts." Four phases later, they are **empirically validated** — every instance follows them and the rollout never needed an exception. Future FV work should treat them as load-bearing, not aspirational.

### C1 — Toolchain pin policy

**Rule:** Each `.proofs/` pins its own `lean-toolchain`. Pin the lowest version that compiles the theorem. Don't force Mathlib where `omega`/`ring`/core tactics suffice.

**Validated split:**
- `RiirChainProof`, `NeuronDbProof`, `KatgptProof` → `leanprover/lean4:v4.31.0` (Mathlib-free; theorems reduce to integer linear arithmetic, algebraic identities, lattice ops — all in Lean core).
- `RiirAiProof` → `leanprover/lean4:v4.32.0-rc1` (Mathlib required — sigmoid boundedness needs `Real.exp_pos`, not in Lean core).

**Lesson:** Three of four instances avoided Mathlib. The Mathlib dependency is a real cost (8592-file precompiled cache download on first build, ~5 min one-time). Only pull it in when the theorem is genuinely transcendental. The `omega`/`ring`/lattice tactic toolbox covers far more than expected — quorum determinism, layout offsets, slashing monotonicity, and freeze-gate booleans all stayed Mathlib-free.

### C2 — Axiom policy

**Rule:** Target axioms = `{propext, Classical.choice, Quot.sound}` only (Mathlib's standard foundation). No `sorry`, ever. Verified by `#print axioms` in CI.

**Validated:** All 79 theorems stay within budget. Notably, the freeze/thaw theorems *under*-use it — they depend only on `[propext]` because the atomicity is structural. The axiom budget is a ceiling, not a target; under-shooting it is a strength.

### C3 — Spec-match test convention (the two-way gate)

**Rule:** Every Lean theorem has a paired Rust spec-match test. Lean proves the math (over `ℝ` or abstract types); Rust catches spec drift (on `f32`, with NaN/Inf/saturation edge cases). Both must pass for the proof to be valid.

**Pattern (three concrete examples):**

| Theorem | Spec-match test | What the test catches that Lean can't |
|---|---|---|
| `read_snapshot_consistent` (riir-ai) | `concurrent_lora_no_torn_read` (100K iterations, A=i / B=i*2 fill, hard-fails on any torn read) | f32 memory reordering the proof doesn't model; the *actual* ArcSwap implementation vs the abstract SC model |
| `hla_scalar_via_sigmoid_bounded` (riir-ai) | `hla_bounds_spec_match` (6/6, incl. 10K-trial sigmoid stress on `[-10,10]` strict + `[-50,50]` closed + NaN/Inf edge cases) | f32 saturation at `\|x\| > 17` (Lean's `ℝ` has no saturation); NaN→0 clamp path (Lean's `ℝ` has no NaN) |
| `computeRootFromProof_injective_in_leaf` (neuron-db) | `merkle_soundness_spec_match` (7/7, incl. 1000-trial × 16K single-byte tamper stress) | BLAKE3's actual collision resistance (the Lean theorem parameterizes over any injective `hashFn`; the test validates BLAKE3 satisfies the hypothesis) |

**Why the two-way gate matters:** Lean proves the math is correct *if the Rust matches the spec*. The Rust test proves the Rust matches the spec *on the actual hardware*. Neither half is sufficient alone. A spec drift (e.g. someone swaps `sigmoid` for a piecewise approximation) breaks the Rust test even though the Lean theorem still type-checks; a Mathlib bug would break the Lean theorem even though the Rust test still passes.

### C4 — Private proofs stay private

**Rule:** Lean files in `riir-*/.proofs/` are internal-only. The open/private FV split mirrors the open/private code split: `katgpt-rs/.proofs/` proves generic math (the WHAT); `riir-*/.proofs/` proves the HOW over private code. **Do not cross-port private proofs into the public repo, even as "reference".**

**Validated:** `KatgptProof` (public) proves ranking preservation over abstract monotone direction vectors — no game/chain/shard semantics. `RiirAiProof` (private) extends it to boundedness over the actual `NpcEmotionScalars` and freeze/thaw over the actual `LoRAWeightVersion`. The public theorem is the adoption hook; the private theorems are the moat.

### C5 — Build isolation

**Rule:** `lake build` artifacts (`.lake/`) must not pollute Cargo `target/`. Add `.lake/` to each repo's `.gitignore`. CI invokes `lake build` separately from `cargo test`.

**Validated:** All four instances gitignore `.lake/` and `lake-manifest.json`. The CI hooks (`scripts/ci_feature_guard.sh` in each repo) run `lake build` as a separate layer, skipping gracefully if `elan` is absent. No cross-contamination between Lean and Cargo build graphs.

### C6 — README discipline

**Rule:** Each `.proofs/README.md` documents: theorem list, axiom inventory, Mathlib-dependency rationale, spec-match test path, regeneration protocol (what to do when the Rust side changes).

**Validated:** All four instances have READMEs following this template. The regeneration protocol is the load-bearing part — it tells the next agent "if you change `sigmoid`, here's exactly which theorem to re-check and which test to re-run." Without it, the proof becomes unmaintained ceremony.

---

## 3. The pattern that made the long pole collapse

### 3.1 The intended sequencing (and why it worked)

The coordinator sequenced phases by tractability: leaf crate (neuron-db, pure algebra) → chain (LatCal composition) → fill-ins (Merkle, slashing, split-key) → riir-ai (the "1–2 week long pole" — memory model).

This sequencing de-risked each phase: Phase 1 proved the toolchain + conventions worked on the easiest case; Phase 2 proved they composed across repos; Phase 3 proved the parameterized-over-hash trick; Phase 4 inherited all of that and attacked the hard case last, with maximum confidence.

### 3.2 The bug-finding payoff (the Issue 354 case study)

The strongest evidence that FV pays for itself: **scoping the riir-ai freeze/thaw theorem found a real concurrency bug that testing missed.**

`LoRAWeightVersion` (`riir-ai/crates/riir-engine/src/episode_buffer.rs`) stored (version, A, B) in three independent atomics. The struct's doc comment claimed:

> "A reader that observes the new version is guaranteed to see both new matrices because the Release store of version happens after both ArcSwap stores."

**This claim was false.** The Release/Acquire pair on `version` only orders the version load relative to stores that happen-before the version store *on the writer side*. It does not prevent the reader's `a.load()` from observing an OLD value while `b.load()` and `version.load()` observe NEW values — a torn `{old_A, new_B, new_version}` snapshot.

The existing test (`concurrent_lora_update_read`, 1000 iterations) filled both matrices with the same scalar per update but **never cross-checked `first_a == first_b`**, so torn reads passed silently. The test was green; the bug was live.

**Only the act of trying to prove the invariant surfaced the gap.** The proof requires "readers observe a consistent triple" as a hypothesis; examining the code to justify that hypothesis revealed it wasn't justified.

**The fix:** single-`ArcSwap<Inner>` packing, where `Inner = { version, a, b }`. One atomic pointer swap for the whole triple → torn read impossible by construction. The Lean proof then reduced to a definitional unfold (`read_snapshot_consistent`, depending only on `[propext]` — the SC atomicity is structural, not an axiom). Strengthened spec-match: `concurrent_lora_no_torn_read` (100K iterations, A=i / B=i*2 fill, hard-fails on any torn read).

**The pattern repeats the `merkle_root` and `can_freeze` bug classes:** every past invariant violation was a property we asserted (in doc comments or test names) but didn't prove. The Lean proof effort is the only mechanism that reliably surfaces the gap, because it refuses to accept "obviously true" as a justification.

### 3.3 The "fix the code, not the proof" move

When the code violates the invariant the proof claims, the temptation is to weaken the theorem to match the code. **Don't.** The invariant in the doc comment is usually the *intended* contract; the code is wrong. Fixing the code to satisfy the invariant (and re-running the proof) produces both a correct implementation *and* a proof of correctness. Weakening the proof produces a proof of a weaker property that may not be the one the runtime depends on.

In Issue 354, this move collapsed a projected 1–2 week proof effort into a definitional unfold — not because the proof got easier, but because the *code* got correct. The lesson: **if the proof is hard, suspect the code.**

---

## 4. When to apply this pattern (and when not to)

### Apply when the invariant is:
- **Static** (doesn't depend on runtime data distributions, adversary behavior, or learning dynamics)
- **Invariant-shaped** (a universal quantification over constructors, operations, or code paths — "every constructor sets `merkle_root`", "every read returns a consistent triple")
- **Asserted in doc comments or test names** but not enforced by the type system — this is the smell that the assertion is load-bearing but unproven
- **Cross-constructor or cross-path** (the bug class that `--all-features` CI catches but single-feature CI misses)

### Do NOT apply when the property is:
- **Probabilistic** (training convergence, win-rate improvement, quality gain) — these are empirical GOAT gates, not theorems
- **Behavioral** (collapse recovery, curiosity-driven exploration, emergent NPC behavior) — Lean is a poor fit for "this agent explores well"
- **Performance** (latency, throughput, alloc-free) — these are benchmark gates; Lean proves correctness, not speed
- **Single-implementation** (no constructor variance, no multi-path hazard) — a Rust test is cheaper and sufficient

The riir-ai AGENTS.md captures this as the **static-vs-dynamic verification split**: prove the static invariants the runtime *depends on*; empirically test the dynamic behaviors the runtime *produces*. The two are complements, not substitutes.

---

## 5. The reusable scaffold (for the next FV effort)

When starting a new FV instance (a new repo, a new invariant class), follow this scaffold:

1. **Classify the invariant** (§4). If it's not static + invariant-shaped, stop — use an empirical gate instead.
2. **Find the bug-shaped version** — grep for doc comments and test names that assert the invariant. These are the assertions most likely to be load-bearing-but-unproven. The Issue 354 bug was hiding behind a doc comment that said "this can't happen."
3. **Pin the lowest toolchain** that compiles (C1). Start Mathlib-free; only escalate to Mathlib if the theorem is genuinely transcendental.
4. **Write the spec in `Basic.lean`** — model the Rust struct/operation as Lean definitions. Abstract away anything the theorem doesn't care about (e.g. the freeze/thaw proof abstracts matrix contents to a token `Matrix : Type` because the theorem is about structural atomicity, not matrix data).
5. **Write the theorems in `<Class>.lean`** — aim for the axiom budget ceiling (C2); celebrate under-shooting it.
6. **Write the spec-match test** (C3) — the test must exercise the f32/NaN/edge-case paths Lean can't model. If the test can't distinguish the correct implementation from the buggy one, the test is insufficient (this is exactly the `concurrent_lora_update_read` failure — strengthen it before proceeding).
7. **Document regeneration** (C6) — the README must tell the next agent what to re-check when the Rust side changes.
8. **Add the CI hook** (C5) — `lake build` as a separate layer, skipping if `elan` absent.
9. **Update the coordinator** (`katgpt-rs/.issues/012_*`) — add the instance to the state table with theorem count.

The scaffold is deliberately small. The pattern's value is that it's *repeatable* — four instances in seven days, one of which found a real bug.

---

## 6. Open questions / future work

- **Phase 5 (riir-ai bridge ordering)** — largely redundant with the public `action_bridge_ranking_preserved` theorem (which is fully parameterized over arbitrary direction vectors). The honest path is a thin specialization doc; the substantive alternative (linear independence of the committed-blend direction basis) is a *different* theorem than originally scoped. Pending decision.
- **UQ-bearing primitive GOAT gate** (the "Report the Floor" rule, adopted 2026-06-28 per Research 322 / Plan 340; retroactive audit COMPLETE, consolidated in `katgpt-rs/.benchmarks/010_report_the_floor_consolidated.md`) — requires UQ-bearing primitives to benchmark against the conformal-naive floor. This is an empirical gate, not a theorem, but it's the next GOAT-gate discipline worth distilling as a sibling pattern note.
- **Cross-instance lemma sharing** — currently each `.proofs/` is standalone. If a fifth instance emerges that needs, e.g., the LatCal round-trip lemma, it would re-prove it. A shared `katgpt-rs/.proofs/Common/` library is conceivable but premature (YAGNI — four instances, zero shared lemmas needed so far).
- **Memory model for richer concurrency** — the freeze/thaw proof models `ArcSwap` as a single-field SC atomic. Richer concurrency patterns (seqlock, RCU, cross-thread epoch reclamation) would need a real memory model in Lean, which doesn't ship by default. Defer until a theorem actually requires it.

---

## 7. Cross-references

- **Coordinator:** `katgpt-rs/.issues/012_cross_repo_fv_lean4_rollout_coordinator.md` (Phases 1–4 COMPLETE, 79 thms)
- **Capability-face:** `riir-ai/.research/003_Commercial_Open_Source_Strategy_Verdict.md` §"Formal Verification"
- **Predecessor (gap analysis):** `katgpt-rs/.research/292_Bridge_Neuro_Symbolic_Formal_Verification_Gap.md`
- **Bug-finding case study:** `riir-ai/.issues/354_lora_weight_version_torn_read.md` (riir-ai issue, closed + removed; all tasks DONE)
- **riir-ai FV sibling issue:** `riir-ai/.issues/348_freeze_thaw_runtime_lean_proofs.md` (riir-ai issue, closed + removed; all tasks DONE) (T1–T8 all done)
- **Sibling proof instances:** `katgpt-rs/.proofs/KatgptProof/` (public), `riir-chain/.proofs/RiirChainProof/`, `riir-neuron-db/.proofs/NeuronDbProof/`, `riir-ai/.proofs/RiirAiProof/`
- **Past bugs prevented:** `merkle_root` (riir-neuron-db AGENTS.md), `can_freeze` (riir-neuron-db Plan 002 Phase 5), AC-Prefix G1 (`katgpt-rs` Plan 313), torn-read (riir-ai Issue 354)

---

## TL;DR

Research 292 said "zero machine-checked proofs." Seven days later: 79 Lean 4 theorems across four `.proofs/` instances, all within Mathlib's axiom budget. The rollout worked because of six conventions (C1–C6) that were locked in *before* Phase 1 and empirically validated across all four instances. The strongest evidence the pattern pays for itself: scoping the freeze/thaw theorem found a real concurrency bug (Issue 354) that doc comments claimed couldn't happen and tests couldn't catch — the canonical FV-as-bug-finder story. The pattern repeats the `merkle_root`/`can_freeze` bug class: every past invariant violation was a property we asserted but didn't prove. **The lesson, distilled: if the proof is hard, suspect the code.**
