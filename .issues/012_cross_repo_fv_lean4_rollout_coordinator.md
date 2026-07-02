# Issue 012: Cross-Repo Formal Verification (Lean 4) Rollout — Coordinator

**Date:** 2026-06-23 (coordinator strategy; filed retroactively 2026-07-02)
**Severity:** 🟡 COORDINATION / TRACKING
**Status:** ✅ **ALL FOUR INSTANCES COMPLETE.** Phases 1–4 shipped. Phase 5 (bridge ordering on learned directions) CLOSED 2026-06-30 via specialization — no new theorem needed.
**Referenced by:** `riir-ai/AGENTS.md` → Static vs Dynamic Verification Split; `riir-neuron-db/.proofs/README.md`; `riir-ai/.proofs/README.md`

---

## Summary

The 5-repo quintet (katgpt-rs / riir-ai / riir-chain / riir-neuron-db / riir-train) carries **four** Lean 4 formal-verification instances — one per proof-able repo (riir-train is training-method research, no static invariants to prove). This issue is the coordinator: it tracks the rollout strategy, the per-instance invariant classes, and the cross-repo dependencies.

**Rule C4 (private proofs stay private):** Lean files in private repos (`riir-ai`, `riir-chain`, `riir-neuron-db`) are internal-only. Do NOT cross-port them into the public `katgpt-rs` repo, even as "reference". The public repo ships only `KatgptProof`.

---

## The four instances — current state

| # | Instance | Repo | Plan(s) | Toolchain | Theorems | Axioms | Status |
|---|---|---|---|---|---|---|---|
| 1 | `RiirChainProof` | `riir-chain` (private) | [004](../../riir-chain/.plans/004_latcal_lean4_roundtrip_proof.md) + [008](../../riir-chain/.plans/008_chain_fv_phase2_quorum_and_merkle_root.md) + [009](../../riir-chain/.plans/009_chain_fv_phase3_slashing_and_split_key.md) | `v4.31.0` (Mathlib-free) | ~25 (LatCal roundtrip, quorum determinism, `merkle_root` stamping, slashing monotonicity, split-key wire-safety) | `{propext}` or **none** (most theorems axiom-free) | ✅ Phases 1–3 COMPLETE |
| 2 | `KatgptProof` | `katgpt-rs` (public MIT) | [293](../.plans/293_action_bridge_lean4_monotonicity_proof.md) | `v4.32.0-rc1` (Mathlib) | 2 (sigmoid ranking preservation + argmax preservation) | `{propext, Classical.choice, Quot.sound}` | ✅ COMPLETE |
| 3 | `NeuronDbProof` | `riir-neuron-db` (private) | [007](../../riir-neuron-db/.plans/007_neuron_shard_fv_phase1_layout_and_freeze_gate.md) + [008](../../riir-neuron-db/.plans/008_neuron_shard_fv_phase2_merkle_proof_soundness.md) | `v4.31.0` (Mathlib-free) | 19 (shard layout, freeze gate self-consistency, Merkle tamper-evidence) | `{propext, Classical.choice, Quot.sound}` or **none** (7 layout + 4 Merkle axiom-free) | ✅ Phases 1–2 COMPLETE |
| 4 | `RiirAiProof` | `riir-ai` (private) | [353](../../riir-ai/.plans/353_*.md) + [Issue 348 T2](../../riir-ai/.issues/348_*.md) | `v4.32.0-rc1` (Mathlib) | 16 (HLA boundedness + freeze/thaw reader invariant) + Phase 5 specialization | `{propext, Classical.choice, Quot.sound}` (FreezeThaw: `{propext}` only) | ✅ COMPLETE (Phase 5 CLOSED via specialization) |

**Total: ~62 theorems across 4 instances, 4 theorem classes, zero `sorry`.**

---

## Rollout phases (all shipped)

### Phase 1 — `RiirChainProof` (riir-chain, Plan 004) ✅
The first instance. Proves the sync-boundary bridge: LatCal fixed-point round-trip tolerance (`∀ x, isClean x → |fromFixed(toFixed x) − x| ≤ 10⁻⁸`). Deliberately Mathlib-free (theorem reduces to integer linear arithmetic, decided by `omega`, `lake build` < 5s, pinned `v4.31.0`).

### Phase 2 — `KatgptProof` (katgpt-rs, Plan 293) ✅
The public instance. Proves sigmoid ranking preservation: `∀ q d₁ d₂, dot q d₁ > dot q d₂ → sigmoid (dot q d₁) > sigmoid (dot q d₂)`. Requires Mathlib (`Real.sigmoid_strictMono` — transcendental analysis of `exp` not in Lean core). Toolchain `v4.32.0-rc1` (Mathlib's requirement). Paired Rust spec-match test (`bridge_spec_match`) catches drift between Rust `fast_sigmoid` and Mathlib `Real.sigmoid`.

### Phase 3 — `NeuronDbProof` (riir-neuron-db, Plans 007 + 008) ✅
The highest-ROI target. Two past bugs (`merkle_root` forgotten in `new_spectral`, `can_freeze` gate decoupling) are textbook invariant violations — now Lean theorems. The crate is a leaf (no chain dep), theorems are pure algebra/boolean logic, Mathlib-free. 19 theorems: 7 shard-layout (axiom-free), 8 freeze-gate self-consistency, 4 Merkle tamper-evidence (axiom-free, cryptographic injectivity is a *hypothesis* not an axiom).

### Phase 4 — `RiirAiProof` (riir-ai, Plan 353 + Issue 348 T2) ✅
The runtime instance. `riir-ai`'s selling point is **dynamic** (self-play, curiosity, collapse recovery — empirical domain, poor Lean fit). The proof-able subset is the **static invariants the runtime depends on**: HLA scalar boundedness (14 theorems — every sigmoid-derived scalar in `(0,1)`, every clamped emotion scalar in `[0,1]`) and freeze/thaw reader invariant (2 theorems — no torn `{old_A, new_B, new_version}` snapshot; readers observe one atomic `ArcSwap` load). Toolchain `v4.32.0-rc1` (Mathlib).

### Phase 5 — Bridge ordering on learned directions ✅ CLOSED (2026-06-30)
**Originally P2/P3.** The goal was to extend `KatgptProof`'s public sigmoid-monotonicity (over fixed/public direction vectors) to `riir-ai`'s private tuned direction vectors.

**Resolution: no new theorem needed.** The public `action_bridge_ranking_preserved` theorem is **fully parameterized** over arbitrary direction vectors `d₁ d₂ : ι → ℝ` — there is no hypothesis restricting them to "public" or "fixed" directions. Instantiating `ι = Fin 8` (HLA dimension) and `d₁ d₂` with riir-ai's learned/tuned direction vectors yields the Phase 5 property directly.

Formal record: `riir-ai/.proofs/RiirAiProof/Phase5Specialization.lean`.

---

## What's modeled vs what's assumed

| Aspect | Modeled (Lean over `ℝ`) | Assumed (Rust spec-match test) |
|---|---|---|
| HLA boundedness | Strict `(0,1)` via `Real.sigmoid` | f32 saturation near `\|x\| > 17`; spec-match enforces strict on `[-10,10]`, closed `[0,1]` on `[-50,50]` |
| NaN handling | Not modeled (`ℝ` has no NaN) | `clamped()` (NaN → 0 before clamp) validated by `hla_bounds_spec_match` |
| Freeze/thaw memory model | Single-`ArcSwap<Inner>` load (Issue 354 fix made it hold by construction) | Torn-read impossibility (Issue 354) |
| Merkle soundness | `computeRootFromProof` injective in leaf (under `hashFn` injectivity hypothesis) | BLAKE3 injectivity (cryptographic assumption, not provable in Lean) |

---

## Regenerating after Rust changes

If a new sigmoid-derived scalar is added to the runtime:

1. Add a matching theorem in `riir-ai/.proofs/RiirAiProof/Hla/Bounded.lean` mirroring `hla_scalar_via_sigmoid_bounded` (A4) — one-line proof (`sigmoid_bounded (dot q d)`).
2. If a new field is added to `NpcEmotionScalars`, add a matching `clamped_<field>_bounded` theorem (mirror B2–B6) and extend `clamped_all_bounded`.
3. Run `cd riir-ai/.proofs && lake build` — must pass with no `sorry`.
4. Run `cargo test -p riir-engine --test hla_bounds_spec_match` — must pass.
5. Verify axiom inventory: `cd riir-ai/.proofs && lake env lean PrintAxioms.lean` — must be `{propext, Classical.choice, Quot.sound}` only.

See `riir-ai/.proofs/README.md` §"Regenerating after Rust changes" for the full protocol.

---

## Stale status line (ACTION ITEM for this issue) — ✅ DONE

`riir-ai/AGENTS.md` → Static vs Dynamic Verification Split table previously listed:

```
| Bridge ordering on learned directions | 🟡 P2/P3 (Phase 5) | Coordinator `katgpt-rs/.issues/012_*` |
```

This **was** stale. **Fixed (verified 2026-07-02):** the table now reads:

```
| **Bridge ordering on learned directions** — extends public sigmoid-monotonicity to private tuned direction vectors | ✅ DONE (Phase 5, 2026-06-30) | `.proofs/RiirAiProof/Phase5Specialization.lean` — covered by specialization ... Coordinator `katgpt-rs/.issues/012` |
```

Verified via grep of `riir-ai/AGENTS.md` (Static vs Dynamic Verification Split table). No further action needed.

---

## References

- **Public sibling (extends):** `katgpt-rs/.proofs/KatgptProof` (Plan 293) — sigmoid ranking preservation.
- **Chain sibling:** `riir-chain/.proofs/RiirChainProof` (Plans 004 + 008 + 009).
- **Neuron-db sibling:** `riir-neuron-db/.proofs/NeuronDbProof` (Plans 007 + 008).
- **riir-ai instance:** `riir-ai/.proofs/RiirAiProof` (Plan 353 + Issue 348 T2).
- **riir-ai issue tracker:** `riir-ai/.issues/348_*` (T1/T3/T5/T6 done; T2 freeze/thaw done; T4 blocked on T2; T7/T8 quick follow-ups).
