# Research 292: Bridge Neuro-Symbolic Formal Verification — Gap Analysis & Open Primitive

> **Cross-reference note.** Primary chain-side guide: `riir-chain/.research/004_LatCal_Fixed_Point_Bridge_Lean4_Proof_Guide.md`.
> Sibling crossref: `riir-neuron-db/.research/005_Shard_Snapshot_Atomicity_Iris_Proof_Crossref.md`.
> Plan (open primitive): `katgpt-rs/.plans/293_action_bridge_lean4_monotonicity_proof.md`.
> **Date:** 2026-06-23
> **Status:** Active — GOAT (open primitive, strengthens commercial moat).
> **Classification:** Public (katgpt-rs)

---

## TL;DR

User asked: *"our formal verification proof is on bridge neuro symbolic am i right? do we have atomic proof? where? and how with what?"*

**Honest verdict:** No. We have ~40 empirical `#[test]` FV gates across the 5-repo quintet (`neuron_db_fv.rs` g1–g12, `consensus_fv.rs`, `rsm_fv.rs` g1–g14, `econ_fv.rs` G-ECON-1..10, `chaind_fv.rs`, `latcal_fixed.rs` g1–g2). **Zero are machine-checked.** "Atomic" in our codebase means `std::sync::atomic` (thread-safe hot-swap), not "indivisible theorem". The closest formal-verification-adjacent work is Plan 128 "Proof Sketch Evolution" (AlphaProof-style search *infrastructure*, not proofs of our system) and Plan 223 Lean4Agent fusion (which *deliberately* avoided Lean, going Rust-native + WASM + BLAKE3).

This note documents the gap and proposes the open-primitive tier (Tier 3) of the fix: a Lean 4 monotonicity proof for `ActionBridge::select_action` (the latent→raw bridge in `katgpt-rs/crates/katgpt-core/src/bridge/mod.rs`). The chain-side LatCal round-trip proof (Tier 1) and neuron-db Iris linearizability proof (Tier 2) live in sibling repos.

---

## 1. What "bridge neuro-symbolic" actually means in this codebase

Three distinct things share the name "bridge":

| Bridge | File | What it does | Latent/Raw |
|---|---|---|---|
| **LatCal fixed-point bridge** | `riir-chain/src/encoding/latcal_fixed.rs` | f32 ↔ i64×10^6 for cross-node deterministic commitment | **Raw** (synced, quorum-committed) |
| **ActionBridge** | `katgpt-rs/crates/katgpt-core/src/bridge/mod.rs` | Latent Q-values → raw action index via `sigmoid(dot())` | **Latent→Raw** (one-way projection) |
| **ShardIndex snapshot** | `riir-neuron-db/src/index.rs` | papaya lock-free HashMap zone→shard lookup | **Latent** (local, but must be atomic) |

The "neuro-symbolic" layer is Plan 211 (Three-Mode Router), Plan 210 (INSIGHT), Plan 209 (FOL-LNN) — all empirical GOAT gates, no proofs.

The user's mental model conflates these. The honest answer: **none of the three bridges has a formal proof; the neuro-symbolic layer has only empirical gates.**

---

## 2. What exists vs what's missing

### Existing FV (all empirical `#[test]`)

```
riir-neuron-db/src/neuron_db_fv.rs           g1..g12   (12 gates)
riir-chain/src/consensus_fv.rs                4 invariants + 2 statistical
riir-chain/src/consensus/rsm_fv.rs            g1..g14   (14 gates)
riir-chain/src/economics/econ_fv.rs           G-ECON-1..10
riir-chain/crates/riir-chaind/tests/chaind_fv.rs   daemon FV
riir-chain/src/encoding/latcal_fixed.rs       g1..g2    (bridge round-trip, 2 samples)
riir-chain/src/goat_proof.rs                  g1..g13   (chain GOAT)
```

### Existing formal-verification-*adjacent* research

| Note | What it actually proposed | Why it's not a proof |
|---|---|---|
| `katgpt-rs/.research/198_Lean4Agent_Formal_Workflow_Verification.md` | 3-layer Lean4 framework for agent workflows | Distilled to Plan 223 with **explicit decision to NOT use Lean** — went Rust-native + WASM + BLAKE3 |
| `katgpt-rs/.research/106_Shock_Confidence_Formal_PDE_Verification.md` | Hierarchical proof methodology | Methodology paper; no Lean/Coq code shipped. Plan 145 `proof_cert` only documents the methodology |
| `katgpt-rs/.research/088_AlphaProof_Nexus_Formal_Proof_Search.md` + Plan 128 | Elo-rated proof sketch population | Infrastructure for *searching* proofs; proves nothing about *our* system |
| `katgpt-rs/.research/170_LEAP_Blueprint_DAG_Proof_Search.md` | AND-OR DAG proof search | Same — search infrastructure |

### What's missing (the gap)

- **No `.lean` / `.coq` / `.v` / `.isabelle` / `.tla` files** anywhere in any of the 5 repos (`find_path` returned zero hits).
- **No Lean/Coq/Iris dependency** in any `Cargo.toml` (`grep` for `lean|coq|tlaplus|smt|z3|proof` returned only feature-flag names like `proof_cert`, `proof_sketch_evolution` — all Rust-native).
- **No theorem, lemma, or axiom declarations** in any `.rs` file (`grep` for `theorem|lemma|axiom` returned only docstrings).
- Plan 223's `WASMProofWitness` (Idea 5) only adds a BLAKE3 witness hash to WASM validator results — it is *not* a Lean proof.

---

## 3. Distillation — the open primitive (Tier 3)

### Paper-side prior art (online research)

| Source | What it proves | Relevance |
|---|---|---|
| Lean 4 Mathlib `Real.strictMono_sigmoid` (or equivalent) | sigmoid is strictly monotone | Direct — Tier 3 |
| Boldo et al., "Formal Verification of Floating-Point Programs" (ARITH18) | Frama-C/Why3 methodology for FP | Reference — methodology |
| "Formal Verification of Floating Point Arithmetic" (arXiv:2512.06850) | IEEE 754 adder in Lean | Reference — FP library maturity |
| "End-To-End Formal Verification of a Fast and Accurate Floating…" (ITP 2024) | binary64 approximation in Coq | Reference — large-scale FP proofs are feasible |
| VeriTrans (arXiv:2604.10341) | NL→PL compiler with verified backend | Adjacent — neuro-symbolic compiler pattern |
| NeuS-V | Neuro-symbolic formal verification of text-to-video | Adjacent — neuro-symbolic FV is a 2025 theme |

### The primitive (Tier 3 — `katgpt-rs`)

**Property to prove:**

```lean
-- For all action-direction vectors d and Q-value vectors q,
-- the ActionBridge::select_action ranking is monotone in dot product.
theorem action_bridge_ranking_preserved
  {D : ℕ} (q : Vector D Float32) (d₁ d₂ : Vector D Float32)
  (h : dot q d₁ > dot q d₂) :
  sigmoid (dot q d₁) > sigmoid (dot q d₂) := by
  exact strictMono_sigmoid (dot q d₁) (dot q d₂) h
```

5 lines. The proof value is **not** the theorem (it's a Mathlib lemma) — it's:

1. **Compositionality chain.** Once we have `monotone_sigmoid` and `monotone_dot_product` as named Lean lemmas, we can compose them into `monotone_action_bridge` and ship a `.lean` proof certificate alongside the Rust binary.
2. **Spec-match test.** Rust `static_assert` confirms `ActionBridge::select_action` actually computes `sigmoid(dot())` (not e.g. softmax, not ReLU). The proof is only valid if the Rust matches the Lean spec — the test enforces that.
3. **Promotion path.** Today Plan 262 G1.3 asserts ranking preservation over 1000 random triples. After Tier 3, the same property is `∀`. The empirical test stays as a smoke test; the proof is the contract.

### Why Lean 4 (not Coq, not F*, not TLA+)

- **Coq** would also work but the Mathlib FP+sigmoid library is more mature in Lean 4.
- **F*** is for verified C extraction; we don't extract — we prove the math and assert the Rust matches.
- **TLA+** is for concurrent systems; Tier 3 has no concurrency. Tier 2 (ShardIndex atomicity) is where TLA+ or Iris would fit, not here.
- **Dafny** is Java/C#-first; weaker ecosystem fit for a Rust codebase.
- **Lean 4** has `elan` (Rust-style toolchain), Mathlib, and an active FP library. It's the closest cultural fit.

### What this primitive does NOT do

- Does NOT verify the LatCal round-trip (Tier 1, riir-chain).
- Does NOT verify ShardIndex atomicity (Tier 2, riir-neuron-db).
- Does NOT verify the neuro-symbolic router (Plan 211) — that has no clean formal spec.
- Does NOT extract verified Rust code. We prove the math; Rust matches by test.

---

## 4. Verdict

**GOAT (open primitive).** Not Super-GOAT — sigmoid monotonicity is a known Mathlib lemma, not a novel mechanism. The value is (a) closing the empirical→machine-checked gap for the cheapest property first, (b) establishing the Lean 4 toolchain pattern that Tiers 1 and 2 build on, (c) shipping the first real proof certificate in the quintet.

**One-line reasoning:** First real machine-checked proof in the codebase; trivial theorem, non-trivial infrastructure; sets pattern for the harder Tier 1 (LatCal round-trip) and Tier 2 (Iris atomicity).

**Routing:** Open primitive in `katgpt-rs/.proofs/` + plan `katgpt-rs/.plans/293_action_bridge_lean4_monotonicity_proof.md`. No private guide (sigmoid monotonicity is public math).

---

## 5. Related research

| # | File | Connection |
|---|---|---|
| 198 | Lean4Agent Formal Workflow Verification | Predecessor — deliberately avoided Lean; we reverse that decision for the bridge only |
| 106 | Shock Confidence Formal PDE Verification | Methodology — hierarchical proofs, IEEE 754 awareness |
| 088/104 | AlphaProof Nexus | Search infrastructure (not proofs of our system) |
| 170 | LEAP Blueprint DAG | Same — search infrastructure |
| 145 (Plan) | Hierarchical GOAT Proof Certificates | Methodology doc; Tier 3 makes it concrete |
| 223 (Plan) | Lean4Agent Fusion | Idea 5 `WASMProofWitness` — BLAKE3 witness; Tier 3 adds the Lean proof the witness *claims* |
| 262 (Plan) | Latent Physics Primitives (ActionBridge) | The thing being proved |
| 276 (Plan) | MicroRecurrentBeliefState G1.5 | Tier 2 atomicity test that Iris would promote |

---

## TL;DR

User's question revealed a real gap: we have ~40 empirical FV gates, zero machine-checked proofs. This note proposes the cheapest open primitive to close it — a 5-line Lean 4 monotonicity proof for `ActionBridge::select_action`, gated by a spec-match Rust test. Sets the pattern for Tier 1 (LatCal round-trip, riir-chain) and Tier 2 (ShardIndex atomicity, riir-neuron-db). Public math, MIT-tier primitive, strengthens the moat by establishing the toolchain. **Not Super-GOAT — GOAT infrastructure.**
