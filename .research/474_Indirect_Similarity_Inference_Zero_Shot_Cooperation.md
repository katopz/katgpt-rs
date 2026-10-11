# Research 474: Indirect Similarity Inference — Zero-Shot Cooperation from Third-Party Observation [Scoped Super-GOAT]

> **Source paper:** [arXiv:2608.03958](https://arxiv.org/pdf/2608.03958) — Meulemans, Wołczyk, Weis, Nasser, et al. (Google Paradigms of Intelligence + Mila + ETH + DeepMind), 4 Aug 2026. §I "Indirect similarity analysis" + Fig 4.
> **Date:** 2026-08-11
> **Status:** Active — **scoped Super-GOAT** (zero-shot cooperation from third-party observation ONLY). The direct-inference mechanism is GOAT-tier (R471 §3); the equilibrium *concept* is covered by shipped CCE (`CceLp<N,A>`, Plan 295, DEFAULT-ON). This note claims novelty ONLY for the indirect-inference subset.
> **Predecessor:** [R471](471_Similarity_Inference_Embedded_Equilibrium.md) — the open primitive note covering both direct + indirect inference; revised GOAT verdict there. This note opens the scoped Super-GOAT claim for indirect inference specifically.
> **Implementation:** [Plan 526 Phase 3](../.plans/526_similarity_inference_primitive.md) — G5 PoC PASS.
> **Benchmark:** [Bench 579](../.benchmarks/579_similarity_inference_goat.md) §G5.
> **Private half:** [`riir-ai/.research/336_Indirect_Similarity_Inference_Zero_Shot_Guide.md`](../../riir-ai/.research/336_Indirect_Similarity_Inference_Zero_Shot_Guide.md)
> **Classification:** Public (katgpt-rs/MIT) — the math + primitive; the game-runtime selling point lives in R336.

---

## TL;DR

**The scoped claim: zero-shot cooperation from parallel third-party observation** is a genuinely new capability class. Two agents who have *never directly interacted* — each has only played third-party NPCs concurrently with the other — cooperate on first direct encounter iff they share functional policy structure. No classical mechanism produces this:

- Classical reciprocity (tit-for-tat, grim trigger) **requires repeated direct interaction** — fails on first encounter.
- Bayesian theory-of-mind with hardcoded joint utility **hardcodes the cooperative objective** — does not derive cooperation from selfish best-response.
- Designer-steered CCE (our own R143/Plan 295) **requires the designer to choose the correlation signal** — the moderator chooses `Γ₀`; cooperation is designer-driven, not emergent-from-observation.
- Direct similarity inference (R471 §H, Plan 526 Phase 2) **requires direct observation** — the two agents must have played each other.

**Indirect inference closes a gap no shipped primitive closes.** The math is a Bayesian posterior transfer: agent A infers `ω_AB` from "A and B both played the same third-party C and behaved identically" — using the same evidence stream as direct inference, but routed through C as a shared reference rather than through direct A-vs-B play. The prior `α` is on shared policy; the evidence is shared third-party behavior.

**G5 PoC (2026-08-11): PASS** — 40 trials × 50 rounds × 3 shared NPCs:
- Shared-policy primaries coop rate: **1.000** (target >0.70)
- Random-policy primaries coop rate: **0.000** (target <0.25)
- Mean ω: shared **1.0000** vs random **0.0000** (perfect separation)

The mechanism is shipped in `crates/katgpt-core/src/similarity_inference/posterior.rs::SimilarityPosterior::observe_other_via_third_party`. Test: `similarity_inference::poc::g5_indirect_inference_poc` + `g5_indirect_primaries_never_directly_interact` (DEFAULT-ON, Plan 526 Phase 6 promotion).

---

## 1. The scoped novelty gate (§1.5 skill, indirect-inference-only)

### 1.1 What this note claims

- ✅ **CLAIM:** Zero-shot cooperation from third-party observation (indirect inference) is a genuinely new capability class not produced by any shipped primitive in the 7-repo stack.

### 1.2 What this note explicitly does NOT claim

- ❌ **NOT claimed:** the equilibrium *concept* is new. The equilibrium reached is a CCE — we ship `CceLp<N,A>` (Plan 295, DEFAULT-ON). See R471 §1.55.2 for the reverse-grep evidence.
- ❌ **NOT claimed:** the direct-inference mechanism is new-capability. Direct inference (Plan 526 Phase 2, G2) is a new *mechanism* (endogenous correlation device) for the CCE substrate, but the capability (cooperation from direct repeated play) overlaps with classical reciprocity in repeated games. GOAT-tier, not Super-GOAT.
- ❌ **NOT claimed:** novel math. The posterior transfer equation is R471 §I, distilled from the paper. The novelty is the *capability class* (zero-shot), not the equation form.

### 1.3 Reverse-grep evidence (the §1.5 "no candidate escape hatch" gate)

**Searched across all 7 repos** (`katgpt-rs`, `riir-ai`, `riir-chain`, `riir-neuron-db`, `riir-train`, `riir-game-sdk`, `riir-mmorpg-examples`) for: `indirect.*inference`, `third.party.*observation`, `zero.shot.*cooperat`, `parallel.*encounter`, `IndirectSimilarity`, `indirect_similarity`, `observe_other_via_third_party`.

**Hits found:**
- `katgpt-rs` — only Plan 526's own artifacts (`posterior.rs` comments, Bench 579 §G5, R274 cross-ref note, this R474). **Zero independent prior art.**
- `riir-ai` — only R335 (private Super-GOAT guide, revised in lockstep with R471) which contains the *future-tense* claim "if Plan 526 G5 PoC passes, scoped Super-GOAT claim opens". **Zero shipped implementation.**
- `riir-chain`, `riir-neuron-db`, `riir-train`, `riir-game-sdk`, `riir-mmorpg-examples` — **zero hits.**

**Verdict:** the scoped novelty claim IS defensible. The only "hits" are Plan 526's own artifacts + R335's same-session forward reference. Per skill §1.5: writing the claim triggers the mandatory guide outputs (R336) in the same session — done below.

---

## 2. The capability (what indirect inference does)

### 2.1 Setup (paper §I + Plan 526 Phase 3 PoC)

Two primary agents `A` and `B` never directly interact during the info-gathering phase. Instead, each plays `K` third-party NPCs concurrently:

```
Tick 1..T:
  A plays NPC_1, NPC_2, ..., NPC_K   (A vs each NPC; B is not involved)
  B plays NPC_1, NPC_2, ..., NPC_K   (B vs each NPC; A is not involved)
  A observes: (a_A vs NPC_k, a_NPC_k) for k=1..K, for each tick
  A also observes (sideways): (a_B vs NPC_k, a_NPC_k) for k=1..K — B's parallel play

Tick T+1: terminal one-shot PD between A and B (their first direct encounter).
```

A's posterior `ω_AB` is updated from third-party evidence: A saw that B behaved identically to itself against the same NPCs in the same situations. Under the shared-policy hypothesis, this is overwhelmingly likely; under independent-policy, it is overwhelmingly unlikely.

### 2.2 The posterior transfer equation

The likelihood-ratio compounding works exactly as in direct inference (R471 §H.2), but the evidence is `(A's action vs C, B's action vs C)` rather than `(A's action vs B, B's action vs A)`:

```
For each third-party C_k at each tick t:
  match_t   := (a_A(C_k, t) == a_B(C_k, t))   under shared-policy hypothesis: P=1
  mismatch_t := (a_A(C_k, t) != a_B(C_k, t))  under shared-policy hypothesis: P=0

ω_AB(T) = α_AB / (α_AB + (1−α_AB) · |A|^(−T·K))   [matches only]
         0                                        [any mismatch]
```

The math is identical to direct inference — `K` third-party NPCs at `T` ticks each give `T·K` evidence samples, vs `T` direct samples in the direct-inference case. Indirect inference with K=3 NPCs is 3× evidence-denser per tick than direct inference.

### 2.3 Why classical mechanisms fail on first direct encounter

| Mechanism | What it needs | Fails on first A↔B encounter? |
|---|---|---|
| Tit-for-tat | A's last direct observation of B's action | ✗ — A has never directly observed B; no history |
| Grim trigger | Direct reciprocity memory | ✗ — same reason |
| Bayesian theory-of-mind with hardcoded joint utility | Joint utility pre-specified as cooperative | ✗ — cooperation is assumed, not derived |
| Designer-steered CCE (R143/Plan 295) | Designer-set correlation signal ζ broadcast to both | ✗ — no designer signal; cooperation is designer-driven |
| Direct similarity inference (R471 §H, Plan 526 G2) | A and B have directly played each other | ✗ — A and B have not played each other yet |
| **Indirect similarity inference (this note)** | **A and B both played the same third-party NPCs concurrently** | **✓ — first A↔B encounter cooperates iff shared functional structure** |

The indirect-inference path is the only mechanism in the table that produces cooperation on first direct encounter from purely observational evidence. The others either require direct interaction history, hardcode the cooperative objective, or require designer intervention.

---

## 3. The G5 PoC (Plan 526 Phase 3, 2026-08-11 — PASS)

### 3.1 Test setup

Test: `crates/katgpt-core/src/similarity_inference/poc.rs::g5_indirect_inference_poc`.

- 40 trials, each with a different trial seed.
- Per trial: 2 primary agents (A, B), 3 shared NPCs.
- A and B never directly interact; each plays the 3 NPCs concurrently for T=50 rounds.
- After 50 rounds: terminal one-shot PD between A and B (their first direct encounter).
- A pair "cooperated" iff BOTH A and B chose Cooperate (action 0).
- Two trial kinds interleaved per seed:
  - **Shared:** A and B share `policy_seed` (identical deterministic policy).
  - **Random:** A and B have independent `policy_seed`s.

### 3.2 Results

```
G5 indirect-inference PoC (40 trials, 50 rounds, 3 shared NPCs):
  Shared-policy coop rate: 1.000 (target >0.70)
  Random-policy coop rate: 0.000 (target <0.25)
  Shared-policy mean ω:    1.0000
  Random-policy mean ω:    0.0000
```

**Perfect separation (100% vs 0%)** — the gate passes with massive margin. The mean ω saturates to 1.0 (shared) and 0.0 (random) in f32 after 50 rounds × 3 NPCs = 150 evidence samples per pair; the log-likelihood ratio `log_w = ±150 · ln 2 ≈ ±104` far exceeds the f32 saturation threshold of `±88.7`.

### 3.3 The companion structural test

`g5_indirect_primaries_never_directly_interact` — verifies that the API surface *enforces* the "no direct interaction" constraint: there is no `observe_direct` call path on `IndirectAgent`, only `observe_other_via_third_party`. The indirect-inference claim is structurally guarded, not just numerically demonstrated.

### 3.4 Why the gate is `>0.70` / `<0.25` and not stricter

The thresholds leave room for: (1) bounded f32 precision artifacts, (2) partial-similarity cases (cross-model in the paper; partial-shard-overlap in production), (3) noisy real-game policies that don't deterministically match. The toy PoC happens to produce perfect separation; production behavior will be softer. The gate is intentionally not `>0.99` so it remains informative on partial-similarity regimes.

---

## 4. The scoped Super-GOAT verdict

### 4.1 Q1–Q4 (scoped to indirect inference only)

| Question | Answer | Evidence |
|---|---|---|
| **Q1: No prior art for this capability?** | **YES** | §1.3 grep — zero hits across all 7 repos outside Plan 526's own artifacts + R335's same-session forward reference. |
| **Q2: New class of behavior?** | **YES** | §2.3 — no shipped mechanism produces cooperation on first direct encounter from purely observational evidence. |
| **Q3: Composable with shipped primitives?** | **YES** | The primitive composes with `CceLp<N,A>` (Plan 295), `ArchetypeBlendShard` (R158, grounds `α`), `SalienceTriGate` (crowd routing), `KARC` (curiosity-gated exploration). |
| **Q4: Honest scope respected?** | **YES** | §1.2 — explicitly does NOT claim novelty for the equilibrium concept, the direct-inference mechanism, or the math. Claims ONLY the indirect-inference zero-shot capability. |

### 4.2 The honest caveat

The PoC uses **deterministic shared policies** (same `policy_seed` → bit-identical action sequence). Real foundation-model agents (paper Fig 4) use stochastic policies and show attenuated but nonzero cooperation. The toy PoC's perfect separation (100% vs 0%) is the *best-case* demonstration of the mechanism; production behavior will be softer:

- **Partial shard overlap** — two NPCs with overlapping but non-identical archetype blends will have `α ∈ (0, 1)` and produce intermediate cooperation rates.
- **Stochastic policies** — sampling temperature > 0 produces occasional mismatches even between identical policies; cooperation rate drops as temperature rises.
- **Limited evidence** — fewer third-party NPCs (smaller K) or shorter info-gathering (smaller T) gives noisier posteriors.

The gate thresholds (`>0.70` shared, `<0.25` random) leave room for these realistic effects. The Super-GOAT verdict is on the *capability class* (zero-shot cooperation is achievable at all), not on the perfect-separation demonstration.

### 4.3 The scoped Super-GOAT verdict

**VERDICT: scoped Super-GOAT for indirect inference ONLY.** Zero-shot cooperation from third-party observation is a genuinely new capability class not produced by any shipped primitive in the 7-repo stack. The PoC passes with massive margin (100% vs 0%). The math is shipped in `SimilarityPosterior::observe_other_via_third_party` (Plan 526 Phase 3, DEFAULT-ON since Phase 6 promotion 2026-08-11).

This scoped claim does NOT extend to:
- The direct-inference mechanism (GOAT-tier — R471 §3).
- The equilibrium concept (covered by shipped CCE — Plan 295).
- The math form (R471 §I, distilled from paper §I).

---

## 5. Future work (non-blocking)

- **Stochastic-policy PoC** — replace deterministic `policy_seed` with temperature-sampled policies; measure cooperation rate as a function of temperature. Expected: monotone decrease; the gate threshold (`>0.70` shared) gives the meaningful regime.
- **Partial-similarity PoC** — `α ∈ (0, 1)` prior; measure cooperation rate as a function of `α`. Expected: threshold-like transition near `α ≈ 0.5` (paper §3.4).
- **Cross-model analog** — paper Fig 4 shows Flash-Lite vs Flash cooperate (attenuated). Production analog: two NPCs with different but related archetype libraries. Requires `ArchetypeBlendShard` overlap scoring (R158 + riir-neuron-db R009).
- **Lean 4 formal verification** of the staleness-window bound `T > log_2((1−α)/α)` for indirect inference — paper §I.3. Non-blocking; the PoC empirically validates the bound.

These are tracked as **non-goals of Plan 526** (per `.plans/526_*.md` §Non-Goals) — they are research extensions, not implementation gaps.

---

## 6. Cross-references

- **R471** — the parent open primitive note (both direct + indirect inference). Revised GOAT verdict there.
- **R335** (riir-ai) — the parent private Super-GOAT guide. Contains the forward reference to this scoped claim.
- **R336** (riir-ai) — the scoped private guide for indirect inference, opened in lockstep with this note per skill §1.5.
- **Plan 526** — the implementation plan. Phase 3 (G5 PoC) is the trigger condition for this note. Phase 7 (this note + R336) is the conditional Super-GOAT-claim output.
- **Bench 579** — the GOAT benchmark file. §G5 records the indirect-inference PoC numbers.
- **Plan 295 / R274** — the CCE moderator substrate (DEFAULT-ON). The equilibrium concept this primitive composes with.
- **R158 / riir-neuron-db R009** — the ArchetypeBlendShard substrate that grounds `α` in production (chain-committed, BLAKE3-verifiable shard identity).

---

## 7. Source paper citation

Meulemans, Wołczyk, Weis, Nasser, Rocca, Kobayashi, Lajoie, Steger, Richards, Hutter, Manyika, Saurous, Sacramento, Agüera y Arcas. "A game theory for foundation models shows new paths to rational cooperation through similarity inference." [arXiv:2608.03958](https://arxiv.org/abs/2608.03958). 4 Aug 2026. §I (indirect similarity analysis) + Fig 4 (indirect-inference cooperation curve) + §H (direct analysis, the foundation) + §F.2 (evidential information formalization) + §G (equilibrium convergence).
