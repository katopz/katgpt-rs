# Research 315: Neural Scaling Universality — Exponents Fixed, Coefficients Actionable

> **Source:** [Neural Scaling Universality: If Exponents Are Fixed, Time to Understand Coefficients](https://arxiv.org/abs/2606.25008) — Yizhou Liu, Jeff Gore (MIT), 23 Jun 2026 (17-page position paper, 6 figures)
> **Date:** 2026-06-26
> **Status:** Done
> **Related Research:** 131 (UNSL — sibling scaling-law theory), 222 (Spectral Scaling Laws of Muon — direct predecessor, *empirical* power laws by layer; this paper is the *mechanistic* companion), 295 (AC-GPT — explicitly defers the sigmoid-vs-softmax justification to here), 240 (CGSP — confirms sigmoid scaling as codebase default)
> **Related Plans:** 254 (Spectral Budget Router — already implements Research 222's coefficient-space navigation), 297 (PersonalityWeightedComposition — coefficient-space drift), 321 (CommittedFieldBlend — coefficient-space commit)
> **Classification:** Public (katgpt-rs — modelless inference justification)

---

## TL;DR

Liu & Gore (MIT) argue a position: the **exponents** of neural scaling power laws are **universal**, locked in by three generic mechanisms — **1/3 time scaling from softmax nonlinearity, inverse-width scaling from representational superposition, inverse-depth scaling from transformer-layer ensemble averaging**. They argue the actionable lever for near-term performance is therefore not changing exponents (impossible inside this universality class) but **navigating coefficient space** — coefficients are sensitive to data and architecture, and they determine the compute-optimal frontier and optimal model shape.

For this codebase, the paper is the third in a scaling-law trilogy (UNSL R131 → Spectral Scaling Laws R222 → this). Its distinct contribution is **mechanistic**: it names *why* exponents take the values they do. That gives us two things:

1. **Theoretical justification for the AGENTS.md sigmoid-not-softmax rule.** The codebase mandates sigmoid (constraint #2); `fast_sigmoid` / `simd_sigmoid` / G6 gate enforce it everywhere. The paper claims softmax's strong nonlinearity *fixes* the 1/3 time exponent. **By using sigmoid instead of softmax in blending gates, projection gates, and personality blends, the runtime deliberately operates outside the paper's "fixed-exponent" universality class.** This is the missing theoretical grounding for the rule — Research 295 (AC-GPT) explicitly flagged this gap.

2. **Theoretical justification for the coefficient-space navigation strategy.** `CommittedFieldBlend` (Plan 321), `PersonalityWeightedComposition` (Plan 297), and adapter routing already move the runtime in blend-weight / personality-weight / archetype-commitment space. Per Liu-Gore, this is *exactly* the actionable lever — coefficients are what you can change; exponents are not. The existing committed-blend primitives are the modelless inference instantiation of the paper's recommendation.

**Distilled for katgpt-rs (modelless, inference-time):**
This paper ships no new mechanism; it ships a *frame*. The frame re-casts two existing design choices (sigmoid-not-softmax, committed-blend coefficient navigation) as theoretically-grounded rather than stylistic. The one small implementable angle is a **universality-class canary** in the CUCG bench: a synthetic test that demonstrates the sigmoid gate produces a different (gentler) effective nonlinearity than a softmax gate of equivalent temperature, with the predicted consequence on tail-decay of the corresponding "loss-like" quantity. That's a documentation/canary task, not a new primitive.

---

## 1. Paper Core Findings (position paper — argued, not measured)

### 1.1 The three universal exponent mechanisms

| Exponent | Mechanism | Robustness claim |
|---|---|---|
| **Time: 1/3** | Strong nonlinearity of softmax — gradient vanishes as `e^{-δ}` near a saturation boundary, but the mass being moved scales as `δ^2`, yielding a 1/3 exponent in time-to-saturation | Holds across data structures; locked by softmax specifically |
| **Width: inverse** | Representational superposition — wider models pack more features into the same latent space; the cost of interference scales inversely with width | Holds across architectures with superposition |
| **Depth: inverse** | Ensemble averaging of transformer layers — depth acts as an ensemble of weak estimators; variance reduction scales inversely with depth | Holds for transformer-style stacked layers |

### 1.2 Coefficients are the actionable lever

Exponents are claimed to be locked by generic mechanisms; coefficients are claimed to be sensitive to data + architecture. Practical quantities that depend on coefficients:
- **Optimal model shape** (depth × width for a given compute budget)
- **Compute-optimal frontier** (Chinchilla-style allocation)
- The constants of the per-axis power laws (loss vs. time / size / compute)

The paper's recommendation: stop trying to change exponents (you can't, inside the universality class); focus engineering effort on coefficient-moving interventions (data curation, architecture shapes, training recipes).

### 1.3 What "universality class" means here

The paper borrows the physics notion (cf. critical exponents being universal within a class). Inside the (softmax + transformer + superposition) class, exponents are universal; coefficients vary. To get *different* exponents, you must change one of the three load-bearing mechanisms — most relevantly for us: **the nonlinearity**.

---

## 2. Distillation

### 2.1 The sigmoid escape valve (modelless, validates constraint #2)

The AGENTS.md rule "Use sigmoid not softmax" is enforced across the codebase:

- `katgpt-rs/crates/katgpt-core/src/simd/` — `fast_sigmoid`, `simd_sigmoid_inplace`, `simd_sigmoid_tanh_clamp_inplace`
- `katgpt-rs/crates/katgpt-core/src/analytic_lattice/decoder.rs` — `direction_vector_decode` uses `fast_sigmoid(z * temperature)`
- `katgpt-rs/crates/katgpt-core/src/bridge/mod.rs` — `select_action` uses `fast_sigmoid(dot)`, comment: "zero allocation, fixed-size. Uses `sigmoid(dot())` — never softmax"
- `katgpt-rs/benches/cucg_goat.rs` — **G6: sigmoid never softmax (0 softmax hits)** — statically enforced
- `katgpt-rs/crates/katgpt-core/benches/bench_322_phase_rotation_goat.rs` — G6 proves sigmoid at `dot=0` returns `0.5` (would be `1.0` under softmax of a single value)
- `CommittedFieldBlend` / `PersonalityWeightedComposition` — both compute blend weights via sigmoid, never softmax

**The Liu-Gore claim closes a documentation gap.** Sigmoid's nonlinearity is gentler than softmax's:
- Softmax is **scale-sensitive** (logits × T → distribution sharpens exponentially) and produces a partition-of-unity (one winner takes all in the limit).
- Sigmoid is **scale-bounded per-coordinate** (each output ∈ (0,1) independently, no partition constraint) and produces a soft `2σ(x)−1 ∈ (−1, +1)` symmetric gate used pervasively here.

If the paper is right that softmax's nonlinearity is what fixes the 1/3 time exponent, then **swapping softmax→sigmoid in the gates (which we already do) places us in a different universality class**, with different (and per the paper's logic, not 1/3-locked) effective exponents. We are not claiming the sigmoid class is *better* (the paper makes no such claim and neither do we); we are claiming it is *different*, and the difference is principled rather than stylistic.

This is **the** missing justification Research 295 deferred:

> `katgpt-rs/crates/katgpt-core/benches/bench_313_ac_prefix_goat.rs:132-133`
> "log-softmax over vocab — AGENTS.md 'sigmoid not softmax' rule applies to blending gates, not the LM head."

Per Liu-Gore, the LM head softmax is exactly where the 1/3 exponent is fixed; the blending-gate sigmoid is exactly where we already escape it. The comment is now theoretically grounded.

### 2.2 Coefficient-space navigation (modelless, validates Plans 297 + 321)

Liu-Gore: coefficients are the actionable lever; exponents are not. The codebase's committed-blend stack is the inference-time instantiation of that recommendation:

| Primitive | Coefficient axis it navigates | Lock-in status |
|---|---|---|
| `PersonalityWeightedComposition` (Plan 297) | `w[N]` personality weights (sigmoid-gated drift: `Δw = α·(R_obs−R_exp)·d_recent`) | Mutable per-tick (drift rule) |
| `CommittedFieldBlend` (Plan 321) | `pi[N]` blend logits, BLAKE3-committed once from trajectory summary | Frozen until major personality event (re-commit) |
| Adapter routing (dMoE / Dynamic Pair / Polytope, R091/R161/R227) | which frozen expert per token | Per-token, modelless |
| Freeze/thaw (`MerkleFrozenEnvelope` in riir-neuron-db) | which committed snapshot is active | Atomic version swap |

All four navigate coefficient space without changing the (universal) exponent structure. The paper's recommendation is, in effect, *already our architecture*.

The reframe is consequential for prioritization: **coefficient-space navigation is not a hack around scaling limits — it is the theoretically-correct lever per Liu-Gore.** Resources spent improving committed-blend granularity, archetype-blend commit frequency, or personality drift stability are spending on the actionable axis the paper identifies, not on a workaround.

### 2.3 Latent-space reframing (mandatory per workflow step 3)

Re-cast the paper's mechanism on each Super-GOAT factory module:

- **HLA per-NPC latent state** (`katgpt-rs/crates/katgpt-core/src/sense/`): The 5 synced affect scalars (valence/arousal/desperation/calm/fear) are **coefficient outputs** of the latent projection, not exponent-determining quantities. The nonlinearity that determines the HLA "effective exponent" is the sigmoid in `evolve_hla`, not softmax. This puts HLA's per-NPC affect dynamics in a different universality class than a softmax-mixture-of-emotions would be — and per the paper, that's where actionable coefficient variation lives.
- **`latent_functor/`** (`riir-ai/crates/riir-engine/src/latent_functor/`): Functor applications are stage-gated via sigmoid (zone gating, coherence re-estimation, k-selector). The paper's "inverse-depth" mechanism assumes transformer-layer ensembling; **a sigmoid-gated functor stack is not a transformer stack**, so the inverse-depth exponent does not directly apply. Again, deliberate universality-class divergence.
- **`cgsp_runtime/`**: Curiosity signals drive exploration. The exploration-exploitation tradeoff has its own scaling, but the paper's softmax-fixed 1/3 exponent does not bind it — curiosity-driven exploration is not softmax-saturated training.
- **LatCal fixed-point** (`riir-chain/src/encoding/latcal*.rs`): LatCal is deterministic fixed-point arithmetic, not a softmax-trained model. It has no exponent in the paper's sense. **It is the canonical raw-scalar bridge at the sync boundary** (per AGENTS.md latent-vs-raw rules) — coefficients only, no exponents.
- **`NeuronShard` / `MerkleFrozenEnvelope`** (`riir-neuron-db/src/`): Frozen snapshots are coefficient-space points, not exponent-space moves. Freezing/thawing navigates coefficient space atomically. Consolidation (Raven/δ-Mem) moves shards along a coefficient trajectory; the paper's frame says this is the right thing to be doing.
- **DEC Stokes operators** (`katgpt-rs/crates/katgpt-core/src/dec/`): `d∘d=0` is exact linear-algebra identity, no exponent. The DEC substrate is pure structure; the coefficient variation lives in the cochain values, not in operator dynamics.

**The pattern:** every modelless inference primitive we ship either operates on coefficient space directly (blend, route, freeze) or is structurally outside the paper's universality class entirely (sigmoid gates, linear DEC operators, deterministic LatCal). The paper's "exponents are fixed" claim therefore does not constrain anything we actually do — and the "coefficients are actionable" claim endorses everything we already do.

### 2.4 Fusion — what novel combination does this enable?

**Fusion A — "Universality-class canary" bench (small, ship-worthy, GOAT-grade):**

A synthetic bench that constructs two equivalent blending systems — one softmax-gated, one sigmoid-gated — and measures a "loss-like" tail-decay quantity under a controlled input distribution. The bench *demonstrates* (does not prove) that the two gates produce different effective decay exponents. This is a modelless, pure-arithmetic documentation of the sigmoid escape valve. Lives in `katgpt-rs/benches/` next to `cucg_goat.rs`'s G6 gate. **This is the only implementation-worthy fusion from this paper.** No new primitive, just a canary that makes the existing G6 gate quantitatively defensible rather than just rule-enforced.

**Fusion B — "Coefficient navigation ledger" (documentation only, no code):**

A `.docs/` note cross-referencing Plan 297 (PersonalityWeightedComposition), Plan 321 (CommittedFieldBlend), Plan 254 (Spectral Budget Router), and the freeze/thaw envelope, framed explicitly as "the four modelless coefficient-navigation mechanisms per Liu-Gore 2606.25008". This re-prioritization rationale is the paper's main transferable value.

**Fusion C (speculative, NOT planned here) — nonlinearity→exponent calculator:**

If softmax gives 1/3 and the paper's argument is "nonlinearity fixes exponent", one could in principle build a small lookup / closed-form calculator: given a bounded nonlinearity φ (sigmoid, tanh, GELU, etc.), predict the effective exponent. This connects to Research 222 (Muon spectral exponents per layer). **This is riir-train territory** (it requires actually training models to verify the prediction) and is mentioned only to close the fusion search — do not plan it here.

---

## 3. Verdict

**GOAT** — provable gain in conceptual clarity and theoretical justification of existing design choices; no new capability class.

| Criterion | Assessment |
|---|---|
| **Strengthens moat?** | ⚠️ Mild — the sigmoid-not-softmax rule gains a theoretical justification (universality-class escape per Liu-Gore); the committed-blend stack gains a re-prioritization rationale (coefficient-space navigation is the actionable lever). Neither is a new moat; both harden existing moats. |
| **Uses existing traits?** | ✅ Yes — fully validates `fast_sigmoid` / `simd_sigmoid` pervasive usage and the `CommittedFieldBlend` / `PersonalityWeightedComposition` coefficient-navigation stack. |
| **Modelless?** | ✅ Yes — the distilled angles are documentation, canary benches, and re-prioritization rationale. No training, no backprop. The paper's training-side content (compute-optimal allocation, model shape) is acknowledged as riir-train territory. |
| **Commercial alignment** | ✅ Public-engine justification — katgpt-rs. The sigmoid escape valve and the coefficient-navigation frame are generic modelless-inference arguments, no game/chain/shard IP. |
| **Perf impact** | ❌ None directly — this is a justification note, not a speedup. The Fusion A canary would measure but not improve perf. |
| **Proof of gain** | ⚠️ Conceptual, not empirical. The paper is a position paper (argued, not measured). Our claim is "existing design choices are now theoretically grounded", which is a clarity gain, not a measurable speedup. |

**One-line reasoning:** The paper is the third in a scaling-law trilogy (UNSL R131 → Spectral Scaling Laws R222 → this). R131 was "Marginal — research-only"; R222 produced a GOAT-graded Plan 254 (Spectral Budget Router). This paper produces no new primitive but validates two existing design choices on first-principles grounds — the sigmoid-not-softmax rule and the committed-blend coefficient-navigation stack — and adds a small implementable canary (Fusion A) to the CUCG bench. That's a GOAT, not a Super-GOAT.

**Why not Super-GOAT:** fails novelty gate Q2 (new class of behavior). The paper does not unlock any capability the codebase did not already have; it grounds choices already made. Q1 (no prior art for the *justification*) passes; Q3 (mild selling point — "our sigmoid rule is a universality-class escape") passes; Q4 (force multiplier — connects Plans 254, 297, 321, the AGENTS.md rule, and Research 222) passes. But Q2 fails, and Q2 is a hard gate. No private guide warranted.

**Why not Pass / Gain:** the paper is *not* training-only (§3.5 modelless-unblock protocol applies). The sigmoid escape valve is a modelless inference justification; the coefficient-navigation frame is a modelless re-prioritization. These are GOAT-grade (clarity gain, force-multiplier across multiple existing pillars), not Gain-grade (which would be a single small optimization).

### Action items

- [ ] **T1 (optional, low priority):** Add a "universality-class canary" sub-bench to `katgpt-rs/benches/cucg_goat.rs` extending G6 — construct softmax-gated vs sigmoid-gated equivalent blend systems and report the tail-decay difference. This makes G6 quantitatively defensible rather than just rule-enforced. No new primitive; pure bench addition.
- [ ] **T2 (documentation):** Add a one-paragraph note to `katgpt-rs/README.md` (or `.docs/`) citing this paper as the theoretical grounding for the sigmoid-not-softmax rule and the committed-blend coefficient-navigation strategy. Low effort, high documentation value.
- [ ] **T3 (cross-reference update):** Update `katgpt-rs/crates/katgpt-core/benches/bench_313_ac_prefix_goat.rs:132-133` comment to reference this research note as the theoretical grounding, closing the deferred Research 295 thread.

---

## 4. What This Is NOT

- **Not a riir-train deferral.** The paper is about training scaling laws, but its modelless inference implications (sigmoid escape valve, coefficient-space navigation) are in-scope here. The training-side content (compute-optimal allocation, optimal model shape) is acknowledged as riir-train but not deferred — we note it and move on, no riir-train plan needed.
- **Not a new primitive.** No new module, no new trait, no new feature flag. Fusion A is a bench addition only.
- **Not a Super-GOAT.** Fails Q2 (no new capability class). The existing `CommittedFieldBlend` / `PersonalityWeightedComposition` / `fast_sigmoid` infrastructure is what makes the paper's recommendation actionable; this paper does not add to that infrastructure, only justifies it.
- **Not a claim that sigmoid is *better* than softmax.** The paper makes no such claim and neither do we. The claim is that sigmoid is *in a different universality class* per the paper's argument — different, not strictly better. Empirical comparison of which class is better for which inference tasks is riir-train territory.
- **Not a Chinchilla-style compute-optimal allocation recipe.** The paper argues coefficients matter but does not provide a recipe for navigating them at training time. Our coefficient-navigation is inference-time (blend, route, freeze) and is already shipped.

---

## 5. Cross-references

- **katgpt-rs/.research/131_UNSL_Unified_Neural_Scaling_Laws.md** — sibling scaling-law theory; verdict "Marginal — research-only". This paper's mechanistic framing is sharper; UNSL's functional-form framing is broader.
- **katgpt-rs/.research/222_Spectral_Scaling_Laws_Muon_Adaptive_Inference.md** — direct predecessor. R222 ships the *empirical* power laws per layer type (Muon momentum singular values follow `σ_q(M) = c_q · M^(-α_layer)`); this paper is the *mechanistic* companion explaining why those α's take the values they do. R222 → Plan 254 (Spectral Budget Router, shipped, GOAT). This paper does not modify Plan 254; it justifies its premise.
- **katgpt-rs/.research/295_AC_GPT_Arbitrary_Conditionals_Prefix.md** — explicitly defers the sigmoid-vs-softmax rule's theoretical grounding. This paper is the deferred grounding.
- **katgpt-rs/.research/240_SGS_Curiosity_Guided_Self_Play.md §1.3** — confirms sigmoid scaling (`R_C = R_0 + (A − R_0) · σ(B · ln(C / C_mid))`) as the codebase default. Consistent with the "different universality class" claim here.
- **katgpt-rs/.plans/254_spectral_budget_router.md** — coefficient-space navigation at the NS-depth layer (shipped, GOAT).
- **katgpt-rs/.plans/297_personality_weighted_composition.md** — coefficient-space navigation at the personality-weight layer (shipped, GOAT).
- **katgpt-rs/.plans/321_sampling_invariant_per_entity_moe_primitive.md** — coefficient-space navigation at the archetype-commitment layer (shipped, GOAT).
- **riir-neuron-db/src/freeze.rs** (`MerkleFrozenEnvelope`) — atomic coefficient-space snapshot swap.
- **riir-chain/src/encoding/latcal*.rs** — deterministic fixed-point raw-scalar bridge; no exponent in the paper's sense (pure coefficient space).

---

## TL;DR

Third paper in the scaling-law trilogy (UNSL R131 → Spectral R222 → this). Argued (not measured) position: scaling exponents are universal (locked by softmax/superposition/ensembling); coefficients are the actionable lever. For this codebase, that means two things. **(1) The AGENTS.md sigmoid-not-softmax rule — already enforced everywhere via `fast_sigmoid` and the CUCG G6 gate — is a deliberate universality-class escape per this paper's argument, not a stylistic preference; this closes the deferred Research 295 thread. (2) The committed-blend stack (Plans 297, 321, 254) and freeze/thaw envelope are the modelless inference instantiation of the paper's "navigate coefficient space" recommendation — what we already do is what the paper says to do.** Verdict: **GOAT** — conceptual clarity and theoretical grounding for existing design choices, plus one small canary-bench action item. Not Super-GOAT (no new capability class). No private guide warranted. No new primitive ships; the existing infrastructure *is* the implementation.
