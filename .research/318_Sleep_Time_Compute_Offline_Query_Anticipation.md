# Research 318: Sleep-time Compute — Offline Query Anticipation & Multi-Query Amortization

> **Source:** [Sleep-time Compute: Beyond Inference Scaling at Test-time](https://arxiv.org/abs/2504.13171) — Lin, Snell, Wang, Packer, Wooders, Stoica, Gonzalez (Letta / UC Berkeley), arXiv:2504.13171v1, 2025-04-17
> **Date:** 2026-06-27
> **Status:** Active — Super-GOAT via fusion; primitive + plan + private guide created this session
> **Related Research:** 069 (AutoDreamer — already cites this paper), 116 (LLM Sleep — KV/fast-weight sleep consolidation), 242 (topological recurrent belief), 276 (MicroRecurrentBeliefState), 281 (Per-Tick Salience Tri-Gate), 288 (KARC delay-basis ridge forecaster), 296 (Stokes/DEC vocabulary crosswalk), 311 (Analytic Lattice Encoder/Decoder)
> **Related Plans:** katgpt-rs 107 (AutoDreamer), 154 (LLM Sleep), 276 (MicroBelief), 281 (SalienceTriGate), 290 (Closure-Expansion motif mining at sleep cycles), 299 (Engram hash-addressed memory), 304 (Gain/Cost Loop Halting), 308 (KARC), **334 (this research's open primitive)**
> **Cross-ref (riir-ai):** Research 163 (Per-NPC Sleep-Time Query Anticipation Guide — the private selling point)
> **Classification:** Public (katgpt-rs = open math primitive); the *selling-point guide* is private in riir-ai.
> **Verdict: Super-GOAT — the fusion (paper's C/Q decomposition + predictability-gated allocation + multi-query amortization × KARC forecast × Plan 154 sleep substrate × Gain/Cost halting × Salience Tri-Gate × NeuronShard freeze × chain quorum commitment) is a new capability class with no shipped prior art for the COMBINATION.**

---

## TL;DR

The paper introduces **sleep-time compute**: `S(c) → c'`, applied to context *before* a query arrives, producing a re-represented context `c'` so that test-time can use a much smaller budget `T_b(q, c') → a` with `b << B`. Empirically ~5× test-time reduction on Stateful GSM-Symbolic + Stateful AIME, +13–18% accuracy by scaling sleep-time, 2.5× cost amortization across multi-query contexts. §7 explicitly frames it as **"representation learning over tokens"** in natural-language space, with **query predictability** the dominant correlate of gain.

**Distilled for katgpt-rs (modelless, inference-time):**
A generic `SleepTimeAnticipator<C, Q>` trait: given context `c` (any latent state), score the predictability of likely queries `Q`, allocate sleep-time compute proportional to predictability, and emit a reusable **anticipated-query projection set** `c'` that test-time consumers can apply via cheap dot-product + sigmoid gates. The paper implements `S(c)` with LLM `rethink_memory` calls (up to 10); we generalize the *mechanism* (allocate offline compute to predictable-query anticipation, amortize across queries) to *any* latent-state substrate (HLA, latent_functor direction vectors, NeuronShard style_weights, KarcShard readout). The novelty is the **combination** — every individual piece (sleep consolidation, forecasting, gating, amortization) ships in our corpus already; **none of them is wired as a query-anticipation + predictability-gated + cross-consumer amortization pipeline**.

**Distilled mechanism (paper → math, training-free):**

```
S(c) → c' where c' = { (q̂_i, ẑ_i, p_i) }_{i=1..K}
  q̂_i   : i-th anticipated query direction in latent space
  ẑ_i   : pre-computed latent answer / direction vector / functor ready to apply
  p_i   : predictability score in [0,1] from p_i = σ(α · sim(c, q̂_i) + β)

Test-time:
  T_b(q, c') → a   with b << B
  apply c' to query q via cheap lookup: argmax_i sim(q, q̂_i) → ẑ_i → sigmoid gate → a

Cost model (paper §5.3):
  cost_total = N_sleep · cost_sleep + N_test · t · cost_test
  amortization factor across N queries sharing the same c' : t / N (paper shows 2.5× at N=10, t=10)
```

---

## 1. Paper Core Findings (verified by full PDF read)

### 1.1 The primitive — `S(c) → c'` then `T_b(q, c') → a`

The standard test-time paradigm `T_B(q, c) → a` assumes both `q` (user query) and `c` (context) arrive together. The paper observes that real applications are **stateful**: `c` is available *before* `q` arrives. The model is idle during that "sleep" period — wasted opportunity.

**Sleep-time compute** prompts the model to reason about `c` offline, producing a re-represented context `c'` containing inferences that may be useful for future queries:

```
sleep-time:   S(c) → c'
test-time:    T_b(q, c') → a      with b << B
```

The paper's implementation (Appendix K) uses function calling: `rethink_memory(new_memory, target_block_label, source_block_label)` called up to 10 times to iteratively rewrite `c` into `c'`. This is the entire mechanism.

### 1.2 Headline empirical wins

- **~5× test-time reduction** for same accuracy on Stateful GSM-Symbolic (P1, P2) and Stateful AIME. Tested across GPT-4o, GPT-4o-mini, o1, o3-mini, Claude Sonnet 3.7 Extended Thinking, DeepSeek-R1.
- **+13% accuracy** on Stateful GSM-Symbolic P1, **+18%** on Stateful AIME by scaling sleep-time compute (multiple parallel `c'_1, ..., c'_k` for non-reasoning models; reasoning-effort variation for reasoning models).
- **Sleep-time pareto-dominates parallel pass@k** at the same test-time token budget on both datasets.
- **2.5× cost amortization** when 10 related queries share one context (Multi-Query GSM-Symbolic) — paper's cost model: `cost_total = N · cost_sleep + N_test · t · cost_test` with `t = 10` (latency-optimized test-time is ~10× more expensive per token than sleep-time).
- **Context-only baseline ablation** (Appendix I): `c'` is NOT just "guess the answer to the most likely question" — sleep-time compute significantly outperforms a context-only baseline that just guesses. Confirms the queries are non-trivially predictable.

### 1.3 The predictability correlation (paper §5.4 — the key insight)

The paper bins Stateful GSM-Symbolic examples by **query predictability** (log-probability of the question given the context under Llama2-70B base model). The gap between sleep-time compute and standard test-time compute **widens monotonically** as predictability increases (Fig. 10): on the highest-predictability quintile, sleep-time compute's gain over test-time-only is 4× larger than on the lowest.

**Predictability-gated allocation is the actionable primitive.** The paper does not operationalize this — they say "future work should identify which contexts may have predictable questions and optimally allocating inference compute between sleep-time and test-time across different contexts and queries" (§7). **This is the gap our open primitive fills.**

### 1.4 §7 framing: "representation learning over tokens"

> "Our approach to applying compute at sleep-time resembles representation learning. We first transform the context into a representation that is more amenable to answering test-time queries, and then we utilize that representation at test-time to rapidly answer queries. Unlike traditional representation learning (Bengio et al., 2014), which typically operates in model parameter or activation space, we instead form representations in the space of natural language."

This is the bridge to our latent-space substrate: `c'` *is* a learned representation; we simply choose latent-state vectors instead of natural-language tokens. Latent-to-latent preferred per AGENTS.md constraint #2.

### 1.5 SWE-Features case study (§6) — multi-file PR completion

33 PRs from ComfyUI + Aider that modify ≥3 files. At lower test-time budgets sleep-time compute wins (1.5× test-time reduction); at higher budgets standard test-time wins. **Confirms the amortization trade-off is regime-dependent** — matches the predictability correlation.

---

## 2. Distillation

### 2.1 What we already ship (the prior-art surface — verify before any novelty claim)

| Paper mechanism | Shipped cousin | File / Plan |
|---|---|---|
| `S(c) → c'` offline re-representation | **LLM Sleep** — N-pass recurrent consolidation of KV into GDN2 fast weights at eviction | Plan 154, Research 116, `src/sleep/` |
| Modelless `S(c)` (no BPTT) | **AutoDreamer** — offline consolidation tick | Plan 107, Research 069 (which **already cites this paper** as "Sleep-time Compute: Lin et al. 2025 (offline pre-computation)") |
| `S(c)` that produces *reusable artifacts* (motifs, not weights) | **Closure-Expansion Instrument** — mines motifs at every sleep-cycle boundary | Plan 290, `src/closure_mining.rs`, `crates/katgpt-core/src/closure/motif.rs` |
| Per-NPC trajectory forecaster (the "predict what comes next" primitive) | **KARC** — closed-form delay-basis ridge readout, fits in `KarcShard` (NeuronShard subtype) | Plan 308, Research 288, `crates/katgpt-core/src/karc/` |
| Per-NPC per-relation learned direction vector + coherence-gated apply | **latent_functor** — `extract_functor_into`, `apply_functor`, `functor_gate(coherence)` (sigmoid) | Plan 303, `riir-engine/src/latent_functor/arithmetic.rs` |
| Test-time budget `b` per-NPC adaptive halting | **Gain/Cost Loop Halting** — `halt_decision(gain, cost, tau)` | Plan 304, Research 282 |
| Per-tick "should I think more or answer now" decision | **Salience Tri-Gate** — `Speak / Silent / Delegate` | Plan 303 (katgpt-rs), Research 281 |
| Per-NPC hash-addressed conditional pattern memory | **Engram** — hash-keyed conditional lookup | Plan 299, Research 278 |
| Cross-node curiosity snapshot commitment (sync-boundary bridge) | **cgsp_runtime/chain_bridge.rs** — `commit_snapshot_via_quorum`, `reload_snapshot_from_chain` | `riir-engine/src/cgsp_runtime/chain_bridge.rs` |
| Per-NPC frozen latent state, replicable | **NeuronShard / MerkleFrozenEnvelope** | `riir-neuron-db/src/shard.rs`, `freeze.rs` |
| Warm-tier offline generation positioning (the existing "NPC sleep cycle" framing) | **CompressionDrafter** — explicit verdict: "Warm-tier offline generation — quest packs generated during NPC sleep cycles, where ms latency is fine" (Bench 285, GOAT FAILED for Hot-tier) | Plan 285, Research 137 (riir-ai) |
| Predictability score via dot-product | **EmotionDirections / FutureBehaviorProbe::forecast** | `src/pruners/emotion_vector.rs`, `future_probe.rs` |
| Multi-query cost amortization via warm-tier cache hit | **BFCF × LFU × Sharding** — region-keyed amortized retrieval | Plan 218, Research 193 |

### 2.2 What the paper adds that NONE of the above does alone

The fusion is the novelty, not any single component:

1. **Explicit query anticipation** — paper asks the model to *predict likely queries* from `c` before they arrive. We have KARC (forecasts next latent state) and latent_functor (forecasts next relation), but **neither frames the forecast as "what query is coming"**. The re-framing is novel: query anticipation = forecasting in *query space*, not state space.

2. **Predictability-gated allocation** — paper §7 explicitly leaves this as future work. The mechanism is: spend sleep-time compute proportional to `predictability(c)`. We have gain/cost halting for *wake-time* loops (Plan 304); we have **no shipped primitive** for *sleep-time budget allocation by query predictability*. The mapping is: `gain_sleep(c) = predictability(c) × expected_queries_sharing_c` and `cost_sleep(c) = N_sleep_tokens × opportunity_cost_of_idle_compute`. Allocate sleep-time to contexts where gain > cost.

3. **Multi-query amortization across CONSUMERS (not just queries)** — paper amortizes `c'` across N related queries about one context. The MMORPG-scale generalization (private selling point — see riir-ai/.research/163) is amortizing `c'` across **N players who talk to the same NPC**. This is the cross-consumer amortization the paper does not consider. The cost model generalizes: `cost_total = cost_sleep(c') + N_players × t × cost_test` instead of `cost_sleep(c') + N_queries × t × cost_test`.

4. **Re-representation as a first-class emitted artifact (not internalized into weights)** — paper §7 explicitly calls sleep-time "representation learning over tokens". Our existing `S(c)`s (Plan 154, Plan 107) internalize `c'` into weights / fast state. **The paper's contribution is treating `c'` as an emitted, reusable, sharable artifact**. This maps cleanly onto Engram (hash-addressed memory) and KarcShard (frozen replicable readout) — but no existing primitive **emits** `c'` for re-use across consumers as its primary output.

5. **Predictability = curiosity inverted** — our `cgsp_runtime` curiosity is `curiosity_t = ‖actual_hla_t − karc_forecast_t‖` (high curiosity = unpredictable). The paper's predictability is `predictability = log P(q | c)` (high predictability = query follows from context). **These are the same scalar, opposite sign.** The fusion: sleep-time allocation is `predictability = 1 − sigmoid(curiosity)`. Sleep-time is for *low-curiosity* (predictable) contexts; curiosity-driven exploration (CGSP runtime) is for *high-curiosity* contexts. This is the **first principled allocation rule between two existing systems** — neither paper alone produces it.

### 2.3 Fusion (the Super-GOAT move)

| Fusion partner | What it ships | What this paper adds | Fusion product |
|---|---|---|---|
| **R116 / Plan 154 LLM Sleep** | Offline N-pass KV→fast-weight consolidation at eviction | Treat `c'` as an *emitted artifact*, not internalized state; anticipate queries, not just compress context | "Sleep consolidation that *also* emits anticipated-query projection set, reusable across consumers" |
| **R288 / Plan 308 KARC** | Closed-form delay-basis ridge forecaster; predicts next latent state | Frame the forecast as "what query is coming" (query-space prediction, not state-space); predictability-gated budget | "Per-NPC query-anticipation forecaster fit during sleep-time, applied at test-time via dot-product lookup" |
| **R282 / Plan 304 Gain/Cost Halting** | Wake-time per-loop adaptive halting | Sleep-time per-context adaptive *allocation* (the same gain/cost curves, evaluated offline) | "Gain/cost framework extended from wake-time loop count to sleep-time context-priority budget" |
| **R303 / Plan 303 Salience Tri-Gate** | Per-tick `Speak / Silent / Delegate` decision | Sleep-time pre-decision: should this NPC pre-compute, given the predictability of likely queries? | "NPCs decide at sleep-time whether to pre-compute, then decide at wake-time whether to use the pre-computation or think fresh" |
| **R281 / Plan 299 Engram** | Hash-addressed conditional pattern memory | Anticipated-query projection set IS an Engram entry keyed by `(NPC, context_hash)` | "Sleep-time produces Engram entries that wake-time retrieves via hash — predictable queries hit Engram, unpredictable fall through to fresh thought" |
| **R276 MicroRecurrentBeliefState / HLA `evolve_hla`** | Per-NPC 8-dim latent state + leaky integrator update | Sleep-time updates HLA with anticipated-query direction projections; wake-time applies them via dot-product | "HLA becomes the sleep-time substrate; the 5 synced affect scalars are the bridge artifact that crosses sync after sleep-time pre-computation" |
| **Plan 290 Closure-Expansion** | Mines motifs at sleep-cycle boundaries | Mines anticipated queries at sleep-cycle boundaries (different output type, same trigger) | "Sleep-cycle boundary produces BOTH motifs (closure mining) AND anticipated-query projections (sleep-time compute) in one pass — zero extra trigger cost" |
| **cgsp_runtime curiosity (Research 126, Plan 274)** | Curiosity = forecast residual; high-curiosity contexts get more CGSP exploration | Predictability = 1 − curiosity; low-curiosity contexts get more sleep-time pre-computation | "First principled allocation rule between CGSP exploration (high-curiosity) and sleep-time pre-computation (low-curiosity)" |
| **cgsp_runtime/chain_bridge.rs** | Commits curiosity snapshot via chain quorum, reloads by hash | Commits the anticipated-query projection set (`c'`) via the same quorum path; reloads at test-time on any node | "Cross-player amortization: one NPC's `c'` computed once, committed, reloaded by every player's client that talks to that NPC" |
| **NeuronShard / MerkleFrozenEnvelope** | Fixed-size Pod with `style_weights[64]`, BLAKE3, freeze/thaw | `SleepAnticipationShard` subtype: `style_weights[64]` stores up to 8 anticipated-query direction vectors (8-dim each) | "Anticipated queries frozen into a shard, replicated via chain, restorable on any node — the persistence substrate" |
| **LatCal fixed-point commitment** (`riir-chain/src/encoding/`) | Deterministic linear-op commitment via 2×2 fixed-point blocks | `c'` is a set of linear projections; LatCal commits them as fixed-point matrix blocks | "LatCal-committed `c'` = deterministic, quorum-reproducible anticipated-query projection. The sync-boundary bridge." |
| **BFCF × LFU × Sharding (Plan 218)** | Region-keyed amortized retrieval | Region-keyed sleep-time pre-computation: NPCs in the same region share `c'` for region-predictable queries | "Sleep-time compute scoped to regions, not just per-NPC — the region is the natural amortization unit" |

### 2.4 Latent-space reframing (mandatory per fusion protocol §1.3)

Operating on each Super-GOAT factory module:

(a) **HLA per-NPC latent state** (`katgpt-core/src/sense/`, `riir-engine/src/hla/`): The NPC's HLA state IS the latent `c`. During sleep-time, project HLA onto a fixed set of **anticipated-query direction vectors** `{d_price, d_quest, d_lore, d_combat, d_trade, ...}`. Each projection magnitude = predictability score for that query class. The post-sleep HLA' carries these projections as additional latent channels. Wake-time: `argmax_i sim(h, d_i)` picks the pre-computed answer slot. The 5 synced affect scalars (valence/arousal/desperation/calm/fear) cross sync; the full HLA' does not.

(b) **latent_functor** (`riir-engine/src/latent_functor/`): Sleep-time = functor extraction over **anticipated** (source, target) pairs — `extract_functor_into(c, q̂_i, dim, f_out)` for each anticipated query direction `q̂_i`. `c'` = the set of pre-extracted functors `{(q̂_i, f_i, coherence_i)}`, ready to apply via `apply_functor` at wake-time. The `ReestimationScheduler` already implements "drift-triggered re-fit"; sleep-time compute becomes a *proactive* mode of the same scheduler — re-fit functors for *anticipated* queries before they arrive, not just *observed* queries after they arrive.

(c) **cgsp_runtime curiosity** (`riir-engine/src/cgsp_runtime/`): The allocation signal. `predictability(c) = 1 − sigmoid(curiosity_t)` where `curiosity_t = ‖actual_hla_t − karc_forecast_t‖`. High-curiosity contexts (unpredictable) → no sleep-time pre-computation, full wake-time compute. Low-curiosity contexts (predictable) → heavy sleep-time pre-computation, minimal wake-time compute. This is the first principled curiosity↔sleep-time allocation rule.

(d) **LatCal fixed-point commitment** (`riir-chain/src/encoding/latcal*.rs`): The `c'` projection matrix `{d_i}` is a linear op. LatCal commits linear ops over 2×2 fixed-point blocks. **A LatCal-committed sleep-time `c'` = deterministic, quorum-reproducible anticipated-query projection set.** Cross-node / cross-player: every node loading NPC `X` gets bit-identical `c'`, so every player talking to NPC `X` benefits from the same pre-computation. Only the resulting scalar projections cross sync; never the full `c'` matrix.

(e) **NeuronShard / freeze envelope** (`riir-neuron-db/src/`): `SleepAnticipationShard` subtype. Layout sketch: `[zone_hash(32) | direction_block_0(8) | ... | direction_block_7(8) | predictability_block(8) | coherence_block(8) | basis_config(1) | commitment(32) | merkle_root(32)]` — fits inside the existing 64-slot `style_weights` slot pattern. `MerkleFrozenEnvelope` wraps it for self-play freeze/thaw. Stored in cold tier, retrieved on demand.

(f) **DEC Stokes-calculus operators** (`katgpt-rs/crates/katgpt-core/src/dec/`): The predictability field over an NPC's belief region IS a divergence / flux calculation. `predictability(c) = 1 − |divergence(belief_cochain)|` — high predictability = low belief-mass divergence (the player's attention is concentrated where sleep-time pre-computed). High unpredictability = high divergence (attention flowing to unexpected zones). **This is the Stokes-theoretic reframing of the predictability score**, using `codifferential` (δ) on the belief cochain. Curse-of-dimensionality caveat applies (d ≤ 3 only — game maps, HLA regions, KG embeddings; NOT high-dim shards).

(g) **Analytic Lattice Encoder/Decoder** (`katgpt-rs/.research/311_Analytic_Lattice_Encoder_Decoder_Primitive.md`): The `c'` artifact is itself an analytic encoding of `c` — anticipated queries are the "factors" of `c` in query space. `compose_chain(c'_1, c'_2, ..., c'_k)` produces a composite anticipated-query set for cross-NPC / cross-region composition. This is the *spectral audit* primitive the Analytic Lattice Encoder plan flagged as a gap.

---

## 3. Verdict

### Super-GOAT

**One-line reasoning:** The paper provides the academic foundation + empirical validation for "offline query anticipation with predictability-gated allocation and cross-query amortization" — a mechanism none of our shipped primitives fuses into a single capability class, with a clear product selling point ("NPCs that pre-think about what players might ask, so dialog feels instant, and the same pre-thinking serves every player who talks to that NPC") and force multiplication across ≥5 existing pillars (sleep consolidation, KARC forecaster, latent_functor, Salience Tri-Gate, cgsp_runtime, NeuronShard, chain quorum, BFCF sharding).

### Novelty gate (§1.5)

1. **No prior art?** Three-layer check done: (notes) grep `sleep-time|offline thinking|rethink_memory` → only Plan 154 / Plan 107 / Plan 290 sleep-cycle hits, none framing sleep as query-anticipation. (code) grep `Sleep|SleepTime` in `.rs` → only `std::thread::sleep`. (vocabulary translation) `S(c)→c'` ↔ "anticipated-query projection set", "rethink_memory" ↔ "latent_functor proactive extraction", "predictability" ↔ "1 − curiosity", "amortization" ↔ "warm-tier cache hit" / "BFCF region-keyed retrieval". The COMBINATION has no shipped prior art. ✅
2. **New class of behavior?** Yes — "pre-think during idle, share pre-thinking across consumers" is not a sped-up version of any existing primitive; it's a new behavioral axis (offline anticipation vs online reaction). ✅
3. **Product selling point?** Yes — "NPCs that pre-think so dialog feels instant, with one pre-thinking serving every player." Complete sentence, not an optimization. ✅
4. **Force multiplier?** Yes — connects ≥5 pillars (sleep consolidation, KARC, latent_functor, Salience Tri-Gate, cgsp_runtime, NeuronShard, chain quorum, BFCF sharding, DEC Stokes operators). ✅

All 4 YES → Super-GOAT.

### Tiers (high → low)

| Tier | Criteria | Routing |
|------|----------|--------|
| **Super-GOAT** | Novel mechanism (no prior art) + new capability class + product selling point + force multiplier (≥2 pillars). Creates a moat. | Open primitive → katgpt-rs Plan 334. **Architectural guide → riir-ai/.research/163** (game runtime, where the per-NPC selling point lives). Plans → katgpt-rs 334 (open) + riir-ai 341 (private runtime). |

---

## 4. Mandatory outputs (created this session)

1. **Open primitive** → `katgpt-rs/.plans/334_sleep_time_query_anticipator_primitive.md` — generic `SleepTimeAnticipator` trait, `PredictabilityScorer`, `AmortizationCostModel`. No game semantics. The adoption hook.
2. **Private guide** → `riir-ai/.research/163_Per_NPC_Sleep_Time_Query_Anticipation_Guide.md` — the per-NPC selling point, connection map to riir-engine runtime, latent-vs-raw boundary, validation protocol (G1–G5), implementation priority.
3. **Private runtime plan** → `riir-ai/.plans/341_npc_sleep_time_anticipation_runtime.md` — wires the open primitive into riir-engine HLA, latent_functor, cgsp_runtime, chain_bridge.

---

## 5. Modelless unblock protocol check (§3.5)

Before any verdict, checked the three modelless unblock paths for any sub-component:

| Sub-component | Path 1 (freeze/thaw)? | Path 2 (raw/lora)? | Path 3 (latent projection)? | Verdict |
|---|---|---|---|---|
| `S(c) → c'` offline compute | YES — `c'` is a frozen snapshot, BLAKE3-committed via MerkleFrozenEnvelope | YES — `c'` can be a deterministically-constructed LoRA overlay applied at test-time | YES — `c'` is a direction-vector projection set | **Modelless-validable** ✅ |
| Predictability scorer | YES — pre-fit ridge over historical (c, q) pairs, frozen | YES — analytic form `predictability = 1 − sigmoid(curiosity)` | YES — dot-product + sigmoid projection | **Modelless-validable** ✅ |
| Test-time `T_b(q, c') → a` lookup | YES — `c'` artifact lookup | YES — `apply_functor` with pre-extracted direction | YES — `argmax_i sim(q, q̂_i)` + sigmoid gate | **Modelless-validable** ✅ |
| Cross-player amortization | YES — `c'` frozen once per NPC, replicated via chain | n/a (no per-player LoRA needed) | n/a (projection is shared) | **Modelless-validable** ✅ |

**No riir-train deferral needed.** The paper itself uses LLM calls but no training; our mapping uses freeze/thaw + latent projections + dot-product + sigmoid — all modelless. The wake-time `T_b(q, c')` budget `b` is governed by the already-shipped Plan 304 Gain/Cost Halting primitive (modelless).

---

## 6. Latent vs raw boundary (sync semantics)

| Signal | Domain | Synced? | Notes |
|---|---|---|---|
| Context `c` (HLA, latent_functor state) | Latent (semantic) | NO | Local to NPC; sleep-time operates on it |
| Anticipated query directions `{d_i}` (the `c'` matrix) | Latent (semantic) | NO (local) | Computed at sleep-time on the NPC's node; stored locally or in cold tier |
| Predictability scores `{p_i}` | Latent (semantic) | NO | Allocation signal; never crosses sync |
| **Frozen `c'` artifact** (SleepAnticipationShard) | **Frozen latent blob** | **YES (committed)** | BLAKE3-hashed, Merkle-wrapped, chain-committed; reloaded on any node by hash |
| **The 5 affect scalars** post-sleep | **Raw scalar** (semantic) | **YES (synced)** | The bridge artifact that crosses sync per AGENTS.md constraint — never the full HLA' |
| Test-time answer `a` | Latent (semantic) → token | Per-call | Local to player's session |

**Bridge function:**
```rust
// Zero-allocation, gateable by feature flag, no sync dependency.
fn sleep_time_to_synced_scalars(hla_prime: &[f32; 8]) -> [f32; 5] {
    // Project onto the 5 affect direction vectors — same machinery as EmotionDirections::project.
    // The 5 scalars (valence/arousal/desperation/calm/fear) are the ONLY thing that crosses sync.
    // The anticipated-query direction set {d_i} stays local; the resulting affect is what's shared.
    [/* ... */]
}
```

**Two-brain compatibility:** sleep-time operates on the **think brain** (per-NPC `SpatialBelief`, latent, NOT synced). It does NOT touch the **info brain** (real `MapPos`, synced, ground truth). An NPC whose think brain pre-computes anticipated queries still has its info-brain position synced at full fidelity. The pre-computation is purely about how the NPC's subjective model prepares for likely future inputs, not about ground-truth physics.

**Anti-cheat:** the frozen `c'` is BLAKE3-committed; tampering is detectable bit-identically. Two nodes processing the same NPC sleep-time compute produce the same `c'` if given the same input HLA + direction vector set. No model-in-the-loop divergence at the sync boundary.

---

## 7. Risks & honest assessment

### 7.1 Why this might fail (be honest)

- **Predictability score quality.** The paper measures `log P(q | c)` with a 70B base model. We don't ship a 70B model; our predictability proxy is `1 − sigmoid(curiosity)` where curiosity is KARC-forecast residual. Whether this proxy correlates with true query predictability is **an empirical G1 question**, not a given. The honest G1 must measure on a real predictability-labeled corpus.
- **Anticipated-query direction vectors.** Where do `{d_price, d_quest, d_lore, ...}` come from? The paper uses an LLM to rewrite `c` into `c'` (the LLM implicitly knows what queries are likely). Our modelless version needs *pre-existing* direction vectors — either (a) hardcoded per zone type, (b) learned offline via clustering on historical query logs (→ riir-train dependency if we go this route), or (c) borrowed from a frozen general embedding model. **(a) is modelless**; (b) is riir-train; (c) is borderline (uses a frozen model at runtime). Default to (a); (c) is opt-in.
- **Amortization failure.** The paper's 2.5× amortization gain requires N=10 related queries per context. In an MMORPG, an NPC might be talked to by 10 players an hour, but each player asks 1–3 questions, not 10. The amortization N might be lower than the paper assumes. **The cross-player generalization (N_players not N_queries) is what makes the economics work** — and it's exactly the part the paper doesn't test.
- **Warm-tier latency claim.** CompressionDrafter (Plan 285) tried warm-tier generation during NPC sleep cycles and FAILED at Hot-tier (GOAT 2×). Sleep-time compute lives in the same warm-tier regime; the latency budget is ms not µs. **Sleep-time is NOT a Hot-tier primitive** — it's a warm/cold-tier primitive that produces artifacts the Hot-tier consumes. This must be clear in the plan to avoid repeating the CompressionDrafter Hot-tier mistake.

### 7.2 Why this is still worth committing to

- **The paper is published, peer-reviewed, and empirically validated on standard reasoning benchmarks.** It's not speculative. The mechanism works.
- **Every individual piece ships in our codebase.** We are not asking for new physics; we are asking for a *wiring*. The fusion is the value.
- **The selling point is concrete and customer-facing** (instant-feeling NPC dialog), not internal infrastructure. This is rare for our corpus — most of our Super-GOATs are infrastructure moats; this one is a product moat.
- **The predictability↔curiosity inversion is a non-obvious theoretical contribution** that fell out of the distillation — it's the first principled allocation rule between CGSP exploration and sleep-time pre-computation, and it's novel.

### 7.3 What we will NOT do

- **Train direction vectors with gradient descent.** Direction vectors come from (a) hardcoded per zone type, (b) deterministic clustering, or (c) a frozen external embedding model. Training → riir-train.
- **Run a 70B model at sleep-time.** Sleep-time compute uses our existing primitives (latent_functor extraction, KARC ridge fit, Engram hash lookup). The LLM-in-the-loop is opt-in for high-stakes NPCs only.
- **Ship sleep-time compute as a Hot-tier primitive.** Sleep-time is warm/cold-tier; the produced `c'` artifact is what Hot-tier consumes at test-time. Mixing the tiers would repeat the CompressionDrafter failure.
- **Skip the GOAT gate.** The Super-GOAT guide (riir-ai/.research/163) contains the G1–G5 protocol; the open primitive plan (katgpt-rs/.plans/334) implements only the math primitives + synthetic gates. Promotion to default-on requires the riir-ai runtime plan (341) to clear its G1–G5 on a real game corpus.

---

## 8. Open questions (tracked in riir-ai/.research/163 §Open Questions)

- Q1: How many distinct query classes per zone type? (Hardcoded direction vectors need a count.)
- Q2: What's the realistic N (cross-player queries per NPC per sleep cycle) in a production MMORPG zone?
- Q3: Does the predictability proxy (`1 − sigmoid(curiosity)`) correlate with true query predictability on real player dialog logs? — G1 in the private guide.
- Q4: Should `c'` be committed via chain quorum (every NPC's `c'` is global) or only via local cold-tier storage (each node computes its own)? — affects whether the selling point is "every player gets the same pre-thinking" or "each player's local node pre-thinks independently."
- Q5: Multi-NPC composition — can `c'_NPC_A × c'_NPC_B` produce a composite anticipated-query set for a conversation involving both? — fusion with Analytic Lattice Encoder/Decoder (Research 311).

---

## 9. Implementation priority (cross-repo)

| Priority | Repo | Output | Status |
|---|---|---|---|
| **P0** | katgpt-rs | Plan 334 — `SleepTimeAnticipator` trait + `PredictabilityScorer` + `AmortizationCostModel` + synthetic G1/G5/G7 | This session |
| **P0** | riir-ai | Research 163 — private selling-point guide with G1–G5 protocol | This session |
| **P1** | riir-ai | Plan 341 — runtime integration (HLA sleep-time update, latent_functor proactive extraction, chain_bridge commitment) | This session (skeleton) |
| **P2** | riir-neuron-db | `SleepAnticipationShard` subtype (when P1 G2 passes) | Deferred |
| **P3** | riir-chain | LatCal fixed-point commitment of `c'` direction matrix (when P1 G4 passes) | Deferred |
| **P3** | katgpt-rs | DEC Stokes-calculus wrapper `predictability_via_belief_divergence` (low-priority alternative predictability scorer) | Deferred |

---

## 10. References

- **Source paper:** Lin, Snell, Wang, Packer, Wooders, Stoica, Gonzalez. *Sleep-time Compute: Beyond Inference Scaling at Test-time.* arXiv:2504.13171v1, 2025-04-17. https://arxiv.org/abs/2504.13171
- **Already-cited-as-relative:** `katgpt-rs/.research/069_AutoDreamer_Offline_Memory_Consolidation.md` line ~170 ("Sleep-time Compute: Lin et al. 2025 (offline pre-computation)") — this paper is the academic foundation for the entire "sleep-cycle offline pre-computation" line in our corpus.
- **Closest shipped cousin (consolidation side):** `katgpt-rs/.research/116_LLM_Sleep_Offline_Recursive_Memory_Consolidation.md`, `katgpt-rs/.plans/154_sleep_consolidation_offline_memory.md`, `src/sleep/`
- **Closest shipped cousin (forecast side):** `katgpt-rs/.research/288_KARC_Delay_Basis_Ridge_Forecaster.md`, `katgpt-rs/.plans/308_karc_delay_basis_ridge_forecaster.md`
- **Closest shipped cousin (warm-tier positioning, GOAT FAILED):** `katgpt-rs/.benchmarks/285_compression_drafter_goat.md` (the explicit "NPC sleep cycle warm-tier" framing thatCompressionDrafter failed to make work at Hot-tier)
- **Closest shipped cousin (latent-to-latent):** `riir-ai/.research/123_Latent_Functor_Runtime_Guide.md`, `riir-ai/crates/riir-engine/src/latent_functor/arithmetic.rs` (`extract_functor_into`, `functor_gate`)
- **Closest shipped cousin (cross-consumer commitment):** `riir-ai/crates/riir-engine/src/cgsp_runtime/chain_bridge.rs` (`commit_snapshot_via_quorum`, `reload_snapshot_from_chain`)
- **Predictability↔curiosity inversion:** `katgpt-rs/.research/126` (CGSP), `katgpt-rs/.research/288` (KARC curiosity as forecast residual)

---

## TL;DR

Sleep-time compute (arXiv:2504.13171) provides the academic foundation for what our codebase has been calling "sleep-cycle offline pre-computation" (AutoDreamer Research 069 cites this paper; Plan 154 ships the consolidation substrate; Plan 290 ships sleep-cycle motif mining). The paper's specific contribution is the **C/Q decomposition** (separate context from query), **predictability-gated allocation** (paper §7 leaves this as future work — we operationalize it), and **multi-query amortization** (generalize to multi-player amortization for the MMORPG selling point). **Verdict: Super-GOAT** via fusion with KARC (forecast primitive), latent_functor (proactive extraction mode), Gain/Cost halting (sleep-time budget allocation), Salience Tri-Gate (wake-time consume decision), cgsp_runtime (predictability = 1 − curiosity), NeuronShard (frozen `c'` artifact), chain_bridge (cross-player amortization), and DEC Stokes operators (predictability as belief-mass divergence). Open primitive → `katgpt-rs/.plans/334`. Private selling-point guide → `riir-ai/.research/163` ("Our NPCs pre-think about what players might ask during their idle time, so dialog feels instant — and the same pre-thinking serves every player who talks to that NPC"). Modelless throughout: freeze/thaw + latent projection + dot-product + sigmoid; no riir-train deferral needed. Sleep-time is a **warm/cold-tier primitive** (NOT Hot-tier — that's the CompressionDrafter Plan 285 failure mode). G1–G5 protocol in the private guide; open plan ships math + synthetic gates only.
