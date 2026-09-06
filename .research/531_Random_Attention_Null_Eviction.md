# Research 531: Random Attention — the Null-Hypothesis Evictor (Protection Dominates Scoring)

> **Source:** "Random Attention: Rethinking KV Cache Eviction for Efficient Reasoning" — Heng Wang et al. (Salesforce AI Research + UIUC), arXiv:2609.03430, Sep 2026. Code: github.com/SalesforceAIResearch/Random-Attention.
> **Date:** 2026-09-04
> **Status:** RECORD — Gain verdict landed (Plan 585 addendum: null-arm task + re-read of G8)
> **Related Research:** 523 (H2O norm-age eviction — Plan 585's parent), 100 (EGA salience — the `ega_eviction.rs` scoring arm), 487 (massive activations / sink-aware quant), 159 (KVarN), 483 (KEEP), 401 (KV consolidation), 435 (TTPO — per-track precedent), 516 (TTT-KVB)
> **Related Plans:** katgpt-rs 585 (usage-rate eviction — addendum tasks added 2026-09-04), riir-train 367 (policy-in-place co-training — rider recorded below)
> **Classification:** Public

---

## TL;DR

A Salesforce study **refutes the premise behind every scoring KV evictor**: keep the prompt, evict uniformly at random per KV head, compute no score — and it matches the strongest prior evictor (SnapKV / R-KV / VaSE / TriAttention) across 4 models × 6 reasoning tasks, while serving **32–43% higher throughput** in vLLM (no scoring pass). Two controlled experiments explain it: (1) **the prompt is the fragile part** — most of the gap between selectors is whether their score happened to keep it (giving every method the prompt-protection rule closes the big gaps: SnapKV +12.6/+22.5 points); (2) **the reasoning trace protects itself** — redundancy at two levels (the model restates what it needs; every KV head holds its own copy) makes a random draw retain enough copies. The residual for a signal: facts stated once and never restated at depth (passcode probe: R-KV 84% vs random 0%).

**Actionable for the stack:** Plan 585's Bench 697 (T3.2) compared six scored/structural policies — **none against the null**. This paper's central methodological claim is that prompt-pinned per-head random at matched budget + matched protection is the control every scored policy must beat ("Any signal-based selector that cannot beat it at matched budget is not extracting usable information from its signal"). → Plan 585 addendum: add the `random_prompt_pin` arm, re-read G8 against it.

---

## 1. Paper Core Findings

1. **The method** (Eq. 3): `s_i = +∞ if i ≤ ℓ_p (the whole prefill: system prompt + chat template + question), else u_i ~ Uniform(0,1)`, drawn independently **per KV head** at every eviction; top-K per head. Periodic eviction with budget K + recent buffer r (never scored). Per-eviction cost = one rand + one topk + the shared compaction.
2. **Accuracy parity** (Table 1): at ~4× compression, Random Attention is significantly AHEAD in 31 of 60 baseline cells and behind in exactly one (code reasoning on Qwen3-32B, traced to prompt-length not signal — TriAttention keeps the prompt by construction and its +3-point lead survives). Compression sweep 2×→16×: gap WIDENS in random's favor vs VaSE; ties TriAttention throughout.
3. **The prompt confound** (§5.1, Table 2): prior evictors differ in prompt protection (TriAttention pins by rule; SnapKV/VaSE/R-KV leave it to the score). With the rule given to everyone, the three scored baselines land within 2.2 points of each other everywhere; the payoff of the rule is ordered exactly by how much prompt each score was losing (SnapKV lowest retention → biggest gains; R-KV already keeps the prompt → ±1.9). **The prior papers' "random baselines lose badly" results (Liu 2025, Yuan 2026) are re-attributed: their random baselines lost the prompt.**
4. **Cross-head redundancy is real and superadditive** (§5.2 planted-fact probe): a fact kept in ONE head retrieves 3% (best head alone), TWO heads 60%, three 83%, all eight 99%. Two different facts in different heads: R = 0.31 together vs 0.10 + 0.16 alone. The SHAPE of surviving copies is irrelevant (scattered tokens ≈ intact sentences). Consequence: per-head independent draws maximize the probability some head holds a copy — exactly what per-(b,h) eviction gives.
5. **Implicit age bias**: no score is computed, but per-round random survival is geometric: `((K−ℓp)/(K+r−ℓp))^n ≈ 0.94^n` — **random eviction IS a soft recency window** (recent ≈ always present; old survive as a thin per-head tail). Cross-head union at 1–2k age: 0.776 (random) vs 0.161 (shared draw) — the shared-draw control TIES on real traces (text redundancy suffices) but loses the probe regime; the two redundancy levels are substitutes and per-head independence preserves both.
6. **The boundary** (§5.3 passcode probe): a fact stated once, never restated, 57 rounds deep — Random retrieval 0.000 / log p −18.35; R-KV (cumulative attention) 0.836; recent-window scores (SnapKV, TriAttention) ≈ 0. **Retrieval tracks the attention statistic each signal scores by** — needle-finding is real selection skill; it just rarely matters because traces restate.
7. **Throughput** (Table 4/9): vLLM paged serving, 32k generations: 1.58–2.67× full attention, **+32–43% over TriAttention at identical kernels/paging/scheduler**. Mechanism: scoring is cheap alone (0.30 ms compaction-only vs 1.47–1.64 ms scored per round, single-stream) but **serving multiplies it** — compressions pile up at sync points across 128 concurrent requests, and content-dependent scoring under paging needs an extra pass over block tables (cache-statistic scores) or recomputation (attention-weight scores — fused kernels never materialize the weights). **Random reads nothing; the runtime's existing compaction path is the whole integration** (adding it to TriAttention's plugin took one function).
8. **Equal-memory capacity**: every evictor collects most of the batch win (small cache → more sequences); the residual ordering IS the scoring-pass cost. At K=1024: Random reaches 28.8× full-attention throughput, +16–20% over the scored policies at their own largest batches.

## 2. Distillation

### 2.1 Path 0 inventory (per-component)

| Component | Training? | Modelless form | Coverage in-tree |
|---|---|---|---|
| Prompt-pinned per-head uniform-random eviction | No | Exclusion predicate + seeded RNG draw + per-head top-K | **NONE as an arm.** Bench 697 T3.2 ran {ring, raw_h2o, mass_age, mass_age_sink, ega_energy, ega_x_usage} — no null arm. The `pinned: &[bool]` parameter in `kv_eviction::select_evict` (Plan 585 T1.3) already supports the +∞-score form (pin all prompt rows); the draw is ~10 lines on the caller side. |
| Matched-protection protocol (score-alone vs +prompt) | No | Bench arm factorial | **NONE.** Bench 697 compared policies at (implicitly) their own protection behavior. The factorial (each scored policy ± prompt-pin) is the diagnostic that attributes quality to protection vs signal. |
| Round-cost axis (scoring pass vs compaction-only) | No | Criterion round timing | **PARTIAL.** G2 measured the score update (1.22 ns/row CPU) but not the eviction ROUND; the kernel-side scorer (T4.1 mass byproduct, unlanded — Issue 836) is where the paper's 32–43% would bite us. Our kernels return no column sums (note 523 §1.5) — the random policy needs no kernel change at all, which is ALSO the T4.1 alternative: skip the kernel, ship the null. |
| Cross-head redundancy pooling law | No | Per-head independence already by-construction; the probe instrument (pin-in-chosen-heads) is a test fixture | **Insight only.** Validates Plan 585's per-(b,h) design; answers half of T3.3's open question (τ disagreement 0.69–0.75 measured; the paper adds WHY per-head matters: pooled copies). |
| Geometric-survival soft-recency property | No | Closed-form `((K−ℓp)/(K+r−ℓp))^n` | **NONE.** A free analytical lens on any random arm's behavior (connects to the age axis of note 523's score — random's implicit survival IS an age profile). |
| Eviction-in-training (MEMENTO / Prefix Sliding, cited as concurrent) | Yes | — | Reinforces riir-train Plan 367 (co-training against the deployed policy). New rider: co-train against the SIGNAL-FREE policy — cheapest co-adaptation target, no kernel work. |

### 2.2 Signal-diff vs the closest in-tree cousins (§3.6)

- **vs `ega_eviction.rs` (riir-engine, Research 100):** static key-energy score `dot(key, w_proj)` → sigmoid gate → evict low-energy. No positional pin (config has count floors only: `min_retain`, `max_evict_fraction`). Already a Bench 697 arm and LOST to mass_age (recall 30/42/56% vs 50/100/100% at caps 32/48/64). Unwired (zero production callers of `ega_evict`). **The paper's prediction for EGA: its quality is dominated by whether its energy incidentally protects the keystone rows** — exactly what the +prompt factorial arm will measure. No live demotion (unwired); the bench arm is the audit.
- **vs `mass_age` (Plan 585):** mass/age retains the prompt WELL in principle (sinks accumulate attention mass → high usage rate; note 523's sink-pin covers the rest) — the paper's Table 2 predicts R-KV-class policies gain little from the rule because they already keep the prompt (R-KV ±1.9 points). So the paper PREDICTS mass_age survives the null-arm comparison on protection, and the comparison isolates pure signal value on our workload. Our workload (recurring needles, caps 32–64) is the intermediate regime: recurring needles are the text-redundancy analog (random may tie at high caps), while the cap=16 miss (0/384) is needle-at-depth (random should collapse there — the paper's passcode regime). Either outcome is informative: tie at high caps → the complexity of mass_age buys only the deep-needle tail; win → G8 strengthens with the null controlled.
- **vs `decay_confidence` / think-brain staleness:** sigmoid(−λ·age) on beliefs — a scored soft-eviction for the think brain. Not a KV policy; the game hot path has no KV to evict (HLA/GDN are fixed-size recurrent). Analogy only; recorded in the game reframe below.

### 2.3 Fusion

**Paper's null × our bench discipline × our canary:** the full GOAT-gate shape for ANY future lossy KV policy on the serving path becomes: {policy, policy+prompt-pin, random+prompt-pin, ring} × {recall/accuracy, RunawayStats canary, round-cost} at matched budget — protection factorial first, signal second. None of the four alone is the gate; the paper supplies the missing arm and the attribution protocol.

**Second fusion (training side):** if the serving path adopts random+pin (the throughput winner), riir-train Plan 367's co-adaptation should target the SAME policy — co-training against a signal-free policy is the cheapest possible recipe (no column-sum kernel, no scorer in the training loop) and the paper's own co-adaptation logic ("same conditions as inference") demands the match. Rider recorded on Plan 367 via note 523 cross-ref.

### 2.4 Game-runtime reframe (priority #1) + healer reframe (priority #2)

- **Game runtime:** no live KV eviction exists on the NPC hot path — per-NPC cognition runs on fixed-size recurrent state (HLA/GDN), and think-brain staleness is sigmoid decay, not eviction. The concrete future consumer is the **serving path's long-context condition windows** (riir-ai router / engram conditioning): if a memory cap ever lands there, this paper says the FIRST policy to try is pin-the-keystone (system prompt + condition window) + random the trace, gated by the already-landed `RunawayStats` canary — with mass_age as the challenger only if deep-needle recall (once-stated facts at depth: quest constraints, earlier evidence) measurably matters on our workloads. Pull-gated exactly like Plan 585 (Issue 836 consumer).
- **Healer (consumer-context check, answered honestly):** the fix-trajectory store (`.heal/`, Warm tier) has **no eviction today** — nothing to demote or fix. If a memory cap ever lands, this paper's shape applies directly: Certified tier = keystone (pin), everything else random-or-recency — NOT a new Elo/Beta scoring pass for eviction. Recorded as design guidance, not an issue.

## 3. Verdict

**One verdict, one track** (pure inference paper; no training content of its own — the eviction-in-training prior art reinforces the EXISTING riir-train Plan 367):

| Axis | Call |
|---|---|
| Novelty (Q1) | **No** — the method is published (this paper + close cousins below); our content is integration + gate discipline. |
| New behavior class (Q2) | No — bounded-memory eviction is literature-established; our serving path simply doesn't have it yet. |
| Selling point (Q3) | No — "matches random" is a null result, not a moat; "validated against the null" is discipline. |
| Force multiplier (Q4) | Moderate — sharpens Plan 585's gate, feeds the KV stack + league serving follow-ups. |

**Verdict: GAIN** (ships + actionable). Files: this note + **Plan 585 addendum** (T3.6 null arm + T3.7 round-cost axis; re-read G8 against the null). NOT Super-GOAT. No demotion executed today: Bench 697's opt-in-no-consumer standing state is unchanged; the addendum's arm decides whether mass_age's signal value is real beyond protection once the null is controlled.

**MOAT gate:** katgpt-rs in-scope (KV/eviction ledger slot; the gate-design upgrade IS the moat — a bench that can't be fooled by the protection confound). No routing conflicts; healer + game reframes recorded above with the honest "no live consumer" answers.

## 4. Prior-Art Search Record (§4)

**Instrument limitation, recorded:** the web-search API was unavailable this session (MCP key error). The record below is the paper's own related-work section (a 2026-09-03 submission surveying exactly this landscape) — adequate for the routing verdict because our claim is NOT novelty of the method.

- **Garcia, arXiv:2605.18053 "Protection is (nearly) all you need: Structural protection dominates scoring in globally capped KV eviction"** — the closest published cousin of the headline claim, in the LONG-CONTEXT QA regime (global cache cap, prompt-boundary guarding). The paper differentiates: decode-phase reasoning (short prompt, cache filled by the model's own trace) vs long-input QA; whole-question protection vs boundary tokens; per-head eviction + the two-level redundancy mechanism are new here.
- **Wu et al., ICLR 2026 "Randomization boosts KV caching, learning balances query load"** — random eviction at the serving level for prefix-sharing robustness (different objective).
- **Chen et al. 2024 (score-guided randomized eviction; VaSE's ancestor)** — randomness in the keep-set, guided by a score — the opposite conclusion; re-examined here as partially protection-confounded.
- **StreamingLLM (Xiao et al. 2024)** — sink + recent window: signal-free structural protection; the `recency+prompt` control row (lands within ~2 points of the best baseline once protected).
- **Wang, arXiv:2604.17935** — theory: random caches lose pointer-chasing when nothing is redundant (the boundary the passcode probe instantiates).
- **Muennighoff et al., arXiv:2608.26070 "Prefix Sliding"** (concurrent) — prompt + recent window for test-time scaling; also eviction-in-training. **Kontonis et al. "MEMENTO"** (concurrent) — teaching LLMs to manage context / eviction-in-training.
- Conclusion: the null-evictor METHOD is published prior art; no novelty claim survives for us. Our claims: the null-arm gate discipline applied to Plan 585's bench + the integration/round-cost analysis for OUR serving path.

**Web-search completion (2026-09-06, second pass):** the instrument limitation
above is CLOSED — an independent web search this date confirms the routing
verdict with fresh IDs. No standalone score-free random evictor with
prompt-pinning was published before this paper (Sep 2026). Closest prior art,
now pinned by ID: **VaSE arXiv:2606.03928** (stochastic eviction FOR reasoning
but signal-bearing — value-magnitude protection + attention-sampled fill; the
Random-Attention repo builds on VaSE's harness and benchmarks it directly),
**NACL arXiv:2408.03675** (per-head diversified random eviction HYBRIDIZED with
scoring, long-context, 2024 — the earliest random-as-component), **R-KV
arXiv:2505.24133** (NeurIPS 2025; the text-level-redundancy PRIOR ART — this
paper's contribution is showing even the redundancy score is unnecessary),
**StreamingLLM arXiv:2309.17453**. Two PRO-scoring contrarians the result runs
against: Learning-to-Evict arXiv:2602.10238 (ICML 2026; RL ranker beats random)
and arXiv:2605.25085 (Wyner–Ziv theory; heavy-hitter provably beats random in
the general model) — both re-attributed here as protection-confounded. Confirmed
novel to the paper: the prompt-fragility attribution and the cross-head-
redundancy mechanism; confirmed NOT novel: text-level redundancy (R-KV). Q1
unchanged — we claim no method novelty; routing verdict stands.

## 5. PoC Addendum

**RECORDED 2026-09-06 (Bench 697 addendum, Plan 585 T3.6–T3.9 landed):** the
falsifier ran on the constructed induction-pair fixture (32 seeds × 12 needles,
M3, release). **The passcode regime CONFIRMED and the signal verdict is
CONFIRMED — the demote-the-loser branch did not fire:** the unpinned null
collapses at cap=16 (7/384, floor-class — and BEATS mass_age's 0/384 there: the
null's geometric survival `((K−ℓp)/(K+r−ℓp))^n` retains a thin recent tail,
the paper's implicit soft-recency finding, which is mass_age's second recorded
extreme-pressure loss), while mass_age strictly beats the null at every regime
cap (192/38 vs 384/80 vs 384/100 — 5.0×/4.8×/3.8×). Protection factorial: every
pinned arm hits 100% at every cap (pin-honored PASS; the paper's Table-2 shape
reproduced — rand_keystone = the null-with-oracle ceiling at zero scoring
cost). **Instrument find:** rand PASSES the runaway canary at cap=32 (R=1.0,
p_cap 0.0) while recalling 9.9% — generation health rides the recent tail the
null keeps, so the canary and the null control are complementary instruments,
not substitutes. Gates landed: `kv_eviction::beats_random_prompt_pin` (strict;
NaN fail-closed) — the standing promotion rule for any lossy KV policy is now
`runaway_gate` ∧ `beats_random_prompt_pin` ∧ the protection factorial.
Record: [Bench 697 §T3.6/T3.7 Addendum](../.benchmarks/697_usage_rate_eviction_goat.md).

Original entry: the falsifier was cheap and scheduled via Plan 585 addendum
T3.6 (superseded by the record above).

## 6. Cross-Ref (2026-09-04)

> **Cross-ref (Research 523 / Plan 585):** this paper is the null hypothesis Plan 585's bench lacked. Any future promotion of `usage_rate_eviction` (or ANY lossy KV policy) must include the prompt-pinned per-head random control at matched budget — see Plan 585 addendum. The throughput half (32–43% over scored evictors under paged serving) lands on T4.1's decision: the mass-byproduct kernel (Issue 836) is the expensive path; the null policy needs no kernel at all. If deep-needle recall is not load-bearing on our serving workloads, **skip the kernel and ship the null** — that is now a registered alternative, not a strawman.
> **Cross-ref (riir-train Plan 367):** if serving adopts random+pin, co-train against random+pin — the cheapest co-adaptation target (no scorer in the loop, no kernel work).
