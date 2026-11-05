# Research 622: HybridAgent — Hybrid Latent-Text Reasoning Agents

> **Source:** "Hybrid Reasoning Agents" (HybridAgent) — [arXiv:2610.18861](https://arxiv.org/abs/2610.18861), Zou, Qi, Chen, Bianchi, Zhu, Lu, He, Zou (Stanford + UIUC + Together AI), 2026-10-13. MIT code `github.com/togethercomputer/HybridAgent` @ `8ad7d8f56a80` (cloned to `.raw/`, pinned, read, removed post-distill). Models open on HF: `togethercomputer/HybridAgent-8B/14B-Agentic` (Qwen3-8B/14B-Base backbones + `adapter.pt`).
> **Date:** 2026-10-15
> **Status:** CLOSED-RESOLVED 2026-11-05 (Outcome record in §3) — was "Active — Gain (modelless GOAT) + Path 0.5 (training, secondary)". Modelless GOAT → Plan 626 **CLOSED-NEGATIVE** 2026-10-16 (Bench 941: anchor exit fires 1/12 frozen-model problems; `swir_anchor_exit` stays opt-in). Training track → riir-train Plan 461 Phase 2 **COMPLETE** (Bench 655: adapter map PASS retrieval@1 97.4%, **G2 structural FAIL 0.88×** — Bonsai LM head is 0.9% of a step, ceiling ≈1.009×; Phase 3 owner-gated). Fusion casualty: `swir_switch_thinking` **DEMOTED default→opt-in** 2026-10-16 (owner call on riir-ai Issue 1075 — corrected-host G2 1.00× vs ≥1.3× gate, katgpt-rs `12db0eeb7`). Durable keepers: `position_provenance.rs` (`142dfcece`) + `pseudo_text.rs` (`76d6d4fef`).
> **Related Research:** 241 (SwiReasoning — closest shipped cousin), 158 (MUX/CODI), 192 (NextLat — Coconut verdict lineage), 204 (NFCoT), 073 (LT2/Ouro/MoR verdict family), 170 (LEAP thinking bandit)
> **Related Plans:** 275 (SwiR, DEFAULT-ON — the surface this extends), 194 (thinking_cot), 211 (three-mode router), 172 (RiM slots), 283 (Self-Advantage Gate), **626 (this distill's modelless plan)**, **riir-train 461 (training recipe, secondary)**
> **Classification:** Public

---

## TL;DR

HybridAgent interleaves **latent-reasoning spans** with **text tool-calling** in one generation process: a tiny trained residual MLP ("HybridLink", `R(h) = post_ln(h + W2·GELU(W1·pre_ln(h)))`, ~2·D² params) maps the last-layer hidden state back into input-embedding space so the model continues generating from its own hidden state without decoding tokens; `<latent>`/`</latent>` anchors switch modes; a **continuous working memory** (the KV cache, with a per-position token-vs-latent ledger) carries hybrid context across multi-turn agent interactions. HybridSFT (3 stages) then HybridRL (GRPO) train the when-to-go-latent policy. +10.4% over text and latent agentic baselines, 35–55% token reduction.

**Distilled for katgpt-rs (modelless, inference-time):** the trained bridge is training-bound (Path 0.5), but four runtime mechanisms extract cleanly onto the frozen-model stack: (1) the **anchor-exit law** — terminate latent/soft-embedding spans when the implicit argmax readout lands in a boundary-token set after a min dwell (a *semantic marker* exit signal, distinct from Plan 275's *entropy trend*); (2) the **KV position ledger** — per-position provenance `Tok(id) | Lat{span,step}`, whose first consumer is sampler correctness (repetition penalties must never count non-materialized latent positions); (3) the **pseudo-text readout** — cosine nearest-token inspection for any soft-embedding lane; (4) forced min/max dwell caps (config deltas on existing SwiR knobs). The multi-turn/span-reuse and FSM pieces are standard practice or consumer-less today — audited discards below.

---

## 1. Paper Core Findings

| Finding | Detail |
|---|---|
| Architecture | HybridLink adapter bridges last-hidden-state → input-embedding space; latent spans run inside `<think>`; tool calls (search/python) are text-mode; results fed back token-by-token (FEED mode) |
| Mode switching | Entry: model samples `<latent>` (learned). Exit: **the implicit argmax readout of the current state equals `</latent>` after `min_latent_steps`, hard-capped at `max_latent_steps`** (forced exit materializes the anchor token) |
| Working memory | KV cache persists across turns; `kv_ledger` marks each position token-id or `None` (latent). Used so repetition penalty counts only real tokens; positions still attend normally |
| Interpretability | Each latent step's injected vector decoded by cosine-nearest-token against a normalized embedding bank; spans render as pseudo-text with a readability score |
| Training | HybridSFT: (1) format activation (backbone learns anchor emission), (2) hybridlink activation (adapter regression onto teacher target embeddings), (3) self-conditioned joint (backbone+adapter on the model's OWN latent inputs — closes the DAgger-class train–inference gap). Then HybridRL: GRPO, reward = `r_correct + r_format + r_tool`, exact rollout replay (bit-consistent trainer recomputation) |
| Results | 3 scales, 11 benchmarks: +10.4% vs text/latent agentic baselines; 35.5–54.7% token reduction; 1.8–2.6× faster; 2.7–3.7× cheaper |

Prior-art verdict (§4 search, 2-day-old paper): **no published work pre-empts the headline combination** (tool-integrated multi-turn hybrid latent-text agent). Closest: HRPO (arXiv:2505.18454, GRPO-trained latent/discrete interleaving — no tools/bridge), Coconut (arXiv:2412.06769 — anchor skeleton, no adapter/tools), SwiReasoning (arXiv:2510.05069 — training-free switching, no tools/memory). Trained-form prior for the exit law: arXiv:2610.17189 (soft tokens until max-prob readout hits an end-of-thought token) and arXiv:2511.21581 (RL-learned latent length). No prior found on repetition-penalty-over-latent-steps.

## 2. Distillation

### 2.1 Path 0 inventory (component → coverage + extraction)

| # | Component | Modelless-extractable? | Coverage in stack | Disposition |
|---|---|---|---|---|
| 1 | HybridLink trained bridge `R(h)` | **No** — the gain IS the learned representation-gap correction; closed-form substitutes are the paper's own contrast class | `swir/soft_embedding.rs` (Plan 275, DEFAULT-ON) = the training-free substitute; CODI/SoftCoT/2608.08888 are the trained-bridge family | **Path 0.5 → riir-train Plan 461** (secondary). Substitutes: audited discard (§2.4) |
| 2 | Format activation (trained anchor emission) | No (frozen models have no trained reason to project onto anchors) | thinking_cot bandit + Plan 283 = the runtime-surrogate class for "when to spend compute" | Bandit latent-arm extension deferred in Plan 626 |
| 3 | **Anchor-exit law** | **Yes** — dwell-gated boundary-argmax predicate over `next_logits`, boundary set DERIVED per model (EOS + discourse boundaries), not trained | Plan 275 exits on **entropy trend** (statistical signal); anchor exit is a **boundary-identity** signal (semantic marker). Signal-diff below. Published trained-form prior: 2610.17189, 2511.21581 — the *modelless adaptation on frozen models* is unclaimed | **→ Plan 626 (GOAT)** |
| 4 | Min/max dwell caps | Trivially | SwiR has dwell windows (`w_e_to_l`/`w_l_to_e`), `c_max`, answer budget | Config deltas folded into 626 |
| 5 | **KV position ledger + ledger-correct sampling** | **Yes** — pure data structure, zero model assumptions | **Ships host-side**: riir-engine's Gemma2 backend excludes soft-embedding positions from the penalty history (`generated_tokens` pushes only explicit ids) — HybridAgent's `kv_ledger` semantics under another name. Not contract-ized katgpt-rs-side; two LIVE host/bench defects found in the same subsystem during verdict review (penalty compounding over duplicates; latent pseudo-argmaxes in the bench scoring path) | **→ Issue 941** (contract lift; reference impl = the riir-engine backend) **+ riir-ai Issue 1075** (the live defects) |
| 6 | **Pseudo-text readout + readability/loop/collapse sensors** | **Yes** — cosine NN against a pre-normalized embedding bank | Not shipped as an inspection surface for swir/rim lanes | **→ Issue 942** |
| 7 | Continuous working memory / span-atomic cache reuse | Yes (infra) | Multi-turn KV reuse + prefix caching = standard serving practice (prior art confirms); the span-atomic delta needs a demonstrated multi-turn consumer | **Audited discard** — revisit trigger: heal-retry prefix reuse landing as a real consumer |
| 8 | Bridge-injection modes (hard one-hot / entropy-matched top-k) | Yes | Hard one-hot = the existing explicit mode (decode-then-feed); top-k = a soft_embedding config knob with no evidence of need | **Audited discard** |
| 9 | FSM materialization invariants | Yes (tests) | SwiR already materializes control tokens at forced transitions (CloseThink/ForceAnswerPrefix enqueueing); tool-boundary invariants belong to host loops that don't exist in this stack (no agent product) | **Audited discard** — revisit with the first soft-embedding host loop |
| 10 | HybridSFT/HybridRL recipe | **No → Path 0.5** | GRPO substrate ships (`loss_grpo.rs`, `pooled_grpo.rs`, `rollout_pool.rs`, `rlvr_reward.rs`, `sil_replay.rs`) | **→ riir-train Plan 461** (secondary): E1 adapter-only minimum ~3–5 GPU-h; E2/E3 owner-gated |

### 2.2 Signal-diff (§3.6) — anchor exit vs the shipped Plan 275 controller

The containing expression read one level up: SwiR's exit-from-latent consumes an **entropy trend against a reference** (`H_t < H̄` → L→E switch), and its termination is **count-based** (`c_max` → CloseThink/ForceAnswerPrefix). HybridAgent's latent exit consumes **argmax identity against a boundary-token set** — a semantic marker the model's own distribution emits — gated by a minimum dwell and a hard cap. Different signal classes (statistical trend vs boundary identity), different failure modes (entropy plateaus chatter; anchors never fire on untrained projections — or, the opposite risk Plan 626 must also measure, they OVER-fire when the boundary set contains frequent-in-domain tokens like newline/period, collapsing the anchor arm into a fixed min-dwell cap). They **compose**, they don't compete: `exit = anchor ∨ entropy ∨ cap`, first match wins, with a per-span reason tag — and the reason histogram is itself the calibration instrument (Anchor at 0% ⇒ the arm demotes inert; Anchor near 100% ⇒ the set is over-broad and narrows). Both risks are measured, neither assumed: the Phase-0 probe carries a kill gate on the under-firing side and an over-firing telemetry arm on the other. The decisive risk stays named honestly: on a frozen model the anchor signal is a *completeness proxy* (controller vocabulary, not model vocabulary) — decline is a correct answer.

### 2.3 Reframes

**Game context (step 4):** the KV ledger is the **two-brain law expressed at cache level** — world observations persist as `Tok` positions (info brain, raw, syncable); per-deliberation latent positions are `Lat` scratch that evaporates (think brain, local-only, never serialized). Our sync-boundary rule already forbids syncing latent state; the ledger is its runtime enforcement surface. Anchor-exit maps to deliberation-span budgeting against the 20 Hz tick (bounded dwell = tick-budget law); the trained when-to-think policy's modelless surrogate is the existing drive/bandit class — never a magic threshold (behaviour-through-brain rule).

**Consumer context (healer, priority #2):** the fixer loop's draft→oracle→feed cycle is HybridAgent's FEED-mode grammar, and heal-retry prefix reuse is the named revisit trigger for the span-reuse discard. Ledger-correct sampling applies to any host loop running penalties over soft-embedding steps. The healer runs no soft-embedding lane today, so the anchor-exit has no direct healer consumer — recorded, not forced.

### 2.4 Fusion

The genuine fusion is **cross-inform between the two tracks**: riir-train Plan 461's E1 (adapter-only regression on a frozen backbone) produces `(h, argmax-readout)` statistics at latent-completion positions — the **trained upper bound** of the anchor signal — while Plan 626's Phase-0 probe measures the **frozen lower bound** on the models we actually serve. Together they bound whether the anchor-exit is worth having, and E1 needs no consumer to be worth running (it is the cheapest measurement of the paper's central claim — that a 2·D² adapter closes the hidden-state→embedding gap — on our own backbone). Secondary fusion surfaces recorded for the owner's routing: game think-brain tick budget (latent deliberation + action token turns token reduction into tick-budget reduction), ternary/low-rank HybridLink (CPU-class serving; a later kernel_opt mining target), decision-serving escalation rung (serving arm = rethink charter, private; reflex at most a public comparison lane), healer hosted-expert cost cut (~2–3× on the miss lane).

## 3. Verdict

### Outcome record (2026-11-05 — both tracks resolved; stale grade struck per ungoat discipline)

Every filing below has since resolved; verified live this session (files read, commit hashes):

| Filing | Outcome | Evidence |
|---|---|---|
| Plan 626 (modelless GOAT — anchor exit) | **CLOSED-NEGATIVE** — Phase 0 kill gate failed on under-firing: the minimal anchor arm fires on **1/12** MATH-500 problems in latent mode (Gemma 2 2B IT); all observed anchor activity sat ≥ step 212 vs the production span cap c_max=64, so the under-firing conclusion is STRONGER on production surfaces. Exactly the pre-named decisive risk (frozen models lack trained anchor vocabulary — the signal is a completeness proxy). Re-open trigger (recorded in the plan): a thinking-format model re-runs the landed instruments unchanged. | Bench 941; plan Status line; `swir_anchor_exit` stays OPT-IN (Cargo.toml) |
| Issues 941 + 942 (katgpt-rs) | **BOTH LANDED** — `position_provenance.rs` (Tok/Lat position-provenance host contract; the note's most durable extraction) and `pseudo_text.rs` (cosine nearest-token readout instrument); issue files removed per noise-reduction, HISTORY entries hash-pinned | `142dfcece`, `76d6d4fef`, `322a6c2fd`, `21843a1fb` |
| riir-ai Issue 1075 (host defects) | **FIXED** `6fd3af867` (+ `c9002cb33`) — distinct-once repetition penalty + fed-token answer history. **Consequence: the corrected re-measurement supersedes riir-ai Plan 313's G2 promotion cells — 1.43× → 1.00×** | fix commits; `.benchmarks/313_swir_real_model_goat.md` SUPERSEDED header |
| `swir_switch_thinking` (the fusion surface, Plan 275) | **DEMOTED default→OPT-IN 2026-10-16** (owner call, katgpt-rs `12db0eeb7`); re-promotion needs a corrected G2 gate — the synthetic G3–G6 invariants stand (3.1 ns/step, convex-hull 1000/1000, no-regression, kurtosis escape); `thinking_cot` itself stays default (transitively via chain_fold/collapse_aware_thinking); riir-engine `swir_validation` forward verified both postures | Cargo.toml feature comment; Bench 313 SUPERSEDED header |
| riir-train Plan 461 (training recipe, secondary) | **Phase 2 COMPLETE** — stage-2 adapter on frozen Bonsai-27B-PQ2: map PASS (held-out MSE 0.050× constant, retrieval@1 97.4%), G1/G3/G4 PASS, **G2 FAILS STRUCTURALLY (0.88×)**: the LM head is 0.9% of a token step ⇒ decode-skip ceiling ≈1.009× even with a free adapter. T2.4 trained upper bound (span-end readout = teacher 95.8%) published as the Plan 626 re-open reference. Phase 3 (E2/E3) stays owner-gated — justified by the ceiling: a full hybrid post-training campaign has no efficiency payoff on this serving posture | Bench 655; plan Status/lease lines |

**Net verdict after outcomes: the paper's efficiency claims do not transfer to this stack's serving posture.** Modelless: the anchor-exit arm is inert on frozen models (closed negative). Trained: the per-step saving is nil on Bonsai — the paper's own cost model charges each latent step a full backbone forward pass (its Table 9), so its gains come from *trained trajectory shortening*, which our frozen serving models lack. Durable yield: two correctness/instrument modules (`position_provenance.rs`, `pseudo_text.rs`), two fixed live defects, and the trained-upper-bound statistics behind the recorded re-open triggers. Discard #7's revisit trigger (heal-retry prefix reuse as a real consumer) checked this session — not fired (no soft-embedding/swir lane in riir-refine `src/`). **No further filings from this note.**

~~**GOAT — modelless track**~~ (anchor-exit composition → Plan 626; **STRUCK 2026-11-05 — Plan 626 closed negative, see Outcome record above**). Not Super-GOAT: Q2 fails — an alternative exit signal for existing latent-compute allocation is not a new capability class, and the trained-form exit prior (2610.17189/2511.21581) plus the trained headline means the novel axis is only the *modelless adaptation on frozen models*, which is exactly what the Phase-0 probe must earn.

**Verdict-review corrections (round 1, both verified by direct code read before filing):** (1) the KV-ledger row above originally claimed "not shipped" — wrong: riir-engine's Gemma2 backend already implements the discipline host-side; Issue 941 is a contract LIFT, not a new primitive. (2) The review surfaced two LIVE host/bench defects (penalty compounding over duplicate history entries, contradicting the code's own HuggingFace-convention comment; latent pseudo-argmaxes pushed into the bench's scored answer history against the "held back" intent comment) — filed as riir-ai Issue 1075 with verification quotes.

**Path 0.5 — training track** (recipe → riir-train Plan 461, **SECONDARY** by serving-envelope fit: no production consumer today; the modelless plan reaches the swir hot path immediately and is PRIMARY). One verdict per track — neither cascades.

**MOAT gate:** `katgpt-rs` in scope (base primitive, thinking-control slot beside swir; per-stack ledger row: feature `swir_anchor_exit`, opt-in until GOAT, entropy stays as the composite's fallback — the demotable unit is the anchor ARM, never the entropy law). `riir-infer`: ledger substrate follow-up noted in Issue 941 (substrate home when a cache-side implementation lands). `riir-train`: active moat, Plan 461. No rethink-class content in any public file.

### Filings

| File | What |
|---|---|
| `.plans/626_swir_anchor_exit.md` | Anchor-exit composite (Phase 0 kill-gate probe → controller extension → GOAT bench) |
| `.issues/941_ledger_correct_sampling_soft_embedding.md` | Position-provenance CONTRACT LIFT (behavior ships in riir-engine's Gemma2 backend; contract + katgpt-rs tests missing) |
| `../riir-ai/.issues/1075_swir_host_penalty_compounding_latent_token_leak.md` | Two verified live defects in the swir real-model host/bench (penalty compounding; latent tokens in scored answer) |
| `.issues/942_latent_pseudo_text_readout_instrument.md` | Cosine nearest-token readout + readability/loop/collapse sensors (report-only) |
| `../riir-train/.plans/461_hybridlink_adapter_recipe.md` | HybridLink recipe distill; E0/E1 minimum experiments; E2/E3 owner-gated |

## 4. References

- HybridAgent — arXiv:2610.18861 (this note's source; MIT, @ `8ad7d8f56a80`)
- HRPO — arXiv:2505.18454 (closest headline prior: GRPO latent/discrete interleaving, no tools)
- Coconut — arXiv:2412.06769 (PASS-redirect lineage: R192; latent core GD-dependent, verdict stands)
- SwiReasoning — arXiv:2510.05069 (R241/P275 — the training-free switching we extend)
- Soft Teacher Forcing — arXiv:2610.17189 (trained implicit-readout exit; surfaced in §4 search)
- Learning when to stop — arXiv:2511.21581 (RL latent length)
- Full-bandwidth transformer — arXiv:2608.08888 (trained latent feedback; PASS verdict R073 — same family, pretraining-scale)
- Deliberation in latent space via cache augmentation — arXiv:2412.17747 (mixed-provenance KV, coprocessor shape)
- Survey — arXiv:2507.06203 "A Survey on Latent Reasoning" (names no tool-integrated latent agent)
