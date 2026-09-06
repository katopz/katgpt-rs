# Research 535: XBridge — Entity-Grounded Latent Bridge for Heterogeneous LLM Communication

> **Source:** "XBridge: Entity-Grounded Latent Bridge for Heterogeneous LLM Communication" — Yang, Huang, Zhang, Wang, Yu, Lee (UIC / Capital One / HKUST / Noah's Farm), arXiv:2608.11676, 2026-08-12. Code: github.com/WooseongYang/XBridge
> **Date:** 2026-09-06
> **Status:** RECORD — modelless extractions filed as riir-clippy Issue 073; training recipes + cross-model bridge trigger filed as riir-train Issue 523
> **Related Research:** 178 (Rosetta Neurons cross-model alignment — closest cousin), riir-ai 133 (NPC mind reading, cross-model KV adapter → riir-train), riir-ai 364 §5 (training-track trigger R3 "cross-model shared-latent serving"), riir-clippy 121 walk-#6 row 12 (arXiv:2608.30963 graded C)
> **Related Plans:** none (no plan — no consumer for the trained core; issue-level routing)
> **Cross-ref (riir-ai / riir-train):** riir-train Issue 523; riir-clippy Issue 073; riir-ai Benches 868/864 (gist negative), riir-train Bench 422 (Procrustes NO-GO)
> **Classification:** Public

---

## TL;DR

XBridge identifies the **entity grounding problem** in heterogeneous LLM→LLM communication: continuous bridges (projections, cross-attention) across model families preserve contextual semantics but lose discrete entity identity — **rare-token compression collapse** (bridge-only F1 ≈ 30%). The fix is a **dual-channel decode-free protocol**: (1) **LAM** — deterministic, training-free cross-vocabulary token mapping (ID-to-ID lookup + string re-tokenization fallback) placing sender context tokens as *receiver-native* discrete anchors; (2) **LEB** — a small trained gated cross-attention bridge (264M params, 3.8% of receiver, 587 balanced samples, <10 min) letting the receiver query the sender's last-layer hidden states. The measured role separation is the transferable finding: **the discrete anchor channel determines *which* entities appear in output; the latent channel determines *how* the receiver reasons about them** (entity-perturbation intervention, §4.4). Beats text communication on 7/7 tasks × 3 model pairs at 11× lower latency; beats same-architecture KV sharing on 6/7.

**Distilled for our stack (modelless, inference-time):**
1. **Anchor-primacy rule** — identity-critical fields (names, ids, codes, numbers) crossing any compression boundary (prompt, summary, embedding, KG) must cross as *discrete receiver-native anchors*, never inside a continuous summary. Continuous channels carry enrichment; anchors carry identity.
2. **The entity-perturbation eval** — a falsifiable identity-preservation test for ANY compression surface: swap the entity in the input, assert the output follows the swap. We ship NO such test anywhere (grep-verified gap).
3. **LAM cross-vocabulary mapping** — a trivially modelless primitive (deterministic vocab translation with lossless re-tokenization fallback); no consumer today (all our drafter/verifier pairs share vocabularies), recorded with trigger.
4. **Rare-token compression collapse** — the named mechanism behind our own Bench 868 negative (dense gist 0.93× vs plain prose: the gist is a continuous bottleneck that loses rule ids/error codes first).

---

## 1. Paper Core Findings

- **Entity grounding problem**: cross-architecture bridges transfer context but not identity. Formalized as an identifiability failure — two contexts differing only by entity substitution produce near-identical continuous messages (`d(m_Ce, m_Ce') ≤ ε`) while the required receiver-native anchors differ (`a_R(e) ≠ a_R(e')`).
- **LAM** (modelless): precomputed per-tokenizer-pair mapping; Llama3.1→Qwen2.5 direct-maps 97.1% of tokens (85.4% shared vocab), Mistral→Qwen 87.8% (32.2% shared); fallback is lossless surface-string re-tokenization; <1 ms.
- **LEB** (trained): 4 gated cross-attention modules at receiver layers {6,13,20,27}, receiver hidden state = query, sender last-layer states = K/V, `h' = h + tanh(α)·A`, **warm init α=1.0** (beats Flamingo zero-init 78.8 vs 72.7 F1 at 587 samples).
- **Training-data composition**: 587 BALANCED samples (~100/task) beat 42K unbalanced (63.2 vs 59.3 avg F1) and 20K single-task (49.5) — balanced small data wins; task-gated gate behavior emerges (near-closed on saturated tasks, active on reasoning tasks).
- **Sender layer**: bridge quality monotonic in sender depth — last layer optimal (L31 78.8 vs L8 64.6).
- **Role separation** (the load-bearing intervention): swapping LEB's input alone never changes the output entity; swapping LAM's input always does. LAM = "which entities", LEB = "how to reason about them".
- **Zero-shot composability**: two independently trained bridges (Llama + Mistral senders) combine at inference: 70.4 vs 67.0 single-sender vs 56.8 dual-NLComm.
- **LAM is sender-size invariant** (1.5B sender reaches 59.5 ≈ FullComm) — identity transport is capacity-free.

## 2. Path 0 Decomposition + Signal-Diffs (per §3.5/§3.6)

| # | Component | Coverage in stack? | Modelless-extractable? | Signal-diff vs shipped cousin | Disposition |
|---|---|---|---|---|---|
| 1 | LAM cross-vocab deterministic mapping | NO mapper shipped (`cross.vocab\|vocab.*map\|tokenizer.*map` = 0 as cross-model translation) | YES (pure deterministic; `katgpt-tokenizer` + GGUF `BpeTokenizer::from_gguf` are the substrate) | N/A — no cousin | Open primitive, **zero consumer** (all drafter/verifier pairs share vocab; in-vocab contract `ngram_drafter.rs::with_vocab_size` is memory-safety, not translation) → riir-train Issue 523 trigger + recorded here |
| 2 | Dual-channel anchor⊕enrichment | PARTIAL | YES (architectural principle) | **riir-rag hybrid** fuses at RETRIEVAL time (`rrf_fuse` K=60 over BM25 rank ⊕ latent KNN rank; `FusionStrategy` "swaps the sort key only"); scenelab `trace_index.rs:866` is a second retrieval-time site. XBridge fuses at GENERATION time (anchors in the prompt + cross-attn enrichment). Bench 068's same-rule channels are generation-time but carry no pinned anchor-primacy rule and no perturbation gate | Sharpening → riir-clippy Issue 073 |
| 3 | LEB trained gated cross-attention | INTRA-model analog ships: `LoraStillCompactor::cross_attention_synthesis` (riir-engine `lora_still/compactor.rs:185` + riir-gpu `lora_still_forward.rs:111`) — latent-bank queries attend teacher KV. Cross-model: NO | NO core (translation needs GD) | Signal-diff: LoRA-Still consumes teacher KV to *synthesize adapter weights within one family*; XBridge LEB conditions a *runtime answer across families*. `Flamingo`/`gated residual` = 0 workspace; our `tanh`-gate hits are MoE softcap arithmetic, not gated residuals | Training track → riir-train Issue 523 (defer-with-trigger; see §4) |
| 4 | Warm gate init α=1.0 | NO (Flamingo = 0 everywhere) | Recipe | — | riir-train Issue 523 recipe backlog |
| 5 | Balanced-sample curriculum (587 > 42K unbalanced) | Not pinned | Recipe | Maps to quest_grammar/edge_lora small-data regimes + Bonsai SFT corpus composition | riir-train Issue 523 |
| 6 | Last-layer teacher taps; 4-module density | `DSPARK_CAPTURE_LAYERS` / `memory_targets_raw` tap specific layers, choice not derived from this finding | Recipe | — | riir-train Issue 523 |
| 7 | Entity-perturbation eval | **NO — gap confirmed** (`perturb` hits are index-chaos/FD-sweep/routing-independence senses; "swap a symbol, assert the output follows" = 0) | YES — falsifiable A/B methodology | Closest in spirit: `bench_279:442` perturb-one-expert-row independence gate — routing independence, not identity following | riir-clippy Issue 073 (T2) — the strongest modelless extraction |
| 8 | Rare-token compression collapse (named failure mode) | Unnamed but MEASURED: riir-ai Bench 868 — dense gist 0.93×/0.95× vs plain prose (2.0× bar); Issue 852 closed-negative with reopen trigger "materially different compressor" | YES — explanatory principle + test axis | Bench 868 measured the loss; XBridge names and isolates the mechanism (identity vs enrichment) and predicts the fix shape (verbatim anchor block alongside prose) | riir-clippy Issue 073 (T3: Bt8 = prose + verbatim anchor ledger) |

**Three-track panel (folded into the table):** No-GD advocate findings = rows 1, 2, 7, 8 — all filed (modelless extractions, nothing discarded). Model-based advocate findings = rows 3–6 — filed as riir-train Issue 523. **Audited discard:** "implement LEB cross-model bridge now" — reason: no consumer surface exists (no shared-latent serving path), the pre-existing trigger R3 (riir-ai Research 364 §5) is not fired, and measured negative priors bound the value: `gemma2_directions` GOAT FAIL (~9 s/embedding), L4 fixer 0/60 at ~191 s/fix, and Bench 422's cross-arch Procrustes NO-GO (mean cos ≤ −0.08).

## 3. Landscape (prior-art reconciliation)

The latent-MAS communication field is dense as of 2026; XBridge's delta over each cousin is the **asymmetric setting** (sender sees context, receiver sees question) + the **discrete-anchor channel**:

- **LatentMAS** [arXiv:2511.20639] — training-free latent MAS via shared KV caches; same-architecture assumption.
- **Dense Latent Communication Across Heterogeneous Agents** [arXiv:2606.13594] — aligned KV-cache comms across families; continuous-only (exactly the surface XBridge shows collapses on identity).
- **KVComm** [NeurIPS 2025] — training-free cross-context KV reuse; matching architectures.
- **Universal Context-Reuse** [arXiv:2608.30963] — cross-model KV sharing for serving; graded **C not-mineable** in riir-clippy Research 121 row 12 (trained translator, heterogeneous-serving framing, no stack consumer). XBridge reconciles: same trained-component cost class, but adds a modelless discrete channel + the entity-grounding diagnosis; still no stack consumer → stays issue-level, does not overturn the C grade for 30963 itself.
- **Causal audit of latent communication** [arXiv:2608.04893] — audits whether KV-relay gains are causal; relevant skepticism for any future bridge GOAT gate (our perturbation/intervention gates answer the same demand).

## 4. Fusion

### Fusion A — Healer (investment #2, the actionable surface) → riir-clippy Issue 073

The paper × Bench 068 × Bench 868 produces what none has alone: Bench 068 proved same-rule anchor channels help (6/10 vs 5/10 strict-keep) but pinned no rule for WHY; Bench 868 measured the dense-gist loss without a mechanism. XBridge supplies the unifying principle (anchor-primacy + role separation) and the missing falsifiable test (entity perturbation):

1. **Anchor-primacy rule**: lint ids, rustc error codes, symbol names, span ranges cross fixer prompts / TraceQuery / gists VERBATIM in a dedicated anchor position — never paraphrased, never summarized, never only inside an embedding.
2. **Entity-perturbation fixture class**: swap the error code/symbol in the frozen fixture context; assert the healed fix follows the swap. A prompt builder or compressor that paraphrases away the anchor fails loudly.
3. **Bt8 gist variant**: plain prose + a verbatim anchor ledger (ids/codes/symbols as discrete fields) vs Bench 868's prose baseline — a "materially different compressor" that fires Issue 852's documented reopen trigger.

### Fusion B — Game runtime (investment #1: validation, not new work)

The stack already implements the XBridge shape at the NPC layer — this paper is external validation plus one sharpening:

- **KG triples** (`kg_gate.rs`: latent sigmoid gate + raw corroboration → discrete triple; scenelab `svo.rs` name-anchored triples) = the discrete anchor channel; **zone attention / HLA scalars** = the enrichment channel. The two-brain model (raw info brain / latent think brain) is the same decomposition at the cognition layer.
- **`RefTag` verbatim anchors** (riir-agents `shared_context.rs`: admission requires every anchor verbatim + BLAKE3 recompute) is already a *stronger* anchor-integrity discipline than LAM (which trusts the mapping).
- Sharpening (recorded, not filed — thin gain): an entity-perturbation gate on `kg_gate` (perturb the observed entity, assert the emitted triple follows) would pin identity-following in KG emission the way Bench 068's gates pin channel placement. File with the next kg_gate touch.
- **Sync-boundary corroboration**: the paper's collapse mechanism is a new argument for the existing raw-sync rule ("never encode position as embedding then decode back") — continuous bottlenecks lose rare-token identity FIRST, which is exactly the anti-cheat/anti-replay hazard class.

### Fusion C — Training track → riir-train Issue 523

LEB's core is genuinely training-bound (learned cross-space translation; the deterministic variant is refuted twice — XBridge's own bridge-only ≈30% F1 AND our Bench 422 Procrustes NO-GO). **Bench 422 refinement worth recording: the NO-GO kills deterministic LINEAR alignment; XBridge is the published counter-shape — a LEARNED cross-attention + discrete anchors succeeds where linear alignment fails. Any future bridge attempt must be learned + anchored, never Procrustes.** Recipes (warm gate init, balanced curriculum, last-layer taps, 4-module density) filed as backlog; the cross-model bridge itself stays defer-with-trigger per Research 133 R3 (trigger: a real cross-family consumer where text handoff measurably fails).

## 5. Verdict

**Tier: Gain.** One-line reasoning: the dual-channel principle is now published territory (§3), no new behavior class or product selling point emerges for the stack, but four modelless extractions are actionable (anchor-primacy rule, perturbation eval axis, Bt8 compressor, LAM-with-trigger) and four training recipes map to live pipelines.

**Novelty gate:** Q1 (prior art) — the principle is published; in-stack fusion is unclaimed but is engineering, not research novelty. Q2 (new behavior class) — no. Q3 (selling point) — cannot finish "our X does Y no competitor can". Q4 (force multiplier) — yes (retrieval + prompts + eval + training). 1 YES + 1 partial → not Super-GOAT; actionable → Gain.

**MOAT gate per domain:** katgpt-rs = principle note + LAM recorded with trigger (no flag without a benchmark consumer). riir-ai = validation of shipped designs; kg_gate perturbation idea recorded. riir-clippy = the actionable surface (Issue 073, consumer-first moat). riir-train = recipes + trigger (Issue 523).

**Pinned claim (pre-search):** "Generation-time dual-channel composition (discrete receiver-native anchors + continuous enrichment) with an entity-perturbation eval axis, for the healer's fixer-prompt/retrieval surfaces, consuming identity-critical fields (lint ids, rustc error codes, symbols), distinguished from riir-rag's retrieval-time RRF fusion by operating at prompt-composition time and from Bench 068's same-rule channels by adding a falsifiable identity-following gate."

## PASS-Redirects / closest-cousin cross-refs

> **Closest cousins:** Yang et al. [arXiv:2608.11676 "XBridge: Entity-Grounded Latent Bridge for Heterogeneous LLM Communication"] — this note; dual-channel anchor⊕enrichment, Gain. LatentMAS [arXiv:2511.20639 "Latent Collaboration in Multi-Agent Systems"] — same-family KV sharing, no anchor channel. [arXiv:2606.13594 "Dense Latent Communication Across Heterogeneous Agents"] — the continuous-only surface XBridge refutes. [arXiv:2608.30963 "Universal Context-Reuse"] — riir-clippy Research 121 row 12, C not-mineable, reconciled §3.
