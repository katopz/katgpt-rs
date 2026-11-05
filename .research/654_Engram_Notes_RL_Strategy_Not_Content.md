# Research 654: Engram Notes-RL — Training Compounds Text Memory; the Learned Strategies Are Protocol-Shaped

> **Source:** Engram (company blog), "Learning in a world of notes", <https://engram.com/blog/learning-in-a-world-of-notes>, 2026-11-04. ⚠ **Namesake disambiguation:** NOT the conditional-memory Engram paper of Research 278 (arXiv:2601.07372) — same brand name, entirely different subject.
> **Date:** 2026-11-05
> **Status:** Active — verdict GAIN; §5 AGREE (round 2; round-1 REVISE edits: λ-floor saturation rule, T1.5 memory-reading reward, base pin Kimi→smoke-only, G-B ≤5pp absolute cap, F3 per-predicate precision, G-D deployment-scope line)
> **Related Research:** 278 (namesake only), 500 (EnvHarness — GRPO env-shaping, adversarial-panel precedent), 509 (PAV prover advantages), 037 (REAP model-based/modelless duality)
> **Related Plans:** `../riir-train/.plans/486_notes_usage_rl_cost_sweep_drift_curriculum.md` (the training track)
> **Cross-ref:** katgpt-rs Issue 983 (the modelless track — coverage-manifest gap-check PoC); the escalation value-of-information readout in the private serving lane (the pre-built Pareto/value instrument — named with path in the private plan)
> **Classification:** Public

---

## TL;DR

An industry blog reports that RL (GRPO + LoRA) training of an agent that *reads* a fixed set of text notes adds +7–20pp accuracy over notes+optimized-prompt, matches frontier models given the same notes at ~1/3 the per-query cost (2 of 3 workspaces), costs only 5–30% of what building the notes cost (break-even 0.7–8.7 days at 400 queries/day), and — the headline — a model trained **once** stays ahead of the base model while the notes evolve by **23M tokens** (3× their size): it learned *strategies for using notes* (a conditional gap-check of notes coverage vs raw source, narrow ids-first lookups, link-following), not their *contents*.

**Distilled for the stack (three tracks):**
- **Training track (c):** the recipe is directly adoptable and one seam is already pinned in-tree — `degrpo.rs::game_efficiency_reward` bakes a **fixed** cost coefficient (`0.8 − 0.01·thinking_tokens`, reflex-win tiered 1.0): exactly the hand-pinned λ the source sweeps. → `riir-train` Plan 486.
- **Modelless track (a):** the learned gap-check IS a shell set-difference (`comm -23`); the weights bought only the *disposition* to run it conditionally. The deterministic protocol (coverage manifest + suspicion predicates) is extractable → katgpt-rs Issue 983. The blog's own +7–20pp accuracy delta is **not** modelless-claimable (W1–W6 bounds).
- **Self-adaptive track (b):** pure corroboration — the robustness result is external evidence for the strategy/content separation our stack already practices (Elo/BetaPosterior selection over rules, corpus-lease epoch swaps, artifact staleness probe). No new work.

---

## 1. Source core findings, per claim — with published-prior-art class (§4 search, 13 queries)

| # | Claim | Closest published prior art | Class |
|---|---|---|---|
| C1 | RL-trained note-*user* beats notes+prompt and frontier-with-same-notes | mem-agent (GSPO over markdown-file memory, OpenReview `VguhK2kwHk`); CAMG-RL (arXiv:2609.34422, RL "use files as memory" for coding agents) | **Adjacent** — no exact match; the fixed-corpus + frontier-comparison axis is open |
| C2 | reward = accuracy + λ·token-cost → accuracy/cost Pareto frontier | **OTC** (arXiv:2504.14870, joint correctness+tool-efficiency RL); ToolRL (NeurIPS 2025 reward-design study); Token-Budget-Aware reasoning (arXiv:2412.18547, prompt-level) | **COVERED** at mechanism level — cite, never claim |
| C3 | trained-once robust to 23M-token note drift ("strategies, not contents") | none found; 2026 surveys (arXiv:2603.07670 mechanisms; arXiv:2602.06052 3-axis taxonomy) name staleness/forgetting as **open problems**; no trained-policy-vs-evolving-corpus result surfaced | **Relatively unexplored** — the source's strongest axis, and Plan 486 P2's claim to *prove* on our corpus generations |
| C4 | learned conditional gap-check (set-difference notes-vs-source on suspicion), narrow lookups, link-following | Self-RAG (arXiv:2310.11511, ~4k cites — *trained* when-to-retrieve/critique); Evidence Tracing & Execution Provenance (arXiv:2606.04990 — *scaffolded* memory verification) | **Partially covered** — a learned coverage-**set-difference** on a notes corpus was not found published; the deterministic protocol form (Issue 983) is the stack-novel claim |

Blog's own citations confirmed: Memory-R1 (arXiv:2508.19828 — RL trains memory *management* ops, the note-taking side), Librarian Agents (OpenReview `bqRWdCXKJH` — RL shapes memory *construction*; the inverse setup), Search-R1 (arXiv:2503.09516 — live search engine, not a fixed corpus), Harness Learning (arXiv:2609.35738 — adaptation in code artifacts), GEPA, DeepSeekMath GRPO, on-policy distillation.

---

## 2. Path 0 decomposition — component inventory

| # | Component | Analog ships? | Modelless extraction? | Outcome |
|---|---|---|---|---|
| 1 | GRPO+LoRA estimator (no value model) | **YES** — five variants in-tree: `degrpo.rs`, `pooled_grpo.rs`, `nf_grpo.rs`, `riir-train-gpu/src/loss_grpo.rs`, `tether_grpo.rs` | n/a (training substrate) | covered — no work |
| 2 | λ·token-cost reward sweep | **PARTIAL** — `degrpo.rs::game_efficiency_reward` pins `0.8 − 0.01·thinking_tokens` (verified by grep; reflex tier 1.0, floor 0.3): a hand-pinned λ, never swept | the sweep itself is config, but the *trained frontier* needs rollouts | → Plan 486 P1 |
| 3 | harness-freeze + disjoint train/eval | **YES** as discipline (A5 manifest law, frozen-read eval, ESC_POSTURES boot cross-check) | n/a | covered; Plan 486 P0 pins digests |
| 4 | fixed-notes training + drift curriculum | machinery exists (`continual_lora.rs`, corpus generations) but **no plan treats corpus-generation index as the robustness axis** | n/a (needs GD) | → Plan 486 P2 |
| 5 | 3-arm ablation (notes/no-notes/base) + two-recipe generality | arms exist in `degrpo_training.rs`; ablation not pinned for memory surfaces | n/a | → Plan 486 P3 |
| 6 | GEPA prompt co-lever | **SHIPS** — `katgpt-pruners/src/gepa_reflective.rs` (Research 146 / Plan 164, feature-gated `gepa_reflective`) | n/a | covered; Plan 486 P4 folds it as an arm |
| 7 | coverage manifest + set-difference gap-check | **NO runtime analog** (§5 signal-diff: cousins are static reports, artifact-space probes, or distance gates) | **YES** — deterministic protocol | → Issue 983 |
| 8 | suspicion trigger predicate | **PARTIAL** — `CorpusDistanceGate` = embedding distance; ESC gate flags = output shape; count-vs-census mismatch = none | **YES** — closed-form predicate over pinned counts | → Issue 983 |
| 9 | truncation-refusal rule (`returned == cap` ⇒ read untrustworthy) | unmeasured in retrieval lanes | **YES** — 100% mechanical | → Issue 983 scope |
| 10 | break-even formula (capital / daily savings) | no standing probe | **YES** — closed-form, report-only | → Plan 486 P5 gate line |
| 11 | strategy/content two-lane schema law | **YES implicitly** — rules corpus vs fact shards, corpus-lease epochs, the two-brain split | design principle, already practiced | corroboration, note-level |
| 12 | λ-sweep frontier instrument | **YES (observer shape)** — the escalation value readout (private serving lane; report-only rate/precision/value legs + CI) | the sweep framing joins the instrument | → Plan 486 P1 output format |

---

## 3. Distillation — three tracks

### Track (c) trained weights → `riir-train` Plan 486

Recipe items (model-based advocate table, verified in-tree), priority-ordered:

| Pri | Item | Consumer | Delta |
|---|---|---|---|
| P1 | λ-cost sweep on the GRPO reward | `degrpo.rs::game_efficiency_reward` (the fixed 0.01) | config-swept `λ_cost`, default byte-identical to today; sweep {0, 1e-3, 1e-2, 1e-1} → Pareto table; 10–30 GPU-h on a ≤1B base (Kimi-K3 0.4B / MiniCPM5-1B), rollouts offloaded to bmo |
| P2 | drift-robustness curriculum | `continual_lora.rs` + corpus-generation machinery | train gen-N, eval frozen gen-N+1..N+k (k≥2) — **hot-swap corpora with NO reader retrain is the deployment-enabling property**; rank sweep {8,16,32} × memorization probe (high rank + flat retention = content memorization → reject) |
| P3 | 3-arm ablation + recipe generality | `degrpo_training.rs` arms | with-corpus / without-corpus / base on one frozen read; second corpus recipe arm |
| P4 | GEPA co-lever | `gepa_reflective.rs` (feature-gated) | aim the bandit at the trained lane's serving config BEFORE RL; two paired reads |
| P5 | break-even gate + 27B posture | plan discipline | print `training_$ / daily_hosted_savings_$` before any >10 GPU-h spend; **27B QLoRA deferred behind this gate** (100–400 GPU-h on the 4090 at batch 1–2; Bonsai-2-27B PQ2_0 as a *training* base = research project — no LoRA-over-frozen-ternary kernels — out of scope) |

**Deployment honesty (the advocate's flagged weakness, adopted):** the one genuine consumer is **the private serving lane's ESC lanes** — hosted-spend substitution at equal accuracy (the value/Pareto instrument lives there; a trained arm enters as a candidate under the serving law, T2 strict superiority vs BOTH modelless and laya arms on one frozen read). The refine fixer's `l4_lora` seam is high-bar (0/60 measured precedent). The game side is toy. The cost claim bites only where hosted tokens are paid; against in-process modelless lanes it is void. The honest value proposition: **hosted-spend substitution + corpus-drift survival without retraining**, nothing broader.

### Track (a) modelless → katgpt-rs Issue 983

The blog's Figure-8/10 traces show the trained model running `comm -23 dir_ids.txt meta_ids.txt` — the check is a deterministic set-difference; only its conditional firing is weight-carried. Modelless form (full mechanism + falsification gates in Issue 983): the memory **builder** stamps a coverage manifest (covered-key sets/counts + source-census digest); the **reader** fires deterministic suspicion predicates (count-vs-census mismatch, truncation-at-cap, coverage-deficient section) and acts by set-difference (targeted gap lookups / escalate to primary source / emit a coverage-gap event). Enforced-by-design ordering law: firing rate ∝ monotone(1 − coverage), ≈0 at complete coverage.

**Requires-weights bounds (panel W1–W6, adopted verbatim in the issue):** semantic suspicion beyond enumerable anomalies (W1), open-ended query composition (W2), selective link-following (W3), the +7–20pp delta itself (W4), calibrated doubt on plausible-but-wrong notes (W5), and the Hex capability ceiling (W6 — training-with-notes still failed to match frontier there) are weight-carried. The protocol claims the **cost axis and the coverage-accounting axis only** — never accuracy parity.

### Track (b) self-adaptive → corroboration, no files

`self_evolve`'s Elo/BetaPosterior selection is strategy-not-content at the selection layer; corpus-lease epochs are the drift axis the blog's staged-note-reveal replay harness mirrors; `riir-instinct/src/staleness.rs` is the frozen-procedure-holder probe shape (artifact-vs-reference divergence, REPORT-ONLY). The blog's robustness result is external evidence for the strategy/content separation this stack already practices. Recorded as corroboration; no new work.

---

## 4. Fusion

**notes-RL × ESC ladder × corpus lease** — a trained cheap-lane reader that survives corpus swaps without retraining (Plan 486 P2's property) + coverage-manifest gap events as an ADDITIONAL escalation trigger beside the ESC gate flags (Issue 983's predicate class). None of the three alone has this: the blog has no escalation ladder; ESC has no trained reader; the corpus lease has no trained-reader drift guarantee. The combination is carried as Plan 486 P2's consumer framing + Issue 983's ESC consumer row; any serving adoption is owner-gated per the serving law (a trained arm is a candidate like any other).

---

## 5. Signal-diff cousin map (§3.6 — every coverage dismissal carries its signal)

| Cousin | Signal it consumes | Diff vs the gap-check primitive |
|---|---|---|
| `riir-refine` coverage-report | lint-API surface coverage (clippy-driver `-W help` diffed vs corpus), static report | API-surface × static; NOT memory-vs-source, NOT runtime, NOT query-time |
| `riir-refine --corpus-delta` | oracle-feed hits partitioned "not leased YET" vs "unsupported" | requires an external clippy JSON feed; partition of *hits*, never a set-difference against the workspace inventory |
| `riir-instinct` staleness.rs | artifact-vs-reference divergence on a pinned probe set (fire: flips>0 OR mean\|Δgold\|≥0.02) | artifact-space impact read; NOT source-vs-memory coverage |
| `riir-reflex` CorpusDistanceGate | input embedding distance from corpus | abstains on distance; set-membership coverage is a different signal — complementary (abstain vs escalate) |
| ESC value readout (private serving lane) | escalation rate / precision / value-per-escalation CI | the measurement instrument for the *frontier*; no coverage accounting |
| two-brain spatial cognition | observation-gated belief updates (`visible_radius` re-observation, confidence decay) | re-observation trigger is visibility, not census-vs-belief set-difference; the gap list concept is absent |

**The gap stands:** no shipped surface runs a memory-vs-source set-difference at decision time on a suspicion trigger. Bounded by W1 (the semantic suspicion tail is weight-carried).

---

## 6. Verdict

**GAIN** — actionable on two tracks with verified in-tree seams; not Super-GOAT: Q1 is bounded by published prior art (OTC covers the λ mechanism; Self-RAG occupies when-to-consult; CAMG-RL/mem-agent occupy file-memory RL), and the composite drift-robustness axis — while relatively unexplored in the literature — is Plan 486's *pre-registered claim to prove*, not a today claim. Per-track separation (TTPO law): the training track's tier is independent of the modelless track's, and vice versa.

- **Training track:** → `riir-train` Plan 486 (GOAT-gated; gates G-A..G-D pre-registered in the plan before any run). Path 0.5 satisfied: recipe + GPU-hours + GOAT gates vs modelless baseline; Path 0 decomposition recorded (rows 2/4/5/10 lack modelless analogs *because they need rollouts + gradient updates*, everything extractable was extracted).
- **Modelless track:** → katgpt-rs Issue 983 (fusion idea, novelty TBD pending the PoC precision gate F3). No Super-GOAT guide now — ungoat discipline: smallest artifact that carries the verdict.
- **Corroboration:** this note (track b).

**MOAT gate:** katgpt-rs = generic protocol primitive (Issue 983) + this public note — in scope. riir-train = training-method plan (active moat) — in scope. Consumer-first check for riir-refine: named as an Issue-983 consumer surface; no refine file ships from this verdict.

**Proof paths (ungoat discipline):** Plan 486 gates G-A (λ-Pareto dominance), G-B (drift retention bar + memorization probe), G-C (ablation ordering), G-D (break-even line, expected-FAIL scoping of the 27B defer). Issue 983 gates F1 (set-equality, two-derivation), F2 (stamp-time cost), F3 (trigger precision + ordering law), F4 (staged-replay determinism + stale-census loud refusal).

---

## 7. Adversarial panel record (§3.5 — three tracks, one spawn round)

- **No-GD advocate:** 14 extraction candidates + W1–W6 requires-weights table. **Adopted:** candidates folded into Issue 983 (coverage manifest, suspicion predicate, ordering law, truncation refusal, ids-first grammar, trace linter) and Plan 486 P5 (break-even formula); frontier-instrument framing joined Plan 486 P1; the rest covered/corroboration.
- **Model-based advocate:** 10 recipe items R1–R10 with in-tree consumers. **Adopted:** R2/R4/R5/R6/R8/R9 → Plan 486; R1/R3/R7 covered-no-work; R10 deferred behind G-D with the advocate's own feasibility math (100–400 GPU-h).
- **Web prior-art searcher:** 13 searches; per-claim classes adopted into §1. Snippet-level caveat recorded in §8.
- **Discard audit:** **none discarded** — every advocate finding landed in a file, a covered-row, or the requires-weights bounds.
- **Coordinator corrections (grep-verified):** the model-based advocate called `gepa_reflective.rs` "default-on" — it is **feature-gated** (`#[cfg(feature = "gepa_reflective")]`, Research 146/Plan 164); the escalation value readout is feature-gated in its private lane (reported correctly); the R2 seam fixed-coefficient claim verified byte-exact.

## 8. Caveats / re-open triggers

- The §4 searches were snippet-level; Memory-R1 / mem-agent / CAMG-RL internals were not PDF-verified. **Re-open trigger:** if any is found to test fixed-corpus usage under content drift, Plan 486 P2's novelty framing downgrades (recipe adoption is unaffected — the plan's value is the property proof on OUR corpus generations, not literature priority).
- All λ-sweep magnitudes are the blog's, on their workspaces. Our gates are pre-registered against OUR floors (G-A/G-B); magnitude transfer is not implied (the advocate's baseline-gap note: our floors are tuned modelless engines, not bare base models).
