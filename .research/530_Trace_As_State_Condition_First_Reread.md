# Research 530: Trace as State — Condition-First Reread for Causal Long-Context Processors

> **Source:** [Trace as State: Reasoning Traces as Conditional States for Long-Context Transformers](https://arxiv.org/abs/2609.02702) — Xu Zou (Z.ai), Jie Tang (Tsinghua), arXiv:2609.02702 [cs.CL], submitted 2026-09-02. CC BY 4.0.
> **Date:** distilled 2026-09-04
> **Verdict: GAIN** — an inference-time prompt-placement protocol with a clean worst-case memory theorem. No in-stack consumer exists for the protocol itself (no reasoning model with exposed traces in the serving path; no long-context quality bench; ~2× prefill cost), but it **recalibrates a closed negative** (riir-clippy Bench 051) and supplies the placement + relevance-gating rules for every future trace-in-prompt surface. Actionables filed: riir-clippy `.issues/067`.
> **Status:** RECORD — issue 067 filed same day; cousin notes 313/512 updated with redirects. Re-open triggers recorded in §6.
> **Related Research:** 313 (Thinking-to-Recall — fact-only conditioning), 512 (Meta^n growing-input), 213 (StillCoT trace folding), 295 (AcPrefix prefix-conditioning), 525 (TRACE halting — trace *length* axis; this paper is the trace *placement* axis)
> **Classification:** Public

---

## TL;DR

The paper makes two claims. **Theory:** for *conditional state update tasks* — a causal processor reads an information sequence `C` and must apply a condition `z` (the initial state) — reading order is a memory-order separation: condition-first `[z,C]` needs `⌈b⌉` bits of working state (`b = log₂|S|`), condition-last `[C,z]` needs `⌈b·2^b⌉` bits in the worst case (Appendix A realizes the bound with all self-maps of `S` at `n=1`). Causal transformers with finite context + finite precision ARE causal state-update processors, so the separation applies. **Method (trace as state):** run `n_tr` first passes, collect the reasoning traces, serialize them as `T` (fixed labels, 50k-char truncation per trace), then a fresh second pass receives `[T, x, q]` — trace BEFORE the long context — versus the matched control `[x, T, q]` (trace append). Order is the only variable.

**Results:** trace-as-state beats trace-append in **26/27** model×task×metric cells (3 frontier models × GraphWalks-256K / MRCRv2 / NUB-1M). GraphWalks Parents EM: DeepSeek V4 Pro 29.2 → **81.8**, Qwen 3.7 Max 60.8 → **96.4**, GLM-5.2 66.4 → **100.0**. Paired CIs fully above zero in 20/24 cells. It beats Re2 (repeat-question reread: 50.0 vs 81.8 Parents EM), beats Answer-Feedback (answers-only conditioning: 45.4), **beats Oracle@5** (best-of-5 first passes: 50.0 → 81.8), and beats Trace-Only (no context reread). **Random Trace (traces from other problems) scores BELOW the no-trace first pass** (14.2 vs 29.2 Parents EM) — weakly-relevant conditioning content is actively harmful. Trace count 1→5 (same problem) is monotone.

**Distilled for the stack (modelless, inference-time):** four transferable rules —

1. **Placement principle:** derived task state (traces, evidence, conditioning text) belongs BEFORE the context on any reread pass, never after. Worst-case justification is exponential; empirical justification is dominant (26/27).
2. **Relevance hazard:** conditioning content that is topically similar but not task-relevant is net NEGATIVE (Random-Trace arm). Content gating beats content volume.
3. **Process evidence > outcome evidence:** full reasoning traces ≫ final answers alone as conditioning state (Answer-Feedback arm: 45.4 vs 81.8 Parents EM).
4. **Cost model:** ~2× total tokens (first pass + reread); prefix-cache reuse degrades in multi-round settings because the trace changes the prefix (the paper's own Limitations).

---

## 1. Paper core findings (compressed)

| Finding | Number |
|---|---|
| Memory separation | `[z,C]` = `⌈b⌉` bits; `[C,z]` = `⌈b·2^b⌉` bits worst case (exponential in state-space size) |
| Placement dominance | Tas > Ta in 26/27 cells; ≥ +16.8 EM in every GraphWalks-Parents cell |
| vs rereading alone | Re2 `[x,q,x,q]` improves over first pass (31.6→49.0 BFS EM) but stays well below Tas (58.8) — repetition ≠ state |
| vs answer conditioning | Answer Feedback `[a,x,q]` 35.4/45.4 — answers are much weaker state than traces |
| Random content harms | Random Trace `[T_rand,x,q]` 22.4/14.2 — BELOW first pass (31.6/29.2) |
| Trace-only ≈ append | `[T,q]` ≈ `[x,T,q]` (46.2 vs 41.8 BFS EM) — the trace carries answer-relevant info, but the original context placed AFTER the trace is still worth rereading |
| Count scaling | same-problem traces n_tr 1→5 monotone (for every n_tr ≥ 1, Tas > Ta) |
| Question-first | `[q,x,q]` +23.8 EM Parents over first pass — the question itself is task state for some tasks |
| Cost | ~2× tokens; e.g. DeepSeek GraphWalks: 51.1M missed-input (first pass) → 113.6M (Tas total) |

Prior-art anchors (from §4 search): RE2 (Xu et al., EMNLP 2024, arXiv:2309.06275 — repeat-question, no derived state, no order-isolating control), S2A (arXiv:2311.05732 — first pass *replaces* context, opposite direction), least-to-most (ICLR 2023 — derived answers as *suffix*), MemoRAG (2024 — trained compressor, memory feeds retrieval not prefix), Gist tokens (NeurIPS 2023 — learned prefix), ReContext (arXiv:2607.02509 — recursive evidence replay; cited by the paper), Ok & Lee 2026 (prompt-order sensitivity), Hao et al. 2026 / State-over-Tokens (traces carry state but models under-report using it). The paper's own delta is honest and real: the **order-isolating matched control** (`[T,x,q]` vs `[x,T,q]`, same T, same fresh pass) had no published antecedent, and the formal separation had no published statement for causal processors. The theorem's information-theoretic core is INDEX-problem-shaped (one-way communication complexity: condition-first `O(log n)` vs condition-last `Ω(n)` — folklore-adjacent), so the contribution is the formalization + the protocol, not a new lower-bound technique.

## 2. Why this matters for our stack (three reframes)

**Healer reframe (priority #2 — the only surface with a live measured negative to recalibrate).** riir-clippy Bench 051 ran the closest experiment we have: arm G appended trajectory outcomes + failure patterns AFTER the code module (`prompt_growing`, channels spliced before the closing line). Verdict `NEGATIVE-no-gain`: strict-keep tied 5/9 across arms A/B/G at 2.8× arm-B tokens; channels were **cross-rule** (no same-rule history exists — the reachable lints are off-corpus by construction). The paper recalibrates that negative along two axes without overturning the closed verdict: (i) **placement** — the channels were condition-LAST relative to the code (the paper's controlled variable); (ii) **relevance** — cross-rule exemplars are exactly the Random-Trace hazard class the paper shows is net harmful, so Bench 051 may have measured positive-channel × negative-relevance cancellation, not "growing input doesn't help". Both deltas are cheap to falsify when the trigger condition exists (same-rule exemplars in the TrajStore) → riir-clippy `.issues/067`. The Answer-Feedback arm additionally predicts the few-shot design already in `prompt_prepass` (before/after fix pairs = outcome evidence) is weaker conditioning than process evidence would be — noted, not actionable at current store density.

**Game/NPC reframe (priority #1 — honest answer: validation, not capability).** The two-brain model already embodies the principle: think-brain belief (derived state) gates perception (evidence processing) via `visible_radius` — i.e. state is available *before* evidence is interpreted, one-way, exactly the ordering the paper proves is the cheap direction. The bridge rules (belief updates only within `visible_radius`; think brain never writes info brain) already respect the causal-state-update structure the theorem formalizes. No new per-NPC scalar, no new behavior class, no selling point that a competitor lacks. Recorded as design validation.

**Inference-perf reframe (priority #3 — a quality mode we cannot currently serve).** Trace-as-state is a Cold-tier quality lever: ~2× prefill+decode for large accuracy gains on long-context reasoning. Our serving path (Bonsai-27B ternary) is (a) throughput-constrained (the league measures tok/s), (b) a base model that emits no reasoning traces — the paper's protocol REQUIRES an exposed trace interface (their own Limitations). The CF/GLM lanes (Bench 057's GLM-5.3-flash fixer) do have reasoning models but no long-context task. No consumer today. Trigger conditions in §6.

## 3. Signal-diff table (§3.6 — closest shipped cousins)

| Paper mechanism | Closest shipped analog | What the analog consumes | Delta (the gap) — audited disposition |
|---|---|---|---|
| Condition-first prefix placement | `AcPrefix` (katgpt-core, DEFAULT-ON; R295/Plan 313/riir-ai Plan 398 bridge) | designated conditioning tokens `xc` copied to the front with original-position RoPE, leakage-controlled, single-pass `p(xe\|xc)` scoring | AcPrefix is the *placement machinery* (right position discipline); trace-as-state is a *two-pass protocol* where the conditioning content is generated by pass 1 itself. Composition (pass-1 trace → front placement → reread) is unoccupied — but unreachable without a model that emits traces (below). **Not filed:** no executable surface. |
| Two-pass collect-then-condition (latent) | `SelfCondDraft` (katgpt-core `dllm_solver.rs` L354–440) + DFlash multi-pass fusion | pass-1 marginals → best-path reinforcement into an SC buffer → pass-2 marginals conditioned on pass-1 output | Latent marginal blending (lossy) vs textual trace prefix (near-lossless). The paper's Answer-Feedback ≪ Trace ablation is evidence that lossy distillation of pass-1 state leaves gains on the table — a real signal-diff against `SelfCondDraft`'s best-path reinforcement. **Audited discard of the "textual SC variant" idea:** requires an in-stack model emitting trace text (Bonsai = base model; the dllm lane is a latent drafting component feeding spec-decode, where a textual prefix would break the latent pipeline). Revisit trigger: a reasoning-capable in-stack model (§6). |
| Trace text in an inference prompt (healer) | `prompt_growing` (riir-clippy `cf_workers_ai.rs` L404–453; Bench 051) | trajectory outcomes + failure patterns appended AFTER the code module | Ran; `NEGATIVE-no-gain` (5/9 ties, 2.8× tokens). Paper adds two falsifiable deltas: condition-FIRST placement, and same-rule relevance gating (Random-Trace hazard). → **riir-clippy `.issues/067`** (the one filed action). |
| Fact-extracted conditioning | R313 Thinking-to-Recall ("fact-only conditioning recovers most of CoT's gain") | concrete facts extracted from the trace → short fact-list conditioning | Same family; 313 isolated *what* to condition on (facts), this paper isolates *where* (before vs after) and proves the order matters exponentially. 313's latent reframing (anchor injection) remains the in-stack realization path. Redirect appended to 313. |
| Conditioning-order sensitivity in two-pass pipelines | F-check class (riir-ai Issues 669/749/860: draft token `i` scored against position `i+1`; precompute_cache off-by-one; ring-inject once-per-position) | KV/verify-layer slot alignment between condition and prediction | The empirical shadow of the paper's theorem at the KV layer: *where conditioning sits* in a two-pass pipeline silently invalidates results. The theorem is the formal account; the F-checks are the recurring bug class. Note-level connection only. |

## 4. Per-track dispositions (auditable)

- **Track (a) modelless inference:** the protocol is modelless (prompt scheduling; no weights touched). Generic serving-primitive implementation **not filed** — audited discard: zero consumers (no in-stack trace-emitting model; no long-context quality bench; the league is throughput-only and 2× prefill is a league loss). §6 records the re-open triggers.
- **Track (b) self-adaptive runtime:** no interaction (the protocol mutates no latent state; the traces are external text).
- **Track (c) model-based training:** the paper trains nothing (frozen frontier models, inference-only). The conclusion's "a broader training framework could jointly learn the model and the feedback interface" is future-work hand-waving with no recipe, no loss, no schedule. §3.5 Path 0.5: **not applicable — genuinely no training content.** No riir-train plan. (Auditable one-liner, per the training-papers third defense.)

## 5. Verdict + MOAT gate

**Tier: GAIN.** Not PASS — two actionable calibrations exist (Bench 051 reopen conditions; the placement/relevance rules for future prompt surfaces). Not GOAT/Super-GOAT for the stack — the protocol has no in-stack consumer; the theorem is recorded principle, not implementable primitive.

| Domain | MOAT fit | Action |
|---|---|---|
| katgpt-rs | Placement principle recorded as a serving/prompt-assembly rule; no primitive without consumer | this note |
| riir-clippy (consumer-first, #2) | Recalibrates Bench 051's negative; measured gain unproven until the Issue 067 A/B | `.issues/067` |
| riir-ai | Two-brain ordering validated (state-before-evidence is the provably cheap direction) | this note only |
| league | Quality-mode option recorded; 2× prefill = a throughput league loss; no quality bench exists | this note only |

## 6. Re-open triggers (any one)

1. **An in-stack model that emits reasoning traces** (a fine-tuned Bonsai reasoning variant, or an exposed-trace API lane with a long-context task) → the generic two-pass reread primitive becomes implementable; the fact-extracted × AcPrefix-front-placed × reread composition (R313 × this paper × R295) is the novel combination to attempt first.
2. **The TrajStore densifies with same-rule exemplars** (≥K Certified same-rule trajectories for a queried lint) → riir-clippy Issue 067's arm G' A/B fires (same-rule-only channels, condition-first placement, vs arm B on the Bench 049 harness).
3. **A long-context quality bench lands** (GraphWalks-class) in the league or elsewhere → add the Tas/Ta placement cells as quality rows.

## 7. Actions taken this session

- riir-clippy `.issues/067_trace_as_state_growing_input_reopen.md` filed (the one executable action).
- R313 + R512 updated with dated redirects (512's stale conditional — "the POC will upgrade the finding" — corrected against Bench 051's measured negative).

## References

- Zou & Tang, arXiv:2609.02702 (the paper; fetched full text 2026-09-04).
- Xu et al., EMNLP 2024, arXiv:2309.06275 (RE2 — repeat-question reread; the paper's strongest ablation control).
- Weston & Sukhbaatar, arXiv:2311.05732 (S2A — context rewrite-and-replace).
- Zhao et al., arXiv:2607.02509 (ReContext — recursive evidence replay).
- Ok & Lee, ACL 2026 Findings (prompt-order sensitivity under causal attention).
- Workspace: R295 (AcPrefix), R313, R512, Bench 049/051 (riir-clippy), riir-clippy `cf_workers_ai.rs` `prompt_growing`.
