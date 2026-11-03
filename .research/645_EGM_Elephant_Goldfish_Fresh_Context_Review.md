# Research 645: EGM — Elephant-Goldfish Fresh-Context Review Protocol

**Status:** Active (Gain — adoption set filed as Issue 977)
> **Source:** google/elephant-goldfish-model (EGM, egm.dev) @ `5ba5afa03f62627639fa4f71b7ac3bd4d8d07e24`, Apache-2.0; ~10.7k stars; "Thousands of Google engineers use EGM". Not a paper — a productized agent-skill corpus (7 skills, 9 reviewer personas, 5 document standards, per-skill `test_scenarios.yaml` prompt tests).
> **Date:** 2026-11-03
> **Classification:** Public
> **Related Research:** — (zero prior distills of this repo; internal grep clean)
> **Related Plans:** — (adoption issues, not plans)

---

## TL;DR

EGM productizes the **fresh-context adversarial review** pattern the workspace already runs ad-hoc (research §5 verdict ping-pong, `spawn_agent` isolation): an Elephant (deep-context drafter) writes a spec, and Goldfish subagents review it with **zero conversation history** — only the artifact + a curated `GoldfishContext` reading list. The distill value is NOT the class (fresh-context review is known: LLM-as-judge, debate, Constitutional-AI critique) but four **mechanisms the workspace does not ship**: paraphrase-first comprehension gating, plan-side implementation-readiness auditing, semantic doc↔code drift reporting, and failed-reviewer "not run" disclosure. Filed as adoption set → katgpt-rs Issue 977.

**Distilled for the workspace (workflow layer, no runtime code):**
A reviewer that cannot paraphrase what it just read has not understood it — and a list of defined terms cannot catch a document whose terms are all defined but whose *parts do not connect*. The paraphrase does (EGM comprehension agent, verbatim).

---

## 1. What EGM Is

Roles: **Elephant** (project context, drafts specs) · **Goldfish** (fresh-subagent reviewers, no conversation history) · **Human** (trade-offs, approval). Pipeline: `/egm-brainstorm` → `concepts_brief.md` → `/egm-draft` (section-by-section approval pauses) → `/egm-review` (5 stages) → `/egm-resolve` (PROPOSE FIX vs CONTEST) → approved spec under `docs/designs/`. Plus `/egm-bootstrap` (document an existing codebase), `/egm-review-loop` (configured reviewer group), `/egm-drift-report` (docs vs code).

### 1.1 The reviewer swarm (9 personas, all stateless + read-only)

| Persona | Lens | Distinctive mechanism |
|---|---|---|
| `comprehension` | naive fresh reader | **Paraphrase First** — cannot paraphrase a section ⇒ 🔴 Blocker. Self-containment over term lists (quote in TL;DR). |
| `design_critic` | hostile architect | stress-test assumptions, unmitigated risks, cost/quota economics (brainstorm mode) |
| `structural` | layout validator | heading hierarchy + template conformance vs a Target Standard |
| `style` | copy editor | hyperbole/jargon/filler vectors + **claim calibration**: each hedged claim classified `DROP \| CALIBRATE \| RELOCATE \| KEEP` (KEEP = backed; never reported as a finding) |
| `zoom_out` | YAGNI skeptic | code-search-verified reuse ("MUST perform code searches" before flagging reinvention) — this is substrate-first as a reviewer persona |
| `implementation_readiness` | the engineer who writes the code | *"Do I know exactly what to code, where, and how, based on this text?"* — any decision left to the implementer is a finding |
| `codebase_details` | compliance auditor | doc↔code exactness: hardcoded consts, env vars, edge cases in code missing from the doc |
| `synthesizer` | swarm compiler | dedup, severity triage 🔴/🟡/🟢, line-ref normalization, **A+B dual instances per reviewer for recall**, failed reviewer = `not run`, never zero-findings |
| `bootstrap_orchestrator` | docgen | codebase → docs |

### 1.2 The 5-stage pipeline (Stage 5→9, strict order)

5 **Initial Zoom-Out** (full file, YAGNI/reuse) → 6 **Comprehension Test** (paraphrase-first, FeatureRange+DetailedRange, no standard — "the test is self-containment, not detail") → 7 **Critic Review** (design+structural+style on FeatureRange vs `feature-spec-standard`) → 8 **Intermediate Zoom-Out** (re-run AFTER critique, given earlier reports, focused on additions/mitigations) → 9 **Implementation Readiness** (implementation-readiness+codebase-details+style on DetailedRange vs `implementation-readiness-standard`).

Supporting mechanics: **semantic split discovery** (find `## Detailed Implementation Plan`, review the two halves with different reviewer sets at different altitudes) · preset gates (interactive / consolidated / feature-spec-only / implementation-only) · checkpoint gates with a **1-audit convergence cap** (second manual-edit round ⇒ residual findings documented as accepted trade-offs, auto-advance) · **linked-file context assembly** (resolve relative links from the doc, realpath-scoped inside the working dir, URL approval gate, headless = leave out + disclose) · execution charter (banned validating adjectives, banned apologies under correction, critique-first, mandatory counter-arguments, "Earn Your Agreement", strictly-one-question interactive invariant) · artifact-first delivery (never dump large artifacts in chat).

### 1.3 Drift report + standards

`egm-drift-report`: per design doc — contextualize → locate source (grep if paths not explicit) → **read current code state, not memory** → report ONLY differences (missing features, incomplete models, logic shortcuts, architectural divergence); perfect match prints "No drift detected". Standards include `heilmeier-standard.md` — the DARPA catechism as a self-containment bar: *"a competent engineer (or fresh AI session) who is unfamiliar with the project can understand the goals, risks, and value"*.

---

## 2. Cousin Map + Signal-Diffs (§3.6 discipline)

Internal-first grep: `elephant|goldfish` across all `.research/`/`.plans/`/`.issues/` — **zero hits**. Mechanism-level cousins:

| Workspace cousin | Shared | Signal-diff (what EGM adds that the cousin does not consume) |
|---|---|---|
| **research SKILL.md §5 verdict ping-pong** (katgpt-rs, SKILL.md:409) | context isolation ("a fresh reviewer that cannot see your context"), round cap 3, evidence-based REVISE | Verdict reviewer is ONE lens, asked to verdict, never to **paraphrase-first**; no severity triage; no A/B recall instances; fallback is REPLACE (equal generic reviewer), EGM's is DISCLOSE (`not run` + retry-once). Same green-zero concern, different mechanism. |
| **doc-sync skill** (katgpt-rs) | doc hygiene, drift rows | doc-sync anchors on **git history** ("diffing git history against the last documented entry"); drift-report anchors on **current code state** (reads the code, verifies named symbols/claims, reports only differences). feature-gate-audit covers the feature-gate claim subset only — no general doc↔code semantic lane. |
| **code-reviewer skill** (seal-game-editor) | review verdict + verification gates | Reviews a **landed branch** (fmt/clippy/build/tests + plan-to-code fidelity). No **plan-side** actionability audit: no surface asks "can an agent execute each task without making independent design choices?" |
| **grammar-not-answers skill** (riir-reflex) | contamination control (echo-gate, leak-scan) | Different domain — eval contamination, not document review. Named because both defend against context leakage into a judgment. |
| **substrate-first skill** (katgpt-rs) | reuse-before-build via vocabulary translation | EGM `zoom_out` is the same law as a reviewer persona ("MUST perform code searches" before flagging reinvention) — validates the law, adds nothing. |
| **proposal skill** (katgpt-rs) | proposal format, honest caveats | No grep hits for catechism/self-containment — the "unfamiliar engineer can understand" bar is not pinned anywhere. |

Greps run 2026-11-03: `paraphrase` and `readiness` — **zero hits** across the workspace `.agents/` trees and the global `~/.agents` (10 workspace trees, derived at grep time; independently re-confirmed by the verdict reviewer, case-insensitive) — the two strongest deltas. `fresh context` — one hit (research §5). `drift` — all mechanical/contract drift (boundary ledger, highwater, gate labels), zero semantic doc↔code lanes.

## 3. Two-Sided Consumer Check (priority-ladder discipline)

- **Healer (riir-refine):** no direct consumer — EGM is prompt corpus, not Rust rule material; would not route to kernel_opt/clippy_lints/sec/rust_perf corpora. Recorded, not mined.
- **Decision-serving self-evolve loop:** the goldfish shape IS the shipped ESC/`seat_rerank` posture — a fresh-context hosted expert consulted over a candidate set without the drafter's context. EGM at Google scale is external validation of that design; its A/B dual-instance recall lever is a plausible ESC recall experiment (rethink-side, not filed here — no consumer asks for it today).
- **Workflow layer (the actual consumer):** the research §5 gate, the plan corpus, doc-sync, and the skill templates — see Issue 977.

## 4. Prior-Art Context (class-level, so nobody overclaims)

Fresh-context / adversarial review is a known class: LLM-as-a-judge (Zheng et al. 2023, MT-Bench), multi-agent debate (Du et al. 2023), Constitutional-AI critique-revision (Bai et al. 2022), CriticGPT (Saunders et al. 2024), and every harness's generic subagent. EGM's contribution is **productization**: a fixed 5-stage pipeline with per-stage reviewer sets and standards, paraphrase-first gating, semantic-split reviewer routing, prompt tests (`test_scenarios.yaml` per skill), and an honest limits section ("a review does not replace tests or human judgment"). The pinned claim is **adoption deltas vs our own surfaces**, not a new review class.

## 5. Verdict

**GAIN** — external repo ships; four mechanisms are actionable and unshipped (grep-proven). Not GOAT (no measurable gain over a shipped approach on a benchmark axis), not Super-GOAT (no new capability class; the class ships). Routing: note + issue in **katgpt-rs** (public-safe; the consumers are katgpt-rs-hosted skills and the workspace plan corpus). No private-tier content involved.

### Adoption set (filed as katgpt-rs Issue 977, each independently revertible)

1. **Paraphrase-first comprehension arm** — research §5 Summary contract: the reviewer paraphrases the pinned claim and the verdict-bearing claims before verdicting. A paraphrase that contradicts the note is a finding in EITHER direction — the drafter must then fix the note or quote the passage showing the paraphrase misread it; a contradiction is never a silent accept. Catches the self-AGREE-drift class (§5's own named failure mode) that severity bullets do not. Rides the §5 Summary + reply contract, so BOTH verdict paths inherit it — the `request_verdict` main path and the generic-subagent fallback share one contract. Proof path: the paraphrase is IN the verdict reply — auditable per round; grep shows no shipped equivalent.
2. **Failed-reviewer "not run" disclosure** — a reviewer round that returns no verdict is retried once, then disclosed as not-run; never folded into an AGREE. Applies to both verdict paths (same shared reply contract as item 1). Extends the existing Reviewer-unavailable clause (replace) with EGM's disclose semantics for the partial-failure case. Same green-zero family as cfg_gated_floor_gate; wording-level change.
3. **Implementation-readiness audit for plans** — convention carried in `/.agents/skills/research/SKILL.md` (the workflow skill any plan-executing session loads): before executing any `.plans/NNN` task set, audit *"does each task name what to change, where, and how — or is a design choice left to the implementer?"* Left choices are findings, resolved into the plan before implementation. Proof path: run on the two most recent plans; every finding must name the exact task + the decision it leaves unnamed (human-confirmed real), and the audit's self-test seeds one deliberately vague task that MUST be flagged — a zero-finding run fails the audit, never reads as clean plans.
4. **Semantic drift lane in doc-sync (optional stage)** — for each doc the quarter touched, verify named symbols/claims against current code; report ONLY differences. feature-gate-audit already owns the feature-gate subset; this covers the rest. Proof path: deliberately drifted doc pair as the fixture.

### Recorded, not filed

- Heilmeier-catechism self-containment bar for proposals ("a fresh session unfamiliar with the project can understand goals, risks, value") — a proposal-skill change, standing alone; not folded into the plan-readiness item.
- A/B dual-instance recall for verdict rounds (cost doubles; no consumer demand) — noted for ESC recall experiments instead.
- Prompt-test lane for skills (`test_scenarios.yaml` analog) — real gap, bigger than this distill; one line here so it is findable.

## 6. Provenance

Clone `katgpt-rs/.raw/elephant-goldfish-model` @ `5ba5afa03f62627639fa4f71b7ac3bd4d8d07e24` (shallow, Apache-2.0), read-only, deleted after this note lands. All quotes above re-verifiable at that sha: `agents/*.md` (personas), `skills/egm-review/SKILL.md` (pipeline + charter), `skills/egm-drift-report/SKILL.md`, `skills/egm-resolve/SKILL.md`, `resources/document_standards/heilmeier-standard.md`, `agents/comprehension.md` (Paraphrase First).
