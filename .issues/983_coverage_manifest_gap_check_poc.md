# Issue 983: PoC — Coverage-Manifested Memory + Suspicion-Triggered Gap-Check (modelless protocol)

**Status:** Open — fusion idea, novelty TBD pending the PoC precision gate (F3). F3 ordering-law scope AMENDED pre-data 2026-11-05 (see **F3 amendment (verdict correction, pre-data)** below) — the amendment is recorded before the harness ran; no precision/ordering numbers existed when it landed.
**Claimed:** zed-m3-i983poc, 1793953746, expires 2026-11-07T20:00:00+00:00
**Date:** 2026-11-05
**Source:** `../.research/654_Engram_Notes_RL_Strategy_Not_Content.md` — Engram blog "Learning in a world of notes" (2026-11-04); the trained model's learned gap-check behavior, extracted modelless. Training-track sibling: `../riir-train/.plans/486_notes_usage_rl_cost_sweep_drift_curriculum.md`.

## Mechanism

The blog's trained agent runs `comm -23 dir_ids.txt meta_ids.txt` — a shell **set-difference** between the workspace's folder inventory and its notes' covered IDs — but only when suspicious (a lookup returned fewer entries than expected). Figure 8 of the source: the trained model fires the check on 47% of queries at 50% notes coverage, scaling to 0% at 100% coverage; the base model fires on 0–2 of 125 queries at every stage. The weights bought only the *disposition to fire conditionally*; the check itself is deterministic. Modelless form:

1. **Coverage manifest** — the memory builder stamps, at build time, per-section covered-key sets (or counts + digests where key sets are large) plus a source-census digest.
2. **Suspicion predicates** (deterministic, over pinned expectations): (a) *count-vs-census mismatch* — a lookup returned < the manifest's expected count for its query class; (b) *truncation* — returned tokens == cap ⇒ the read may not be evidence for a final answer; re-issue narrow; (c) *section-coverage deficit* — the query touches a section whose coverage ratio < 1.
3. **Action** — deterministic set-difference (source inventory − covered keys) → targeted lookup of the gap keys / escalate to the primary source / emit a coverage-gap event (feeds `--misses`-style intake queues).

**Enforced-by-design ordering law** (the source's emergent behavior, made a contract): firing rate ∝ monotone(1 − coverage), ≈ 0 at complete coverage. On game surfaces this ships as a sigmoid drive over the deficit consumed by the existing FSM — never a hard edge bolted beside it (the oscillation trap).

## Consumer surfaces

- **riir-refine heal lane:** a corpus-gap event when the drafter's span family matches zero/poor corpus coverage → post-mining intake (the `--misses` queue).
- **riir-reflex:** per-suite corpus coverage readout beside abstention — complementary, not redundant: distance-based abstain (embedding space) vs set-membership escalate (coverage space).
- **riir-rethink ESC:** coverage-deficit as an ADDITIONAL escalation trigger beside the gate flags.
- **Game two-brain:** think-brain belief census vs info-brain zone inventory — one-way gate preserved (belief reads the census, never writes it).

## Falsification gates (pre-registered)

- **F1 (set-equality):** the gap set equals an independent second-derivation census EXACTLY (two-derivation agreement); a dangling gap key (no executable lookup) refuses loud.
- **F2 (cost):** the diff computes at stamp time / incrementally — never a full per-query re-enumeration; per-query trigger cost O(1) over the manifest.
- **F3 (precision + ordering — verdict-round-1 correction, per-predicate):** precision is measured **PER PREDICATE, never pooled** — truncation (b) can fire for coverage-unrelated reasons, and pooled "no false fires" would fail for reasons that say nothing. Predicate (a) count-vs-census: a fire is a true positive iff the expected-count deficit is real against a fresh census. Predicate (b) truncation: true positive iff the narrowed re-issue changes the answer or recovers cut-off content. Predicate (c) coverage-deficit: a fire is a true positive iff it is **answer-relevant** — the gap set contains the gold answer's key on eval rows carrying gold; merely touching an under-covered section does not count. **Ordering-law assertion (firing rate monotone in 1 − coverage, ≈ 0 at complete coverage) is scoped to predicates (a) and (b)** — for (c) monotonicity holds by construction (it fires ∝ deficit by definition) and is RECORDED as such, never asserted as a finding.
- **F4 (drift determinism):** staged-replay — grow the fact lane 3× with the manifest versioned per stage; protocol outputs byte-identical given identical inputs; a stale census refuses loud (never silently validates).

## Honest bounds (requires-weights, Research 654 panel W1–W6)

Semantic suspicion beyond the enumerable anomaly classes (W1), open-ended query composition (W2), selective link-following (W3), the blog's +7–20pp accuracy delta (W4), calibrated doubt on plausible-but-wrong notes (W5), and the Hex capability ceiling (W6) are weight-carried by the trained policy. This PoC claims the **cost axis and the coverage-accounting axis only** — never accuracy parity with a trained reader.

## Prior art

Self-RAG (arXiv:2310.11511 — *trained* when-to-retrieve/critique), Evidence Tracing & Execution Provenance (arXiv:2606.04990 — *scaffolded* memory verification), OTC (arXiv:2504.14870 — cost-penalized RL, the training sibling). A learned coverage-set-difference on a notes corpus was not found published (Research 654 §1/§4 searches, snippet-level caveat there); the deterministic protocol form is the stack-novel claim — pending this PoC. Signal-diff vs shipped cousins (coverage-report, corpus-delta, staleness probe, CorpusDistanceGate, esc_voi, two-brain): Research 654 §5.

## Tasks

- [ ] T1 Choose the first consumer (refine heal lane vs reflex readout) and pin its manifest row format.
- [ ] T2 Implement the manifest stamp + trigger predicates behind an opt-in feature flag (default-off; report-only first — annotate, never block).
- [ ] T3 F1–F4 as tests; the four behavior metrics (checks/query, lookups/query, tokens/lookup, link-follow share) as a report-only trace linter.
- [ ] T4 Staged-replay harness on the corpus-lease epoch machinery; GOAT gate vs the static coverage-report cousin on precision-per-fire.

**F3 amendment (verdict correction, pre-data) — 2026-11-05, delegated-owner Claude verdict round 1 (session 5c33f8b4-a47f-427a-ae5a-f9e7dcabf261, REVISE → Option 2 + three additions; landed BEFORE any harness/precision/ordering number existed — a correction from analysis, never from data):**

- **Original wording, quoted:** "Ordering-law assertion (firing rate monotone in 1 − coverage, ≈ 0 at complete coverage) is scoped to predicates (a) and (b) — for (c) monotonicity holds by construction (it fires ∝ deficit by definition) and is RECORDED as such, never asserted as a finding."
- **The c=0 proof (why the original scope is unsatisfiable for (b)):** under any model where returned content scales with what the memory holds, at coverage c=0 the lookup returns 0 tokens — below every positive cap — so predicate (b) (truncation: returned tokens == cap) NEVER triggers at the largest deficit. A law reading "monotone in 1 − c, ≈ 0 at c = 1" cannot hold for (b): its rate is 0 at BOTH endpoints (no trigger at c=0 for lack of content, no fire at c=1 for lack of gap). No measurement design satisfies the original wording for (b); the pre-registration contained a mistake.
- **Revised scopes (per predicate, never pooled, never the union (a ∨ b)):**
  - **(a) count-vs-census:** asserts the ordering law — firing rate monotone non-decreasing in 1 − c, ≈ 0 (exactly 0 on well-formed lookups) at c = 1.
  - **(b) truncation:** asserts its OWN ordering law on its own axis — trigger rate monotone non-decreasing in `covered_tokens / cap`, ≈ 0 (exactly 0) when the covered content fits under the cap; precision still measured per F3's TP criterion (the narrow re-issue recovers cut-off content).
  - **(c) coverage-deficit:** unchanged — monotonicity holds by construction and is RECORDED, never asserted as a finding.
  - **ALL THREE predicates:** the c=1 endpoint is asserted across the board — at complete coverage the gap set is empty and the difference was computed once at stamp time (F2), so gap events = 0 at c = 1 for every predicate. Only the "monotone in 1 − c" half is narrowed to (a).
- **Blind spot this exposes (predicate docs + TraceLinter coverage notes carry it):** (b) CANNOT detect gaps in sparse memory — a truncation suspicion requires covered content ≥ cap, so low-coverage detection belongs to (a) alone. A deployment relying on (b) for gap discovery reads silence where the memory is thinnest.
- **Rejected alternatives (recorded):** asserting the law on the union (a ∨ b) would pass empirically only because (a) dominates at low c, hiding (b)'s non-monotone behavior; asserting on TP-qualified (confirmed) firing would hold by definition — the Lean-restatement class (a check that holds whatever the inputs).
