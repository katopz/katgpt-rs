# Plan 632 — grid-reselect λ calibration + GOAT G1 (the 966 continuation after the premise read)

**Status:** OPEN — filed 2026-10-28 from [Issue 966](../.issues/966_modelless_prefix_first_grid_reselection.md) after [riir-ai Bench 1012](../../riir-ai/.benchmarks/1012_issue966_premise_gate_read.md) read the premise ALIVE (pool 0.6174 vs 0.0625 floor on the DFlash2 keyed lane posture, sel-rank mode 2). Nothing implemented yet; the plan exists so the arc starts plan-first with pre-registered bars. NO lease taken — claim per Issue-943 before implementing.
**Lease closed:** zed-m3-632-lambda-g1 (the sweep + records landed same session 2026-10-28; the expiry token above is kept only as the claim's historical record). (claim record: zed-m3-632-lambda-g1, epoch 1793139820, was expires 2026-10-29T22:37:00+00:00 — folded per the lease-reshape rule)

## Why now

The premise gate was 966's blocking unknown ("if pool coverage ≈ top-1, the class is dead — close"). It read ALIVE at 9.88× floor. The next tasks in the issue (λ calibration, then G1) are unblocked but un-designed: the issue's prose still conflates two lanes, and the premise result sharpened the question this plan resolves FIRST.

## T0 — the lane-hosting decision (resolves before any sweep)

The issue ships TWO drafter lanes and they must not be silently blended:

- **Lane K (katgpt-speculative)**: the `grid_reselect` substrate's own lane — `dflash.rs` marginals + `bigram_markov` top-16 table + the prefix-first walk. Modelless, in-crate, testable without the 27B (the G2 instrument's synthetic shapes). Its verify loop / accepted-length harness does NOT exist yet.
- **Lane D (DFlash2, riir-ai plan 614)**: the z-lab drafter lane where the premise was actually measured — real 27B target, keyed posture, mean_j 3.793 (Bench 1012's side finding). Its loop is production-shaped (byte-exact gates) and NOW carries the premise evidence. The corrector is NOT built here.

Decision criteria (record the pick + one-line rationale in this file, then strike the other rows):
- (A) **Measure in Lane K, argue transfer.** Cheapest (no 27B), keeps the substrate self-contained; but the premise evidence lives in Lane D — a Lane-K-only G1 would repeat the Bench-965 "offline signal did not transfer" class.
- (B) **Port the compose to Lane D.** The premise is certified THERE; the loop already reports slot-resolved acceptance (hist) and the premise read's plumbing (KeyedPremiseRead) is the natural host for a re-selection arm. Cost: the bigram table + walk must be built for the DFlash2 candidate shape (candidates[i] + scores[i] already flow out of DraftWalk — the lattice source exists); riir-infer would consume katgpt-core's bigram substrate (already a dep) — no new repo edges.
- (C) **Both, Lane D first** — the honest default IF (B)'s cost is confirmed small (≤ a day): the Lane-D result is the decision-grade one; a Lane-K mirror only if it changes the verdict.

⛔ The pick is recorded HERE before any sweep runs; a sweep run against the un-picked lane is a wasted 4090 window.

## T1 — the λ sweep (pre-registered before data)

- Grid: λ ∈ {0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6} over the SAME corpus/posture as the premise read (doc-prose, seed `0x5eed`, T = 0.6) plus ONE code-cell (`B614_DOC` = a code file) — Bench 965's lesson: prose-only cells overstate.
- λ = 0 is the byte-identity control: the re-selected stream must equal the argmax walk's stream EXACTLY (the substrate's own λ=0 pin re-asserted at the compose site).
- The bigram term scale: log-probability vs log-probability (the substrate-half note) — record the chosen scale beside λ in the results file.
- Report per λ: mean_j, effective tokens/cycle (draft-phase cost + verify cost amortized — the C_K/L_eff shape from plan 614's Phase-0 gate), per-slot hit-rate, and the premise columns (the KeyedPremiseRead analog under re-selection: does the re-pick convert corrections?).
- **Posture:** 4090, `E:\git\p614-target` (kept warm, Bench 1012), Issue-834 exclusivity gate; Plan-337 zero-contention if a sibling measurement is live.

## T2 — GOAT G1 (pre-registered bars)

- **G1**: the re-selection arm's mean_j at its best λ must exceed the argmax baseline's mean_j on BOTH workloads (prose + code), with the paired interleaved protocol (the ab_timing shape) and a pre-registered minimum margin. Pre-register the margin as **+5% relative mean_j on both cells** (picks up Bench 965's noise scale; anything lower is window noise by that record's own decomposition).
- **G2 (no-regression)**: the draft-phase cost of the re-selection pass must not exceed +15% of the walk phase (the extraction is µs-class; the bigram lookups are O(B·K) — pin it, don't assume it).
- **G3**: lossless contract untouched — the verify stream remains the serial keyed decode BY CONSTRUCTION (re-selection changes WHICH tokens are proposed, never the acceptance rule); the G1-keyed byte-identity gate re-runs green with the corrector arm on.
- **G4**: zero-alloc in the compose (the substrate's existing canary pattern).
- Honest-negative clause stands: G1 fail at every λ ⇒ close 966 with the negative record (the premise was necessary, not sufficient — the bigram prior may be too weak over a 151k vocab even where the grid is right); the walk + coverage instrument survive as the reusable parts (already landed).

## T3 — records + hygiene

- [x] One bench file carrying the sweep table, both workloads, the box provenance lines, and the verdict per gate — **[riir-ai Bench 1013](../../riir-ai/.benchmarks/1013_issue966_grid_reselect_lambda_g1.md)** (window 2; window 1's prose pass rode the same log, bit-exact counts).
- [x] Update 966 (T-table + status) + this plan's checkboxes in the same commit as the landing.
- [x] The second-window replication note from Bench 1012's limitations is DISCHARGED by T2's paired protocol — and EXCEEDED: the prose workload ran twice (window 1's full pass + window 2) and every count replicated bit-exactly, including through a mid-run GPU co-tenant (the counts are seed-deterministic; window 2's timing columns are the only contaminated cells, disclosed in the bench).

## Tasks

- [x] T0: record the lane-hosting pick (A/B/C) here — **PICKED (C) Lane D first, 2026-10-28.** The cost estimate that gated the owner call collapsed on the boundary read: riir-infer's BOUNDARY.md (line 116) already allowlists `katgpt-speculative` as a NON-OPTIONAL public-upstream dep AND it is a live dep today (the DDTree/weaver substrate flows through it) — the port is compose-only (~150–250 LOC: the bigram table + the prefix-first walk over `DraftWalk`'s per-slot `candidates[i]`/`scores[i]`), well under the one-day threshold, so the criterion's owner-call trigger does not fire. The verdict lane (claude_code reviewer) was UNAVAILABLE this session ("Authentication required", two rounds) — noted here; the pick rides the delegated-owner precedent (opt-in arms land, PROMOTION is the owner-gated part — the Issue-1113/1124 arm precedents) and stays revisitable. The compose lands as an OPT-IN default-off arm (config knob, byte-identical when off, the KeyedPremiseRead telemetry pattern); a Lane-K mirror happens ONLY if the Lane-D verdict differs from the premise read's direction. **POST-RUN note:** the landed port differs from this estimate in one design point — the compose is NOT a post-pass over stored grids (`chain_step_scores` conditions on `pred_tok`, so a stale-grid compose is wrong by construction); the arm is a sibling walk (`lattice_walk_reselect`) that re-scores along its own prefix, cost-identical to the walk family. The Lane-D verdict below is the decision-grade read; the Lane-K mirror is NOT run — it cannot convert a verdict that failed in the lane where the premise was certified.
- [x] T1: the λ sweep per the pre-registered grid (both workloads, λ=0 control) — **RAN 2026-10-28 (4090, `2bc7c691b`, Bench 1013).** Prose: λ=0 3.514, every λ>0 ≤ baseline (best λ=0.05 +0.00%), keyed 3.793 (+7.94%). Code: λ=0 4.300, λ=0.05/0.1 4.526 (+5.26%), keyed 4.526 (EXACT TIE with best re-select). Scale recorded: ln U + λ·ln P (log-prob + log-prob; effective pair weight λ·T at T=0.6).
- [x] T2: GOAT G1–G4 per the pre-registered bars (paired interleaved; the honest-negative clause honored on fail) — **G1 FAIL (1/2 cells: prose +0.00% vs the +5% both-cells bar; code +5.26% at-bar). G2 PASS (0.985/0.978). G3 PASS (stream byte-identity across all 16 arm-runs — the lossless contract held end to end). G4: one reused pool buffer per walk call, zero per-slot alloc beyond the walk family's existing profile (disclosed shape).** Coverage ≠ conversion measured directly: premise pool_rate RISES with λ (0.696→0.799 prose) while mean_j FALLS (3.514→2.672) — the corpus-bigram re-picks stay in-grid but leave the target's keyed sample. **THE HONEST-NEGATIVE CLAUSE FIRES: close katgpt-rs 966 with the negative record.** The Lane-K mirror is not run (would not change the verdict); the keyed incumbent ≥ best re-select λ everywhere makes table-improvement variants (top_m, per-row backoff) moot on the registered posture — keyed needs no table and already owns the win.
- [x] T3: bench record + 966/plan updates + commit+push — this commit train (Bench 1013 + 966 close + plan ticks).
