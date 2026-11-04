# Research 650: Probe-Grammar × Supergoat-Candidates — Consumer Fusion Sweep (refine + rethink)

> **Source:** user-directed sweep (2026-11-04): fuse Research 649 (probe-authored circuit
> grammar) with each `riir-ai/.docs/04_supergoat_candidates` doc — one by one, via
> delegated subagents — aimed at the self-evolve product stack (fusion-priority-1):
> **riir-refine** (the rulewright) and **riir-rethink** (ESC/encoder serving) as consumers.
> **Date:** 2026-11-04
> **Status:** Active — sweep recorded; 2 ISSUE + 2 NOTE + 6 PASS; reviewer verdict
> negotiated this session: round 1 REVISE (6 reasons) addressed — round 2 recorded in §9
> **Related Research:** 649 (the anchor primitive), 648 (parent)
> **Files created this session:** this note; `riir-refine/.issues/205_campaign_failure_decomposition.md`
> (LEAP × 649); `riir-refine/.issues/206_probe_authored_corpus_repair.md` (ILC × 649, ILC leg dropped)
> **Classification:** Public

---

## TL;DR

Ten candidate × 649 fusions were analyzed by ten delegated subagents (each brief: read the
candidate doc + Research 649 + both consumer surfaces; grep BOTH vocab sets; stationary
test; prior-art search; one verdict). Load-bearing code facts were spot-checked by
coordinator grep before filing (`RefusalTally`, `EscalationDispatch::arm`, the three
`record_outcome` sites, `demotion_suggestions`, `ESC_GATE_RHOS`, `BorelPartition` — all
verified at the cited locations). Outcome: **2 ISSUE** — LEAP × 649 → campaign failure
decomposition in the refine fixer lane (`riir-refine/.issues/205`), and ILC × 649 →
probe-authored corpus repair (`riir-refine/.issues/206`; the ILC leg itself DROPPED as
redundant — the verified finding behind that filing is that **no deterministic
telemetry-driven corpus-mutation writer exists** in `riir-refine/src` or
`riir-rethink/src` — scoped after review: `kat_corpus_pr.rs` IS a deterministic corpus
writer, but it consumes owner-attested mined pairs, not telemetry). **2 NOTE** — NS-CSG
× 649 (per-region probe-authored records: a real primitive shape, but both aimed consumers
are blocked — refine is disclaimed by 649 §5 itself plus Bench 126's routed-fanout
negative, and rethink's referee is blind until the decstat outcome join); LEO × 649 (the
ESC re-arm three-stage shape already landed MANUALLY as rethink plan 016 §8-A2 / Bench 078,
and the prereg law forbids the calibration loop the fusion implies — salvage rungs
recorded, not filed). **6 PASS** — DynamicPair, LatCal, OFT, MoA/AM, Conformal UQ, and
Phase Separation, each killed by a specific named check with the closest cousin cited
(§5). Cross-cutting: the **stationary test killed 4 of 10**; 649 §8's consumer veto-gates
held everywhere tested; rethink's ESC lane carries **exactly ONE armed escalate row**
(`emotion_syn`, N=1), making every multi-suite fusion vacuous today.

---

## 1. Method

- Ten `spawn_agent` analysts in two parallel batches; each returned one report (thesis,
  consumer mapping, cousin grep with file:line, stationary test, prior art, verdict).
  No subagent wrote files; all filings below were made by the coordinator after
  spot-checking the facts each verdict leans on.
- Consumers read per brief: `riir-refine/.docs/README.md` + the retrieval / self_evolve /
  score-bench / escalation-chain / fix_verify trees; `riir-rethink/README.md` +
  `escalating_backend.rs` + `esc_voi.rs` + `esc_audit.rs` + `arsenal.toml`.
- Both vocab sets grepped per fusion (649 words: `gate scan|wire slope|CircuitBinding|
  EditForm|strength grid|dose sweep|probe_circuit`; plus each candidate's own words)
  across `katgpt-rs/.research`, `riir-refine/src`, `riir-rethink/src`, and the relevant
  riir-ai crates. **649 vocabulary has ZERO consumer-tree hits anywhere** — the lane is
  consumerless, exactly as 649 §5 recorded; every fusion below is greenfield, not a
  collision.

## 2. The matrix

| # | candidate | verdict | consumer aim | the deciding check |
|---|---|---|---|---|
| 1 | LEAP Strategy (Plan 215) | **ISSUE** → `riir-refine/.issues/205` | refine fixer lane | strong: the shipped `RefusalTally` taxonomy + the fix_verify loop ARE the grammar's gate keys and the dose-sweep referee; campaign EMA admissible in-campaign (non-stationary world: every edit changes state) |
| 2 | NS-CSG (Plan 243) | **NOTE** | rethink ESC (blocked) | per-region records are a real shape (the polytope region lifts 649 caveat 3's band-on-one-scalar limit), but the only referee (`esc_voi` value legs) is receipts-only blind; refine side disclaimed by 649 §5 itself + `kernel_route_narrow` measured negative (Bench 126: p50 45.8 ms > flat 36.3 ms) |
| 3 | Dynamic Pair LoRA (Plan 260) | PASS | — | stationary test fails both consumers (refine retrieval has no target/state axis to condition; rethink has N=1 armed arm) AND the pair scorer's only action point sits behind Bench 018's representation ceiling — `RerankMode::Structural` (visitor-firing evidence) already occupies the rerank slot with a strictly stronger signal |
| 4 | LatCal fixed-point (Plans 258/265) | PASS | — | no commit customer: every value crossing a commit edge in either consumer is already a string digest, an integer counter, or an ad-hoc fixed-point (`taint_composite_256`, ×256 u32 — the pattern in miniature, shipped); 649's binding record already rides its own BLAKE3 envelope |
| 5 | OFT / Orbit (Research 362) | PASS | — | double-conditional: no documented weight-repairable degeneracy in either consumer (refine's ρ/W classes are routing/parse pathologies; the L4 adapter measured 0/60 degenerate; rethink's Bench 062/079/0057 degeneracies are threshold/mask-shaped at the gate layer), AND OFT has zero kernels anywhere; rotation-as-EditForm forced (1 probe scalar vs d(d−1)/2 rotation params) |
| 6 | MoA / Attention Matching (Plan 297) | PASS | — | the seed premise is factually broken: AM's head-importance "by-product" is `measure_sensitivity_stub` (synthetic curves; real per-head sensitivity deferred to Plan 297 Phase D/E, unstarted); rethink's documented degeneracies are gate-layer not head-shaped; head-importance→head-edit is published (CAP ICLR 2026, Voita, AT2) |
| 7 | LEO All-Goals (Plan 155) | **NOTE** | rethink ESC re-arm | the three-stage shape already landed by hand (plan 016 §8-A2: the 6-rung cal sweep = the dose sweep; the committed-before-read rho* pin = the grammar record; the one frozen read + G-RATE/G-WORTH/G1/G2 = behavior disposes); the all-goals no-regression referee holds BY CONSTRUCTION (per-suite `ESC_GATE_RHOS` + the §7 lockstep arming pins); plan 016 §6's prereg law ("no second read, no threshold iteration against test") forbids the autonomous loop |
| 8 | Latent Prediction / ILC (Research 025) | **ISSUE** → `riir-refine/.issues/206` | refine corpus | ILC leg redundant (k-means over embeddings refine already has); the durable finding is the verified-empty seam — no deterministic telemetry→corpus writer exists; vocab restricted to the two probe-derivable arms (templated `context_notes` append; blueprint-named demote execution) |
| 9 | Conformal Predictive UQ (Plan 340) | PASS | — | bound-class redundancy: every finite-sample acceptance in both consumers already carries the CORRECT bound for its statistic (Wilson in `esc_audit`, Beta-LCB in `frontier_report`/instinct T2, paired LCB95 in 649's own P3); the one unbounded acceptance (649 `emit_word`) is a one-shot sweep over a static pool — precisely the shape CRC's own impossibility line disclaims (no residual stream, non-monotone exchangeability) |
| 10 | Phase Separation LRC (Plan 571) | PASS | — | stationary kill: LRC phases need periodic per-entity state (cycle speed × tick); refine mining is agent-session-driven with zero in-binary cadence, refine retry is deadline-capped event-driven, rethink's rate guard is a rolling event window with N=1 armed suites — nothing cycles |

## 3. The two ISSUE filings

### 3.1 `riir-refine/.issues/205` — campaign failure decomposition (LEAP × 649)

The strong consumer fusion of the sweep. A failed `cargo refine --fix` campaign
decomposes AND-OR into sub-failures keyed by the shipped `RefusalTally` classes
(`src/fix_verify.rs`; run-end roll-up at `src/bin/heal.rs`, Plan 329 P2); each sub-failure
gets a 649-style grammar record (`fail ↚ drive[rule | stage] —wire(word, span)→
gap[rejected edit]`, BLAKE3-digested, additive row — traj-store extension or
blueprint-history v2 mode row); recovery is a DISCRETE vocabulary (`SplitEdit /
NarrowSpan / AddContextNotes / EscalateSeatRerank` — the last delegates to the existing
`EscalationDispatch::arm`, no new escalation machinery); emission is the ascending
first-accept sweep with the compile gate as referee — **grammar proposes; verify
disposes**. LEAP's dead-end signal lands as an in-campaign verify pass-rate EMA prune
(the mutated state is the in-memory branch ledger only; campaign end appends to ledgers
that already exist). Pre-registered demotion law in the issue: the **base-rate gate
runs first** — if refusal/loop-exhausted is near-zero on real campaigns, the lane folds
pre-PoC as a recorded negative. The issue also carries the honest weak link: clippy
spans/verify diagnostics are a coarse proxy for 649's activation taps — what transfers
is the emission structure, not the probes. Proof path:
`tests/campaign_decomposition_gates.rs` + flat-retry blind baseline + one bench record.

### 3.2 `riir-refine/.issues/206` — probe-authored corpus repair (ILC × 649; ILC dropped)

The durable artifact of this fusion is a **verified empty seam**: no deterministic
telemetry→corpus-mutation writer exists in `riir-refine/src` or `riir-rethink/src`
(grepped this session; scoped per review — `kat_corpus_pr.rs` IS a deterministic
corpus writer whose rebase keys `record_id + blake3(fix)`, but its input is
owner-attested mined pairs over the PR wire, never runtime verify/score telemetry).
The report-only lanes: `--gate-calibration` strata
verdicts, blueprint `demotion_suggestions`, `--corpus-delta` are report-only;
`--active-set-demote` is explicit CLI (the marker file is the only demotion writer);
post-mining intake is human/agent-authored. The filing scopes the loop to the two arms
649's discipline can actually derive: (arm A) templated `context_notes` append keyed by
failure class, landed via the pins-manifest same-commit law; (arm B) deterministic
execution of the blueprint-named demote command behind an explicit flag. Template split
and span narrowing are EXCLUDED (not probe-derivable — the probe names the broken rule,
it cannot author fix content). Referee: `--score-bench` A/B, generation N vs N+edit,
pre-registered bar. Kill criteria pre-registered: arm A misses the bar AND demote is the
only gain ⇒ downgrade to NOTE (fold demote-execution into blueprint as opt-in), PASS the
rest. The new artifact class is the `corpus_binding` record (drive rule × gate stage ×
failure class, BLAKE3, corpus-epoch-keyed, stale-refusing — the `sst_certified`
verify-against shape), which is the part cleanlab (the nearest published cousin) does
not carry.

## 4. The two NOTE rows (salvage rungs + unblock triggers; deliberately not filed as issues)

### 4.1 NS-CSG × 649 — per-region probe-authored records

The primitive shape that survives published prior art: run the 649 pipeline PER BFCP
REGION (the region is the gate predicate — lifting 649 caveat 3's band-on-one-scalar
limitation), with coverage-guaranteed assignment (`BorelPartition::classify` — every
input lands in exactly one region). Prior art narrows it sharply: Provable Polytope
Repair (PLDI 2021) + REASSURE already own solver-based per-region weight repair; the
unpublished remainder is the probe-authored + discrete-vocab + behavior-gate axes made
region-conditional.

- **Blocked on refine:** 649 §5 itself disclaims the repo ("the rulewright fixes code,
  not weights"), and the nearest region-keyed fan-out cousin measured negative
  (`kernel_route_narrow`, Bench 126). Record Bench 126 as the standing refutation for
  region-keyed fan-out in refine.
- **Blocked on rethink:** `esc_voi`'s value legs are receipts-only blind until the
  decstat outcome join; region-heterogeneity is unmeasured at N=1 armed row.
- **Smallest consumer-free step (recorded, not run):** extend the Bench-978
  injected-degeneracy MLP to TWO disjoint band-gates in two half-space regions;
  `classify` keys the region; the per-region pipeline must recover both k₀ exactly.
  Pre-registered bar: both constants exact, referee emits exact k on 3/3 pinned configs.
- **Unblock triggers:** (a) the decstat outcome join lands (esc_voi unblinds);
  (b) a second suite re-arms (the rethink issue-023 family) — then region-keyed
  rho/margin fitting becomes measurable against the global fit.

### 4.2 LEO × 649 — ESC gate calibration

The honest finding is that the fusion's pipeline **already exists, hand-run**: rethink
plan 016 §8-A2 landed the 6-rung cal sweep (the dose sweep), the rho* pin committed
BEFORE the frozen read (the grammar record — verified at `escalating_backend.rs`
`ESC_GATE_RHOS = &[("emotion_syn", 0.22)]` with the "committed BEFORE the frozen read"
comment), and the one-shot frozen read + G-RATE/G-WORTH/G1/G2 (behavior disposes). The
LEO all-goals referee holds by construction: per-suite gate params + the §7 lockstep
arming family (ESC_POSTURES + ESC_GATE_RHOS + esc_gates pins + the instinct byte pin)
make cross-suite regression structurally impossible. And plan 016 §6's prereg law —
"no second read, no threshold iteration against test — a re-pin is a NEW prereg" —
forbids exactly the autonomous calibration loop the fusion imagined. That is the lane's
own discipline working, not a gap.

- **Salvage rung (a), rethink:** at the NEXT legitimate re-arm (re-source-gated via
  rethink issue 023), replace the hand rho ladder with 649-style probe-derived window
  centering inside the existing `--cal-sweep` instrument — fixture = Bench 077/078
  replay records, referee verbatim, bar pre-registered per plan 016 §6.
- **Salvage rung (b), refine:** report-only cross-family coupling table from traj-store
  replays feeding `active_set` cap ordering (the regime-readout posture; no admission
  change). Neither rung is filed as an issue — both are conditional on events that may
  never fire, and the sweep note is their home.

## 5. The six PASS rows (kill evidence + closest cousin)

1. **DynamicPair × 649 — PASS.** Both consumers fail the stationary test: refine's
   retrieval surface has no target axis to condition (the span is the whole query;
   `RerankMode`/posture fixed per invocation) and rethink has exactly one armed ESC arm
   (nothing to route — the pair axis is N=1). The pair scorer's only action point —
   reranking over the 8-D hash embeddings — sits behind Bench 018's measured
   representation ceiling (`with_capacity` outscored by text similarity on a one-token
   discriminator), and `RerankMode::Structural` already ships the strictly stronger
   signal (visitor-firing evidence, not a gate). The shipped GATv2 anchor lives in
   `riir-router/src/dynamic_polytope_router.rs` consuming dense trained-state inputs
   refine does not have. Cross-encoder pair reranking is the published standard
   (monoBERT/monoT5-class); `katgpt-attn-match` MaxSim is the in-workspace published-shape
   cousin, unused by refine.
2. **LatCal × 649 — PASS.** LatCal converts mutated f32 state crossing a quorum;
   neither consumer has that. Refine commits code-fix corpora and Elo tables locally
   (append-only, single host); rethink commits integer counters and BLAKE3 digests
   (decstat rows are integer maps, Ed25519-signed; receipt fingerprints are digests of
   manifest text). The only f64→commit seam in the workspace (`taint_composite_256`,
   `domains/rustc_errors/driver.rs`) is the LatCal pattern already shipped in 4 lines.
   649's outputs are stationary one-shot records (binding string + integer grid +
   episode-count fraction) — BLAKE3-of-canonical-representation is sufficient. Re-open
   trigger: a lane committing mutated latent-f32 state across Byzantine agreement (none
   exists).
3. **OFT × 649 — PASS.** Double-conditional with the stronger half failing. The
   decision-ladder shape (probe-authored deterministic overlay as modelless floor, OFT
   retrain as trained arm) already exists as 649 §5 Q4 + §8's own consumer-gated
   promotion posture — the fusion restates it. No documented degeneracy in either
   consumer is weight-repairable: refine's ρ is retrieval (remedied by rerank, Bench
   027), W is parse/compile-gate output failure (no weight edit fixes a parse failure),
   and the one measured adapter degeneracy (L4 LoRA 0/60, Bench 037/467) argues the
   trained lane adds nothing; rethink's Bench 062/079/0057 degeneracies were repaired
   gate-side (Bench 078 re-arm) — the correct altitude. Rotation-as-EditForm is forced:
   649's probe yields one scalar feeding a 1-D ascending window; a block-diagonal
   rotation needs d(d−1)/2 generator params with no dose-sweep analog. OFT itself has
   zero kernels (research-only). A8 note: any rethink arm is a mint-time re-mint +
   vessel hot-swap (monotonic `artifact_version`), never a runtime edit — the ladder is
   over artifact VERSIONS. Flip path recorded: a weight-attribution study on ONE named
   failure family re-opens 649 §8 on its own terms.
4. **MoA/AM × 649 — PASS.** The seed premise ("AM's matched-head importance as the
   probe source") is factually broken: `katgpt-attn-match/src/head_budget/measure.rs`
   ships only `measure_sensitivity_stub` (synthetic `(1−r)^1.5` curves); the real
   per-head sensitivity is Plan 297 Phase D/E, unstarted. Rethink's served scorer
   (`instinct_laya_head.rs`) is an MLP over cached marker rows — attention lives in
   riir-infer-laya — and its documented degeneracies (Bench 062 gate-abstain, 079
   class-mask, 0057 production-seat abstain) are threshold/mask-shaped at the gate
   layer, already repaired gate-side; none is head-shaped, so 649 §8's veto holds.
   Head-importance→head-edit localization is published (CAP ICLR 2026 training-free
   causal-attribution pruning; Voita "Story of Heads"; AT2). Revisit triggers recorded:
   real per-head sensitivity lands in the AM head_budget lane AND a head-shaped
   degeneracy is documented at the served checkpoints — at which point this rides 649
   §8's own consumer-gated path, not a new fusion.
5. **Conformal UQ × 649 — PASS.** The signals don't align on any live leg. Every
   finite-sample acceptance in both consumers already carries the correct bound class
   for its statistic: `esc_audit` uses Wilson bounds (golden-pinned) on a Bernoulli
   proportion — the correct finite-sample instrument; `frontier_report` uses Beta-LCB
   (`best_belief_score`); instinct's T2 uses paired non-inferiority + Beta-LCB; 649's
   own P3 uses paired LCB95. The one unbounded acceptance (649's `emit_word`) is a
   one-shot dose sweep over a static episode pool — no residual ring, no
   calibration/deployment split, strengths dependent (monotone dose): the exact shape
   Conformal Risk Control's impossibility line disclaims. The esc_voi value leg has a
   stream but is REPORT-ONLY and its deciding quantity is the realized delta mean (the
   row-2/3 failure shape: direct observation dominates). The house conformal floor is
   referenced as doctrine (`Report the Floor`) and never consumed as a bound in either
   consumer. Re-open trigger: a continuous authoring service (per-word losses as a
   stream) — then CRC (arXiv:2208.02814) becomes well-posed and needs its own GOAT +
   floor pre-registration.
6. **PhaseSep × 649 — PASS.** The joint requires a shared tick; neither consumer has
   periodic per-entity state. Refine mining is the agent idle-loop + CLI verbs (zero
   in-binary scheduler; no cron/tick/interval in src); refine escalation retry is
   deadline-capped event retry on 503 (terminal lanes never retried); rethink's
   `EscRateGuard` is a rolling 200-decision event ring with exactly one armed suite
   (N=1 — the LRC bound is trivially 1/1). Both vocabularies are producer-side only in
   katgpt-core; neither consumer imports either. Closest shipped cousins that already
   do the candidates' jobs: the rate guard ("don't escalate too often") and the
   idle-loop skill's batch ledger ("which domain mines next" as a human/agent calendar).

## 6. Prior-art register (IDs + titles, so future greps hit)

- **ORBIT** — arXiv:2606.22357, "ORBIT: Training-Free Multi-Attribute Behavioral
  Steering" (the Orthogonal Rotation-Based Intervention Technique; activation-level
  rotations). Nearest neighbor to "rotation as an edit vocabulary"; shrinks the
  OFT-fusion remainder. ID + title verified this session.
- **Provable Polytope Repair** — Sotoudeh & Thakur, PLDI 2021, arXiv:2104.04413;
  **REASSURE** (Tao et al. 2023) reduces repair to the buggy linear regions. Owns
  region-keyed weight repair (solver-based); the NS-CSG fusion's unpublished remainder
  is probe-authored + vocab + behavior-gate made region-conditional.
- **Conformal Risk Control** — Angelopoulos et al., arXiv:2208.02814 ("Conformal Risk
  Control"). Finite-sample guarantees on bounded monotone losses for threshold-tuned
  rules; the exchangeability/monotonicity requirements are what disqualify 649's
  one-shot dose sweep.
- **Confident Learning / cleanlab** — Northcutt et al. (dcai.csail.mit.edu class on
  label errors). Deterministic class-level label-error detection + prune/reassign from
  observed probabilities — covers issue 206's demote half; the binding-record artifact
  is the delta.
- **DRAGO** — arXiv:2610.22568, "Repairing LLM-Generated Specifications with
  Deterministic Diagnosis" (DRAGO; formalism-specific deterministic diagnosis +
  repair; telemetry constrains repair). ID + title verified this session.
- **CAP** — ICLR 2026, training-free causal-attribution pruning of attention heads
  (+ Voita et al. "Story of Heads"; AT2 attribution-with-attention). Owns
  head-importance→head-edit localization.
- **LLM-APR line** — arXiv:2506.23749 (taxonomy); Xia & Wei, ISSTA 2024 (ChatGPT APR).
  Diagnosis-driven retry pervasive in automated program repair — adjacent to issue 205,
  not killing it (no verify-stage-gated discrete recovery vocabulary found).

## 7. Cross-cutting findings

1. **The stationary test is the highest-yield fusion killer (4 of 10).** Dynamics
   signals (EMA, drift, calibration loops) need a named mutated state variable; pure/
   exact surfaces take LEVEL readouts. Standing precedent: `esc_voi`'s own
   LEVEL-not-slope retraction. Any future fusion brief should run it before the
   consumer mapping, not after.
2. **649 §8's consumer veto-gates held everywhere tested.** No documented
   weight-repairable degeneracy class exists in refine (ρ/W are routing/parse; L4 0/60)
   or rethink (gate-layer threshold/mask, repaired gate-side by Bench 078). The veto
   architecture works: without it, three fusions would have filed noise.
3. **rethink's ESC lane is N=1** (one armed escalate row, `emotion_syn`). Every
   multi-suite/multi-arm fusion is vacuous until a re-source lands (issue-023 family).
   This single fact killed or weakened four candidate mappings.
4. **The prereg/one-way-writer laws are load-bearing, not gaps.** plan 016 §6 (no
   second read), the active-set marker as the only demotion writer, pins-manifest
   same-commit — several imagined "loops" are structurally prohibited, by design.
   Fusion briefs should read the target lane's discipline laws BEFORE proposing loops.
5. **The one verified-empty seam of the sweep** — no telemetry-driven corpus
   writer in riir-refine (grepped `riir-refine/src` + `riir-rethink/src`; the nearest
   shipped writer, `kat_corpus_pr.rs`, consumes attested mined pairs, not telemetry) —
   is filed as issue 206 with pinned kill criteria.
6. **Refine's rerank slot is occupied by a strictly stronger signal**
   (`RerankMode::Structural`, Bench 018 ceiling). Any future retrieval-fusion must beat
   visitor-firing evidence, not the static embedding score.

## 8. What would kill / promote the consumer lanes

- **Issue 205 dies if:** the base-rate gate shows refusal/loop-exhausted near-zero on
  real campaigns (fold pre-PoC, recorded negative); or the EMA prune false-fires on any
  recoverable seed (the 0-false-prune bar is absolute).
- **Issue 206 dies if:** arm A misses the pre-registered score-bench bar AND demote
  execution is the only gain ⇒ downgrade to NOTE (fold demote-execution into blueprint
  as an opt-in flag); record the negative.
- **NS-CSG NOTE unblocks at:** decstat outcome join (esc_voi unblinds) + a second armed
  suite; smallest step is the consumer-free two-region fixture (§4.1).
- **LEO NOTE activates at:** the next legitimate re-arm (re-source-gated) — then rung
  (a) replaces the hand rho ladder with probe-derived window centering inside the
  existing `--cal-sweep` instrument.
- **Sweep-level re-open trigger:** any documented weight-repairable degeneracy class at
  a named consumer re-opens the whole 649 §8 promotion family at once (OFT/AM/NS-CSG
  rows all wait on the same event).

## 9. Verdict

Self-grade: **record** — none of the ten fusions individually clears GOAT at consumer
level today, which is the honest outcome of running the sweep cheaply before any
implementation. Two actionable fusion ideas are filed with proof paths and
pre-registered kill criteria (`riir-refine/.issues/205`, `riir-refine/.issues/206`);
two conditional NOTEs are recorded here with unblock triggers; six PASSes carry named
kills with checkable evidence paths. Fusion-priority-1 aim preserved: every actionable
outcome lands on the self-evolve stack (both issues are refine-side; both NOTEs name
their rethink-side rungs).

### Verdict record

- **Round 1 — REVISE** (reviewer session `6a18619e`, 2026-11-04): 6 reasons, no row
  verdict changes. Addressed with evidence: (1) 206's seam claim scoped to
  telemetry-driven writers, grepped repos named, `kat_corpus_pr.rs` added as the
  nearest cousin with its signal-diff (attested mined pairs ≠ telemetry — rebase keyed
  `record_id + blake3(fix)` verified at `kat_corpus_pr.rs` L312–332); (2) 206's
  referee bar replaced (the "+1 shaped verdict OR +0.5pp" point-estimate OR dropped)
  with paired LCB95 > 0 on affected strata + n ≥ 30 minimum support; (3) 206's arm B
  re-ruled as a second candidate arm against the unedited generation N baseline (not a
  control); (4) 205's base-rate gate pinned (<5% over ≥20 campaigns) and EMA seed
  counts pinned (≥12 recoverable / ≥6 unrecoverable, ≥3-of-6 detection); (5) 205's
  held-out coverage arm added (real harvested campaigns scored for word-clear fraction,
  de-circularizing the fixture claim); (6) this §9 record + the Status line.
  Two arXiv titles in §6 corrected to their canonical forms after re-search (ORBIT,
  DRAGO — IDs verified unchanged).
- **Round 2 — AGREE** (same session `6a18619e`, 2026-11-04): all six round-1 fixes
  verified on disk by the reviewer at the cited lines; per-row verdicts, routing, and
  numbering unchanged and confirmed. One pre-commit fix directed and applied: the two
  PASS-Redirects titles at the foot of this note were corrected to the canonical §6
  forms (ORBIT, DRAGO) — the block future greps hit now carries the verified titles.
  Closing round confirmed the fix on disk; commits: katgpt-rs `e7a746274` (this note
  + highwater), riir-refine `52126109` (issues 205+206 + highwater, both referencing
  this note's hash in their headers).

## PASS-Redirects (synthesis)

> **PASS-Redirects (synthesis):** ORBIT [arXiv:2606.22357 "ORBIT: Training-Free
> Multi-Attribute Behavioral Steering"] — rotation-as-intervention published at
> activation level; caps the OFT-fusion remainder.
> **PASS-Redirects (synthesis):** Sotoudeh & Thakur [arXiv:2104.04413 "Provable
> Polytope Repair of Piecewise Linear Networks"] + REASSURE (Tao et al. 2023) —
> region-keyed weight repair is published; NS-CSG × 649 survives only as the
> probe-authored region-conditional composition.
> **PASS-Redirects (synthesis):** Angelopoulos et al. [arXiv:2208.02814 "Conformal
> Risk Control"] — the coverage-guaranteed acceptance framework; its exchangeability +
> monotone-threshold requirements are the named instrument-mismatch that PASSes the
> Conformal × 649 row.
> **PASS-Redirects (synthesis):** Northcutt et al. ["Confident Learning" / cleanlab,
> dcai.csail.mit.edu] — deterministic label-error prune/reassign; covers issue 206's
> demote half, motivating the binding-record delta.
> **PASS-Redirects (synthesis):** [arXiv:2610.22568 "Repairing LLM-Generated
> Specifications with Deterministic Diagnosis" (DRAGO)] — diagnosis-constrained
> repair, general method; no corpus-record discipline.
> **PASS-Redirects (synthesis):** CAP [ICLR 2026, training-free causal-attribution
> head pruning] + Voita et al. ["Analyzing Multi-Head Self-Attention" / "Story of
> Heads"] + AT2 — head-importance→head-edit published; PASSes the AM × 649 row.
> **PASS-Redirects (synthesis):** LLM-APR surveys [arXiv:2506.23749; Xia & Wei ISSTA
> 2024] — diagnosis-driven retry pervasive in APR; adjacent to issue 205, no
> verify-stage-gated discrete recovery vocabulary found.

The internal-candidate PASSes (DynamicPair, LatCal, PhaseSep) are candidate docs, not
papers — their cousin citations live in the §5 rows (file:line evidence paths), which
is the anti-invisibility treatment for this sweep's shape.
