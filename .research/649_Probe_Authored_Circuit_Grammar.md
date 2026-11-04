# Research 649: Probe-Authored Circuit Grammar — activation probes as the weight-edit author (fn-grammar + params-vocab)

> **Source:** user-idea fusion (2026-11-04), executed as PoC + proof this session. Parent distillations: [Research 648 — Weight-Space Programming](648_Weight_Space_Programming.md) (Takehi/Teoh/Zhang/Khattab, MIT CSAIL blog 2026-11-02) + the landed Issue-980 `WeightEditOverlay` lane (`b0f1ed615`, Bench 976 GOAT). The idea: *648's author was an LLM reading weights — cheat by PEEKING instead: probe which neurons activate, use that as the learning guide, and structure the result like quest grammar's topologam+vocab but at neuron level (fn-grammar + params-vocab).*
> **Date:** 2026-11-04
> **Status:** Active — PoC PASS on 4 pinned configs (Bench 978, verdict round 2 revised), opt-in primitive landed
> **Related Research:** 648 (the parent), 287 (probe/steering claim-evidence ladder), 301 (probe bank), 094 (2D probing)
> **Related Plans/Issues:** katgpt-rs `.benchmarks/978` (the PoC record); Issue 980 (the carrier lane, closed `b0f1ed615`); katgpt_core::tpr (the binding-algebra cousin)
> **Files created this session:** `crates/katgpt-core/src/probe_circuit_grammar.rs` + feature `probe_circuit_grammar`; `crates/katgpt-core/tests/probe_circuit_grammar_poc.rs` (P0–P3 + P2b); `.benchmarks/978_probe_authored_circuit_grammar_poc.md`
> **Classification:** Public

---

## TL;DR

Research 648's honest fault line was the **proposer layer**: "authoring quality was purchased with someone else's training run" — the LLM read the weights and knew where to edit. This research files the alternative the 648 note left open: **the author does not need to be a reasoning model, because activation probes already answer the two questions the LLM was answering** — *where does the behavior live?* (which neuron columns couple to the failing decision) and *how strong is the coupling, to within a window?* The lane peeks at rollout taps, parses the failing circuit into a **fn-grammar** record (`fail ↚ drive[src[j] | gate] —wire(W[:, j], ŝ)→ gap[row o]` — the quest-grammar role⊗filler structure at neuron level), and emits the repair as a **params-vocab** sentence over the shipped 980 `EditForm` words: the strength GRID is derived from the probe (`⌊|ŝ|⌋−3 ..= ⌈|ŝ|⌉+3`, hard-capped), and the WORD is chosen inside that window by a behavior-verified dose sweep whose acceptance statistic is the **failure rate** — an edit that turns oscillation into drift does not count. **PoC PASS on FOUR pinned configs** (the 980 original + three held-out carrier/gap/band/k₀ combinations; constants pinned before the run, fixture inputs for gap rows 3/5 repaired after the first run — see §4): the probe-blind parse recovered every injected constant on all four (column and gap row exactly, band as the right grid cell, slope in the ±3 window), and the referee emitted the **exact k₀ on all four** (8, 5, 10, 6). The slope UNDERESTIMATES k₀ by 0.40–0.55 in every config — floor fails 4/4, nearest-rounding fails 1/4 (HELD_C: −9.45 → 9 ≠ 10) — and the BASE floor word (k=7) provably oscillates 8/8: the sweep's job is verification, picking the rounding with no prior knowledge of which way the bias runs. The matched blind baseline — same words, same grid, same acceptance, gap row given free — needed **69 authoring evals vs the probe's 5 (13.8×; an order-dependent lower bound — see §4)** to find its repair; random-direction arms never repair (paired LCB95 = +1.0). No LLM, no gradient, anywhere.

**Distilled for katgpt-rs (modelless, inference-time):**
The transferable primitive is a **three-stage authoring pipeline that replaces the reasoning-model proposer of any weight-edit lane**: (1) **gate scan** — one-sided magnitude edge-contrast over (column, band) pairs finds the band-gated drive column; (2) **wire slope** — pooled least-squares coupling of `dst_pre[gap]` on `src[drive]` ranks the anomalous arc and CENTERS the strength window (the slope's job is window centering, never exact strength); (3) **behavior-verified emission** — the grid `derive_strength_grid(ŝ, margin, cap)` is probe-derived, and the word is the smallest strength in the window whose overlay drives the caller's failure-rate to zero. The grammar record is a BLAKE3-digested artifact that rides the 980 envelope's provenance trailer. Authoring is probes + arithmetic; application and verdict stay deterministic (the 648 two-layer split, preserved).

---

## 1. The idea, sharpened

The user's framing, in research vocabulary:

| user words | lane primitive |
|---|---|
| "check probe and see what neuron activate" | activation taps on rollout captures (`StepTap { gate_var, src, dst_pre }`) |
| "use that as a learning guide" | the fn-grammar parse (gate scan → wire slope → gap row) — the guide is a RECORD, not a gradient |
| "in between their method that use LLM to adjust" | the 648 two-layer split: authoring may be heuristic; application + verdict stay deterministic. This lane pushes the author below the LLM floor entirely |
| "cheat by peek" | the LLM's read of weights ≈ a probe's read of activations — both answer "which neurons matter", but the probe needs no frontier model, crosses no egress boundary, and is reproducible byte-for-byte |
| "quest grammar technics aka topologam and vocab at neuron level" | fn-grammar = role grammar over circuit coordinates (drive/wire/gap/gate ≙ subject/verb/object/modifier); params-vocab = the `EditForm` × derived-strength-window word set; an edit = a sentence; the parse = a topological read of the activation graph |
| "revert engineering as fn-grammar and params-vocab to get results" | the PoC: parse → emit → repair, proven on four pinned configs |

The 648 lane already ships the carrier (`WeightEditOverlay`), the arming telemetry (`arm_from_telemetry`), the gate battery (dose-response, norm-matched controls, isolation, retention), and the freeze envelope. **What it lacked was a local author.** That is the gap this fills.

## 2. Prior art (two parallel web sweeps, 2026-11-04)

The full composition is **not published**; every component is. The closest works and the exact deltas:

| work | has | misses |
|---|---|---|
| **AI-Lancet** (ACM CCS 2021) | activation-diff localization + discrete sign-flip edit, gradient-free | no grammar record, no vocabulary, no dose/behavior gate; CNN classifiers |
| **ROME** (NeurIPS 2022) | causal-trace localization + closed-form rank-one write | edit is SOLVED per-edit from data, never emitted from a discrete vocabulary; fact insertion, not behavior repair; no battery |
| **Compiling Activation Steering into Weights** (arXiv:2604.12359, ACL 2026) | class-level activation diff → closed-form weight edit, modelless | no neuron-level localization, no circuit record, no repair gate; demonstrated objective is backdoor injection |
| **INNER** (ISSTA 2024) | probes + clean-vs-failing diffs + neuron localization — the strongest (a) match | repairs by masked gradient descent |
| **ASPD** (arXiv:2609.37731) | weight components with read/write roles → "parameter-level mechanism circuits" — the closest hint of fn-grammar | decomposition is LEARNED; edits are causal experiments, not an emission pipeline |
| **RASP/Tracr/ALTA** (2021–24) | a grammar whose sentences become weights | inverted direction (program→weights on hand-built models); no probe-derived roles; no checkpoint-edit semantics |
| **Knowledge Neurons / circuit discovery (ACDC, attribution graphs)** | probe-flavored localization | read-only, no weight edit; attribution usually gradient-based |
| **Task arithmetic / TIES / DARE** | ±scaled delta algebra | dense deltas, no circuit parse, no role grammar |

Negative searches: `"circuit grammar"`, `"weight grammar"`, `"vocabulary of edits"` → **zero ML hits**; the WSL survey (arXiv:2603.10090, Mar 2026) has **no edit-language axis** in its taxonomy — a citable gap witness. The 648 blog itself is the capability witness, not the language: its edits are free-form tensor operations chosen by a prompt.

**Unpublished composition (the four axes no single work carries):** probe-localized circuit (a) → role-grammar record (b) → discrete-vocab emission over a probe-derived strength window (c) → dose/behavior acceptance battery, all modelless with no LLM (d). AI-Lancet is the nearest end-to-end neighbor and carries only (a) + a weaker (c).

## 3. The primitive (what landed)

`katgpt-core::probe_circuit_grammar` (opt-in feature, implies `weight_edit_overlay`, zero new deps):

1. **Capture** — `ProbePools` splits rollout taps by the policy's OWN outcome (failed vs succeeded episodes — twin-free by construction; the fixture's degenerate policy fails from every start so both pools carry failed episodes, and the statistics pool over all steps, with the split kept as the API's behavior-conditional extension point).
2. **Gate scan** (`discover_gated_drive`) — for each source column × candidate band: one-sided magnitude edge-contrast `mean(|src[j]| | |g| ≤ b) − mean(|src[j]| | b < |g| ≤ b+w)` over pooled rms. A band-gated drive scores large; smooth/constant features score ≈ 0 or negative. First-index tie rules; z-bar refusal.
3. **Gap row** (`discover_gap_row`) — `argmax |W_readout[+axis] − W_readout[−axis]|` from readout rows the caller supplies (weight arithmetic; the axis pairing comes from the arming telemetry, not the injection record).
4. **Wire slope** (`measure_wire_slope`) — pooled LS slope of `dst_pre[gap]` on `src[drive]`; ranked across candidates by `|slope|`. **The slope's job is window centering, never exact strength** (verdict round 2: the estimate is confounded by legitimately-driven channels).
5. **The fn-grammar record** (`CircuitBinding`) — `fail ↚ drive[src[j]|gate |x|≤b] —wire(W[:,j], ŝ)→ gap[row o]` rendered as a stable string + BLAKE3 digest; rides `ProvenanceTrailer.prompt_digest` in the 980 freeze envelope (the record IS this lane's authoring prompt).
6. **The params-vocab emission** — `derive_strength_grid(ŝ, margin, cap)` derives the window FROM the probe (empty return = the loud cap refusal: floor above ceiling → no allowed strength can reach the anomaly; a window whose TOP reaches the cap may be silently truncated there — the grid is a bare Vec); `binding_to_spec` builds the word `DirectionSubtraction { columns: [j], direction: sign(ŝ)·e_o, strength: k }`; `emit_word` sweeps the window ascending and accepts the FIRST overlay whose **failure rate** (fraction of authoring episodes that never reach the goal) clears the bar. **Grammar proposes; behavior disposes.**

Determinism: sequential folds, first-index tie rules, BLAKE3-seeded control arms — repeated captures produce byte-identical bindings (asserted on all four configs).

## 4. The PoC + proof (Bench 978, verdict round 2)

Fixture: the 980 `InjectedDegeneracyMlp` shape, PARAMETRIZED (`carrier_col`, `gap_row`, `k₀`, `band` — all off-diagonal injections; the vertical readout routes through `h2[gap_row]`). **Four configs with constants pinned before the run** (the fixture INPUTS for gap rows 3/5 were repaired after the first held-out run — see the design-stability note below; design constraints from the analytic shape, never tuned per run: carrier ∈ {3,4,5} disjoint from the x/y/goal inputs; gap ∈ {1,3,5} disjoint from the horizontal readout rows; carrier ≠ gap so the diag survives and every injection is off-diagonal with twin value 0):

| config | carrier | gap row | band | k₀ | probe-blind parse (col/row/band) | slope (true) | emitted k |
|---|---|---|---|---|---|---|---|
| BASE (the 980 original) | 4 | 1 | 4 | 8 | 4 / 1 / 4 ✓ | −7.5238 (−8) | **8** ✓ exact |
| HELD_B | 3 | 5 | 3 | 5 | 3 / 5 / 3 ✓ | −4.6030 (−5) | **5** ✓ exact |
| HELD_C | 5 | 3 | 5 | 10 | 5 / 3 / 5 ✓ | −9.4471 (−10) | **10** ✓ exact |
| HELD_D | 3 | 1 | 3 | 6 | 3 / 1 / 3 ✓ | −5.6029 (−6) | **6** ✓ exact |

Exactly ONE column clears the gate bar on every config (gate_z 1.38–1.40 vs bar 1.0); twin telemetry refuses at the SHIPPED arming precondition before any parse, on every config; the parse digest is stable across captures on every config. The emission grid is derived per config (`margin 3, cap 12` — pinned before the run) and CONTAINS the true k₀ on all four (asserted).

**P2 BASE battery:** full-eval obstacle rate 0.0 (16×12 seeded starts) AND failure rate 0.0; success parity with the twin; dose-response `{1,1,1,0}` monotone (oscillation statistic, the 980 G1a arm); neighbor-column isolation null (rate stays ≥ 0.95); out-of-region forwards bit-identical + action prefixes identical; provenance digest round-trips through the freeze envelope. **The slope-bias fact:** ⌈|ŝ|⌉ = k₀ in ALL FOUR configs — the single-regressor slope underestimates by 0.40–0.55 every time. Floor fails 4/4 (the BASE k=7 floor word oscillates 8/8 — the recorded negative); nearest-rounding fails 1/4 (HELD_C: −9.45 → 9 ≠ 10); round-UP would hit 4/4 but only with the bias direction known in advance, which a real lane never has. **The sweep's job is verification: it picks the rounding without prior knowledge of which way the bias runs**, and its dose arms double as the dose-response evidence. **P2b held-outs:** all three held-out configs pass the repair core (authoring failure 0 → full-eval obstacle 0 + failure 0 + success parity) with the SAME design constants.

**P3 peek vs blind search (verdict round 2 framing):**
- **Reachability control (kept):** 16 norm-matched random-direction arms at the RIGHT column + 16 blind column×direction arms, each given the same derived-grid dose sweep with the same failure-rate acceptance: NONE repairs (mean failure rate 1.0); the probe word reaches 0.0; paired LCB95(best random − probe) = **+1.0 > 0** with every per-seed delta +1. A random 64-dim unit direction puts `|d[gap]| ≥ 0.875` (what repair needs) with probability ≈ 0.
- **Matched blind baseline (new — the strongest blind searcher):** ±e_gap words at EVERY column, the SAME derived grid, the SAME failure-rate acceptance, the gap row GIVEN FREE. Measured: **69 authoring evals vs the probe's 5 — 13.8×** for parse-and-verify over enumeration (≈ 6 vs 69 counting the probe's one capture rollout). **The 69 is arithmetic of the search order, not a measurement**: ascending columns, sign − first, 8 strengths → `16·carrier_col + 5` (64 failures at cols 0–3 + 5 at col 4); 13.8× is an ORDER-DEPENDENT LOWER BOUND measured on BASE only — under a uniformly random column order the expected cost is ≈ 510 (≈ 100×), worst case ≈ 1,029. The blind sweep's first-found repair lands at the TRUE column (4, −, 8) — the acceptance statistic killed the plausible compensating edits (e.g. amplifying the legit y-channel at col 1) before they cleared, so the LOCUS difference did not materialize; the honest claim is the eval saving + attribution (the probe names the anomalous column from ONE capture; the sweep had to test 69 words to find it).
- The design-stability honesty: the four configs were run WITHOUT retuning the statistics. The first HELD_B/HELD_C run DID expose two fixture-design flaws (a twin that could not reach the goal because the gap row's diag channel carried a constant bias, and an oscillation whose active↔zero swing never pushed the consecutive-state cosine negative so the shipped halter could not arm) — both were FIXTURE repairs required by the fixture's own analytic contract (twin centering + halter-visible sign-flip), not parse retuning. The parse statistics (gate scan form, band grid, z-bar, slope window, grid margin/cap) are byte-identical across all runs and all four configs.

Validation: `cargo test -p katgpt-core --features probe_circuit_grammar --lib` → 2231 passed (+9 module tests incl. the derived-grid windows/cap arms); the PoC's 5 tests green (P0, P1×4 configs, P2 base battery, P2b held-outs, P3); clippy clean at default / `--features probe_circuit_grammar` postures.

## 5. Novelty gate (the four questions)

1. **No prior art?** No (qualified) — components published (AI-Lancet / ROME / 2604.12359 / INNER / ASPD / RASP-Tracr); the four-axis composition unpublished per the sweeps. Fails Q1 for Super-GOAT, exactly as 648 did.
2. **New behavior class?** Yes — *probe-authored repair-without-retrain-without-LLM*: a local, byte-reproducible author for the 648 lane, whose strength window is itself probe-derived. No stack lane had a local weight-edit author at all.
3. **Product selling point?** Yes — "systematic behavioral failure repaired from activation probes alone: the circuit is parsed into a grammar record, the fix is emitted from a discrete edit vocabulary over a probe-derived strength window, and behavior accepts the word — no LLM in the loop, no gradient, artifact-frozen." The 648 author-capability caveat is REMOVED for this lane (no frontier-model dependency at any scale the taps can cover).
4. **Force multiplier?** Yes — consumes the 980 overlay + telemetry + envelope + battery (≥2 pillars), and the grammar record is a reusable artifact class (TPR-adjacent binding).

**3 solid + Q1 qualified ⇒ GOAT (fusion)** — the same tier as the 648 filing, one layer deeper in the stack: 648 filed the lane with an LLM author; this files the lane's local author.

**MOAT gate:** katgpt-rs is correct (base primitive via fusion, zero game/chain/shard IP; the fixture is generic RL/policy shape). Two-sided consumer check recorded honestly: no named consumer ships today (the rulewright fixes code, not weights; the decision-serving rung is veto-gated per 648 §4) — promotion stays consumer-gated, same posture as 980. The TernaryDraftModel `.bits` repair idea remains veto-gated on a documented drafter-degeneracy class.

## 6. Honest caveats

1. **Single-arc scope.** The parse covers one `drive → wire → gap` arc with one scalar gate variable. Multi-arc/multi-layer circuits (the general "topologam") are future work — `CircuitBinding` is the extension point.
2. **The slope is an estimate; the fixture proves it.** BASE slope −7.52 vs k₀ 8, with the k=7 negative recorded. The slope CENTERS the window; the behavior referee picks the word. Any consumer that skips the dose sweep will repair nothing.
3. **The gate statistic assumes the gate is expressible as a band on one scalar.** Region gates on compound predicates need the multidimensional extension (future).
4. **Fixture-scale proof, four configs.** 8.7K-parameter MLPs. Held-out configs vary carrier/gap/band/k₀ but share the architecture and the degeneracy FAMILY (band-gated rank-1 carrier). The LLM-author line of 648 demonstrated up to 0.8B; the probe author has no scale evidence beyond the fixture family yet. The mechanism is dimension-fair, but the honest statement is: proven on four pinned fixture instances.
5. **Pooled statistics, twin-free.** The fixture's degenerate policy fails from every start, so the outcome split degenerates (all-failed pools) and the statistics pool over all steps. The split exists as the API's extension point for models where some episodes succeed — but the "outcome-split" wording is NOT part of the claim until a fixture exercises it.
6. **Egress: zero by construction.** Nothing leaves the box — the probes replaced the wire entirely (648 caveat 6 resolved for this lane).
7. **The gate scan needs a halter-visible obstacle.** The arming precondition (the shipped halter) requires a sign-flipping oscillation in the hidden series; a degeneracy whose swing is active↔zero (never sign-flipping) arms nothing and the lane never parses. This is the obstacle law working as designed, but it bounds the lane's reach.

## 7. Fusion map (what × what produced this)

- **Research 648 / Issue 980** (the carrier: overlay + arming + battery + envelope) ×
- **the probe bank line** (Research 287's claim-evidence ladder, Research 301's probe bank, the act_tap posture in riir-infer) — probes as measurement instruments, here promoted to AUTHOR ×
- **the quest-grammar lineage** (riir-ai `quest_grammar`: NpcVocabulary + role bindings + `katgpt_core::tpr` compose/surgery + the validator) — role⊗filler grammar thinking, lifted from quest text to neuron coordinates ×
- **RandOpt (Plan 121)** — the norm-matched random control that makes "peek beats search" measurable.

None of the four alone gives a local weight-edit author; the composition does.

## 8. What would kill / promote this lane

- **Kill (demotion law, pre-registered):** the PoC battery failing its identification or repair arms records the negative and the lane folds into `weight_edit_overlay` (no zombie module).
- **Promote (consumer-gated):** a real lane with a documented degeneracy class + tappable forwards — candidates in order of realism: (1) the decision-serving rung's encoder checkpoints IF a loop/degeneracy class is ever documented there (the 648 veto-gate stands); (2) `TernaryDraftModel .bits` drafters IF a degeneracy class is documented; (3) a game-side policy with obstacle telemetry (riir-ai halters already emit it). Each promotion needs its own GOAT gate at consumer scale.
- **Re-open triggers (prior art):** a published probe→grammar→vocab→gate pipeline (the four-axis composition), or an LLM-free author in the fast-moving weight-space-programming thread (4 days old at filing — community replications are running).

## 9. Verdict

**GOAT (fusion), PoC PASS on four pinned configs, opt-in.** The user's idea is filed and proven at fixture scale: the LLM author of weight-space programming is replaceable by probes + a grammar record + a behavior-verified vocabulary emission over a probe-derived strength window — "cheating by peeking" is not a hack but a strictly stronger authoring posture on every axis the 648 caveats named (egress, reproducibility, author-capability provenance), at the cost of single-arc scope, the dose-sweep dependency, and fixture-family-scale proof.
