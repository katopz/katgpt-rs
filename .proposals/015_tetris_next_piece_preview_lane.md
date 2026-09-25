# Proposal 015 — Tetris next-piece preview: the guideline information set (`laya-tetris-v4`, after the in-flight v3)

Status: **draft — Claude verdict AGREE (round 3, 2026-09-25, session `7cd12b60`) with pre-commit text fixes applied — awaiting owner call on the Option B sequencing**
Branch: `develop` (per global rule — no feature branches)
Owner: unassigned (filed by the katgpt-rs session, 2026-09-25)
Fusion of: Issue 878 (fidelity-boundary rule) × Issue 885 (the in-flight v3 hard-drop lane) × Plan 607 (the game-lane grammar/oracle/head substrate) × Research 363 (real-time planning budgets, Tetris RT rows)
Related: [Issue 878](../.issues/878_tetris_sim_fidelity_boundary.md), [Issue 885](../.issues/885_laya_tetris_v3_real_hard_drop_lane.md), [Plan 607](../.plans/607_modelless_game_lane.md), [Research 363](../.research/363_realtime_rl_planning_budget_gate.md), [Research 576](../.research/576_Laya_Open_Jev_Generalist_RLCD_Comparison_Arena.md), [Proposal 014](014_katgpt_decision_engine_site.md), riir-ai [Proposal 028](../../../riir-ai/.proposals/028_civ_flow_field_navigation.md) (lookahead-capable navigation — the game-stack complement), riir-reflex [Issue 030 lever 4](../../../riir-reflex/.issues/030_noul_route_antialignment.md) (precedent citation; independent except the optionally-batched arena republish)

## TL;DR

Real guideline Tetris shows the next piece; our arena's Tetris lane does not — the spot question "Does the stack look clean?" is asked with the next drop blinded, on every lane equally. This proposal adds the preview as **`laya-tetris-v4`, sequenced AFTER the in-flight v3 lands** (verdict round 1: do not fold into v3 — its 102-forward oracle and its numerics-parity read are the only riir-reflex drift detector, and a near-done change must not inherit an open design). The design has three load-bearing parts: (1) the preview slot lives in the STATE sentence only ("The next piece is the {piece} piece." — the existing `TETRIS_PIECE` vocabulary, template phrasing matching "The {piece} piece is falling."); (2) **the head consumes it only through spot×piece cross features** — verified in verdict round 1: a state-constant additive column cannot change any argmax (every spot in a state shares the same next piece), so crosses are the whole feature, not a phase; (3) the oracle moves to the two-line state+option envelope — which is itself a fidelity change (the v2 oracle is **verified** board-blind: `laya_oracle_batch.rs` forwards only the option sentence as the laya state), so v4 carries a two-arm attribution: full envelope, and two-line-without-preview as the masked ablation. The corpus is **paired** — each of the 120 boards authored with all 7 next pieces (~840 states, ~18.6k forwards, tens of minutes on M3 Metal) so the preview effect is measurable within a single board. **The win is fidelity, not score**: every lane gets the same new information, and the honest expectation is that the model-based lane benefits at least as much as ours. This is a protocol-design proposal importing standard Tetris-AI practice (next-piece-conditioned evaluation), not a new research claim.

## The problem this solves

The arena's Tetris board measures evaluator quality at depth 1: the harness enumerates landing spots (macro-actions), each lane scores one typed question per spot, argmax P(clean) plays. That factorization is deliberate and stays. But the *information set* it hands the evaluator is narrower than the real game's, in TWO verified ways:

1. **No preview.** Guideline Tetris reveals the next piece; every serious Tetris controller in the literature conditions on it — accept a hole now when the next piece fills it, save the flat landing for the I-piece. The grammar has no next-piece slot (`tetris_spot()`: 5 spot slots; `tetris_state()`: names only the *falling* piece). reflex-site already deals live play from a guideline 7-bag (Issue 878 records `1cc76de`) — the site knows the true next piece and discards it at render time.
2. **The oracle is board-blind (verified, verdict round 1 + this session's grep).** `riir-reflex/examples/laya_oracle_batch.rs` forwards each option's `sentence` ALONE as the laya state ("Each option's `sentence` is forwarded as the laya `state`"). The state sentence exists in the grammar (`tetris_state()`, rendered by `render_state_sentence`) but the v2 oracle never received it — so the 44/120-class agreement anchor is measured against an oracle that sees neither the board summary nor even the falling piece. The spot sentence is the oracle's entire input.

The gap: we measure evaluation under an information set no real player (and, as of the second point, not even the full grammar) provides. Every lane is blinded equally, so the standings are fair — but the benchmark is less faithful to the task it names than it could be, and the substrate for fixing it (piece vocabulary, two-line wire shape, width-generic fitter, fixture-rule drift detector) is already shipped.

## The proposed design

### 1. Grammar — one slot in the state sentence, zero new vocabulary

`tetris_state()` gains a fifth vocabulary slot on **both** state templates (spread + flat), reusing `TETRIS_PIECE` verbatim, phrased to match the falling-piece sentence:

```text
… {TETRIS_HOLES2} The {TETRIS_PIECE} piece is falling. The next piece is the {TETRIS_PIECE} piece.
```

The option (spot) grammar is UNCHANGED — spots keep their 5 slots and stay self-contained; the preview lives in the state context, exactly where a player sees it. Reusing a vocabulary across two slots is precedented (`TETRIS_SIDE3` already fills two slots). `verify_all_closed()` picks the new slot up automatically; per-consumer grammar tables (`riir-reflex/src/game_heads.rs`, reflex-site JS) mirror the template. **Note (intended, disclosed):** once the state line reaches the oracle, the falling piece reaches it too — that is part of the envelope change below, not a side effect to hide. Per the Issue 878 rule: new grammar name (`laya-tetris-v4`), v2 byte-identical and committed everywhere.

### 2. The oracle moves to the two-line envelope — with a masked ablation arm

Today the oracle forwards the option sentence alone (§2 of the problem). v4's oracle forwards the **state line + `\n` + option line** — the flappy two-line shape the wire already speaks. Because that changes TWO things at once (state context: board + falling piece; then preview), the campaign runs **two arms**:

- **Arm A (ablation):** two-line envelope, preview line masked — the state-context-only oracle. Because a masked preview makes all 7 variants of a board send IDENTICAL bytes, Arm A runs once per BOARD (120 states, ~2.7k forwards — the two-arm cost is ~1.14× a single full campaign, not 2×), plus one duplicated board as the determinism check (its forwards must match). Labels recorded under `.benchmarks/` with a **BLAKE3 pin** (verdict round 2's condition: Arm A is the drift-control input for the v2 → A → B attribution step, and an unpinned input cannot be reproduced; three copies are not needed). Not a served fixture.
- **Arm B (candidate):** two-line envelope, preview line present — the v4 fixture's labels, the full ~18.6k-forward paired campaign.

The attribution chain is then clean: v2 (option-only) → Arm A (state context) → Arm B (state context + preview), each delta measured on the same boards.

### 3. The head — one-hot piece × spot crosses from the start (no additive-only phase)

**Verdict-round-1 correction, adopted:** the next piece is constant within a state. In a linear head, any state-constant column adds the same amount to every spot's score and **cannot change the argmax** — the round-1 "additive column first" design would have shipped preview-conditioned calibration and zero preview-conditioned decisions. The v4 head therefore consumes the preview only through interactions:

- next-piece **one-hot** (6 columns for 7 classes — pieces are categorical, not ordinal),
- crossed with the spot features that can accept the interaction (minimum: holes × piece; full form: each spot feature × piece),
- `HeadFitter` is width-generic (Plan 607 T2) — the design grows to ~40 columns (5 spot + 6 one-hot + 30 crosses); the fit recipe is unchanged.

Note the division of labor: the 6 one-hot main effects are themselves argmax-inert (state-constant) and exist only to absorb calibration; **the crosses alone carry the preview**.

**Corpus design — paired, not sampled:** 120 states × one authored preview each gives ~17 states per next piece — too thin to identify crosses. v4 authors **all 7 next pieces for every board** (~840 states, ~18.6k option forwards at v2's ~22 options/state). Cost: ~7× a v2-scale campaign — tens of minutes on the M3 Metal lane (`laya_oracle_batch`), measured before the consumer pass. The pairing is what makes the preview effect identifiable *within* a board, and it gives the T1.4 go/no-go for free (§Phased rollout).

**Holdout discipline — board-grouped, not state-grouped (verdict round 2, verified):** the shipped `loo_select` (`riir-reflex/src/game_heads.rs`) holds out one STATE at a time (`corpus.offsets[s]..offsets[s+1]` is the LOO unit). On a paired corpus that leaks: the 7 preview variants of a board share identical spot sentences, so leaving one variant out leaves its 6 siblings — same board, nearly the same labels — in training, inflating every held-out number. **Both λ selection and every G1 metric must hold out a whole BOARD (all 7 states together)**; the fitter gains a board-grouped LOO path (group id = board), the state-level path unchanged for v2/v3 corpora.

### 4. Serving, fixture, pins

`riir-reflex` tetris serving adopts the two-line shape (the flappy convention, already shipped). New fixture + fresh BLAKE3/sha256 pins per the Issue 878 rule: three byte-identical copies (katgpt-rs `tests/fixtures/`, riir-reflex `assets/game_heads/`, reflex-site `tests/fixtures/`), **hashed in the same commit** (the 885 "pins were length-only" lesson). Pins that do NOT move: the entire v2 surface (fixture `f32c8577…`, λ=1 anchors, tables). The recorded demo walks re-record; the wasm-head corpus blob + anchors regenerate.

### 5. Live-lane parity — the fairness claim is a checklist item, not a vibe

"Byte-identical questions to every lane" under v4 requires the **live laya lanes (Rust port AND the torch reference)** to receive the same two-line payload as the fitted head and the oracle. Parity test: for a pinned state set, the laya lane's answer distribution under option-only vs two-line payloads is recorded, and every lane's serving path is asserted to send the identical bytes. This rides the existing G5-parity harness.

## Sequencing — verdict round 1: Option B (v4 after v3), adopted

The round-1 recommendation was fold-into-v3; the verdict rejected it and the rejection stands on three grounds, all accepted:

1. **v3 is nearly done, low-risk, and mid-flight in another session** — tying it to an open design (bag policy, envelope, crosses) is how a finished change stalls.
2. **v3's numerics-parity read is load-bearing and must not be cancelled.** riir-reflex numerics moved 164 commits since the v2 generator; the 99 unchanged options inside v3's 3 fresh states are the only detector for that drift. A v4 full re-run would silently mix numerics drift with the preview effect — so v4 **inherits v3's fixture as its baseline** and quotes the parity read as its drift control.
3. **The cost argument was weaker than it looked.** v3's oracle step is 102 forwards; the duplicated cost is the mechanical consumer pass (anchors, wasm, demo, deploy). The one sanctioned economy: run and pin v3's oracle as planned, then decide whether to deploy v3 alone or batch its consumer pass with v4's. v4 starts from v3's `FromTop` drop rule — attribution chain: drop rule → state envelope → preview.

## Honest caveats — READ BEFORE IMPLEMENTING

1. **This is not a KatGPT score play, and must not be sold as one.** Every lane sees the same preview; the model-based lane (laya) plausibly conditions on it *better* than a ~40-column linear head. The published agreement number may go **down**. The claim this proposal makes is task fidelity; if the owner wants a score lever, this is not it. (The standing lever remains riir-reflex Issue 030 lever 4's fitted-heads arc.)
2. **Coarse spot classes may leave the oracle nothing to use the preview FOR.** "Accept a hole because the I-piece fills it later" needs well/column-shape information the 5 class slots may not express. A null outcome must be read through the T1.4 argmax-flip fraction (the decision-reliable readout) — if the oracle's argmax never changes across previews, the grammar's resolution is the binding constraint, not laya's skill — named here so a null is read honestly instead of as "preview doesn't matter."
3. **The envelope change is bundled by construction; the two-arm design prices the bundle but does not eliminate it.** Arm A/B attribution is exact only on the paired corpus; the served v4 fixture is Arm B only, and its delta vs v2 conflates envelope + preview by design. The bench record must state this.
4. **Sim fidelity boundary still applies** (Issue 878): `tetris_sim` remains a placement-enumeration oracle. The authored 7-piece preview grid is a corpus-construction choice, disclosed in the fixture `_meta`; the 7-bag truth lives site-side for live play.
5. **Rejection condition:** if the crossed head does not beat the spot-only head fitted on the same Arm B labels, under the board-grouped holdout (the G1 comparator), the preview ships for fidelity (the sentence is the game's truth) but the served head claim is limited to "preview-aware calibration, no measured ranking gain" — publish whatever is true, per the arena's honesty law.

## Fusion lineage

- **Issue 878's grammar-version rule** (new grammar + own fixture + v2 byte-identical) × **Issue 885's pipeline + parity read** × **Plan 607's head substrate** (width-generic `HeadFitter`, decoded-ordinal convention): none of the three alone gives preview-conditioned serving — the rule keeps it honest, the pipeline makes it one pass, the substrate makes it columns, not a rewrite.
- **Research 363** (real-time RL planning budgets; the Tetris-RT rows): the preview is also step 1 of the *planner* lane — preview + deterministic 7-bag is the substrate deterministic lookahead needs. This proposal deliberately does NOT build the planner.
- **riir-ai Proposal 028** (lookahead-capable civ navigation): the same fidelity arc in the game stack — information set first, lookahead second.

## GOAT gate (binds any promotion of the v4 head to the served default)

- **G1 (quality — within-state, single-delta comparator; verdict round 2):** the comparator is a **spot-only head fitted on the SAME Arm B labels** (no preview columns, no crosses) — on the paired corpus that head ranks all 7 preview variants of a board identically, so it is exactly the preview-blind baseline, and the comparison changes ONE thing (the features) at fixed training labels. The crossed head must beat it on (a) **argmax agreement with the Arm B oracle per state, across all ~840 states, with folds grouped by board**, (b) pairwise ranking accuracy within board, (c) board-centered MSE — **all evaluated with the board-grouped holdout** (§3; state-level LOO numbers are inadmissible on a paired corpus). Pooled MSE is expressly NOT the gate (round 1: it moves with calibration, not decisions). The Arm-A-fitted head is a SECONDARY disclosure (the envelope-delta head), never the gate comparator (round 2: label-and-feature-at-once comparisons attribute nothing).
- **G2 (perf):** serving stays µs-tier (~40-column linear head; two-line state) — `decision_set_goat` unchanged posture.
- **G3 (no regression):** the v2 AND v3 lanes byte-identical end to end (fixture hashes, anchors, tables); prior serving postures untouched; **the `Embedder` is untouched** (per riir-reflex Issue 030's blast-radius warning — the two-line payload changes request shape, never the feature hasher).
- **G4 (alloc):** zero-alloc serving law holds (columns and a line-split add no allocation).
- **Disclosure:** the bench page's tetris row labels the grammar version and the envelope arm; lanes compared only within a version+arm.

## What ships now (katgpt-rs) vs deferred

### Ships now — katgpt-rs (this proposal's repo)

- `grammar_tables.rs`: the state-template slot + `verify_all_closed` (behind `laya-tetris-v4`; v2/v3 tables untouched).
- `tetris_sim`: the paired next-piece authoring (7 previews per board) + `render_state_sentence` extension.
- `tetris_01_state_enum`: the v4 dump mode (paired states, two-arm oracle manifest) + fixture `_meta` disclosure (bag policy, envelope arms).
- Fixtures + digest pins under `tests/fixtures/`; the Arm-A ablation labels recorded under `.benchmarks/`.

### Deferred — riir-reflex / reflex-site (consumers; own plans/issues per repo)

- reflex: two-line serving, crossed-head refit + G1 gate + board-grouped holdout path + anchors, `fixture_pins()` triple-hash, wasm head regen. **Independence note (verdict round 2, correcting round 1):** riir-reflex Issue 030 lever 4 (per-label heads for the harness suites) NEVER reads the tetris fixture — it is a precedent citation, not a consumer, so there is **no shared fixture generation and no blocking dependency**. The only real coupling is publication timing: batch v4's arena republish (the issue-030 L61-64 item) with lever 4's post-promotion clean-window rerun if their timing overlaps; otherwise fully independent. The two-line serving must not touch the `Embedder` (Issue 030's blast-radius warning) — as designed, it does not.
- reflex-site: next-piece render + UI box, re-recorded walks, wasm regen, golden sha256 pins, deploy.

### Explicitly NOT shipped by this proposal

- **No planner lane.** Preview ≠ lookahead; multi-piece tree search over the bag needs harness-side hypothetical-future serving — a protocol extension (Research 363 is its prior-art home), not this.
- **No changes to the flappy/lanes grammars**; **no change to the depth-1 decision factorization** (lanes keep answering one typed question per spot; the harness keeps owning enumeration); **no hold-piece / SRS / wall-kick fidelity work** (Issue 878's boundary holds).

## Phased rollout (sketch — a plan would expand this)

### Phase 1 — katgpt-rs (this repo)
- [ ] T1.1 Paired-corpus dump mode (7 previews × 120 boards) + two-arm oracle manifest + `_meta`
- [ ] T1.2 Grammar slot (`laya-tetris-v4`) + `verify_all_closed` + per-consumer table sync + template phrasing pin
- [ ] T1.3 Arm B campaign (~18.6k forwards) + Arm A campaign (120 boards only — masked previews make all 7 variants byte-identical — plus one duplicated board as the determinism check); cost measured before commit; Arm A labels BLAKE3-pinned in the `.benchmarks/` record
- [ ] T1.4 **Go/no-go (verdict round 2):** measure the fraction of boards where the Arm B oracle's argmax CHANGES across the 7 previews. ≈ 0 → no head can gain anything within a state: ship the preview in the sentence only (fidelity), STOP the crossed-head/consumer/G1 work, record the negative. The decision-relevant readout — the raw MI picks up calibration shifts and cannot separate them; this fraction can
- [ ] T1.5 Fixture + digest pins + Arm A record

### Phase 2 — consumers (sequenced after v3 lands; see §Sequencing)
- [ ] T2.1 reflex: two-line serving + crossed head + G1 within-board gate + board-grouped holdout path + anchors (+ live-lane parity test, §5)
- [ ] T2.2 reflex-site: render + re-record + wasm regen + goldens + deploy
- [ ] T2.3 Bench record + arena copy + version-and-arm-labeled tables (batch the republish with Issue 030 lever 4's rerun only if timing overlaps)

### Phase 3 — evidence-gated
- [ ] T3.1 Wider crosses or richer spot classes ONLY if G1 passes AND a per-board read of the T1.4 flip cases shows the spot classes cannot express the difference the oracle makes (the grammar-resolution trigger — never a raw-MI threshold, which cannot separate grammar limits from model skill)
- [ ] T3.2 Planner-lane proposal only if this lands green (separate proposal)

## Risks

1. **Perf risk: ~none.** ~40-column linear head; two-line state; G2/G4 are formalities.
2. **Correctness risk: moderate, contained.** The three-copy fixture pin and the wasm corpus blob have gone stale before (885's finding) — hash all three copies in the same commit. The Arm A + Arm B campaign is priced (~2.7k + ~18.6k forwards); measured before the consumer pass (T1.3). The state-level-LOO leak on a paired corpus is a designed-against hazard (§3, board-grouped holdout) — the requirement must land IN the fitter, not in the bench prose.
3. **Architectural risk: low.** No new deps, no boundary crossing (decoded ordinals stay raw per the plan-607 convention). Coordination risks, both named: Issue 885's in-flight session (v3 must land first) and riir-reflex Issue 030 (independent except the optionally-batched arena republish).

## Out of scope

Hold piece; wall-kick/SRS fidelity; the planner board; any score-side narrative.

## References

Supplied by the owner's prior-art paste (cited-only, summaries verified against the paste; not re-fetched this session):
1. "Playing Tetris Using Bandit-Based Monte-Carlo Planning" (ResearchGate 229067780) — UCT over landing-spot arms; the macro-action root this arena already factorizes.
2. hrpan/tetris_mcts — macro-action MCTS + heuristic leaf evaluation; the depth-1 shape the lanes occupy.
3. Bodoia & Puranik, "Applying RL to Competitive Tetris" (Stanford CS229) — TD value functions with next-piece-conditioned features; the standard evaluation practice this proposal imports into the sentence protocol.
4. arXiv:1904.03646; arXiv:1706.02986; arXiv:2204.13307 — MCTS/TD Tetris agent lineage (cited-only).
In-workspace: Issues 878/885, Plan 607, Research 363/576, Proposal 014, riir-ai Proposal 028, riir-reflex Issue 030 (lever 4).

## Verdict trail

- **Round 1 — `#Verdict: REVISE`** (Claude, 2026-09-25, session `7cd12b60`): head design rejected (state-constant column cannot change an argmax; pooled-MSE gate misleading; ordinal encoding wrong; crosses are the whole feature; paired 7-preview corpus required) — **adopted in full** (§3, G1). Caveat 3 verified as certainty (`laya_oracle_batch.rs` forwards option sentence only) — **adopted, verified by this session's own grep** (§2, problem §2). Sequencing: Option B — **adopted** (§Sequencing). State-sentence placement approved; template phrasing fixed. Two TL;DR sections — fixed. Issue 030 citation rescoped to "lever 4"; shared fixture generation promoted to hard dependency.
- **Round 2 — `#Verdict: REVISE` (narrow)** (same session): (1) G1's comparator changed labels AND features at once (Arm-B-fitted crossed head vs Arm-A-fitted head, both scored against Arm B) — replaced with the spot-only head fitted on the SAME Arm B labels as the preview-blind baseline; the Arm-A head demoted to secondary disclosure — **adopted** (G1). (2) The state-level LOO leaks on a paired corpus (7 sibling states per board share spot sentences; `loo_select`'s LOO unit is the state) — board-grouped holdout required for λ selection and every G1 metric — **adopted, verified by this session's own grep of `game_heads.rs` L786-799** (§3, G1, T2.1). (3) The round-1 Issue 030 shared-fixture dependency was the reviewer's own category error, self-corrected: lever 4 never reads the tetris fixture — replaced with the optionally-batched arena republish + the Embedder-untouched note — **adopted** (§Deferred, §Risks, T2.3). Arm A as bench-data-only ACCEPTED on the BLAKE3-pin condition — **adopted** (§2, T1.3). Cheap improvements adopted: Arm A over 120 boards only (~2.7k forwards + one duplicate board as the determinism check), the T1.4 argmax-flip go/no-go (decision-reliable where raw MI is not), and the one-hot-main-effects-are-argmax-inert note (§3).
- **Round 3 — `#Verdict: AGREE`** (same session): all round-2 fixes confirmed in place; four stale sentences named for pre-commit repair (caveat 5's old comparator, the Related line's shared-fixture claim, the stale Status line, Sequencing ground 1's Issue 030 item) — **all four applied**; two optional wording fixes adopted (caveat 2 + T3.1 now read nulls through the T1.4 argmax-flip fraction, never a raw MI threshold; G1(a) disambiguated to per-state argmax agreement over ~840 states with board-grouped folds). **The design is approved.**
- **Status:** awaiting owner sequencing confirmation. Next actions per the verdict: Issue 885 (v3) finishes its oracle step first; a plan files from Phase 1 after the owner's call.
