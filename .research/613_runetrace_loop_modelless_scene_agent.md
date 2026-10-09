# Research 613 — The Runetrace Loop: a modelless scene agent from shipped GOAT primitives

**Status:** DESIGN — fusion over shipped substrate (no new primitive); the composition is the contribution. Feeds seal-remake Proposal 005 Phase 4 (the instinct/rethink lane). Quality claims UNPROVEN until the lane's G-gates run (§6). Verdict: **AGREE** after 2 rounds (2026-10-09) — sub-agent reviewer path (`request_verdict` claude backend rate-limited until 22:00 ICT; the §5-documented equal fallback; 4 REVISE bullets discharged: ignition Bench 666, GOAT-blanket scoped, KARC Issue-866 QUALIFY + Bench 849, §8 harness homed instinct-side).
**EXECUTED (first lane): instinct Plan 010, `ffdc9a8` (2026-10-09)** — the modelless loop + G1/G2 harness landed behind `runetrace_loop` (G1 byte-determinism PASS; G2 136 µs/decision @ 32 entities; the Ext-seat serve wiring + rethink arm + hosted leg stay deferred).

**Date:** 2026-10-09
**Provoked by:** owner ask — "apply all tactics: karc, elo, leo duel leo, shallow-reasoning looped transform, reflex, instinct, all GOAT, on top of Runetrace; think novel, e2e agent + latent-space first."
**Pinned claim (§1.5 form):** *A per-entity decision loop that embeds the structured Runetrace scene DOC (never parsing rendered text), forecasts drive trajectories with KARC, rates strategy arms with an Elo book (the AttackEloBook generalization), selects goals via LEO/dual-LEO salience, refines belief in a bounded looped shallow-reasoning pass (gain_cost_halt + state_probe + risk_control_exit + ignition + saddle_escape), answers on decision_wire with first-class abstention, and renders its OWN reasoning trace as a Runetrace EntityBlock — digest-stable, self-inspectable, freezable — distinguished from Voyager/Generative-Agents (LLM-in-the-loop) by the modelless floor, with the LLM as escalation (instinct Ext seat), never the core.*

**Primitive-quality disclosure:** every §2 row SHIPS; most are GOAT-proved, but the refine-control tails (`saddle_escape`, `entropic_tilt`, `state_probe`) are opt-in with pending or PoC-gated quality records — see §2's Gate cells + §6 caveat 1.

## TL;DR

It is all connected already — by design. Every stage of an agent loop (sense → forecast → rate → goal → refine → decide → explain → remember) ships as a named primitive in this workspace (most GOAT-proved; the refine-control tails opt-in — §2). Runetrace (Plan 624) and decision_wire (Plan 603) are the two wire edges that close the loop: **scene in as a deterministic DOC, judgment out as a typed answer, and the agent's own reasoning back in as another Runetrace block.** This note pins the composition so the Phase 4 lane plan consumes it directly. No new primitive is proposed; the novelty is the loop (§3).

## §1 The loop

```
        ┌────────────────────────── RunetraceDoc (tick T) ─────────────────────────┐
        │  entities: drives (think rows), vitals, dag rows — RAW exact, digested   │
        └──────┬───────────────────────────────────────────────────────────┬──────┘
               ▼                                                           │
   [1] EMBED  doc → latent  (reflex embed.rs hashed-feature discipline —    │
               hash the STRUCTURED fields, never parse the rendered text)   │
               ▼                                                           │
   [2] FORECAST  KARC delay-embedding over the tick-T capture window →     │
               next-belt drive forecast per entity (karc_bridge shape)     │
               ▼                                                           │
   [3] RATE  Elo book over (strategy-arm, context-class) — the             │
               AttackEloBook generalization; outcomes = next capture's     │
               vitals deltas (raw), book stays think-brain-local           │
               ▼                                                           │
   [4] GOAL  LEO / dual-LEO salience over forecast+ratings →              │
               sigmoid_bounded_q goal ranking (the engine's own law)       │
               ▼                                                           │
   [5] REFINE  bounded looped shallow reasoning on the latent belief:     │
               iterate contrast_combine step · entropic_tilt until        │
               risk_control_exit (confident) or gain_cost_halt (marginal  │
               gain < drift cost × τ) or ignition patience expires;       │
               saddle_escape kicks a trapped loop                         │
               ▼                                                           │
   [6] DECIDE  decision_wire answer (choice/score/noul), abstention       │
               first-class — CorpusDistanceGate shape; any confidence     │
               claim floors against conformal-naive (Report the Floor)    │
               ▼                                                           │
   [7] SELF-TRACE  the loop's own fired conditions + path rendered as a  │
               Runetrace EntityBlock (stage rows: perceive→…→action) ◄────┘
               ▼                                                           │
   [8] REMEMBER  digest-stable captures (blake3 over canonical text) →   │
               freeze/sleep consolidation (Proposal 005 Phase 5)
```

Stages 1–6 are advisory inputs to the existing brain+FSM (proposal 005 caveat 7) — never a bypass.

## §2 What ships (grep-verified homes)

| Stage | Primitive | Home | Gate |
|---|---|---|---|
| wire-in | `runetrace` (`RunetraceDoc`, digest) | katgpt-core (Plan 624, this session) | G1 byte-pin, opt-in |
| embed | hashed-feature embedding | riir-reflex `embed.rs` | reflex G1–G5 |
| forecast | `karc::KarcForecaster` + `karc_bridge` (`NpcKarcState`) | katgpt-core `karc_forecaster` (default) + riir-engine `karc_runtime` | Bench 308 GOAT — **split-config; Issue 866 QUALIFY** (passing legs sit on Chebyshev configs no consumer constructs; every deployed Fourier R=1 shape fails both D3 bars) + **Bench 849** (deployed-shape one-step NRMSE 1.2–4.1e-3) — the stage-2 usage IS the deployed shape |
| rate | `rating` (Elo expected-score) + `AttackEloBook` precedent | katgpt-core `rating` (default) + riir-games `swarm/attack_reasoning.rs` | Bench 675 / 898 GOAT |
| goal | `leo_all_goals` (`LeoHead`, `sigmoid_bounded_q`) + `dual_leo` (`DualLeoMixer`) | katgpt-core (both default) | promoted |
| refine-halt | `gain_cost_halt` (`GainCostLoopHalter`) | katgpt-core | kernel bench |
| refine-trap | `saddle_escape` (Continue/Converged/Trapped/Kick) | katgpt-core | opt-in — quality gated on the Phase-3 defend-wrong PoC |
| refine-stop | `risk_control_exit` (dual-threshold stop-when-confident) | katgpt-core | Bench 681 lane |
| refine-worth | `state_probe` (V̂ pass-fraction — is more compute worth it) | katgpt-core | Plan 621 — Phase-1 substrate (2026-10-08); quality claims wait on its lane |
| refine-patience | `ignition` (`commit_time_star`) | katgpt-core `ignition_schedule` | Bench 666 GOAT G1–G4 |
| refine-step | `contrast_combine` (LoopCD `z_R + ω(z_R − z_k)`), `entropic_tilt` | katgpt-core | contrast_combine gated by its consumers; entropic_tilt opt-in pending the Plan 341 Phase 2/3 GOAT |
| decide | `decision_wire` + fused abstain + `SigmoidGateCalibrator` | katgpt-core + riir-reflex | reflex gates |
| UQ floor | `conformal` (`ConformalIntervalCalibrator`) | katgpt-core (default) | Report-the-Floor law |
| compose | H1 cascade + H2 prior fusion `p′ᵢ ∝ pᵢ·exp(g·β·mᵢ)` | riir-instinct `hybrid.rs` | Bench 0029 etc. |
| escalate | Ext seat / `LaneBackend`, `EscalateSpec` grammar | riir-instinct `server.rs` / `arsenal.rs` | rethink byte-pin |

## §3 The novel combination

No single shipped piece does this, and the cited prior art does not do it modellessly:

- **Voyager** (arXiv:2305.16291) and **Generative Agents** (arXiv:2304.03442) put an LLM inside the loop; ours is the modelless floor with the LLM as an escalation rung (instinct/rethink ESC) — the loop runs at game cadence with zero inference spend by default.
- KARC ships as a per-NPC HLA forecaster but nothing composes its forecast into a DECISION prior.
- The Elo book ships for attack options only; nothing rates strategy arms against forecast trajectories.
- LEO ships as goal machinery; nothing feeds it KARC forecasts + Elo priors.
- The loop-control primitives (halt/stop/worth/patience/escape) each ship standalone; nothing composes them into ONE bounded reasoning pass.
- Runetrace just shipped as a format; nothing closes the loop by rendering the agent's own trace back into it — **the self-trace closure is the genuinely new surface**: the agent is inspectable by its own observation format, and the digest law makes every loop trajectory a freezable memory atom.

## §4 Latent-first laws (the loop's constraints)

1. **Embed the DOC, never parse the text** (the ConvexTok law, one layer up): the renderer's output is for humans/LLM lanes; the modelless agent hashes structured fields.
2. **Raw stays raw**: vitals/positions ride the doc raw; outcomes for Elo are raw deltas; only ratings/forecasts/goal-rank are latent. Nothing latent is ever parsed back into game truth.
3. **Sigmoid, never softmax** — every gate/projection/fusion (H2, salience, calibration).
4. **Think-brain locality**: Elo books, KARC rings, beliefs are per-entity local, never synced; only the decision (and its receipt) crosses.
5. **Cadence follows observation**: the refine budget (stage 5) is spent only where state_probe says worthiness holds — limelight's law applied to reasoning itself.

## §5 Self-trace closure (why stage 7 matters)

The loop's own iteration renders as an `EntityBlock`: `think` rows = the fired control conditions (`gain < drift×τ`, `confidence ≥ θ_hi`) with their drive values; `dag` rows = the actual path (perceive the forecast → believe the rating → drive the tilt → goal → transition → action the decision). Consequences: (a) the panel/debug surface shows the agent with the same inspector as heroes/monsters; (b) blake3(content) gives every trajectory a stable identity for diffing; (c) Phase 5 sleep-consolidation consumes trajectory digests as first-class memory atoms; (d) an escalated LLM rung receives the agent's own trace as context — the "why did you do that" artifact — without any new format.

## §6 Honest caveats

1. **Architectural note, not a measured quality claim.** No PoC yet; §3.6 discipline applies — the lane plan must run the G-gates (G1 determinism of the end-to-end trace; G2 inside the lane's latency bar; G5-style abstention honesty vs the modelless floor) before any "beats/parities" wording. This disclaimer covers the COMPOSITION; it does not blanket-bless the primitives — three refine-control tails (`saddle_escape`, `entropic_tilt`, `state_probe`) ship opt-in with pending/PoC-gated quality records of their own (§2 Gate cells), and KARC's passing GOAT legs sit on configs no consumer constructs (Issue 866 QUALIFY; the deployed-shape record is Bench 849).
2. **KARC needs capture history** — the forecast window requires belt-cadence Runetrace captures to exist first (a producer-side dependency, not free).
3. **Elo outcome normalization is a design knob** (damage_frac precedent normalizes to target max HP; a scene-general normalization must be defined per question kind).
4. **The refine loop is bounded but not free** — the composition must state its worst-case compute per decision and let state_probe/risk_control_exit actually gate it, or stage 5 becomes the anti-pattern its own halt primitives exist to stop.
5. **No second brain** — outputs are advisory to the FSM (standing owner rule; a bypass oscillates).
6. **Hosted-LLM escalation stays owner-gated** (refine `hosted_expert` consent/spend pattern; Glacial cadence).

## §7 Prior art + internal cousins

- External (cited-only): Voyager (2305.16291), Generative Agents (2304.03442) — LLM-in-the-loop; the modelless floor is the delta. No external "modelless scene agent" prior art was searched for yet — the lane plan runs the §4 search before any novelty claim beyond the composition-over-internal-substrate one pinned here.
- Internal: no `.research/` note composes these (grep 2026-10-09: `runetrace|scene-as-text|agent loop` — the phrase-cousins Research 364's parked F2 "Reflex-Auto", Research 289 RecursiveMAS, Research 104's "Basic agent loop → G-Zero" row all lack both the composition and the self-trace); closest per-stage cousins are karc_bridge, attack_reasoning, hybrid.rs — none closes the loop or self-traces. The lane plan's §4 search sweeps those cousins by name.

## §8 Route

- The lane implementation is **instinct** (open grammar + Ext seat; Phase 4 of seal-remake Proposal 005) with **rethink** as the trained escalation arm (private, per the split law); reflex contributes the calibration/abstention discipline it already owns. A G1/G2 composition harness wires naturally in the **instinct lane** — where the riir-reflex dep is legal — over stages 2–7 + stage 8's digest (all katgpt-core: KarcForecaster, rating, LEO, the refine family, decision_wire, runetrace; stage 8's freeze/sleep consolidation half stays ndb / Proposal 005 Phase 5); stage 1's shipped embedder homes in riir-reflex `embed.rs`, and katgpt-core has no hashed-bag substrate of its own (katgpt-rs is upstream of every riir repo — a reflex dep there would cycle), so a katgpt-core-side harness would need a fixture-local mirrored embedder as test scaffolding, explicitly NOT shipped substrate. That harness is its own plan, not this note.
- This note records the design; it creates no files beyond itself and edits only Proposal 005's Phase 4 pointer.
