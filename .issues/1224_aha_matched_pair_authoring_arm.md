# Issue 1224: AHA matched-pair authoring arm + cut-off-rate headroom gate (weight_edit_overlay)

**Status:** open — fixture-scale authoring-arm spec, CLAIMED (the 993/1209 posture; no real-model arm, the 976 serving veto stands)
**Claimed:** zed-m3-aha-1116a@m3, epoch=1794806580, expires 2026-11-18T05:23:00+00:00

**Research:** [`.research/676_AHA_Self_Supervised_Weight_Space_Programming.md`](../.research/676_AHA_Self_Supervised_Weight_Space_Programming.md)
**Source:** AHA — "Who Needs an LLM to Program an LLM?" (Khamis/Tutunov/Bou Ammar, 2026-11-15) — haithamb.github.io/aha · github.com/Eppy-EpHard/AHA (MIT)
**Family:** 980 (overlay lane, GOAT G1–G4, Bench 976) · 993 (LP authoring arm, LANDED `eb33e08e4`) · 1209 (discrete arm)

---

## Summary

AHA removes the last two agent inputs from the WSP loop-repair edit: the **direction** (matched pairs — the model's own same-prefix resampled branches, detector-labeled looping vs productive, aligned-window residual diff-of-means) and the **layers** (LP solved at every layer, ranked by held-out hit rate). Result: parity with the agent's edit on Qwen3.5-0.8B/2B, ahead of Barry's unmatched diff-of-means by 5.8/4.0/2.3 pts (GSM8K test/MATH-500/IFEval). This issue files the arm + the headroom gate as the THIRD authoring arm of `weight_edit_overlay` — un-gated, single-parent additive child, per the Issue-992/Plan-635 feature budget (no new flag; katgpt-core sits under the crates.io 300-feature ceiling).

## Scope law

- **Fixture-scale only** (the 990 precedent: the family's real-model PoC record is 0/1; no real-model claim made or implied — any arm drifting toward a real model is out of scope and closes as such).
- **No new feature flag** (the 993 budget law).
- **The 976 promotion veto is unchanged** (re-verified 2026-11-16, Research 676 §4.6): arming still requires a documented loop class in a SERVED model. The Bonsai Go arena's degenerate position-bias (riir-ai Bench 866) is a different failure class on an offline arena lane, already worked via trained LoRA (riir-ai Plan 528, riir-ai Bench 618).

## Tasks

- [ ] **T1 — pair builder on the wsp980_common fixture.** Shared-prefix branch rollouts from the fixture's looping episodes (seeded variants from identical prefix states — the fixture analog of AHA's 8-continuation resample); label every branch with the fixture's oscillation detector (the wsp980_common labeling step); keep prefixes with ≥ 1 looping and ≥ 1 productive branch; pair earliest-looping × longest-productive. Yield assertion with loud refusal on starved prefixes (the paper's 228/300 discipline — a yield floor, named in the gate test).
- [ ] **T2 — matched-pair direction + comparison arms.** `u_matched` = normalized mean over 80% of pairs of aligned-window state differences (20% held out — the window anchor is PINNED to the repo README's onset-relative reading: 160 states starting 64 after the looping branch's detected onset, the same offsets applied to both branches; the blog's "tokens 64–224 of each branch" is the looser wording). Feed `u_matched` to the UNCHANGED 993 LP (`compose_direction_overlay`) and run the FULL 993 battery (held-out separation, determinism digests, dose grid, isolation null, 16-arm random control, twin pre-screen, retention bit-identical). **Pre-registered bar: PARITY, not superiority** — the fixture cannot reproduce the real-model confound that makes matched pairs matter (state this in the gate doc); `u_matched` passing the battery ≡ construction correctness + determinism.
- [ ] **T2b — the confound falsifiers (the arms that CAN fire at fixture scale).** F1: deliberately mismatched-prefix pairs (pairs built from DIFFERENT prefixes — the unmatched-pile shape that made the whitened direction fail) must measurably degrade held-out separation or trip a battery arm. F2: sampler-confound analog (branches generated under different seed statistics) must degrade or trip the confounder audit. If F1/F2 do NOT degrade, record honestly — the fixture lacks confound structure and the matched-pair parity claim stays construction-only (demotion-law input).
- [ ] **T3 — cut-off-rate headroom gate.** `arm_from_telemetry` gains the headroom readout: recoverable share = fraction of obstacle episodes ending in truncation (the fixture analog of the base model's cut-off rate). **Report beside the arm verdict; hard-refuse `HeadroomBelowFloor` ONLY at ~0 recoverable.** Pre-registered anti-falsifier: a FIXED-rate skip gate is forbidden — the paper's own "<0.40 skip" gate wrongly skipped the 2B (0.23 cut-off, +14.1 pts). The bound is a ceiling REPORT, not a threshold refusal.
- [ ] **T4 — layer-ranker doc row (no code).** The fixture is layer-agnostic by construction — no fixture test is possible. Record the two no-agent layer answers + the placement-insensitivity law in the `lp_authoring`/`discrete_authoring` module docs: 668's forward write-score scan, AHA's LP-hit-rate ranker (different signals, same purpose), four derived-vs-agent comparisons all < 1.6 pts. Fixture-scale arms stop agonizing over layer choice.
- [ ] **T5 — landing + close.** Battery + bench record per the family pattern (`.benchmarks/NNN_issue1224_*.md`); 648 §9 / 664 §7 pointers already landed with Research 676 (2026-11-16); close this issue with the HISTORY.md row per the noise rule and fold the claim line per the lease law.

## Demotion law (pre-registered)

If the matched-pair arm fails battery arms the heuristic diff-of-means direction passes, or T2b's falsifiers fail to fire AND the fixture is shown to lack confound structure entirely, record the negative and strike the arm from the lane docs — the LP arm (993) and discrete arm (1209) are untouched either way.
