# Issue 1247: Looped-lane copy-through cure — WSM weight-space forget mask + rotary position reset

**Status:** Open — filed from `../riir-infer/.research/045_Weight_Space_Micromachines_Implants.md` §5 T5 (verdict AGREE round 1, 2026-11-26). Own-repo issue for the looped family; back-reference only in riir-infer Issue 1256.
**Claimed:** zed-m3-wsm-followup, 1795673675, expires 2026-11-28T06:14:35Z

Source: "Weight-Space Micromachines" (blog 2026-11-26, papail.io/papers/wsm). Two of their GSM8K single-call implants are candidate cures for a pathology OUR looped lanes document but never cured:

- Their **weight-space forget mask**: 4 fixed extra q/k dims per head hide earlier tries from later ones — plain retries copy the failed attempt through attention (their datum: on a retry the model copied its failed attempt and arrived at an impossible answer in 62 of 80 cases).
- Their **weight-space position reset (WPR)**: an offset register + fixed rotary table make every retry see the question at the first retry's distance — their datum: 8 tries in one call WITHOUT WPR = no gain (71.9 vs 72.3 with one try); WITH WPR = 80.7 (+8.8), matching 8 separate calls + majority vote; WPR ≈ the programmatic reset it replaces (−0.3, n.s.).

Our family: the looped lanes — katgpt-rs `.research/097_Training_Free_Looped` (training-free ODE-substep looping of a frozen checkpoint, Plan 136) and `414_Fully_Looped_Readout_Blind_Spot` (parameter-free loop-stability fix taxonomy + readout-blind-spot diagnostics); `073_LT2` (training-time looping, shipped). Later loop iterations attend earlier ones at longer effective distances — the same copy-through/positional-drift class WSM names, with no weight-space cure recorded in `414`'s taxonomy.

- [ ] **T1 — Read the family against the two mechanisms before any build.** Grep the looped lanes for how iteration k attends iterations < k (mask? distance? nothing?) and whether `414`'s fix taxonomy already contains a forget/position analog under another name (vocabulary translation: forget-mask ≈ attention embargo / causal gap / iteration isolation; position reset ≈ rotary re-basing / distance clamp / re-anchoring). Record the signal-diff per hit.
- [ ] **T2 — Pre-register the cell** (only if T1 shows the gap is real): a looped decode on the frozen checkpoint fixture, iterations k ∈ {4, 8, 16}, arms: incumbent vs +forget-mask (extra q/k dims per head) vs +position-reset (rotary offset table) vs both. Bars: per-iteration answer divergence (copy-through rate) down; final-answer quality not worse; byte-identical with the arms off (the standing flag-off law). Quote bench_preflight PROVENANCE.
- [ ] **T3 — Implementation behind an opt-in feature** (`looped_forget_wpr` or split), never default without the GOAT gate; the mask/reset tables are deterministic constructions (no training — same law-set as katgpt-rs 621/690).
