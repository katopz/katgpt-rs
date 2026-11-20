# Issue 1237: Modelless neuron-attribution primitives — AUROC screen, typed control verdicts, decoy battery, ColumnScale edit form

**Status:** in progress — claimed by katgpt-rs-92@m3; consumer lane riir-infer Issue 1241.

**Claimed:** katgpt-rs-92@m3, 1795184362, expires 2026-11-22T14:19:22+00:00

**Filed from:** katgpt-rs Research 684 (Fermion "Mapping every neuron in Qwen3-8B" distill, 2026-11-20). The reviewer round-1 repo-split: these are modelless scoring primitives with zero riir deps — katgpt-rs owns them, riir-infer consumes. Carrier surfaces: katgpt-pruners `mech_attribution` (contrastive attribution already ships there, `cna_steering` feature) + katgpt-core `weight_edit_overlay` (Issue 980, `EditForm` family).

## Tasks

- [ ] **T1 — `screen_auroc`** in katgpt-pruners `mech_attribution`: closed-form rank AUROC of a unit's activations on labeled data vs 20 scrambled-label searches, admission margin 0.20; params pinned in a param-pin test. G1 canaries both directions: planted labels fire, genuinely shuffled labels refuse. G4: alloc-free scorer.
- [ ] **T2 — typed control-verdict battery** beside it: `SwitchFound` / `NoSwitch{band_measured}` / `Undecided{under_floor}` — never pooled; a confident negative requires the random band measured at floor (the semantic that makes Fermion's "no liar neuron" a result rather than an absence). G1: three canary arms — planted strong effect → SwitchFound; under-floor read count → Undecided; full read with band measured → NoSwitch permitted.
- [ ] **T3 — two-sided decoy battery** beside katgpt-core `closed_form_weight_write`: per candidate `WeightEditSpec`, score chosen edit vs 20 decoys (chosen direction on random sites / random directions on chosen site) on ONE frozen held-out read; verdict = chosen ≥ best-decoy + pre-registered margin. G1 canary: a decoy planted as the edit must FAIL the battery. The two-sided upgrade of the 980 gate battery's single norm-matched control arm.
- [ ] **T4 — `EditForm::ColumnScale`** in `weight_edit_overlay`: down_proj output column at dose d ∈ [0,1] (the fader — Fermion's switch surface). G1 dose-linearity: forward delta at d ≈ d × full-dose delta within a documented float-assoc tolerance; apply at d ∈ {0,1} bit-exact; round-trip through `forward_delta_into` pins the algebra. G3: unarmed builds byte-identical.

**Kill bar (shared with 1241):** if the consumer census cell (riir-infer 1241 T2) closes NEGATIVE, these primitives demote with it — they are scorer/edit-form halves of that lane, not standalone products; nothing here lands without the consumer gate existing. Boundary: no game/chain/shard semantics; generic modelless primitives per this repo's domain test.
