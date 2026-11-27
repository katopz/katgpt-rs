# Issue 1248: Closed-form loop-depth extension for looped forwards (FP-Solve class)

**Status:** OPEN — GAIN filing; PoC not run; real-model cell CONSUMER-GATED (no converged looped checkpoint on disk — Plan 324 postmortem ask / Plan 447 base rows) + hermetic-G1-first.
**Claimed:** zed-i1248-t1@m3, 1795756939, expires 2026-11-29T05:22Z

> **Source:** katgpt-rs/.research/697_Recursive_Self_Improving_Loopers_FP_Solve.md — rse_looper @ `765453d13878804c49876a656f39b57a4384dca4` (pin recorded; clone removed at filing).

## The primitive

An iterated computation that already runs at depth d provides its own supervision for extending to depth d+1: per control matrix, solve the protected ridge

```
dWᵀ = (G_edit + λ_keep·G_keep + λ₂·I)⁻¹ · (X_editᵀ·dY / N_e)
X_edit = pass-(d+1) inputs; dY = Y_clean(d) − Y_current(d+1)
G_keep = pass-1..d input Grams (pass-domain soft null-space)
λ₂ = ridge_frac · tr(G_edit)/D  (scale-invariant)
```

and apply as a rank-r overlay pair (`QuantErrorLora` — reused, never forked) at the looped layers' control tensors, base artifact unmoved (the overlay law: BLAKE3-pinned envelope, empty overlay demotes byte-identically, structural undo). **The loop is the only label source** — no facts, no teacher, no gradient (the §3.5 path-2 class: deterministically constructed weight correction).

Signal-diff vs shipped cousins (Research 697 §2): `closed_form_weight_write` = authored-fact targets + prompt-key keep set; Bench 699 `lt2_deep_stability` = state-space damping, per-request cost; this class = loop-internal targets + pass-domain keep + one-time weight-space repair. **Cost posture:** "zero per-request cost" holds only in the MERGED-overlay posture (deltas folded into the loaded weights at mount); an UNMERGED rank-r overlay costs O(r·d) per token at serving — the cost census is T7.

Novelty boundary (pinned): using the loop's own pass output as a target is KNOWN at training time (DEQ fixed-point/Jacobian regularization; pass-consistency losses in looped-LM training). What is claimed new here is narrow and inference-side: the CLOSED-FORM solve (no gradient step) with a PASS-DOMAIN keep set, applied to FROZEN weights at serving, to license depth extrapolation — the escape hatch Research 608's law lacked.

## Why the fixture is NEW (the Bench 699 fixture cannot host this cell — verified 2026-11-27)

The Bench 699 DESTABILIZED arm carries state as `h ← ρ·h + h̃` with ρ_τ = 1.1 — the multiplier is 1.1 FROM PASS 1: there is no healthy pass for `dY = Y_d − Y_{d+1}` to target, the divergence is a scalar carry mode no edit to F's (RMSNorm-fed) matrices can cancel, and after RMSNorm the pass-(d+1) inputs sit inside span(G_keep) — the solve would come out near zero, while the governor arm reduces to the already-shipped radial `direction_scales` knob (Bench 699 T4: radial ×0.25 rescues). Both directions of the ladder would be rigged. Bench 699 stays in the cell as the DAMPING INCUMBENT's control, never as this primitive's harness.

## The fixture: `loop_depth_ood` (constructed, zero training)

An `lt2_looped` micro-class config with **n_layer ≥ 2** whose pass map is a **contraction to depth d by construction** (residual scaled so passes 1..d converge), plus ONE **pass-indexed gate module** at a single layer: a gate whose input carries a **continuous pass indicator — a scalar ramp t/d (low-frequency sinusoid as the fallback spelling), PINNED over one-hot: a one-hot indicator is orthogonal to span(G_keep) at d+1, so G0 would pass trivially, the edit would write into empty space, and bar (b)'s d+2/d+3 extrapolation would fail for the wrong reason (new orthogonal directions each pass, uncarriable by construction) — while a continuous encoding gives the d+1 input partial overlap with span(G_keep), the Inkling regime: a SEEN direction whose influence flips** — at pass d+1 the indicator leaves the constructed region, the gate output collapses/flips (the Inkling fault class), and the fault is LOCALIZED at the constructed layer. Real gate modules ⇒ R3 testable; n_layer ≥ 2 ⇒ R2 testable; healthy ladder to depth d ⇒ `dY` meaningful. The encoding + parameters land in T1 before any code; the G0 overlap number is REPORTED alongside every result (verdict round 2, pin 1).

**G0 sanity bar (runs before any arm):** span census — assert `X_edit` NOT inside span(G_keep) (min-eigenvalue of the edit-keep joint Gram + span-overlap metric reported), and assert the edit arm's dW clears a norm floor. A rigged ladder fails here, in either direction.

## Proof path (pre-registered)

Arms on ONE frozen read:

1. baseline — `loop_depth_ood` at depth d+1, no intervention (fault signature must reproduce: gate collapse + sign flip at the constructed layer);
2. damping incumbent — `lt2_deep_stability` (the shipped control, Bench 699 lineage);
3. R4 governor — project-out at the diagnosed site;
4. FP-Solve edit — closed-form, one-time, then depth d+1/+2/+3 with no runtime knob (4b: merged vs unmerged posture).

Bars: (a) stabilization parity arm 4 vs arms 2/3 on state norm + readout consistency; (b) **ALL transitions** repaired (F3), measured at d+1 **and extrapolated d+2/d+3** (the F10 generalization claim becomes a BAR: fit at d+1, assert d+2/d+3 hold); (c) depth-d outputs preserved within pre-registered per-suite floors (F2: soft keep, mixed signs allowed, never identity); (d) ‖dY‖ decay census (F8: a productive limit cycle is DETECTED and left alone — the contraction must not annihilate productive depth); (e) G0 span census passes; G4 alloc-free solve+apply; Issue-940 determinism (pinned f32 folds, golden digests, capture on the serving posture — F6/F7); (f) R2 localizes the fault to the constructed layer only; R3's alarm fires at d+1 and never at 1..d, and graceful exit ≥ refusing the depth.

**Winner rule:** the whole ladder gates on the same frozen read — if the governor (one vector) matches the edit (many matrices), the governor wins and the edit stays opt-in; if the edit wins it must ALSO beat the damping incumbent, else the incumbent stays. Demote the loser in the same landing.

## Tasks

- [ ] **T1** — `loop_depth_ood` fixture (constructed pass-indexed gate with the PINNED continuous pass-indicator encoding — scalar ramp t/d, sinusoid fallback — written here BEFORE code; n_layer ≥ 2, contractive to depth d) + G0 span census (overlap number REPORTED with every result) + fault-signature self-check (arm-1 baseline reproduces the Inkling shape)
- [ ] **T2** — R2 fault-signature instrument (offline rank-1 SVD of the pass-shift + per-pass projection census; readout-screen class, never a serving path)
- [ ] **T3** — R3 gate-saturation alarm + graceful early exit (never worse than refusing the depth)
- [ ] **T4** — R4 spectral governor (`h' = h − γ·(v₁ᵀh)v₁`, one frozen vector per fault site, γ from gate telemetry)
- [ ] **T5** — the edit arm (protected-ridge solve → rank-r `QuantErrorLora` overlay pair; capture on the serving posture; Issue-940 folds; envelope digest binds base + capture posture)
- [ ] **T6** — the GOAT ladder A/B (arms 1–4, one frozen read) + bars (a)–(f); demote the loser in the same landing
- [ ] **T7** — merged-vs-unmerged overlay cost census (the O(r·d)/token disclosure, measured)
- [-] **T8** — real-model cell — CONSUMER-GATED: needs a converged looped checkpoint on disk (Plan 324 postmortem ask; Plan 447's ARLoop/Ouro base rows are the candidates). Hermetic G1 (T1–T7) passes first; the cell opens when the base lands, under Plan 447's own status line — no new gate created here.

## Explicitly NOT claimed

- Depth UTILITY (F1): the repair stops harm, never claimed to add capability — the source's own 4-pass 42.2% < 3-pass 50.0% row is the standing refutation; only the riir-train GSPO bonus lane (713) tests "depth pays".
- 3-pass preservation as identity (F2): soft keep only, floors per suite.
- Fault elimination (F3): containment only; all-transitions gating is the honest read.
- Training-time novelty: pass-target/consistency objectives are known in DEQ and looped-LM training — the claim is pinned to the closed-form inference-side solve with the pass-domain keep set (Research 697 §3).

## References

- Research 697 (master record, Path 0 table, falsifier catalog F1–F10, verdict record)
- Research 608 serving law — amended with the conditional escape hatch (same commit as this filing)
- Issue 717 / Bench 699 (`lt2_deep_stability`, damping toolbox, T4 radial/tangential probe) — CLOSED, record lives in the bench file; the destabilized ρ-carry arm is WHY the fixture here is new
- Issues 990/1189 + Research 659 (`closed_form_weight_write` — the MEMIT-family solve substrate, PoC-NEGATIVE precedent for honest gated retention)
- riir-train Issue 713 (the GSPO control-tensor recipe routing — the track-c counterpart)
