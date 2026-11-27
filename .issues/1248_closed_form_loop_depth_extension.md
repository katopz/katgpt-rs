# Issue 1248: Closed-form loop-depth extension for looped forwards (FP-Solve class)

**Status:** OPEN — GAIN filing; T1–T5 LANDED (fixture + G0 + fault-signature; R2 localizer; R3 alarm + graceful exit; R4 governor arm; the FP-Solve edit arm — 2026-11-27); T6–T7 open (the frozen-read ladder + the cost census); real-model cell CONSUMER-GATED (no converged looped checkpoint on disk — Plan 324 postmortem ask / Plan 447 base rows) + hermetic-G1-first.
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

## T1 fixture design (pre-registered parameters — REVISED at implementation, 2026-11-27)

**Revision note (honest):** the design below was "written before code" per T1 in an earlier shape (a rank-1-dominant two-axis geometry, `u`/`b`). Implementation + measurement showed that geometry cannot satisfy its own G0 bar and the sign-flip bar at the same time: in a converged loop the fixed point always carries the write directions, so the keep span swallows the collapse deficit (overlap ≈ 0.95-0.99 — solve-dead, the round-1 kill reproduced), and per-row warm residuals fatten the span further. The revision follows the SOURCE's own telemetry shape: the healthy loop's shifts and the fault shift live on ONE axis — the fault REVERSES a steady drift (their `v₁` per-pass projections `[−137.2, −121.2, +114.6, …]`) — which both separates the signs structurally and keeps the keep span thin. The parameters below are AS-BUILT (committed with the fixture; the fixture constants carry the same values); all bars held unchanged.

**Encoding (verdict round-2 pin 1):** continuous pass indicator `φ_k = k/d` — the scalar ramp, PRIMARY; sinusoid fallback `φ̃_k = sin(π·k/(d+1))` compiled with the SAME κ (band-derived hinge, no re-tuning), not the GOAT arm's encoding. Never one-hot (G0 would pass trivially; bar (b) would fail for the wrong reason). The G0 overlap number ships beside every result.

**Geometry (as-built):** `D = 64`; gate input `[h1; φ] ∈ R^65`; `d = 3`; ladder `K = 6 = d+3`; batch `B = 12` seeded starts (the prompt-batch analog). Seeded orthonormal axes `e₁, e₂, e₃` (SplitMix64 master seed 1248 → Gram–Schmidt).

- **rmsnorm direction-contraction:** every layer applies `rmsnorm` BEFORE its linear map and the pass carry feeds the readout back — the state converges in DIRECTION at stable magnitude; no affine anchor needed.
- **Layer 1 (fault-free):** `h1 = A1·rmsnorm(x)`; `A1 = 0.85·e₁e₁ᵀ + 0.008·(I − e₁e₁ᵀ)` — the alignment mode pulls the state's direction toward e₁ and the near-zero off-mode kills off-content (0.03 was measured insufficient: y's 80%-off share × 0.03 still left ~25% off in the fault mix).
- **Gate (the constructed fault site, after layer 1):** `z = w_hᵀh1 + 0.3·φ + c − κ·relu(φ − 1)`; `g = σ(z)`; `h1g = h1 + g·W_AMP·ŵ`. In-band mild upward drift (the source's 0.566→0.638 shape); `W_AMP = 40`, `κ = 26`: in-band `g ≈ 0.70`, at `φ = 4/3` the hinge `26/3 = 8.67` drops `g ≈ 0.0002` — a ~3500× collapse. The RATIO (not the amplitude) is the discriminator: with `W_AMP = 40` the fault's residual write must LOSE to the state content or the fault mix stays write-flavored (measured: κ = 14 left the fault write ≈ the state norm and cos(y₃,y₄) = 0.98).
- **Write direction `ŵ = normalize(0.30·e₁ + 0.55·e₂ + 0.77·e₃)`** — positive on all three axes: the `+0.30` e₁-weight puts `⟨e₁, A2ŵ⟩ > 0`, which is the sign-flip's sign convention; the e₂/e₃ dominance is what the collapse swings.
- **Layer 2 (the OOD-consuming layer — the edit target, mirroring the source's 168 control matrices, never the gate itself):** `y = A2·rmsnorm(h1g)`; `A2` diagonal in the seeded basis with modes `(0.3, 0.9, 0.9, 0.1…)` on `(e₁, e₂, e₃, rest)`. The small e₁-mode is load-bearing: at the fault the write dies, layer 1's off-kill leaves the layer-2 input e₁-pure, and the fault output collapses onto the (weak) e₁ channel — measured `cos(y₃,y₄) = 0.14-0.44-class` — while the healthy output rides the write through the strong e₂/e₃ channels.
- **Warm start (the serving analog):** the loop run to its fixed point from a canonical seed (row-independent), then a SHARED `+0.15·e₁` displacement (excess) plus per-row noise `0.005`. The shared displacement gives the healthy ladder its pull-back shifts — decaying, ONE sign (−e₁-ward), every row — OPPOSITE the fault's +e₁-ward collapse shift; that is the source's `[−, −, +]` sign trajectory, structurally, not seed luck. The tiny noise keeps rows distinct without fattening the keep span (the G0 margin lives in the thin span).
- **Why the edit must target layer 2, not the gate (structural, G0-checked):** the gate's fault input differs from seen inputs only by the φ excess (in-span — `e_φ` is loaded by every seen row) plus a tiny contraction tail, so a gate-weight solve comes out near zero (the round-1 kill, reproduced honestly); the fault the solve CAN express is the corrupted layer-2 input. R2's localization mirrors this in the edit magnitudes (`‖dW_layer1‖ ≈ 0 << ‖dW_layer2‖`).
- **F8 disclosure:** this fixture is contractive by construction — it has NO productive limit cycle; the F8 census instrument ships and reports the healthy ladder's monotone ‖dY‖ decay (the repair must not flatten it), and the dedicated limit-cycle variant stays out of scope.
- **Seeding:** one master seed (1248); SplitMix64; orthonormal `e₁, e₂, e₃, Q2` via seeded Gram–Schmidt on normal draws; f64 census/solve math, f32 state (Issue-940 hygiene — local instrument, sequential folds).

**Fault-signature self-check (arm-1 baseline, standing; MEASURED 2026-11-27, all green):** healthy transitions `1→2, 2→3`: pass-cos ≥ 0.95 (measured ≈ 1.0 — deeply converged); fault transition `d→d+1`: pass-cos ≤ 0.45; gate means: `g ∈ [0.5, 0.75]` for passes `1..d`, `g_{d+1} ≤ 0.15`; shift-SVD: `|proj(s_fault, v₁)| ≥ 10×` median healthy projection (measured ≈ 400×: `[−0.0157, −0.0036, +6.275, +0.698, +0.0197]` — the flip AND the post-fault parking in the fault basin, both source-shaped) AND `sign(proj_healthy) = −sign(proj_fault)`; layer-1 pass-cosines ≥ 0.95 at ALL transitions (localization); ‖dY‖ strictly decreasing across healthy passes. **G0 measured: overlap mean 0.678 / min 0.636** (per-row rows printed beside the census), joint-Gram min-nz-eig 0.62 — the honest middle regime, both rigged poles excluded.

## Proof path (pre-registered)

Arms on ONE frozen read:

1. baseline — `loop_depth_ood` at depth d+1, no intervention (fault signature must reproduce: gate collapse + sign flip at the constructed layer);
2. damping incumbent — `lt2_deep_stability` (the shipped control, Bench 699 lineage);
3. R4 governor — project-out at the diagnosed site;
4. FP-Solve edit — closed-form, one-time, then depth d+1/+2/+3 with no runtime knob (4b: merged vs unmerged posture).

Bars: (a) stabilization parity arm 4 vs arms 2/3 on state norm + readout consistency; (b) **ALL transitions** repaired (F3), measured at d+1 **and extrapolated d+2/d+3** (the F10 generalization claim becomes a BAR: fit at d+1, assert d+2/d+3 hold); (c) depth-d outputs preserved within pre-registered per-suite floors (F2: soft keep, mixed signs allowed, never identity); (d) ‖dY‖ decay census (F8: a productive limit cycle is DETECTED and left alone — the contraction must not annihilate productive depth); (e) G0 span census passes; G4 alloc-free solve+apply; Issue-940 determinism (pinned f32 folds, golden digests, capture on the serving posture — F6/F7); (f) R2 localizes the fault to the constructed layer only; R3's alarm fires at d+1 and never at 1..d, and graceful exit ≥ refusing the depth.

**Winner rule:** the whole ladder gates on the same frozen read — if the governor (one vector) matches the edit (many matrices), the governor wins and the edit stays opt-in; if the edit wins it must ALSO beat the damping incumbent, else the incumbent stays. Demote the loser in the same landing.

## Tasks

- [x] **T1** — `loop_depth_ood` fixture (constructed pass-indexed gate with the PINNED continuous pass-indicator encoding — scalar ramp t/d, sinusoid fallback — written here BEFORE code; n_layer ≥ 2, contractive to depth d) + G0 span census (overlap number REPORTED with every result) + fault-signature self-check (arm-1 baseline reproduces the Inkling shape) — LANDED 2026-11-27 (`tests/common/loop_depth_ood.rs` + `tests/loop_depth_ood_g0.rs`; all bars green, measured numbers in the design section above; geometry revised at implementation — revision note there)
- [x] **T2** — R2 fault-signature instrument (offline rank-1 SVD of the pass-shift + per-pass projection census; readout-screen class, never a serving path) — LANDED 2026-11-27 (`r2_shift_census` + `r2_fault_localizer` in the fixture module, typed `R2Report` with the `first_fault` site verdict; measured: site = (layer 1, transition d→d+1), the fault-free layer's own verdict = the downstream contamination wave one pass later — F3's containment story made visible)
- [x] **T3** — R3 gate-saturation alarm + graceful early exit (never worse than refusing the depth) — LANDED 2026-11-27 (`run_with_gate_alarm` + `GATE_FLOOR` 0.35, two-phase per pass: batch-mean telemetry read BEFORE commit; measured: fires exactly at pass d+1 (telemetry 0.66/0.68/0.70 → 0.00045), never at 1..d; the kept passes are the unarmed ladder's outputs bit-identically)
- [x] **T4** — R4 spectral governor (`h' = h − γ·(v₁ᵀh)v₁`, one frozen vector per fault site, γ from gate telemetry) — LANDED 2026-11-27 (`governor_run` + `governor_gamma`, γ = clamp(1 − g_prev/GATE_FLOOR, 0, 1) — the lagged-telemetry form: pass d+1 reads pass d's healthy mean and stays γ = 0; measured: γ = [0,0,0,0, 0.9987, 1.0], passes 1..=d+1 byte-identical to the unarmed ladder, and the post-fault outputs hold cos = 1.0000 vs the healthy depth-d reference at ALL extrapolated depths d+1/d+2/d+3 with stable norms (6.836) — the mechanism: the projection deletes the collapsed e₁ content (fault AND the healthy e₁-share alike — the governor cannot distinguish them) and the write-derived v₁⊥ residue re-normalizes onto the healthy direction. T6's adjudication consequence: the governor reaches stabilization parity with ONE vector and zero weight edits — the edit must beat the GOVERNOR, not just the baseline, to win the ladder)
- [x] **T5** — the edit arm (protected-ridge solve → rank-r `QuantErrorLora` overlay pair; capture on the serving posture; Issue-940 folds; envelope digest binds base + capture posture) — LANDED 2026-11-27 (`capture_edit` + `fp_solve_edit` + `layer2_overlay` in the fixture module, `EditConfig` pinning λ_keep 1.0 / ridge_frac 0.06 / rank_r 8; f64 solve, f32 apply rows — the serving posture IS the fixture's f32 forward; BLAKE3 envelope binds capture + config + the artifact bytes). MEASURED: rank-r truncation drops ~0 energy (the source's near-lossless-here, reproduced); REPAIR 0.403 → 0.9988 at d+1; **bar (b) PASSED — the one overlay fit at d+1 holds at d+2 (0.9989) and d+3 (0.9989)** (F10 converted to a bar and SURVIVED); keep drift ≤ 2e-6 cosine (soft, not identity); Issue-940 determinism golden green; the empty-overlay demote byte-identical, config-nudge moves the envelope digest. Landing fought two transcription bugs, both caught by the fit census: the solve-side transpose (dWᵀ stored where the apply needs Δ) and the thin-SVD U-row extraction (Δᵀ·v where the factorization needs Δ·v — the rank-r response read 1.9e6 against a 6.27 target before the fix; the FULL-Δ fit census reading cos = 1.0000 is what localized the defect to the truncation half)
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
