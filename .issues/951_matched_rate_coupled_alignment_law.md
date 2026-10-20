# Issue 951: Matched-Rate Coupled Alignment Law as a Runtime Store-Pair Invariant

**Status:** Open — fusion idea, novelty TBD. Not a primitive filing: the mechanism is published prior art (KP 1994 / Akrout 2019) and the closest in-workspace dual-stream latent extraction settled negative (katgpt-rs Research 448 / Plan 456). Builds only with a concrete consumer + a task-usefulness gate.

> **Source:** arXiv:2412.20018 — "How Biological Synapses Self-Assemble Gradient Learning" (SAM, PNAS 2026), Lemma 1. Distill home: `riir-train/.research/478_SAM_Self_Assembling_Motif_Gradient_Learning.md`.

## Pinned claim (one sentence)

Matched-rate coupled alignment — two stores updated with the SAME outer-product drive and matched rates under a shared decay γ satisfy `ΔE = −γE` (exponential decay of the transpose-misalignment `E = V − V̄ᵀ`, unconditional in the data under EXACT LINEAR updates) — shipped as a **runtime maintenance invariant for latent store pairs**, distinguished from Kolen-Pollack / weight-mirroring prior art (which maintain forward/backward WEIGHT pairs during gradient training) by operating on runtime latent store pairs that our stack already couples: `LoraPair { reader, writer }` overlays, paired EMA estimators (`elo.rs` strength estimators, `self_evolve.rs` direction-vector stores), and MAPE-K monitored shard pairs.

## What Lemma 1 gives (from the paper, verified against the v2 SI proof)

- `ΔV = η·ā(a)ᵀ − γV` and `Δ(V̄)ᵀ = η·ā(a)ᵀ − γ·V̄ᵀ` share the drive term ⇒ the difference
  obeys `ΔE = −γE` exactly when η matches, where **`E = V − V̄ᵀ` (the TRANSPOSE — no
  inverse is involved, and none is needed: non-square V̄ at 5–8× downstream width has no
  inverse and the law does not want one)**; `E(t) = (1−γ)ᵗ·E(0)`.
- Shipable three ways, all no-GD: (a) **certificate** — k steps at γ guarantees
  ‖E‖ ≤ (1−γ)ᵏ‖E₀‖, committed in the `MerkleFrozenEnvelope` provenance and verified
  fail-closed at thaw; (b) **monitor** — a `mape_k.rs` vital sign (observed residual vs
  the γ-prediction; violation = rate mismatch or corruption); (c) **reconciler** — any two
  latent estimators of one signal under matched blend rates converge geometrically for free.
- **Linearity precondition (part of the pinned claim, not a footnote):** the law holds for
  exact matched LINEAR updates only. Clamping, quantization, sigmoid gating, sparse/
  async writes on EITHER store break the exact cancellation — and our LoraPair, EMA, and
  MAPE-K stores do several of these. Any build must either (i) apply the updates in exact
  shared arithmetic, or (ii) reclassify the guarantee as approximate with a measured
  residual band — (ii) forfeits the "certificate" framing and the G1 exact-decay gate.
- Sizing companion: the feedback-facing side wants 5–8× the capacity (paper's downstream
  overparametrization law) — a one-line construction policy.

## Why issue, not primitive (the two kills, stated)

1. **Q1 (no prior art) fails at the mechanism level:** KP (1994) and Akrout et al.
   (NeurIPS 2019) publish matched-update + shared-decay transpose alignment — in the
   weight-training context. The runtime-invariant *application* is unclaimed, but an
   application of a published mechanism is GOAT-at-best territory, never Super-GOAT.
2. **The 448 prior:** the closest dual-stream latent extraction (Research 448 / Plan 456,
   latent error diffusion) was modelless-correct (G1–G6 PASS) but **task-useless**
   (G7 −0.51 pp vs Frozen). An invariant that merely certifies alignment, without a
   consumer whose task improves, is exactly that failure shape.

## Consumers-if-built (from the Research 478 advocate pass; each must show TASK gain, not just invariant satisfaction)

- `LoraPair { reader, writer }` (riir-engine): alignment-certificate overlay — construct
  from a transpose-mirror seed, k matched-rate refinement steps at consolidation time,
  commit {‖E‖, γ, k} in the freeze envelope; every hot-swap carries its own certificate.
- `mape_k.rs` (riir-neuron-db): the alignment vital sign + thaw-time re-alignment trigger.
- `self_evolve.rs` / `elo.rs` (riir-refine): paired EMA estimators of one signal aligned
  by law instead of by tuning.
- `vibe.rs` KG candidates via a decayed co-activation (eligibility-trace) ring — the
  σ(−λΔt) decay is already the mandated workspace law; RawWakeCycle (lossless, no decay)
  is the partial analog.

## Gate-if-built (mandatory shape)

- G1: measured residual = (1−γ)ᵏ·‖E₀‖ exactly on a fixture **under exact linear updates**;
  mismatched-rate posture **refused** (fail-closed — shipping the primitive without the
  matched-rate constraint ships a lie); a non-linear store posture (clamped/quantized/
  gated/async) **refused or explicitly reclassified approximate** with the certificate
  framing withdrawn.
- G3: invariant-off = bit-identical boot.
- G4: fixed-size scratch, zero steady-state alloc; any freeze-time eigendecomposition is
  freeze-time-only, never hot-path.
- **G7 (the 448 lesson, non-negotiable):** a defend-wrong PoC (riir-poc, three competitors:
  invariant-carrying pair vs constructed-once pair vs unaligned pair) must show a task-axis
  gain on a real consumer — an invariant without task gain does not ship (Research 448 §7).

## Provenance

- Paper: arXiv:2412.20018 (Lemma 1 + SI Lemma 1 proof; v2, 2026-10-12).
- Prior art: Kolen & Pollack 1994; Akrout et al., "Deep Learning without Weight Transport", NeurIPS 2019.
- Workspace lineage: katgpt-rs Research 448 (negative prior), riir-train Research 404 (family), riir-train Research 478 (distill home + Plan 466 training lane).
