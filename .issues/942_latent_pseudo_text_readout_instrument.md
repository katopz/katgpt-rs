# Issue 942: Latent Pseudo-Text Readout + Health Sensors (Report-Only Instrument)

**Status:** CLOSED 2026-10-16 — landed `76d6d4fef` (module + example + gates, report-only; claim lease `c88264c85` honoured)
**Claimed:** shikuwa-4090-agent, 1792100883, expires 2026-10-17T21:48:03Z
**Date:** 2026-10-15
**Source:** katgpt-rs/.research/622 (HybridAgent distill — interpretable latent mode; the readout technique family is established prior art: logit-lens/coda-lens arXiv:2507.02199 et al.; the delta here is only that ours doesn't ship)

## Problem

The latent lanes (Plan 275 soft embedding, Plan 172 RiM slots) have no inspection surface: when a span degrades or loops, there is no way to see what the model was "thinking". HybridAgent renders every latent step as pseudo-text (cosine nearest-token against a normalized embedding bank) with a per-span readability score.

## Deliverables

- [x] Readout fn: pre-normalized embedding bank built once per table; `readout(v) = argmax_j cos(v/‖v‖, ê_j)` + top-k; span readability = mean max-cosine.
- [x] Health sensors for `collapse_aware_thinking` consumption: loop sensor (`cos(v_t, v_{t−1}) > θ_c` for k consecutive steps), collapse sensor (nearest-token constant over window w with score > θ).
- [x] Report-only debug lane + trace surface (glyph-led markdown, Copy-as-Markdown): exit 0 always, WITH a blindness floor (min steps inspected) so an instrument that inspects nothing cannot print green.
- [x] Gates: G1 sanity — for a peaked distribution the readout equals argmax with ≈1.0 agreement; bank unit-norm verified at build. G2 — armed cost = one V·D matvec per INSPECTED step; disarmed hot path unchanged. G4 — bank precomputed, zero per-call alloc.
- [ ] Optional later: low readability as an abstention feature for decision-serving — UQ-flavored, must beat the conformal-naive floor before anyone says "calibrated" (out of scope here).
