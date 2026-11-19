# Issue 1236 — Issue 978's round-2 "Thm-3 bound HELD in all 4 estimable arms" was measured on the laya head residual; on the true encoder output banking77 k=256 is VIOLATED

**Status:** IN PROGRESS — filed 2026-11-20 (katgpt-rs-93@amo) from riir-reflex Bench 201.
**Claimed:** katgpt-rs-93@amo, 1795114000, expires 2026-11-22T00:00:00Z

## The finding

riir-infer Issue 1239 (fix `954b27be`) found that `EncodedQuestion.hidden` was the laya head's post-type-emb, post-two-layer residual, not the encoder output. riir-reflex's `rho_suite_probe` (the Issue 978 T5 consumer lane) mean-pools that field. Re-run on the same box, before and after the fix (riir-reflex Bench 201, job `6cdf8e7c`):

- banking77 k=256: held-out post **468.40 → 449.14** against a floor of 467.41. The bound is now **VIOLATED** (3.9% below the floor, outside the pinned 2% estimator band; null/2k = 0.997, so the estimate is VALID).
- The other three estimable arms still HOLD. The lane tally (6 permitted, FAIL, no saving) and the T4 promotion refusal are unchanged.

The round-2 revision (`f8df1746d`) retracted run-1's "floor violated 46–74%" as the in-sample artifact and recorded "HELD in all 4 estimable arms". That second half no longer holds on the corrected input.

## Tasks

- [ ] **T1 — re-adjudicate (Claude verdict):** is the single held-out violation (a) the Thm-3 independence premise failing for same-input cross-embedder pairs (the structural reading `50152705a` first proposed), (b) estimator noise at k = 256, n_fit = 600, or (c) an instrument defect? A seeded-split robustness read (several split seeds) decides between (a) and (b). Record the verdict in HISTORY.md's Issue 978 row. The primitive stays opt-in either way; nothing here promotes it.

## Pre-registration (2026-11-20, before any seeded number)

- **Instrument:** riir-reflex `rho_suite_probe`, plus an opt-in `RHO_SPLIT_SEEDS` env. Unset, the even/odd record stays byte-identical. Set, each seed s draws a blake3(PERM_SEED ⊕ s)-seeded random half split (n_fit = ⌊n/2⌋). Fit, held-out score and null all use that split, with the same pinned bars (2% band, null/2k ≥ 0.95).
- **Seeds:** 1..=16 at riir-infer ≥ `954b27be` (the true encoder output). The adjudicated arm is banking77 k=256; the other arms are reported too.
- **Verdict rule** over the VALID seeds of banking77 k=256:
  - VIOLATED on ≥ 75% → (a) **structural** (the Thm-3 premise fails for this pair);
  - VIOLATED on ≤ 25% → (b) **estimator noise** (978's "bound HELD" stands with a split-sensitivity caveat);
  - otherwise **undecided** (report the rate).
- (c) instrument defect is checked first: the seeded path at the even/odd split must reproduce the recorded 449.14.
