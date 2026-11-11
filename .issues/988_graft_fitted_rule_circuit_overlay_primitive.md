# Issue 988: `grafts` — fitted rule-circuit overlay primitive (Research 658)

**Status:** OPEN

**Claimed:** amo-i988-grafts@amo, 1794423582, expires 2026-11-13T19:26:22+00:00

**Claimed:** katgpt-rs research session (mac m3), 1794160000, expires 2026-11-10T21:00:00Z

## Summary

Modelless capability-graft primitive distilled from [Circuit grafts (Levi Foster, 2026-11-07)](https://lf.wtf/research/circuit-grafts/) — see [Research 658](../.research/658_Circuit_Grafts_Fitted_Rule_Circuit_Overlays.md). A four-part sidecar overlay that adds a deterministic skill to a served model with no gradient training:

- **`GraftRead`** — closed-form least-squares linear probes over provided activation slices. The solver ALREADY SHIPS — `katgpt-attn-match/src/beta_fitter.rs::fit_beta_nnls` (normal equations `(AᵀA)w = Aᵀm`, Cholesky with diagonal jitter, NNLS refinement) and `value_fitter::cholesky_decompose`; T1 lifts it into the leaf rather than writing a second one (DRY).
- **`GraftTable`** — deterministic rule table computing the answer from probe reads. Authored rows AND a `mined_table()` constructor: minimal-deciding-set search over ~300 solved examples (per output cell, smallest input subset that decides it on every example; reuse found rules upward — the carry pattern).
- **`GraftWrite`** — computed logit bias at the last layer (sampling-distribution seam; PhraseBoost precedent — never residual-stream surgery, FPCG's Constraint 5 untouched and unrelied-upon).
- **`GraftSwitch`** — fitted per-class gates (sigmoid of a linear read; same closed-form family as `SigmoidGateCalibrator`, `katgpt-core/src/sigmoid_calibration.rs`) + optional position-conditioned arming.

Everything is fitted or written, nothing trained. The primitive is model-agnostic: callbacks (`ActivationReader`, `LogitWriter` traits), no model coupling in katgpt-core.

## Tasks

- [ ] **T1** — `katgpt-core` module `graft.rs` behind feature `grafts` (opt-in): `GraftSpec { reads, tables, writes, switches }`. `fit_reads()`/`fit_switches()` consume a SHARED closed-form normal-equation solver: lift the `katgpt-attn-match` Cholesky+jitter solver into the leaf (e.g. `katgpt-core/src/lstsq.rs`) and consume it here; re-point attn-match in the same commit ONLY if mechanical, else keep its local copy and file a dedup issue — never a second solver. `compile_table()` + `mined_table()`: deterministic minimal-deciding-set search — candidate input subsets enumerated by ASCENDING SIZE then lexicographic input index, **maximum subset size k ≤ 3 PINNED**; kept only if it decides every example; cap exhausted without a deciding subset → typed `Undecidable` (loud — never an unbounded search, never a guess). `fit_switches()` — closed-form, same family as `sigmoid_calibration.rs`. `Graft::apply(reader, writer) -> GraftApplied` (zero-alloc hot path; writes only when the switch fires; deterministic float-reduction order per Issue 940).
- [ ] **T2a** — correctness fixture ONLY: a fixed synthetic activation generator pinned by BLAKE3 (or a deterministic tiny forward from the stack) — validates table compilation, switch fitting, wiring, and the alloc census. NEVER an accuracy cell: synthetic activations make digit readability true by construction, so G1 and the scramble arm would pass for free on it.
- [ ] **T2b** — the REAL cell, REQUIRED for any G1/GOAT accuracy claim: real hidden states + real logits from a real frozen model. Host: **riir-infer** (boundary-correct consumer — riir-infer path-deps katgpt-core, so the bench lives there consuming the katgpt-core primitive; it files its own issue there on execution), forward = the GGUF loader over a small dense checkpoint (SmolLM2-135M-class, the paper's own model class; pin the exact file + blake3 at fixture creation), activation slices from a PINNED ≥2-layer mid-layer set (the paper's joined-14/20/21 pattern), solved/dev/sealed splits fixed BEFORE any graft is built.
- [ ] **T3** — in-repo falsification arms (both must be green for any GOAT claim): (a) **scrambled wiring** — same sidecar, shuffled table wiring, must land at/below baseline; (b) **switch-off** — graft present, switches force-off, must be BYTE-IDENTICAL to baseline (the unarmed no-op law, npc_brain_graft precedent). The trained-overlay comparison is NOT a katgpt-rs cell (BOUNDARY.md: no gradient code in this repo) — it is riir-train's OPTIONAL cell (riir-train commits trained fixture bytes; the bench only loads bytes), and the paper's own head-to-heads (gradient grafts 23.7–25.6%, LoRA 39.4% vs circuit 83.5–88.7%) stand as THEIR claims, cited in Research 658 §1.
- [ ] **T4** — GOAT bench `grafts_goat` (`--release` mandatory at G2); accuracy legs run on the T2b cell only: G1 held-out exact-answer accuracy ≥ baseline + a pre-registered bar (set at T2b from baseline measured, BEFORE graft construction); G2 graft hot-path latency ≤ a pre-registered µs/token bar; **G3 no-regression, metric + bar pre-registered NOW — switches disarmed: byte-identical outputs (exact); switches armed on the unrelated split: per-token KL(top-40) vs baseline ≤ 0.02 nats AND top-1 flip rate ≤ 0.5%**; G4 alloc census (zero allocation on the armed hot path). Bench runs under `bench_preflight.sh` provenance.
- [ ] **T5** — digest-pinned sidecar serialization (`GraftArtifact`, BLAKE3 over canonical encoding, versioned format; monotonic apply refuses downgrade — freeze/thaw envelope law) so a graft can mount through the LoraPair hot-swap registry later without re-litigating the envelope.
- [ ] **T6** — docs: module doc carries the mechanism + the modelless doctrine citation (Plan 025 deterministically-constructed overlays; Research 658), and the honest scope note: arithmetic is the VALIDATION domain; the primitive claims structured tasks whose answers are decided by small tables over linearly-readable state.

## Non-goals

- No LM training, no fine-tuning, no gradient code anywhere in this repo (BOUNDARY.md) — the trained-overlay comparison lives in riir-train's optional cell or stays the paper's cited claim.
- No riir-* deps: the primitive is pure katgpt-core math; a GGUF-hosted seam (riir-infer) files its own issue when a consumer exists.
- No serving-family wiring (reflex/instinct/rethink): recorded as future consumer surface in Research 658 §3E; files separately on first real suite.

## Related

- Research 658 (verdict + prior art + cousins); Plan 025 (LoraPair doctrine); Plan 292 (FPCG, Constraint 5); riir-ai Proposal 055 / Plan 629 / Bench 979 (npc_brain_graft — authored predecessor, ProbeSet + exact-LP gate); riir-infer Plan 622 (TokenGate key-bias overlay); riir-train Research 409 + Bench 430 (G-GRAFT FAIL — the trained-router graft precedent); `katgpt-attn-match` `beta_fitter.rs`/`value_fitter.rs` (the existing closed-form solver this lane lifts — DRY, no second solver); `katgpt-core/src/sigmoid_calibration.rs` (SigmoidGateCalibrator — the fitted-gate family).
