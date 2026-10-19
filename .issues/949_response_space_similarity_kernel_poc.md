# Issue 949: POC — Sensitivity-Fingerprint Kernel (`path_kernel_sim`) in katgpt-core

**Status:** Open — POC filed from Research 625 (arXiv:2012.00152 path-kernel lens), 2026-10-19

## Summary

Extracted from [Research 625](../.research/625_Path_Kernel_Exemplar_Superposition_Lens.md): the degenerate path kernel of an *authored, untrained* deterministic scorer is a training-free, positive-semidefinite **response-space similarity** — `K(x,x') = ⟨Ŝ(x), Ŝ(x')⟩` where `Ŝ(x) = ∇_w f_w(x)|_{w₀}` is the scorer's sensitivity fingerprint over its `p ≤ 64` authored knobs (closed-form for sigmoid-gated scorers; central finite differences, pinned `h`, for opaque ones). No analog ships: `similarity.rs` is input-space; `subspace_phase_gate::participation_ratio` runs over latent directions, not knob-sensitivity directions. Class-level prior art (Fisher kernels, TracIn) exists for trained models — the unshipped consumption pattern is zero-training scorers as the kernel source.

⛔ **THE COLLAPSE CAVEAT (verdict round 1, load-bearing).** For any scorer LINEAR in its knobs, `f = σ(w·φ(x))`, the normalized fingerprint kernel is ALGEBRAICALLY the input cosine: `∇_w f = σ'(w·φ(x))·φ(x)` and the scalar `σ'` cancels on normalization. New signal appears ONLY where knobs enter nonlinearly — stacked gates, a knob inside a composition, multiplicative knob coupling, per-stage gains. Everything below is scoped to knob-nonlinear scorers; the linear case is kept as a correctness oracle (T5a), not a target.

Unclaimed — open for any session; no `Claimed:` lease is held (2026-10-19, distill session).

**Claimed:** zed-m3-idle/M3 idle loop, 1792437452, expires 2026-10-22T02:17+07
Implementing T1–T9 (T10 stays deferred) this session; push is the arbiter.

## Tasks

- [ ] T1 `types.rs` — `SensitivityFingerprint` (p-dim, scratch-buffer API), `FingerprintKernel` config (knob set, `h`, normalization, fold rule); **zero-norm guard**: a near-zero-norm fingerprint (e.g. a frequency-knob scorer at its stationary points, where `∇_w sin(wx) = x·cos(wx) = 0`) abstains with a sentinel, NEVER NaN
- [ ] T2 fingerprint computation — closed-form derivative arm for sigmoid-gated scorers (`σ'` computed with the SAME `fast_sigmoid` the scorer itself uses) + central-difference arm (pinned `h`, fixed evaluation order, `dot_f32_ordered` sequential fold)
- [ ] T3 kernel + diagnostic — normalized dot kernel over guarded fingerprints; Spearman ρ between input-space distance and fingerprint similarity (E4's symmetry-class detector)
- [ ] T4 E3 spectrum arm — PR / effective rank over the fingerprint Gram (reuse `subspace_phase_gate::participation_ratio` math over the new object)
- [ ] T5a G1 correctness oracle — the LINEAR-in-knobs identity: for `f = σ(w·φ(x))` the normalized fingerprint cosine equals input-feature cosine exactly (to f32 tolerance). Known answer; proves the kernel code, not the signal
- [ ] T5b G1 sine fixture (CORRECTNESS-ONLY, same class as T5a) — knob pinned to AMPLITUDE (`a·sin(kx)`, fingerprint `= sin(kx)`, cosine 1 at distinct maxima at maximal input distance); ⚠ the amplitude knob is LINEAR, so this fixture proves the fingerprint code, never the signal — do not quote it as support for the new similarity channel (signal evidence lives in T5c/T5d only); the frequency-knob parameterization is the zero-norm case and is covered by T1's guard test; PSD sanity (Gram eigenvalues ≥ −ε·‖G‖) rides T5c's generator
- [ ] T5c G1 planted-class fixture — **independent label generator**: latent class `z ∈ {0,1}` from a seeded RNG, class-conditional `x` independent of the scorer's construction; scorer carries multiplicative knob coupling such that only the nonlinear cross-term structure can recover `z`. Fixture is INVALID if labels are derived from the scorer's own response (passes by construction)
- [ ] T5d G1 baseline battery — fingerprint must beat BOTH (i) input cosine and (ii) the **output-vector baseline**: `r(x) = [f_{w₁}(x), …, f_{w_m}(x)]` over m authored knob settings (gradient-free response similarity). If the output-vector baseline ties or beats the Jacobian fingerprint, the tangent kernel adds nothing → do not promote
- [ ] T6 G2 bench — `--release`, p ≤ 64 knobs / 256 anchors, µs-class per-pair bars, `black_box` args + receiver
- [ ] T7 G3 — every existing cosine/dot consumer byte-identical with the feature off (hash-pinned decision corpus)
- [ ] T8 G4 — zero-alloc hot path, warm scratch buffers
- [ ] T9 GOAT verdict — feature `path_kernel_sim` opt-in; promote ONLY if T5c shows the fingerprint beating BOTH T5d baselines on the independent-label fixture; demote/keep-off otherwise; record verdict in the research note. If T9 output is ever routed into SYNCED belief attribution, the determinism law binds (sequential fold, first-index ties, `freeze_version_hash` coverage) — the POC scope keeps all outputs LOCAL to the computing process
- [-] T10 (defer) E5/E11 fit-first ordering arm — loss-derivative-shaped one-pass candidate priority; only if T5c shows signal

## Consumers recorded (no separate files)

- riir-refine: function-space rule dedup (token-disjoint spans with identical sensitivity profiles) + retrieval re-rank axis
- riir-reflex: routing regime signal; E6 fingerprint-KDE abstention stays UNFILED pending riir-reflex Issue 093 drift evidence
- riir-ai: experience-attribution receipts (`evolve_belief`, `hla` style_weights, `vibe` KG triples) — E2/E12 split
