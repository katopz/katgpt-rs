# Issue 958: `sst_certified` — tail-subspace substrate for weight-delta anatomy and factored-tail shipping

**Status:** Open — claimed

**Claimed:** zed-m3-sst958, epoch 1792784038, expires 2026-10-25T19:33:58+00:00

> **Source:** [katgpt-rs/.research/631_Subspace_Tuning_SST_Composition_Law.md](../.research/631_Subspace_Tuning_SST_Composition_Law.md) · arXiv:2610.22293 (Subspace Tuning, Oct 2026)
> **Kind:** poc + proof (modelless substrate; consumer sides live in riir-neuron-db Issue 629, riir-ai Issue 1116; the producer is riir-train Plan 470)
> **Feature:** `sst_project` (opt-in, katgpt-core), substrate only — zero runtime cost unless invoked
> **Boundary:** THIS issue is katgpt-rs scope ONLY (substrate + theorem). The freeze-envelope sidecar is riir-neuron-db Issue 629; the merge-admission gate and hot-swap wiring are riir-ai Issue 1116.

## Problem

The stack composes weight mutations at runtime — freeze/thaw deltas, `LoraPair { reader, writer }` overlays, TIES-style merges, consolidation deltas, quantized (transcoded) deltas — with no spectral accounting of WHERE a delta writes relative to the base weights' capability substrate. arXiv:2610.22293 supplies the algebra and two load-bearing facts:

1. **~98% of an unconstrained delta's energy lands in the top-99% protected span** (their Fig. 4 controlled study). Existing stack deltas were produced WITHOUT tail confinement — they would fail a tail-compliance check by construction. So the modelless tool for EXISTING deltas is an **anatomy report** (protected/free energy split — interference predictor for merges, drift quantifier for freeze deltas), never a compliance gate.
2. **The composition-order law:** any elementwise transform between accumulation and projection leaks. Training face: Adam's diagonal D (69% leak for gradient-space projection). Shipping face: quantizer SCALES have the same diagonal shape, but quantizer ROUNDING is isotropic noise — a fraction `1−(1−r)² ≈ 99.99%` of it lands in the protected block at r = 0.99, at magnitude comparable to the delta. **Dense-project-then-quantize is therefore unsalvageable at near-zero tolerance.** The only certifiable shipping form for a quantized tail delta is **factored-tail**: `ΔW = U_⊥ C V_⊥ᵀ` with C quantized — the tail span is closed under the product, so confinement is preserved EXACTLY, certified by construction; the residual certifier is a RELATIVE check vs the expected quantization-noise floor.

Internal prior art (Research 406 `spectral_rewire` = keep the TOP ~1% block; Research 231 `off_principal_project` = vector projection off the top-10%) covers different cuts of the same spectrum as purification/retrieval operators — neither carries the anatomy/composition/factored-tail residue. This issue files that residue at substrate level.

## Scope (katgpt-rs only)

1. **Lane 1 — per-side bases + projector + anatomy** (`katgpt-core/src/sst_certified.rs`, feature `sst_project`): `fit_anchor` (Gram eigendecomposition on the smaller side; `thin_svd_into`/`jacobi_eigen`/`newton_schulz` consumed, never forked; `Σ_k⁻¹` as precomputed reciprocal + zero-guard per `algebraic_op_ban_gate`). **Per-side representation choice:** on the NARROW side store the complement `U_⊥` (~1%·min cols — truly skinny); on the WIDE side store the protected `V_k` (k cols — narrower than the `d_in−k` complement, e.g. 4055 vs 6953 at 4096×11008); apply `Q = U_⊥U_⊥ᵀ` or `Q = I − V_kV_kᵀ` per side; never materialize m×m. ⚠ **Anchor cost honesty:** at MLP shapes the wide-side sidecar is roughly BASE-MATRIX-SIZED (4055×11008 ≈ 44.6M floats vs the ~45M base) — anchors are NOT cheap for wide MLP matrices; document the per-matrix cost in the anatomy report and consider deriving `V_k` from `W_ref` + the narrow-side bases on demand where the sidecar budget refuses. `protected_energy` via the stored bases (`E_prot = ‖P_UΔ‖² + ‖ΔP_V‖² − ‖P_UΔP_V‖²`), ordered folds at every reduction crossing a commitment boundary (`dot_f32_ordered` precedent, Issue 940 determinism law). **Width guard:** `k = max(0, min − f_min)`; a caller at `k == 0` is an SST non-case and is REFUSED (nothing protected ⇒ plain-SFT behavior wearing the flag).
2. **Lane 2 — scoped Lean theorem** (`.proofs/KatgptProof`): for rank-1 elementwise `A⊙X = diag(a)·X·diag(b)`, zero protected-block leak holds iff each protected span is invariant under its diagonal — span(U_k) [resp. span(V_k)] splits along the LEVEL SETS of a [resp. b] — with the added clause that the opposite diagonal does not vanish on the complement (a vanishing diagonal can mask a violation); scalar diagonals the trivial case. Ship exactly two arms: (i) level-set-aligned / scalar ⇒ no leak (axiom-free, concrete instances, Plan-016 spec-self-test convention); (ii) an explicit 2×2 diagonal counterexample that leaks (the perturbation arm's negative, `scripts/proof_negative_test.sh`). ⚠ The general iff over arbitrary (non-rank-1) A is OUT OF SCOPE — ill-posed as a single statement and beyond this issue.
3. **Lane 3 — factored-tail form + relative-tolerance certifier:** `factor_tail(a, delta) -> TailFactor { u_perp, c, v_perp_t }` and `certify_decoded_tail(a, decoded, rel_tol)` where `rel_tol` is calibrated against the expected quantization-noise floor (never a near-zero absolute bar). Planted-leak negative: perturb a decoded tail delta by a known-magnitude protected component; the certifier must refuse with the leak measured to tolerance. The isotropic-rounding arm: a dense-projected-then-quantized delta must FAIL the check (proving the check is not vacuous) while the factored arm PASSES at the same bit budget.
4. **Lane 4 — rotation-invariant anchor commitment:** commit `BLAKE3` over the PROJECTOR (`U_⊥U_⊥ᵀ` / `V_kV_kᵀ`, ordered folds, pinned fp tolerance) — never raw basis bytes. At r=0.99 the cut sits inside the Marchenko–Pastur bulk where near-degenerate gaps make the tail basis defined only up to rotation; raw U/V are solver-dependent and not portable across devices.
5. **Lane 5 — chain composition (tail deltas):** chained factored-tail overlays stay tail-confined under addition; the composite is certified if every link is; free-block drift budget additive (triangle inequality). Certification ledger rides the consumer's lineage (riir-neuron-db Issue 629 owns the envelope side).
6. **Lane 6 — re-anchor churn policy (substrate API):** re-anchor invalidates outstanding certifications; stale certs REFUSE loud (non-parent anchor). Cost model inherits the paper's K-schedule shape (R100 +5.1% / R10 +37.6% as the curve family; our absolutes measured).

## Consumer dependency (explicit)

Certification lanes (3/5/6) have NO existing-delta consumers: every delta the stack ships today was produced without tail confinement (~98% protected-block energy). Their value is gated on riir-train Plan 470 producing tail-confined deltas, or on hand-constructed factored-tail overlays (a deterministically-constructed LoraPair: reader = U_⊥, writer = V_⊥ᵀ, payload = quantized C). Lane 1's anatomy report is the consumer-independent modelless value.

## GOAT gate sketch

- **G1** `r=0 ⇒ Π = I` byte-identical (disarmed posture); anatomy agrees with a brute-force two-sided-projection oracle on random deltas; planted-leak refusals; the isotropic-rounding arm FAILS while the factored arm PASSES at equal bit budget; the 2×2 theorem counterexample REDs the compliant claim.
- **G2** **ms-class per matrix at 4k-class shapes** (measured, `--release`, bench_preflight PROVENANCE line — one application moves ~0.7–5 GFLOP at r=0.99; the round-1 µs target was wrong at this rank).
- **G3** `--all-features` + `--no-default-features` clean.
- **G4** zero alloc per call over 1000 cycles.
- Anchor recompute determinism: projector digest stable across recompute on the same device (pinned tolerance); cross-device portability documented as digest-level only.

## Caveats

- NOT for 64-dim vector shards — Research 406's G1b ledger measured weak top-1% concentration at NPC style_weights scale (a different claim than tail confinement, but the small-width hazard stands); lanes target matrix deltas at LLM/adapter widths.
- Small widths need a free-rank floor: at min-dim 64, r=0.99 leaves a rank-≤1 tail (capacity confound). `fit_anchor` takes `min_free_rank` (default `max(8, ⌈0.02·min⌉)`); Plan 470's B1 pilot is the first consumer.
- `algebraic_op_ban_gate` compliance check required for the `Σ_k⁻¹` reciprocal in `fit_anchor` (precomputed reciprocal + zero-guard).

## Consumers

- riir-train Plan 470 — the PRODUCER (tail-confined deltas certified by construction; unblocks lanes 3/5/6)
- riir-neuron-db Issue 629 — freeze anchor sidecar + stale-cert refusal (lanes 4/6 envelope side)
- riir-ai Issue 1116 — merge-admission gate + hot-swap wiring (anatomy + certification consumers)
- Quantized-delta shipping lanes — lane 3 certifier API (hook locations are consumer-side)

**Claimed:** (append `**Claimed:** session, epoch, expires <ISO-UTC>` here before implementing; 48 h expiry, advisory)
