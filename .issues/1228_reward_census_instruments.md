# Issue 1228 — Reward-census instruments: closed-form TD/overlap/sparsity/readout kit (katgpt-core)

**Status:** OPEN — unclaimed. Opt-in instrumentation joining the existing `diagnostics` family (I4's readout lands in katgpt-pruners); pure-math fixtures are the proof path (no model, no capture lane required to land the primitives).
**Source:** riir-reflex `.research/018_Sparse_Reward_Subsystem_Probes.md` (arXiv:2602.00986, sparse reward subsystem) — the paper's measurement layer made generic. Companion consumers: katgpt-core `mb_value.rs` (Issue 767 — the RPE-learning sibling that consumes these signals), reflex esc-capture significance arms (I2).
**Date filed:** 2026-11-17

## Goal

Four gradient-free instruments that measure/localize reward-relevant signal in labeled (state, outcome) captures. None claims novel math — each is textbook statistics composed into a census kit with typed verdicts and loud refusals, and TWO OF THE FOUR reuse shipped code (I2 the shared log-gamma kernel, I4 the shipped mean-difference constructor). Determinism: blake3-reproducible, fixed tie rules (first-index), G4 alloc-free on the hot paths.

## Instruments

- [ ] **I1 — `td_delta` estimator + MC significance gate.** API carries the GENERAL Bellman form δt = r_t + γ·V̂(t) − V̂(t−1) (per-step reward first-class); the paper's sparse-terminal-reward shape (r_t = 0 for t < T, terminal δ = r − V̂(T)) is the DEFAULT instance, so a consumer with per-step rewards cannot get a silently wrong δ. Significance gate `|δ| > z_α·σ̂/√K` with the K-budget inverse `K = ⌈(z_α·σ̂/δ_min)²⌉`. Fixture: hand-computed trajectory with a known V table → deltas bit-exact (both the terminal-reward instance and one per-step-reward case); synthetic Gaussian-δ coverage at nominal α within band. Falsifiers: shuffled state order must shift deltas per the identity (wrong order ≠ same output); K=1 with mis-set σ̂ measurably inflates FP (the gate must catch it).
- [ ] **I2 — hypergeometric set-overlap null.** `P(overlap ≥ x | two random k-of-N sets)` + `E[IoU]` — the paper's Appendix C baseline, made a p-value. **Reuse the shared Lanczos log-gamma kernel** (`crate::special_fn::ln_gamma`, Plan 597 T1.1 — the kernel `best_belief.rs`'s ln B(a,b) and `bmr.rs` already consume); NO new log-space combinatorics. ⚠ Implementer note (from the verdict review): `special_fn` is `pub(crate)` and the comment at `katgpt-core/src/lib.rs:1715` says it is limited to its consumers — but the reviewer found no `#[cfg]` gating the module, so comment and code disagree; check what actually controls the module when this lands (if gated, add the `diagnostics` member to the gate) and run `cargo clippy --no-default-features` so any dead-code warning surfaces at implementation time, not after merge. Fixture: small-N exhaustive enumeration cross-check, max abs err < 1e-12. Falsifier: N=0 / k>N typed refusals; asymptotic-vs-exact sanity arm.
- [ ] **I3 — L1 pruning-curve knee census.** Keep-fraction sweep zeroing coordinates by |w| on ANY weight vector, metric-vs-keep-fraction curve + knee detection with a TYPED `NoKnee` verdict (never invents a knee). **First read the existing elbow logic** in `katgpt-core/src/log_frontier.rs` and `subspace_phase_gate.rs` — reuse/extend whichever detector fits; a THIRD parallel knee implementation is a DRY violation, not a kit member. Fixture: planted k-sparse signal → knee recovers k. Falsifier: dense ground truth → `NoKnee` (the instrument must report it, loud).
- [ ] **I4 — difference-of-means direction readout — CONSUME, don't re-derive.** `construct_probe_via_mean_difference` ALREADY SHIPS at `katgpt-pruners/src/fpcg_modelless.rs:101` (planted-direction recovery + empty-input refusal, tested). The delta lands BESIDE it in katgpt-pruners (not katgpt-core — the constructor lives there): a sigmoid scoring readout σ((h − μ̄)·d) over the constructed direction + a typed no-signal refusal (μ₁ = μ₀ → refuse, promotion REFUSES loud, never a silent AUC ≈ 0.5 pass). Fixture: planted direction → AUC ≥ the analytic linear-Bayes bound on the fixture. Falsifier: the no-signal typed refusal arm.

## Fold-in candidate (only if a consumer appears)

- PAVA/isotonic recalibration (exact monotone L2 fit, closed form) as a `SigmoidGateCalibrator` sibling — recorded, NOT in scope until a consumer asks for calibrated probabilities; any such landing owes the conformal-naive floor.

## Consumers named up front (so this is not speculative substrate)

- `mb_value` (Issue 767): I1 is the RPE signal-source sibling — today the modulator arrives caller-side.
- reflex ESC δ-target cell (reflex Issue 122): I2 is the significance arm for any set-overlap claim.
- Any future capture lane (hidden-state taps, outcome captures): I3/I4 are the read-only census posture (the `sink_audit`/`weight_audit` precedent — offline, report-only, zero serving-path cost).

## Constraints

- **Feature placement: join the existing `diagnostics` family** (katgpt-core Cargo.toml carries it — 24 members, all measurement tools; these instruments are exactly that shape). Do NOT add a new top-level `reward_census` flag: feature_cap_gate has katgpt-core near the registry cap (inside the warning band) and the growth ratchet in `feature_count_pins.txt` would flag a new top-level flag as GROWTH. I4's readout lands in katgpt-pruners beside the shipped constructor (see I4).
- Default builds byte-identical (the env-switch law applies if an env is ever added).
- No sync-boundary semantics cross unless raw scalars; blake3 pins on any frozen census output.
- Bench: a census example running all four fixtures + falsifiers, exit non-zero on any tampered arm; GOAT gate = the fixtures themselves (pure math) — promote to default only when a real consumer wires it and shows the gain, per the feature-flag discipline.
