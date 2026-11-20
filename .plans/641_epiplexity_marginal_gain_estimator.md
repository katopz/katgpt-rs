# Plan 641: Epiplexity Marginal-Gain Estimator (katgpt-core primitive)

**Status:** OPEN — filed from katgpt-rs Research 685 (arXiv:2608.11746); novelty 4/4 at mechanism level, value bar UNPROVEN, all gates pre-registered; demote-in-landing on any G2 FAIL (Research 591 precedent).
**Paper:** EpiSelect (Su et al., NeurIPS 2026) — online epiplexity gain from cross-domain scaling-law fits.
**Substrate:** extends `katgpt-pruners::epiplexity` (Plan 130, Bench 041 — post-hoc area only; the default-on `epiplexity_bandit` consumes binned S_T/H_T as bandit CONTEXT, no fits) with the ONLINE allocation half, homed in **katgpt-core** (the leaf refine/reflex may consume; pruners re-exports/wraps — DRY direction pinned).
**Feature gate:** the new katgpt-core module ships as a member of katgpt-core's existing **`estimators` umbrella** (`crates/katgpt-core/Cargo.toml` — the `estimators = [...]` row) **with NO key of its own** — the exact Plan 635 Tranche 2 Q4 landing shape, precedent: the `lepski_cascade` module rides the same umbrella keyless (the umbrella's comment records katgpt-core at 296/300 of the crates.io cap; feature_count ratchet: a new primitive lands under an existing family umbrella, never as a new top-level [features] key). Fallback if the umbrella is refused at implementation review: a reasoned grant row in `scripts/feature_count_pins.txt` in the same commit — `grant katgpt-core epiplexity_gain  # cross-repo consumer primitive (refine/reflex consume katgpt-core only; generic fit math cannot home in katgpt-pruners without a boundary expansion)`. katgpt-pruners and the root crate then forward their existing `epiplexity_scoring` chain DOWN to the umbrella-gated core module (dependencies stay pointing downward; core sits upstream of both).

## Why katgpt-core

riir-refine and riir-reflex consume katgpt-core only (verified in their manifests). The fit + closed-form derivative is generic math (no pruner semantics) — core is the correct layer; `katgpt-pruners::epiplexity` gains a thin wrapper for its own screening consumers. This gives the epiplexity family its **first consumer outside katgpt-rs** (the family's in-stack consumers today: the post-hoc screening pruner + the default-on `epiplexity_bandit` bandit-context heuristic — both in-stack only; verdict round 1 caught that the original "dormant, zero consumers" claim was wrong and was narrowed accordingly).

## Phase 1 — primitive

- [ ] `katgpt-core/src/epiplexity_gain.rs`: `CrossDomainCurveFit` — per-domain params (α_m, log β_m, log ε_m, γ_m,k on the simplex), GD-free fit: grid over (α, ε) + coordinate descent on γ (log-space Huber objective, median-of-residuals robustness), bounded iterations.
- [ ] `marginal_gain_k(&self) -> [f32; K]` — the closed form `Σ_m [n_m·α_m·γ_m,k / (Σ_i γ_m,i·n_i)]·(L̂_m − ε_m)`; fixed-size `[f32; K]` (bounded-domain array law), sequential fold (Issue 940).
- [ ] `sigmoid_weights(gains, τ_s) -> [f32; K]` — `σ((g_k − median)/τ_s)` normalized (NEVER softmax; σ saturation replaces the paper's clip); momentum/hysteresis helper.
- [ ] `Saturation` verdict type + `Sparse` typed verdict: under-floor mix variation (effective rank < K−1) refuses the cross-domain fit and falls back to univariate per-domain (α, ε only) — never a silent γ.
- [ ] Re-export through `katgpt-pruners::epiplexity` for the screening consumers.

## Phase 2 — gates (pre-registered bars; `--release` at G2)

- [ ] **G1 recovery:** synthetic curves from the family + noise → recovered (α, ε, γ) within pre-registered tolerance; closed-form derivative vs central finite difference of Ŝ within 1e-3 relative (n ≥ 1000 loop, black_box at result AND arguments AND receiver — the timed_region law). Two-sided canary: perturb one γ coordinate → marginal-gain ranking flips where the arithmetic says (proves the gate live).
- [ ] **G1b identifiability:** single-mix fixture returns typed `Sparse` + univariate params; cross-mix fixture returns full γ.
- [ ] **Saturation tripwire:** planted ε-dominated domain ⇒ its allocation weight demotes to ~0 (loud-zero defence for the gate itself).
- [ ] **G2 allocation value (three-arm minimum + misspecified fixture):** paired A/B on synthetic bandits with planted headroom structure — sigmoid-weighted allocation vs **(a) uniform**, **(b) greedy-exp**, and **(c) a cheap learning-progress arm: windowed finite-difference slope per domain, NO fit** (the Graves/TSCL-class baseline — if the closed-form marginal cannot beat the raw-slope arm with LB95 > 0, the fitting machinery is unjustified and the primitive demotes). Plus a **misspecified-family fixture** (exponential decay; plateau-then-drop) where the bar is fitted-allocator NO WORSE than the LP arm — recovery on the well-specified power-law family is guaranteed by construction and grades nothing. ≥3 seeds, interleaved pairs (the ab_timing law: no sequential-arm ratios). Consumer-surface G2s live in the consumers' own filings (riir-refine Plan 392, riir-reflex Issue 129); per-surface Spearman bars are measured THERE, never on synthetic fixtures.
- [ ] **G3:** feature off → byte-identical (no new default-path code).
- [ ] **G4:** O(K·W) per decision, zero steady-state alloc; refit cost bounded (paper's own ~12 s-class budget is the inherited ceiling; our W is small — assert wall ≤ that budget on the bench box with `bench_preflight` PROVENANCE line).

## Phase 3 — wiring notes (advisory, no code here)

- [ ] Consumers: riir-refine Plan 392 (mining queue), riir-reflex Issue 129 (corpus admission), riir-ai Issue 1247 (curiosity budget) — each consumes this primitive via katgpt-core; none re-implements the fit.
- [ ] goat-audit note: this filing gives the epiplexity family its first consumer OUTSIDE katgpt-rs (in-stack consumers already exist: the post-hoc screening pruner + default-on `epiplexity_bandit` context heuristic — corrected at verdict round 1); boundary direction is DOWN only (riir-* → katgpt-core), never a katgpt-rs dep on any riir-* crate.

## Out of scope (explicit)

REINFORCE/EMA machinery (training-side — riir-train Plan 506); any serving/decide-path routing (stationary-surface ban, Research 685); per-surface thresholds (consumer repos own them privately).
