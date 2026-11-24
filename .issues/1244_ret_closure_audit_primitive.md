# Issue 1244 — `closure_audit`: matched-capacity self-prediction gap as a macrostate gate

**Status:** OPEN — claimed (m3); T0+T1 LANDED `c80cf19e7` (G1 fixtures 14/14 + G4 4/4, clippy clean, default posture 2522/0); T2 bench QUEUED on amo exclusive (job 01a20e44, ridge ≤ 50 ms floor bar + kNN disclosed walls) — bench record + verdict land when the rig returns. POC filed 2026-11-24 from `../katgpt-rs/.research/692_RET_Macrostate_Closure_Audit.md` (arXiv:2605.09294 "Representational Effective Theory"); verdict Gain (modelless track), GOAT-eligible primitive.
**Claimed:** katgpt-rs-m3-zed@m3, 1795533160, expires 2026-11-26T15:12:00Z

## What

A generic modelless measurement primitive in `katgpt-core`: given a recorded trajectory of latent vectors plus an optional microstate (the un-summarized remainder), measure how well `z_{t+1}` is predicted from `z_t` alone vs from `(z_t, x_t)` under a MATCHED closed-form predictor family, and emit a typed verdict on whether the summary is a closed macrostate. The "is my summary sufficient?" question nothing in the workspace can answer today (667's certificates are analytical over designed kernels; 342's geometry is explicitly non-predictive; Plan 324's bisimulation quotients discrete graphs).

## API sketch

```
ClosureAudit::run(trajectories: &[&[Vector]], micro: Option<&[&[Vector]]>) -> ClosureVerdict
  arms: ridge sweep (λ grid as capacity proxy, closed-form normal equations, f64 folds)
      + kNN {1, 8, 64} per side
  split: blocked FORWARD (prefix windows train / suffix windows test) — never iid
  gap = R²(augmented) − R²(summary-only)
  verdict: Closed     iff gap ≤ ε AND R²(summary) ≥ floor
           Lossy      iff gap >  ε AND R²(summary) ≥ floor
           Undecidable iff support < floor OR R²(summary) < floor
```

Feature flag opt-in (e.g. `closure_audit`), report-only consumers first.

- [x] T0: fixtures — injection arm (microstate carries next-z ⇒ gap > bar), noise arm (gap ≈ 0), identity arm (z_{t+1}=A·z_t ⇒ both R²=1, gap=0); degenerate-abstraction plant (one summary dim a pure function of another) must NOT read Closed. Exact numbers pinned in-test. **LANDED `c80cf19e7`:** identity = product of disjoint Givens rotations (a DECAYING A underflows the late windows under the absolute ridge λ — the orthogonal form keeps the gram at O(n) scale); noise = AR(0.9) stationary-variance 1, gap −0.09..+0.03 pinned ±0.03; injection = 0.8z + 0.6x (Lossy, gap ≈ 0.38); degenerate = two INDEPENDENT chains (ρ_u 0.3 / ρ_h 0.95), summary `x = u + 0.5·h` recorded `[x, 0.5·x]`, micro `[u, h]` → Lossy at gap ≈ 0.095. ⚠ TWO earlier plants measured **Closed and were replaced**: the coupled `u' = 0.9u + 0.6v` form (steady-state Cov(u,v) = 15.79 makes u nearly sufficient — R² 0.98, gap 0.011) and the same-ρ `x = u + v` mixture (a mixture of equal-ρ chains is EXACTLY AR(ρ) — `m' = 0.9m + noise`, gap ≈ 1e-5); both closed systems were the AUDIT being right, not wrong. The h-chain's noise enters the mixture scaled by w², so the gap is maximized at SMALL w — the w = 0.5 pin carries that derivation in the FixtureKind docs.
- [x] T1: primitive impl (ridge sweep + kNN, blocked-forward split, two-condition verdict, typed `Undecidable` on weak arms — never a green zero; zero per-tick alloc in online consumers ([`WindowRing`], G4 hard-zero over 1000 wraparound pushes), bounded disclosed offline budget (per-arm walls + kNN reference posture in `ClosureBudget`). **API NOTE:** trajectories are FLAT row-major recordings with EXPLICIT `dim`/`micro_dim` (a flat recording's length cannot encode its width — the inferred-dim shape collapsed every run to zero windows and was caught by the fixtures). Determinism: bit-identical reports fixture-asserted; `(distance², reference index)` kNN tie rule; Cholesky back substitution pushes via `L[i][j]` — the `L[j][i]` spelling reads the STALE pre-Cholesky upper triangle (found by the G1 fixtures printing R² = −8311 / −10²⁷).
- [ ] T2: GOAT bench — G1 fixtures above; G2 **split budgets, pre-registered at filing (this line IS the pre-registration — never re-pinned after a failed run)**: ridge sweep ≤ 50 ms at T=10⁴, D=64 release (closed-form normal equations, ~10⁸–10⁹ flops per λ, fine); **kNN arms carry their own disclosed budget line** — the shared n_test×n_train×D distance pass ≈ 0.3–2 s single-threaded at T=10⁴, so either pin a kNN reference-subsample cap as part of the design or budget the full pass explicitly in the bench output; G4 alloc census. Calibration reference (provenance, never our bars; **v3** App. D + App. N): RET held-out R² 0.908/0.924/0.979/0.925 vs matched ceilings 0.716/0.574/0.786/0.705; config-variant closure 0.795–0.949.
- [ ] T3 (follow-up, this lane): light phase-clustering arm (Lloyd's + agglomerative, pinned seeds, first-index ties) + NMI-across-≥5-pinned-seeds disclosure (`StochasticBelow(floor)` typed, never promoted on one seed) for the FSM-agreement audit consumer.
- [ ] T4: hand consumers their first census (riir-ai `.issues/1259` sync surface; riir-refine `.issues/256` fix trajectories) once T1+T2 land.

## Proof path (pre-registered)

Falsifier: if the injection and noise fixtures cannot be separated by the gap at the pinned ε (i.e. the matched-capacity design fails to isolate the microstate's marginal predictability with closed-form arms), the primitive is DEMOTED in the same commit — the linear+kNN family is disclosed as the modelless floor; the paper's MLP capacity sweep is the trained-predictor escalation (training track, Research 692 §5).

## Related

Research 692 (the note) · riir-ai `.issues/1259` · riir-refine `.issues/256` · Research 667 (analytical twin — compose: the audit validates ObservationGate cadences 667 derives) · Plan 324 bisimulation (discrete cousin) · Plan 342 `latent_trajectory_geometry` (the diagnostic-primitive precedent).
