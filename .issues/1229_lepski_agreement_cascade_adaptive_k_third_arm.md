# Issue 1229: Lepski agreement cascade + MAD σ̂ — the adaptive_k third arm (Research 681 poc)

**Status:** open — primitives LANDED in katgpt-core `lepski_cascade` (rides the `estimators` umbrella); G1 PASS at A = 1 (cell 2) after A = 4.5 was FALSIFIED as a serving constant (cell 1); G4 PASS; G2 + G3 deferred to a quiet rig; consumers unwired — claimable
**Lease closed:** katgpt-rs-9a@m3, epoch 1795177976, expires 2026-11-22T12:32Z — primitives + G1/G4 landed; G2/G3/consumers left unclaimed for the next lane

Modelless extraction of arXiv:2611.22198 (*Training Neural Networks to be Locally Adaptive*):
a selection-layer primitive — per-query adaptive neighbourhood size by **agreement cascade**
over nested candidates, with a **computable** noise scale. Completes the adaptive_k FAMILY:
{variance-sigmoid heuristic (Plan 256), Lemma-2 closed form (Issue 747 / Research 549),
**agreement cascade with oracle inequality** (this issue)}.

**Placement (explicit):** the primitive functions live in **`katgpt-core`** (generic, no attn
semantics — the substrate-first law); the `dash_attn` `AdaptiveKRouter` arm is a CONSUMER of
the core primitive, never the definition site.

**Route:** m3 interactive lane (fixture + selection fn + tests only; no heavy build) — claimed
above; the G3 reflex-harness cell routes to a quiet rig when it comes up.

## Primitive surface (katgpt-core; consumers wire, never fork)

- [x] `lepski_cascade`: geometric grid over nested neighbourhood sizes (prefixes of ONE top-K
      retrieval — no per-candidate retrieval), sequential scan upward, stop at FIRST rejection,
      select the largest k whose agreement test passed before that rejection. A pinned (≥ 4;
      4.5 default). **Determinism (Issue-940 law):** the pass predicate is `|f̂_k − f̂_ℓ| ≤ A·τ_ℓ`
      (ties pass, `≤` never `<`), the scan order is ascending grid index, the selected k̂ is
      the last passing index before the first rejection — one fixed strict rule, named here.
      The selected k̂ is a LOCAL per-query value, never a sync-crossing quantity; any future
      surface that commits it across the sync boundary must re-pin this order + tie rule at
      the site per the determinism law.
- [x] `sigma_hat_nn`: σ̂ = median_i |Y_i − Y_nn(i)| / (0.6745√2) — nearest-neighbour MAD
      (continuous-response form); derived discrete-label analog = neighbour disagreement rate
      (write the derivation, do not copy the MAD form onto labels).
- [x] `lambert_w`: Halley's method (not plain fixed-point — it converges slowly and is
      unstable at large arguments, and δ̄²'s argument k(L+1)V²/(16(s+1)) gets large), log-based
      starting guess W₀(z) ≈ log z − log log z for z > e, ~4–6 Halley steps to f64 eps, no dep.
- [x] Entropy-derived floor δ̄²_k = 2048(s+1)/k·W(k(L+1)V²/(16(s+1))) — turns the measured
      K-floor into a derived floor.
- [x] τ̂_k = σ̂·√(log n / k) practical scale (paper §5 form); the full τ_k (§4.4) when the
      architecture constants are held.
- [x] Feature flag; zero-alloc hot path; constants read from the paper at implementation time
      (Research 681 §4 hygiene note), never from the note's prose.

## Gates (pre-registered; the falsifiers kill, they do not get re-tuned)

**G1 operationalisation (pinned 2026-11-20 by katgpt-rs-9a BEFORE the first run; the issue
text left the aggregation open):** n = 2000, X ~ U[0,1] (5 seeds, splitmix64), planted
σ = 0.1 (the paper's verbatim §1.1 level), 200 equally spaced queries on [0,1], estimator =
kNN mean, grid {4, 8, …, 512} (the paper's k-NN–Lepski grid family), A = 4.5 (the pinned
default). Fixtures: Doppler f₀ = √(x(1−x))·sin(2.1π/(x+0.05)) (paper eq. 1.2) and
piecewise-constant levels (0, 1, −0.5, 0.5) on quarters. Oracle k*ₓ = largest grid k with
b_ℓ ≤ τ_ℓ for every ℓ ≤ k (paper eq. 4.6 as an initial segment), b_ℓ = |noiseless kNN mean
of f₀ − f₀(x)|, τ_ℓ = σ√(log n/ℓ) at the TRUE σ.
- **Structure** (per fixture, 1000 pooled queries): PASS iff (a) ≥ 80% of queries have
  |log₂(k̂/k*)| ≤ 1 AND (b) Spearman ρ(k̂, k*) ≥ 0.5 (average ranks).
- **Property** (per fixture): |f̂_k̂ − f₀(x)| ≤ (A+2)·τ̂_{k*} with σ̂ estimated from the
  sample (consecutive-difference MAD) on ≥ 99% of pooled queries.
- **Discrete**: n = 4000, p(x) = 0.2 below ½ and 0.8 above (planted σ = 0.4); per seed
  σ̂ = √(D/2) within 5% of 0.4; the copied MAD form reads 0 (the degeneracy the derivation
  names); vote-share cascade (A = 4.5): mean k̂ within 0.05 of the boundary < mean k̂ beyond
  0.2; adaptive majority vote agrees with the Bayes label on ≥ 95% of pooled queries.
- Disclosed, NOT a verdict input: the same structure numbers at A = 1 (the paper's own
  experiment constant, outside the proved A ≥ 4 range).

**Cell 1 RESULT (A = 4.5, 2026-11-20, katgpt-rs-9a, `/tmp` dev build, m3):** property
PASS both fixtures (1.000/1.000); structure piecewise PASS (0.989, ρ 0.804); structure
**Doppler FAIL** (0.536 < 0.80, ρ 0.318 < 0.5; upper end binding on 39.7% of queries);
discrete σ̂ PASS (0.3836–0.4119 vs 0.4), copied-MAD degeneracy PASS (reads 0), Bayes
agreement PASS (0.997), discrete **tracking FAIL** (k̂ = 512 near and far — the cascade
never rejects: A·τ̂_ℓ ≥ 1 for ℓ ≲ 27 on a [0,1] vote share and ≥ 0.32 at ℓ = 256, above the
boundary-induced shift). Verdict: **the pinned A = 4.5 is FALSIFIED as a serving constant**
— the theory-admissible constant is too wide at n ≤ 4000 to see bias (the paper itself
runs A = 1). The premise is NOT refuted by this cell: the disclosed A = 1 arm tracked
Doppler (0.939, ρ 0.790) — disclosed, not a verdict input.

**Cell 2 (pinned 2026-11-20 BEFORE its run, after cell 1):** the hypothesis "A = 1 (the
paper's §5 constant, not a sweep pick) tracks" on FRESH seeds 0x1229_0101…0105, every
cell-1 threshold unchanged (structure 0.80/0.5 per fixture, property (A+2)·τ̂_{k*} ≥ 99%,
discrete σ̂ 5% + degeneracy + near < far + Bayes ≥ 95%). PASS ⇒ the module's serving
constant is A = 1 with the A ≥ 4 theory constant kept for the oracle-bound reading; FAIL ⇒
demote to watch row (premise refuted at practical n). No third cell.

- [x] G1 structure: selected k tracks planted smoothness within ONE grid step of the oracle
      k*_x on the Doppler + piecewise-constant fixtures (planted σ). **Falsifier:** no tracking
      ⇒ the agreement-is-smoothness premise is refuted on that surface ⇒ demote to watch row.
- [x] G1 property gate: |estimate − truth| ≤ (A+2)·τ̂ on ≥ 99% of queries at planted σ;
      violation indicts σ̂/agreement statistic, never the rule. **Report-the-Floor exemption
      (stated, per the gate's own discipline):** this is NOT a calibrated-interval claim — the
      bound is checked at a KNOWN planted noise level (σ is planted, not estimated-then-served),
      so no interval is being calibrated and the conformal-naive floor does not adjudicate this
      gate. The floor BECOMES the comparison arm if a surface later serves τ̂ as a calibrated
      interval (e.g. reflex abstention wiring) — that cell carries its own
      `ConformalIntervalCalibrator<SeasonalNaiveForecaster>` bar, per-predicate, never pooled.
- [x] G1 known answers: Lambert-W vectors (W(e)=1, W(1)≈0.567143, monotone, branch checks,
      **large-argument vector W(1e6)** for the Halley path).
- [x] G1 discrete: disagreement-rate σ̂ analog on a labeled fixture (majority-vote estimator).
- [-] G2: latency parity vs fixed-K incumbent at the serving posture (expected ≤ fixed-K_max —
      one retrieval either way, +log-n folds).
- [x] G4: alloc census zero.
- [-] G3 (the promotion gate, separate cell): paired accuracy vs incumbent fixed-depth scoring
      on the reflex harness suite slice (cal + test, paired LB95, ONE frozen read) — runs on a
      quiet rig; no promotion talk before it runs.

**Landed (2026-11-20, katgpt-rs-9a):** `crates/katgpt-core/src/lepski_cascade.rs` —
`lepski_cascade` (eq. 5.1 every-pair-in-prefix form, so stop-at-first-rejection is exact),
`prefix_means_into`, `tau_hat(_into)`, `sigma_hat_from_abs_diffs` / `sigma_hat_consecutive` /
`sigma_hat_disagreement` (√(D/2), derivation in the doc comment), `lambert_w0` (Halley on
w − z·e^{−w}, overflow-free), `critical_radius_sq`, `network_volume_v`, `kappa_sigma`,
`tau_full`, `implementable_k0/j`, `geometric_grid_into`. Constants were checked against the
paper's TeX source (arXiv src v1: eq. 4.2, 4.4, 4.6, 5.1, Lemma 4.3), not the note.
`A_DEFAULT = 1` (serving, cell 2 PASS), `A_THEORY = 4.5` (reads the oracle bound; cell 1
falsified it as a serving constant). No own feature key — gated on the `estimators`
umbrella (compounding/Issue-1222 precedent; katgpt-core 296/300 of the crates.io cap).
Gates: `tests/issue_1229_lepski_g1.rs` (cell 1 pinned as the recorded negative, cell 2 the
deciding gate), `tests/issue_1229_lepski_g4_alloc.rs` (0 allocations), 14 unit tests.
**Cell 2 RESULT (A = 1, fresh seeds):** Doppler 0.935 within one step, ρ 0.833, property
0.999; piecewise 0.985, ρ 0.951, property 0.999; discrete σ̂ 0.4015–0.4142 (≤ 3.6% off),
mean k̂ near 229 < far 501, Bayes 0.997 — every pinned bar PASS. Dev profile on m3; G1 is a
correctness claim, not a latency one. **G2/G3 deferred:** m3 was BUSY (Zed contender) and
the owner scoped this session to light m3 work with no other rig, so the latency cell and
the reflex-harness promotion cell route to a quiet amo/bmo lane. No promotion talk before G3.

## Consumers (wire the katgpt-core primitive; each consumer its own flag + gate)

- [ ] `dash_attn::AdaptiveKRouter` — cascade arm beside the two shipped arms (estimates are
      folds over already-computed block scores).
- [ ] reflex adaptive retrieval depth + abstain-on-cascade-failure (fusable with
      `CorpusDistanceGate`; the honest open question — whether prefix-truncated corpus scoring
      degrades gracefully the way kNN regression does — is measured by G1/G3 before any wiring).
- [ ] refine fan-out per-span adaptive K (prefix-form keeps ONE retrieval; Bench-126 call-count
      floor not re-paid).
- [ ] Watch row (not a task): `elo_k_eff` per-rule adaptive evidence floors.

Ref: `katgpt-rs/.research/681_Locally_Adaptive_kNN_Neighbourhood_Selection.md`
