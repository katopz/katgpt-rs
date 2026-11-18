# Issue 1229: Lepski agreement cascade + MAD σ̂ — the adaptive_k third arm (Research 681 poc)

**Status:** open — poc/proof task, GOAT-tier pending the pre-registered gates (Research 681 §4)
**Claimed:** m3-research-lane, 1795019668, expires 2026-11-20T00:45Z

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

- [ ] `lepski_cascade`: geometric grid over nested neighbourhood sizes (prefixes of ONE top-K
      retrieval — no per-candidate retrieval), sequential scan upward, stop at FIRST rejection,
      select the largest k whose agreement test passed before that rejection. A pinned (≥ 4;
      4.5 default). **Determinism (Issue-940 law):** the pass predicate is `|f̂_k − f̂_ℓ| ≤ A·τ_ℓ`
      (ties pass, `≤` never `<`), the scan order is ascending grid index, the selected k̂ is
      the last passing index before the first rejection — one fixed strict rule, named here.
      The selected k̂ is a LOCAL per-query value, never a sync-crossing quantity; any future
      surface that commits it across the sync boundary must re-pin this order + tie rule at
      the site per the determinism law.
- [ ] `sigma_hat_nn`: σ̂ = median_i |Y_i − Y_nn(i)| / (0.6745√2) — nearest-neighbour MAD
      (continuous-response form); derived discrete-label analog = neighbour disagreement rate
      (write the derivation, do not copy the MAD form onto labels).
- [ ] `lambert_w`: Halley's method (not plain fixed-point — it converges slowly and is
      unstable at large arguments, and δ̄²'s argument k(L+1)V²/(16(s+1)) gets large), log-based
      starting guess W₀(z) ≈ log z − log log z for z > e, ~4–6 Halley steps to f64 eps, no dep.
- [ ] Entropy-derived floor δ̄²_k = 2048(s+1)/k·W(k(L+1)V²/(16(s+1))) — turns the measured
      K-floor into a derived floor.
- [ ] τ̂_k = σ̂·√(log n / k) practical scale (paper §5 form); the full τ_k (§4.4) when the
      architecture constants are held.
- [ ] Feature flag; zero-alloc hot path; constants read from the paper at implementation time
      (Research 681 §4 hygiene note), never from the note's prose.

## Gates (pre-registered; the falsifiers kill, they do not get re-tuned)

- [ ] G1 structure: selected k tracks planted smoothness within ONE grid step of the oracle
      k*_x on the Doppler + piecewise-constant fixtures (planted σ). **Falsifier:** no tracking
      ⇒ the agreement-is-smoothness premise is refuted on that surface ⇒ demote to watch row.
- [ ] G1 property gate: |estimate − truth| ≤ (A+2)·τ̂ on ≥ 99% of queries at planted σ;
      violation indicts σ̂/agreement statistic, never the rule. **Report-the-Floor exemption
      (stated, per the gate's own discipline):** this is NOT a calibrated-interval claim — the
      bound is checked at a KNOWN planted noise level (σ is planted, not estimated-then-served),
      so no interval is being calibrated and the conformal-naive floor does not adjudicate this
      gate. The floor BECOMES the comparison arm if a surface later serves τ̂ as a calibrated
      interval (e.g. reflex abstention wiring) — that cell carries its own
      `ConformalIntervalCalibrator<SeasonalNaiveForecaster>` bar, per-predicate, never pooled.
- [ ] G1 known answers: Lambert-W vectors (W(e)=1, W(1)≈0.567143, monotone, branch checks,
      **large-argument vector W(1e6)** for the Halley path).
- [ ] G1 discrete: disagreement-rate σ̂ analog on a labeled fixture (majority-vote estimator).
- [ ] G2: latency parity vs fixed-K incumbent at the serving posture (expected ≤ fixed-K_max —
      one retrieval either way, +log-n folds).
- [ ] G4: alloc census zero.
- [ ] G3 (the promotion gate, separate cell): paired accuracy vs incumbent fixed-depth scoring
      on the reflex harness suite slice (cal + test, paired LB95, ONE frozen read) — runs on a
      quiet rig; no promotion talk before it runs.

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
