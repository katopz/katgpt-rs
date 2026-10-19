# Issue 950: InfiLoop gated N/D summary — modelless extraction POC (extends `PrecisionVector`)

**Status:** Open — unclaimed. Research 626 (arXiv:2610.11570, InfiLoop). Modelless track; training track is riir-train Plan 465.

**Claimed:** — (append `**Claimed:** <session>, <epoch>, expires <ISO-UTC>` under this line before implementing; 48 h expiry)

## The primitive

InfiLoop's loop-native residual, stripped of training: a two-counter normalized running summary whose writes are gated by content score vs decayed accumulated mass.

```text
N_t = β·N_{t−1} + e_t·z_t      e_t = exp(clip(s_t, ±8))
D_t = β·D_{t−1} + e_t          β ∈ (0,1) asserted at construction
h_t = N_t / D_t                (mean view; convex hull of candidates)
γ_t = σ(s_t − ln(β·D_{t−1}))   (step view; sigmoid of log-ratio — never softmax)
```

- O(d) memory + O(d) work per write, zero-alloc, f32, deterministic.
- β ∈ (0,1) is load-bearing (paper ablation: β=1.82 → 2.0%, β=1 → 38.4%, β=0.55 → 74.0%) — refuse loud outside the range.
- Properties by construction: `‖h_t‖ ≤ max‖z_i‖` (convex hull), summary never decays to zero (unlike leaky EMA), no clamping (vs Issue 934's 96.9%-clamped `evolve_belief` cumulative variant).
- **t=1 edge:** first write has `D_{t−1} = 0`, so `ln(β·0) = −∞` ⇒ γ = 1 exactly — the first solid write is accepted fully. Pin as a test (P3), never left to log-domain luck.

## Substrate-first: EXTEND, never fork

The **undecayed** form already ships: `PrecisionVector` (`katgpt-pruners/src/posterior/precision.rs`, BAKE-style sequential Bayesian) runs `precision_new = precision_old + w; mean_new = mean_old + (obs − mean_old)·(w/precision_new)` — InfiLoop's gate with **β=1 and a fixed caller weight `w`**. What it lacks is exactly the paper's contribution set: forward decay (precision only grows, never forgets), content-scored weights (`w` is a constant, never `exp(score(z))`), the loop/belief-tick axis, and the convex-hull invariant.

Design: a generic `GatedSummary<const D>` (in the same crate/leaf as the posterior substrate or its consumer leaf — substrate-first decides the home at implementation) carrying a `beta: f32` field:

- **β = 1 must reproduce `PrecisionVector`'s update bit-for-bit** (same fixed-`w` scoring) — this is the G1 oracle and the compatibility proof.
- β ∈ (0,1) arms the decay; content-scored weights (`e = exp(clip(s, ±8))`) arm the gating. Each delta arms independently so the P7 battery can attribute gain per delta.
- `PrecisionVector` itself is NOT modified in this POC (its consumers keep byte-identical behavior); the extension relationship is proven by the oracle test.

## Modelless scorer source

The trained pseudo-query `q` is replaceable by any shipped per-write score: fog-of-war observation confidence (two-brain), coherence (`reestimation` family), cadence shapes, `quality_gate` separation, precision-KL surprise (`posterior/surprise.rs`). Calibration via the shipped sigmoid-calibration discipline. A constant score is NOT a valid primary arm (paper §D: q=0 costs 6–8 pts — the content signal is the point), but it IS a required control (below).

## Tasks

- [ ] P1: `GatedSummary<const D>` kernel (fixed-D `n: [f32; D]` + `mass: f32` + `beta: f32`, `step(z, s)`, `read() -> h`), zero-alloc, byte-deterministic reference test.
- [ ] P2: numerics — score clip ±8; must-fail-when-broken arm for the clip (unclipped large scores overflow `exp`); f32 accumulation order pinned (Issue 940 determinism law applies if any result crosses a sync boundary).
- [ ] P3: sigmoid step-view `gamma(s, mass, beta)` with (a) the identity test `σ(s − ln(β·D)) == e/(β·D + e)` within f32 tolerance, (b) bit-identity where both views consume the same state, (c) the **t=1 edge**: `D=0 ⇒ γ=1` exactly (first write fully accepted).
- [ ] P4: **`PrecisionVector` oracle** — β=1 + fixed-`w` scoring reproduces the shipped `PrecisionVector` mean update **bit-for-bit**: store the state the way `PrecisionVector` updates it (a mean nudged toward each new value, `h += (z−h)·w/(βD+w)`, same in-order floating-point ops), not as a separate weighted sum and total that get divided; the N/D form derives from the nudged-mean state for reading. Bit-for-bit is the compatibility proof — a tolerance match is NOT acceptable here (a tolerance-oracle hides an ordering divergence the oracle exists to catch).
- [ ] P5: convex-hull invariant test (`‖h‖ ≤ max‖z_i‖` across randomized streams, including adversarial single-dominant-candidate cases) + never-decays-to-zero differential vs `leaky_step` on a stopped-write stream.
- [ ] P6: config law + perturbation protocol — `beta` range assert (refuse β ≤ 0 or β ≥ 1 at construction; β=1 allowed ONLY for the oracle posture, named); defaults β₀ = 0.55 with the paper's 2.0/38.4/74.0 ablation cited. Perturbation-resilience replay: inject ε ∈ [1e−4, 0.3] into a simulated write stream, measure summary displacement vs the controls (the paper's Fig. 5 protocol, modelless).
- [ ] P7: GOAT gate — **four-control battery**, one frozen read:
  1. **`PrecisionVector` posture (β=1, fixed w)** — today's shipped behavior; the gain must NOT come from here (this control isolates the decay+content deltas);
  2. **decay only** (β<1, fixed w);
  3. **content only** (β=1, `e = exp(s)`);
  4. **full InfiLoop posture** (β<1 + content);
  plus leaky-EMA and carry-last controls for the external comparison. G1 correctness (properties above), G2 perf (ns/write at D=8/64/512, `--release`, bench_preflight provenance line), G3 no regression (kernel is additive, no incumbent changes, `PrecisionVector` byte-identical), G4 alloc-free (counting allocator canary). Modelless gain = the full posture demonstrably resists noisy overwrites where every control flips or drifts (controlled toy, verdict table in the bench output; per-delta attribution from the four cells).
- [ ] P8 (contingent, defer-until-pull): phase_belief two-signal commit fusion (Research 626 §5.1) — magnitude residual r_free AND content gate γ; follows Issue 939's posture. Do not start before the P7 verdict.

## Constraints

- Brain+FSM law: the gate is DATA feeding the existing deciders (cadence, FSM, frustration drives) — never a bypass, never its own decision layer.
- Sigmoid, never softmax — the full-history softmax must never be materialized; P3's identity is the proof.
- Keep the kernel generic (no game vocabulary) — game wiring is a downstream consumer posture, not this issue.
- Substrate-first: `PrecisionVector` is consumed/oracle-tested, never forked or modified; if the kernel lands in a different leaf than the posterior substrate, the boundary rule (BOUNDARY.md) decides the home.
