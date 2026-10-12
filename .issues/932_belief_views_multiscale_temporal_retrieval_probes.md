# Issue 932: Belief Views — Multi-Scale Temporal Retrieval Probes (MSAR time-depth transpose, novelty TBD)

**Status:** Open — probes not started; five scripted probes gate any code. Fusion idea filed per research-skill §1.5 (novelty TBD — NOT a plan, NOT a Super-GOAT claim until E1+E2 clear).

**Source:** Research 619 ([arXiv:2610.13630](https://arxiv.org/abs/2610.13630) MSAR, 2026-10-12). Adversarial-panel findings (No-GD advocate) are the basis for the probe order.

## Claim (pinned — kill or confirm by the probes)

*Weight-free multi-view temporal retrieval over a bounded per-agent latent state chain* for think-brain belief evolution: block summaries of the `evolve_belief` trajectory (`[f32; 8]`, katgpt-sense `ReconstructionState`), m views via causal temporal convs, content query = current belief / `temporal_deriv` fast component / drive projection, **independent sigmoid gate per (stream, entry) over normalized similarity** (house rule: sigmoid, never softmax, for our primitives — the paper's softmax belongs to Plan 458's faithful reproduction, not to this modelless primitive; sigmoid also lets the gate ABSTAIN — all-gates-low = "history carries nothing relevant" — instead of being forced into a convex combination over past summaries, which is exactly the EMA-with-extra-steps failure shape), weighted fusion into the update. Distinguished from `ObservationGate` decay (content-independent `sigmoid(-λ·Δt)`) by content-dependence; from GDN2 (trained projections, token-stream) by zero learned parameters + state-chain consumer. ~30 flops/NPC/tick at m=4, K=8, d=8. Latent-local; only scalars cross sync.

## Why probes first — three named failure modes

1. **Recency degeneracy:** `dot(current_belief, past_view)` over a smoothly-evolving 8-dim signal ≈ monotone in age → any normalized-combination readout rebuilds an exponential-decay prior = "EMA with extra steps." Sigmoid gates soften this (they can abstain instead of being forced to sum to 1) but do not remove it — T4 measures both readouts rather than assuming.
2. **Kernel-scale mismatch:** literal kernels {4,8,12} ticks at 20 Hz = 0.2–0.6 s ≪ belief time constants → near-identical views (identical DC component). The paper's own conv-removal ablation (1.847→1.862) says the views ARE the engine — wrong-scale views are dead views.
3. **Destruction disanalogy (viability decider):** depth-residual retrieval recovers info destroyed by layers; over time `b_{t-1}` is an *input* to `b_t` — headroom ∝ the lossiness of `evolve_belief` (clamps, sigmoid saturation, event bumps). No lossiness → no headroom → "single-stream stays" is the honest outcome.

## Tasks

- [ ] T1 (E2): per-channel autocorrelation half-life τ of belief trajectories (riir-ai logs or synthetic scenarios) → derived kernel schedule κ_r ≈ τ·{0.5, 2, 8}. Script only. RED if τ ≪ 1 tick on all channels (signal too smooth to have multi-scale structure).
- [ ] T2 (E6): detrended (DC-removed) vs raw similarity — does only detrended produce non-monotone age profiles? Script.
- [ ] T3 (E1): lossiness probe — variance of `b_{t-1} | b_t` over logged trajectories (or closed-form invertibility read of the update ops: clamp, sigmoid saturation, evidence injection). If ~self-restoring → close as recorded negative ("single-stream belief stays").
- [ ] T4 (E3): argmax-age entropy on a frozen scenario corpus — context-dependent age jumps must exist; monotone-in-age reds. **Two readout arms, measured not assumed: sigmoid-gate retrieval vs softmax-retrieval** — the paper's softmax forces a convex combination (the EMA attractor); sigmoid can abstain. Record which readout survives the entropy floor; the loser is closed in writing.
- [ ] T5: retrieval-age diagnostics spike (machine-readable affective persistence — grudge/grief signal shape); report-only, ≤2 scalars across any sync surface.
- [ ] T6 (E4, only if T1+T3 clear): paired LB95 on a frozen corpus, multi-view vs single-stream `evolve_belief`, downstream label = next observable event / aggression switch — **at the T4-winning readout only**. Gate must include the degeneracy floor from T4 (loud-zero defense).
- [ ] T7 (only if T6 passes): primitive lands in katgpt-sense as `belief_views` (feature-gated, default-off, operator-named — paper vocabulary in docs only), one view constructor composing with `temporal_deriv`/`SyncBank` (never duplicating), zero-alloc fixed arrays, **sigmoid-gated** (never softmax); consumer wiring in riir-ai is a separate riir-ai issue citing this one.

## Notes

- Probe corpus: existing logs preferred; synthetic fallback must reproduce real-boot cognition (house rule on behavior tests).
- Expansion-on-demand is explicitly DECLINED here (storage/wire law, not a hot-path law at d=8 — a stored multi-scale pyramid is cheaper per tick); do not re-litigate inside this issue.
- Close-negative is a first-class outcome: the recorded negative pins single-stream belief with evidence and stops blind retries.
