# Issue 939: Phase-carrying belief kernel + constructed phase-code counters (underdamped κ-blocks)

**Status:** Open — POC, **defer-until-pull** (no committed consumer; promotion requires a named pull — posture set at verdict round 1).

## The idea

Per-NPC belief kernels (`katgpt-sense/src/reconstruction.rs:996` `evolve_belief`, `:1055` `evolve_belief_additive`, plus `evolve_belief_regularized` in `reconstruction_depth_invariance.rs:141` — the RECONSTRUCTION lane, not the drive/FSM stack) are leaky integrators: magnitude-only memory — a belief knows *how much*, never *when* or *with what period*. arXiv:2610.13341's damping-regime framework (Research 620) gives a drop-in upgrade: replace a scalar leak per channel with a 2×2 underdamped κ-block,

```
z_{t+1} = ρ·R(θ_t)·z_t + write     (R = 2D rotation; κ = -ν² ⇒ 0 < ζ < 1)
```

— **phase-carrying belief**: the channel's rotation phase carries input-conditional increments, so a belief channel gains memory of *when*, not just *how much* (last-encounter phase, cadence, periodicity). The paper's E.1 shows gradient descent converges to exactly this shape as a **modular counter**: input-conditional phase increment `Δφ ≈ 2π/m` per event class, cosine readout = count mod m (trained pairs: ≈180°/parity, ≈120°/mod-3, ≈0°/other).

**TARGET consumer (does not exist yet — this is why the issue is defer-until-pull):** mod-m event-count drives for the FSM/drive stack (combo cadence, aggro cycles, spawn patterns) with `sigmoid(cos φ)` readout — smooth, house-compliant, no hard edges beside the FSM.

## Why not already covered (signal-diff vs the oscillation cousins)

- **Research 169 (LinOSS/Plan 189):** *free* oscillation (imaginary-axis eigenvalues) as architecture for KV evolution / spec decode. No input-conditional phase increments, no damping-regime placement, no modular-counter construction.
- **Research 371 (oscillatory DMFT):** mean-field regime classification of low-rank RNN populations with adaptation — population-level theory, not a per-channel constructed primitive on the belief kernel.
- Delta: **constructed input-conditional phase codes + ζ-as-a-design-knob on the belief substrate.** Novelty TBD in the strong sense: the mechanism is published mathematics (second-order oscillators are classical); the claim is only ever "first consumer on our belief substrate," which is a GOAT-shaped engineering claim, never a Super-GOAT.

## POC tasks

- [ ] P1. Constructed counter correctness — **integer formulation is the deliverable, not f32 phase accumulation** (verdict round 1): the undamped code `φ += 2π/m·1[event]` drifts in f32 over 10⁶ events; the exact form is a `u8` count mod m + lookup `cos(2π·c/m)` — bit-deterministic by construction, zero drift, cheaper than sincos. The **non-trivial primitive is the damped block (ρ < 1)**: phase and magnitude fade together, no exact integer counter exists — that is where the 2×2 κ-block earns its keep. Gate: integer counter exact over 10⁶ random streams; damped block's phase/magnitude decay matches the closed form `ρⁿ·R(nθ)` to f32 tolerance.
- [ ] P2. Underdamped belief block vs leaky kernel on parity/mod-3 fixtures — **SANITY CHECK, not a GOAT gate** (verdict round 1): a real-spectrum leaky integrator cannot represent parity, so "beats it by ≥20 pts" is true by construction and proves too little. Keep as the reproduction smoke test; the deciding gates are P3 + a named consumer pull.
- [ ] P3. FSM interaction real-boot test (the house behaviour law — a DECIDING gate): the ringing drive must assert PROGRESS through the FSM — no oscillation-induced flip-flop between two states; the ζ-budget floor is the control, the test is the proof. (The paper's own dynamics warn: uncontrolled oscillation causes back-and-forth — exactly the failure class the FSM rule exists for.)
- [ ] P4. Accumulation bound as a DESIGN constraint for the new block itself: the proposed damped κ-block `z = ρ·R(θ)·z + u` IS an additive write, so the paper's C.3 steady state `‖u‖/|1-ρe^{iθ}|` bounds its own saturation — use it to set per-channel write budgets. ⚠ θ-dependence (reviewer round 3): at ρ→1 the gap → `2·sin(θ/2)`, so the bound scales as `‖u‖/(2·sin(θ/2))` — large-θ channels (e.g. θ = 2π/3 ⇒ ‖u‖/√3) get the rotation benefit, but SMALL-θ channels need the same budget as a leaky kernel; P4's per-channel budget computes this anyway. Scope note (verdict round 2): NO shipped lane is an additive-write accumulator (GDN = delta rule, belief kernels = convex EMA — all bounded by construction), so this constrains the NEW primitive only; Research 620 §2 carries the full no-consumer verdict.

## Feature flag

`phase_belief` (opt-in, katgpt-sense); **promotion requires P1–P4 green AND a named consumer pull from the drive/FSM stack** (defer-until-pull). Zero-alloc hot path; fixed `[f32; 2]` per channel block; integer counter where undamped.

## References

- Research: `.research/620_Mamba3_Oscillatory_Dynamics_Damping_Regimes.md` (§2 Path 0 table, §3.3 game reframe, §3.1 duality)
- Paper: arXiv:2610.13341 §3 (equivalence), §E.1 (phase codes), §C.3 (accumulation bound — additive-write scope)
- Cousins: `.research/169` (LinOSS), `.research/371` (DMFT), `.research/301` (Subspace Phase-Gate — the real-eigenvalue phase cousin; Hopf extension lives there)
- Substrate seam: `crates/katgpt-sense/src/reconstruction.rs` (`evolve_belief`, `evolve_belief_additive`)
