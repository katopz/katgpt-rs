# Plan 639: Novelty-Gated Retention — Input-Gated Decay for Persistent Latents

**Status:** Open — Phase 2 DONE (Bench 987): G1 ✓ · G2 ✓ (34/36 cells, rational law) · G3 ✗ (1.53× vs ≤1.15×; floor ≥1.40× measured) · G4 ✓ → NOT GOAT yet; primitive stays DEFAULT-OFF; Phase 3 BLOCKED on Issue 1231 (G3 fused kernel / pre-registered re-price). Phase 1 `127a145dd`. Research: riir-train/.research/500 §4. Mechanism provenance: input-gated decay is PUBLISHED (Mamba input-dependent Δ; arXiv:2611.21608 §3.2.2) — an ADOPTION fused with our shipped novelty signal, never claimed novel to the world.
**Lease closed:** katgpt-rs-2b@m3, 1795068547, expires 2026-11-21T06:09:07+00:00 — Phase 1 landed 127a145dd
**Lease closed:** katgpt-rs-49@m3, 1795071538, expires 2026-11-21T06:58:58+00:00 — Phase 2 landed (Bench 987)

**Date:** 2026-11-19
**Research:** [riir-train/.research/500_Latent_Lattice_Language_Models.md](../riir-train/.research/500_Latent_Lattice_Language_Models.md) §4
**Source papers:** [arXiv:2611.21608](https://arxiv.org/abs/2611.21608) §3.2.2 (affine horizontal update, input-derived g) · Mamba (Gu & Dao 2024, input-dependent Δ) — adoption framing, no world-novelty claim
**Target:** `crates/katgpt-sense/src/` (retention law + `evolve_belief_additive` wiring). Feature `novelty_retention = ["modality_additive"]` (the kernel sits behind `modality_additive`, katgpt-sense Cargo.toml default-on; the new flag is default-off with DEMOTE literal `NOVELTY_RETENTION=0`).
**Incumbent (the thing to beat):** `evolve_belief_additive` (reconstruction.rs:986–1004) — per-channel FIXED retention `config.belief_retention[m]` (default `[0.9; 6]`, reconstruction.rs:163), sigmoid-bounded drive. The incumbent is the GOAT loser candidate, never edited ungated.
**Plan-277-F1 relationship:** reconstruction.rs documents "Plan 277 Fusion F1" for the dual-(fast−slow) derivative lane (:92, :433) with post-GOAT wiring deferred. **This plan IS that post-GOAT wiring lane** — the F1 deferral is satisfied/superseded by this plan's own pre-registered GOAT gate; cite this plan at the F1 hook sites when wiring lands.
**Determinism law (Issue 940, binding):** the belief vector crosses the sync boundary (SyncBank observes the kernel — reconstruction.rs own doc; the SIMD sibling is deliberately routed to scalar code per Issue 940 SE-2). Therefore `g_t` and every per-channel novelty signal are computed with **fixed sequential ordered folds** (`dot_f32_ordered` or plain `+=`) — never `simd_dot_f32` lane-order sums. `sigmoid_surprise_gate`'s own body uses `simd_dot_f32` (katgpt-types/src/temporal.rs:210) and MUST NOT be reused as-is for this path.
**Consumers (follow-ups, each its own issue only if the gate passes):** riir-engine goal_salience `reach_decay` 0.995 · desperation_dual EMA β · katgpt-hla cache γ (local half only — sync crossing untouched) · riir-refine `ema_step` α/β.

---

## Goal

Close the one lattice delta the census found: the fleet's persistent latents all have **input-dependent drive, fixed retention**; the paper (and Mamba before it) derives retention `g` from the current input. Fuse a per-channel novelty signal (built ON the shipped dual-EMA machinery) into a deterministic retention law and prove on a fixture that gated retention **strictly dominates the incumbent's entire fixed-α trade-off curve** — adapting faster at every noise level the incumbent can reach by retuning α alone. GOAT gate below; the loser (fixed α) is demoted only per gate, incumbent bytes intact at `NOVELTY_RETENTION=0`.

## Phase 1 — Primitive + fixture + gates (0 GPU, m3/CPU)

### Tasks

- [x] **T1.1** Per-channel retention law behind `novelty_retention`: per kind m, `g_t[m] = clamp(α_base[m]·(1 − κ·σ(β·s_t[m])) + μ·σ(β·n_t[m]), g_min, g_max)` with **`g_max < 1` (default 0.995) and `g_min > 0`** — a saturated gate must never zero `(1−g)·drive` and freeze a channel (the stuck-behavior the brain/FSM law forbids). Signals per kind: `s_t[m]` = sustained novelty (per-kind SLOW-EMA displacement over the window — the world really changed for that kind), `n_t[m]` = transient share (per-kind fast−slow divergence not explained by slow movement — noise). This needs **new per-kind fast/slow EMA state (6×2 floats)** — O(1) per channel per tick, no new passes over the input beyond the existing accumulate; all folds sequential-ordered per the determinism law above. κ/μ/β/g_min/g_max fixed config. NaN → `g_t[m] = α_base[m]` (safe default: incumbent behavior).
- [x] **T1.2** Wire into `evolve_belief_additive` behind the flag. **G1 identity, named exactly:** flag-off takes the INCUMBENT CODE PATH (α_base used directly, `g_t` never computed) — not an arithmetic-equivalent reformulation; and the armed-path arithmetic identity holds at κ=0 **AND** μ=0 (both zero, else the `μ·σ(β·n)` term survives). `NOVELTY_RETENTION=0` is the exact DEMOTE literal (env-switch law).
- [x] **T1.3** Fixture: synthetic hidden world — (a) steady state + noise, (b) sustained steps at known ticks, (c) transient spikes (noise bursts), (d) mixed, **(e) sustained high noise (long run — the channel-freeze case)**. Belief-tracking error vs hidden state, per tick. Commit the fixture; it is the gate's only judge.
- [x] **T1.4** Gate suite: G1 identity arm (flag-off digest ≡ incumbent); arithmetic arm (κ=0 ∧ μ=0 ≡ incumbent on the fixture); polarity arms (sustained step must LOWER retention — belief adapts; pure-transient burst must RAISE effective retention vs drive — belief holds); NaN arm; boundary arm (`g_t ∈ [g_min, g_max]` always); **channel-freeze arm — under case (e), `(1−g_t[m])·drive` stays > 0 for every channel every tick** (no lock).

**Phase 1 record (`127a145dd`, katgpt-rs-2b@m3):** `crates/katgpt-sense/src/novelty_retention.rs` (law + state), armed via `ReconstructionState::arm_novelty_retention` (returns false under `NOVELTY_RETENTION=0` or a guard-breaking config), fixture `crates/katgpt-sense/tests/novelty_fixture/mod.rs`, suite `tests/novelty_retention_gates.rs` (8/8, also green under the DEMOTE env). Two design verdicts (Claude, recorded here, Phase 2 judges them on the fixture):
- **Centering:** the law uses `ŝ = 2σ(β·s) − 1`, `n̂ = 2σ(β·n) − 1` instead of the literal `σ(β··)` — the literal form is 0.5 at zero novelty and silently shifts a steady channel to `α(1−κ/2)+μ/2`; the centered form keeps EXACTLY `α_base` with no novelty (unit-tested), so the gate acts only on evidence. κ=0 ∧ μ=0 identity unaffected.
- **Signals:** `s = |fast_{t−1} − slow_{t−1}|` (PRE-update — a one-tick spike cannot read as sustained; persistence is the evidence), `n = max(|x_t − fast_{t−1}| − s, 0)`. Defaults κ=0.4, μ=0.2, β=8, g∈[0.05, 0.995], EMA 0.3/0.03 (the shipped F1 ratio) — starting point only; T2.2's grid decides.
- G4 (zero-alloc fixture assertion) and the per-tick debug-pod report remain Phase 2 items (accessors `last_gate/last_sustained/last_transient` exist).

**Phase 2 record ([Bench 987](../.benchmarks/987_novelty_retention_goat.md), katgpt-rs-49@m3):** harness `crates/katgpt-sense/tests/novelty_retention_goat.rs`, metrics + verdict arithmetic committed BEFORE the first run (`f2be3ef2a`); (ii)/(iii) judged against a DENSE α curve (0.50..0.99 step 0.01), closing the sparse-grid escape up front.
- **G2 PASS:** logistic (Phase 1) law 36/36 cells; shipped rational law 34/36. Best: κ=0.8 μ=0.4 β=8 g_max=0.995 → adaptation 5.04 ticks (fixed-α≈0.55 speed) at noise_rmse 0.0231 (fixed-α≈0.93 noise). Caveat: single-RMSE on case (d) Mixed ties, does not beat, the best fixed α (0.0664).
- **G3 FAIL:** logistic 3.06× → rational `u/(1+u)` (one shared division, 8-lane padded, `#[inline]`) **1.53×**. Floor probe (hard ramp, no division) 1.40× — the bar premise "6 sigmoids are the cost floor" is FALSIFIED (dual-EMA state + wiring ≈ +40%). **Claude verdict:** bar NOT moved post-hoc; smooth rational law shipped (kink-free, −50% vs logistic); routes → Issue 1231 (fused kernel T1, else a consumer-budget re-price pre-registered BEFORE re-judging).
- **Saturation-law change (Claude verdict):** `ŝ = r(β·s/2)`, `n̂ = r(β·n/2)`, `r(u) = u/(1+u)` replaces `2σ(β·)−1` — same zero, same slope at 0, same range, monotone; κ=0∧μ=0 identity still bit-exact (gates 8/8, also under DEMOTE).
- **G4 PASS:** armed forward 0 allocations over 400 ticks (binary-local counting allocator). The per-tick debug-pod report stays deferred to Phase 3 wiring (accessors exist).

## Phase 2 — GOAT bench (CPU fixture lane, pre-registered)

### Tasks

- [x] **T2.1** **Incumbent trade-off curve first:** sweep fixed α ∈ {0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97} × 3 seeds — the incumbent's (adaptation, noise-RMSE) Pareto curve, recorded as the baseline artifact. A pass against α=0.9 alone proves nothing that retuning `belief_retention` wouldn't.
- [x] **T2.2** Gated sweep (κ, μ, β, g_max) over a pinned grid (e.g. κ ∈ {0.2, 0.4, 0.8}, μ ∈ {0, 0.2, 0.4}, β ∈ {4, 8}, g_max ∈ {0.99, 0.995}), 3 seeds per cell. **Pre-registered bar — beat the curve, not the default point (and not the impossible corner):** a gated cell passes iff (i) there EXISTS a fixed-α point p on the T2.1 curve with `gated.adaptation ≤ 0.7·p.adaptation` AND `gated.noise_rmse ≤ 1.1·p.noise_rmse` — the matched-point margins, 3/3 seeds — AND (ii) **no fixed-α point dominates the cell** (`≤` on both axes for no single p — the cell lies on/below the curve, not strictly inside it). Also report the cell's distance below the straight-line curve between neighbouring α points, and make it a third passing condition (verdict-round-3 advisory, adopted): **distance-below-line > 0** — the sparse low-α grid must not let an interpolated fixed-α retune slip through the margins. If a candidate cell's neighbourhood is too sparse to compute the distance honestly, tighten the α grid around the passing cell and re-measure before claiming the cell. ≥1 passing cell = GOAT candidate; report the best cell + BOTH full tables (the sweep table is the artifact, no cherry-picks).
- [x] **T2.3** _(not triggered — 34/36 cells pass)_ Honest falsifier: if NO gated cell satisfies (i)+(ii), the per-kind dual-EMA novelty signal is insufficient for retention gating — **close the plan DONE-NEGATIVE**, incumbent stands, both tables recorded in `.benchmarks/`. A "close but noisy" result without a passing cell is a FAIL, not a partial.
- [x] **T2.4** Cost row: per-tick overhead of the gated kernel (INCLUDING the new 6×2 per-kind EMA state) vs incumbent on the fixture (bar ≤1.15× kernel wall — the belief kernel is hot-path; the 6-sigmoid structure is already the cost floor).

## Phase 3 — Consumer wiring (only if T2.2 passes; each its own issue + own gate)

### Tasks

- [-] **T3.1** goal_salience `reach_decay` gated variant (the same law, scalar tier) — separate issue at GOAT time.
- [-] **T3.2** desperation_dual expectation-EMA β gated variant — separate issue.
- [-] **T3.3** katgpt-hla cache γ novelty gate (LOCAL state only; the 5-scalar sync crossing and the bridge law untouched) — separate issue, careful posture.
- [-] **T3.4** riir-refine `ema_step` rate gating (outcome-signed α/β → evidence-shaped) — separate issue; touches the durable learn-row format, needs its own determinism review.

## GOAT gate (pre-registered)

| Gate | Bar | Demote |
|---|---|---|
| **G1** identity | flag-off takes the incumbent code path (`g_t` never computed), fixture digest ≡ incumbent; armed path at κ=0 ∧ μ=0 ≡ incumbent | any drift = defect, fix before bench |
| **G2** quality | ONE gated cell passing T2.2's (i)+(ii): matched-point margins vs SOME curve point (adaptation ≤0.7×, noise RMSE ≤1.1×, the SAME p) AND dominated by no fixed-α point, 3/3 seeds | no passing cell → DONE-NEGATIVE close, both tables recorded |
| **G3** cost | gated kernel ≤1.15× incumbent wall on the fixture, new per-kind EMA state included | over → shrink the signal path before re-judging |
| **G4** allocation | zero-alloc forward (fixture-asserted) | any per-tick alloc → fix before verdict |

---

**Discipline:** sigmoid everywhere (never softmax); all novelty-signal math in sequential ordered folds (Issue 940 — the belief crosses SyncBank); the per-channel `s_t/n_t/g_t` are REPORTED per tick in the debug pod when armed (observability before adoption); the incumbent's `belief_retention` config is never edited — the gated path reads it as `α_base`; every quoted number cites the fixture seed and cell.
