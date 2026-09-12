# Research 553: Flybody — Frozen Pattern Core + Residual Control for Whole-Body Locomotion

> **Source:** "Whole-body physics simulation of fruit fly locomotion" — Vaxenburg, Siwanowicz, Merel, Robie, Morrow, Novati, Stefanidi, Both, Card, Reiser, Botvinick, Branson, Tassa, Turaga. *Nature* 643:1312–1320 (2025). https://doi.org/10.1038/s41586-025-09029-4 · code: [TuragaLab/flybody](https://github.com/TuragaLab/flybody) (MuJoCo + dm_control + Acme DMPO + Ray). HHMI Janelia + Google DeepMind. CC-BY 4.0.
> **Date:** 2026-09-12
> **Status:** Done — verdict **Gain (modest)**: one modelless fusion idea (novelty TBD, no consumer) + one training-recipe re-attack. No Super-GOAT.
> **Related Research:** 014 (Learning Beyond Gradients — the workspace's only prior CPG note; this paper is its embodied, Nature-scale confirmation), 374 (OTF-LAM factorized motor primitives), 308 (NSM/VLA frozen low-level skills)
> **Related Issues:** riir-ai #935 (gait-pattern fusion, novelty TBD) · riir-train #535 (residual-over-frozen-heuristic G5' re-attack)
> **Classification:** Public

---

## TL;DR

DeepMind/Janelia built a 102-DoF anatomically detailed MuJoCo fruit fly and trained it (via imitation RL) to fly and walk like real flies, steered by high-level 6-DoF commands. The transferable architecture is **not** the RL — it is a **frozen closed-form core with thin learned residuals**: a lookup-table wing-beat pattern generator (WPG) + offline-fit closed-form fluid coefficients + a frozen pretrained low-level controller reused under a newly trained high-level navigator. The paper's own insight — *large maneuvers arise from small deviations off a nominal periodic pattern* — is a license to linearize: near baseline, the learned residual approximates an offline-derivable control-effectiveness Jacobian.

**Distilled for katgpt-rs (modelless, inference-time):**
1. **WPG pattern class** — frozen periodic pattern table `g(φ)` + phase accumulator (`φ += f·dt`, ±10% frequency) + smooth phase blending + *additive bounded residual*. Zero-alloc O(1) readout.
2. **Small-deviation license** — near a strong periodic baseline, replace a learned controller with `u = σ(gain)·B⁺·Δcmd` where `B = ∂wrench/∂asymmetry` is derived offline by finite differences over the closed-form system.
3. **Offline-fit-then-frozen coefficient lifecycle** — quasi-steady force laws calibrated once offline, evaluated closed-form at runtime (their fluid model; robust to ±20% coefficient error).
4. Recorded non-actionables with reopen triggers (adhesion/friction-margin, gait invariants, DLS-IK, Floquet certificate, optic-flow autopilot) — §4.

---

## 1. Paper Core Findings

- **Body**: 67 rigid segments, 66 joints, 102 DoFs, from confocal microscopy → Blender → MuJoCo. Position actuators (joints) + torque actuators (wings); tendon-coupled DoF groups (tarsi, abdomen).
- **Fluid model**: phenomenological, *stateless*, quasi-steady — closed-form polynomial force/torque laws on ellipsoids covering 5 phenomena (added mass, viscous drag, viscous resistance, Magnus lift, Kutta lift). Coefficients optimized **offline** until stable hover; runtime is pure evaluation. Dominant forces: viscous drag + Kutta lift. Robust to ±20% coefficient variation.
- **Adhesion actuators**: inject normal force at tarsal contacts → enlarges Coulomb friction-cone slip margin → wall/ceiling walking (now a general MuJoCo feature).
- **Flight controller = WPG + residual MLP**: fixed lookup-table baseline wing-beat pattern (from hovering *D. melanogaster* data) + MLP whose output ADDS small torque deviations; one scalar controls WPG frequency (±10%); DMPO action penalty keeps motion near baseline. The WPG doubles as initialization → large training speedup.
- **Steerable low-level controllers**: high-level command = desired 6-DoF CoM position+orientation delta per timestep; low-level converts to 12-dim (flight) / 59-dim (walking) motor signals. Walking input 286-dim egocentric proprioception. Results: flight median CoM error 0.25 mm, orientation <5°; walking median 0.4 cm / 4°; emergent gait stats match real flies (≥3 legs in stance at all speeds, inter-leg phase relations, turning via asymmetric stride lengths; adhesion activates naturally during stance).
- **Hierarchical reuse**: FROZEN pretrained low-level flight controller + jointly trained CNN (2× 32×32 egocentric cameras, 150° FOV) + high-level navigator MLP → vision-guided altitude (bumps) + lateral avoidance (trench). Low-level weights unchanged.
- **Training**: DMPO (MPO + distributional critic), ~1e9 env steps (walking) / 1e8 (flight); Ray-distributed CPU actors + GPU learner; strictly egocentric observables; early termination = failure (future reward zeroed) vs time-limit end (bootstrapped).
- **Regularized IK**: 2D keypoint tracks → full 3D poses via `min_q Σ‖s_i(q)−s_i*‖² + λ‖q−q₀‖²` (default-pose prior, warm start).

## 2. Why mechanism novelty fails (prior-art sweep, all confirmed published)

| Paper mechanism | Published prior art |
|---|---|
| CPG/pattern-generator + residual policy | CPG-actor-critic (AAAI 2004); CPG-RL (RA-L 2022); learned action residuals (CoRL 2023) |
| Frozen low-level + trained high-level | HiPPO (ICML 2018, explicit); ETH quadruped agility 2023; standard HRL practice per surveys |
| Physics/procedural locomotion in games | Motion Matching (Ubisoft, GDC 2016, shipped *For Honor*); learned NN controllers (Ubisoft, GDC 2018) |
| Quasi-steady closed-form flapping aero | Sane & Dickinson 2002 (canonical, 1000+ citations); extensive FWMAV literature |
| Steerable imitation locomotion | DeepMimic (2018); virtual rodent (ICLR 2020; Nature 2024); steerable-from-animal-video (2025) |
| Whole-body insect sims | NeuroMechFly (Nat. Methods 2022, v2 2024) |
| DMPO | MPO (Abdolmaleki 2018) + C51 (Bellemare 2017); ships in acme |

None of the mechanisms is individually novel; the paper's contribution is the *integration* (whole-fly flight+walking unified, RL-trainable full-body behaviors). Our workspace adds no novelty on top: it has **no physics engine, no articulated-body surface, no animation roadmap** (verified — zero rapier/mujoco/physx/avian deps workspace-wide; no skeletal/animation plans in riir-ai or riir-game-sdk).

## 3. Path 0 component table (coverage × extraction)

| Component | Ships? (workspace) | Modelless extraction? | Verdict |
|---|---|---|---|
| WPG: frozen periodic pattern + phase accumulator + freq modulation | **NO** — closest: semantic-domain LinOSS NPC oscillator (personality-mood trajectories, 5-dim affect basis) + `phase_separation` (measures inter-phase distance, generates no pattern) + `kinematics` Sched (closed-form motion scheduling) | YES — O(1) table readout, zero-alloc | Fusion idea, novelty TBD → riir-ai #935 |
| CPG + bounded residual control (small-deviation) | NO (one research-note mention, 014) | YES — sigmoid-bounded projection | Rides with #935 |
| Offline Jacobian decoder (`B⁺·Δcmd` near baseline) | NO | YES (finite differences over closed-form system) | Principle recorded (§4); trim-linearization is classic aerospace prior art |
| Frozen low-level + new high-level split | **Pattern ships** (freeze/thaw envelope + atomic adapter hot-swap) | YES (already the shipped pattern) | Covered architecturally; no action |
| QS fluid kernel (ellipsoid polynomial forces) | NO | YES | Not actionable — no consumer; Sane & Dickinson prior art |
| Adhesion / friction-cone margin | NO | YES — closed-form contact algebra; margin→fear affect coupling is a neat fit for the force→emotion bridge | Reopen trigger inside #935 |
| Gait invariants (≥3 stance, phase bands) | NO | YES — spec predicates / replay validators | Reopen trigger inside #935 |
| Regularized IK (DLS) | NO | YES (solver ≠ trainer) | Not actionable — no articulated body |
| Floquet monodromy certificate | NO | YES (offline) | Not actionable — no periodic physical system |
| Optic-flow autopilot (τ = z/ż, ventral flow) | NO | YES (closed-form insect vision) | Not actionable — no flying-entity vision consumer |
| Walking policy (286→59 dim) | NO | **NO** — genuinely learned | Conceded; blocked harder by *no physics sim* than by gradients |
| CNN vision stack | NO | NO | Conceded; out of scope |
| DMPO / imitation training loop | GRPO/DPO exist (G-Zero); trajectory pipeline exists (civ, closed negative) | NO | **One real mapping** → riir-train #535 |

## 4. Recorded non-actionables (reopen triggers)

- **QS fluid kernel** — reopen if a flying/swimming entity physics surface ever lands (would be a katgpt-rs closed-form kernel: 5 polynomial force laws on ellipsoids, offline-fit coefficients, BLAKE3-frozen).
- **Adhesion/friction-cone margin** — reopen with #935's consumer; fear-from-low-margin is a one-line force→affect bridge.
- **Gait invariants** — reopen with #935's consumer as deterministic-replay validators (the ≥3-stance-legs invariant is a clean spec predicate).
- **DLS-IK pose lifting** — reopen if an articulated/retarget surface lands.
- **Floquet certificate** — reopen if a periodic physical limit cycle ever needs an offline stability proof artifact.
- **Optic-flow autopilot** — reopen if flying NPCs with vision land.

## 5. Signal-diff checks performed (§3.6 discipline)

- **LinOSS NPC oscillator vs WPG**: the shipped oscillator evolves *semantic* state (5-dim affect basis [aggressive/passive/curious/fearful/social]) via a 2nd-order ODE with player forcing — it consumes interactions, produces personality-mood trajectories. The WPG is a *motor-domain* phase-indexed lookup table producing periodic command patterns with additive residual. Different signal class, different mechanism → **not covered**; but also no motor-pattern consumer exists today.
- **`phase_separation` vs gait inter-leg phase relations**: both operate on phases, but `phase_separation` *measures* minimum circular distance between entity phases (Lonely-Runner coverage guarantee) while gait coordination *generates and maintains* phase offsets. Complementary, not substitutable — a fusion candidate inside #935 (crowd cadence = inter-agent phase coordination).
- **Freeze/thaw + adapter hot-swap vs frozen-low-level reuse**: same architectural pattern (frozen base, swappable head) — genuinely covered at the pattern level.

## 6. Per-track verdicts (one verdict per track)

- **Track (a) modelless inference**: *fusion idea, novelty TBD* → riir-ai Issue 935. Not a plan: no consumer surface (no animation/embodiment roadmap), published prior art exists at class level (CPG-RL, motion matching), Q2/Q3 of the Super-GOAT gate fail. The fusion (frozen gait tables × phase accumulator × sigmoid-bounded residual × `phase_separation` crowd cadence × freeze/thaw pattern libraries × emotion→modulation bridge) is recorded with explicit reopen triggers.
- **Track (b) self-adaptive runtime**: covered by the same issue — the emotion→forcing bridge pattern (already shipped for the semantic oscillator) inverts into emotion→residual-modulation on a motor pattern.
- **Track (c) model-based**: *one actionable recipe* → riir-train Issue 535 — residual-over-frozen-heuristic re-attack of the closed civ G5' gate (the flybody "WPG-as-init + deviation penalty" recipe applied to our closed negative). 2–6 GPU-hrs + CPU rollouts; no-regression-by-construction; honest domain-ceiling blocker documented (a clean second NEGATIVE is an acceptable outcome).

### Auditable discard reasons (adversarial-panel merges)

- **DMPO-for-G-Zero** — discarded: no observed GRPO failure that DMPO demonstrably fixes; 20–60 GPU-hrs for a statistically powered A/B; acme/reverb absent. Class-level speculation.
- **Regularized-IK dataset lifting for civ traces** — discarded: NPC states are fully observed; the underdetermined 2D→3D lift problem does not exist here. The paper's IK solves a problem we don't have.
- **Mirror augmentation** — not discarded, recorded as a cheap rider inside #535 (verify L/R symmetry holds first).
- **Termination semantics (time-limit bootstrap vs failure-zeroing)** — recorded inside #535 as a pre-run audit item (verify the civ env actually truncates; otherwise inert).
- **QS kernel as open primitive** — discarded *today*: no consumer + canonical prior art; reopen trigger in §4.

## 7. MOAT gate

- katgpt-rs: nothing planned (no consumer for a standalone primitive; the semantic oscillator + phase_separation already cover the live surfaces).
- riir-ai: issue-only (935) — pillar-level? No: it would connect Frame-Sampling Bridge + Reasoning Pack + neuron-db freeze/thaw *if* an embodied surface ever existed. Correctly parked.
- riir-train: issue-only (535) — active-moat training-method candidate, correctly parked behind owner priority (4090/prefill work first; riir-train lowest priority).

## 8. Honest caveats

- No physics simulation exists anywhere in the workspace (verified: zero physics-engine deps across 13 repos). Building embodied simulation would be a platform bet, not a recipe landing — out of scope today (panel row 10, unanimous).
- The panel's strongest modelless claim (offline Jacobian decoder) is exactly the aerospace trim-linearization pattern — *recorded as principle*, not claimed as novelty.
- riir-ai's sibling-agent WIP (`riir-poc`) was untouched; only this note's own files were committed.

## Citation

```bibtex
@article{vaxenburg2025flybody,
  title={Whole-body physics simulation of fruit fly locomotion},
  author={Vaxenburg, Roman and Siwanowicz, Igor and Merel, Josh and Robie, Alice A and Morrow, Carmen and Novati, Guido and Stefanidi, Zinovia and Both, Gert-Jan and Card, Gwyneth M and Reiser, Michael B and Botvinick, Matthew M and Branson, Kristin M and Tassa, Yuval and Turaga, Srinivas C},
  journal={Nature},
  volume={643},
  pages={1312--1320},
  year={2025},
  doi={10.1038/s41586-025-09029-4}
}
```
