# Research 550: Metabolic Gate — Energy-Coupled Compute Gating

> **Source:** "Tapes Together Strong: The Co-evolution of Computation and Cooperation" [arXiv:2609.10817](https://arxiv.org/abs/2609.10817) — Jha, Cicala, Agüera y Arcas, Richards, Jaques, Kleiman-Weiner, Niklasson, 2026-09-09
> **Date:** 2026-09-11
> **Status:** Active — primitive filed, awaiting implementation (consumed by riir-ai Plan 585)
> **Related Research:** 363 (state-dependent compute budget — closest shipped cousin), 167→Plan 187 (WealthBanditPruner — resource-coupled selection)
> **Related Plans:** riir-ai `.plans/585` (consumer); katgpt-core implementation rides its Phase 1
> **Classification:** Public

---

## TL;DR

The paper couples an agent's **computation capacity to its available energy**: every op costs energy, and execution share = energy share. Cooperation then self-stabilizes WITHOUT memory, assortment, or punishment because **lossy theft destroys the energy a defector needs to finish its own computation** (Theorem 1). Distilled for katgpt-core: a family of zero-alloc, f32, sigmoid-gated primitives — **stock-gated compute-tier selection**, the **closed-form starvation design law** (`2ε < L(1+(1−α)δ)` with the `K(ε,L)` break-even solver), **lossy transfer**, and **energy-share scheduling** — a sibling of `gain_cost_halt.rs` in the value-of-computation family, but gating on a metabolic **stock** instead of a utility flow.

**Distilled for katgpt-rs (modelless, inference-time):**
- `depth = σ((stock − E_base)/E_scale)` → tier/depth multiplier (sigmoid, never softmax)
- `starvation_bound(eps, repl_cost, alpha, delta) -> bool` — `2ε < L(1+(1−α)δ)`
- `metabolic_drag_threshold(eps, repl_cost) -> f32` — solves `ln(2ε/(2ε−L(1+K))) = (1+K)·ln(ε/(ε−L))` (bisection, ~20 iters, no alloc)
- `steal_lossy(&mut thief, &mut victim, delta, alpha)` — destroys `(1−α)δ` from the system
- `execution_share(e_i, e_j) -> f32` — `E_i/(E_i+E_j)` lottery-share scheduling

## 1. Paper core findings (math only — no training anywhere)

- Execution kinetics `dt = 1/(E_i+E_j)`, `p_i = E_i/(E_i+E_j)`; thermodynamic grounding: Margolus-Levitin `Δt ≥ h/4E`, Landauer, DVFS.
- **Lemma 1 (endogenous metabolic drag):** with per-op system drain `δ(α−1)−1 ≤ −1`, a defector whose inefficiency `(1−α)δ` exceeds the threshold `K(ε,L)` is *strictly slower* than mutual cooperators even in its best case.
- **Lemma 2 + Theorem 1 (starvation limit):** `2ε < L(1+(1−α)δ)` ⇒ defector exhausts the pool before completing L writes ⇒ cooperation locally favored in well-mixed, memoryless populations.
- Timing result: replication priority from **pre-interaction** energy favors cooperation; post-interaction favors defection. (Design rule for any scheduler that interleaves resource accounting with interaction.)
- Scarcity⇄computation: free-energy abundance *negatively* correlates with task solving — rewards must be earned against a budget to incentivize compute.
- SHARE (α=1) + eligibility cap ⇒ reciprocal transfer survives exactly when it raises system energy.

External prior-art: substrate is Tierra/Avida lineage; spatial assortment is textbook (Nowak-May); Avida's task→CPU bonus anticipates tasks-for-compute. The defensible novelty = the energy↔computation coupling as the game-theoretic payoff axis + the starvation law. Our primitive ships the LAW, not the ALife substrate.

## 2. Distillation

**Proposed primitive:** `katgpt-core/src/metabolic_gate.rs` behind feature `metabolic_gate` (opt-in, GOAT-gated before any default promotion).

```rust
pub struct MetabolicGate { e_base: f32, e_scale: f32 }
impl MetabolicGate {
    /// σ((stock − base)/scale) ∈ (0,1) — compute-depth multiplier from an energy stock.
    pub fn depth_factor(&self, energy_stock: f32) -> f32;
}
/// Paper Theorem 1: cooperation/defector-starvation regime check.
pub fn defector_starves(eps: f32, repl_cost: f32, alpha: f32, delta: f32) -> bool;
/// Break-even inefficiency K(ε,L) — continuous approximation, bisection solve.
pub fn metabolic_drag_threshold(eps: f32, repl_cost: f32) -> f32;
/// Lossy transfer: thief +αδ, victim −δ (system loses (1−α)δ). Returns destroyed amount.
pub fn steal_lossy(thief: &mut f32, victim: &mut f32, delta: f32, alpha: f32) -> f32;
/// Energy-share scheduling probability (lottery share over a metabolic stock).
pub fn execution_share(e_i: f32, e_j: f32) -> f32;
```

Zero-alloc, const-friendly, `#[inline]`, f32. Tests: monotonicity of `depth_factor`; starvation-bound boundary cases (α→1 recovers zero-sum; δ=0 ⇒ `2ε < L` false when viable); `K(ε,L)` solver vs the transcendental equation at pinned tolerances; `steal_lossy` conservation identity (`thief_gain + destroyed == victim_loss`); `execution_share` symmetry + zero-zero guard (paper C.1: both-zero → ½).

**Slot in the per-stack ledger:** compute-budgeting family beside `gain_cost_halt` (value-of-computation halting) — this adds the **stock axis** (resource inventory) where gain_cost_halt uses a **utility flow**. Signal-diff: gain_cost_halt consumes (gain, cost) per decision; metabolic_gate consumes a persistent stock + design-law constants. Distinct, complementary.

## 3. Fusion

The private consumer (riir-ai Plan 585) composes the gate with thermal tiering and swarm foraging; the open primitive stays game-agnostic (any bounded-agency scheduler: budgeted agent loops, rate-limited workers, auction-priced compute). Cross-repo fusion candidates recorded there. Secondary distill from the same paper (recorded, unplanned): **MAP trace embeddings** — `u⊛v = u⊙ρ(v)` (circular-shift + Hadamard binding) n-gram sums with RoPE → fixed-size trace vectors; VSA-adjacent, candidate for trajectory clustering if ever needed.

## 4. Verdict

**Gain** (open primitive; filed, not yet implemented). GOAT gate for promotion: G1 correctness (unit identities above), G2 latency (sub-ns gate; the K-solver budgeted ~20 bisection iters), G3 no-regression (feature-off default), G4 zero-alloc bench. No UQ surface — no conformal floor required. Per-stack ledger row: *compute budgeting / metabolic stock axis — opt-in `metabolic_gate`, consumer riir-ai Plan 585.*

> **PASS-Redirects (synthesis):** Jha et al. [arXiv:2609.10817 "Tapes Together Strong: The Co-evolution of Computation and Cooperation"] — validates the stock-gated compute-budget family this note extends; closest shipped cousin is `gain_cost_halt` (Research 363's six-implementation index gains a seventh consumer slot via riir-ai Plan 585).
