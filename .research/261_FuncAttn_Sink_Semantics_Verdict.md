# Research 261: FuncAttn Sink-Semantics Verdict — Negative Result

> **Question:** Can the sink-aware dual-policy gate (Plan 287, Research 258) be wired into FuncAttn's forward path the same way Plan 289 wired it into Parallax?
> **Date:** 2026-06-18
> **Status:** Closed — negative result. Sink-aware attention does **not** apply to FuncAttn's basis-mode structure.
> **Trigger:** Plan 289 §A2 deferred FuncAttn wiring with "sink classification semantics on basis-modes need separate research." This note is that research.
> **Related Research:** 257 (Functional Attention), 258 (Attention Sinks — NOP vs Broadcast)
> **Related Plans:** 286 (FUNCATTN open primitive), 287 (sink-aware classifier), 289 (forward-path wiring — Parallax only)
> **Classification:** Public

---

## TL;DR

Sink-aware attention (Plan 287/289) targets the `(n, n)` softmax attention matrix — it scans columns of `A = softmax(QK^T/√d)` for high-mass sinks, then applies a per-head NOP/Broadcast gate. **FuncAttn has no `(n, n)` attention matrix.** Its forward path is `out = Φ · C · Ṽ` where `Φ ∈ R^{n×k}` (partition-of-unity basis, k≪n), `C ∈ R^{k×k}` (closed-form Tikhonov-solved operator), and `Ṽ ∈ R^{k×d}` (sliced values). The closest structural analog of an "attention column" is a basis mode `Φ[:, g]` for `g ∈ [0, k)` — but basis modes are *designed* to be diffuse (that is the entire premise of functional correspondence), so the sink classifier would either find no sinks (correctly, by design) or flag degenerate training (a column-norm check would catch the same thing cheaper).

**Verdict: do not wire.** Plan 289 §A2's deferral becomes a closure: the deferral was for "research"; the research says "not applicable." Documented here so future sessions don't re-investigate.

---

## 1. Why sinks exist in softmax attention

A sink is an emergent pathology of softmax over a pairwise token affinity matrix. Given `A = softmax(QK^T/√d)` ∈ `R^{n×n}`:

- Softmax **must** allocate probability mass per row (rows sum to 1).
- When query `i` has no semantically relevant key, softmax still has to put its mass somewhere — it piles onto whichever key `j` has the highest `q_i · k_j` even if that dot product is negative. The result is a **vertical stripe**: column `j` receives disproportionate mass from many rows.
- Two algorithmic regimes emerge (Research 258 §1.1):
  - **Adaptive NOP** — sink value `‖v_j‖ ≈ 0`, so the mass piles onto a "trash can" token that contributes nothing. Residual update ≈ 0.
  - **Broadcast** — sink value has content, the mass writes a rank-1 update `O ≈ a_j v_j^T` to many positions.

Both are *symptoms of softmax*. The fix is either to gate away the mass (NOP case) or give the broadcaster a dedicated workspace token (register tokens).

---

## 2. Why FuncAttn structurally lacks sinks

FuncAttn (Research 257, paper arxiv 2605.31559) replaces softmax-over-n with a partition-of-unity-over-k:

```
Φ = row_normalize(activation(Linear(X) / τ))      ∈ R^{n × k}
slice_token[g] = (Σ_n Φ[n,g] · x_value[n]) / col_sum[g]   ∈ R^{k × d}
Q̃, K̃, Ṽ = slice_token · w_{q,k,v}^T              ∈ R^{k × d}
C = Q̃ · ((1-α)·K̃ᵀK̃ + α·I_d)^{-1} · K̃ᵀ          ∈ R^{k × k}   (closed-form Tikhonov)
out = Φ · C · Ṽ                                    ∈ R^{n × d}
```

The "attention" is the `k×k` operator `C`, not an `n×n` matrix. Each row of `Φ` is a partition-of-unity over basis modes by construction (`Σ_g Φ[n,g] = 1`, non-negative).

### 2.1 The structural analogs and why they don't carry sink semantics

| Softmax concept | FuncAttn analog | Why the analogy breaks |
|---|---|---|
| Attention matrix `A ∈ R^{n×n}` | `Φ · C ∈ R^{n×k}` (synthetic) | Not square; not the operator C itself. Synthesizing it costs `O(nk)` and defeats the linear-in-n property. |
| Attention column `A[:, j]` | Basis column `Φ[:, g]` | Basis modes are partition-of-unity — every mode absorbs mass by design. High column-sum is the *healthy* state, not a sink. |
| Sink strength `sink(s; I) = mean_i A_is` | Column-sum `Σ_n Φ[n,g] / n` | Bounded in `[0, 1]` by partition-of-unity. The τ→0 hard-P0 limit (Prop 4.3) makes this a hard clustering indicator — still not a sink. |
| Sink value `‖v_s‖ ≈ 0` | `‖Ṽ[g,:]‖ ≈ 0` | This is a **dead basis mode** — a training failure, not a runtime pathology. Better detected by column-norm thresholding on `Ṽ` than by a stable-rank classifier. |
| Stable rank of `O = AV` | Stable rank of `Φ · C · Ṽ` | Always low-rank (`k ≤ 64` typically). Plan 287's stable-rank classifier thresholds at `≈ 1.5` — FuncAttn's output is structurally rank-`k` regardless of inputs, so the classifier has no signal. |

### 2.2 The NOP/Broadcast discrimination collapses

Research 258's classifier returns `{None, Nop, Broadcast}` based on (sink strength, value-norm ratio, stable rank). Applied to FuncAttn:

- **Nop case** would require a basis mode `g` where `Φ[:, g]` has high column-sum but `Ṽ[g, :] ≈ 0`. That is exactly the "dead basis mode" failure mode — the fix is retraining or pruning the basis dimension, not a runtime gate.
- **Broadcast case** would require `Φ[:, g]` to have high column-sum AND `Ṽ[g, :]` to have content. That is the **intended operating point** of every basis mode. Calling it a "Broadcast sink" and preserving it via the dual-policy gate is a no-op — the operator `C` already mixes all `k` modes.

There is no input-dependent regime where some basis modes should be NOP-gated and others preserved. The partition-of-unity structure means *every* mode is a broadcaster by construction; degenerate modes are a training signal.

---

## 3. What would be needed to make sinks meaningful in FuncAttn

For completeness — if a future paper identifies sink-like phenomena in functional correspondence, here is what the infrastructure would need:

1. **A definition of "basis-mode sink" that is input-dependent.** The current partition-of-unity makes all modes active for all inputs. A sink-like phenomenon would require certain modes to activate only for specific input patterns (e.g., a mode that "lights up" for outlier tokens) — that is a research hypothesis, not an observed phenomenon in the FUNCATTN paper.
2. **A value-side analog with meaningful per-token structure.** The current `Ṽ ∈ R^{k×d}` is per-mode, not per-token. A sink classifier needs per-token values to compute `‖v_s‖`. This would require re-introducing an `n×d` value path, which FuncAttn deliberately eliminates for the `O(ndk)` complexity win.
3. **Empirical evidence of stripe patterns in `Φ`.** Research 258's diagnostics were motivated by observed vertical stripes in DINOv2-G / OpenCLIP-L attention maps. No such observation exists for FuncAttn's basis matrices in the paper or our G1/G4 benchmarks.

None of these are tractable without new research. The deferral in Plan 289 §A2 was correct; the answer is "no."

---

## 4. Decision and consequences

**Decision:** Plan 289 §A2's deferred FuncAttn wiring is **closed as not-applicable**, not postponed. The forward-path composition for sink-aware attention applies to softmax-style `n×n` attention paths only (Parallax — done; standard SDPA — out of scope, no caller).

**Consequences:**
- The `SinkAwarePolicy` API (Plan 287) and the forward-path composition (Plan 289) remain Parallax-specific by design, not by deferral.
- The `apply_dual_policy_gate_flat` / `apply_dual_policy_gate_cached_flat` functions in `data_probe.rs` retain their `n×n` attention contract — no API change needed to support FuncAttn.
- FuncAttn's existing basis-level degeneracy detection (column-norm check on `Φ` or `Ṽ`) is the correct diagnostic for its failure modes. If a basis-mode degeneracy detector becomes interesting, it should be a separate `funcattn_diagnostics` module, not a reuse of the sink classifier.

**If evidence emerges** (e.g., a follow-up paper observes stripe-like patterns in functional correspondence bases on a specific domain), reopen as a new research note citing this one.
