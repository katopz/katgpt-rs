# Plan 134: MGR Stability Proof — Validate delta_routing Convex Combination

> **Source:** Research 095 — MGR Multi-Gat Residuals (arXiv:2605.23259)
> **Status:** ✅ Complete (T1–T3 ✓, GOAT norm stability proof passing)
> **Priority:** Low — Documentation + GOAT proof enhancement
> **GOAT Pillar:** ❌ Not a pillar — see [MMO GOAT Pillars Decision Matrix](../../riir-ai/.docs/27_mmo_goat_pillars_decision_matrix.md). General transformer architecture, not game-specific. Stays in `katgpt-rs` domain.
> **Domain:** `katgpt-rs` — no game IP, no secret, no selling point. The multi-stream residual idea is public knowledge from the paper.

## Context

MGR paper §3.2 proves that **convex-combination residual updates** (β-gated lerp) guarantee bounded activation norms through arbitrary depth. Our existing `delta_routing` (Plan 097) uses additive routing via `depth_route()`, which is a **softmax-weighted sum** added to the residual — structurally similar to AttnPool but without the explicit lerp gate.

The key question: **Does our `depth_route` already satisfy the convex-combination stability guarantee?**

Answer: **Yes, partially.** Our routing is additive (`residual += weighted_sum`), not a convex combination. The stability proof from MGR applies to the **lerp gate** (Eq. 5), not to the AttnPool aggregator. Our system gains stability from:
1. RMSNorm before routing (bounds input scale)
2. Softmax normalization (weights sum to 1)
3. But the **addition** to residual means norms can still grow — unlike MGR's convex lerp

## Tasks

### T1: Document Stability Analysis in `depth_route` ✏️
- [x] Add doc comment to `depth_route()` referencing MGR §3.2
- [x] Note: our additive routing is **not** a convex combination (it's residual + weighted_sum), so the per-layer norm ceiling does NOT formally apply
- [x] Note: practical stability comes from RMSNorm + softmax normalization, not from convex-combination guarantee
- **Files:** `src/transformer.rs` (doc comments on `depth_route` at ~L964)
- **No new code** — documentation only

### T2: GOAT Proof — Verify `depth_route` Norm Stability 🐐
- [x] Add a GOAT test that verifies activation norm doesn't grow unboundedly over 36+ layers with `depth_route` enabled
- [x] Test: forward pass through all layers, check `‖x_L‖ ≤ C × ‖x_0‖` for some reasonable constant C (e.g., C < 10)
- [x] This is an **empirical** stability check, not a formal proof (unlike MGR's theoretical guarantee)
- **Files:** New test in `src/transformer.rs`
- **Feature gate:** Uses existing `delta_routing` — no new gate needed

### T3: Record Bias Initialization Formula (Eq. 14) 📐
- [x] Add Eq. 14 as a comment in `depth_route` or `TransformerWeights::new`
- [x] Useful if/when we add training infrastructure
- **Files:** `src/transformer.rs` — comment only

## What We're NOT Doing

- ❌ Multi-stream residual topology (training architecture, n× memory)
- ❌ Gated interpolation lerp mixer (training-only, requires weight format change)
- ❌ Competitive vs independent gate variants (training ablation)
- ❌ Fallback inversion for backward pass (training-only)
- ❌ New feature gate (everything uses existing `delta_routing`)

## Why No Feature Gate

Our `delta_routing` already captures the inference-time subset of MGR (AttnPool-style depth routing). The novel parts of MGR (multi-stream residuals, gated lerp) are training-time architecture changes that would require:
1. n× stream memory per layer
2. Per-stream gating weights in `TransformerWeights`
3. Per-layer lerp gate bias parameters
4. Changes to weight checkpoint format

These belong in a training framework (e.g., `riir-ai/riir-engine`), not in our inference codebase. If implemented in the future, the feature gate would be `mgr_streams` in `riir-engine`.

## Domain Decision — katgpt-rs vs riir-ai

Per [MMO GOAT Pillars Decision Matrix](../../riir-ai/.docs/27_mmo_goat_pillars_decision_matrix.md), MGR is **NOT a GOAT pillar**:

| Criterion | Score | Reason |
|-----------|-------|--------|
| MMO-product | ❌ | General transformer architecture, no game contribution |
| LoRA-independent | ❌ | Requires model weights |
| Defensible | ❌ | Public paper, straightforward algorithm |
| Secret coverage | ❌ | No secret protected |

**Why this stays in `katgpt-rs` (MIT open):**
- The convex-combination stability proof (§3.2) is a mathematical result, not IP
- Our `depth_route` implementation is already in the open codebase
- No game-specific tuning, no private domain knowledge

**When it would move to `riir-ai` (private):**
- If multi-stream residuals were applied to **game-specific** reasoning (e.g., separate streams for spatial tactics, resource planning, and opponent modeling in Bomber/Go/FFT)
- If the stream gating biases encode **accumulated game knowledge** from self-play
- If the AttnPool query vectors are **tuned per game domain** via Fourier frequencies or similar game-specific features

The boundary follows the same rule as the decision matrix: `katgpt-rs` ships the generic framework (plug sockets), `riir-ai` ships game-specific implementations (plugs).

## GOAT Proof Target

| Proof | Description | Expected |
|---|---|---|
| Norm stability | `‖x_L‖ ≤ 10 × ‖x_0‖` for 36-layer forward | ✅ Pass (empirical) |
| Routing sharpness | `max_weight ≥ 0.4` in deep layers (existing Plan 097 T8) | ✅ Already passing |

## Time Estimate

- T1 (docs): 15 min
- T2 (GOAT test): 30 min
- T3 (formula comment): 5 min
- **Total: ~50 min** — documentation/validation only
