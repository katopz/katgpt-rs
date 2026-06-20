# Issue 002: Sigmoid Parallax diverges to NaN under naive FD-SGD (W_R positive feedback)

**Filed:** 2026-06-18
**Source:** `.benchmarks/058_funcattn_goat.md` G2 Results "Caveat 3"
**Plan:** [135_parallax_attn](../.plans/) (historical; current issue is post-shipping)
**Status:** OPEN. Discovered during Plan 286 T3.2 G2 regression gate.

---

## Problem

When sigmoid-basis `tiled_attention_parallax_forward` is trained via naive
finite-difference SGD with `LR=1.0`, the `W_R` correction path diverges to
NaN around step 350–375 (after starting from a stable descent at MSE 0.163
in step 350).

Concrete trace from
`tests/funcattn_g2_funcattn_vs_parallax_vs_sdpa.rs` (release, STEPS=500):

```
[parallax] step  300/500   mse = 0.283481  rel-L2 = 0.864314
[parallax] step  325/500   mse = 0.226900  rel-L2 = 0.773263
[parallax] step  350/500   mse = 0.163051  rel-L2 = 0.655499   ← still descending
[parallax] step  375/500   mse = NaN       rel-L2 = NaN        ← sudden blowup
[parallax] step  400/500   mse = NaN       rel-L2 = NaN
```

Setup: n=64 tokens, d=8 features, sigmoid activation, `gate_scale=1.0`,
orthogonal init on W_Q, identity W_K/W_V, zero W_R (recovers plain sigmoid
attention at init). Inputs Gaussian-scaled by 0.5.

---

## Root cause (analysis)

The Parallax correction `o_PLX = o_SA − gate_scale · Σ_KV · ρ` has a
positive feedback loop when W_R is trained naively:

1. As |ρ| = |W_R · x| grows, the correction `Σ_KV · ρ` grows.
2. The loss gradient w.r.t. W_R is proportional to `(Σ_KV · x)`, which
   grows as the correction grows.
3. The W_R update amplifies ρ further, which amplifies the correction,
   which amplifies the gradient. Classic positive feedback.
4. Sigmoid normalization's softer saturation (vs softmax's sharper
   max-subpression) means attention weights near 0.5 let the covariance
   correction amplify rather than compress. Softmax Parallax at the same
   setup stays stable past step 500.
5. Once ρ magnitude exceeds the softmax/sigmoid numerical range,
   `exp(s_j)` in normalization overflows → NaN propagates.

This is **not** a bug in the shipped `tiled_attention_parallax_forward` —
the forward path is numerically stable for any finite ρ. The divergence
is a **training dynamics** issue: naive SGD on W_R is unstable without
regularization.

---

## Why softmax Parallax doesn't hit this

Softmax's sharper normalization (max-subtraction + exp) means that once
the attention pattern saturates (one weight → 1, others → 0), the
covariance `Σ_KV` becomes rank-1 with bounded magnitude, so the
correction `Σ_KV · ρ` cannot grow unboundedly. Sigmoid's softer
saturation keeps multiple weights near 0.5, leaving `Σ_KV` higher-rank
and the correction free to grow.

This is the same trade-off documented in Research 140 / Plan 161: sigmoid
has higher COR capacity but lower implicit regularization than softmax.
The capacity-regularization trade-off is the root cause of this
instability.

---

## Tasks

- [ ] T1: Reproduce in isolation — a minimal test in `tests/parallax_*.rs`
      that trains W_R from zero with sigmoid activation and documents the
      divergence step.
- [ ] T2: Verify softmax Parallax stays stable at the same setup (control).
- [ ] T3: Try mitigations in order of preference:
  - [ ] T3a: W_R weight decay (e.g. WD=0.01 AdamW-style decoupled).
  - [ ] T3b: Gradient clipping on W_R (e.g. ||∇_W_R|| ≤ 1.0).
  - [ ] T3c: LR annealing (halve LR every 100 steps).
  - [ ] T3d: `gate_scale` annealing (start at 0.0, ramp to 1.0 over 200 steps).
- [ ] T4: Document the chosen mitigation in `crates/katgpt-core/src/parallax_attn.rs`
      module doc as a caller requirement.
- [ ] T5: If no mitigation is found that's competitive with softmax Parallax,
      escalate as a research question (is sigmoid Parallax actually viable
      for end-to-end training?).

## Acceptance criteria

- `cargo test --features parallax_attn --test parallax_<mitigation>_stability`
  runs 500 FD-SGD steps on W_R without diverging.
- The chosen mitigation is documented in the parallax_attn module doc.
- Sigmoid Parallax reaches a finite MSE after 500 steps at LR=1.0 (or the
  LR is documented as needing reduction).

---

## Severity

**Medium.** The shipped forward path is correct and stable; only end-to-end
training is affected. The Plan 161 G3 result (sigmoid Parallax has higher
COR capacity than softmax on real LM data) suggests sigmoid Parallax is
worth the regularization complexity, but this issue shows the training
cost of that capacity.

Production callers using pre-trained W_R weights (the modelless inference
path) are not affected — the forward path is finite for any finite ρ.

---

## Related

- Plan 286 T3.2 (G2): `.plans/286_functional_attention_spectral_transport.md`
- Bench 058 G2: `.benchmarks/058_funcattn_goat.md` "G2 Results" Caveat 3
- Research 140: sigmoid Parallax COR capacity
- Plan 161: G3 sigmoid vs softmax Parallax on LM data
- Test: `tests/funcattn_g2_funcattn_vs_parallax_vs_sdpa.rs` (steps 350→375)
- Source: `crates/katgpt-core/src/parallax_attn.rs`
