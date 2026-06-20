# Issue 002: Sigmoid Parallax diverges to NaN under naive FD-SGD (W_R positive feedback)

**Filed:** 2026-06-18
**Re-investigated:** 2026-06-19 (findings inverted — see below)
**Source:** `.benchmarks/058_funcattn_goat.md` G2 Results "Caveat 3"
**Plan:** [135_parallax_attn](../.plans/) (historical; current issue is post-shipping)
**Status:** OPEN. Original sigmoid divergence no longer reproduces; softmax now diverges instead. Root cause of the inversion not diagnosed. See T6.

---

## Re-investigation finding (2026-06-19)

Running the exact Issue 002 setup against the current
`tiled_attention_parallax_forward` produces the **opposite** of the original
report:

| Variant | Original (2026-06-18) | Current (2026-06-19) |
|---------|------------------------|----------------------|
| Sigmoid Parallax, no mitigation | diverges @ step 350–375 | **stable through step 500** (mse 0.382 → 0.354) |
| Softmax Parallax, no mitigation | stable past step 500 | **diverges @ step 333** (NaN propagates from there) |
| Sigmoid Parallax + W_R clip ‖∇‖≤1 | (not tested) | stable through step 500 (mse 0.382 → 0.362) |

The softmax divergence step (~333) sits inside the range originally
reported for sigmoid (350–375), so the divergence *regime* looks the same
— only the activation that triggers it has flipped.

Test file pinning the current behavior as regression anchors:
`tests/parallax_sigmoid_stability_grad_clip.rs` (3 tests, all PASS in
release). Run with:

```bash
cargo test --features parallax_attn --release \
  --test parallax_sigmoid_stability_grad_clip -- --nocapture --test-threads=1
```

**Candidate explanation (not verified):** softmax's `exp` saturates faster
than sigmoid's bounded output as attention pre-activations grow late in
training, so softmax's sharper normalization amplifies numerical
instability rather than suppressing it once the W_R correction feeds back
into the scores. This is the opposite of what the original Research 140
analysis predicted. **Verifying this is T6.**

The W_R gradient clip (T3b) is now a defensive measure rather than a
required mitigation for sigmoid: it costs nothing when the gradient is
small, and it caps the feedback loop's per-step amplification for whichever
activation ends up unstable.

---

## Problem (original report — sigmoid claim now falsified)

When sigmoid-basis `tiled_attention_parallax_forward` is trained via naive
finite-difference SGD with `LR=1.0`, the `W_R` correction path was
*originally reported* to diverge to NaN around step 350–375 (after
starting from a stable descent at MSE 0.163 in step 350).

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

- [x] T1: Reproduce in isolation — `tests/parallax_sigmoid_stability_grad_clip.rs::t1_sigmoid_parallax_stays_stable_unclipped`.
      **Finding inverted:** sigmoid stays stable through 500 steps
      (mse 0.382 → 0.354). The original divergence no longer reproduces.
      Test now pins current behavior as a regression anchor.
- [x] T2: Softmax Parallax control — `t2_softmax_parallax_diverges_unclipped`.
      **Finding inverted:** softmax now diverges at step 333 (NaN propagates).
      Sigmoid is the stable variant, softmax is the diverging one.
- [ ] T3: Try mitigations in order of preference:
  - [ ] T3a: W_R weight decay (e.g. WD=0.01 AdamW-style decoupled).
  - [x] T3b: Gradient clipping on W_R (`‖∇_W_R‖₂ ≤ 1.0`) —
        `t3b_sigmoid_parallax_stabilized_by_wr_grad_clip`. Stable through
        500 steps (mse 0.382 → 0.362). Note: sigmoid was already stable
        without the clip in the current code; the clip is now a defensive
        measure, not a required mitigation. The clip has NOT been verified
        to rescue softmax (T7).
  - [ ] T3c: LR annealing (halve LR every 100 steps).
  - [ ] T3d: `gate_scale` annealing (start at 0.0, ramp to 1.0 over 200 steps).
- [x] T4: Document the chosen mitigation in `crates/katgpt-core/src/parallax_attn.rs`
      module doc as a caller requirement. ✅ added "Training-time caller
      requirement: W_R gradient clipping (Issue 002)" section. The doc
      notes both activations' current behavior and recommends W_R clipping
      when training W_R with either activation.
- [ ] T5: If no mitigation is found that's competitive with softmax Parallax,
      escalate as a research question (is sigmoid Parallax actually viable
      for end-to-end training?). **Now flipped:** sigmoid is the stable one
      in the current code; softmax is the open question.
- [ ] T6 (NEW): Diagnose why the sigmoid/softmax divergence pattern has
      inverted vs the 2026-06-18 report. Bisect `parallax_attn.rs` history
      to find the commit that flipped the behavior. Candidates: a change
      to normalization (sigmoid vs softmax ordering), a change to
      `column-sum` factorization, a change to the W_R probe application
      point. Without T6 the issue stays OPEN — the inversion is a signal
      that something changed and we don't yet understand what.
- [ ] T7 (NEW): Verify whether W_R gradient clipping (T3b mitigation)
      rescues softmax Parallax (which now diverges at step 333). If yes,
      recommend W_R clipping as a universal caller requirement. If no,
      softmax Parallax may need to be demoted to inference-only (frozen
      W_R), per AGENTS.md "demote loser" rule.

## Acceptance criteria

- `cargo test --features parallax_attn --release --test parallax_sigmoid_stability_grad_clip`
  → 3/3 PASS (regression anchors for current behavior). ✅ measured
  2026-06-19.
- The W_R gradient clip mitigation is documented in the parallax_attn
  module doc. ✅
- **Revised:** sigmoid Parallax reaches a finite MSE after 500 steps at
  LR=1.0 (✅ measured: 0.354). Softmax Parallax does NOT — it diverges at
  step 333. The acceptance criterion has flipped along with the empirical
  finding; the original sigmoid requirement is now trivially met.

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
