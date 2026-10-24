# Canonical Intent Space — Plug-and-Play Adapter Substrate (katgpt-canon)

> **Status:** opt-in (`canon`, `canon_subspace`, `canon_mask` features, default-off).
> The cross-arch Super-GOAT headline was **PERMANENTLY DEMOTED** (Bench 427, 2026-07-27)
> — see `negative_results.md` §15. The intra-arch + substrate GOAT is real (Bench 562,
> 17/17 PASS).

## What this is

`katgpt-canon` is a standalone crate (publishable to crates.io, MIT) that ships the
vocabulary for **canonical intent directions** — steering vectors that live in a
model-independent canonical space, projected into a specific base model's latent
space by a `ModelAdapter` at apply time. The crate depends on `katgpt-core`
(for SVD) and `katgpt-spectral` (for Procrustes) — both in-tree.

The three adapters ship behind independent feature gates:

| Adapter | Feature | Math | Hot-path use case |
|---|---|---|---|
| `ProcrustesAdapter` | `canon` | Orthogonal rotation `R` from `orthogonal_procrustes`; `project_into = R·h` | Same-arch snapshot swap (setup-time; **1.3ms at d=2304**, NOT gated against 50µs) |
| `SubspaceAdapter` | `canon_subspace` | Joint SVD `M=[A\|B] = UΣV^T`, top-k right singular vectors = shared subspace; `project_into = V_k^T·h` | Cross-arch steering (per-direction per-tick; **417ns at k=4, d_b=1536** ≤ 50µs) |
| `MaskAdapter` | `canon_mask` | Elementwise `mask ⊙ h` (lottery-ticket *apply*, not discovery) | Sparse-prune application (**1.38µs at d=2304** ≤ 50µs) |

## The cross-arch Super-GOAT claim and why it was permanently demoted

Proposal 009's headline was **"plug-and-play any base model"** — mine a steering
direction on Gemma2-2B, re-apply it on MiniCPM5-1B or Llama, no retraining. The
make-or-break gate was G6: the canonical direction discriminates Rust-idiom vs
non-Rust across architectures at ≥0.5 mean cosine agreement (floor: a good system
prompt, per Research 322 "Report the Floor" rule).

Four hidden-state construction methods failed G6:

| Phase | Method | Best cross-arch agreement | Why it failed |
|---|---|---|---|
| P3 (Bench 424) | Per-model centroid + Procrustes | **−0.33** | Procrustes aligns shape, not location; centroids point opposite ways |
| P3 (Bench 424) | Difference-of-means `d_diff` | +0.46 (borderline) | JS discrimination negative — apparent signal was a token-count confound |
| P3b (Bench 425) | Intermediate-layer probe | +0.19 (layer 0 best) | Git Re-Basin contradicted — layer 0 captures surface features, not semantic idiom |
| P3c (Bench 426) | Length-detrended `d_diff` | **−0.15** | Length detrending REVERSES Python discrimination — signal was prompt length |
| Recipe D (Bench 427) | Length-matched corpus, k-sweep | **+0.009** | Length-matching works but cross-arch agreement is structurally ~0 — the failure is cross-arch disagreement, not length, not noise |

**Recipe E (gradient-descent stitching) was NOT opened** — the failure pattern
(structural cross-arch disagreement, not non-linearity) rules it out. The
modelless path is **declared exhausted**.

## What still ships (the intra-arch + substrate GOAT, Bench 562)

Despite the cross-arch demotion, all three adapters carry a measured G1/G2/G4 GOAT
stamp on the paths that remain load-bearing:

| Gate | ProcrustesAdapter | SubspaceAdapter | MaskAdapter |
|---|---|---|---|
| **G1** correctness | residual 0.0000%, round-trip 4.47e-8, BLAKE3-deterministic | fit shapes ✓, no-NaN, held-out mean cos 0.257 (frac positive 0.78) | all-ones identity ✓, half-zero preserve ✓, BLAKE3-deterministic |
| **G2** perf | d=256 **16.17µs** ≤ 50µs (post-SIMD, was 29µs) | k=4 d_b=1536 **417ns** ≤ 50µs | d=2304 **1.38µs** ≤ 50µs |
| **G4** alloc-free | 0 allocs / 1000 hot-path calls | 0 allocs / 1000 hot-path calls | 0 allocs / 1000 hot-path calls |

**⛔ REFUTED 2026-10-24 (riir-train Issue 644 T4, riir-train `0a290e94`): on the routing-fixed Gemma-2 forward Bench 423 reads k=2 +0.62, k=4 +0.56 (AMBIGUOUS), k=16 +0.48 (NO-GO); a reverted-fix control reproduces the +0.82 exactly — the G5 GO was the delta-routing bug. No cross-arch measurement supports SubspaceAdapter.** Prior record: **SubspaceAdapter carries the load-bearing Bench 423 G5 GO at k∈{2,4}** — mean
cosine +0.87/+0.75 on Gemma2-2B ↔ MiniCPM5-1B real weights, held-out prompts. The
cross-arch shared-subspace ALIGNMENT is genuinely real; the cross-arch DIRECTION
claim is what failed. Replacing the P2 random projection with joint SVD recovered
the cross-arch path modellessly (Recipe C trained stitching no longer a blocker).

## The d=2304 Procrustes honest scaling limitation

`ProcrustesAdapter::project_into` at production model dim (d=2304, Gemma2-2B) is
**1.328ms post-SIMD-optimization** (commit `e5efd20e`, 8-wide FMA accumulator
pattern). This is O(d²) scaling and is **NOT gated against the 50µs target**.

The theoretical SIMD floor at d=2304 is ~220µs (5.3M flops / 8-wide AVX2 FMA /
3 GHz). Even with perfect SIMD, 50µs is physically unachievable at d=2304 on
commodity hardware.

**Why this is acceptable:** the 50µs G2 floor applies to the per-direction-per-tick
hot path — that's SubspaceAdapter (O(d·k), k≪d) and MaskAdapter (O(d)).
ProcrustesAdapter's use case is same-arch snapshot swap, a setup-time operation
(not per-token), where 1.3ms is acceptable. The d=256 gate covers the low-rank
same-arch steering case (adapter fit in a 256-dim subspace of a larger model).

Further gains would require BLAS-backed matvec (ndarray+openblas) — not added per
the "prefer existing dependencies" rule.

## Why opt-in despite GOAT PASS

Promotion to default-on would require a new proposal re-arguing the substrate's
value proposition post-demotion. The cross-arch Super-GOAT headline is gone; the
substrate is useful (intra-arch snapshot swap, cross-arch alignment preservation,
lottery-ticket application) but narrower than originally sold.

## What reopens the cross-arch claim

Only a **non-hidden-state construction** (AST node-type histogram, Clippy
lint-category fingerprint, ownership/borrow graph topology) — features that are
identical regardless of which model processes the source code. Sketched in
[Proposal 010](../../.proposals/010_non_hidden_state_canonical_construction.md)
(draft, HIGHLY SPECULATIVE — the hidden-state approach had a plausible mechanism;
the source-feature approach has no such guarantee).

## See also

- [`.docs/09_feature_catalog/opt_in_features.md` §25](../09_feature_catalog/opt_in_features.md) — feature-flag detail
- [`.docs/09_feature_catalog/negative_results.md` §15](../09_feature_catalog/negative_results.md) — cross-arch permanent demotion
- [`poincare_navigator.md`](poincare_navigator.md) — sibling closed-form navigation primitive (DEFAULT-ON)
- [`tilr_subspace_family.md`](tilr_subspace_family.md) — sibling subspace-projection family
