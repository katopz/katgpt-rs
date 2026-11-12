# Research 665: e-means — EMA Mini-Batch k-means (One-Tenth the Epochs)

> **Source:** Bessa, Mazaré, Szilvasy, Touvron, Picard, Douze, Jégou (Meta FAIR), [arXiv:2611.09501](https://arxiv.org/abs/2611.09501) "e-means: outperforming k-means (in one-tenth the epochs)" (2026-11-06) · site [e-means.github.io](https://e-means.github.io) · code `facebookresearch/faiss/contrib/e_means`
> **Status:** GOAT (demote-on-fail — the PoC issue 997 runs the gates; failure demotes this grade in the same commit) — verdict via Claude ping-pong 2026-11-12, 2-round negotiation (R1 REVISE → R2 AGREE + final close), reviewer session `65c0ee85-095b-4730-b57d-2d33c37c78ad`
> **Track:** SINGLE — modelless (classical optimization math; no training loop anywhere). riir-train Plan 445 (trained VQ adapter-codebook distillation) is UNTOUCHED: its VQ-VAE codebook learning is the moving-distribution regime this paper explicitly excludes ("a static dataset, unlike codebook learning inside VQ-VAEs").

## TL;DR

e-means replaces full-batch Lloyd with **mini-batch EMA updates** on per-cluster sufficient statistics — `S_j ← (1−α)S_j + α·s_j`, `N_j ← (1−α)N_j + α·n_j`, centroid `c_j = S_j/N_j` — with the batch size `B` and initial step `α₀` set by **fitted power laws** in `(n, k, T)` (no sweep), dead clusters revived by **splitting a large cluster** sampled ∝ size (Faiss-derived, not EnCodec's batch resample), cosine anneal `α₀ → 0.1·α₀`, and a finalization stage that ends on each cluster's exact last-epoch mean. Headline: Lloyd@25-epoch loss beaten in 1.4–2.6 epochs → 10–45× GPU wall-clock vs Flash-KMeans, 8–15× CPU vs Faiss, at *lower* final loss; dead clusters 47/16384 vs 4313 (off) / 2343 (resample); Lloyd stuck 26–46% of starts on multi-modal blob pathologies vs 0–4% for e-means (mini-batch noise escapes local minima Lloyd's monotone descent cannot).

**Workspace claim (pinned, §1.5 precondition form):** EMA mini-batch updates + scaling-law-derived (B, α₀) + proportional-split revival as a deterministic (fixed-batch-schedule) upgrade path for the stack's offline k-means fits — `katgpt-core::factorized_action::fit_codebook_kmeans_into` (Plan 375) and the GMM EM seed it feeds (`gmm_support::em::fit_diag_gmm`, Plan 618) — distinguished from the shipped full-batch Lloyd + farthest-patch reinit + Jeffreys-smoothed EM by update-rule / revival-mechanism / zero-sweep-hyperparameter deltas. **No class novelty is claimed** — the mini-batch EMA class is old prior art (MacQueen 1967; Sculley 2010; Newling & Fleuret 2016; VQ-VAE EMA 2017 — the paper's own citations). First-of-kind in the public literature: the (B, α₀) scaling laws (arXiv API `"k-means"+"scaling law"` returns only this paper); Flash-KMeans (arXiv:2603.09229) is kernel co-design for EXACT Lloyd (algorithm unchanged), Qkmeans (arXiv:2602.01104) is seeding-side.

## Path 0 inventory (all components modelless-validable — no riir-train deferral)

| # | Component | Extraction | Shipped analog? | Signal-diff |
|---|---|---|---|---|
| 1 | EMA centroid update `(1−α)` recurrence | closed-form recurrence | **No centroid analog.** The only shipped EMA is the `gmm_support::gate` SIGNAL smoother (`s ← α·g + (1−α)·s`, tests.rs L340) — consumes a gate log-ratio scalar, never assignment counts. Not coverage. | run |
| 2 | Scaling laws `B = 0.035·k^0.13·n^0.88·T^0.81`, `α₀ = min(0.824·k^−0.46·B^0.90·n^−0.45·T^−0.11, 1)` | closed-form constants (fitted offline by the authors; we extract the fitted coefficients) | **None** anywhere | n/a |
| 3 | Dead-cluster revival by proportional split | deterministic given seeded sampling | Shipped cousin: **farthest-patch reinit** (codebook.rs L247–260, Issue 940 LI tie pin). Signal-diff: picks the farthest DATA POINT, never splits an existing cluster's mass. Second cousin: GMM Jeffreys smoothing (λ=0.5) — mass-padding, never re-seeds positions. | run |
| 4 | Cosine schedule `α₀ → 0.1·α₀` | closed-form | none for clustering | n/a |
| 5 | Finalization (exact last-epoch mean) | IS one Lloyd step | trivially composeable | n/a |

Three-track adversarial panel SKIPPED per the exemption: no training-loop classification anywhere (clearly classical-optimization side).

## The mechanism hypothesis (what the PoC actually tests)

**Farthest-point revival is outlier-biased; split revival seeds from dense mass.** The farthest point from any centroid is disproportionately an outlier, so a revived centroid seeds a junk cluster that must migrate back into a mode; splitting a large cluster sampled ∝ size lands inside a real cluster — the split half-centroids start near a local mode. This is sharper than "split is better" and is falsifiable head-to-head.

**Incumbent defect found en route (verified in-source, standalone fix filed in issue 997 T1):** the shipped revival recomputes `d2_nearest` ONCE per pass (codebook.rs L232–245), then the per-cluster placement loop only zeroes the chosen point (`d2_nearest[far_idx] = 0.0`, L259). When SEVERAL clusters are empty in the same pass, the second revival selects from distances that never saw the first revival's new centroid — two revived centroids can land in the same far-outlier clump. The fix helps the incumbent regardless of the split verdict.

## Consumers — one change, two surfaces (consume, never fork)

1. **`fit_codebook_kmeans_into`** (Plan 375 factorized-action codebooks; bench posture K=128, D=32, 16K patches, deterministic from seed, f64 accumulation, f32 artifact). Its G1 gate (factorized 0.0287 vs monolithic 0.1401) is a DOWNSTREAM abstraction ratio — a better fit may not move it; the primary PoC metric is the codebook's own reconstruction SSE. G1 stays as a no-regression check only.
2. **`fit_diag_gmm`** (Plan 618 support-gate plane; consumer: the decision-serving family per Research 604's consumer issue riir-reflex/.issues/066) — consumes the k-means seed directly (em.rs L100–102, `kmeans_iters: 16`). **The upgrade reaches the GMM automatically; building a parallel GMM seed path is forbidden** (consume, never fork).

**Fusion-priority ladder (two-sided check, stated honestly):** rulewright — NO k-means surface ships there (retrieval is flat KNN+BM25; `kernel_route_narrow` measured cost-negative on query count, which clustering cannot fix) → none today. Decision-serving/self-evolve — INDIRECT via the Plan-618 gate plane. Game — Plan-375 codebooks compress multi-modal NPC transition patches: exactly the paper's pathology class (Lloyd stuck 26–46% of starts on multi-modal blobs).

## Closest cousins (why they don't kill)

1. **`fit_codebook_kmeans_into`** — full-batch exact mean vs EMA; farthest-POINT reinit vs split; no hyperparameter law.
2. **`gmm_support` Jeffreys smoothing** — keeps components alive by mass-padding, never re-seeds positions; orthogonal to revival.
3. **TurboQuant** (Research 020) — Lloyd-Max is 1D scalar quantization on ANALYTIC distributions (closed-form centroids); no data-space clustering.
4. **FlashLib** (Research 130) — flags fused scoring "for future clustering ops"; kernel-side, no algorithm change.
5. **negative_results §31** (factorized_action G2b FAIL) — the recorded modelless-k-means negative is about the GATE parity (state-aware relevance needs trained FiLM), NOT fit quality (G1 passed 4.9×). This filing targets the fit, not the dead gate; reopening trigger discipline unaffected.

## Weakest points (named at verdict, carried forward)

1. **No at-scale consumer** — headline speedups live at n=10⁶–10¹⁰; the stack fits 16K patches where Lloyd converges in ≤16 iters. The speedup claim likely evaporates at stack scale; the durable claim is QUALITY (escape + revival), measured or the grade demotes.
2. **The paper never ran split-vs-farthest-reinit** (its ablation arms: split / resample / off) — "split beats what ships" is unproven until the head-to-head.
3. **Order dependence** — mini-batch EMA is batch-order-dependent; the determinism contract below is what makes it shippable.
4. **Scaling laws used outside their fitted range** — fitted at n≥10⁶; stack scale n≈16K, K=128 gives B≈hundreds (~2–4% of the corpus). The "no sweep needed" claim is explicitly NOT carried at our scale until the 5×5 (B, α₀) sanity grid passes.

## Determinism contract (Issue 940 law, fully pinned)

- Batch order: contiguous fixed-order chunks, fixed once, pinned (no unseeded shuffle).
- EMA accumulation: plain sequential f64 fold — never a SIMD/platform-lane reduction.
- Size-proportional sampling: last-index tie rule (matching the Issue 940 LI pin).
- Final codebook BLAKE3-committed; parity test asserts bit-exactness across two runs at the same seed.

## PoC + gates (execution: `.issues/997_emeans_codebook_fit_upgrade.md`)

Four arms — **shipped / fixed-farthest / split / split+EMA** — on the bench_375 patch corpus. Gates (each can fail):
- **Primary:** final SSE (sequential f64 fold) — the winning arm must beat fixed-farthest at ≤ wall-time.
- **Stuck rate:** ~50 seeds on an uneven multi-blob fixture (two very-different-mass blobs + a light middle blob); reported as COUNTS (e.g. 3/50), threshold = final SSE > 1% above best-of-arms.
- **Revival count:** DIAGNOSTIC ONLY — never a veto at unequal SSE (split may revive more often on purpose; "lower is healthier" holds only at equal SSE).
- **No-regression:** `bench_375` G1 ≥ the 4.9× margin; GMM arm: EM final-LL ≥ shipped seed on the gmm fixtures (rides the seed swap automatically).

**Demote clause:** split arms fail to beat fixed-farthest ⇒ GOAT grade struck, PoC Addendum records the raw numbers in this note in the same commit, only T1 (the defect fix) survives.


## PoC Addendum — Issue 997 verdict (2026-11-12, DEMOTED)

The Issue 997 PoC (split revival vs farthest revival, bench_375 posture) ran as
**one negative**: batch Lloyd's with the shipped k-means++ init never produces
empty clusters on any reachable posture (zero deaths across continuous,
quantized, duplicate-heavy, and the real bench_375 corpora), and forced
starvation provably collapses both revival policies to identical picks (all
revival distances zero under first-index tie-dups). Split did NOT beat
farthest — the arms are unreachable/identical. **GOAT grade for the batch
split-revival track: DEMOTED** per the pre-registered T7 rule. T1 (the
`d2_nearest` staleness fix) stands on its own state-level regression test.
The split-vs-farthest head-to-head moves to the mini-batch EMA regime
(Issue 997 T4), where empty clusters occur naturally; the PoC harness lifts
from git history (tree of `b6b85a906`, 2026-11-12).


## PoC Addendum 2 — Issue 997 T4/T5 (2026-11-12)

The mini-batch EMA lane SHIPPED (`fit_codebook_kmeans_ema_into`, opt-in
`emeans_poc`): Sculley-style EMA over fixed contiguous batches, cosine anneal
alpha0 -> 0.1*alpha0, exact-means finalization, farthest revival. The ~5x5
(B, alpha0) sanity grid at n=16K/K=128/D=32: smooth band 3.90-4.01e5 SSE,
low-loss at B=1000/alpha0=0.3 — the paper's law region (B=hundreds, mid a0)
sits inside it, so the OUT-OF-RANGE extrapolation holds at stack scale (the
"no sweep needed" claim is still NOT made — the grid IS the evidence). The
split-revival challenger measured behaviorally identical to farthest in every
reachable regime (including the 400-cluster overcrowding corner), so the
revival dimension was collapsed to the incumbent. Caveat recorded: mini-batch
seed-variance is real (stuck-rate 49/50 on the uneven fixture vs 0/50 for
batch Lloyd's) — consumers wanting seed-robustness keep the batch fit.
T5: gmm_support 19/19 at HEAD. GOAT grade for the e-means capability:
**OPT-IN CAPABILITY, no promotion** (no SSE win over the batch incumbent on
these corpora; the lane's value is the fit-regime option + the validated
grid instrument).
