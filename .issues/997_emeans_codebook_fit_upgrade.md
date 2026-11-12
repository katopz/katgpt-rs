# 997 — e-means codebook-fit upgrade PoC (split revival + EMA arm + defect fix)

**Status:** OPEN — filed 2026-11-12 from Research 665 (arXiv:2611.09501) · GOAT demote-on-fail · consumer surfaces: `katgpt-core::factorized_action::fit_codebook_kmeans_into` + `gmm_support::fit_diag_gmm` (one change, two consumers — the GMM seed rides the codebook upgrade through its existing `fit_codebook_kmeans_into` call at `em.rs` L100–102; building a parallel GMM seed path is FORBIDDEN)
**Lease closed:** T1 landed by zed-84acf899@amo (2026-11-12, this commit); T2-T7 remain open for the PoC lane / a free window
**Lease closed:** T2-T7 LANDED-AS-NEGATIVE by zed-84acf899@amo (2026-11-12, commit b6b85a906 lineage): the PoC ran and the head-to-head is EMPTILY UNREACHABLE in batch Lloyd's — zero dead clusters across 4 synthetic fixture families (incl. quantized/duplicate-heavy) AND the real bench_375 corpus (K=128, D=32, 16K); structural cause: this k-means++ init (D2-sampling + degenerate-branch) only starves centroids via first-index tie-dups after all distinct positions are covered, which provably zeroes all revival distances, making split and farthest revivals coincide BY CONSTRUCTION (forced-death synthetic: bit-identical SSE both arms, 50/50 seeds). The split-vs-farthest question is live only in the MINI-BATCH/EMA regime (deaths are the norm there) — T4's lane, which re-introduces split revival where it is reachable (harness instrument: `bench_997_emeans_split_poc.rs` at commit b6b85a906's tree, liftable from git history). Per the T7 Otherwise rule: the GOAT grade in Research 665 is DEMOTED (split did not beat farthest — they are identical in the reachable regime); T1's fix stands (the staleness mechanism is real, verified at the state level by `revival_refresh_moves_second_pick_off_the_clump`). The split arm + `emeans_poc` feature + harness were removed post-verdict per the noise-reduction law.
**Claimed:** zed-84acf899@amo, 1794463483, expires 2026-11-14T06:04:43Z (T4-T5: the mini-batch EMA lane - where empty clusters occur naturally - + the (B, a0) sanity grid + the T5 GMM consumer check; the split-vs-farthest head-to-head re-run where it is live)

**Verdict provenance:** Claude ping-pong 2026-11-12, R1 REVISE → R2 AGREE (+ final close), session `65c0ee85-095b-4730-b57d-2d33c37c78ad`.

## Context

e-means (mini-batch EMA k-means, Meta FAIR) beats Lloyd's loss in 1.4–2.6 epochs at embedding scale and eliminates dead clusters via proportional-split revival. The stack's offline k-means (`factorized_action/codebook.rs`) is full-batch Lloyd + k-means++ + **farthest-patch reinit** for dead centroids. All e-means components are closed-form/modelless. The paper's ablation NEVER compared split against farthest-reinit, so the head-to-head has real information value. Primary tested hypothesis: **farthest-point revival is outlier-biased; split revival seeds from dense mass.**

**En-route defect (verified in-source):** when several clusters are empty in the same Lloyd pass, `d2_nearest` is recomputed ONCE before the per-cluster placement loop (codebook.rs L232–245), and each placement only zeroes its chosen point (L259) — a second empty cluster selects from stale distances and can clump onto the first revival's outlier position. Fix benefits the incumbent regardless of the split verdict.

## Tasks

- [x] **T1 — standalone defect fix (do first, independent of e-means):** in `fit_codebook_kmeans_into`'s revival loop, update `d2_nearest` against each newly placed centroid before the next empty cluster picks (incremental update or per-placement recompute; keep the f64/last-index tie rules). Add a regression test: a fixture forcing TWO empty clusters in one pass must place the two revived centroids in distinct clusters (not the same clump), plus `kmeans_deterministic_from_seed` must stay green. LANDED 2026-11-12: the refresh extracted as the pure helper `refresh_d2_nearest_against` (codebook.rs, called after each revival placement); regression tests `revival_refresh_moves_second_pick_off_the_clump` (stale second pick = the pair-mate clump, refreshed second pick = the runner-up region — pins the exact defect mechanism at the state level, since a minimal synthetic fixture cannot reach a distinguishable two-empty pass end-to-end: this init's tie-starvation only fires after every distinct patch position is covered, which zeroes all revival distances) + `revival_two_dead_clusters_stays_deterministic` (duplicate-heavy fixture, K=6 > 3 distinct positions, determinism + finiteness); full katgpt-core lib suite 2527 passed / 0 failed at `--features factorized_action`; clippy `-D warnings` clean; default-feature build byte-identical (module gated).
- [x] **T2 — four-arm PoC harness** on the bench_375 patch corpus posture (K=128, D=32, 16K patches, release profile, isolated `CARGO_TARGET_DIR`): arms = shipped / fixed-farthest (T1) / split / split+EMA. Primary metric: **final SSE** (sum of squared nearest-centroid distances, plain sequential f64 fold — never a SIMD reduction). Report per arm: SSE, wall-time, revival count (DIAGNOSTIC ONLY — never a veto at unequal SSE), G1 no-regression (must hold the 4.9× factorized-vs-monolithic margin). Feature-gate the split/EMA arms (opt-in), keep default byte-identical.
- [x] **T3 — stuck-rate fixture:** ~50 seeds per arm on an uneven multi-blob fixture (two blobs of very different mass + a light middle blob — the paper's pathology class); a seed is STUCK when its final SSE lands > 1% above the best-of-arms SSE; report COUNTS (e.g. `shipped 15/50, split 2/50`), never bare percentages.
- [ ] **T4 — EMA arm + scaling-law constants:** mini-batch EMA updates (fixed contiguous batch schedule, cosine anneal α₀→0.1α₀, finalization = exact last-epoch mean) behind the same opt-in feature. The (B, α₀) constants from arXiv:2611.09501 are recorded as OUT-OF-RANGE EXTRAPOLATIONS (fitted at n≥10⁶; here n≈16K, K=128 → B≈hundreds). Include a ~5×5 (B, α₀) sanity grid; "no sweep needed" is NOT claimed at stack scale until this grid shows the law's (B, α₀) sits in the low-loss band.
- [ ] **T5 — GMM consumer check:** with the upgraded seed, measure `fit_diag_gmm` EM final log-likelihood + iterations-to-tol on the existing gmm_support fixtures (must be ≥ shipped seed on final LL; EM iteration count reported). Rides T1/T2 — no separate seed path.
- [x] **T6 — determinism gate:** BLAKE3 parity test — two runs at the same seed produce byte-identical codebooks for every arm; sequential f64 accumulation asserted (no platform-lane SIMD paths); size-proportional sampling keeps the last-index tie rule.
- [x] **T7 — verdict:** record raw numbers in Research 665 §"PoC Addendum" in the SAME commit as the outcome. Split arm beats fixed-farthest on SSE at ≤ wall-time AND stuck-rate improves ⇒ GOAT stands, promote the winner behind its feature flag per the flag discipline. Otherwise ⇒ strike the GOAT grade in 665 (demote), keep only T1, record the negative.

## Notes

- The paper's quality claims are scale-independent (escape + revival); its speed claims are NOT expected to transfer to 16K patches — do not gate on speedup, gate on SSE at ≤ wall-time.
- `bench_375` G1 is a no-regression check only — it is a downstream abstraction ratio, not a fit-quality metric (verdict R1 correction).
- Bench numbers quote `scripts/bench_preflight.sh` PROVENANCE (or the riir-reflex equivalent) beside them.



## Outcome (2026-11-12, T2-T7 executed as ONE NEGATIVE)

- **T2/T3/T6 ran** (harness at commit `b6b85a906`'s tree, release posture, isolated
  target dir): T6 BLAKE3 parity PASS (3 seeds x 2 runs); T3 50-seed stuck-rate:
  farthest 0/50, split 0/50 — arms bit-identical on every seed; T2 @16K:
  SSE ratio split/farthest = 1.0000, wall within noise (432 vs 434 ms).
- **Root finding (structural, not incidental):** batch Lloyd's + this k-means++
  init never starves a centroid on reachable postures (verified on continuous,
  quantized, duplicate-heavy, and the real bench_375 corpus — the
  `counts.contains(&0)` branch is dead code there). Forced starvation (K > distinct
  positions) goes through first-index tie-dups, which zero every revival distance —
  then both revival policies pick identically. The arm question is REAL only under
  mini-batch EMA dynamics (T4), where empty clusters are the norm.
- **T7 verdict: NEGATIVE per the pre-registered rule.** Research 665's GOAT grade
  DEMOTED (split did not beat farthest — unreachable/identical). T1 kept: the
  staleness mechanism is real and pinned by the state-level regression test.
  T4 remains open as the decisive lane (EMA + split in the mini-batch regime);
  the harness lifts from git history (`b6b85a906` tree).
