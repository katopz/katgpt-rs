# Bench 997 — i1250 arm B T7 cell: cross-layer KV-sharing init loss-delta (NULL — the bar's premise not met)

**Date:** 2026-11-28 · **Box:** amo (M5 Ultra 96 GB, exclusive rig lane) · **Verdict: NULL — the halving bar's premise (a naive-copy degradation) never materialized; the arm closes null per the closure law.**

## Cell posture

Pre-registered in `.issues/1250` arm B T5 (commit `3b45a85be`, pinned BEFORE the run). Harness: riir-infer `kv_share_align` (landed `56a0308b`, gates 7/7) + the cell bin `i1250_kvshare_cell`. Run: amo exclusive rig lane, 2026-11-28 06:35–07:35 UTC (≈60 min: release build + fit + 3 arms × 13 windows). **Poison disclosure:** the rig flagged ×24 sibling rustc/cargo processes starting 1 min into the run (clean for the final 6 min) — the outputs are deterministic teacher-forced NLL counts over fixed windows with paired arms (load hits all arms identically and cancels in the deltas); nothing crashed, the artifact is complete; the run is VALID with the window disclosed. Box state (from the artifact): `PROVENANCE: power=AC Power load=6.89 swap=4014.12M canary=skipped powermode=0(auto) ane=ok`, loadavg {6.89, 7.59, 8.27}.

## Configuration

Checkpoint `Ternary-Bonsai-2.5-27B-PQ2_0` (the league model); FA pair **(3, 7)** (closest-gap convention, gap-4 confound disclosed in the pre-registration); anchors `tokens.fit.tokens` first 4096 (sha256 `1a862fe7…`), eval `tokens.chat.tokens` 14,075 tokens (sha256 `2e8ef853…` — the pin-label correction story: the original pre-registration labeled the file's BLAKE3 as "sha256"; the file never drifted, mtime 2026-11-27 11:12; the v2 cell's fail-closed refusal was the gate working); 13 × 1024-token windows with 512 warmup; arms baseline / naive-copy / aligned-init (rotate-half constrained key rotation via `katgpt_spectral::rope_commuting_procrustes_paired` + free orthogonal on the non-rotary slice + rank-128 Eckart–Young value maps via `katgpt_core::linalg::symmetric_eig`, O-fold per head pre-rotation); bootstrap 10,000 resamples, SplitMix64 seeded from the corpus digests.

## Fit facts

Angles per kv head ‖θ‖₂ ≈ [9.97, 10.18, 8.70, 10.98] rad — **the producer/consumer key spaces are substantially misaligned pre-fit** (the NO-OP falsifier does not fire); constrained+rank-truncated fit residual 0.605 (the map explains ~60% of the anchor-scale key/value mismatch).

## Results (verbatim from the artifact)

- pooled deltas: **naive −0.0300, aligned −0.0272** (nats/token vs baseline) — **both sharing arms slightly IMPROVE on the unshared baseline** (~1.2% of the ~2.5 nats/token loss; no [arm − baseline] CI was instrumented — disclosed)
- bootstrap CI [aligned − 0.5·naive]: mean −0.0122, **CI [−0.0243, −0.0040]** (entirely ≤ 0)
- bootstrap CI [aligned − naive]: mean +0.0028, **CI [−0.0107, +0.0127]** (straddles zero — aligned ≈ naive)

## Adjudication (delegated-owner verdict round 1, session `ad265a92`, 2026-11-28 — REVISE, executed)

The bin PRINTED `NEGATIVE` via a verdict-branch defect (`mean > 0 && hi > 0` instead of the CI's lower bound; fixed `4193bded`). Under the corrected gates the printed verdict would have been PASS — **and the verdict records NULL instead, per the REVISE**: the halving bar "aligned ≤ 0.5× naive" assumes naive copy DOES damage (naive delta > 0); with naive delta = −0.0300 the ratio flips meaning (it would test "aligned captured half of naive's improvement"), which the pre-registration never registered. A sign-artifact PASS would fire the closure law's positive branch (the riir-train note row) for a null substance — exactly what the pre-registration exists to prevent. UNDECIDED was also rejected: its registered remedy ("more windows") cannot fix a test that does not apply; the closure law's null branch closes the arm.

**Recorded verdict: NULL — bar premise not met (naive delta < 0, the halving ratio is undefined in this sign regime); the ratio did not decide anything.** Substantive findings: (1) aligned ≈ naive (CI straddles 0) — the closed-form alignment shows NO measurable benefit over naive copy on this pair; (2) both arms ≈ baseline −0.03 nats/token (the unexplained anomaly is disclosed as an advisory confound: the affected tensors run f32 while the baseline's layer-7 KV rides the native quantized backbone — a precision difference could produce a uniform gain; a self-copy identity control (layer 7's own KV through the same f32 path) would separate a precision effect from a sharing effect — optional, not required to close); (3) the key spaces were misaligned by ~10 rad per head and the constrained fit closed only ~60% of the anchor residual — the recipe's machinery WORKS (the substrate + harness are landed, gates green) but had no degradation to reduce.

**Consequences per the closure law:** NULL files NO riir-train note row; the CPT-window question is NOT ASKED (the "better init → cheaper CPT" premise got no evidence either way: no damage to reduce, and the CPT lane is budget-infeasible per Research 698 §3 track c). **Arm B CLOSES at this landing** — re-open precondition: a pair or checkpoint where naive copy MEASURABLY HURTS (naive-delta CI lower bound > 0) — only then can the halving bar decide anything. Never a retune on this pair.

**Substrate yield (the landing that outlives the null):** `katgpt-spectral` `rope_commuting_procrustes(_paired)` (i1251, both RoPE conventions, bit-exact commutation pins) + riir-infer `kv_share_align` (the fit + three-arm measurement forward, gates 7/7) + the premise-guarded verdict ladder (NO-OP / NULL / NEGATIVE / PASS / UNDECIDED / FAIL) — all standing for the next sharing/alignment measurement that meets the bar's premise.

Artifacts: `/tmp/i1250_kvshare_cell_artifact.json` (5,366 bytes; box-state, corpora digests, fit facts, per-window rows, bootstraps, verdict). Log: rig job `01a220db-7574` (`/Users/katopz/.agent-mail/jobs/logs/01a220db-7574-753c-808f-4df3c959aa6f.log`). Session: zed-7bab0f17@amo, epoch 1795832278; verdict session `ad265a92`.
