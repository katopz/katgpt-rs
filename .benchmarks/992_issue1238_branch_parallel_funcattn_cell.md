# Bench 992 — Issue 1238 fused multi-branch FUNCATTN cell (select-vs-fuse-vs-wide)

**Status:** RECORDED 2026-11-22 — A6 gate WIN by the pre-registered rule; **promotion REFUSED on attribution** (verdict below). `funcattn_branch_parallel` stays opt-in.

## Provenance

- Pre-registration: `.issues/1238_*` PINNED block, committed `f8aa01e67` before any run.
- Harness: `crates/katgpt-attn/tests/issue_1238_branch_parallel_goat.rs` (release, `--features funcattn_branch_parallel,funcattn_freeze_thaw`); G4: `tests/issue_1238_branch_parallel_alloc.rs`. Primitive: `katgpt_attn::funcattn_compose::branch_parallel`.
- Box: bmo (M5 Ultra 96 GB) — test binary shipped from amo (no source on bmo). `agent-mail rig status` at launch: bmo QUIET, load 1.1/30, 79/96 GB free, AC. `bench_preflight.sh` PROVENANCE not captured (script not on bmo; binaries-only rule) — the only timing figure is G5's interleaved ratio. 35 training runs in 7.3 s; whole test 13.4 s.
- Task: Plan-286 T3.2 sinusoidal regression (n=64, d_model=8, k=8, α=0.01, τ=0.1), FD-SGD central differences, pilot-picked LR = 1.0 (largest on {1.0, 0.5, 0.25} for which every arm learned on the pilot seed). Five gate seeds; 150-step figure = checkpoint of the 500-step run.
- Metric: PRIMARY held-out MSE (fresh 64-token sequence); decision rule r̄ (pp) vs δ = max(1.0, 2.5·SE).

## Arms

A1 d_b=8 K=1 (384) · A2 d_b=5 K=2 Σ/√2 joint (390) · A3 = A2 + learned concat-projection (518) · A4 = A2 + shared slice basis (390) · A5 d_b=12 K=1 (720) · A6 two frozen independently trained d_b=5 snapshots (FUNCATTN weights round-tripped through BLAKE3-committed `FuncAttnWeightsSnapshot` / `FuncAttnSnapshotStore`), mean-fused (390) · A6b = A6 fused Σ/√2.

## Results — held-out MSE, mean of 5 seeds (train mean in brackets)

| arm | 150 steps | 500 steps |
|---|---|---|
| A1 | 0.0802 [0.0127] | 0.0913 [0.0095] |
| A2 | 0.0719 [0.0137] | 0.0598 [0.0060] |
| A3 | 0.0615 [0.0126] | 0.0558 [0.0053] |
| A4 | 0.1019 [0.0139] | 0.1158 [0.0109] |
| A5 | 0.0857 [0.0117] | 0.1075 [0.0087] |
| A6 | 0.0469 [0.0130] | 0.0595 [0.0101] |
| A6b | 0.0325 [0.0634] | 0.0376 [0.0616] |
| halfA (single snapshot) | 0.0446 [0.0145] | 0.0580 [0.0109] |
| halfB (single snapshot) | 0.0614 [0.0137] | 0.0662 [0.0105] |

## Gates

| gate | 150 | 500 | verdict |
|---|---|---|---|
| G1 bit-identical re-run | — | — | PASS (asserted) |
| G2 A2 vs A1 | +3.5 pp TIE | −20.8 pp TIE | PASS (not LOSS) |
| G3 A2 vs A5 | −12.5 pp TIE | −43.1 pp WIN | PASS |
| G4 alloc-free hot path | — | — | PASS (0 allocs, K=2, both fusions, shared + private basis) |
| G5 A2/A1 forward wall | — | — | median 1.771 ≤ 2.2 PASS |
| A3 vs A2 (Table 6 on this host) | −11.8 pp TIE | −6.6 pp WIN | learned concat slightly better at 500 (optimisation geometry — same function class) |
| A4 vs A2 (shared basis, Table 5 axis) | +42.9 pp TIE | +103.5 pp LOSS | shared slice basis LOSES |
| **A6 vs A1 (promotion-bearing)** | **−35.7 pp WIN** | **−31.3 pp WIN** | **WIN by the pinned rule** |
| A6b vs A6 (scale-law control) | −23.5 pp TIE | −29.7 pp TIE | control did NOT lose on held-out (lost 6× on train) |
| A6 vs halfA (attribution) | +11.8 pp TIE | +3.8 pp TIE | fusion adds nothing over ONE snapshot |

## Verdict (Claude, recorded under the 2026-11-11 gating law)

**Promotion REFUSED; the A6 WIN stands as recorded but is not a fusion gain.**

1. **Attribution failure.** A6 ties a single half-width snapshot at both budgets. The win over A1 is capacity/regularisation: A1 overfits (train 0.0095 vs held-out 0.091 at 500), and a 195-param half model alone beats it. The same cause drives G3's WIN (A5, the widest arm, overfits most).
2. **The metric is shrinkage-dominated.** The pre-registered negative control A6b (Σ/√2, expected to lose) loses 6× on TRAIN but beats A6 on HELD-OUT, i.e. held-out predictions are under-scaled and any output inflation helps. A metric where the known-wrong scale wins cannot adjudicate fusion quality. The scale law itself is confirmed on train.
3. **What does stand:** the primitive is correct, deterministic (G1), alloc-free (G4), within the latency bar (G5 1.77×); the shared-slice-basis axis LOSES on this host (A4); joint K=2 composition does not lose to the single wide branch (G2).

**Re-open bar for promotion:** a regime where A1 does not overfit (multi-sequence training corpus, held-out train/test gap < 2×) in which (a) A6 WINs vs its best single snapshot and (b) A6b LOSES to A6 on held-out. Both are needed before a promotion proposal is filed.

## Carried from Issue 1238 (issue file removed at close; full text — translation table, riders, LEO/Elo cross-refs — in git history)

Guard rows (grep bait; each restates an existing law):
- **G-1 `mbt_table6_not_an_artifact_blend_license`** — Table 6 is about ACTIVATIONS in one model; never a license to blend decision ARTIFACTS (arsenal law A2: selection is monotonic atomic hot-swap).
- **G-2 `per_npc_branch_fusion_adverse_prior`** — K fused personality branches per NPC break runetrace attribution, HlaCacheProxy 5-scalar provenance and block-λ keying; the legitimate form is identity-keyed unit directions (Plan 514 / Issue 1239). W1-K2 (0.93–0.99×) is the named adverse prior.
- **G-3 `shared_cache_stays_local_compute`** — Joint-KV is in-process compute; at a sync seam the raw-scalars-only bridge law holds.
- **G-4 MCGS position-invariance** — `multi_branch_aggregate` stays parameter-free unless weights are proven position-invariant (Plans 061/065).

Open advisory riders are owned by doc-sync lanes (MoA doc-rot + stale `fusion_arm.rs` "Bench 027" pointer; Plan 376 uniform-η ablation at next touch; Bench 027→799 corroboration at next hero-fusion rerun; uniform-mix-MoA falsifier, proposer-gated). Optional `funcattn_compose` parent feature in katgpt-attn: deferred, no consumer.
