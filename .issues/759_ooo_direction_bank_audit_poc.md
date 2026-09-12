# Issue 759: `direction_bank_audit` PoC — asymmetric OOO + Cross-OOO curation gate

> **Status:** OPEN (2026-09-12)
> **Source research:** `.research/552_Asymmetric_OOO_Direction_Bank_Audit.md` (Issa, Liu, Ballé, Klindt — bioRxiv 2026.09.05.748439)
> **Repo:** katgpt-rs (open primitive); consumers: riir-ai (MAG bank freeze gate), riir-clippy (rule-corpus dedup axis)
> **Priority:** P2 (offline primitive; does not compete with the 4090 prefill lane)

## Problem

The direction-vector ecosystem (MAG R397 acquisition → LFS/PWC/CFB injection → freeze/thaw commit) has **no audit or curation stage**: nothing verifies a bank member is self-consistent (interpretable) or non-redundant vs other members. Because blends **sum** sigmoid-gated directions, duplicated concepts double-count effective weight — a correctness bug class at crowd scale, not just waste.

## Primitive (modelless, closed-form)

Over activation matrix `A ∈ R^{U×N}` + pluggable exemplar-similarity matrix `S ∈ R^{N×N}`:
- `ooo_score(unit)` — asymmetric intruder fraction (threshold = MEI self-similarity; score = fraction of non-MEIs below it)
- `cross_ooo(a, b)` — symmetric U×U attribution matrix (0.5 = chance)
- `greedy_curate(ooo_cut=0.8, cross_cut=0.8)` → curated index list + unique-feature count
- Unique-feature count as codebook K-selection saturation curve (EffectCodebook / cluster_map / shard-VQ)

Substrate check: K-means ships (`fit_codebook_kmeans_into`); OOO/Cross-OOO do not ship anywhere (grep 0 hits).

## Tasks

- [ ] T1: `ooo_audit` module in katgpt-core behind `direction_bank_audit` feature — `ooo_score`, `cross_ooo_matrix`, `greedy_curate`; zero-alloc, pre-allocated scratch, chunked reductions; unit tests on hand-computed fixtures
- [ ] T2: G1 synthetic fixture — planted bank (distinct + near-duplicate + noise units); recover unique count ±1; prune duplicates; drop noise; curated subset preserves OOO ordering
- [ ] T3: **defend-wrong PoC (the kill condition)** — run audit on a REAL bank (Plan 418 MAG fixtures or a live `EffectCodebook`); measure actual redundancy. If redundancy < metric noise floor → gate is hygiene-only → demote to opt-in, record negative result in Research 552, close
- [ ] T4: G2 latency bench (U×U + U×N reductions) + GOAT verdict; promote to default only if T3 shows load-bearing redundancy AND G1–G4 pass; demote loser if a simpler dedup (e.g. plain cosine on direction vectors) matches it — signal-diff: cosine dedup compares direction VECTORS, Cross-OOO compares ACTIVATION EVIDENCE (two units can be vector-dissimilar yet trigger on indistinguishable exemplars)
- [ ] T5 (deferred, on T3 pass): riir-ai guide — freeze-time bank curation cadence (consolidation window); riir-clippy axis — Cross-OOO over rule top-K oracle spans feeding `frontier_report`

## Acceptance

T1–T4 complete with GOAT verdict recorded in `.benchmarks/`; Research 552 status updated (Super-GOAT + guides on pass; honest negative on kill).
