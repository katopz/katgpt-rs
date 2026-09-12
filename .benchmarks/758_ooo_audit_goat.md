# Bench 758: direction_bank_audit GOAT Gate — Asymmetric OOO + Cross-OOO + Greedy Curation

> **Feature:** `direction_bank_audit` (katgpt-core) — **OPT-IN** (no-default-consumer rule; promotion needs a production consumer win, see below)
> **Source:** Research 552 / Issue 759 — Issa, Liu, Ballé, Klindt, "High-dimensional population codes reveal interpretable and diverse features underlying visual perception" [bioRxiv 2026.09.05.748439](https://www.biorxiv.org/content/10.64898/2026.09.05.748439v1)
> **Date:** 2026-09-12
> **Gate:** `cargo bench -p katgpt-core --features direction_bank_audit --bench bench_759_ooo_audit_goat` (bench profile, 4090 box, quiet)
> **Companion tests:** 13 lib tests under `--features direction_bank_audit` (+3 more with `factorized_action`), `ooo_audit` module

## Verdict: GOAT G1 + G2 + G4 ALL PASS — primitive ships OPT-IN

```
================ GOAT VERDICT ================
  [PASS] G1 correctness — count=4 min_true_ooo=0.903 pruned=2 dropped=2
  [PASS] G2 latency — audit=1119µs curate=1µs total=1120µs < 10ms
  [PASS] G4 zero-alloc — 0 allocs / 100 cycles
```

- **G1 correctness** — the planted bank (4 true clusters + 2 activation-duplicate
  units + 2 noise units, K=8, paper cutoffs 0.8/0.8) curates to exactly the 4
  planted features across seeds {7, 42, 2026}; duplicates pruned via Cross-OOO,
  noise dropped via OOO, min true-unit OOO 0.9028 ≥ 0.8.
- **G2 latency** — `audit_bank_into` 1119 µs + `greedy_curate` 1 µs at U=64
  units × N=1024 exemplars × K=8 (11.2× under the 10 ms offline-gate budget;
  the audit runs at freeze/consolidation cadence, never in a tick).
- **G3 feature isolation** — `cargo check -p katgpt-core` (default) clean;
  module compiles to nothing without the feature.
- **G4 alloc** — 0 allocations across 100 steady-state audit+curate cycles
  (per-thread CountingAllocator + the Issue 714 live-counter canary). **G4
  caught a real bug on first run**: `AuditScratch::reserve_for` sized with
  `Vec::reserve(k)` while buffers still held the previous run's contents
  (`len == k`), doubling each buffer's capacity exactly once → 2 steady-state
  allocs. Fixed to a capacity-check + clear-then-reserve; the gate is the
  reason the bug is not shipping.

## T3 defend-wrong PoC (the kill condition) — PASSED, redundancy measured

The Issue 759 kill condition: if a REAL bank shows redundancy below the
metric's noise floor, the gate is hygiene-only → demote. Verdict: **redundancy
is real and curation recovers it exactly.**

- **Substrate:** the shipped deterministic k-means (`factorized_action::
  fit_codebook_kmeans_into` — Lloyd + k-means++, no GD) fit over-complete
  (K=12) on 4 planted clusters (N=120, D=8), read as a UTM bank (activation =
  −Euclidean distance to each centroid — the paper's exact construction).
- **Concept-level similarity** (the DreamSim role: same-concept exemplars
  similar regardless of micro-position): the 12-centroid bank curates to
  **exactly 4** distinct features — the audit measures the true
  distinct-content capacity of an over-complete codebook. Exact-fit K=4 also
  curates to 4 (the elbow anchor). This is the F3 K-selection signal from
  Research 552: pick K where the curated count saturates.
- **The metric-granularity law (measured, pinned as a test):** under a
  geometric RBF (γ=1) on raw coordinates, the SAME K=12 bank keeps ~11
  features — the kernel resolves the intra-cluster micro-splits kmeans
  creates. Neither reading is wrong: **Cross-OOO detects redundancy at the
  similarity metric's concept granularity — the metric defines the concept.**
  Consumers must choose a metric at their semantic granularity (span
  embeddings for rules, latent cosine for directions); a geometric metric
  honestly reports micro-features. Pinned by
  `t3_metric_granularity_law_geometric_rbf_resolves_micro_splits`.

## Promotion status

Stays OPT-IN per the no-default-consumer rule. The GOAT is primitive-level;
promotion needs a production consumer win on one of the Research 552 fusions:

- **F1 (riir-ai):** freeze-time curation of MAG-mined direction banks — audit
  before `MerkleFrozenEnvelope` commit; duplicated directions double-count
  effective weight in sigmoid blends (PWC/CFB sum them). Guide: riir-ai
  Research 378.
- **F2 (riir-clippy):** rule-corpus Cross-OOO axis — rules = units, top-K
  oracle-triggering spans = MEIs, span embeddings = similarity; the
  absorption/splitting detector the 580+-rule corpus has never had, feeding
  `frontier_report` as a new per-rule axis. Tracked in riir-clippy.

## Provenance

- Paper: bioRxiv 2026.09.05.748439 (Eq. 6–8 asymmetric OOO, Eq. 9–11
  Cross-OOO, Fig. 4a greedy curation, UTM construction).
- Implementation: `crates/katgpt-core/src/ooo_audit.rs` (pure f32, zero deps,
  deterministic, zero-alloc `*_into` + caller-owned scratch); bench
  `crates/katgpt-core/benches/bench_759_ooo_audit_goat.rs`.
