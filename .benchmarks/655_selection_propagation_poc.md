# Bench 655 — Selection-Set Fixpoint Propagation POC (GOAT G1–G4)

> **Source:** katgpt-rs `Issue 655` /
> [Research 483](../.research/483_KEEP_KV_Centric_Memory_Management.md) (KEEP,
> arXiv:2602.23592, DAC 2026; HippoRAG PPR class)
> **Date:** 2026-08-16
> **Feature:** `selection_propagation` (opt-in — promotion awaits the first
> production consumer, see T5 routing below)
> **Machine:** M3 Max (arm64), `cargo test`/`cargo bench` release, GPU idle
> (CPU-only POC; no GPU exclusivity concern).

## Verdict: **PASS** — claim CONFIRMED on quality, cost honestly reported

A query-seeded importance propagation iterated until the top-`r` selected set
stabilizes **decisively beats** the shipped BFS k-hop + inverse-sigmoid
hop-decay traversal (`KgTripleIndex::k_hop_neighbors` + `riir-rag
fuse_graph_candidates`) on multi-hop chain recall at equal selection budget,
**36/36 cells won, 0 ties, 0 losses** at hop ≥ 2. Single-hop is not
competitive (as expected).

## Competitors (equal selection budget `k`)

| # | Arm | Shape |
|---|-----|-------|
| 1 | single-hop | top-k by `dot(query, memory)` — the engram latent-lookup shape |
| 2 | BFS-decay (shipped) | top-1 entity link → `k_hop_neighbors` BFS over the symmetrized adjacency (spo∪osp) → `graph_score = 1/(1+exp(1.5·d))` (the shipped `GraphRagConfig::default`: `k_hop=2, λ=1.5`) → fused `query_sim + graph_score` (the packer's unweighted sum, `packer.rs` L233-244), distance-0 skipped |
| 3 | propagation (Mass) | `propagate_selection_to_fixpoint_into`, `α=0.85, β=4, max_iters=16`, budget r = k, seed = `sigmoid(4·dot)` — PPR-style mass blend `next = (1-α)·seed + α·Σ w_ji·sigmoid(β·score_j)` |
| 4 | propagation (Mean) | the literal KEEP `edge_avg` blend `Σ w·rel / Σ w` — shipped so the single-supporter weight-cancellation degeneracy is MEASURED, not assumed |

## G1 — quality (load-bearing): PASS

Fixture: KEEP Fig-6-shaped planted chains. Per chain: head (query ≈ head +
noise, sim ≈ 0.95), `hop` tail nodes with their OWN random embeddings
(query-sim ≈ 0 — invisible to single-hop BY CONSTRUCTION), chain edges
`U[0.75, 0.95]`, calibrated distractors (query-sim ≈ 0.71, attached to the
head at `U[0.25, 0.5]`), background noise edges `U[0.05, 0.3]` (2/node).
12 chains × 5 seeds × {hop 1-4} × {distractors 4/12/24} × {k 4/8/16/32}.
Metric: chain-recall@k (full chain) / tail-recall@k (chain minus head — the
part only a graph method can find).

**h≥2 means over all 36 cells (chain/tail):**

| arm | chain recall | tail recall |
|---|---|---|
| single-hop | 0.293 | 0.046 |
| BFS-decay (shipped defaults) | 0.267 | 0.058 |
| **propagation (Mass)** | **0.789** | **0.730** |
| propagation (Mean) | 0.297 | 0.051 |

**Per-cell:** propagation (Mass) wins 36/36 cells vs BFS-decay on chain
recall at h≥2 — zero losses. Worst-case propagation cell (h=4, d=4, k=4):
0.423/0.329 vs BFS 0.197/0.004.

### Honest findings (predictions vs measurements)

1. **BFS-decay is WORSE than single-hop under calibrated distractors.**
   h=2/d=24/k=4: BFS 0.100 chain vs single 0.328. Mechanism: the graph bonus
   `1/(1+exp(1.5·d))` rewards proximity, not relevance — distractor neighbors
   of the head at distance 1 (+0.18) outrank the chain tail AND cannibalize
   budget that single-hop would have spent on other chains' heads. The
   shipped `fuse_graph_candidates` fusion is actively harmful in this regime.
   (In the shipped G5 use case — transitive callers with zero lexical
   overlap — distractor pressure is absent, so the shipped path is fine
   there; this is the calibrated-distractor regime the issue specified.)
2. **1-hop does NOT tie — propagation wins there too** (0.79-0.99 vs
   0.50-0.70). The issue predicted a tie; measurement says win. The win
   persists because distractor pressure exists even at h=1 (the successor
   must still outrank distractors). With ZERO distractors the gap narrows
   (h=3/d=0 control: mass 0.431/0.315 vs BFS 0.347/0.130) but does not
   invert — weights never hurt a clean chain, and hop-decay's `1/(1+e^{1.5·3})
   ≈ 0.013` leaves distant tails ranked below other chains' heads even with
   no distractors present.
3. **Mean blend (the literal KEEP edge_avg) is degenerate exactly as
   predicted**: h≥2 tail recall 0.051 vs Mass 0.730. For a node supported by
   exactly one selected node, `w·rel/w = rel` — the edge weight cancels, so a
   w=0.85 chain edge and a w=0.40 distractor edge score identically. Unit
   test `mean_blend_weight_cancellation_documented` pins this bit-exactly.
   **Consumer guidance: use `PropagationBlend::Mass`.**
4. **The membership fixpoint rarely fires in distractor-dense worlds** (G2:
   0/32 queries stable within 16 iters at N=1024/k=32). Boundary churn among
   near-tied distractors keeps the selection moving; `max_iters` is the
   operative stop. The result stays bit-deterministic either way
   (`deterministic_bit_identical_two_runs`). A damped/hysteresis stop is a
   possible follow-up, not needed for the POC verdict.

## G2 — latency: PASS (µs-scale), honest negation of "cheaper than BFS"

N=1024, budget k=32, 32 queries, release build (all arms pay the same
1024-element top-k sort ≈ 44 µs):

| arm | µs/query | notes |
|---|---|---|
| single-hop | 44.7 | sort-bound |
| BFS-decay k_hop=2 | 47.1 | avg BFS visited 78 nodes |
| BFS-decay k_hop=4 | 55.3 | avg visited 816 nodes |
| **propagation (early-stop)** | **73.0** | avg 16.0 iters (max_iters-bounded, 0/32 stable) |
| propagation (max_iters=64) | 218.7 | the no-early-stop upper bound |

- **µs-scale at N≤1024, k≤32: PASS** (73 µs).
- **"Propagation may be cheaper than BFS at equal recall": NEGATED in this
  sparse regime.** BFS visited sets stay small (78-816) at degree ≈ 6; the
  O(degree^k) blowup needs dense graphs. At EQUAL RECALL the comparison is
  vacuous — no `k_hop` reaches 0.73 tail recall under calibrated distractors
  (the BFS arm's ceiling in these cells is 0.16); propagation is
  infinitely cheaper per unit of recall.
- Cost driver is the operator's per-iteration top-r selection (O(n·r) with
  r=32, ×16 iters), not the edge propagation (6,352 edges × 16 iters is
  trivial). A bounded-heap selection is an obvious optimization if a
  consumer needs it hotter.

## G3 — no-regression: PASS

Additive feature-gated module. Default lib tests 1887 passed / 0 failed
(unchanged baseline); `--no-default-features --features
selection_propagation` clean; clippy 0 warnings across
`--features selection_propagation --all-targets` (lib + 2 test bins + bench).

## G4 — alloc-free: PASS

`bench_655_propagation_alloc_check` (CountingAllocator, separate binary):
**0 allocs / 0 deallocs across 100 steady-state calls** at N=1024/k=32 in
BOTH blend modes after 5-call warmup (scratch grow-only; top-r buffers sized
`budget+1` so the insert-then-pop window never reallocates).

## Gates table

| Gate | Status | Evidence |
|---|---|---|
| G1 quality (load-bearing) | **PASS** | h≥2: 0.789/0.730 vs BFS 0.267/0.058; 36/36 cells; single-hop 0.293/0.046 |
| G2 perf | **PASS** (µs-scale) | 73.0 µs @ N=1024/k=32; "cheaper than BFS" negated (see above) |
| G3 no-regression | **PASS** | default 1887/0; feature combos clean; clippy 0 |
| G4 alloc-free | **PASS** | 0/0 across 100 calls × 2 blends |

## Promotion decision

**Stays opt-in** — the GOAT passes modellessly, but per the
`grapem_rodrigues`/`mop_path_entropy` precedent (gain proven on synthetic
POC, no production consumer wired yet), promotion to default-on lands WITH
the first consumer:

- **riir-ai Issue 703** — riir-rag `fuse_graph_candidates` F1: offer a
  propagation-based graph fusion behind `graph_rag` config (Mass blend),
  gated on the existing G5 transitive-caller test extended with calibrated
  distractors.
- **riir-ai Issue 704** — engram conditional-memory chain recall F2: seed
  propagation from the NPC's query/goal direction over the engram-KG
  adjacency (CLR-reliability-weighted), bounded by a tick-budget
  `max_iters`.

## Failure protocol

Not triggered (claim confirmed). Recorded numbers stand as the regression
baseline; the POC test suite (`bench_655_selection_propagation_poc`) is the
standing gate.

## Run commands

```bash
# G1 quality sweep + controls + degeneracy (3 tests)
CARGO_TARGET_DIR=/tmp/k655 cargo test -p katgpt-core \
  --features selection_propagation --test bench_655_selection_propagation_poc -- --nocapture

# G2 latency
CARGO_TARGET_DIR=/tmp/k655 cargo bench -p katgpt-core \
  --features selection_propagation --bench bench_655_propagation_latency

# G4 alloc
CARGO_TARGET_DIR=/tmp/k655 cargo test -p katgpt-core \
  --features selection_propagation --test bench_655_propagation_alloc_check
```
