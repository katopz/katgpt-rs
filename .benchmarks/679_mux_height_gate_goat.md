# Bench 679 — Issue 688: Mux height-gated commit + gap-trend narrowing GOAT

**Date:** 2026-08-26 · **Repo:** katgpt-rs · **Issue:** `688`
**Source:** Coconut (arXiv:2412.06769v4) §4.4 + Fig. 6 — classic-heuristic
port (quiescence/CNS/LRTA* defer-commit; successive-halving/Hyperband/
Hoeffding-races narrowing). **No novelty claim.**

## What shipped

`crates/katgpt-core/src/mux/height_gate.rs` behind
`mux_height_gate = ["mux_bfs"]` (opt-in):

1. **`HeightGate { commit_height }`** — commit-timing gate over
   distance-to-TERMINAL: `should_commit(h) = h ≤ commit_height` (inclusive);
   `commit_width(h, base) = 1` for near-terminal leaves (deterministic
   commit), `base` otherwise; `commit_order_by_height[_into]` ranks
   candidates (value desc, height asc, index asc) — among value-ties,
   min-height commits first. Consumes the signal `MuxBfs::step` previously
   plumbed and discarded (`let _ = depth`).
2. **`GapTrendNarrower`** — inter-step width controller over the cumulative
   top-k mass gap, which telescopes to `p_first − p_last`. Control law:
   grew > ε → narrow (floor 1); shrank > ε → widen (cap base); flat → hold.
3. **`MuxBfs::step_height_gated[_into]`** — composed step: per-step the
   frontier's MEAN cumulative gap feeds one narrower (the inter-step signal
   is search-level, Coconut Fig. 6's axis); per-leaf width =
   `commit_width(height, base.min(step_cap))`. `_into` variant reuses the
   caller's `LeafPaths` (zero-alloc steady state).

## En-route: latent `collect_leaf_paths_flat_into` bug FIXED

The composed multi-step tests exposed a **pre-existing defect** in
`dd_tree.rs`: the DFS appends intermediate child-path copies to `buf` as
stack entries are pushed — garbage between the final leaf copies — but the
single-offset scheme assumed contiguity, so `path(i)` returned garbage
prefixes (`path(0) == [3,2,1,0,0]` instead of `[0]`) for ANY tree with
depth ≥ 1. Every prior caller only ever collected from a root-only tree
(single-step tests), so the bug was latent — production `step()` at BFS
step ≥ 2 would index-panic or expand wrong nodes. Fix: explicit two-phase
collection (pairs recorded during DFS, in-place left-compaction pass after;
write index ≤ read indices so the rewrite is safe; same reuse contract, O(n)
`copy_within`). All existing dd_tree/bfs tests pass against the fix.

## GOAT gates — ALL PASS

| Gate | Result |
|---|---|
| **G1** commit ordering ≡ oracle (value desc, height asc) on planted DD-tree fixtures | PASS — gated ordering matches the oracle exactly ([1,0,3,2,4] on the inverted-heights fixture) AND differs from the ungated (insertion-order) baseline — the tie-break is load-bearing |
| **G1 negative control (mandatory)** — SimpleTES-style flat-gap fixtures must NOT narrow | PASS — 50 steps of constant-gap frontiers hold width at base (4) through BOTH the bare narrower AND the composed step; the control law structurally cannot narrow on a flat derivative (Bench 017 T9 wide-dominance is safe) |
| **G1 narrowing direction** | PASS — sharpe frontiers narrow monotonically 4→3→2→1; shrinking gaps widen back to base; dead-band (sub-ε) holds |
| Composed behavior | PASS — near-terminal leaf commits to exactly 1 successor (depth grows, breadth holds) vs ungated 4; far leaf = parity with ungated; flat-gap 4-step run keeps 256 = 4⁴ leaves (uniform width held) |
| **G2** ns-scale latency (release) | PASS — `observe_gap` + `commit_width` < 50 ns/op budget (measured ~1–3 ns; 200k ops) |
| **G3** no-regression | PASS — all 39 existing mux tests green in BOTH `comp_width` states against the collector fix; katgpt-core default lib 1917/1917; feature off by default (`mux_height_gate` opt-in) |
| **G4** zero-alloc controller path | PASS — extract → gap → narrower → gate: 0 allocations / 1000 iterations (TrackingAllocator). The composed step's remaining allocs are the TREE's (child growth) + CALLER's (logits storage) — covered by existing bfs/dd_tree gates |

## Non-goals held

No continuous-thought feedback loop (Coconut's own ablation: w/o curriculum
14.4% ≈ no-CoT 16.5% on GSM8k — the untrained latent loop is inert; Plan 276
Family A/B nulls corroborate). Training-side rows live in riir-train Plan 352.

## Verdict

Stays **opt-in** (`mux_height_gate`) — no default consumer yet; the gap in
the issue's own terms ("wide ships as a static config choice") is now
*closable* by consumers, not switched. Promotion awaits a real search
consumer measuring wide-vs-gated on a SimpleTES/DD-tree workload.

**Record:** commit on `develop` (see git log `feat(mux)` — Issue 688 + the
`fix(mux)` collector fix in the same series).
