#!/usr/bin/env python3
"""List UNRESOLVED percentile sites for one repo (per-site-read input).

The main audit (`percentile_index_audit.py`) prints site rows only for the
four severe classes; UNRESOLVED rows appear only in the tally. This helper
dumps them so a human per-site read can resolve each to OK / DEGENERATE /
not-a-percentile.

── 2026-09-04 workspace per-site read (the record) ──────────────────────────

Every UNRESOLVED row across all 16 contract repos was read and resolved.
13/14 riir-chain rows verified-OK, 1 real defect found and fixed
(riir-chain `6ad6f2ab`, `.issues/121`: edge_wallet_agent p99 printed the
max at the default WALLET_AGENT_ITERS=10 — the riir-ai Issue 853 class).
The remaining 12 rows across 6 repos, all verified-OK:

  riir-neuron-db  benches/bench_323_local_kv_commit_levels_goat.rs:49
                  → iters=1000 at every call → idx 990, support 10 = MIN_SUPPORT;
                    the G2 gate reads p50 (`pass = p50 < 100_000.0`), p99
                    print-only. OK.
  riir-neuron-db  benches/bench_324_dense_embed_knn.rs:161,217
                  → N_LATENCY_ITER=1000 → support 10, print-only. OK.
  riir-train      tests/issue_307_archetype_library_gates.rs:511
                  → dists ≈ 5000 (5000 sampled pairs) → p95 idx 4750,
                    support ≈ 250; load-bearing via the A4 ratio assert
                    (trained_div >= 0.9 * syn_div). OK.
  riir-train      tests/xhc_grt_lm.rs:234
                  → NOT a percentile: `train_end = ids.len()*95/100` is a
                    train/test split boundary. Pattern false-positive.
  riir-game-sdk   crates/riir-e2e/src/contention.rs:254
                  → `at` helper in ContentionProbe::verdict; production
                    caller samples TICKS=200 → p99 idx 198, support 2, and
                    the asserted fields are stall/sustained (max-based),
                    never p99 — the module doc itself explains why
                    (p99-based stall detection would be blind). OK.
  riir-game-sdk   crates/riir-e2e/src/percentiles.rs:109 [ASSERTED]
                  → the regression CANARY: asserts the naive form == n-1 at
                    n ∈ {1..100} as an explicit counter-example pin (mmorpg
                    Issue 093 D1's sibling). Intentional — not a defect.
  riir-mmorpg     tests/e2e_topology.rs:826 [ASSERTED]
                  → the same canary class (`quantile_pins_the_rank_that_is_
                    not_the_max`, Issue 093 D1). Intentional.
  riir-clippy     benches/bench_002_l2_pruner_syn.rs:115
                  → n_iters=10_000 → idx 9900, support 100; the gate reads
                    p50, p99 print-only. OK.
  seal-remake     crates/seal-view/tests/texture_vessel_bench.rs:589 [ASSERTED]
                  → the same canary class (asserts the naive shape == n-1,
                    "the shape that reads a max as a p99"). Intentional.
  katgpt-rs       crates/katgpt-types/src/simd/tests.rs:850
                  → NOT a percentile: a sine-wave test-data generator
                    (`i as f32 * 0.97` matched the p-pattern). False-positive.
  katgpt-rs       tests/precision_aware_draft_goat.rs:201
                  → NOT a percentile: a weighted score blend
                    (`clean*0.90 + boundary*0.70`). False-positive.

Net: after riir-chain 121's fix, zero unresolved percentile defects remain
workspace-wide; 5 of the 12 residual rows are auditor false-positives
(2 not-percentile arithmetic, 3 intentional canaries), which is the
expected noise floor of the UNRESOLVED class ("needs a per-site read").

Post-read audit state: 0 DEGENERATE / 0 WEAK+ASSERTED / 0 TRUNC-VAR
workspace-wide (the gated classes); UNRESOLVED 38 → all resolved-by-read.

Usage:
    python3 scripts/list_unresolved_percentile_sites.py ../riir-chain
    python3 scripts/list_unresolved_percentile_sites.py   # katgpt-rs itself
"""
import importlib.util
import os
import sys

spec = importlib.util.spec_from_file_location(
    "pia", os.path.join(os.path.dirname(__file__), "percentile_index_audit.py"))
pia = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pia)

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
target = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else root
name = os.path.basename(target)

rows = []
for f in pia.walk_rs(target):
    rows += pia.audit_file(f, os.path.relpath(f, target))

unres = [r for r in rows if r["verdict"] == pia.UNRESOLVED]
print(f"{name}: {len(unres)} UNRESOLVED site(s)")
for r in sorted(unres, key=lambda r: (r["file"], r["line"])):
    asserted = "ASSERTED" if r["asserted"] else "print-only"
    print(f"  {r['file']}:{r['line']}  [{asserted}]  {r['text']}")
