# Bench 621 — escalation probe gate, healer-lane eval (Plan 621 Phase 4, T4.1–T4.4)

**Status:** SUPER-GOAT on the declared healer-lane model (G1 + G2 PASS; G3 default-posture no-regression verified; G4 is the kernel's, already pinned) — promotion stays OPT-IN pending the production consumer (Plan 621 T5.2, riir-refine Issue 156), which re-gates on REAL misses
**Date:** 2026-10-11
**Instrument:** `crates/katgpt-core/tests/state_probe_t4_escalation_eval.rs` (G1, prints the table) + `crates/katgpt-core/benches/bench_621_escalation_eval.rs` (G2 cost cell) over `crates/katgpt-core/src/state_probe_eval.rs` (the corpus + arms)
**Corpus:** planted, seed `0x621_621_621_621`, 150 states = 60 knowledge-like (p = 0.0) + 60 execution-like (p = 0.20) + 30 productive (p = 0.60), CLASS-INTERLEAVED round-robin (layout contract — corpus order is not class-aligned); declared models in `state_probe_eval.rs`'s module doc (scissors law: escalation rescues ONLY the knowledge class, R_ESC = 0.6; local retries k = 4; both arms share the per-state draws)

## The comparison table (T4.2 — one frozen read, deterministic)

G1 tight cap (matched spend, corpus order) — cap = 50 units:

| arm | rescues | esc-net | escs | spend (units) | probe evals | esc-precision |
|---|---|---|---|---|---|---|
| never-escalate | 63 | 0 | 0 | 0.0 | 0 | 0.00 |
| incumbent (cap) | 35 | 10 | 50 | 50.0 | 0 | 0.34 |
| **probe-gated** | **93** | **30** | 50 | 50.0 | 478 | **1.00** |

G1 generous cap (the uncapped incumbent escalates every state) — cap = 150 units:

| arm | rescues | esc-net | escs | spend (units) | probe evals | esc-precision |
|---|---|---|---|---|---|---|
| never-escalate | 63 | 0 | 0 | 0.0 | 0 | 0.00 |
| incumbent (cap) | 98 | 35 | 150 | 150.0 | 0 | 0.40 |
| **probe-gated** | **98** | **35** | 60 | **60.0** | 478 | **1.00** |

## The verdict (T4.4 — the Super-GOAT decision)

- **Matched spend (the plan's own bar):** the probe-gated arm rescues **2.66×** the incumbent at the same 50-unit cap (93 vs 35) — the triage value, measured. The incumbent burns 66 % of its budget on states the escalation cannot rescue (the scissors law); the probe's escalations land on the knowledge class at precision 1.00.
- **Uncapped:** identical rescues (98 = 98 — the probe misses nothing the incumbent's extra 90 blind escalations rescue, because under the scissors law those rescue nothing) at **40 % of the spend** (60 vs 150 units).
- **The gate FIRES** (the T4.1 floor law): 60 escalations, net 30 rescues over the floor — the never-escalate control sits strictly below both escalating arms.
- **Precision mechanism:** probe 1.00 vs incumbent 0.34–0.40 — every probe escalation landed on an escalate-worthy state; ~2/3 of the incumbent's were waste.

**Verdict: probe-gated beats the incumbent at matched spend — the Super-GOAT label is EARNED ON THIS MODEL.** The label's boundary is the declared world: the corpus is planted (the probe reads exactly the planted structure through binomial noise — that is the instrument's design, T4.1), the scissors law is a DECLARED model constant (FlyBy-faithful, but a model), and the rescue/cost constants are documented priors. The production proof is the consumer wiring (T5.2 — riir-refine Issue 156's rescue predicate consuming `ProbeEstimate`), which re-gates on real fix-miss telemetry; **promotion to default waits for that consumer's gate, per flag discipline** — the feature stays `escalation_probe_gate` (opt-in), the incumbent cap-only rule survives as the G3 fallback.

## G2 cost cell (T4.3)

- **Probe wall cost:** 14.4 ns/state (median of 200 full-corpus passes — full adaptive ladder + gate, worst case) against the 5 µs/state ceiling — **~350× headroom**.
- **Declared-units budget ratio:** 478 evals × 10⁻⁴ units = 0.048 units vs the 10 % line at 6.0 units — **125× headroom**; the guard never breached (`budget_fallback: false`).
- **The breach→fallback path is TESTED** (`budget_breach_falls_back_to_the_incumbent_loudly`): an absurd per-eval cost trips the guard and the arm finishes under the incumbent rule with the flag SET — never a silent degradation.

**Box state disclosure (the G2 wall cell only):** M3 Max, AC power, **powermode = 2 (High Power)**, load 1-min 6.03 (a sibling build running — `bench_preflight.sh` REFUSED a publishable latency read). The number is recorded WITH that disclosure under T1.4's load-robustness reasoning: the bar is 5 µs, the reading 14.4 ns — 347× of headroom no scheduler tax closes. A quiet-box re-read rides the next preflight-green window (the re-read is a disclosure refresh, not a gate — the G2 gate's asserted line is the DECLARED-units ratio, which is posture-independent).

## Gates

- **G1** — truth tables + the two-class corpus + the floor-fires law: `state_probe_t4_escalation_eval` 6/6 PASS (this table is its print).
- **G2** — ≤ 1 µs bound math (T1.4, 167 ns/call) + the ≤ 10 % probe-cost budget with the fallback: PASS (above).
- **G3** — probe-off / λ = 0 byte-identical: default-posture lib **2151 passed 0 failed** (unchanged from the 622 landing); default-posture clippy 0; feature-posture clippy (`--all-targets --features escalation_probe_gate`) 0; Phase 1–2 suites re-run green (t1 9/9, t2 6/6).
- **G4** — the kernel's alloc gate (already pinned at Phase 1/3); the eval instrument is bench-side and allocates its corpus by design (setup, not the hot path).

## Provenance chain

Plan 621 Phase 1 (T1.1–T1.4, 2026-10-08) → Phase 2 + 3 (T2.1–T3.4, 2026-10-10) → **this record (Phase 4, 2026-10-11)** → Phase 5 consumers pending (T5.1 reflex lane, T5.2 refine Issue 156, T5.3 docs sync, T5.4 the promotion decision — owner-gated on the consumer's gate).
