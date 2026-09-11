# Bench 707 — Saddle-Trap Escape Gate GOAT (Plan 593 / Research 546)

**Status:** GOAT G1–G5 ALL PASS (2026-09-11) — feature stays **opt-in**
(promotion owner-gated per the 525/Bench-834 precedent; a real-model-trace
demonstration is the remaining promotion condition). The demonstration is
**AUTHORIZED 2026-09-11** (owner call) — schedule in the next idle M3 window;
promotion still requires it to pass (no toy-only promotion).

**Source paper:** arXiv:2609.04963 "Fractal basins trap latent reasoning"
(Lai, Bao, Quinn, Gilpin; UT Austin, Sep 2026).
**Primitive:** `katgpt_core::saddle_escape` — `SaddleEscapeGate` wrapping
(opt-in `saddle_escape = ["gain_cost_halt"]`) the Plan-304
`GainCostLoopHalter`: three-way Continue / Halt{Converged} / Halt{Trapped} /
Kick control with a decode-flip-rate EMA trap detector and a deterministic
BLAKE3-seeded, budget-bounded escape kick.

## Gates

| Gate | Budget | Measured | Verdict |
|---|---|---|---|
| G1a Toy A escape rate ≥ both baselines (K=200 trapped ICs) | gate ≥ halt-only AND ≥ noise | **gate 100.0%** vs halt-only **0.0%** vs always-on-noise 100.0% (tie; noise spends 700 perturbations vs gate's 200, and 100/200 noise halts carry flipping decodes) | ✅ PASS |
| G1b halt-only anchor (defend-wrong) | halt-only solves ~0% | **0/200** — every halt lands in the saddle band with an alternating decode (the two-way family's failure mode) | ✅ PASS |
| G1c Trapped-halt honesty (both toys) | trapped ⊆ halted-in-trap-region; zero Trapped/kicks on clean cohorts | Toy A: 0 trapped, 0 in-trap halts; Toy B: **59 trapped, all 59 in-band**; clean cohorts (K=200 A + K=400 B): 0 Trapped, **0 kicks** | ✅ PASS |
| G2 `decide()` Continue path | ≤ 500 ns/loop | **5.3 ns/loop** (94× headroom) | ✅ PASS |
| G2 `decide()` trap path (seed + eps) | ≤ 5 µs/loop | **21.9 ns/loop** | ✅ PASS |
| G2 `apply_kick` d=1024 | reported (rare: ≤ kick_budget/episode) | 7.84 µs/kick | reported |
| G3 no-regression | halter suite green; default surface compiles the module to nothing | 39/39 `gain_cost_halt` tests PASS; default-feature `cargo check` clean | ✅ PASS |
| G4 alloc-free | 0 allocs steady-state (Continue + Kick emission paths) | **0 bytes** over 2048 decides incl. Kick emissions | ✅ PASS |
| G5 bit-reproducible | identical episodes → identical decisions | seed + eps(`to_bits`) + variants **bit-identical** | ✅ PASS |

## Phase-3 defend-wrong PoC verdict tables (T3.1/T3.2, K=200 each)

`tests/saddle_escape_poc.rs` (deterministic — no RNG beyond BLAKE3 streams):

**Toy A — double-well + saddle-seeking trap band** (1-D; band `|x|<0.5` is a
sustained 2-cycle `x'=-x` whose sign-decode alternates every loop; wells at
±1 approached over-damped; eps0=1.0, kick budget 2):

| competitor | solve | loops(avg) | perturbs | kicks | trapped | flips@halt | in-trap |
|---|---|---|---|---|---|---|---|
| gate | **100.0%** | 5.5 | 200 | 200 | 0 | 200* | 0 |
| halt-only | 0.0% | 4.0 | 0 | 0 | 0 | 200 | 200 |
| always-on-noise | 100.0% | 4.5 | 700 | — | 0 | 100 | 0 |

*The gate's solved runs all halt ONE loop after the crossing flip (the
improvement event); noise's 100/200 flips are mid-wander instability — the
decode at halt was still changing.

**Toy B — decode-keyed flip ring** (2-D; parity-driven stable 2-cycle
straddling decode cells, band `|x−y|<0.2`, solution basin contracts to
(0.05, 0.75); eps0=0.35, budget 2):

| competitor | solve | loops(avg) | perturbs | kicks | trapped | flips@halt | in-trap |
|---|---|---|---|---|---|---|---|
| gate | **45.5%** | 5.7 | 310 | 310 | 59 | 191 | 109 |
| halt-only | 0.0% | 4.0 | 0 | 0 | 0 | 200 | 200 |
| always-on-noise | 100.0% | 4.0 | 600 | — | 0 | 200 | 0 |

### Toy B honest caveats (recorded, not revised)

- The gate's 45.5% is the HARD-difficulty result: half of successful kicks
  escape into the basin whose contraction path RE-CROSSES the band
  (geometry), and the budget exhausts honestly (`Halt{Trapped}`, all 59
  verified in-band at halt).
- Always-on noise wins Toy B's raw in-basin metric because the thin band
  does not tax per-loop perturbation — it exits by luck every loop. Its
  answers are not committed (flips@halt 200/200; unbounded 600
  perturbations). The plan's Toy B gate (T3.2) is measure + honesty, which
  this records; the ≥-both gate is Toy A's (passed, tie-with-noise allowed).
- The differentiated claim (aimed budget beats wasted perturbation as trap
  difficulty rises) is supported analytically (per-kick exit probability ×
  budget vs per-loop re-entry) but NOT asserted here — a real-model-trace
  demonstration is the promotion condition (T4.3).

## Commands

```bash
CARGO_TARGET_DIR=/tmp/plan593 cargo test -p katgpt-core --features saddle_escape \
  --test saddle_escape_poc --test saddle_escape_alloc_check -- --nocapture
CARGO_TARGET_DIR=/tmp/plan593 cargo bench -p katgpt-core --features saddle_escape \
  --bench saddle_escape_bench -- --nocapture
```

Machine: M3 Max (this box), quiet tree. GOAT gate run 2026-09-11.
