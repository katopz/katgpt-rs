# Bench 891 — ladder_gate GOAT (Issue 887: streak-gated advancement + corrective backtracking FSM)

**Status:** COMPLETE — G1 PASS (17/17 unit tests: streak/retreat properties, both negative controls, dwell ordering) · G2 PASS (0.8 ns/eval probe-free, 1.2 ns/eval retained k≤8, k64/k8 paired ratio 0.977 — linear scan confirmed) · G3 PASS (flag-off compiles to nothing; default suite 2063 passed unchanged; feature-on 2080) · G4 PASS (0 steady-state allocs over 10k driven evals, 1333 advances/retreats). Feature `ladder_gate` (katgpt-core) stays **OPT-IN** — no production consumer exists yet; promotion waits on the first consumer's GOAT (the 873/874/875 precedent). Consumers named in the issue: riir-clippy corpus staging (doctrine-gated on Issue 102's ≥3-generation fade clock), riir-reflex cal cadence (config-only), riir-ai CGSP zone unlock, riir-train ATC rig.

**Box:** M3 Max (16 cores), macOS 26.6.2, AC power (battery 98% charging), release profile, isolated `/tmp` target dir. G2 via the shared interleaved `ab_median_ratio` (30 rounds — the paired protocol; ±21.7% sequential-arm drift measured on this workspace) + `best_of_us` with `black_box` on result AND arguments.

## What was measured

`crates/katgpt-core/src/ladder_gate.rs` — the ATC stage controller (arXiv:2609.19717, Research 589): advance iff held-out accuracy ≥ τ for m consecutive evals (any sub-τ resets); at the completing eval re-probe every passed stage — a regression retreats the pointer to the SHALLOWEST failing stage (argmin).

- **G2a**: `on_eval` driven ladder (1000 evals, alternating pass/fail, fresh gate per rep): **0.8 ns/eval** — O(1) streak arithmetic, invisible against any eval's own cost.
- **G2b**: `on_eval_with_retention` (1000 evals, alternating probe shapes k=8/∅): **1.2 ns/eval**.
- **G2c**: k=64 vs k=8 retained scan, paired: **median 0.977** (min 0.945, max 1.045, 30/30 rounds survived) — the scan is linear with the base so small both arms measure the ladder drive, not the scan; blowup bar was 40×.
- **G4**: 10k steady-state evals, alternating probe shapes: **0 allocations** (counting allocator), 1333 non-Hold actions exercised both Advance and Retreat paths.

## G1 (quality, unit-gated in `ladder_gate::tests` — 17 tests)

- Streak properties: advance only on m-consecutive passes; a single sub-τ eval resets (the 91.8→99.9 pin); interleaved pass/fail never advances; non-finite/out-of-range evals are not confirmations (fail-closed).
- Retreat: argmin-shallowest-failing (three failing patterns); malformed probe slice → fail-closed to stage 1 (the paper's collapse warning mechanised); retreat at stage 1 is structurally Hold; `#[repr(u8)]` action discriminants pinned.
- **T3 negative controls (the paper ships them free)**: (a) a λ=0 toy two-stage ladder MUST trip the retention path (Retreat{1} emitted, gate back at stage 1, never passes stage 2 — the 8/8-arms class); (b) the SAME toy driven probe-free (`on_eval` only) climbs while stage-1 skill collapses 0.95 → <0.5 with no possible red — the 51% arm, pinned as the reason the retention path exists.
- **T4 dwell dominance**: on the deterministic toy (256-stage ladder, +0.1/eval train cap 0.95, −0.02/eval unrehearsed decay), the final foundation (min skill of stages 1-2) under (τ=0.9, m=5) strictly beats (τ=0.9, m=1); the rehearsed (λ=0.1) arm holds foundations ≥ τ outright. **Toy-ceiling caveat recorded in-test**: the paper's (τ=0.98, m=1) middle arm is excluded because the toy's 0.95 training cap makes τ=0.98 unmeetable — that arm measures the toy's ceiling, not the gate.

## Findings

1. **`#[repr(u8)]` fieldful enums cannot `as`-cast** — the discriminant wire-stability pin for `Retreat { to_stage }` goes through `PartialEq` + `mem::discriminant` instead (E0605; a unit-only enum cast would have worked but the payload is the API).
2. **Reading B pinned in code**: retention gates the ADVANCE (paper: "on advancement re-probe"), checked at the streak-completing eval only — a sub-τ current eval resets and holds without consulting probes. Documented on `on_eval_with_retention` + asserted, so a future "check every eval" refactor is a deliberate semantics change, not a drive-by.
3. **The toy ladder needs depth ≥ the eval budget** when m=1 can advance every eval — the first draft's 3-stage toy overflowed at stage 4. Deep-ladder tests need the 256-stage shape (or a consumer-owned cap).

## Gates

- G1: `ladder_gate::tests` (17 tests, unit-gated in-module — `cargo test -p katgpt-core --features ladder_gate --lib ladder_gate`).
- G2 + G4: this bench (`cargo test --release -p katgpt-core --features ladder_gate --test bench_887_ladder_gate_goat`).
- G3: flag-off `cargo check`/`clippy` clean at 0 warnings (module cfg'd out); default-features lib suite 2063 passed / 0 failed — unchanged floor; feature-on lib 2080 (+17).

## Posture

Opt-in (`ladder_gate = []`, zero deps); no default feature set changes; the module is leaf-clean (no katgpt-core cross-feature implications). Per the issue's T7: **GOAT PASS, stays opt-in** — the Feature-Flag modelless-gain rule needs a first consumer measuring the gain on a real ladder before any default promotion. Issue 887 remains the tracker; this record closes its T1–T7 implementation tasks with consumers explicitly out of scope.
