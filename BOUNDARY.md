# katgpt-rs — boundary contract

> The single source of truth for what may live in and depend on this repo.
> Audited by the `boundary-guard` skill + two scripts:
> `../riir-ai/scripts/ci_boundary_contract.sh` (workspace — is the dep graph
> what this file says, and is this file still honest?) and
> `../riir-mmorpg-examples/scripts/ci_boundary_guard.sh` (per-repo — is the
> CODE in the right repo?). Cross-repo rules LINK to their one
> canonical home — never copied. Rollout record:
> `../riir-ai/.docs/01_orientation/737_boundary_contract_rollout_record.md`.

## Owns

- **Public modelless inference primitives** (`katgpt-core` + member crates) —
  the public funnel per Research 003. No training, no backprop, no gradient
  descent; runtime weight mutations limited to freeze/thaw, deterministic
  raw/lora hot-swap, latent-space updates. Includes the zero-dep wire-format
  modules: `decision_wire` (choice/score/noul, v1 frozen additive-only) and
  `runetrace` (the scene-as-text DOC model + byte-deterministic renderer +
  BLAKE3 digest, Plan 624 — the format half of seal-remake Proposal 005,
  ACCEPTED 2026-10-09; consumers: riir-ai Plan 626 producers, the Phase 3
  panels, the Phase 4 instinct lane per Research 613).
- **Spec-driven constrained assignment solver** (`katgpt-assign`, Plan 620 /
  Research 607 — the OSDI'24 Rebalancer distillation): objects → containers
  under capacity/balance/movement specs over an integer expression DAG with
  strict-improvement local search and byte-identical per-seed determinism.
  Admitted on the domain test read as written: *modelless*, *zero deps*, no
  riir dep, upstream of everything. It is an **optimization/verification**
  primitive rather than an *inference* primitive — a widening of the Owns
  line recorded here rather than waved through (the `katgpt-device-verify`
  precedent); the load-bearing half is the same: integer-deterministic output
  fit for the consensus replay path (consumer riir-chain Issue 164, Phase 3
  of the plan — its own repo's issue, not this repo's code).
- **`no_std` device-side verification primitives** (`katgpt-device-verify`) —
  fair-roll die verification and binary-Merkle inclusion verification for the
  `Satellite` tier (`riir-chain` Issues 108/109, this repo's Issue 685; the
  device side of that tier moved 2026-09-06 to the private POC repo
  `riir-esp32`, which consumes this crate by git pin — the chain node and the
  device still resolve the SAME fair-roll implementation from here).
  Admitted 2026-08-24 on the domain test read as written: *modelless*, *no riir
  dep*, and this repo is upstream of everything. It is a **verification**
  rather than an *inference* primitive, which is a widening of the Owns line
  above and is recorded as such rather than waved through — the load-bearing
  half of the test is the dependency direction, and the alternatives
  (`riir-chain`, `riir-neuron-db`) are both downstream, both std-only, and
  both drag a 100 K-LOC manifest into every MCU build's resolution (the Plan
  040 recursive-path-dep lesson). One dep: `blake3`, `default-features =
  false`. No `getrandom` — verification consumes entropy, never produces it.
- Workspace-level agent skills in `.agents/skills/` (boundary-guard,
  substrate-first, feature-gate-audit, goat-audit, proposal, research) — these
  are workspace *tooling*, not crate surface.

## Does not own

Everything downstream is a consumer, never a dependency:

| Concern | Correct home |
|---|---|
| Game runtime / cognition wiring | `../riir-ai` |
| Chain / storage / training / SDK / dApps | the respective `../riir-*` repos |
| The trained-artifact lane (the `artifacts/` convention: classes, manifests, protected remote lanes) | the artifact-owning repos (`../riir-train`, `../riir-instinct`, `../riir-rethink`; the gate lives in `../riir-ai`, the vocabulary in `../riir-reflexer`) — this repo carries NO artifact lane; its fixtures are committed repo bytes |

## May depend on

| Crate | Location | Condition |
|---|---|---|
| — | — | **Nothing workspace-internal.** katgpt-rs is UPSTREAM of riir-ai; any riir dep here is a dependency cycle. |
| `blake3` | crates.io | `katgpt-device-verify` pins `default-features = false` (pure-Rust core-only). The `std` feature is host/CI only; a device build must not enable it. |
| `bevy_ecs` | crates.io | OPTIONAL only (features `bomber` / `monopoly` / `bomber-wasm`), used as a schedule-free data-structure layer (`World` + `query{,_filtered}` + `Messages` drain + derives; zero scheduler/Commands/change-detection — Issue 799 API audit). The bomber/monopoly arenas are GOAT evidence infrastructure (Bench 432 kernel_blend gates measured on bomber tournaments) and pass the domain test: an evaluation harness with no riir dep, upstream of everything. Pinned at the workspace 0.19 wave (bumped 0.15→0.19, Issue 799; measured: game semantics byte-identical before/after, `run_tick` World paths ~1.8× slower — floors still pass with ≥10× headroom, see the Bench 799 record). |
| `syn` | crates.io | OPTIONAL only (feature `canon_source_features` in `katgpt-canon`, Issue 867 T1): Rust-source parsing for the AST node-type histogram extractor — setup-time only, NEVER on a per-token path. Opt-in feature, default build's dep graph unchanged (syn was already transitive in the lock; the direct edge activates nothing without the feature). |

### Consumers of `katgpt-device-verify` (outbound, informational)

Not a dependency of this repo — recorded so the drift invariant has a home.
`riir-chain` (fair roll) and `riir-neuron-db` (Merkle) each own an
implementation this crate mirrors, and each asserts the **same pinned vector
fixture** by value. Copying the code across the seam is what creates drift;
copying the fixture is what detects it.

The consume step **landed 2026-08-24** (`develop` → `main` fast-forward,
`51be354a` → `4d6749fa`). `riir-chain`'s `FairRng` delegates to
`FairRollVerifier`; `riir-neuron-db/src/merkle.rs` `pub use`s the verify half.
Exactly one implementation of each now exists, and it is the one a device
links. The shared fixture remains the gate that would catch a re-divergence —
it was written against the pre-move implementations and still passes against
the delegated ones, which is what proved the move behaviour-preserving.

## Inherited boundaries (links)

- Cross-repo dep direction: `../riir-ai/BOUNDARY.md`
- Chain admission (three-test): `../riir-chain/BOUNDARY.md`
- Public/private split doctrine (what may be public at all): `../riir-ai/.research/003_Commercial_Open_Source_Strategy_Verdict.md` — Research 003 §"Decision Rules for AI": what = public, how = private; private-workspace governance/ops content never lands here; when unsure, keep it private.

## Drift ledger (target vs actual)

None. (Clean at last guard run.)

## WASM / wasmi compatibility contract (audited 2026-08-23)

- Targets: `wasm32-unknown-unknown` (browser + CF Worker via wasm-bindgen) and
  `wasm32-wasip2` (wasmtime-run benches). Both verified clean via
  `cargo check -p katgpt-core --target <t>` with `+simd128` (2026-08-23).
- **simd128 rule (the Issue 205 lesson)**: every wasm build that runs perf
  kernels MUST pass `-C target-feature=+simd128` — without it the SIMD paths
  silently compile the scalar fallback (~16× slower; encoded in
  `scripts/build-moka-wasm.sh`) — and `wasm-opt --enable-simd` at the
  optimize step (without it SIMD ops are stripped).
- `getrandom` wasm backends (0.2 `js` + 0.3 `wasm_js` + 0.4 `wasm_js`) stay
  pinned at workspace level — transitive bevy/uuid consumers break without
  them (the 0.4 pin added with the bevy_ecs 0.19 bump, Issue 799).
- **wasmi hosts carry `features = ["simd"]`** so they load modules built
  under any sibling workspace's `+simd128` rustflag. One version-aligned
  wasmi 1.x in every tree ("one wasmi" rule).
- Footgun: env `RUSTFLAGS` REPLACES `.cargo/config.toml` rustflags (they do
  not merge) — include every needed flag when overriding.
