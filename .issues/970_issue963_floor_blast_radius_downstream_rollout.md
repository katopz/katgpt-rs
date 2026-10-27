# Issue 970 — the 963 floor bump hard-broke every downstream pin that still reads 1.98.1: measured matrix + the T3-generalized rollout ledger

**Status:** OPEN — the rollout is the remedy; two quiet repos landed same-session (reflexer `7a6c255`, instinct `22fb8a9`); every other broken repo needs its own lane's quiet touch or the one-line override. No anchor-side change requested (see "The floor-hold fallback" — declined unless the rollout stalls).
**Claimed:** zed-amo-idle-b309, 1793136990, expires 2026-10-29T21:36:30+00:00
(scope: the riir-mmorpg-examples rollout row only — the one unchecked cell landable from amo, which carries the checkout the m3 lane flagged absent)
**Date:** 2026-10-28
**Type:** cross-repo toolchain ripple (files against [Issue 963](963_toolchain_bump_1_99_0.md) T3's scope, not its anchor half)
**Discoverer:** zed-m3-floor-sweep (m3 idle cycle, 2026-10-28 ~03:1x–04:1x +07), while running the idle-loop clippy sweep — `riir-refine` and `riir-reflexer` builds failed with `error: rustc 1.98.1 is not supported by the following package: katgpt-device-verify@0.1.0 requires rustc 1.99.0`.

## The mechanism (one paragraph)

`8d54ddb96` (963 T1) raised `[workspace.package] rust-version` to **1.99.0** in the same commit as the channel bump. **25 of 34 katgpt-rs members** declare `rust-version.workspace = true` and inherit that floor (the 9 that carry none: assign, core, dec, hla, micro-belief, personality, sense, sleep, types). Cargo's rust-version gate fires at resolve/build-plan time for any package in the graph whose floor exceeds the active toolchain — so every consumer repo whose `rust-toolchain.toml` still pins `channel = "1.98.1"` fails any fresh-resolve build whose graph reaches a floored member. 963's T3 named **only** seal-remake + seal-game-editor as downstream syncs; the rest of the blast radius was untracked. This issue is that tracking.

## Measured matrix (fresh-resolve probes, 2026-10-28, m3)

| Repo | Pin | Verdict | Floored member in graph |
|---|---|---|---|
| riir-ai | 1.98.1 | **BROKEN** (measured, `cargo check -p riir-engine --lib`) | katgpt-attn + katgpt-attn-match |
| riir-refine | 1.98.1 | **BROKEN** (measured) | katgpt-device-verify (via riir-kat) |
| riir-auth | 1.98.1 | **BROKEN** (measured) | katgpt-device-verify (direct) |
| riir-kat | 1.98.1 | **BROKEN** (measured) | katgpt-device-verify (direct) |
| riir-reflexer | 1.98.1 | **BROKEN → FIXED** (`7a6c255`) | katgpt-tetris (direct, non-optional) |
| riir-instinct | 1.98.1 | default PASS, tetris posture BROKEN → **FIXED** (`22fb8a9`) | katgpt-tetris (opt-in `tetris` feature) |
| riir-infer | **1.99.0** | fine | — (already follows the anchor) |
| seal-remake | **1.99.0** | fine | — (already follows) |
| riir-dao | 1.98.1 | PASS (measured, fresh target) | graph = protocol tier only (katgpt-core + riir-kat default-features=false; no floored member) |
| riir-chain | 1.98.1 | likely broken (untested) | katgpt-device-verify (`chain` feature dep) |
| riir-dapps | 1.98.1 | likely broken (untested) | katgpt-device-verify |
| riir-neuron-db | 1.98.1 | likely broken (untested) | katgpt-device-verify (git dep, branch main — the lock's rev matters; a lock refresh + the ai-side patch row re-points it at the local floored checkout) |
| riir-game-sdk | 1.98.1 | likely broken (untested) | katgpt-device-verify |
| riir-mmorpg-examples | 1.98.1 | likely broken (untested) | katgpt-device-verify |
| riir-rethink | 1.98.1 | likely broken (untested; sibling lane live) | katgpt-device-verify |
| seal-game-editor | 1.98.1 | likely broken (untested; 963 T3 already defers it) | via riir-neuron-db (artb) + riir-ai (static-ai) edges |
| riir-reflex | 1.98.1 | untested (sibling lane live — 013 T4 on amo) | katgpt-core only by default → may PASS; laya edge unresolved |
| riir-llm / riir-refab / riir-deployer / riir-shader / jup-tool | 1.98.1 (various) | PASS (no katgpt dep / no floored member) | — |
| riir-esp32 | (no file) | PASS | device-verify rides a **git pin** (`6f392727`, pre-floor manifest) — insulated until repinned |
| zed / katgpt-web | — | n/a | no katgpt dep |

Box coverage for the override: m3 rustup has 1.99.0 installed (stable default = 1.99.1); the 4090 has 1.99.0 installed.

## The urgent one: the league closure

riir-ai is measured broken at the ENGINE target — the plan-641 **T5 league re-cell builds fail** until riir-ai's pin follows. ⛔ **Do NOT bump riir-ai's pin mid-bout**: a toolchain flip between league cells changes codegen and invalidates cross-cell comparisons (the same reason 963 T3 deferred under live sibling sessions). The mid-lane unblock is the override, not a pin edit:

```sh
RUSTUP_TOOLCHAIN=1.99.0 cargo build ...   # per-invocation; byte-safe, no tree change
```

The pin follows on the lane's own quiet touch (the 963 anchor semantics, "downstream follows, never the reverse"). Warning row posted: riir-refine `.distill/001_mining_queue_snapshot.md` (`3718c8fc`).

## Rollout ledger (T3 generalized — tick per repo, each with its own validation)

- [x] riir-reflexer → 1.99.0 — `7a6c255` (clippy --all-targets clean; lib tests 17/17)
- [x] riir-instinct → 1.99.0 — `22fb8a9` (clippy clean at default AND `--features tetris` fresh-target; lib tests 103/103)
- [x] seal-remake — already at 1.99.0 (pre-963 follow; nothing to do)
- [x] riir-infer — already at 1.99.0 (nothing to do)
- [ ] riir-ai — **post-bout only** (live league lane; override until then; guard run after)
- [ ] riir-refine — live T2.2 lease to 21:15Z 10-28 + B298 authoring on amo; bump at the next quiet touch
- [ ] riir-kat / riir-auth / riir-chain / riir-neuron-db / riir-dapps — the migrate/recovery rehearsal plane (refine 175 T7 live); bump together as one quiet group
- [x] riir-game-sdk — **measured BROKEN → FIXED** (`43ba18d`): fresh `cargo check --workspace` at the 1.98.1 pin failed resolve (`katgpt-transformer`/`katgpt-validator` require rustc 1.99.0 — the ledger's device-verify guess was the wrong floored member; measured ones recorded here); validation at 1.99.0 = ci_feature_guard --quick (L1 + the full L2 clippy posture series) PASS + lib tests 29/29 + ci_wasm32_guard 8/8 combos
- [ ] riir-mmorpg-examples — NOT MOUNTED on the m3 box (absent from the workspace roots, 2026-10-28): the bump must land from a box that carries the checkout (it still pins 1.98.1 per the matrix above and its graph reaches floored members like sdk's — expect the same resolve failure)
- [x] seal-game-editor — live 963 T3 claim (zed-m3-t3-toolchain-sync, expires 2026-10-28T23:01Z): not this session's row
- [ ] riir-rethink / riir-reflex — the amo measurement lanes (032 claim to 12:00Z, 013 T4 running); bump at their quiet touch
- [x] riir-dao — **landed** (`6f79dec`, the opportunistic row discharged early): passed at 1.98.1 as measured, bumped preventively; validation at 1.99.0 = the repo's own ci_feature_guard ALL LAYERS PASS (direction gate + self-test + clippy ×2 + tests ×2 incl. the dapps contract pins)

## The floor-hold fallback (declined for now, recorded)

Holding `[workspace.package] rust-version` back at 1.98.1 while the channel stays 1.99.0 would instantly unbreak every downstream with a one-line anchor commit — and was declined because (a) the loud gate error is the honest signal and a floor below the true MSRV is a lie that fails obscurely later, (b) the rollout has a documented recipe and is already executing, and (c) the anchor's own protection value is real. Re-open this fallback ONLY if the rollout stalls past ~48 h with lanes blocked (the issue's status line is the place to record that).

## Acceptance

- [x] Every repo in the "likely broken" rows either bumped (with its own clippy+test evidence) or carrying a dated reason in this ledger. — sdk landed `43ba18d`; mmorpg-examples carries the not-mounted reason; sge is the live T3 claim's; rethink/reflex the amo lanes'
- [ ] The migrate-plane group lands as one commit train, after the 175 rehearsal closes.
- [ ] riir-ai's bump lands with the next post-bout quiet window, followed by its own guard.

## Related

- [Issue 963](963_toolchain_bump_1_99_0.md) — the anchor arc; T3's deferred recipe is this issue's mechanism
- Issue 739 — the pin's founding record (the anchor semantics)
- riir-refine `.distill/001_mining_queue_snapshot.md` row `3718c8fc` — the live-lane warning + this session's claim
