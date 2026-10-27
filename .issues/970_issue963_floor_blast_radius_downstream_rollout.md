# Issue 970 — the 963 floor bump hard-broke every downstream pin that still reads 1.98.1: measured matrix + the T3-generalized rollout ledger

**Status:** OPEN — the rollout is the remedy; SIX repos landed (reflexer `7a6c255`, instinct `22fb8a9`, game-sdk `43ba18d`, dao `6f79dec`, riir-infer `31fd00a`, riir-ai `ce236e6bb` — the last MID-BOUT against the guidance below, caveat on its rollout row); the rest need their own lane's quiet touch or the one-line override. No anchor-side change requested (see "The floor-hold fallback" — declined unless the rollout stalls).
**Claimed:** zed-amo-idle-b309, 1793136990, expires 2026-10-29T21:36:30+00:00
(scope: the riir-mmorpg-examples rollout row only — the one unchecked cell landable from amo, which carries the checkout the m3 lane flagged absent)

**Lease closed:** zed-amo-idle-b309 (claimed 2026-10-28 ~04:4x +07) — the row below landed same-session `dce62c2`; the expiry token above is kept only as the claim's historical record.
**Date:** 2026-10-28
**Type:** cross-repo toolchain ripple (files against [Issue 963](963_toolchain_bump_1_99_0.md) T3's scope, not its anchor half)
**Discoverer:** zed-m3-floor-sweep (m3 idle cycle, 2026-10-28 ~03:1x–04:1x +07), while running the idle-loop clippy sweep — `riir-refine` and `riir-reflexer` builds failed with `error: rustc 1.98.1 is not supported by the following package: katgpt-device-verify@0.1.0 requires rustc 1.99.0`.

## The mechanism (one paragraph)

`8d54ddb96` (963 T1) raised `[workspace.package] rust-version` to **1.99.0** in the same commit as the channel bump. **25 of 34 katgpt-rs members** declare `rust-version.workspace = true` and inherit that floor (the 9 that carry none: assign, core, dec, hla, micro-belief, personality, sense, sleep, types). Cargo's rust-version gate fires at resolve/build-plan time for any package in the graph whose floor exceeds the active toolchain — so every consumer repo whose `rust-toolchain.toml` still pins `channel = "1.98.1"` fails any fresh-resolve build whose graph reaches a floored member. 963's T3 named **only** seal-remake + seal-game-editor as downstream syncs; the rest of the blast radius was untracked. This issue is that tracking.

## Measured matrix (fresh-resolve probes, 2026-10-28, m3)

| Repo | Pin | Verdict | Floored member in graph |
|---|---|---|---|
| riir-ai | **1.99.0** | **BROKEN → LANDED** (`ce236e6bb`, 2026-10-27T21:15:49Z — MID-BOUT, see the rollout row) | katgpt-attn + katgpt-attn-match |
| riir-refine | 1.98.1 | **BROKEN** (measured) | katgpt-device-verify (via riir-kat) |
| riir-auth | 1.98.1 | **BROKEN** (measured) | katgpt-device-verify (direct) |
| riir-kat | 1.98.1 | **BROKEN** (measured) | katgpt-device-verify (direct) |
| riir-reflexer | 1.98.1 | **BROKEN → FIXED** (`7a6c255`) | katgpt-tetris (direct, non-optional) |
| riir-instinct | 1.98.1 | default PASS, tetris posture BROKEN → **FIXED** (`22fb8a9`) | katgpt-tetris (opt-in `tetris` feature) |
| riir-infer | **1.99.0** | fine — LANDED `31fd00a` (was 1.98.1; the "already follows" cell was a misread, reconciled 2026-10-27T21:5xZ) | — |
| seal-remake | **1.99.0** | fine (m3 re-read confirms 1.99.0 at 21:5xZ — the 4090's 04:05 "1.98.1" was that box's stale checkout, their own note anticipated the reconcile) | — |
| riir-dao | 1.98.1 | PASS (measured, fresh target) | graph = protocol tier only (katgpt-core + riir-kat default-features=false; no floored member) |
| riir-chain | 1.98.1 | likely broken (untested) | katgpt-device-verify (`chain` feature dep) |
| riir-dapps | 1.98.1 | likely broken (untested) | katgpt-device-verify |
| riir-neuron-db | 1.98.1 | likely broken (untested) | katgpt-device-verify (git dep, branch main — the lock's rev matters; a lock refresh + the ai-side patch row re-points it at the local floored checkout) |
| riir-game-sdk | 1.98.1 | **BROKEN (measured, 2026-10-28) → FIXED** (`43ba18d`) | katgpt-transformer + katgpt-validator (measured resolve errors — not device-verify as first guessed) |
| riir-mmorpg-examples | 1.98.1 | likely broken (untested) | katgpt-device-verify |
| riir-rethink | 1.98.1 | likely broken (untested; sibling lane live) | katgpt-device-verify |
| seal-game-editor | 1.98.1 | **BROKEN (measured by the T3 lane) → FIXED** (`d9f15a4db`, guard PASSED first run) | via riir-neuron-db (artb) + riir-ai (static-ai) edges |
| riir-reflex | 1.98.1 | untested (sibling lane live — 013 T4 on amo) | katgpt-core only by default → may PASS; laya edge unresolved |
| riir-train | 1.98.1 | **BROKEN at all-targets** (measured 2026-10-27T21:45Z, m3: `cargo metadata` PASSES, `check --workspace --all-targets` fails resolve — the posture-dependence class; this row was MISSING from the matrix, gap closed) | 19 floored members at the all-targets layer (katgpt-attn/-match/backend/band/claim/deprecated/forward/kv/percepta/pruners/quant/ruliology/sparse/spectral/speculative/tokenizer/transformer/validator + katgpt-rs), via the riir-engine/riir-games*/riir-gpu path edges + dev-dep/feature-union resolution |
| riir-llm / riir-refab / riir-deployer / riir-shader / jup-tool | 1.98.1 (various) | PASS (no katgpt dep / no floored member) | — |
| riir-esp32 | (no file) | PASS | device-verify rides a **git pin** (`6f392727`, pre-floor manifest) — insulated until repinned |
| zed / katgpt-web | — | n/a | no katgpt dep |

Box coverage for the override: m3 rustup has 1.99.0 installed (stable default = 1.99.1); the 4090 has 1.99.0 installed; **amo VERIFIED 2026-10-27T22:2xZ (m3 970 lane, read-only ssh): 1.99.0 installed (stable default = 1.99.1)** — the riir-train row's "amo's 1.99.0 availability unverified" flag is RESOLVED; every remaining bump (riir-train at claim-close, migrate-plane after the 175 rehearsal, rethink/reflex at quiet touch, riir-refine at quiet touch) can execute on any of the three boxes. 4090 re-read same pass: 1.98.1 + 1.99.0 both installed (its `stable` alias is stale at 1.93.0 — irrelevant where rust-toolchain.toml pins the channel, but worth knowing before a bare-cargo run there).

## The urgent one: the league closure

riir-ai is measured broken at the ENGINE target — the plan-641 **T5 league re-cell builds fail** until riir-ai's pin follows. ⛔ **Do NOT bump riir-ai's pin mid-bout**: a toolchain flip between league cells changes codegen and invalidates cross-cell comparisons (the same reason 963 T3 deferred under live sibling sessions). The mid-lane unblock is the override, not a pin edit:

```sh
RUSTUP_TOOLCHAIN=1.99.0 cargo build ...   # per-invocation; byte-safe, no tree change
```

The pin follows on the lane's own quiet touch (the 963 anchor semantics, "downstream follows, never the reverse"). Warning row posted: riir-refine `.distill/001_mining_queue_snapshot.md` (`3718c8fc`).

**UPDATE 2026-10-27T21:5xZ (m3 970 lane):** the pin HAS followed — `ce236e6bb` (4090 idle lane, 21:15:49Z) — but **MID-BOUT, against the guidance above**: the plan-641 T5 re-cell claim (pushed 20:06:37Z) was live with gate2 running on amo when the flip landed, and the 4090's quiet adjudication missed the claim row. Contained while the amo checkout does not pull mid-gate (a build reads its own tree); the league lane was messaged; the rollout row carries the caveat and the residual.

## Rollout ledger (T3 generalized — tick per repo, each with its own validation)

- [x] riir-reflexer → 1.99.0 — `7a6c255` (clippy --all-targets clean; lib tests 17/17)
- [x] riir-instinct → 1.99.0 — `22fb8a9` (clippy clean at default AND `--features tetris` fresh-target; lib tests 103/103)
- [x] seal-remake — already at 1.99.0 (pre-963 follow; nothing to do; m3 re-read 21:5xZ confirms — the 4090's 04:05 "1.98.1" was that box's stale checkout)
- [x] riir-infer — LANDED by the same 4090 unit `31fd00a` (pin 1.98.1 → 1.99.0 + 3 sweep heals in the fresh riir-infer-dist sockopts crate; clippy --workspace --all-targets -D clean, lib tests 278/0) — the earlier "already 1.99.0" cell was a misread; reconciled
- [x] riir-ai — LANDED by the 4090 idle unit `ce236e6bb` (pin → 1.99.0 + the 75-site KeyBias consumer heal, ALL test-side; clippy --workspace --all-targets -D clean, riir-gpu lib 200/0, every healed cudarc GPU target PASS on the 4090). ⚠ **LANDED MID-BOUT**: the plan-641 T5 re-cell claim (zed-m3-641-t5 → zed-amo-641-t5, pushed 20:06:37Z) was live with gate2 running on amo — the 4090's quiet adjudication missed the claim row. Contained while the amo checkout does not pull mid-gate (league lane messaged 2026-10-27T21:5xZ); cross-cell G2 comparability holds iff amo does not pull before gate2 completes
- [ ] riir-refine — live T2.2 lease to 21:15Z 10-28 + B298 authoring on amo; bump at the next quiet touch
- [ ] riir-kat / riir-auth / riir-chain / riir-neuron-db / riir-dapps — the migrate/recovery rehearsal plane (refine 175 T7 live); bump together as one quiet group
- [ ] riir-train — **measured BROKEN at all-targets** (2026-10-27T21:45Z, m3 — see the new matrix row; the matrix had NO riir-train row before this, gap closed). NOT bumped this session: live foreign claims on the repo (amo issues 620/638, exp 2026-10-29T03:59Z) and amo's 1.99.0 availability unverified — the pin moves at the repo's lanes' next quiet touch on a box that has the toolchain; `RUSTUP_TOOLCHAIN=1.99.0` unblocks any mid-flight build where installed. **UPDATE 2026-10-27T22:2xZ (m3 970 lane): amo's 1.99.0 VERIFIED installed** (read-only ssh; stable default 1.99.1) — the availability flag is closed; 4090 re-confirmed (1.98.1 + 1.99.0, `stable` alias stale at 1.93.0). Claims still live at read time — do not bump under them (the mid-claim toolchain-flip class: their long GPU runs would eat a full workspace rebuild + codegen flip if they pull mid-claim).
- [x] riir-game-sdk — **measured BROKEN → FIXED** (`43ba18d`): fresh `cargo check --workspace` at the 1.98.1 pin failed resolve (`katgpt-transformer`/`katgpt-validator` require rustc 1.99.0 — the ledger's device-verify guess was the wrong floored member; measured ones recorded here); validation at 1.99.0 = ci_feature_guard --quick (L1 + the full L2 clippy posture series) PASS + lib tests 29/29 + ci_wasm32_guard 8/8 combos
- [x] riir-mmorpg-examples — **measured BROKEN → FIXED** (`dce62c2`, amo, 2026-10-28): the ledger's device-verify guess was wrong again — the MEASURED floored set is 20 members (spectral, speculative, tokenizer, transformer, validator, attn, attn-match, backend, band, claim, deprecated, device-verify, forward, kv, percepta, pruners, quant, ruliology, sparse, rs) and `cargo check --workspace` fails resolve at the DEFAULT posture; validation at 1.99.0 = `ci_feature_guard.sh --quick` ALL layers exit 0 + the three latent sibling-drift breaks the first all-targets compile surfaced fixed in the same commit (QuestKind::Fishing E0004 arm; hero_fusion/moe `goal_ticks` [u32;15]→HERO_GOAL_COUNT E0308; the needs_ruliology pin 48→49 for `ae6fde5c9`'s added test, unseen while the floor blocked the posture); warnings-only residue (Dex/SwarmState imports + a dead const at authority-off combos) left for the owning lane. NOT rustfmt'd: the tree is fleet-rustfmt-1.9.0-styled and the local 1.99 rustfmt reformats wholesale — hand-styled insertions only.
- [x] seal-game-editor — **LANDED by the 963 T3 lane** (`d9f15a4db`, pin → 1.99.0, its guard PASSED first run; seal-remake guard 163✓ ALL LAYERS GREEN at `657e1f45`; T3 arc closed `00a7246dc`): my coordination flag about the dirty tools/seal-e2e/Cargo.lock was consumed by that lane — disclosure on record there (the hunk is blaze-run-era churn, swept into their issue-139 commit `6ab1e05bf`, additive-only + `--locked` build-verified; no objection from this lane)
- [ ] riir-rethink / riir-reflex — the amo measurement lanes (032 claim to 12:00Z, 013 T4 running); bump at their quiet touch
- [x] riir-dao — **landed** (`6f79dec`, the opportunistic row discharged early): passed at 1.98.1 as measured, bumped preventively; validation at 1.99.0 = the repo's own ci_feature_guard ALL LAYERS PASS (direction gate + self-test + clippy ×2 + tests ×2 incl. the dapps contract pins)

## The floor-hold fallback (declined for now, recorded)

Holding `[workspace.package] rust-version` back at 1.98.1 while the channel stays 1.99.0 would instantly unbreak every downstream with a one-line anchor commit — and was declined because (a) the loud gate error is the honest signal and a floor below the true MSRV is a lie that fails obscurely later, (b) the rollout has a documented recipe and is already executing, and (c) the anchor's own protection value is real. Re-open this fallback ONLY if the rollout stalls past ~48 h with lanes blocked (the issue's status line is the place to record that).

## Acceptance

- [x] Every repo in the "likely broken" rows either bumped (with its own clippy+test evidence) or carrying a dated reason in this ledger. — sdk landed `43ba18d`; mmorpg-examples carries the not-mounted reason; sge is the live T3 claim's; rethink/reflex the amo lanes'
- [ ] The migrate-plane group lands as one commit train, after the 175 rehearsal closes.
- [ ] riir-ai's bump LANDED (`ce236e6bb`, guard green on the 4090) — but NOT in a post-bout quiet window (the T5 claim was live; mid-bout caveat on the rollout row). Fully discharged when gate2 completes WITHOUT a mid-run pull on amo.

## Related

- [Issue 963](963_toolchain_bump_1_99_0.md) — the anchor arc; T3's deferred recipe is this issue's mechanism
- Issue 739 — the pin's founding record (the anchor semantics)
- riir-refine `.distill/001_mining_queue_snapshot.md` row `3718c8fc` — the live-lane warning + this session's claim
