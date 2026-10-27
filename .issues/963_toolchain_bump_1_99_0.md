# Issue 963 — Toolchain bump 1.98.1 → 1.99.0 (+ rust-src): the anchor pin moves, blaze pilot consumes it

**Status:** ARC LANDED 2026-10-28 at the anchor (katgpt-rs) — T1/T2/T4/T5 done, evidence in HISTORY.md (full gate green at 1.99.0, 0 new lints). **T3 (seal-remake + seal-game-editor downstream pins) DEFERRED**: both repos had a live sibling session mid-flight at claim time (seal-remake writes 02:41, sge 01:51–02:28, 2026-10-28) — a toolchain flip under a running guard would split its layers across two rustc versions and corrupt the verdict. The pins follow on their next quiet touch (anchor semantics: downstream follows, never the reverse). Re-open T3 by re-running this arc's last step: bump both `rust-toolchain.toml` → 1.99.0 (+ rust-src optional there — only blaze needs it, and the pilot is decoupled) + each repo's own guard run.
**Claimed:** zed-m3-t3-toolchain-sync, 1793132071, expires 2026-10-28T23:01:11Z
T3 re-open claim: both repos quiet (no cargo procs, HEAD==origin) at re-claim time.
**Lease released:** katgpt-rs-963-bump (idle loop), claimed 1793145600 — the anchor half landed `8d54ddb96`; T3 deferred for sibling-session safety, not abandoned
**Date:** 2026-10-25
**Type:** refactor / toolchain (the pin authority lives HERE — Issue 739 lineage)
**Consumers:** [seal-remake Proposal 009](../../seal-remake/.proposals/009_blaze_dev_loop_build_driver.md) + [Issue 126](../../seal-remake/.issues/126_blaze_dev_loop_pilot.md) (the blaze dev-loop pilot needs exactly 1.99.0 + rust-src; allocated as Issue 122, renumbered 126 before landing) · seal-game-editor + seal-remake follow this anchor

---

## Summary

The workspace anchor pin (`rust-toolchain.toml`, Issue 739) is 1.98.1. blaze
(indexable-inc/blaze — the incremental build daemon the seal-remake pilot proposes) works with
exactly ONE rustc: 1.99.0, plus the `rust-src` component. Moving the anchor moves `full_gate`
and every lockstep pin — so the bump is a deliberate arc in THIS repo, and the downstream pins
follow it, never the reverse.

**The drift rule (permanent, one-directional):** the workspace pin NEVER follows blaze's
supported version. When blaze's pin drifts from ours, the pilot PAUSES. A dev-loop tool must
never own the toolchain.

The bump is independently worth landing even if the pilot is rejected — it is a normal
toolchain step (1.99.0 shipped its own improvements), and the pin's own comment already mandates
the shape: raise in its own reviewed commit and fix the new lints there.

## Tasks

- [x] T1 Bump this repo's `rust-toolchain.toml` → `channel = "1.99.0"`, add `"rust-src"` to
      components (keep clippy/rustfmt). Own commit. — LANDED `8d54ddb96` with the `rust-version` floor raised in the same commit.
- [x] T2 `scripts/full_gate.sh` (macOS, full claim) + the scoped core tests; fix every new lint
      in the same arc (the pin's documented discipline). Record the `-D` list impact — a lint
      with residual > 0 must NOT join the list. — GREEN with real evidence (the warm-cache
      INCONCLUSIVE discharged by a fingerprint wipe): 347 units, 0 warnings, 0 unbuildable;
      scoped core 2167/0. **No new lint fires at 1.99.0 on this tree — the `-D` list is
      unchanged** (zero residuals to add; zero unknown-lint errors, so nothing to drop).
- [ ] T3 Downstream pin sync in the same arc: seal-remake + seal-game-editor
      `rust-toolchain.toml` follow the anchor (each repo's own guard run). — DEFERRED (live
      sibling sessions in both repos at claim time; see Status for the re-open recipe).
- [x] T4 Document the revert path (pin back to 1.98.1) — a bad 1.99.0 surface is a one-commit
      rollback, not an investigation. — In HISTORY.md: `git revert` of the bump commit; pin +
      floor move together, one commit.
- [x] T5 The wasm32/x86_64 gates re-run at the new pin (Layer 2b/2c) — a toolchain bump is
      exactly the change those lanes exist to catch. — Both green, both arms, triples disclosed
      on the verdict lines (wasm32 simd128 on/off; x86_64-apple-darwin avx2 on/off).

## Acceptance

Full gate green at 1.99.0 on this repo + both consumer repos' guards green, or the bump is
reverted and seal-remake Proposal 009 Phase 1 records the blocker.

## Related

- Issue 739 (the pin's founding record) · `.benchmarks/701_full_workspace_execution_pricing.md`
  (why the full test surface is priced, not run) · seal-remake Proposal 009 §Phase 1 (the
  drift rule in context)
