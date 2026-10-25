# Issue 963 — Toolchain bump 1.98.1 → 1.99.0 (+ rust-src): the anchor pin moves, blaze pilot consumes it

**Status:** OPEN — filed from the Claude verdict on seal-remake Proposal 009; implementation is its own reviewed arc. **Ordering:** arms A/B of the seal-remake pilot run on the CURRENT pin — this bump lands ONLY if arm B (cargo + dev-profile tweaks) does not clear the want-faster bar on its own.
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

- [ ] T1 Bump this repo's `rust-toolchain.toml` → `channel = "1.99.0"`, add `"rust-src"` to
      components (keep clippy/rustfmt). Own commit.
- [ ] T2 `scripts/full_gate.sh` (macOS, full claim) + the scoped core tests; fix every new lint
      in the same arc (the pin's documented discipline). Record the `-D` list impact — a lint
      with residual > 0 must NOT join the list.
- [ ] T3 Downstream pin sync in the same arc: seal-remake + seal-game-editor
      `rust-toolchain.toml` follow the anchor (each repo's own guard run).
- [ ] T4 Document the revert path (pin back to 1.98.1) — a bad 1.99.0 surface is a one-commit
      rollback, not an investigation.
- [ ] T5 The wasm32/x86_64 gates re-run at the new pin (Layer 2b/2c) — a toolchain bump is
      exactly the change those lanes exist to catch.

## Acceptance

Full gate green at 1.99.0 on this repo + both consumer repos' guards green, or the bump is
reverted and seal-remake Proposal 009 Phase 1 records the blocker.

## Related

- Issue 739 (the pin's founding record) · `.benchmarks/701_full_workspace_execution_pricing.md`
  (why the full test surface is priced, not run) · seal-remake Proposal 009 §Phase 1 (the
  drift rule in context)
