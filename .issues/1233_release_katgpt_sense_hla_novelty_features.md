# Issue 1233: release katgpt-sense + katgpt-hla so the Plan 639 features reach crates.io

**Status:** OPEN — ready, release-gated (the only gate is the release credential: `scripts/release.sh` runs release-plz + the CI publish job).
**Claimed:** m3-zed-1233 epoch=1795084180 expires=2026-11-21T10:29:40Z

## Why

Plan 639 (COMPLETE 2026-11-19) landed new opt-in surfaces that the published crates do not carry:

| crate | local | crates.io max | missing feature |
|---|---|---|---|
| katgpt-sense | 0.3.1 | 0.3.1 (2026-11-18) | `novelty_retention` — `NoveltyChannel` + the fused armed kernel (`f907edece`, `c24c8cce7`) |
| katgpt-hla | 0.1.4 | 0.1.4 (2026-11-10) | `novelty_gamma` (issue 1232, `91f80e745`) — depends on katgpt-sense's `novelty_retention` |
| katgpt-core | — | — | the `novelty_retention` forward (`f907edece`) |

Published versions are immutable, so this needs version bumps (release-plz does them) and **katgpt-sense must publish before katgpt-hla** (release-plz orders by the workspace graph). Off-crates.io consumers (riir-ai `desperation_dual_novelty` / `goal_salience_novelty`, riir-refine `evolve_novelty`) build from path deps and are unaffected.

## Tasks

- [ ] `./scripts/release.sh` from develop (needs `gh auth`; GitHub free-tier limits may force the manual path, as in AGENTS/CLAUDE notes).
- [ ] Verify on crates.io: katgpt-sense ≥ 0.3.2 lists `novelty_retention`; katgpt-hla ≥ 0.1.5 lists `novelty_gamma`; katgpt-core's new version lists `novelty_retention`.

Local readiness: `cargo publish --dry-run -p katgpt-sense` packages and builds clean at `b0f841a44` (2026-11-19).
