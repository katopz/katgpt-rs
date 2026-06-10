# 002 Deprecate & Remove py_to_rs Domain

`py_to_rs` transpiler is stale, blocked, and won't ship. Solely focus on mmorpg.
Release current develop (with py2rs intact) to main and tag it, THEN remove.
Everything py2rs vanishes — code, docs, plans, config. No crumbs.

## Scope

**Repos:** `katgpt-rs` (first), `riir-ai` (second)
**Branch:** `develop` → release to `main`, tag, then clean up on `develop`

## Tasks

### Phase 1: Git Flow Release — katgpt-rs (preserve py2rs snapshot)
- [x] `git checkout main && git merge develop --no-ff` (or ff-only if safe)
- [x] `git tag py_to_rs` on main (archive point — all py2rs code preserved here)
- [x] `git push origin main --follow-tags`

### Phase 2: Git Flow Release — riir-ai (preserve py2rs snapshot)
- [x] `git checkout main && git merge develop --no-ff`
- [x] `git tag py_to_rs` on main
- [x] `git push origin main --follow-tags`

### Phase 3: Remove Code — katgpt-rs (back on develop)
- [x] `git checkout develop`
- [x] Delete `examples/hello_py2rs.rs`
- [x] Remove `[[example]] name = "hello_py2rs"` from `Cargo.toml`
- [x] Remove `[[domain]] name = "py2rs"` block from `domains.toml`
- [x] Remove py2rs entries from `examples/README.md` (catalog row + getting started section)
- [x] Scrub py2rs refs from `.docs/01_overview.md`
- [x] Scrub py2rs refs from `.docs/02_architecture.md`
- [x] Scrub py2rs refs from `.plans/025_bidirectional_prefill_lora_switch.md`
- [x] Scrub py2rs refs from `.plans/029_agentic_streaming_lessons.md`
- [x] Scrub py2rs refs from `.plans/042_ttt_inspired_e2e_feedback_loop.md`
- [x] Scrub py2rs refs from `.plans/048_research_audit_fixes.md`
- [x] Scrub py2rs refs from `.plans/065_autogo_distillation.md`
- [x] Scrub py2rs refs from `.research/021_G-Zero_Self-Play_Open-Ended_Generation.md`
- [x] Scrub py2rs refs from `.research/033_autogo_distillation_strategy.md`
- [x] `cargo check --workspace`
- [x] `cargo test --quiet --workspace`
- [ ] Commit: `refactor: remove deprecated py2rs domain (archived at tag py_to_rs)`

### Phase 4: Remove Code — riir-ai (back on develop)
- [x] `git checkout develop`
- [x] Delete `crates/riir-transpiler/` directory
- [x] Remove commented `# "crates/riir-transpiler"` from workspace `Cargo.toml`
- [x] Delete `.docs/07_transpiler.md` (entirely py2rs)
- [x] Delete `.docs/39_python_rust_pipeline.md` (entirely py2rs)
- [x] Scrub py2rs refs from `.docs/01_overview.md`
- [x] Scrub py2rs refs from `.docs/05_router.md`
- [x] Scrub py2rs refs from `.docs/08_examples.md`
- [x] Scrub py2rs refs from `.docs/09_training_data_pipeline.md`
- [x] Scrub py2rs refs from `.docs/10_diagrams.md`
- [x] Scrub py2rs refs from `.docs/14_mtp_domain_config.md`
- [x] Scrub py2rs refs from `.docs/29_wasmi_migration_proof.md`
- [x] Delete `.plans/003_e2e_py2rs_pipeline.md` (entirely py2rs)
- [x] Scrub py2rs refs from `.plans/025_model_vs_modelless_bandit.md`
- [x] Scrub py2rs refs from `.plans/026_autotts_dynamic_inference_budget.md`
- [x] Scrub py2rs refs from `.plans/045_bomber_tech_isolation_ab.md`
- [x] Delete `crates/riir-validator-sdk/examples/python_validator.rs` — pure py2rs, no game use
- [x] Delete `crates/riir-validator-sdk/examples/rust_validator.rs` — code-gen validation, not game
- [x] Delete `crates/riir-validator-sdk/examples/rust_validator_simd.rs` — simd variant of above
- [x] Remove `python_validator` / `rust_validator` / `rust_validator_simd` `[[example]]` entries from `riir-validator-sdk/Cargo.toml`
- [x] Update `bandit_with_real_model_demo.rs` — replace `py2rs_lora.bin` + `rust_validator.wasm` refs with generic game lora/game validator
- [x] Scrub py2rs / python_validator / rust_validator refs from `.docs/02_wasm_validator_sdk.md`
- [x] `cargo check --workspace`
- [x] `cargo test --quiet --workspace`
- [ ] Commit: `refactor: remove deprecated py2rs domain (archived at tag py_to_rs)`

## What stays (game validators only)
- `riir-validator-sdk/examples/bracket_validator.rs` — generic, used by game validators internally
- `riir-validator-sdk/examples/keyword_validator.rs` — generic
- `riir-validator-sdk/examples/game_action_validator.rs` — game domain
- `riir-validator-sdk/examples/quest_fsm_validator.rs` — game domain
- `riir-validator-sdk/examples/bomber_validator.rs` — game domain
- `LoraAdapter` type — generic, used by all domains
- `bandit_with_real_model_demo.rs` — stays, refs updated to game lora/game validator

## Risk
- Low. py2rs is already commented out of riir-ai workspace. katgpt-rs domain is config-only (runtime routing).
- All code recoverable at `git tag py_to_rs` on main.
