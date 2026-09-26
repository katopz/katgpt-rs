# Issue 900: Issue-717 Damping G4 Fails Under the Wider Feature Combo (pre-existing)

**Status:** Open — pre-existing, found 2026-09-26 during Issue-898 twin validation (NOT a regression from any 898 work; reproduced without the 898 features).

**Repro:**

```bash
# FAILS (16 allocs / 1024 B over 8 deep runs):
cargo test --features lt2_deep_stability,gain_cost_halt,cadence_gate,loop_stability_fix \
  --test issue_717_t3_t4_damping_goat g4
# PASSES (its own declared features — lt2_looped,lt2_deep_stability):
cargo test --features lt2_deep_stability \
  --test issue_717_t3_t4_damping_goat g4
```

Failure: `assertion left == right failed: stabilization hot loop allocated 16 times (1024 B) over 8 deep runs` (`tests/issue_717_t3_t4_damping_goat.rs:478`).

**Diagnosis (first pass, unverified):** some cfg'd allocation path guarded by one of the three added features (`gain_cost_halt` / `cadence_gate` / `loop_stability_fix`) enters the measured loop's region when the combo is wider than the gate's own `required-features` — candidate suspects: the halter's `prev_step_buf`/`curr_step_buf` (`Vec::with_capacity(n)` per call), the cadence probe path, or the conditional-gate `prev_prev_h` copy's buffers. 16 allocs over 8 runs = 2/call — consistent with two `with_capacity` calls per `forward_looped` invocation.

**Why it matters:** the Gate-Coverage law — "a gate nobody runs is documentation" extends to a gate nobody runs AT A COMBO: the G4 zero-alloc claim for the damping hot loop holds only at the narrow feature set, and no pinned suite runs the wider one. The allocation contract should be combo-stable or the gate should pin the combos it claims.

**Fix direction:** either (a) make the measured region combo-stable (pre-size the offending buffers outside the measured span — the kimi_k3_g4 pattern), or (b) add a pinned suite row running the wider combo so the contract is actually asserted where it fails, or (c) scope the G4 claim in the test doc to the declared features (weakest — last resort).

**Found-by provenance:** Issue-898 twin session (the duplicate landed as `kl_depth_probe` `aa3a7d382`; the local twin `057147b2b` died in the reflog per the Batch-169 precedent) while validating that the 898 features did not disturb the looped gates. First run: FAILED at the wide combo; reproduced WITHOUT the 898 features → pre-existing.
