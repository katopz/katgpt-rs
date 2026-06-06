//! Full lifecycle integration test for the MUSE skill evolution pipeline.
//!
//! Exercises: append memory → test gate → catalog register → evolve status → verify.
//! Also tests freeze/thaw persistence roundtrip.

#![cfg(feature = "skill_lifecycle")]

use katgpt_rs::pruners::{
    BomberTestGate, MemoryEntry, PrunerMemory, PrunerTestGate, SkillCatalog, SkillDescriptor,
    TestStatus,
};

// ── Phase helpers ────────────────────────────────────────────────────

fn append_episodes(memory: &PrunerMemory, count: u64) {
    for i in 0..count {
        let arm = (i % 5) as u16;
        // Arm 2 is the best arm (highest reward).
        let reward = if arm == 2 { 0.9 } else { 0.3 };
        let is_edge = reward > 0.8;
        let is_failure = reward < 0.1;
        memory.append(MemoryEntry::new(arm, reward, is_edge, is_failure, i));
    }
}

fn register_arms(catalog: &mut SkillCatalog, arm_count: usize) {
    for idx in 0..arm_count {
        let desc = SkillDescriptor::new(&format!("arm_{idx}"), format!("Test arm {idx}"), idx);
        catalog.register(desc);
    }
}

// ── Test 1: Full lifecycle ──────────────────────────────────────────

#[test]
fn test_full_lifecycle() {
    // ── Phase 1: Setup ───────────────────────────────────────────────
    let memory = PrunerMemory::new(64, "integration_test");
    let mut catalog = SkillCatalog::new();
    let gate = BomberTestGate::new();

    // ── Phase 2: Learn — append 100 episodes ─────────────────────────
    append_episodes(&memory, 100);

    // ── Phase 3: Validate — run test gate on pre-built cases ─────────
    let test_cases = BomberTestGate::bomber_test_cases();
    let result = gate.validate(&test_cases);
    assert!(
        result.passed,
        "test gate should pass: {:?}",
        result.failures
    );
    assert!(
        result.coverage >= 0.8,
        "coverage should be >= 0.8, got {}",
        result.coverage
    );

    // ── Phase 4: Register — create descriptors for all arms ──────────
    register_arms(&mut catalog, 5);

    // ── Phase 5: Evolve — transition statuses ────────────────────────
    // Arm 2 (best arm) passes validation → promoted to Active.
    assert!(catalog.update_status(2, TestStatus::Validated));
    assert!(catalog.update_status(2, TestStatus::Active));
    // Arm 0 passes validation but not yet promoted.
    assert!(catalog.update_status(0, TestStatus::Validated));

    // ── Phase 6: Verify ──────────────────────────────────────────────

    // Memory: 100 entries written, ring buffer capped at 64.
    assert_eq!(memory.total_entries(), 100);

    // Recent entries are retrievable and in chronological order.
    let recent = memory.recent(10);
    assert_eq!(recent.len(), 10);
    // Last entry (index 99): arm = 99 % 5 = 4, reward = 0.3.
    assert_eq!(recent[9].arm, 4);
    assert!((recent[9].reward - 0.3).abs() < f32::EPSILON);

    // Catalog: correct arm counts and status.
    assert_eq!(catalog.active_count(), 1);
    assert_eq!(catalog.get(2).unwrap().test_status, TestStatus::Active);
    assert_eq!(catalog.get(0).unwrap().test_status, TestStatus::Validated);
    assert_eq!(catalog.get(1).unwrap().test_status, TestStatus::Untested);

    // All 5 arms registered.
    let mut registered_count = 0;
    for arm in 0..5 {
        if catalog.get(arm).is_some() {
            registered_count += 1;
        }
    }
    assert_eq!(registered_count, 5);

    // Identity check holds.
    assert!(memory.verify_identity("integration_test"));
}

// ── Test 2: Freeze/thaw persistence ─────────────────────────────────

#[test]
fn test_lifecycle_with_persistence() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let path = dir.path().join("lifecycle_memory.bin");

    // ── Write phase ──────────────────────────────────────────────────
    {
        let memory = PrunerMemory::new(64, "persist_test");
        append_episodes(&memory, 50);
        assert_eq!(memory.total_entries(), 50);

        memory.save_to(&path).expect("save should succeed");
    }

    // ── Read phase — thaw and verify ─────────────────────────────────
    let restored = PrunerMemory::load_from(&path, "persist_test").expect("load should succeed");

    assert_eq!(restored.total_entries(), 50);
    assert!(restored.verify_identity("persist_test"));

    let recent = restored.recent(10);
    assert_eq!(recent.len(), 10);
    // Entry 49: arm = 49 % 5 = 4, reward = 0.3.
    assert_eq!(recent[9].arm, 4);

    // Can continue appending after thaw.
    restored.append(MemoryEntry::new(0, 1.0, true, false, 50));
    assert_eq!(restored.total_entries(), 51);
}

// ── Test 3: Identity mismatch on thaw ───────────────────────────────

#[test]
fn test_lifecycle_identity_mismatch() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let path = dir.path().join("identity_test.bin");

    let memory = PrunerMemory::new(64, "original_id");
    memory.append(MemoryEntry::new(1, 0.5, false, false, 0));
    memory.save_to(&path).expect("save");

    let result = PrunerMemory::load_from(&path, "wrong_id");
    assert!(result.is_err(), "should fail with identity mismatch");
}

// ── Test 4: Catalog lifecycle transitions ───────────────────────────

#[test]
fn test_catalog_status_transitions() {
    let mut catalog = SkillCatalog::new();

    // Register skills.
    for i in 0..3 {
        catalog.register(SkillDescriptor::new(
            &format!("skill_{i}"),
            format!("Skill {i}"),
            i,
        ));
    }
    assert_eq!(catalog.active_count(), 0);

    // Full lifecycle for arm 0: Untested → Validated → Active.
    assert_eq!(catalog.get(0).unwrap().test_status, TestStatus::Untested);
    assert!(catalog.update_status(0, TestStatus::Validated));
    assert_eq!(catalog.get(0).unwrap().test_status, TestStatus::Validated);
    assert!(catalog.update_status(0, TestStatus::Active));
    assert_eq!(catalog.get(0).unwrap().test_status, TestStatus::Active);

    // Arm 1 fails validation.
    assert!(catalog.update_status(1, TestStatus::Failed));
    assert_eq!(catalog.get(1).unwrap().test_status, TestStatus::Failed);

    // Arm 2 stays untested.
    assert_eq!(catalog.get(2).unwrap().test_status, TestStatus::Untested);

    assert_eq!(catalog.active_count(), 1);
}
