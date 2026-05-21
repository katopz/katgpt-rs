//! Unit tests for freeze/thaw round-trip verification (Plan 092).
//!
//! Verifies that the freeze → thaw cycle produces identical data for all player
//! types with frozen knowledge persistence. Also tests disk I/O via save/load.

// ── Bomber HLPlayer ────────────────────────────────────────────

#[cfg(feature = "bomber")]
mod bomber_hl {
    use microgpt_rs::pruners::bomber::{BomberFrozenBandit, HLPlayer};
    use microgpt_rs::pruners::{load_frozen, save_frozen};
    use tempfile::tempdir;

    #[test]
    fn freeze_thaw_roundtrip() {
        let player = HLPlayer::new(0);
        let frozen = player.freeze();
        let thawed = HLPlayer::thaw(&frozen, 1).unwrap();
        let re_frozen = thawed.freeze();

        // Direct field copy — should be exactly equal
        assert_eq!(frozen, re_frozen);
    }

    #[test]
    fn freeze_thaw_with_known_data() {
        let mut frozen = BomberFrozenBandit::new_empty();
        frozen.q_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
        frozen.visits = [10, 20, 30, 40, 50, 60, 70];
        frozen.total_pulls = 280;
        frozen.compressed = [0, 1, 0, 1, 0, 0, 1];

        let thawed = HLPlayer::thaw(&frozen, 0).unwrap();
        let re_frozen = thawed.freeze();

        assert_eq!(frozen.q_values, re_frozen.q_values);
        assert_eq!(frozen.visits, re_frozen.visits);
        assert_eq!(frozen.total_pulls, re_frozen.total_pulls);
        assert_eq!(frozen.compressed, re_frozen.compressed);
    }

    #[test]
    fn frozen_disk_roundtrip() {
        let player = HLPlayer::new(0);
        let frozen = player.freeze();

        let dir = tempdir().unwrap();
        let path = dir.path().join("bomber_hl.bin");

        save_frozen(&path, &frozen).unwrap();
        let loaded: BomberFrozenBandit = load_frozen(&path).unwrap();

        assert_eq!(frozen, loaded);
    }

    #[test]
    fn thaw_rejects_bad_magic() {
        let mut frozen = BomberFrozenBandit::new_empty();
        frozen.magic = *b"XXXX";
        match HLPlayer::thaw(&frozen, 0) {
            Err(e) => assert!(e.contains("Invalid magic"), "unexpected error: {e}"),
            Ok(_) => panic!("expected error, got successful thaw"),
        }
    }

    #[test]
    fn thaw_rejects_bad_version() {
        let mut frozen = BomberFrozenBandit::new_empty();
        frozen.version = 999;
        match HLPlayer::thaw(&frozen, 0) {
            Err(e) => assert!(e.contains("Unsupported version"), "unexpected error: {e}"),
            Ok(_) => panic!("expected error, got successful thaw"),
        }
    }
}

// ── Bomber GZeroPlayer ─────────────────────────────────────────

#[cfg(feature = "g_zero")]
mod bomber_gzero {
    use microgpt_rs::pruners::bomber::{BomberFrozenBandit, GZeroPlayer};
    use microgpt_rs::pruners::{load_frozen, save_frozen};
    use tempfile::tempdir;

    #[test]
    fn freeze_thaw_roundtrip() {
        let player = GZeroPlayer::new(0);
        let frozen = player.freeze();
        let thawed = GZeroPlayer::thaw(&frozen, 1).unwrap();
        let re_frozen = thawed.freeze();

        // GZero directly copies q_values/visits — should be exact
        assert_eq!(frozen.q_values, re_frozen.q_values);
        assert_eq!(frozen.visits, re_frozen.visits);
        assert_eq!(frozen.total_pulls, re_frozen.total_pulls);
        assert_eq!(frozen.compressed, re_frozen.compressed);
    }

    #[test]
    fn freeze_thaw_with_known_data() {
        let mut frozen = BomberFrozenBandit::new_empty();
        frozen.q_values = [0.5, -0.3, 0.8, 0.1, 0.0, 0.2, -0.1];
        frozen.visits = [5, 10, 15, 3, 0, 7, 2];
        // total_pulls consistent with visits (sum = 42)
        frozen.total_pulls = 42;

        let thawed = GZeroPlayer::thaw(&frozen, 0).unwrap();
        let re_frozen = thawed.freeze();

        assert_eq!(frozen.q_values, re_frozen.q_values);
        assert_eq!(frozen.visits, re_frozen.visits);
        // total_pulls is recomputed from visits by freeze()
        assert_eq!(re_frozen.total_pulls, frozen.visits.iter().sum::<u32>());
    }

    #[test]
    fn frozen_disk_roundtrip() {
        let player = GZeroPlayer::new(0);
        let frozen = player.freeze();

        let dir = tempdir().unwrap();
        let path = dir.path().join("bomber_gzero.bin");

        save_frozen(&path, &frozen).unwrap();
        let loaded: BomberFrozenBandit = load_frozen(&path).unwrap();

        assert_eq!(frozen, loaded);
    }
}

// ── Go GoHLPlayer ──────────────────────────────────────────────

#[cfg(feature = "go")]
mod go_hl {
    use microgpt_rs::pruners::go::{GoFrozenBandit, GoHLPlayer};
    use microgpt_rs::pruners::{load_frozen, save_frozen};
    use tempfile::tempdir;

    #[test]
    fn freeze_thaw_roundtrip() {
        let player = GoHLPlayer::new();
        let frozen = player.freeze();
        let thawed = GoHLPlayer::thaw(&frozen).unwrap();
        let re_frozen = thawed.freeze();

        // All zeros — exact match
        assert_eq!(frozen, re_frozen);
    }

    #[test]
    fn freeze_thaw_with_known_data() {
        // Construct frozen with non-zero data; thaw replays bandit updates
        let mut frozen = GoFrozenBandit::new_empty();
        frozen.q_values = [0.0, 0.5, 0.8, 0.0, 0.3, 0.0, 0.0, 0.0];
        frozen.visits = [0, 5, 10, 0, 3, 0, 0, 0];
        frozen.total_pulls = 18;
        frozen.epsilon = 0.1;

        let thawed = GoHLPlayer::thaw(&frozen).unwrap();

        // Verify public API matches frozen data
        let thawed_q = thawed.q_values();
        let thawed_v = thawed.visits();
        for i in 0..8 {
            assert!(
                (thawed_q[i] - frozen.q_values[i]).abs() < 1e-5,
                "q_values[{i}] mismatch via API"
            );
            assert_eq!(
                thawed_v[i], frozen.visits[i],
                "visits[{i}] mismatch via API"
            );
        }

        let re_frozen = thawed.freeze();

        // Replay-based thaw: incremental mean with identical reward converges exactly
        for i in 0..8 {
            let diff = (frozen.q_values[i] - re_frozen.q_values[i]).abs();
            assert!(diff < 1e-5, "q_values[{i}] mismatch after re-freeze");
        }
        assert_eq!(frozen.visits, re_frozen.visits);
        assert_eq!(frozen.epsilon, re_frozen.epsilon);
    }

    #[test]
    fn frozen_disk_roundtrip() {
        let player = GoHLPlayer::new();
        let frozen = player.freeze();

        let dir = tempdir().unwrap();
        let path = dir.path().join("go_hl.bin");

        save_frozen(&path, &frozen).unwrap();
        let loaded: GoFrozenBandit = load_frozen(&path).unwrap();

        assert_eq!(frozen, loaded);
    }

    #[test]
    fn thaw_rejects_bad_magic() {
        let mut frozen = GoFrozenBandit::new_empty();
        frozen.magic = *b"XXXX";
        match GoHLPlayer::thaw(&frozen) {
            Err(e) => assert!(e.contains("Invalid magic"), "unexpected error: {e}"),
            Ok(_) => panic!("expected error, got successful thaw"),
        }
    }

    #[test]
    fn thaw_rejects_bad_version() {
        let mut frozen = GoFrozenBandit::new_empty();
        frozen.version = 999;
        match GoHLPlayer::thaw(&frozen) {
            Err(e) => assert!(e.contains("Unsupported version"), "unexpected error: {e}"),
            Ok(_) => panic!("expected error, got successful thaw"),
        }
    }
}

// ── Go GoGZeroPlayer ───────────────────────────────────────────

#[cfg(feature = "go")]
mod go_gzero {
    use microgpt_rs::pruners::go::{GoFrozenTemplates, GoGZeroPlayer};
    use microgpt_rs::pruners::{load_frozen, save_frozen};
    use tempfile::tempdir;

    #[test]
    fn freeze_thaw_roundtrip() {
        let player = GoGZeroPlayer::new();
        let frozen = player.freeze();
        let thawed = GoGZeroPlayer::thaw(&frozen).unwrap();
        let re_frozen = thawed.freeze();

        // All zeros — exact match
        assert_eq!(frozen, re_frozen);
    }

    #[test]
    fn freeze_thaw_with_known_data() {
        let mut frozen = GoFrozenTemplates::new_empty();
        frozen.q_values = [0.6, 0.3, 0.0, 0.9];
        frozen.visits = [8, 4, 0, 12];
        // total_pulls consistent with visits (sum = 24)
        frozen.total_pulls = 24;

        let thawed = GoGZeroPlayer::thaw(&frozen).unwrap();

        // Verify public API matches frozen data
        let thawed_q = thawed.q_values();
        let thawed_v = thawed.template_visits();
        for i in 0..4 {
            assert!(
                (thawed_q[i] - frozen.q_values[i]).abs() < 1e-5,
                "q_values[{i}] mismatch via API"
            );
            assert_eq!(
                thawed_v[i], frozen.visits[i],
                "visits[{i}] mismatch via API"
            );
        }

        let re_frozen = thawed.freeze();

        // Replay-based thaw: incremental mean with identical reward converges exactly
        for i in 0..4 {
            let diff = (frozen.q_values[i] - re_frozen.q_values[i]).abs();
            assert!(diff < 1e-5, "q_values[{i}] mismatch after re-freeze");
        }
        assert_eq!(frozen.visits, re_frozen.visits);
        assert_eq!(frozen.total_pulls, re_frozen.total_pulls);
    }

    #[test]
    fn frozen_disk_roundtrip() {
        let player = GoGZeroPlayer::new();
        let frozen = player.freeze();

        let dir = tempdir().unwrap();
        let path = dir.path().join("go_gzero.bin");

        save_frozen(&path, &frozen).unwrap();
        let loaded: GoFrozenTemplates = load_frozen(&path).unwrap();

        assert_eq!(frozen, loaded);
    }

    #[test]
    fn thaw_rejects_bad_magic() {
        let mut frozen = GoFrozenTemplates::new_empty();
        frozen.magic = *b"XXXX";
        match GoGZeroPlayer::thaw(&frozen) {
            Err(e) => assert!(e.contains("Invalid magic"), "unexpected error: {e}"),
            Ok(_) => panic!("expected error, got successful thaw"),
        }
    }
}

// ── Struct size assertions ─────────────────────────────────────

#[test]
fn frozen_struct_sizes_reasonable() {
    #[cfg(feature = "bomber")]
    {
        let size = std::mem::size_of::<microgpt_rs::pruners::bomber::BomberFrozenBandit>();
        assert!(size > 0, "BomberFrozenBandit should have non-zero size");
        assert!(
            size <= 256,
            "BomberFrozenBandit should be <= 256 bytes, got {size}"
        );
    }

    #[cfg(feature = "go")]
    {
        let size = std::mem::size_of::<microgpt_rs::pruners::go::GoFrozenBandit>();
        assert!(size > 0, "GoFrozenBandit should have non-zero size");
        assert!(
            size <= 256,
            "GoFrozenBandit should be <= 256 bytes, got {size}"
        );

        let size = std::mem::size_of::<microgpt_rs::pruners::go::GoFrozenTemplates>();
        assert!(size > 0, "GoFrozenTemplates should have non-zero size");
        assert!(
            size <= 256,
            "GoFrozenTemplates should be <= 256 bytes, got {size}"
        );
    }
}
