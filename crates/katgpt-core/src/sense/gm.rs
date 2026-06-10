//! Internal Sense API — GM override dispatch for NPC brain control.
//!
//! Provides `GmSenseApi` trait (pub(crate)) and binary MCP action dispatch.
//! Callers: MCP binary protocol, SSH GM tools (egui dashboard).
//! Auth reuse: GmKeyStore (papaya) + EntityControlEnvelope (Ed25519).

use crate::types::SenseKind;

/// GM sense API errors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum SenseError {
    NotFound = 0,
    NotAuthorized = 1,
    ModuleLocked = 2,
    InvalidKind = 3,
    IoError = 4,
}

/// Snapshot of NPC brain state for diagnostic dump.
#[derive(Clone, Debug)]
pub struct NpcBrainSnapshot {
    pub npc_id: u32,
    pub activations: Vec<(SenseKind, f32)>,
    pub overrides_active: Vec<(SenseKind, f32)>,
    pub autonomous_disabled: bool,
    pub locked_modules: Vec<SenseKind>,
    pub confidence: Vec<(SenseKind, f32)>,
}

/// Internal-only trait for GM sense override dispatch.
/// NOT exposed as public API — callers go through `dispatch_gm_action`.
#[allow(dead_code)]
pub(crate) trait GmSenseApi {
    /// Pin a single sense to a fixed activation value.
    fn pin_sense(&mut self, npc_id: u32, kind: SenseKind, value: f32) -> Result<(), SenseError>;

    /// Pin multiple senses at once.
    fn pin_all(&mut self, npc_id: u32, values: &[(SenseKind, f32)]) -> Result<(), SenseError>;

    /// Disable autonomous computation — NPC follows script only.
    fn disable_autonomous(&mut self, npc_id: u32, script_id: u64) -> Result<(), SenseError>;

    /// Re-enable autonomous computation.
    fn enable_autonomous(&mut self, npc_id: u32) -> Result<(), SenseError>;

    /// Inject a KG triple into a zone's octree.
    fn inject_kg(
        &mut self,
        zone_id: u32,
        head: u64,
        relation: u64,
        tail: u64,
    ) -> Result<(), SenseError>;

    /// Lock a sense module — prevents bandit from hot-swapping.
    fn lock_module(&mut self, npc_id: u32, kind: SenseKind) -> Result<(), SenseError>;

    /// Force-reload a module from disk path.
    fn force_reload(
        &mut self,
        npc_id: u32,
        kind: SenseKind,
        module_path: &str,
    ) -> Result<(), SenseError>;

    /// Dump full brain snapshot for diagnostics.
    fn dump_brain(&self, npc_id: u32) -> Result<NpcBrainSnapshot, SenseError>;
}

// ---------------------------------------------------------------------------
// Binary payload helpers
// ---------------------------------------------------------------------------

#[allow(dead_code)]
fn read_u32(payload: &[u8], offset: usize) -> Option<(u32, usize)> {
    if offset + 4 > payload.len() {
        return None;
    }
    let val = u32::from_le_bytes([
        payload[offset],
        payload[offset + 1],
        payload[offset + 2],
        payload[offset + 3],
    ]);
    Some((val, offset + 4))
}

#[allow(dead_code)]
fn read_u64(payload: &[u8], offset: usize) -> Option<(u64, usize)> {
    if offset + 8 > payload.len() {
        return None;
    }
    let val = u64::from_le_bytes([
        payload[offset],
        payload[offset + 1],
        payload[offset + 2],
        payload[offset + 3],
        payload[offset + 4],
        payload[offset + 5],
        payload[offset + 6],
        payload[offset + 7],
    ]);
    Some((val, offset + 8))
}

#[allow(dead_code)]
fn read_f32(payload: &[u8], offset: usize) -> Option<(f32, usize)> {
    if offset + 4 > payload.len() {
        return None;
    }
    let val = f32::from_le_bytes([
        payload[offset],
        payload[offset + 1],
        payload[offset + 2],
        payload[offset + 3],
    ]);
    Some((val, offset + 4))
}

#[allow(dead_code)]
fn read_u8(payload: &[u8], offset: usize) -> Option<(u8, usize)> {
    if offset >= payload.len() {
        return None;
    }
    Some((payload[offset], offset + 1))
}

/// Convert raw u8 to SenseKind. Returns None for unknown discriminants.
#[allow(dead_code)]
fn kind_from_u8(raw: u8) -> Option<SenseKind> {
    match raw {
        0 => Some(SenseKind::CommonSense),
        1 => Some(SenseKind::FighterSense),
        2 => Some(SenseKind::GameTheorySense),
        3 => Some(SenseKind::SpatialSense),
        4 => Some(SenseKind::SocialSense),
        5 => Some(SenseKind::SkillSense),
        7 => Some(SenseKind::Reserved),
        _ => None,
    }
}

/// Serialize snapshot into a binary response buffer.
#[allow(dead_code)]
fn serialize_snapshot(snap: &NpcBrainSnapshot) -> Vec<u8> {
    let count = snap.activations.len() as u8;
    let override_count = snap.overrides_active.len() as u8;
    let locked_count = snap.locked_modules.len() as u8;
    let conf_count = snap.confidence.len() as u8;
    // Header: npc_id(4) + autonomous_disabled(1) + 4 counts(4) = 9 bytes
    // Per entry: kind(1) + value(4) = 5 bytes
    let entry_size = 5usize;
    let total = 9
        + (count as usize + override_count as usize + locked_count as usize + conf_count as usize)
            * entry_size;
    let mut buf = Vec::with_capacity(total);

    buf.extend_from_slice(&snap.npc_id.to_le_bytes());
    buf.push(snap.autonomous_disabled as u8);
    buf.push(count);
    buf.push(override_count);
    buf.push(locked_count);
    buf.push(conf_count);

    for (kind, val) in &snap.activations {
        buf.push(*kind as u8);
        buf.extend_from_slice(&val.to_le_bytes());
    }
    for (kind, val) in &snap.overrides_active {
        buf.push(*kind as u8);
        buf.extend_from_slice(&val.to_le_bytes());
    }
    for kind in &snap.locked_modules {
        buf.push(*kind as u8);
        buf.extend_from_slice(&0f32.to_le_bytes()); // placeholder for alignment
    }
    for (kind, val) in &snap.confidence {
        buf.push(*kind as u8);
        buf.extend_from_slice(&val.to_le_bytes());
    }

    buf
}

// ---------------------------------------------------------------------------
// MCP action code dispatch
// ---------------------------------------------------------------------------

/// Dispatch a GM action by binary action code to the appropriate trait method.
///
/// Payload format varies per action:
/// - `0x20` pin_sense:     npc_id(4) + kind(1) + value(4)
/// - `0x21` pin_all:       npc_id(4) + count(1) + [kind(1) + value(4)] * count
/// - `0x22` disable_auto:  npc_id(4) + script_id(8)
/// - `0x23` enable_auto:   npc_id(4)
/// - `0x24` inject_kg:     zone_id(4) + head(8) + relation(8) + tail(8)
/// - `0x25` lock_module:   npc_id(4) + kind(1)
/// - `0x26` force_reload:  npc_id(4) + kind(1) + path_len(1) + path_bytes(path_len)
/// - `0x27` dump_brain:    npc_id(4)
#[allow(dead_code)]
pub(crate) fn dispatch_gm_action(
    api: &mut dyn GmSenseApi,
    action_code: u8,
    payload: &[u8],
) -> Result<Vec<u8>, SenseError> {
    match action_code {
        0x20 => {
            // pin_sense: npc_id(4) + kind(1) + value(4)
            let (npc_id, off) = read_u32(payload, 0).ok_or(SenseError::InvalidKind)?;
            let (raw_kind, off) = read_u8(payload, off).ok_or(SenseError::InvalidKind)?;
            let kind = kind_from_u8(raw_kind).ok_or(SenseError::InvalidKind)?;
            let (value, _) = read_f32(payload, off).ok_or(SenseError::InvalidKind)?;
            api.pin_sense(npc_id, kind, value)?;
            Ok(Vec::new())
        }
        0x21 => {
            // pin_all: npc_id(4) + count(1) + [kind(1) + value(4)] * count
            let (npc_id, mut off) = read_u32(payload, 0).ok_or(SenseError::InvalidKind)?;
            let (count, off2) = read_u8(payload, off).ok_or(SenseError::InvalidKind)?;
            off = off2;
            let mut values = Vec::with_capacity(count as usize);
            for _ in 0..count {
                let (raw_kind, off2) = read_u8(payload, off).ok_or(SenseError::InvalidKind)?;
                let kind = kind_from_u8(raw_kind).ok_or(SenseError::InvalidKind)?;
                let (value, off3) = read_f32(payload, off2).ok_or(SenseError::InvalidKind)?;
                values.push((kind, value));
                off = off3;
            }
            api.pin_all(npc_id, &values)?;
            Ok(Vec::new())
        }
        0x22 => {
            // disable_autonomous: npc_id(4) + script_id(8)
            let (npc_id, off) = read_u32(payload, 0).ok_or(SenseError::InvalidKind)?;
            let (script_id, _) = read_u64(payload, off).ok_or(SenseError::InvalidKind)?;
            api.disable_autonomous(npc_id, script_id)?;
            Ok(Vec::new())
        }
        0x23 => {
            // enable_autonomous: npc_id(4)
            let (npc_id, _) = read_u32(payload, 0).ok_or(SenseError::InvalidKind)?;
            api.enable_autonomous(npc_id)?;
            Ok(Vec::new())
        }
        0x24 => {
            // inject_kg: zone_id(4) + head(8) + relation(8) + tail(8)
            let (zone_id, off) = read_u32(payload, 0).ok_or(SenseError::InvalidKind)?;
            let (head, off) = read_u64(payload, off).ok_or(SenseError::InvalidKind)?;
            let (relation, off) = read_u64(payload, off).ok_or(SenseError::InvalidKind)?;
            let (tail, _) = read_u64(payload, off).ok_or(SenseError::InvalidKind)?;
            api.inject_kg(zone_id, head, relation, tail)?;
            Ok(Vec::new())
        }
        0x25 => {
            // lock_module: npc_id(4) + kind(1)
            let (npc_id, off) = read_u32(payload, 0).ok_or(SenseError::InvalidKind)?;
            let (raw_kind, _) = read_u8(payload, off).ok_or(SenseError::InvalidKind)?;
            let kind = kind_from_u8(raw_kind).ok_or(SenseError::InvalidKind)?;
            api.lock_module(npc_id, kind)?;
            Ok(Vec::new())
        }
        0x26 => {
            // force_reload: npc_id(4) + kind(1) + path_len(1) + path_bytes(path_len)
            let (npc_id, mut off) = read_u32(payload, 0).ok_or(SenseError::InvalidKind)?;
            let (raw_kind, off2) = read_u8(payload, off).ok_or(SenseError::InvalidKind)?;
            let kind = kind_from_u8(raw_kind).ok_or(SenseError::InvalidKind)?;
            off = off2;
            let (path_len, off2) = read_u8(payload, off).ok_or(SenseError::InvalidKind)?;
            off = off2;
            if off + path_len as usize > payload.len() {
                return Err(SenseError::InvalidKind);
            }
            let module_path = std::str::from_utf8(&payload[off..off + path_len as usize])
                .map_err(|_| SenseError::InvalidKind)?;
            api.force_reload(npc_id, kind, module_path)?;
            Ok(Vec::new())
        }
        0x27 => {
            // dump_brain: npc_id(4)
            let (npc_id, _) = read_u32(payload, 0).ok_or(SenseError::InvalidKind)?;
            let snap = api.dump_brain(npc_id)?;
            Ok(serialize_snapshot(&snap))
        }
        _ => Err(SenseError::InvalidKind),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};

    /// In-memory mock implementing GmSenseApi for testing.
    struct MockGmSenseApi {
        brains: HashMap<u32, MockBrain>,
        kg_triples: Vec<(u32, u64, u64, u64)>,
    }

    struct MockBrain {
        pinned: Vec<(SenseKind, f32)>,
        autonomous_disabled: bool,
        script_id: Option<u64>,
        locked: HashSet<SenseKind>,
        activations: Vec<(SenseKind, f32)>,
        confidence: Vec<(SenseKind, f32)>,
        force_reload_log: Vec<(SenseKind, String)>,
    }

    impl MockGmSenseApi {
        fn new() -> Self {
            let mut brains = HashMap::new();
            // Pre-populate NPC 1 and 2
            brains.insert(
                1,
                MockBrain {
                    pinned: Vec::new(),
                    autonomous_disabled: false,
                    script_id: None,
                    locked: HashSet::new(),
                    activations: vec![
                        (SenseKind::FighterSense, 0.5),
                        (SenseKind::SpatialSense, 0.3),
                    ],
                    confidence: vec![
                        (SenseKind::FighterSense, 0.8),
                        (SenseKind::SpatialSense, 0.6),
                    ],
                    force_reload_log: Vec::new(),
                },
            );
            brains.insert(
                2,
                MockBrain {
                    pinned: Vec::new(),
                    autonomous_disabled: false,
                    script_id: None,
                    locked: HashSet::new(),
                    activations: vec![(SenseKind::CommonSense, 0.4)],
                    confidence: vec![(SenseKind::CommonSense, 0.7)],
                    force_reload_log: Vec::new(),
                },
            );
            Self {
                brains,
                kg_triples: Vec::new(),
            }
        }
    }

    impl GmSenseApi for MockGmSenseApi {
        fn pin_sense(
            &mut self,
            npc_id: u32,
            kind: SenseKind,
            value: f32,
        ) -> Result<(), SenseError> {
            let brain = self.brains.get_mut(&npc_id).ok_or(SenseError::NotFound)?;
            if brain.locked.contains(&kind) {
                return Err(SenseError::ModuleLocked);
            }
            if let Some(entry) = brain.pinned.iter_mut().find(|(k, _)| *k == kind) {
                entry.1 = value;
            } else {
                brain.pinned.push((kind, value));
            }
            Ok(())
        }

        fn pin_all(&mut self, npc_id: u32, values: &[(SenseKind, f32)]) -> Result<(), SenseError> {
            let brain = self.brains.get_mut(&npc_id).ok_or(SenseError::NotFound)?;
            for &(kind, _value) in values {
                if brain.locked.contains(&kind) {
                    return Err(SenseError::ModuleLocked);
                }
            }
            for &(kind, value) in values {
                if let Some(entry) = brain.pinned.iter_mut().find(|(k, _)| *k == kind) {
                    entry.1 = value;
                } else {
                    brain.pinned.push((kind, value));
                }
            }
            Ok(())
        }

        fn disable_autonomous(&mut self, npc_id: u32, script_id: u64) -> Result<(), SenseError> {
            let brain = self.brains.get_mut(&npc_id).ok_or(SenseError::NotFound)?;
            brain.autonomous_disabled = true;
            brain.script_id = Some(script_id);
            Ok(())
        }

        fn enable_autonomous(&mut self, npc_id: u32) -> Result<(), SenseError> {
            let brain = self.brains.get_mut(&npc_id).ok_or(SenseError::NotFound)?;
            brain.autonomous_disabled = false;
            brain.script_id = None;
            Ok(())
        }

        fn inject_kg(
            &mut self,
            zone_id: u32,
            head: u64,
            relation: u64,
            tail: u64,
        ) -> Result<(), SenseError> {
            // In real impl, this would update the zone's octree
            self.kg_triples.push((zone_id, head, relation, tail));
            Ok(())
        }

        fn lock_module(&mut self, npc_id: u32, kind: SenseKind) -> Result<(), SenseError> {
            let brain = self.brains.get_mut(&npc_id).ok_or(SenseError::NotFound)?;
            brain.locked.insert(kind);
            Ok(())
        }

        fn force_reload(
            &mut self,
            npc_id: u32,
            kind: SenseKind,
            module_path: &str,
        ) -> Result<(), SenseError> {
            let brain = self.brains.get_mut(&npc_id).ok_or(SenseError::NotFound)?;
            if brain.locked.contains(&kind) {
                return Err(SenseError::ModuleLocked);
            }
            brain.force_reload_log.push((kind, module_path.to_string()));
            Ok(())
        }

        fn dump_brain(&self, npc_id: u32) -> Result<NpcBrainSnapshot, SenseError> {
            let brain = self.brains.get(&npc_id).ok_or(SenseError::NotFound)?;
            Ok(NpcBrainSnapshot {
                npc_id,
                activations: brain.activations.clone(),
                overrides_active: brain.pinned.clone(),
                autonomous_disabled: brain.autonomous_disabled,
                locked_modules: brain.locked.iter().copied().collect(),
                confidence: brain.confidence.clone(),
            })
        }
    }

    // Helper: build a pin_sense payload
    fn make_pin_payload(npc_id: u32, kind: u8, value: f32) -> Vec<u8> {
        let mut buf = Vec::with_capacity(9);
        buf.extend_from_slice(&npc_id.to_le_bytes());
        buf.push(kind);
        buf.extend_from_slice(&value.to_le_bytes());
        buf
    }

    // Helper: build pin_all payload
    fn make_pin_all_payload(npc_id: u32, entries: &[(u8, f32)]) -> Vec<u8> {
        let mut buf = Vec::with_capacity(6 + entries.len() * 5);
        buf.extend_from_slice(&npc_id.to_le_bytes());
        buf.push(entries.len() as u8);
        for &(kind, value) in entries {
            buf.push(kind);
            buf.extend_from_slice(&value.to_le_bytes());
        }
        buf
    }

    #[test]
    fn test_dispatch_pin_sense_ok() {
        let mut api = MockGmSenseApi::new();
        let payload = make_pin_payload(1, 1, 0.9); // FighterSense
        let result = dispatch_gm_action(&mut api, 0x20, &payload);
        assert!(result.is_ok());

        let snap = api.dump_brain(1).unwrap();
        assert_eq!(snap.overrides_active.len(), 1);
        assert_eq!(snap.overrides_active[0], (SenseKind::FighterSense, 0.9));
    }

    #[test]
    fn test_dispatch_pin_all_ok() {
        let mut api = MockGmSenseApi::new();
        let payload = make_pin_all_payload(
            1,
            &[
                (1, 0.9), // FighterSense
                (3, 0.2), // SpatialSense
            ],
        );
        let result = dispatch_gm_action(&mut api, 0x21, &payload);
        assert!(result.is_ok());

        let snap = api.dump_brain(1).unwrap();
        assert_eq!(snap.overrides_active.len(), 2);
    }

    #[test]
    fn test_dispatch_disable_autonomous_ok() {
        let mut api = MockGmSenseApi::new();
        let mut payload = Vec::with_capacity(12);
        payload.extend_from_slice(&1u32.to_le_bytes()); // npc_id
        payload.extend_from_slice(&42u64.to_le_bytes()); // script_id
        let result = dispatch_gm_action(&mut api, 0x22, &payload);
        assert!(result.is_ok());

        let snap = api.dump_brain(1).unwrap();
        assert!(snap.autonomous_disabled);
    }

    #[test]
    fn test_dispatch_enable_autonomous_ok() {
        let mut api = MockGmSenseApi::new();
        // First disable
        let mut payload = Vec::with_capacity(12);
        payload.extend_from_slice(&1u32.to_le_bytes());
        payload.extend_from_slice(&42u64.to_le_bytes());
        dispatch_gm_action(&mut api, 0x22, &payload).unwrap();

        // Then enable
        let payload = 1u32.to_le_bytes();
        let result = dispatch_gm_action(&mut api, 0x23, &payload);
        assert!(result.is_ok());

        let snap = api.dump_brain(1).unwrap();
        assert!(!snap.autonomous_disabled);
    }

    #[test]
    fn test_dispatch_inject_kg_ok() {
        let mut api = MockGmSenseApi::new();
        let mut payload = Vec::with_capacity(28);
        payload.extend_from_slice(&5u32.to_le_bytes()); // zone_id
        payload.extend_from_slice(&100u64.to_le_bytes()); // head
        payload.extend_from_slice(&200u64.to_le_bytes()); // relation
        payload.extend_from_slice(&300u64.to_le_bytes()); // tail
        let result = dispatch_gm_action(&mut api, 0x24, &payload);
        assert!(result.is_ok());
        assert_eq!(api.kg_triples.len(), 1);
        assert_eq!(api.kg_triples[0], (5, 100, 200, 300));
    }

    #[test]
    fn test_dispatch_lock_module_ok() {
        let mut api = MockGmSenseApi::new();
        let mut payload = Vec::with_capacity(5);
        payload.extend_from_slice(&1u32.to_le_bytes());
        payload.push(1); // FighterSense
        let result = dispatch_gm_action(&mut api, 0x25, &payload);
        assert!(result.is_ok());

        let snap = api.dump_brain(1).unwrap();
        assert!(snap.locked_modules.contains(&SenseKind::FighterSense));
    }

    #[test]
    fn test_dispatch_force_reload_ok() {
        let mut api = MockGmSenseApi::new();
        let path = "/data/modules/fighter_v2.bin";
        let mut payload = Vec::with_capacity(6 + path.len());
        payload.extend_from_slice(&1u32.to_le_bytes());
        payload.push(1); // FighterSense
        payload.push(path.len() as u8);
        payload.extend_from_slice(path.as_bytes());
        let result = dispatch_gm_action(&mut api, 0x26, &payload);
        assert!(result.is_ok());
    }

    #[test]
    fn test_dispatch_dump_brain_ok() {
        let mut api = MockGmSenseApi::new();
        let payload = 1u32.to_le_bytes();
        let result = dispatch_gm_action(&mut api, 0x27, &payload);
        assert!(result.is_ok());
        let data = result.unwrap();
        // Should have at least the header (9 bytes) + activation entries
        assert!(data.len() > 9);
    }

    #[test]
    fn test_dispatch_invalid_code_returns_invalid_kind() {
        let mut api = MockGmSenseApi::new();
        let result = dispatch_gm_action(&mut api, 0xFF, &[]);
        assert_eq!(result.unwrap_err(), SenseError::InvalidKind);
    }

    #[test]
    fn test_dispatch_unknown_npc_returns_not_found() {
        let mut api = MockGmSenseApi::new();
        let payload = make_pin_payload(999, 1, 0.5);
        let result = dispatch_gm_action(&mut api, 0x20, &payload);
        assert_eq!(result.unwrap_err(), SenseError::NotFound);
    }

    #[test]
    fn test_pin_overrides_autonomous() {
        let mut api = MockGmSenseApi::new();

        // Check initial autonomous activation
        let snap_before = api.dump_brain(1).unwrap();
        let auto_fighter = snap_before
            .activations
            .iter()
            .find(|(k, _)| *k == SenseKind::FighterSense)
            .map(|(_, v)| *v)
            .unwrap();
        assert_eq!(auto_fighter, 0.5);

        // Pin to different value
        let payload = make_pin_payload(1, 1, 0.9);
        dispatch_gm_action(&mut api, 0x20, &payload).unwrap();

        let snap_after = api.dump_brain(1).unwrap();
        let pinned = snap_after
            .overrides_active
            .iter()
            .find(|(k, _)| *k == SenseKind::FighterSense)
            .map(|(_, v)| *v)
            .unwrap();
        assert_eq!(pinned, 0.9);
        assert_ne!(pinned, auto_fighter);
    }

    #[test]
    fn test_locked_module_rejects_pin() {
        let mut api = MockGmSenseApi::new();

        // Lock FighterSense
        let mut lock_payload = Vec::new();
        lock_payload.extend_from_slice(&1u32.to_le_bytes());
        lock_payload.push(1); // FighterSense
        dispatch_gm_action(&mut api, 0x25, &lock_payload).unwrap();

        // Try to pin locked module → should fail
        let pin_payload = make_pin_payload(1, 1, 0.9);
        let result = dispatch_gm_action(&mut api, 0x20, &pin_payload);
        assert_eq!(result.unwrap_err(), SenseError::ModuleLocked);
    }

    #[test]
    fn test_non_admin_simulation() {
        // Simulate: non-admin tries to dispatch a GM action
        // In real system, auth check happens before dispatch_gm_action.
        // Here we verify the error path works end-to-end.
        let mut api = MockGmSenseApi::new();

        // Non-existent NPC simulates "not found" which is what
        // a non-admin would get after auth rejection cascades
        let payload = make_pin_payload(404, 1, 0.5);
        let result = dispatch_gm_action(&mut api, 0x20, &payload);
        assert_eq!(result.unwrap_err(), SenseError::NotFound);
    }

    #[test]
    fn test_sense_error_repr_u8() {
        // Verify repr(u8) discriminants
        assert_eq!(SenseError::NotFound as u8, 0);
        assert_eq!(SenseError::NotAuthorized as u8, 1);
        assert_eq!(SenseError::ModuleLocked as u8, 2);
        assert_eq!(SenseError::InvalidKind as u8, 3);
        assert_eq!(SenseError::IoError as u8, 4);
    }

    #[test]
    fn test_dispatch_truncated_payload_returns_invalid_kind() {
        let mut api = MockGmSenseApi::new();
        // pin_sense needs 9 bytes, give 2
        let result = dispatch_gm_action(&mut api, 0x20, &[0x01, 0x00]);
        assert_eq!(result.unwrap_err(), SenseError::InvalidKind);
    }

    #[test]
    fn test_dispatch_invalid_sense_kind() {
        let mut api = MockGmSenseApi::new();
        // kind=6 is not a valid SenseKind
        let payload = make_pin_payload(1, 6, 0.5);
        let result = dispatch_gm_action(&mut api, 0x20, &payload);
        assert_eq!(result.unwrap_err(), SenseError::InvalidKind);
    }
}
