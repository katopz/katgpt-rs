//! SenseHotSwap — atomic runtime module replacement.
//!
//! Uses a fixed-size array indexed by `SenseKind as usize` for O(1) lookups.
//! `AtomicBool` replaces `Mutex<bool>` for lock-free flag checks.

use crate::types::{SenseKind, SenseModule};
use std::sync::atomic::{AtomicBool, AtomicPtr, Ordering};

/// Number of sense kinds (max discriminant + 1).
/// SenseKind values: 0..5, 7. We need index 7, so 8 slots total.
const SENSE_KIND_COUNT: usize = 8;

/// Atomic sense module hot-swap with module lock support.
pub struct SenseHotSwap {
    /// Fixed-size array indexed by SenseKind discriminant.
    /// `AtomicPtr` holds the boxed SenseModule; `AtomicBool` is the lock flag.
    slots: [Option<(AtomicPtr<SenseModule>, AtomicBool)>; SENSE_KIND_COUNT],
}

impl SenseHotSwap {
    pub fn new(kinds: &[SenseKind]) -> Self {
        let mut slots: [Option<(AtomicPtr<SenseModule>, AtomicBool)>; SENSE_KIND_COUNT] =
            [None, None, None, None, None, None, None, None];
        for &kind in kinds {
            let idx = kind as usize;
            if idx < SENSE_KIND_COUNT {
                let module = Box::new(SenseModule::default());
                slots[idx] = Some((
                    AtomicPtr::new(Box::into_raw(module)),
                    AtomicBool::new(false),
                ));
            }
        }
        Self { slots }
    }

    /// Atomically swap a module. Returns Err if module is locked.
    #[allow(clippy::result_large_err)]
    pub fn swap(&self, kind: SenseKind, new_module: SenseModule) -> Result<(), SenseModule> {
        let idx = kind as usize;
        let Some((ptr, locked)) = self.slots.get(idx).and_then(|s| s.as_ref()) else {
            return Err(new_module);
        };
        // Check lock — lock-free read
        if locked.load(Ordering::Acquire) {
            return Err(new_module);
        }
        let new = Box::into_raw(Box::new(new_module));
        let old = ptr.swap(new, Ordering::AcqRel);
        // Safety: old was allocated by us
        unsafe {
            drop(Box::from_raw(old));
        }
        Ok(())
    }

    /// Get current module for a kind.
    pub fn get(&self, kind: SenseKind) -> Option<SenseModule> {
        let idx = kind as usize;
        let (ptr, _) = self.slots.get(idx)?.as_ref()?;
        let raw = ptr.load(Ordering::Acquire);
        // Safety: pointer was allocated by us
        let module = unsafe { &*raw };
        Some(module.clone())
    }

    /// Lock a module — prevents bandit from swapping.
    pub fn lock(&self, kind: SenseKind) {
        let idx = kind as usize;
        if let Some(Some((_, locked))) = self.slots.get(idx) {
            locked.store(true, Ordering::Release);
        }
    }

    /// Unlock a module.
    pub fn unlock(&self, kind: SenseKind) {
        let idx = kind as usize;
        if let Some(Some((_, locked))) = self.slots.get(idx) {
            locked.store(false, Ordering::Release);
        }
    }
}

impl Drop for SenseHotSwap {
    fn drop(&mut self) {
        for slot in &mut self.slots {
            if let Some((ptr, _)) = slot.take() {
                let raw = ptr.load(Ordering::Acquire);
                unsafe {
                    drop(Box::from_raw(raw));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SenseKind;

    #[test]
    fn test_swap_returns_consistent() {
        let hotswap = SenseHotSwap::new(&[SenseKind::FighterSense]);
        let mut module = SenseModule::default();
        #[allow(clippy::field_reassign_with_default)]
        module.kind = SenseKind::FighterSense;
        module.commit();

        hotswap
            .swap(SenseKind::FighterSense, module.clone())
            .unwrap();
        let got = hotswap.get(SenseKind::FighterSense).unwrap();
        assert_eq!(got.kind, SenseKind::FighterSense);
    }

    #[test]
    fn test_locked_module_not_swapped() {
        let hotswap = SenseHotSwap::new(&[SenseKind::FighterSense]);
        hotswap.lock(SenseKind::FighterSense);

        let module = SenseModule::default();
        assert!(hotswap.swap(SenseKind::FighterSense, module).is_err());
    }

    #[test]
    fn test_unlock_allows_swap() {
        let hotswap = SenseHotSwap::new(&[SenseKind::FighterSense]);
        hotswap.lock(SenseKind::FighterSense);
        assert!(
            hotswap
                .swap(SenseKind::FighterSense, SenseModule::default())
                .is_err()
        );

        hotswap.unlock(SenseKind::FighterSense);
        assert!(
            hotswap
                .swap(SenseKind::FighterSense, SenseModule::default())
                .is_ok()
        );
    }

    #[test]
    fn test_unregistered_kind_returns_none() {
        let hotswap = SenseHotSwap::new(&[SenseKind::FighterSense]);
        assert!(hotswap.get(SenseKind::SpatialSense).is_none());
    }
}
