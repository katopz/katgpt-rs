//! SenseHotSwap — atomic runtime module replacement.

use crate::types::{SenseKind, SenseModule};
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::Mutex;

/// Atomic sense module hot-swap with module lock support.
pub struct SenseHotSwap {
    modules: Vec<(SenseKind, AtomicPtr<SenseModule>, Mutex<bool>)>,
}

impl SenseHotSwap {
    pub fn new(kinds: &[SenseKind]) -> Self {
        Self {
            modules: kinds.iter().map(|&kind| {
                let module = Box::new(SenseModule::default());
                (kind, AtomicPtr::new(Box::into_raw(module)), Mutex::new(false))
            }).collect(),
        }
    }

    /// Atomically swap a module. Returns Err if module is locked.
    pub fn swap(&self, kind: SenseKind, new_module: SenseModule) -> Result<(), SenseModule> {
        for (k, ptr, lock) in &self.modules {
            if *k == kind {
                // Check lock
                if let Ok(guard) = lock.lock() {
                    if *guard {
                        return Err(new_module);
                    }
                }
                let new = Box::into_raw(Box::new(new_module));
                let old = ptr.swap(new, Ordering::AcqRel);
                // Safety: old was allocated by us
                unsafe { drop(Box::from_raw(old)); }
                return Ok(());
            }
        }
        Err(new_module)
    }

    /// Get current module for a kind.
    pub fn get(&self, kind: SenseKind) -> Option<SenseModule> {
        for (k, ptr, _) in &self.modules {
            if *k == kind {
                let raw = ptr.load(Ordering::Acquire);
                // Safety: pointer was allocated by us
                let module = unsafe { &*raw };
                return Some(module.clone());
            }
        }
        None
    }

    /// Lock a module — prevents bandit from swapping.
    pub fn lock(&self, kind: SenseKind) {
        for (k, _, lock) in &self.modules {
            if *k == kind {
                if let Ok(mut guard) = lock.lock() {
                    *guard = true;
                }
                return;
            }
        }
    }

    /// Unlock a module.
    pub fn unlock(&self, kind: SenseKind) {
        for (k, _, lock) in &self.modules {
            if *k == kind {
                if let Ok(mut guard) = lock.lock() {
                    *guard = false;
                }
                return;
            }
        }
    }
}

impl Drop for SenseHotSwap {
    fn drop(&mut self) {
        for (_, ptr, _) in &self.modules {
            let raw = ptr.load(Ordering::Acquire);
            unsafe { drop(Box::from_raw(raw)); }
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
        module.kind = SenseKind::FighterSense;
        module.commit();

        hotswap.swap(SenseKind::FighterSense, module.clone()).unwrap();
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
}
