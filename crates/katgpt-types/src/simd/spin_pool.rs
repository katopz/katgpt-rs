//! Decode-scoped spin-barrier pool (Issue 936).
//!
//! rayon's workers park in the short serial gaps between a decode forward's
//! forks (norms, conv1d, gates, SwiGLU), and waking them costs ~100 µs per
//! fork on an M3 Max — ~14–16% of a Bonsai-2-27B token (riir-infer Issue 045).
//! This pool's workers spin on an epoch counter instead, so a fork published
//! a few µs after the last one is picked up without an OS wake.
//!
//! Shape (llama.cpp's threadpool, adapted): [`SpinPool::run`] publishes
//! `(f, n_tasks)` behind one claim word packing `epoch | n_tasks | next`, and
//! every thread — the caller included — claims task indices by CAS on that
//! word until they run out. Completion is counted per TASK: the caller waits
//! only for tasks that were claimed, never for an idle worker the OS
//! descheduled, and a late worker holding a stale word fails its CAS rather
//! than claiming a task of a job it never saw. `f` (a borrow on the caller's
//! stack) is dereferenced only by a thread holding an unretired task, so it is
//! never touched after `run` returns. No work-stealing deque: a task index
//! names a fixed row range, so results are bit-identical to any other
//! partition of independent rows.
//!
//! Workers spin only while a scope is open (a [`with_spin_pool`] body, or a
//! running [`SpinPool::run`]) and park on a condvar otherwise, so the pool
//! never competes with rayon or anything else outside the decode it serves.
//!
//! Kernels opt in through a thread-local: [`with_spin_pool`] installs a pool
//! for the calling thread, and [`spin_pool_run`] dispatches to it when one is
//! installed. Nothing changes for a thread that never installs one.

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

/// One cache line (128 B covers Apple and x86 adjacent-line prefetch pairs).
#[repr(align(128))]
#[derive(Default)]
struct Padded<T>(T);

/// The published job: a borrow of the caller's closure, valid for one epoch.
struct JobRef<'a> {
    f: &'a (dyn Fn(usize) + Sync),
}

/// The claim word packs `epoch | n_tasks | next index` so one CAS both checks
/// that a task belongs to the CURRENT job and claims it: a worker holding a
/// stale word (a later job was published meanwhile) fails the CAS and never
/// runs a task of a job it did not see.
const FIELD_BITS: u32 = 20;
const FIELD_MASK: u64 = (1 << FIELD_BITS) - 1;
/// Largest `n_tasks` one [`SpinPool::run`] accepts.
pub const MAX_TASKS: usize = FIELD_MASK as usize;

#[inline]
fn pack(epoch: u64, n: usize, idx: usize) -> u64 {
    (epoch << (2 * FIELD_BITS)) | ((n as u64) << FIELD_BITS) | idx as u64
}

#[inline]
fn unpack(w: u64) -> (usize, usize) {
    (
        ((w >> FIELD_BITS) & FIELD_MASK) as usize,
        (w & FIELD_MASK) as usize,
    )
}

#[derive(Default)]
struct Shared {
    claim: Padded<AtomicU64>,
    done: Padded<AtomicUsize>,
    job: AtomicPtr<()>,
    /// Open spin scopes + in-flight runs; workers park while it is 0.
    active: AtomicUsize,
    stop: AtomicBool,
    poisoned: AtomicBool,
    park_lock: Mutex<()>,
    park_cv: Condvar,
}

impl Shared {
    /// Claim one task of the current job; `None` when none is left.
    #[inline]
    fn try_claim(&self) -> Option<usize> {
        let mut w = self.claim.0.load(Ordering::Acquire);
        loop {
            let (n, idx) = unpack(w);
            if idx >= n {
                return None;
            }
            match self
                .claim
                .0
                .compare_exchange_weak(w, w + 1, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => return Some(idx),
                Err(cur) => w = cur,
            }
        }
    }

    /// Run claimed task `i`, then retire it.
    fn execute(&self, i: usize) {
        // SAFETY: a task is claimed only under the current epoch's word, and
        // `run` publishes the job pointer before that word (Release, Acquired
        // by the CAS) and returns only after this task retires (`done`), so
        // the borrow is live and is THIS job's.
        let job = unsafe { &*(self.job.load(Ordering::Acquire) as *const JobRef<'_>) };
        let f = job.f;
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(i))).is_err() {
            self.poisoned.store(true, Ordering::Relaxed);
        }
        self.done.0.fetch_add(1, Ordering::Release);
    }

    fn worker_loop(&self) {
        loop {
            if self.stop.load(Ordering::Acquire) {
                return;
            }
            if self.active.load(Ordering::Acquire) == 0 {
                self.park_while_inactive();
                continue;
            }
            match self.try_claim() {
                Some(i) => self.execute(i),
                // A periodic `yield_now` here was measured (every 64 / 1024
                // idle spins, load avg 10-11): no difference, so plain spin.
                None => std::hint::spin_loop(),
            }
        }
    }

    fn park_while_inactive(&self) {
        let mut g = self.park_lock.lock().unwrap_or_else(|p| p.into_inner());
        while self.active.load(Ordering::SeqCst) == 0 && !self.stop.load(Ordering::SeqCst) {
            g = self.park_cv.wait(g).unwrap_or_else(|p| p.into_inner());
        }
    }

    /// Open a spin scope; wakes parked workers on the 0 → 1 edge.
    fn activate(&self) {
        if self.active.fetch_add(1, Ordering::SeqCst) == 0 {
            let _g = self.park_lock.lock().unwrap_or_else(|p| p.into_inner());
            self.park_cv.notify_all();
        }
    }

    fn deactivate(&self) {
        self.active.fetch_sub(1, Ordering::SeqCst);
    }
}

/// A fixed set of spinning workers that run [`SpinPool::run`] jobs together
/// with the calling thread. See the module docs.
pub struct SpinPool {
    shared: Arc<Shared>,
    handles: Vec<JoinHandle<()>>,
    /// Last published epoch (only `run` writes it, one caller at a time).
    epoch: AtomicU64,
    /// Set while a `run` is in flight: the claim word holds ONE job.
    running: AtomicBool,
}

impl SpinPool {
    /// A pool of `workers` background threads (the caller is one more).
    /// `workers == 0` runs every job serially on the caller.
    pub fn new(workers: usize) -> Self {
        let shared = Arc::new(Shared::default());
        let handles = (0..workers)
            .map(|i| {
                let s = Arc::clone(&shared);
                std::thread::Builder::new()
                    .name(format!("spin-pool-{i}"))
                    .spawn(move || s.worker_loop())
                    .expect("spawn spin-pool worker")
            })
            .collect();
        Self {
            shared,
            handles,
            epoch: AtomicU64::new(0),
            running: AtomicBool::new(false),
        }
    }

    /// One worker per PERFORMANCE core minus one (the caller is the last):
    /// spinning threads on every core leave no room for anything else, and a
    /// descheduled spinner holding a task stalls the fork. Measured on the M3
    /// Max (12 P + 4 E), FFN gate+up after a serial gap: 11 workers ran at
    /// 0.60–0.84× rayon-on-a-worker, 15 workers swung 0.84–1.29× with one
    /// other busy process on the box, 7 workers were bandwidth-short
    /// (0.81–1.22×). `KATGPT_SPIN_POOL_WORKERS` overrides; off macOS the
    /// count is `available_parallelism() - 1`.
    pub fn for_this_machine() -> Self {
        Self::new(default_workers())
    }

    /// Background worker count (the caller is one more thread).
    pub fn workers(&self) -> usize {
        self.handles.len()
    }

    /// Keep the workers spinning (not parked) until the guard drops — wrap a
    /// whole decode token so the gaps between its forks never park them.
    fn spin_scope(&self) -> SpinScope<'_> {
        self.shared.activate();
        SpinScope(&self.shared)
    }

    /// Run `f(i)` for every `i in 0..n_tasks` across the workers and the
    /// calling thread; returns once every task has finished. Only the tasks
    /// actually claimed are waited on, so a descheduled idle worker cannot
    /// stall the caller.
    ///
    /// # Panics
    /// `n_tasks > MAX_TASKS`; a second thread calling `run` while one is in
    /// flight; re-raises (as a new panic) if any task panicked.
    pub fn run(&self, n_tasks: usize, f: &(dyn Fn(usize) + Sync)) {
        assert!(
            n_tasks <= MAX_TASKS,
            "spin pool: {n_tasks} tasks > MAX_TASKS"
        );
        if n_tasks == 0 {
            return;
        }
        assert!(
            !self.running.swap(true, Ordering::Acquire),
            "spin pool: concurrent run from two threads"
        );
        struct Running<'a>(&'a AtomicBool);
        impl Drop for Running<'_> {
            fn drop(&mut self) {
                self.0.store(false, Ordering::Release);
            }
        }
        let _running = Running(&self.running);
        let s = &*self.shared;
        let _scope = self.spin_scope();
        let epoch = (self.epoch.load(Ordering::Relaxed) + 1) & ((1 << (64 - 2 * FIELD_BITS)) - 1);
        self.epoch.store(epoch, Ordering::Relaxed);
        let job = JobRef { f };
        s.job
            .store(&job as *const JobRef<'_> as *mut (), Ordering::Relaxed);
        s.done.0.store(0, Ordering::Relaxed);
        s.claim.0.store(pack(epoch, n_tasks, 0), Ordering::Release);
        while let Some(i) = s.try_claim() {
            s.execute(i);
        }
        while s.done.0.load(Ordering::Acquire) < n_tasks {
            std::hint::spin_loop();
        }
        if s.poisoned.swap(false, Ordering::Relaxed) {
            panic!("spin-pool task panicked");
        }
    }
}

/// Worker count for [`SpinPool::for_this_machine`].
fn default_workers() -> usize {
    if let Some(n) = std::env::var("KATGPT_SPIN_POOL_WORKERS")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        return n;
    }
    let cores = performance_cores()
        .or_else(|| std::thread::available_parallelism().ok().map(|n| n.get()))
        .unwrap_or(1);
    cores.saturating_sub(1)
}

/// Performance-core count (`hw.perflevel0.physicalcpu`); `None` when the
/// sysctl is absent (Intel Macs, older macOS).
#[cfg(target_os = "macos")]
fn performance_cores() -> Option<usize> {
    use std::ffi::{c_char, c_int, c_void};
    unsafe extern "C" {
        fn sysctlbyname(
            name: *const c_char,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *mut c_void,
            newlen: usize,
        ) -> c_int;
    }
    let name = b"hw.perflevel0.physicalcpu\0";
    let mut value: c_int = 0;
    let mut len = core::mem::size_of::<c_int>();
    // SAFETY: a NUL-terminated name, an out-buffer of exactly `len` bytes.
    let rc = unsafe {
        sysctlbyname(
            name.as_ptr().cast(),
            core::ptr::addr_of_mut!(value).cast(),
            core::ptr::addr_of_mut!(len),
            core::ptr::null_mut(),
            0,
        )
    };
    (rc == 0 && value > 0).then_some(value as usize)
}

#[cfg(not(target_os = "macos"))]
fn performance_cores() -> Option<usize> {
    None
}

/// Guard of an open spin scope (see [`SpinPool::spin_scope`]).
struct SpinScope<'a>(&'a Shared);

impl Drop for SpinScope<'_> {
    fn drop(&mut self) {
        self.0.deactivate();
    }
}

impl Drop for SpinPool {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::SeqCst);
        {
            let _g = self
                .shared
                .park_lock
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            self.shared.park_cv.notify_all();
        }
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

thread_local! {
    /// The pool installed on this thread by [`with_spin_pool`], if any.
    static CURRENT: Cell<*const SpinPool> = const { Cell::new(std::ptr::null()) };
}

/// Run `body` with `pool` installed for this thread: the row-parallel kernels
/// it calls fork onto `pool` instead of rayon. Restores the previous pool
/// (nesting is fine) on return or unwind.
pub fn with_spin_pool<R>(pool: &SpinPool, body: impl FnOnce() -> R) -> R {
    struct Restore(*const SpinPool);
    impl Drop for Restore {
        fn drop(&mut self) {
            CURRENT.with(|c| c.set(self.0));
        }
    }
    let _restore = Restore(CURRENT.with(|c| c.replace(pool as *const SpinPool)));
    let _spin = pool.spin_scope();
    body()
}

/// Run `f(i)` for `i in 0..n_tasks` on this thread's installed pool and
/// return `true`; `false` (nothing run) when no pool is installed. The pool
/// is uninstalled while its tasks run, so a task that reaches a kernel again
/// on the caller thread cannot re-enter the pool.
pub fn spin_pool_run(n_tasks: usize, f: &(dyn Fn(usize) + Sync)) -> bool {
    let pool = CURRENT.with(|c| c.replace(std::ptr::null()));
    if pool.is_null() {
        return false;
    }
    struct Reinstall(*const SpinPool);
    impl Drop for Reinstall {
        fn drop(&mut self) {
            CURRENT.with(|c| c.set(self.0));
        }
    }
    let _reinstall = Reinstall(pool);
    // SAFETY: `with_spin_pool` holds `&SpinPool` for the whole body and
    // restores the slot on exit, so a non-null slot points at a live pool.
    unsafe { &*pool }.run(n_tasks, f);
    true
}

/// A `*mut T` that crosses threads; the caller guarantees disjoint access.
#[derive(Clone, Copy)]
pub(crate) struct SyncMutPtr<T>(pub(crate) *mut T);
// SAFETY: used only to hand DISJOINT sub-slices of one buffer to tasks.
unsafe impl<T: Send> Send for SyncMutPtr<T> {}
unsafe impl<T: Send> Sync for SyncMutPtr<T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_task_runs_exactly_once_across_many_epochs() {
        let pool = SpinPool::new(5);
        for n in [1usize, 2, 7, 64, 1000] {
            for _ in 0..50 {
                let hits: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
                pool.run(n, &|i| {
                    hits[i].fetch_add(1, Ordering::Relaxed);
                });
                assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1), "n={n}");
            }
        }
    }

    #[test]
    fn zero_workers_runs_serially_and_parks_wake_up() {
        let serial = SpinPool::new(0);
        let sum = AtomicUsize::new(0);
        serial.run(10, &|i| {
            sum.fetch_add(i, Ordering::Relaxed);
        });
        assert_eq!(sum.load(Ordering::Relaxed), 45);

        // Let the workers park, then publish again: no lost wake-up.
        let pool = SpinPool::new(3);
        std::thread::sleep(std::time::Duration::from_millis(20));
        let sum = AtomicUsize::new(0);
        pool.run(100, &|i| {
            sum.fetch_add(i, Ordering::Relaxed);
        });
        assert_eq!(sum.load(Ordering::Relaxed), 4950);
    }

    #[test]
    fn machine_default_leaves_the_caller_a_core() {
        let n = std::thread::available_parallelism().map_or(1, |n| n.get());
        let w = default_workers();
        assert!(w < n.max(1), "{w} workers on {n} cores");
        #[cfg(target_os = "macos")]
        if let Some(p) = performance_cores() {
            assert!(p <= n, "perflevel0 {p} > {n} cores");
        }
    }

    #[test]
    fn a_panicking_task_is_reraised_and_the_pool_survives() {
        let pool = SpinPool::new(2);
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool.run(8, &|i| assert_ne!(i, 5, "boom"));
        }));
        assert!(r.is_err());
        let sum = AtomicUsize::new(0);
        pool.run(4, &|i| {
            sum.fetch_add(i, Ordering::Relaxed);
        });
        assert_eq!(sum.load(Ordering::Relaxed), 6);
    }

    #[test]
    fn install_is_scoped_and_not_reentrant() {
        let pool = SpinPool::new(2);
        assert!(!spin_pool_run(1, &|_| {}));
        with_spin_pool(&pool, || {
            let inner = AtomicUsize::new(0);
            assert!(spin_pool_run(4, &|_| {
                // A task on the caller thread sees no pool installed.
                if !spin_pool_run(1, &|_| {}) {
                    inner.fetch_add(1, Ordering::Relaxed);
                }
            }));
            // No thread sees a pool inside a task: the caller's slot is
            // emptied for the run, and workers never had one.
            assert_eq!(inner.load(Ordering::Relaxed), 4);
            assert!(spin_pool_run(1, &|_| {}), "reinstalled after run");
        });
        assert!(!spin_pool_run(1, &|_| {}), "uninstalled after the scope");
    }
}
