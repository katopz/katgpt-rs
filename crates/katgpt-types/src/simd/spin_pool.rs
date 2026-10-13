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
    /// Workers parked by the idle window inside an open scope.
    idle_parked: AtomicUsize,
    /// Idle spin window before such a park, ns (`u64::MAX` = spin forever).
    idle_spin_ns: AtomicU64,
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
        let mut idle_since: Option<std::time::Instant> = None;
        let mut spins = 0u32;
        loop {
            if self.stop.load(Ordering::Acquire) {
                return;
            }
            if self.active.load(Ordering::Acquire) == 0 {
                self.park_while_inactive();
                idle_since = None;
                continue;
            }
            match self.try_claim() {
                Some(i) => {
                    idle_since = None;
                    self.execute(i);
                }
                None => {
                    std::hint::spin_loop();
                    spins = spins.wrapping_add(1);
                    // Check the clock every 64 idle spins, not every spin.
                    if !spins.is_multiple_of(64) {
                        continue;
                    }
                    let since = *idle_since.get_or_insert_with(std::time::Instant::now);
                    let limit = self.idle_spin_ns.load(Ordering::Relaxed);
                    if limit != u64::MAX && since.elapsed().as_nanos() as u64 >= limit {
                        self.park_until_job();
                        idle_since = None;
                    }
                }
            }
        }
    }

    /// Idle past the spin window inside an open scope: park until a job with
    /// unclaimed tasks is published, the scope closes, or the pool stops.
    /// Hands the core back on a contended box, where an idle spinner keeps
    /// the thread HOLDING a task off-CPU.
    fn park_until_job(&self) {
        let mut g = self.park_lock.lock().unwrap_or_else(|p| p.into_inner());
        self.idle_parked.fetch_add(1, Ordering::SeqCst);
        loop {
            // SeqCst pairs with `run`'s claim store + `idle_parked` load: either
            // we see the new job here or `run` sees us parked and notifies.
            let (n, idx) = unpack(self.claim.0.load(Ordering::SeqCst));
            if idx < n
                || self.active.load(Ordering::SeqCst) == 0
                || self.stop.load(Ordering::SeqCst)
            {
                break;
            }
            g = self.park_cv.wait(g).unwrap_or_else(|p| p.into_inner());
        }
        self.idle_parked.fetch_sub(1, Ordering::SeqCst);
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
        // The last scope closing must release idle-parked workers to the
        // inactive park (they wait on "job, scope closed, or stop").
        if self.active.fetch_sub(1, Ordering::SeqCst) == 1
            && self.idle_parked.load(Ordering::SeqCst) > 0
        {
            let _g = self.park_lock.lock().unwrap_or_else(|p| p.into_inner());
            self.park_cv.notify_all();
        }
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
        shared
            .idle_spin_ns
            .store(default_idle_spin_ns(), Ordering::Relaxed);
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

    /// Set the idle spin window: a worker idle that long inside an open scope
    /// parks until the next job (`None` = spin for the whole scope).
    pub fn set_idle_spin(&self, window: Option<std::time::Duration>) {
        let ns = window.map_or(u64::MAX, |d| d.as_nanos().min(u64::MAX as u128 - 1) as u64);
        self.shared.idle_spin_ns.store(ns, Ordering::Relaxed);
    }

    /// One worker per PERFORMANCE core minus one (the caller is the last):
    /// spinning threads on every core leave no room for anything else, and a
    /// descheduled spinner holding a task stalls the fork. Measured on the M3
    /// Max (12 P + 4 E), FFN gate+up after a serial gap: 11 workers ran at
    /// 0.60–0.84× rayon-on-a-worker, 15 workers swung 0.84–1.29× with one
    /// other busy process on the box, 7 workers were bandwidth-short
    /// (0.81–1.22×). `KATGPT_SPIN_POOL_WORKERS` overrides. Off macOS the
    /// default is 0 workers below 8 logical CPUs, i.e. where the pool is
    /// unmeasured (see `default_workers`). A pool with 0 workers is never used
    /// by [`spin_pool_run`]: the kernels keep their rayon arm.
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
        s.claim.0.store(pack(epoch, n_tasks, 0), Ordering::SeqCst);
        if s.idle_parked.load(Ordering::SeqCst) > 0 {
            let _g = s.park_lock.lock().unwrap_or_else(|p| p.into_inner());
            s.park_cv.notify_all();
        }
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

/// Default idle spin window: `KATGPT_SPIN_POOL_IDLE_US` (µs; `inf` = spin
/// for the whole scope), else [`DEFAULT_IDLE_SPIN_US`].
fn default_idle_spin_ns() -> u64 {
    match std::env::var("KATGPT_SPIN_POOL_IDLE_US").ok().as_deref() {
        Some("inf") => u64::MAX,
        Some(v) => v.parse::<u64>().map_or(DEFAULT_IDLE_SPIN_US * 1000, |us| {
            us.saturating_mul(1000).min(u64::MAX - 1)
        }),
        None => DEFAULT_IDLE_SPIN_US * 1000,
    }
}

/// Idle spin window before a worker parks inside an open scope, µs.
/// Measured against ∞ and 30 µs only (riir-infer `decode_spin_pool_goat`,
/// Bonsai-2-27B, M3 Max): quiet box ∞ 0.740 / 100 µs 0.735 / 30 µs 0.757;
/// with a 6-thread CPU burner ∞ 1.339 (spin-forever collapses) / 100 µs
/// 0.794 / 30 µs 0.834. A tuning point among three, not an optimum.
pub const DEFAULT_IDLE_SPIN_US: u64 = 100;

/// Worker count for [`SpinPool::for_this_machine`].
fn default_workers() -> usize {
    if let Some(n) = std::env::var("KATGPT_SPIN_POOL_WORKERS")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        return n;
    }
    // Off macOS: DEFAULT-ON with the idle window at every measured count —
    // the 4090 box's i7-13700K (8P+8E, 24 threads, Windows), real Bonsai-2-27B
    // paired gate (Issue 936 T4, 2026-10-13, this commit's re-measure WITH the
    // idle window; supersedes the pre-idle-window 0.97–3.93× losses the old 0
    // default was set from — the mechanism is now established: workers park
    // through the long gaps instead of burning SMT/E-core spin): 0.862 at 23
    // workers, 0.851 at 15, 0.866 at 11, 0.902 at 7 — a win at every grid
    // point, G1 bit-identical ×4, flat plateau 23/15/11. Boundary = the
    // measurement floor: hosts under 8 logical CPUs are unmeasured and keep
    // rayon (0 workers). `KATGPT_SPIN_POOL_WORKERS` overrides everywhere.
    #[cfg(not(target_os = "macos"))]
    {
        std::thread::available_parallelism()
            .map_or(0, |n| if n.get() >= 8 { n.get() - 1 } else { 0 })
    }
    // macOS: P-cores − 1, the measured in-model optimum (Issue 936 T3b:
    // 11 < 10 < 9 < 7 ms/token across 4 interleaved rounds, katgpt-rs-21).
    #[cfg(target_os = "macos")]
    {
        performance_cores().map_or(0, |p| p.saturating_sub(1))
    }
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
    // SAFETY: `with_spin_pool` holds `&SpinPool` for the whole body and
    // restores the slot on exit, so a non-null slot points at a live pool.
    // A 0-worker pool would run every fork serially: fall back to rayon.
    if pool.is_null() || unsafe { &*pool }.workers() == 0 {
        CURRENT.with(|c| c.set(pool));
        return false;
    }
    struct Reinstall(*const SpinPool);
    impl Drop for Reinstall {
        fn drop(&mut self) {
            CURRENT.with(|c| c.set(self.0));
        }
    }
    let _reinstall = Reinstall(pool);
    // SAFETY: as above.
    unsafe { &*pool }.run(n_tasks, f);
    true
}

/// Row-chunked body over ONE output buffer on this thread's installed pool:
/// `body(y_chunk, row0)` for consecutive `rows_per_task`-row chunks of `y`.
/// `false` (nothing run) when no pool is installed. Same partition semantics
/// as `y.par_chunks_mut(rows_per_task).enumerate()`.
pub fn spin_row_chunks(
    y: &mut [f32],
    rows_per_task: usize,
    body: &(dyn Fn(&mut [f32], usize) + Sync),
) -> bool {
    let chunk = rows_per_task.max(1);
    let (ptr, len) = (SyncMutPtr(y.as_mut_ptr()), y.len());
    spin_pool_run(len.div_ceil(chunk), &|t| {
        let row0 = t * chunk;
        let rows = chunk.min(len - row0);
        // SAFETY: task t alone owns rows [row0, row0 + rows) of `y`, which
        // outlives the run (spin_pool_run returns after every task).
        let y_chunk = unsafe { std::slice::from_raw_parts_mut(ptr.get().add(row0), rows) };
        body(y_chunk, row0);
    })
}

/// A `*mut T` that crosses threads, for handing DISJOINT sub-slices of one
/// buffer to [`spin_pool_run`] tasks. Construction is `unsafe`: the caller
/// promises no two tasks touch the same element and the buffer outlives the
/// run.
#[derive(Clone, Copy)]
pub struct SyncMutPtr<T>(pub(crate) *mut T);
// SAFETY: used only to hand DISJOINT sub-slices of one buffer to tasks.
unsafe impl<T: Send> Send for SyncMutPtr<T> {}
unsafe impl<T: Send> Sync for SyncMutPtr<T> {}

impl<T> SyncMutPtr<T> {
    /// Wrap `p` for disjoint per-task access.
    ///
    /// # Safety
    /// Every task derived from the wrapper must touch a disjoint element
    /// range, and the pointee must outlive the [`spin_pool_run`] it feeds.
    #[inline]
    pub unsafe fn new(p: *mut T) -> Self {
        Self(p)
    }

    /// The pointer. A method (not `.0`) so a closure captures the whole
    /// `Sync` wrapper rather than the bare raw-pointer field.
    #[inline]
    pub fn get(self) -> *mut T {
        self.0
    }
}

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
        // Installed, a 0-worker pool is never used: the kernels keep rayon.
        with_spin_pool(&serial, || assert!(!spin_pool_run(4, &|_| {})));
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
    #[cfg(not(target_os = "macos"))]
    if std::env::var("KATGPT_SPIN_POOL_WORKERS").is_err() {
        // Issue 936 T4 promotion: default-on at n ≥ 8 (the measured floor),
        // 0 below it — and always strictly less than the core count.
        if n >= 8 {
            assert_eq!(w, n - 1, "off macOS the default is n − 1 workers");
        } else {
            assert_eq!(w, 0, "under 8 CPUs the pool stays opt-in (unmeasured)");
        }
    }
    }

    #[test]
    fn idle_parked_workers_wake_for_the_next_job_in_scope() {
        let pool = SpinPool::new(3);
        pool.set_idle_spin(Some(std::time::Duration::from_micros(10)));
        let caller = std::thread::current().id();
        with_spin_pool(&pool, || {
            for gap_ms in [0u64, 1, 5] {
                std::thread::sleep(std::time::Duration::from_millis(gap_ms));
                let sum = AtomicUsize::new(0);
                let worker_ran = AtomicBool::new(false);
                let timed_out = AtomicBool::new(false);
                assert!(spin_pool_run(200, &|i| {
                    if std::thread::current().id() != caller {
                        worker_ran.store(true, Ordering::Release);
                    }
                    // The caller drains every task itself, so a correct sum
                    // alone cannot catch a lost wake-up: task 0 waits for a
                    // parked WORKER to wake and run something.
                    if i == 0 {
                        let t = std::time::Instant::now();
                        while !worker_ran.load(Ordering::Acquire) {
                            if t.elapsed() > std::time::Duration::from_secs(1) {
                                timed_out.store(true, Ordering::Relaxed);
                                break;
                            }
                            std::hint::spin_loop();
                        }
                    }
                    sum.fetch_add(i, Ordering::Relaxed);
                }));
                assert!(
                    !timed_out.load(Ordering::Relaxed),
                    "gap {gap_ms} ms: no parked worker woke"
                );
                assert_eq!(
                    sum.load(Ordering::Relaxed),
                    199 * 200 / 2,
                    "gap {gap_ms} ms"
                );
            }
        });
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
