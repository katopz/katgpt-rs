//! Issue 1237 T1/T2 G4 — `screen_auroc` and `control_verdict` allocate
//! nothing once `ScreenScratch` is sized. Counting `GlobalAlloc`; release or
//! dev, the count is exact (no sampling).

use katgpt_pruners::mech_attribution::{
    ControlParams, ScreenParams, ScreenScratch, ScreenVerdict, UnitScreen, control_verdict,
    screen_auroc,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Counting;
static ALLOCS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::SeqCst);
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::SeqCst);
        unsafe { System.realloc(p, l, n) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

#[test]
fn g4_screen_and_verdict_zero_alloc() {
    let (n_units, n) = (48usize, 256usize);
    let labels: Vec<bool> = (0..n).map(|i| i % 3 == 0).collect();
    let acts: Vec<f32> = (0..n_units * n)
        .map(|k| {
            let (u, i) = (k / n, k % n);
            let noise = ((k as f32) * 0.618_034).fract() - 0.5;
            if u == 9 && labels[i] {
                noise + 2.0
            } else {
                noise
            }
        })
        .collect();
    let p = ScreenParams::default();
    let mut scr = ScreenScratch::new(n, p.n_scrambles);
    let mut out = vec![UnitScreen::default(); n_units];
    let controls: Vec<f64> = (0..20).map(|i| (i as f64 * 0.1).sin()).collect();
    let cp = ControlParams::new(32, 5.0, 10.0);

    // Warm-up (one screen) — then count.
    let _ = screen_auroc(&acts, n_units, &labels, &p, &mut scr, &mut out);
    let before = ALLOCS.load(Ordering::SeqCst);
    let v = screen_auroc(&acts, n_units, &labels, &p, &mut scr, &mut out);
    let cv = control_verdict(30.0, 64, &controls, &cp);
    let after = ALLOCS.load(Ordering::SeqCst);
    assert_eq!(
        after - before,
        0,
        "scorer allocated {} times",
        after - before
    );
    // Loud-zero defence: the counted region did real work.
    assert!(matches!(v, ScreenVerdict::Screened { .. }), "{v:?}");
    assert!(
        out[9].admitted,
        "planted unit must be admitted: {:?}",
        out[9]
    );
    assert!(
        matches!(
            cv,
            katgpt_pruners::mech_attribution::ControlVerdict::SwitchFound { .. }
        ),
        "{cv:?}"
    );
}
