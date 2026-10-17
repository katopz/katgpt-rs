//! Issue 930 T5 (G1b) — the REAL-fixture paired gate: δ_surprise vs the
//! Plan-277 dual-EMA value-surprise on a per-NPC belief trajectory replayed
//! from the production lane.
//!
//! Fixture provenance: `fixtures/sync_bank_g1b_boot_fixture.bin`, dumped by
//! riir-ai's sync-relation lane (commit `399b4ee9f`) during the REAL-boot
//! hunt test (`sync_relation_boot_tests::sync_relation_lane_on_the_real_boot`
//! — the stalemate rig: 3 000 hunt ticks + 900 stationary ticks, 48
//! monsters, `EntityId`-keyed) via the env-armed `RIIR_SYNC_RELATION_DUMP`
//! sink: the bank's EXACT input stream (`(entity, tick, z)` — the 8-dim
//! padded channel vector) plus the labeled relation flips (the lane's
//! Formed/Released edges, 49 of them). Format `RSR1` v1, little-endian:
//! header `magic[4] ver u32`; records `tag u8` — `0x01` OBS `tick u64
//! entity[16] z 8×f32`; `0x02` EVENT `tick u64 entity[16] tail_present u8
//! [tail[16]] relation u64`.
//!
//! The paired protocol (the issue's law): per-entity replay through the
//! SAME `SyncBank<8, 3>` the lane feeds, and the Plan-277 dual-EMA
//! value-surprise (`TemporalDerivativeKernel<8>::new(0.3, 0.03)`, the G1a
//! posture) fed the identical stream. Detection windows are ±5 ticks
//! around each labeled edge; the false-positive budget is set per detector
//! on the QUIET population (observation ticks outside every event window
//! whose own ±5-tick neighborhood is contiguous in that entity's stream —
//! the evicted-window correction, so a threshold quantile is a real
//! quantile, not an artifact of the lane's ring gaps). Events split WARM
//! (≥ 32 pre-event observations — the entity's banks carry real history)
//! from COLD (the fresh-entry transient a respawned monster's first
//! coupling produces — counted, reported, excluded from the headline rate:
//! the warm/cold census rides the output).
//!
//! Verdict measured on the frozen fixture (2026-10-11): δ_surprise detects
//! 23/23 warm events at EVERY matched FP budget (1/2/5/10 %) while the
//! value-surprise detects 0/23 — Wilson 95 % intervals disjoint — the
//! strict paired superiority the promotion law requires. THIS TEST IS THE
//! PIN: it re-measures on every run and reds if a detector or fixture
//! drifts past the margins.
//!
//! Promotion posture (Issue 930 / katgpt-rs Cargo.toml `sync_bank` row):
//! G1b PASS keeps the lane OPT-IN — the default promotion is the
//! owner-gated act, never this test's.

use std::collections::BTreeMap;
use std::path::PathBuf;

use katgpt_types::sync_bank::{SYNC_BANK_LADDER, SyncBank};
use katgpt_types::temporal::TemporalDerivativeKernel;

/// The fixture path (committed beside this test).
fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sync_bank_g1b_boot_fixture.bin")
}

/// One observation: tick, entity key, the bank's 8-dim input.
struct Obs {
    tick: u64,
    entity: [u8; 16],
    z: [f32; 8],
}

/// One labeled relation flip: tick, entity key.
struct Ev {
    tick: u64,
    entity: [u8; 16],
}

struct Fixture {
    obs: Vec<Obs>,
    events: Vec<Ev>,
}

/// Parse `RSR1` v1. Structural pins: the header + per-record bounds — a
/// truncated or foreign file refuses here, never replays.
fn parse(data: &[u8]) -> Fixture {
    assert_eq!(&data[..4], b"RSR1", "fixture magic");
    assert_eq!(
        u32::from_le_bytes(data[4..8].try_into().expect("ver")),
        1,
        "fixture version"
    );
    let mut obs = Vec::new();
    let mut events = Vec::new();
    let mut i = 8;
    while i < data.len() {
        match data[i] {
            0x01 => {
                assert!(i + 57 <= data.len(), "truncated OBS at {i}");
                let tick = u64::from_le_bytes(data[i + 1..i + 9].try_into().expect("tick"));
                let mut entity = [0u8; 16];
                entity.copy_from_slice(&data[i + 9..i + 25]);
                let mut z = [0.0f32; 8];
                for (k, slot) in z.iter_mut().enumerate() {
                    *slot = f32::from_le_bytes(
                        data[i + 25 + k * 4..i + 29 + k * 4].try_into().expect("z"),
                    );
                }
                obs.push(Obs { tick, entity, z });
                i += 57;
            }
            0x02 => {
                assert!(i + 26 <= data.len(), "truncated EVENT at {i}");
                let tick = u64::from_le_bytes(data[i + 1..i + 9].try_into().expect("tick"));
                let mut entity = [0u8; 16];
                entity.copy_from_slice(&data[i + 9..i + 25]);
                let present = data[i + 25];
                assert!(present <= 1, "tail-present flag {present} at {i}");
                i += 26 + if present == 1 { 16 } else { 0 };
                assert!(i + 8 <= data.len(), "truncated EVENT relation at {i}");
                let _relation = u64::from_le_bytes(data[i..i + 8].try_into().expect("rel"));
                i += 8;
                events.push(Ev { tick, entity });
            }
            tag => panic!("unknown record tag {tag} at offset {i}"),
        }
    }
    Fixture { obs, events }
}

/// Nearest-rank percentile on a sorted slice — the rank-`⌈p·n⌉` form, so a
/// tail quantile NEVER lands on the max (the n·p index trap: at n = 2 234
/// the p99 index sits 23 ranks below the top, and the max is not a
/// percentile).
fn percentile(sorted: &[f32], p: f64) -> f32 {
    let n = sorted.len();
    assert!(n > 0, "percentile of empty population");
    let idx = ((p * n as f64).ceil() as usize)
        .saturating_sub(1)
        .min(n - 1);
    sorted[idx]
}

/// Wilson 95 % score interval.
fn wilson(k: usize, n: usize) -> (f64, f64) {
    let z = 1.96;
    let n = n as f64;
    let p = k as f64 / n;
    let d = 1.0 + z * z / n;
    let center = (p + z * z / (2.0 * n)) / d;
    let half = z * (p * (1.0 - p) / n + z * z / (4.0 * n * n)).sqrt() / d;
    ((center - half).max(0.0), (center + half).min(1.0))
}

/// Detection windows and population constants (the measured protocol).
const WINDOW: i64 = 5;
/// Warm-event bar: the entity owns at least this many PRE-event
/// observations outside the window (≈ half the slow bank's 64-tick
/// horizon — real covariance history, not a cold-entry transient).
const WARM_PRE: usize = 32;
/// The FP budgets the paired superiority must hold at (fractions).
const FP_BUDGETS: [f64; 4] = [0.01, 0.02, 0.05, 0.10];

#[test]
fn g1b_relation_surprise_beats_value_surprise_on_the_real_boot_fixture() {
    let data = std::fs::read(fixture_path()).expect("fixture readable");
    let fx = parse(&data);

    // The structural census (the fixture's content pin — a re-dump that
    // changes the boot's shape re-reds here for a look, not silently).
    assert_eq!(fx.obs.len(), 3_229, "observation count");
    assert_eq!(fx.events.len(), 49, "labeled event count");
    let entities: std::collections::BTreeSet<[u8; 16]> = fx.obs.iter().map(|o| o.entity).collect();
    assert_eq!(entities.len(), 48, "entity count");

    // Per-entity streams in dump order (the bank's exact input sequence).
    let mut streams: BTreeMap<[u8; 16], Vec<(u64, [f32; 8])>> = BTreeMap::new();
    for o in &fx.obs {
        streams.entry(o.entity).or_default().push((o.tick, o.z));
    }

    // Replay: BOTH detectors over the identical stream, per entity.
    // `(entity, tick) -> (surprise, value)` in one map keyed like the
    // event/quiet sets.
    let mut scores: std::collections::HashMap<([u8; 16], u64), (f32, f32)> =
        std::collections::HashMap::with_capacity(fx.obs.len());
    for (entity, rows) in &streams {
        let mut bank = SyncBank::<8, 3>::new(SYNC_BANK_LADDER);
        let mut tk = TemporalDerivativeKernel::<8>::new(0.3, 0.03);
        for (tick, z) in rows {
            bank.observe(z);
            let sv = tk.observe(z);
            let vn = sv.iter().map(|x| x * x).sum::<f32>().sqrt();
            scores.insert((*entity, *tick), (bank.surprise_max(), vn));
        }
    }
    assert_eq!(scores.len(), fx.obs.len(), "every observation scored");

    let mut entity_ticks: BTreeMap<[u8; 16], std::collections::BTreeSet<u64>> = BTreeMap::new();
    for o in &fx.obs {
        entity_ticks.entry(o.entity).or_default().insert(o.tick);
    }
    let in_stream = |entity: &[u8; 16], t: u64| -> bool {
        entity_ticks.get(entity).is_some_and(|ts| ts.contains(&t))
    };

    // Event-window keys (± WINDOW ticks of a labeled edge) + the warm/cold
    // split (pre-event observation count outside the window).
    let mut event_keys: std::collections::HashSet<([u8; 16], u64)> =
        std::collections::HashSet::new();
    let mut warm_events: Vec<(&Ev, usize)> = Vec::new();
    let mut cold_events: Vec<(&Ev, usize)> = Vec::new();
    for ev in &fx.events {
        let rows = &streams[&ev.entity];
        for (t, _) in rows {
            if t.abs_diff(ev.tick) <= WINDOW as u64 {
                event_keys.insert((ev.entity, *t));
            }
        }
        let n_pre = rows
            .iter()
            .filter(|(t, _)| (*t as i64) < ev.tick as i64 - WINDOW)
            .count();
        if n_pre >= WARM_PRE {
            warm_events.push((ev, n_pre));
        } else {
            cold_events.push((ev, n_pre));
        }
    }

    // The QUIET population: outside every event window AND contiguous at
    // the window's own span (the evicted-window correction — a quantile
    // over ticks whose ±WINDOW neighborhood exists in the entity's stream).
    let quiet: Vec<(f32, f32)> = scores
        .iter()
        .filter(|((entity, t), _)| {
            !event_keys.contains(&(*entity, *t))
                && (-WINDOW..=WINDOW)
                    .step_by(2)
                    .all(|dt| in_stream(entity, (*t as i64 + dt) as u64))
        })
        .map(|(_, v)| *v)
        .collect();
    assert!(
        quiet.len() >= 2_000,
        "quiet support collapsed: {quiet_len}",
        quiet_len = quiet.len()
    );

    let mut quiet_s: Vec<f32> = quiet.iter().map(|(s, _)| *s).collect();
    let mut quiet_v: Vec<f32> = quiet.iter().map(|(_, v)| *v).collect();
    quiet_s.sort_by(|a, b| a.total_cmp(b));
    quiet_v.sort_by(|a, b| a.total_cmp(b));

    // Per-event detection at a matched FP budget: threshold = the quiet
    // quantile; an event detects when ANY of its window's observations
    // crosses its OWN detector's threshold.
    let detects =
        |flagged: &std::collections::HashSet<([u8; 16], u64)>, pop: &[(&Ev, usize)]| -> usize {
            pop.iter()
                .filter(|(ev, _)| {
                    streams[&ev.entity].iter().any(|(t, _)| {
                        t.abs_diff(ev.tick) <= WINDOW as u64 && flagged.contains(&(ev.entity, *t))
                    })
                })
                .count()
        };

    let mut reported = String::with_capacity(512);
    reported.push_str(&format!(
        "G1b: warm events {}, cold events (census only) {}, quiet samples {}\n",
        warm_events.len(),
        cold_events.len(),
        quiet.len()
    ));

    for fp in FP_BUDGETS {
        let s_thr = percentile(&quiet_s, 1.0 - fp);
        let v_thr = percentile(&quiet_v, 1.0 - fp);
        let flagged_s: std::collections::HashSet<([u8; 16], u64)> = scores
            .iter()
            .filter(|(_, (s, _))| *s >= s_thr)
            .map(|(k, _)| *k)
            .collect();
        let flagged_v: std::collections::HashSet<([u8; 16], u64)> = scores
            .iter()
            .filter(|(_, (_, v))| *v >= v_thr)
            .map(|(k, _)| *k)
            .collect();

        let n = warm_events.len();
        let d_s = detects(&flagged_s, &warm_events);
        let d_v = detects(&flagged_v, &warm_events);
        let (s_lo, s_hi) = wilson(d_s, n);
        let (v_lo, v_hi) = wilson(d_v, n);
        reported.push_str(&format!(
            "  FP {:+.0}%: surprise {d_s}/{n} [{s_lo:.3},{s_hi:.3}]  value {d_v}/{n} [{v_lo:.3},{v_hi:.3}]\n",
            fp * 100.0
        ));

        // The measured verdict's margins: δ_surprise detects ≥ 90 % of warm
        // events at EVERY matched budget (measured 23/23); the value
        // baseline ≤ 30 % (measured 0/23); the intervals DISJOINT — strict
        // paired superiority, the promotion law's exact requirement.
        assert!(
            d_s as f64 / n as f64 >= 0.90,
            "δ_surprise detection {d_s}/{n} below 0.90 at FP {fp}"
        );
        assert!(
            d_v as f64 / n as f64 <= 0.30,
            "value-surprise detection {d_v}/{n} above 0.30 at FP {fp} — the baseline got better; re-pin the margins from data"
        );
        assert!(
            s_lo > v_hi,
            "Wilson intervals not disjoint at FP {fp}: surprise [{s_lo},{s_hi}] vs value [{v_lo},{v_hi}]"
        );
    }

    // The threshold quantiles are honest: the measured flag rate on quiet
    // sits within 1.5× of each budget (a percentile-index or off-by-one
    // slip would blow through this).
    for fp in FP_BUDGETS {
        let s_thr = percentile(&quiet_s, 1.0 - fp);
        let rate = quiet.iter().filter(|(s, _)| *s >= s_thr).count() as f64 / quiet.len() as f64;
        assert!(
            rate <= fp * 1.5 + 1e-9,
            "quiet flag rate {rate} exceeds 1.5× the {fp} budget"
        );
    }

    // Both detectors' scores are finite (a NaN poisons every window above).
    for ((entity, tick), (s, v)) in &scores {
        assert!(
            s.is_finite() && v.is_finite(),
            "non-finite score at entity {:?} tick {tick}: ({s}, {v})",
            &entity[..4]
        );
    }

    print!("{reported}");
}
