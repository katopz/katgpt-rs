//! Spectral Threat Arena — GOAT Gate for Plan 241.
//!
//! Compares reactive NPC vs spectral NPC against a variable combo player.
//! The spectral NPC uses LinOSS modal analysis to predict combo timing
//! and prioritize dodging heavy attacks over light ones.
//!
//! **Status:** GOAT gate NOT YET PASSED. The spectral prediction infrastructure
//! is correct, but the urgency heuristic requires frequency tuning to produce
//! actionable predictions. Kept as opt-in experiment.
//!
//! Run: `cargo run --example spectral_threat_arena --features spectral_threat`

use katgpt_core::sense::CombatRhythmTracker;

// ── Constants ──────────────────────────────────────────────────

const DT: f32 = 0.016;
const TPS: u32 = 62;
const DURATION_SECS: u32 = 60;
const TOTAL_TICKS: u32 = TPS * DURATION_SECS;
const MAX_HP: i32 = 2000;

// Player behavior: alternating combo patterns
// Pattern A: slash(60) → slash(60) → heavy(200), intervals: 10, 14, 28
// Pattern B: slash(40) → slash(40) → slash(40) → heavy(250), intervals: 8, 8, 8, 28
// Player alternates: 3 cycles of A, then 2 cycles of B, repeat
const PATTERN_A_DAMAGE: &[i32] = &[60, 60, 200];
const PATTERN_A_INTERVALS: &[u32] = &[10, 14, 28];
const PATTERN_B_DAMAGE: &[i32] = &[40, 40, 40, 250];
const PATTERN_B_INTERVALS: &[u32] = &[8, 8, 8, 28];

const THREAT_VISIBILITY_TICKS: u32 = 5;
const DODGE_COOLDOWN_TICKS: u32 = 16;

// ── NPC State ──────────────────────────────────────────────────

#[derive(Debug)]
struct NpcState {
    hp: i32,
    dodges: u32,
    hits_taken: u32,
    damage_taken: i32,
    last_dodge_tick: u32,
    total_attacks: u32,
    damage_avoided: i32,
}

impl NpcState {
    fn new() -> Self {
        Self {
            hp: MAX_HP,
            dodges: 0,
            hits_taken: 0,
            damage_taken: 0,
            last_dodge_tick: 0,
            total_attacks: 0,
            damage_avoided: 0,
        }
    }

    fn dodge_rate(&self) -> f32 {
        if self.total_attacks == 0 {
            0.0
        } else {
            self.dodges as f32 / self.total_attacks as f32
        }
    }

    fn hp_remaining_ratio(&self) -> f32 {
        (self.hp as f32 / MAX_HP as f32).max(0.0)
    }
}

// ── Attack Schedule ────────────────────────────────────────────

#[derive(Debug)]
struct Attack {
    impact_tick: u32,
    damage: i32,
    is_heavy: bool,
}

struct AttackSchedule {
    attacks: Vec<Attack>,
}

impl AttackSchedule {
    fn generate() -> Self {
        let mut attacks = Vec::with_capacity(400);
        let mut tick = 30u32;
        let mut cycle = 0usize;

        while tick < TOTAL_TICKS {
            let phase = cycle % 5;
            let (damages, intervals) = if phase < 3 {
                (PATTERN_A_DAMAGE, PATTERN_A_INTERVALS)
            } else {
                (PATTERN_B_DAMAGE, PATTERN_B_INTERVALS)
            };

            for (i, &dmg) in damages.iter().enumerate() {
                if tick >= TOTAL_TICKS {
                    break;
                }
                attacks.push(Attack {
                    impact_tick: tick,
                    damage: dmg,
                    is_heavy: i == damages.len() - 1,
                });
                tick += intervals[i];
            }
            cycle += 1;
        }
        Self { attacks }
    }
}

// ── Reactive NPC ───────────────────────────────────────────────

fn run_reactive_npc(schedule: &AttackSchedule) -> NpcState {
    let mut npc = NpcState::new();
    for attack in &schedule.attacks {
        npc.total_attacks += 1;
        let threat_tick = attack.impact_tick.saturating_sub(THREAT_VISIBILITY_TICKS);
        let off_cooldown = threat_tick > npc.last_dodge_tick + DODGE_COOLDOWN_TICKS;
        if off_cooldown {
            npc.dodges += 1;
            npc.damage_avoided += attack.damage;
            npc.last_dodge_tick = threat_tick;
        } else {
            npc.hp = (npc.hp - attack.damage).max(0);
            npc.hits_taken += 1;
            npc.damage_taken += attack.damage;
        }
    }
    npc
}

// ── Spectral NPC ───────────────────────────────────────────────

fn run_spectral_npc(schedule: &AttackSchedule) -> NpcState {
    let mut npc = NpcState::new();
    let mut tracker = CombatRhythmTracker::with_combat_frequencies(DT);
    tracker.register(1);
    let mut prev_tick = 0u32;

    for attack in &schedule.attacks {
        let tick = attack.impact_tick;
        npc.total_attacks += 1;

        // Advance tracker between events (natural decay)
        let gap = tick.saturating_sub(prev_tick);
        for _ in 0..gap {
            tracker.ingest_damage(1, 0.0, tick);
        }
        prev_tick = tick;

        // Feed damage impulse
        tracker.ingest_damage(1, attack.damage as f32, tick);

        let threat_tick = tick.saturating_sub(THREAT_VISIBILITY_TICKS);
        let off_cooldown = threat_tick > npc.last_dodge_tick + DODGE_COOLDOWN_TICKS;

        if off_cooldown {
            let should_dodge = if attack.is_heavy {
                true // Always dodge heavy
            } else {
                // Light attack: use spectral prediction to decide
                let features = tracker.extract_features(1);
                if features.rhythm_confidence > 0.15 {
                    let urgency = features.dodge_urgency();
                    // If urgency is high (imminent peak), save dodge for heavy
                    urgency <= 0.6
                } else {
                    true // No prediction — dodge normally
                }
            };

            if should_dodge {
                npc.dodges += 1;
                npc.damage_avoided += attack.damage;
                npc.last_dodge_tick = threat_tick;
            } else {
                npc.hp = (npc.hp - attack.damage).max(0);
                npc.hits_taken += 1;
                npc.damage_taken += attack.damage;
            }
        } else {
            npc.hp = (npc.hp - attack.damage).max(0);
            npc.hits_taken += 1;
            npc.damage_taken += attack.damage;
        }
    }
    npc
}

// ── Main ───────────────────────────────────────────────────────

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Plan 241: Spectral Threat Arena — GOAT Gate               ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let schedule = AttackSchedule::generate();
    println!(
        "Attack schedule: {} attacks over {}s",
        schedule.attacks.len(),
        DURATION_SECS
    );
    println!("Player pattern: 3×(slash60→slash60→heavy200) + 2×(slash40×3→heavy250)");
    println!(
        "  Threat visibility: {} ticks ({:.0}ms) | Dodge cooldown: {} ticks ({:.0}ms)",
        THREAT_VISIBILITY_TICKS,
        THREAT_VISIBILITY_TICKS as f32 * DT * 1000.0,
        DODGE_COOLDOWN_TICKS,
        DODGE_COOLDOWN_TICKS as f32 * DT * 1000.0
    );
    println!();

    let npc_a = run_reactive_npc(&schedule);
    let npc_b = run_spectral_npc(&schedule);

    println!("┌──────────────────────────────────────────────────────────┐");
    println!("│  RESULTS                                                 │");
    println!("├────────────────────┬─────────────┬─────────────┬─────────┤");
    println!("│  Metric            │  NPC A (R)  │  NPC B (S)  │  Δ      │");
    println!("├────────────────────┼─────────────┼─────────────┼─────────┤");
    println!(
        "│  Dodge Rate        │  {:5.1}%     │  {:5.1}%     │  {:+.1}%  │",
        npc_a.dodge_rate() * 100.0,
        npc_b.dodge_rate() * 100.0,
        (npc_b.dodge_rate() - npc_a.dodge_rate()) * 100.0
    );
    println!(
        "│  HP Remaining      │  {:5}       │  {:5}       │  {:+5}   │",
        npc_a.hp,
        npc_b.hp,
        npc_b.hp - npc_a.hp
    );
    println!(
        "│  HP Ratio          │  {:5.1}%     │  {:5.1}%     │  {:+.1}%  │",
        npc_a.hp_remaining_ratio() * 100.0,
        npc_b.hp_remaining_ratio() * 100.0,
        (npc_b.hp_remaining_ratio() - npc_a.hp_remaining_ratio()) * 100.0
    );
    println!(
        "│  Damage Taken      │  {:5}       │  {:5}       │  {:+5}   │",
        npc_a.damage_taken,
        npc_b.damage_taken,
        npc_b.damage_taken - npc_a.damage_taken
    );
    println!(
        "│  Damage Avoided    │  {:5}       │  {:5}       │  {:+5}   │",
        npc_a.damage_avoided,
        npc_b.damage_avoided,
        npc_b.damage_avoided - npc_a.damage_avoided
    );
    println!(
        "│  Hits Taken        │  {:5}       │  {:5}       │  {:+5}   │",
        npc_a.hits_taken,
        npc_b.hits_taken,
        npc_b.hits_taken as i64 - npc_a.hits_taken as i64
    );
    println!(
        "│  Total Attacks     │  {:5}       │  {:5}       │         │",
        npc_a.total_attacks, npc_b.total_attacks
    );
    println!("└────────────────────┴─────────────┴─────────────┴─────────┘");
    println!();

    // GOAT Gate
    println!("─── GOAT Gate ─────────────────────────────────────────────");
    let a_dmg = npc_a.damage_taken as f32;
    let b_dmg = npc_b.damage_taken as f32;
    let damage_reduction = if a_dmg > 0.0 {
        (a_dmg - b_dmg) / a_dmg * 100.0
    } else {
        0.0
    };

    let hp_pass = npc_b.hp > npc_a.hp;
    let damage_pass = damage_reduction >= 10.0;

    println!(
        "  Damage reduction:   {:.1}% {} (threshold: 10%)",
        damage_reduction,
        if damage_pass { "✅ PASS" } else { "❌ FAIL" }
    );
    println!(
        "  HP higher:          {} {}",
        if hp_pass { "YES" } else { "NO" },
        if hp_pass { "✅ PASS" } else { "❌ FAIL" }
    );

    if damage_pass && hp_pass {
        println!();
        println!("  🐐 GOAT CONFIRMED — spectral_threat passes arena proof.");
        println!("  Proposal: promote spectral_threat to default-ON.");
    } else {
        println!();
        println!("  ❌ GOAT gate not yet passed.");
        println!("  Status: infrastructure correct, urgency heuristic needs frequency tuning.");
        println!("  Keep as opt-in experiment behind `spectral_threat` feature flag.");
    }

    println!("─────────────────────────────────────────────────────────────");
}
