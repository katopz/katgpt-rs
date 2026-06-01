use std::collections::HashMap;

use crate::benchmark::{BenchCategory, BenchResult, FEATURE_DIMS};
use plotters::prelude::*;

/// Plot benchmark results as a horizontal bar chart SVG.
///
/// Each bar is colored per `BenchResult::color` with throughput + μs/step annotation.
pub fn plot_results(
    results: &[BenchResult],
    path: &str,
    title: &str,
    x_label: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let n = results.len();
    if n == 0 {
        let root = SVGBackend::new(path, (800, 200)).into_drawing_area();
        root.fill(&WHITE)?;
        let style = TextStyle::from(("sans-serif", 18).into_font()).color(&BLACK);
        root.draw_text("No benchmark results", &style, (10, 80))?;
        root.present()?;
        return Ok(());
    }

    let max_tp = results.iter().map(|r| r.throughput).fold(0.0f64, f64::max);
    if max_tp <= 0.0 {
        let root = SVGBackend::new(path, (800, 200)).into_drawing_area();
        root.fill(&WHITE)?;
        root.present()?;
        return Ok(());
    }

    // Dynamic sizing: wider for labels, taller for more results
    let bar_px = 30;
    let gap_px = 10;
    let img_w = 1100;
    let img_h: u32 = (70 + n * (bar_px + gap_px) + 30) as u32;
    let max_val = max_tp * 1.45; // 45% room for annotations

    let root = SVGBackend::new(path, (img_w, img_h)).into_drawing_area();
    root.fill(&WHITE)?;

    // Y axis: bar i centered at integer y=i, labels at each integer tick
    let y_lo = -0.5f64;
    let y_hi = n as f64 - 0.5;

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 22).into_font())
        .margin(10)
        .y_label_area_size(200)
        .x_label_area_size(45)
        .build_cartesian_2d(0f64..max_val, y_lo..y_hi)?;

    chart
        .configure_mesh()
        .y_label_formatter(&|&y: &f64| -> String {
            let idx = y.round() as i64;
            if idx >= 0 && (idx as usize) < n {
                results[idx as usize].label.clone()
            } else {
                String::new()
            }
        })
        .y_labels(n)
        .x_desc(x_label)
        .x_label_style(("sans-serif", 11).into_font())
        .y_label_style(("sans-serif", 11).into_font())
        .draw()?;

    for (i, result) in results.iter().enumerate() {
        let y = i as f64;
        let (cr, cg, cb) = result.color;
        let color = RGBColor(cr, cg, cb);

        // Draw horizontal bar (slightly narrower than full slot for visual gap)
        chart.draw_series(std::iter::once(Rectangle::new(
            [(0.0, y - 0.4), (result.throughput, y + 0.4)],
            color.filled(),
        )))?;

        // Format throughput with unit suffix
        let tp = result.throughput;
        let tp_str = if tp >= 1_000_000.0 {
            let m = tp / 1_000_000.0;
            format!("{m:.2}M")
        } else if tp >= 1_000.0 {
            let k = tp / 1_000.0;
            format!("{k:.0}K")
        } else {
            format!("{tp:.0}")
        };

        // Annotation: throughput + latency
        let us = result.time_per_step_us;
        let label = format!("{tp_str}  ({us:.2} μs)");

        // Place text just past the bar end
        let text_x = result.throughput + max_val * 0.012;
        chart.draw_series(std::iter::once(Text::new(
            label,
            (text_x, y),
            ("sans-serif", 12).into_font(),
        )))?;
    }

    root.present()?;
    Ok(())
}

/// Time series record parsed from `bench/timeseries.csv`.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct TsRow {
    run_date: String,
    commit: String,
    features: String,
    category: String,
    method: String,
    feature_dim: String,
    throughput: f64,
    us_per_step: f64,
    avg_accept_len: f64,
}

/// Parse `bench/timeseries.csv` into rows.
fn parse_timeseries_csv(path: &str) -> Result<Vec<TsRow>, Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(path)?;
    // Estimate row count from byte length (avg ~80 bytes/line) to avoid double scan.
    let estimated_rows = (content.len() / 80).max(64);
    let mut rows = Vec::with_capacity(estimated_rows);
    for line in content.lines().skip(1) {
        let fields: Vec<&str> = line.splitn(10, ',').collect();
        if fields.len() < 8 {
            continue;
        }
        let throughput = fields[5].parse::<f64>().ok();
        let us_per_step = fields[6].parse::<f64>().ok();
        let avg_accept_len = fields[7].parse::<f64>().ok();
        let feature_dim = fields.get(8).unwrap_or(&"").to_string();
        match (throughput, us_per_step, avg_accept_len) {
            (Some(tp), Some(us), Some(aal)) => rows.push(TsRow {
                run_date: fields[0].to_string(),
                commit: fields[1].to_string(),
                features: fields[2].to_string(),
                category: fields[3].to_string(),
                method: fields[4].to_string(),
                feature_dim,
                throughput: tp,
                us_per_step: us,
                avg_accept_len: aal,
            }),
            _ => continue,
        }
    }
    Ok(rows)
}

/// Detect regression: if latest throughput dropped >15% from the max seen for this method.
#[allow(dead_code)]
fn check_regression(rows: &[TsRow], cat: &str) -> Vec<(String, f64, f64)> {
    let cat_rows: Vec<_> = rows.iter().filter(|r| r.category == cat).collect();
    check_regression_filtered(&cat_rows, cat)
}

/// Check regressions using pre-filtered rows (avoids redundant filtering).
fn check_regression_filtered(cat_rows: &[&TsRow], _cat: &str) -> Vec<(String, f64, f64)> {
    if cat_rows.is_empty() {
        return Vec::new();
    }

    // Group by method
    let mut methods: std::collections::BTreeMap<&str, Vec<&TsRow>> =
        std::collections::BTreeMap::new();
    for r in cat_rows {
        methods.entry(&r.method).or_default().push(r);
    }

    let mut regressions = Vec::with_capacity(methods.len());
    for (method, entries) in &methods {
        if entries.len() < 2 {
            continue;
        }
        let max_tp = entries
            .iter()
            .map(|e| e.throughput)
            .fold(f64::MIN, f64::max);
        let latest = entries.last().unwrap().throughput;
        let drop_pct = (max_tp - latest) / max_tp * 100.0;
        if drop_pct > 15.0 {
            regressions.push((method.to_string(), max_tp, latest));
        }
    }
    regressions
}

/// Plot time series line charts per category from cumulative CSV data.
/// Generates one SVG per category showing throughput trend over runs.
/// Returns list of detected regressions (method, max_tp, latest_tp).
pub fn plot_timeseries(
    csv_path: &str,
    bench_dir: &str,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let rows = parse_timeseries_csv(csv_path)?;
    if rows.is_empty() {
        return Ok(Vec::new());
    }

    // Collect all unique categories from the data
    let mut cats: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
    for r in &rows {
        cats.insert(&r.category);
    }

    let mut all_regressions = Vec::with_capacity(cats.len());

    for cat in &cats {
        let cat_rows: Vec<_> = rows.iter().filter(|r| r.category == *cat).collect();
        if cat_rows.is_empty() {
            continue;
        }

        // Detect regressions — pass pre-filtered cat_rows to avoid redundant filtering
        let regressions = check_regression_filtered(&cat_rows, cat);
        for (method, max_tp, latest) in &regressions {
            let drop_pct = (max_tp - latest) / max_tp * 100.0;
            all_regressions.push(format!(
                "🔴 REGRESSION: {method} dropped {drop_pct:.1}% (peak {max_tp:.0} → {latest:.0})"
            ));
        }

        // Collect unique run indices (sequential position) and methods
        let mut run_labels: Vec<String> = Vec::new();
        let mut seen_date_commit = std::collections::HashSet::new();
        for r in &cat_rows {
            let key = format!("{}_{}", r.run_date, r.commit);
            if seen_date_commit.insert(key) {
                run_labels.push(r.run_date.clone());
            }
        }
        let n_runs = run_labels.len();
        if n_runs < 1 {
            continue;
        }

        // Build HashMap index for O(1) run_idx lookup
        let run_index: HashMap<&str, usize> = run_labels
            .iter()
            .enumerate()
            .map(|(i, s)| (s.as_str(), i))
            .collect();

        // Group by method
        let mut methods: std::collections::BTreeMap<&str, Vec<(usize, f64)>> =
            std::collections::BTreeMap::new();
        for r in &cat_rows {
            let run_idx = *run_index.get(r.run_date.as_str()).unwrap_or(&0);
            methods
                .entry(&r.method)
                .or_default()
                .push((run_idx, r.throughput));
        }

        // Dynamic image size
        let img_w = 1200;
        let img_h = 400 + methods.len() as u32 * 20;

        let svg_path = format!("{bench_dir}/timeseries_{cat}.svg");
        let root = SVGBackend::new(&svg_path, (img_w, img_h)).into_drawing_area();
        root.fill(&WHITE)?;

        let max_tp = methods
            .values()
            .flat_map(|pts| pts.iter().map(|(_, v)| *v))
            .fold(0.0f64, f64::max);

        let y_max = max_tp * 1.15;

        let latest_features = cat_rows
            .last()
            .map(|r| r.features.as_str())
            .unwrap_or("unknown");

        let title = format!("{cat} — Time Series [{latest_features}]");
        let mut chart = ChartBuilder::on(&root)
            .caption(&title, ("sans-serif", 20).into_font())
            .margin(12)
            .x_label_area_size(50)
            .y_label_area_size(90)
            .right_y_label_area_size(80)
            .build_cartesian_2d(0usize..(n_runs.max(1) - 1), 0f64..y_max)?;

        chart
            .configure_mesh()
            .x_desc("Run")
            .y_desc("Throughput")
            .x_label_formatter(&|&idx| {
                if idx < run_labels.len() {
                    let label = &run_labels[idx];
                    // Trim to just time portion for compactness
                    label.split('T').nth(1).unwrap_or(label).to_string()
                } else {
                    String::new()
                }
            })
            .x_label_style(("sans-serif", 9).into_font())
            .y_label_style(("sans-serif", 10).into_font())
            .draw()?;

        // Color palette for methods
        let palette: [RGBColor; 10] = [
            BLUE,
            RED,
            GREEN,
            MAGENTA,
            CYAN,
            RGBColor(255, 165, 0),   // orange
            RGBColor(128, 0, 128),   // purple
            RGBColor(0, 128, 128),   // teal
            RGBColor(200, 200, 0),   // olive
            RGBColor(255, 105, 180), // hot pink
        ];

        let mut legend_x = 100i32;
        let mut legend_y = img_h as i32 - 30;

        for (mi, (method, mut points)) in methods.into_iter().enumerate() {
            let color = palette[mi % palette.len()];
            points.sort_by_key(|(idx, _)| *idx);

            // Draw line
            if points.len() >= 2 {
                chart.draw_series(LineSeries::new(points.iter().map(|&(x, y)| (x, y)), &color))?;
            }

            // Draw dots
            chart.draw_series(
                points
                    .iter()
                    .map(|&(x, y)| Circle::new((x, y), 3, color.filled())),
            )?;

            // Legend label
            let legend_style = TextStyle::from(("sans-serif", 10).into_font()).color(&color);
            root.draw_text(&format!("■ {method}"), &legend_style, (legend_x, legend_y))?;
            legend_x += 160;
            if legend_x > img_w as i32 - 180 {
                legend_x = 100;
                legend_y -= 18;
            }
        }

        // If regression detected, add warning text
        if !regressions.is_empty() {
            let warn_style = TextStyle::from(("sans-serif", 14).into_font()).color(&RED);
            root.draw_text(
                &format!("⚠ {} REGRESSION(S) DETECTED", regressions.len()),
                &warn_style,
                (20, 35),
            )?;
        }

        root.present()?;
    }

    Ok(all_regressions)
}

/// Plot feature-grouped bar charts: one SVG per feature dimension.
///
/// Groups benchmarks by their `feature_dim` tag and generates a horizontal bar chart
/// for each of the 10 feature dimensions from the paper comparison matrix.
/// Also generates an E2E game timing chart if any E2E results exist.
pub fn plot_feature_grouped(
    results: &[BenchResult],
    bench_dir: &str,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut written = Vec::new();

    // ── Feature-dimension charts ──
    for feat_cat in &FEATURE_DIMS {
        let cat_results: Vec<_> = results
            .iter()
            .filter(|r| r.category == *feat_cat)
            .cloned()
            .collect();
        if cat_results.is_empty() {
            continue;
        }

        let title = crate::benchmark::bench_category_title(*feat_cat);
        let slug = crate::benchmark::bench_category_str(*feat_cat);
        let path = format!("{bench_dir}/feature_{slug}.svg");

        plot_results(&cat_results, &path, title, "ops/s")?;
        written.push(format!("📈 {title} → {path}"));
    }

    // ── E2E game timing chart ──
    let e2e_results: Vec<_> = results
        .iter()
        .filter(|r| r.category == BenchCategory::E2EGame)
        .cloned()
        .collect();
    if !e2e_results.is_empty() {
        let path = format!("{bench_dir}/e2e_game_timing.svg");
        plot_results(
            &e2e_results,
            &path,
            "E2E Game Timing (Plasma/Hot/Warm/Cold)",
            "ops/s",
        )?;
        written.push(format!("📈 E2E Game Timing → {path}"));
    }

    // ── Summary: feature coverage radar chart ──
    let radar_path = format!("{bench_dir}/feature_coverage_radar.svg");
    plot_feature_radar(results, &radar_path)?;
    written.push(format!("📈 Feature Coverage Radar → {radar_path}"));

    Ok(written)
}

/// Plot a radar/spider chart showing benchmark coverage per feature dimension.
///
/// Each axis represents one feature dimension (SD, KV, Attn, etc.).
/// The value is the number of benchmarks that fall under that dimension.
/// This gives a visual "coverage" of how thoroughly each feature is benchmarked.
fn plot_feature_radar(
    results: &[BenchResult],
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let dims = &[
        ("SD", "Speculative Decoding"),
        ("KV", "KV Optimization"),
        ("Attn", "Attention Innovation"),
        ("Noise", "Noise Scheduling"),
        ("Distill", "Distillation"),
        ("TTC", "Test-Time Compute"),
        ("Route", "Routing/MoE"),
        ("Diff", "Diffusion"),
        ("Game", "Game/Self-Play"),
        ("SIMD", "SIMD/Perf"),
    ];

    // Count benchmarks per feature dimension
    let mut counts: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    for (code, _) in dims {
        counts.insert(code, 0);
    }
    for r in results {
        if !r.feature_dim.is_empty() && counts.contains_key(r.feature_dim.as_str()) {
            *counts.get_mut(r.feature_dim.as_str()).unwrap() += 1;
        }
    }

    let max_count = counts.values().copied().max().unwrap_or(1).max(1);

    // Draw as horizontal bar chart (radar is hard in plotters; bar gives same info clearly)
    let n = dims.len();
    let img_w = 900;
    let img_h: u32 = (70 + n * 35 + 30) as u32;

    let root = SVGBackend::new(path, (img_w, img_h)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_val = max_count as f64 * 1.45;
    let y_lo = -0.5f64;
    let y_hi = n as f64 - 0.5;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Feature Coverage — Benchmarks per Dimension",
            ("sans-serif", 20).into_font(),
        )
        .margin(10)
        .y_label_area_size(200)
        .x_label_area_size(45)
        .build_cartesian_2d(0f64..max_val, y_lo..y_hi)?;

    chart
        .configure_mesh()
        .y_label_formatter(&|&y: &f64| -> String {
            let idx = y.round() as i64;
            if idx >= 0 && (idx as usize) < n {
                let (code, name) = dims[idx as usize];
                format!("{code}: {name}")
            } else {
                String::new()
            }
        })
        .y_labels(n)
        .x_desc("# Benchmarks")
        .x_label_style(("sans-serif", 11).into_font())
        .y_label_style(("sans-serif", 11).into_font())
        .draw()?;

    // Color palette for feature dimensions
    let dim_colors: [RGBColor; 10] = [
        RGBColor(0, 114, 178),   // SD - blue
        RGBColor(230, 159, 0),   // KV - orange
        RGBColor(0, 158, 115),   // Attn - teal
        RGBColor(240, 228, 66),  // Noise - yellow
        RGBColor(204, 121, 167), // Distill - pink
        RGBColor(86, 180, 233),  // TTC - light blue
        RGBColor(213, 94, 0),    // Route - red-orange
        RGBColor(0, 0, 0),       // Diff - black (few benchmarks)
        RGBColor(128, 0, 128),   // Game - purple
        RGBColor(70, 130, 180),  // SIMD - steel blue
    ];

    for (i, (code, _name)) in dims.iter().enumerate() {
        let y = i as f64;
        let count = counts[*code] as f64;
        let color = dim_colors[i];

        chart.draw_series(std::iter::once(Rectangle::new(
            [(0.0, y - 0.4), (count, y + 0.4)],
            color.filled(),
        )))?;

        let label = format!("{count}");
        let text_x = count + max_val * 0.015;
        chart.draw_series(std::iter::once(Text::new(
            label,
            (text_x, y),
            ("sans-serif", 12).into_font(),
        )))?;
    }

    root.present()?;
    Ok(())
}
