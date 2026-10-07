//! Research & Financial Analysis skills native Rust implementation.
//! Replaces:
//! - TARA/SKILLS/research/earnings-preview/scripts/make_plan.py
//! - TARA/SKILLS/research/earnings-preview/scripts/run_plan.py
//! - TARA/SKILLS/research/earnings-preview/scripts/validate_plan.py
//! - TARA/SKILLS/research/earnings-preview/scripts/lib/calc.py
//! - TARA/SKILLS/research/earnings-preview/scripts/lib/changelog.py
//! - TARA/SKILLS/research/earnings-preview/scripts/lib/io_utils.py
//! - TARA/SKILLS/research/earnings-preview/scripts/lib/kpi_packs.py
//! - TARA/SKILLS/research/earnings-preview/scripts/lib/qa.py
//! - TARA/SKILLS/research/earnings-preview/scripts/lib/render.py

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

// ── 1. Financial Calculation Engine ─────────────────────────────────────────────

pub fn parse_fiscal_period_id(fiscal_period_id: &str) -> Result<(i32, i32), String> {
    let s = fiscal_period_id.trim();
    let re = regex::Regex::new(r"^FY(?P<year>\d{4})Q(?P<q>[1-4])$").map_err(|e| e.to_string())?;
    let cap = re
        .captures(s)
        .ok_or_else(|| format!("Invalid fiscal period format: {}", s))?;
    let y: i32 = cap["year"]
        .parse()
        .map_err(|e: std::num::ParseIntError| e.to_string())?;
    let q: i32 = cap["q"]
        .parse()
        .map_err(|e: std::num::ParseIntError| e.to_string())?;
    Ok((y, q))
}

pub fn shift_period(fiscal_period_id: &str, delta_quarters: i32) -> Result<String, String> {
    let (y, q) = parse_fiscal_period_id(fiscal_period_id)?;
    let idx = (y * 4 + (q - 1)) + delta_quarters;
    let new_y = idx / 4;
    let new_q = (idx % 4) + 1;
    Ok(format!("FY{:04}Q{}", new_y, new_q))
}

pub fn is_rate_metric(metric_id: &str, unit: &str) -> bool {
    let u = unit.to_lowercase();
    let mid = metric_id.to_lowercase();
    if u == "ratio" || u == "pct" || u == "percent" || u == "bps" {
        return true;
    }
    if mid.ends_with("_margin")
        || mid.ends_with("_rate")
        || mid == "nrr"
        || mid == "grr"
        || mid == "nim"
    {
        return true;
    }
    false
}

pub fn safe_pct_change(curr: Option<f64>, prev: Option<f64>) -> Option<f64> {
    match (curr, prev) {
        (Some(c), Some(p)) if p > 0.0 => Some((c / p) - 1.0),
        _ => None,
    }
}

pub fn safe_abs_change(curr: Option<f64>, prev: Option<f64>) -> Option<f64> {
    match (curr, prev) {
        (Some(c), Some(p)) => Some(c - p),
        _ => None,
    }
}

pub fn safe_bps_change(curr_ratio: Option<f64>, prev_ratio: Option<f64>) -> Option<f64> {
    safe_abs_change(curr_ratio, prev_ratio).map(|d| d * 10_000.0)
}

pub fn trend_slope(values: &[f64]) -> Option<f64> {
    if values.len() < 3 {
        return None;
    }
    let n = values.len() as f64;
    let x_mean = (n - 1.0) / 2.0;
    let y_mean = values.iter().sum::<f64>() / n;

    let mut num = 0.0;
    let mut den = 0.0;
    for (i, &y) in values.iter().enumerate() {
        let x = i as f64;
        num += (x - x_mean) * (y - y_mean);
        den += (x - x_mean).powi(2);
    }
    if den == 0.0 {
        None
    } else {
        Some(num / den)
    }
}

pub fn auto_flag_delta(curr_est: Option<f64>, cons_est: Option<f64>, metric_is_rate: bool) -> bool {
    match (curr_est, cons_est) {
        (Some(curr), Some(cons)) => {
            if metric_is_rate {
                (curr - cons).abs() >= 0.0025 // 25 bps
            } else {
                if cons == 0.0 {
                    return false;
                }
                ((curr / cons) - 1.0).abs() >= 0.01 // 1% difference
            }
        }
        _ => false,
    }
}

// ── 2. KPI Packs, QA & Markdown Rendering ───────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricValue {
    pub metric_id: String,
    pub name: String,
    pub unit: String,
    pub value: f64,
    pub consensus: Option<f64>,
    pub prior_year: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarningsPreviewReport {
    pub ticker: String,
    pub fiscal_period: String,
    pub metrics: Vec<MetricValue>,
    pub summary_markdown: String,
    pub qa_passed: bool,
}

pub fn generate_earnings_preview(
    ticker: &str,
    fiscal_period: &str,
    metrics: Vec<MetricValue>,
) -> Result<EarningsPreviewReport, String> {
    let mut qa_passed = true;
    let mut md = format!(
        "# Earnings Preview: {} ({})\n\n",
        ticker.to_uppercase(),
        fiscal_period
    );
    md.push_str("| Metric | Unit | Estimate | Consensus | YoY Change | Delta Flag |\n");
    md.push_str("| :--- | :--- | :--- | :--- | :--- | :--- |\n");

    for m in &metrics {
        let is_rate = is_rate_metric(&m.metric_id, &m.unit);
        let flag = auto_flag_delta(Some(m.value), m.consensus, is_rate);
        let yoy = safe_pct_change(Some(m.value), m.prior_year);

        let yoy_str = match yoy {
            Some(y) => format!("{:+.1}%", y * 100.0),
            None => "N/A".to_string(),
        };

        let cons_str = match m.consensus {
            Some(c) => format!("{:.2}", c),
            None => "N/A".to_string(),
        };

        let flag_str = if flag { "⚠️ ALERT" } else { "✅ IN-LINE" };

        md.push_str(&format!(
            "| {} | {} | {:.2} | {} | {} | {} |\n",
            m.name, m.unit, m.value, cons_str, yoy_str, flag_str
        ));

        // Basic QA sanity check
        if m.value.is_nan() || m.value.is_infinite() {
            qa_passed = false;
        }
    }

    Ok(EarningsPreviewReport {
        ticker: ticker.to_uppercase(),
        fiscal_period: fiscal_period.to_string(),
        metrics,
        summary_markdown: md,
        qa_passed,
    })
}

// ── JSON Dispatcher ─────────────────────────────────────────────────────────────

pub fn handle_research_skill(action: &str, params: Value) -> Value {
    match action {
        "shift_period" => {
            let period = params
                .get("period")
                .and_then(|v| v.as_str())
                .unwrap_or("FY2026Q1");
            let delta = params.get("delta").and_then(|v| v.as_i64()).unwrap_or(1) as i32;
            match shift_period(period, delta) {
                Ok(shifted) => json!({ "status": "SUCCESS", "shifted_period": shifted }),
                Err(e) => json!({ "status": "ERROR", "error": e }),
            }
        }
        "calculate_slope" => {
            let values_arr = params.get("values").and_then(|v| v.as_array());
            let values: Vec<f64> = values_arr
                .map(|arr| arr.iter().filter_map(|v| v.as_f64()).collect())
                .unwrap_or_default();
            match trend_slope(&values) {
                Some(slope) => json!({ "status": "SUCCESS", "slope": slope }),
                None => json!({ "status": "ERROR", "error": "Insufficient data points" }),
            }
        }
        "generate_preview_report" => {
            let ticker = params
                .get("ticker")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim();
            let period = params
                .get("fiscal_period")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim();
            let Some(metrics_value) = params.get("metrics").and_then(Value::as_array) else {
                return json!({"status":"ERROR","error":"metrics must be supplied as an array of sourced values; no market data is fabricated"});
            };
            let metrics: Result<Vec<MetricValue>, String> = metrics_value
                .iter()
                .map(|v| {
                    serde_json::from_value(v.clone()).map_err(|e| format!("Invalid metric: {e}"))
                })
                .collect();
            let metrics = match metrics {
                Ok(metrics) => metrics,
                Err(error) => return json!({"status":"ERROR","error":error}),
            };
            if ticker.is_empty() || period.is_empty() || parse_fiscal_period_id(period).is_err() {
                return json!({"status":"ERROR","error":"ticker and valid fiscal_period are required"});
            }
            match generate_earnings_preview(ticker, period, metrics) {
                Ok(rep) => json!({ "status": "SUCCESS", "report": rep }),
                Err(e) => json!({ "status": "ERROR", "error": e }),
            }
        }
        _ => json!({ "status": "ERROR", "error": format!("Unknown research action: {}", action) }),
    }
}
