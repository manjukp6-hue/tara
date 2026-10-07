//! MathEngine: Production Native Mathematics Specialist Engine for TARA.
//!
//! Provides unified execution across arithmetic, arbitrary precision, fractions,
//! algebra, calculus, geometry, trigonometry, linear algebra, statistics,
//! probability, numerical methods, and symbolic differentiation.

pub mod constants;
pub mod geometry;
pub mod symbolic;
pub mod trigonometry;

pub use constants::*;
pub use geometry::{Geometry, GeometryError};
pub use symbolic::{Expr, SymbolicError};
pub use trigonometry::{TrigError, Trigonometry};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tara_engine::computation::{
    BigInt, ComputationEngine, Fraction, Logarithms, Matrix, NumericalMethods, Polynomial,
    Probability, Roots,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MathEvaluationResult {
    pub success: bool,
    pub domain: String,
    pub operation: String,
    pub input_summary: String,
    pub exact_result: Value,
    pub explanation: String,
    pub execution_time_us: u64,
}

pub struct MathEngine;

impl Default for MathEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl MathEngine {
    pub fn new() -> Self {
        Self
    }

    /// Evaluates structured math requests with exact computation.
    pub fn evaluate(&self, op: &str, params: &Value) -> Result<MathEvaluationResult, String> {
        let start = std::time::Instant::now();
        let exact = match op {
            // Arithmetic & BigInt
            "add" | "sub" | "mul" | "div_rem" | "bigint_add" | "bigint_sub" | "bigint_mul"
            | "bigint_div_rem" => {
                let op_key = if op.starts_with("bigint_") {
                    op
                } else {
                    match op {
                        "add" => "bigint_add",
                        "sub" => "bigint_sub",
                        "mul" => "bigint_mul",
                        "div_rem" => "bigint_div_rem",
                        _ => op,
                    }
                };
                ComputationEngine::execute(op_key, params)?
            }
            "gcd" | "lcm" | "mod_pow" => ComputationEngine::execute(op, params)?,
            "factorial" => {
                let n = params
                    .get("n")
                    .and_then(Value::as_u64)
                    .ok_or("missing 'n'")? as u32;
                let res = BigInt::factorial(n);
                json!({ "n": n, "factorial": res.to_string() })
            }
            "is_prime" => {
                let n_str = params
                    .get("n")
                    .and_then(Value::as_str)
                    .ok_or("missing 'n'")?;
                let n = BigInt::parse(n_str).map_err(|e| e.to_string())?;
                let is_p = n.is_prime(10);
                json!({ "n": n_str, "is_prime": is_p })
            }

            // Fractions & Decimals
            "fraction" | "fraction_op" => ComputationEngine::execute("fraction_op", params)?,
            "decimal" | "decimal_op" => ComputationEngine::execute("decimal_op", params)?,
            "percentage" => ComputationEngine::execute("percentage", params)?,

            // Roots & Logs
            "isqrt" => {
                let n_str = params
                    .get("n")
                    .and_then(Value::as_str)
                    .ok_or("missing 'n'")?;
                let n = BigInt::parse(n_str).map_err(|e| e.to_string())?;
                let res = Roots::isqrt_bigint(&n).map_err(|e| format!("{:?}", e))?;
                json!({ "n": n_str, "isqrt": res.to_string() })
            }
            "nth_root" => {
                let x = params
                    .get("x")
                    .and_then(Value::as_f64)
                    .ok_or("missing 'x'")?;
                let n = params
                    .get("n")
                    .and_then(Value::as_u64)
                    .ok_or("missing 'n'")? as u32;
                let res = Roots::nth_root(x, n).map_err(|e| format!("{:?}", e))?;
                json!({ "x": x, "n": n, "result": res })
            }
            "log" => {
                let x = params
                    .get("x")
                    .and_then(Value::as_f64)
                    .ok_or("missing 'x'")?;
                let base = params.get("base").and_then(Value::as_f64);
                let res = match base {
                    Some(b) => Logarithms::log_base(x, b),
                    None => Logarithms::ln(x),
                }
                .map_err(|e| format!("{:?}", e))?;
                json!({ "x": x, "base": base.unwrap_or(std::f64::consts::E), "result": res })
            }

            // Algebra
            "solve_quadratic" => ComputationEngine::execute("solve_quadratic", params)?,
            "polynomial_eval" => {
                let coeffs: Vec<f64> = params
                    .get("coeffs")
                    .and_then(Value::as_array)
                    .ok_or("missing 'coeffs'")?
                    .iter()
                    .filter_map(Value::as_f64)
                    .collect();
                let x = params
                    .get("x")
                    .and_then(Value::as_f64)
                    .ok_or("missing 'x'")?;
                let p = Polynomial::new(coeffs);
                let val = p.evaluate(x);
                let deriv = p.derivative();
                json!({ "polynomial": p.to_string(), "x": x, "value": val, "derivative": deriv.to_string() })
            }

            // Geometry
            "geometry_circle" | "circle_area" | "circle" => {
                let r = params
                    .get("r")
                    .or_else(|| params.get("radius"))
                    .and_then(Value::as_f64)
                    .ok_or("missing 'r' or 'radius'")?;
                let area = Geometry::circle_area(r).map_err(|e| format!("{:?}", e))?;
                let perimeter = Geometry::circle_perimeter(r).map_err(|e| format!("{:?}", e))?;
                json!({ "radius": r, "area": area, "perimeter": perimeter, "result": area })
            }
            "geometry_sphere" => {
                let r = params
                    .get("r")
                    .or_else(|| params.get("radius"))
                    .and_then(Value::as_f64)
                    .ok_or("missing 'r' or 'radius'")?;
                let vol = Geometry::sphere_volume(r).map_err(|e| format!("{:?}", e))?;
                let sa = Geometry::sphere_surface_area(r).map_err(|e| format!("{:?}", e))?;
                json!({ "radius": r, "volume": vol, "surface_area": sa })
            }

            // Trigonometry
            "trig" => {
                let func = params
                    .get("func")
                    .and_then(Value::as_str)
                    .ok_or("missing 'func'")?;
                let angle = params
                    .get("angle")
                    .and_then(Value::as_f64)
                    .ok_or("missing 'angle'")?;
                let is_deg = params
                    .get("degrees")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                let rad = if is_deg {
                    Trigonometry::deg_to_rad(angle)
                } else {
                    angle
                };

                let val = match func {
                    "sin" => Trigonometry::sin(rad),
                    "cos" => Trigonometry::cos(rad),
                    "tan" => Trigonometry::tan(rad).map_err(|e| format!("{:?}", e))?,
                    "cot" => Trigonometry::cot(rad).map_err(|e| format!("{:?}", e))?,
                    "sec" => Trigonometry::sec(rad).map_err(|e| format!("{:?}", e))?,
                    "csc" => Trigonometry::csc(rad).map_err(|e| format!("{:?}", e))?,
                    _ => return Err(format!("unknown trig function '{}'", func)),
                };
                json!({ "func": func, "angle": angle, "degrees": is_deg, "result": val })
            }

            // Linear Algebra
            "matrix_mult" => ComputationEngine::execute("matrix_mult", params)?,
            "matrix_det" => {
                let n = params
                    .get("n")
                    .and_then(Value::as_u64)
                    .ok_or("missing 'n'")? as usize;
                let data: Vec<f64> = params
                    .get("data")
                    .and_then(Value::as_array)
                    .ok_or("missing 'data'")?
                    .iter()
                    .filter_map(Value::as_f64)
                    .collect();
                let m = Matrix::new(n, n, data).map_err(|e| format!("{:?}", e))?;
                let det = m.determinant().map_err(|e| format!("{:?}", e))?;
                json!({ "n": n, "determinant": det })
            }

            // Statistics & Probability
            "statistics" => ComputationEngine::execute("statistics", params)?,
            "probability" => {
                let p_type = params
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or("combinations");
                match p_type {
                    "combinations" => {
                        let n = params
                            .get("n")
                            .and_then(Value::as_u64)
                            .ok_or("missing 'n'")?;
                        let k = params
                            .get("k")
                            .and_then(Value::as_u64)
                            .ok_or("missing 'k'")?;
                        let res =
                            Probability::combinations(n, k).map_err(|e| format!("{:?}", e))?;
                        json!({ "n": n, "k": k, "combinations": res.to_string() })
                    }
                    "permutations" => {
                        let n = params
                            .get("n")
                            .and_then(Value::as_u64)
                            .ok_or("missing 'n'")?;
                        let k = params
                            .get("k")
                            .and_then(Value::as_u64)
                            .ok_or("missing 'k'")?;
                        let res =
                            Probability::permutations(n, k).map_err(|e| format!("{:?}", e))?;
                        json!({ "n": n, "k": k, "permutations": res.to_string() })
                    }
                    "normal_cdf" => {
                        let x = params
                            .get("x")
                            .and_then(Value::as_f64)
                            .ok_or("missing 'x'")?;
                        let mean = params.get("mean").and_then(Value::as_f64).unwrap_or(0.0);
                        let std_dev = params.get("std_dev").and_then(Value::as_f64).unwrap_or(1.0);
                        let cdf = Probability::normal_cdf(x, mean, std_dev)
                            .map_err(|e| format!("{:?}", e))?;
                        json!({ "x": x, "mean": mean, "std_dev": std_dev, "cdf": cdf })
                    }
                    _ => return Err(format!("unknown probability type '{}'", p_type)),
                }
            }

            // Numerical Methods & Calculus
            "numerical_integral" => {
                let a = params
                    .get("a")
                    .and_then(Value::as_f64)
                    .ok_or("missing 'a'")?;
                let b = params
                    .get("b")
                    .and_then(Value::as_f64)
                    .ok_or("missing 'b'")?;
                let func_name = params.get("func").and_then(Value::as_str).unwrap_or("x^2");
                let n = params.get("panels").and_then(Value::as_u64).unwrap_or(100) as usize;

                let integral = match func_name {
                    "x^2" => NumericalMethods::simpsons_one_third(|x| x * x, a, b, n),
                    "sin" => NumericalMethods::simpsons_one_third(|x| x.sin(), a, b, n),
                    "cos" => NumericalMethods::simpsons_one_third(|x| x.cos(), a, b, n),
                    "exp" => NumericalMethods::simpsons_one_third(|x| x.exp(), a, b, n),
                    _ => {
                        return Err(format!(
                            "unsupported predefined function '{}' for integration",
                            func_name
                        ))
                    }
                }
                .map_err(|e| format!("{:?}", e))?;

                json!({ "func": func_name, "a": a, "b": b, "integral": integral, "method": "Simpson's 1/3" })
            }
            "monte_carlo_pi" => {
                let samples = params
                    .get("samples")
                    .and_then(Value::as_u64)
                    .unwrap_or(100_000) as usize;
                let seed = params.get("seed").and_then(Value::as_u64).unwrap_or(42);
                if samples == 0 || samples > 10_000_000 {
                    return Err("samples must be between 1 and 10,000,000".into());
                }
                let mut rng = StdRng::seed_from_u64(seed);
                let mut inside_circle = 0usize;
                for _ in 0..samples {
                    let x: f64 = rng.gen_range(0.0..1.0);
                    let y: f64 = rng.gen_range(0.0..1.0);
                    if x * x + y * y <= 1.0 {
                        inside_circle += 1;
                    }
                }
                let estimated_pi = 4.0 * (inside_circle as f64) / (samples as f64);
                let error = (estimated_pi - std::f64::consts::PI).abs();
                let rel_error_pct = (error / std::f64::consts::PI) * 100.0;
                json!({
                    "samples": samples,
                    "seed": seed,
                    "inside_circle": inside_circle,
                    "estimated_pi": estimated_pi,
                    "true_pi": std::f64::consts::PI,
                    "absolute_error": error,
                    "relative_error_pct": rel_error_pct,
                    "result": estimated_pi
                })
            }
            "stochastic_process" | "brownian_motion" => {
                let s0 = params.get("s0").and_then(Value::as_f64).unwrap_or(100.0);
                let mu = params.get("mu").and_then(Value::as_f64).unwrap_or(0.05);
                let sigma = params.get("sigma").and_then(Value::as_f64).unwrap_or(0.20);
                let t_years = params
                    .get("time_years")
                    .and_then(Value::as_f64)
                    .unwrap_or(1.0);
                let steps = params.get("steps").and_then(Value::as_u64).unwrap_or(252) as usize;
                let paths = params.get("paths").and_then(Value::as_u64).unwrap_or(50) as usize;
                let seed = params.get("seed").and_then(Value::as_u64).unwrap_or(1337);
                if steps == 0 || steps > 10_000 {
                    return Err("steps must be between 1 and 10,000".into());
                }
                if paths == 0 || paths > 1_000 {
                    return Err("paths must be between 1 and 1,000".into());
                }
                let dt = t_years / steps as f64;
                let mut rng = StdRng::seed_from_u64(seed);
                let mut final_values = Vec::with_capacity(paths);
                for _ in 0..paths {
                    let mut s = s0;
                    for _ in 0..steps {
                        let u1: f64 = rng.gen_range(1e-12..1.0);
                        let u2: f64 = rng.gen_range(0.0..1.0);
                        let z =
                            (-2.0f64 * u1.ln()).sqrt() * (2.0f64 * std::f64::consts::PI * u2).cos();
                        s *= ((mu - 0.5 * sigma * sigma) * dt + sigma * dt.sqrt() * z).exp();
                    }
                    final_values.push(s);
                }
                let mean_final: f64 = final_values.iter().sum::<f64>() / paths as f64;
                let variance: f64 = final_values
                    .iter()
                    .map(|&x| (x - mean_final).powi(2))
                    .sum::<f64>()
                    / paths as f64;
                json!({
                    "initial_value": s0,
                    "drift_mu": mu,
                    "volatility_sigma": sigma,
                    "steps": steps,
                    "paths": paths,
                    "mean_final": mean_final,
                    "stddev_final": variance.sqrt(),
                    "result": mean_final
                })
            }

            // Units
            "unit_convert" => ComputationEngine::execute("unit_convert", params)?,

            _ => return Err(format!("unsupported MathEngine operation '{}'", op)),
        };

        let elapsed = start.elapsed().as_micros() as u64;
        Ok(MathEvaluationResult {
            success: true,
            domain: "mathematics".to_string(),
            operation: op.to_string(),
            input_summary: params.to_string(),
            explanation: format!("Exact computation for '{}' completed in {} µs", op, elapsed),
            exact_result: exact,
            execution_time_us: elapsed,
        })
    }

    /// Natural query interpreter for standard arithmetic and mathematical expressions.
    pub fn solve_query(&self, query: &str) -> Option<MathEvaluationResult> {
        let mut q = query.trim();
        for prefix in &["calculate ", "compute ", "what is ", "evaluate ", "solve "] {
            if let Some(stripped) = q.to_lowercase().strip_prefix(prefix) {
                q = query[query.len() - stripped.len()..].trim();
                break;
            }
        }

        // Pattern: <bigint> + <bigint>
        if let Some(plus_idx) = q.find(" + ") {
            let left = q[..plus_idx].trim();
            let right = q[plus_idx + 3..].trim();
            if let (Ok(a), Ok(b)) = (BigInt::parse(left), BigInt::parse(right)) {
                let res = a.checked_add(&b);
                return Some(MathEvaluationResult {
                    success: true,
                    domain: "arithmetic".into(),
                    operation: "addition".into(),
                    input_summary: format!("{} + {}", left, right),
                    exact_result: json!({ "sum": res.to_string(), "result": res.to_string() }),
                    explanation: format!("Exact sum: {} + {} = {}", left, right, res),
                    execution_time_us: 1,
                });
            }
        }

        // Pattern: <bigint> * <bigint> or <bigint> × <bigint>
        let mul_sep = if q.contains(" * ") {
            Some(" * ")
        } else if q.contains(" × ") {
            Some(" × ")
        } else {
            None
        };
        if let Some(sep) = mul_sep {
            if let Some(idx) = q.find(sep) {
                let left = q[..idx].trim();
                let right = q[idx + sep.len()..].trim();
                if let (Ok(a), Ok(b)) = (BigInt::parse(left), BigInt::parse(right)) {
                    let res = a.checked_mul(&b);
                    return Some(MathEvaluationResult {
                        success: true,
                        domain: "arithmetic".into(),
                        operation: "multiplication".into(),
                        input_summary: format!("{} × {}", left, right),
                        exact_result: json!({ "product": res.to_string(), "result": res.to_string() }),
                        explanation: format!("Exact product: {} × {} = {}", left, right, res),
                        execution_time_us: 1,
                    });
                }
            }
        }

        // Pattern: <bigint> - <bigint>
        if let Some(minus_idx) = q.find(" - ") {
            let left = q[..minus_idx].trim();
            let right = q[minus_idx + 3..].trim();
            if let (Ok(a), Ok(b)) = (BigInt::parse(left), BigInt::parse(right)) {
                let res = a.checked_sub(&b);
                return Some(MathEvaluationResult {
                    success: true,
                    domain: "arithmetic".into(),
                    operation: "subtraction".into(),
                    input_summary: format!("{} - {}", left, right),
                    exact_result: json!({ "difference": res.to_string(), "result": res.to_string() }),
                    explanation: format!("Exact difference: {} - {} = {}", left, right, res),
                    execution_time_us: 1,
                });
            }
        }

        // Pattern: <fraction> / <fraction> or fraction arithmetic
        if q.contains('/') && (q.contains(" + ") || q.contains(" - ") || q.contains(" * ")) {
            // Attempt fraction parsing
            let (op, sep) = if q.contains(" + ") {
                ("add", " + ")
            } else if q.contains(" - ") {
                ("sub", " - ")
            } else {
                ("mul", " * ")
            };
            let parts: Vec<&str> = q.split(sep).collect();
            if parts.len() == 2 {
                if let (Ok(f1), Ok(f2)) = (Fraction::parse(parts[0]), Fraction::parse(parts[1])) {
                    let res = match op {
                        "add" => f1.checked_add(&f2),
                        "sub" => f1.checked_sub(&f2),
                        _ => f1.checked_mul(&f2),
                    }
                    .ok()?;
                    return Some(MathEvaluationResult {
                        success: true,
                        domain: "fractions".into(),
                        operation: op.into(),
                        input_summary: q.to_string(),
                        exact_result: json!({ "fraction": res.to_string(), "decimal": res.to_f64() }),
                        explanation: format!("Exact fraction result: {}", res),
                        execution_time_us: 1,
                    });
                }
            }
        }

        None
    }
}
