//! TARA Native Exact Mathematics & Scientific Computation Subsystem.
//!
//! Complete, non-fabricated, high-precision computational capabilities:
//! - Arbitrary-precision integer arithmetic (`BigInt`, GCD, LCM, modular exp/inv, primes, factorial)
//! - Exact rational arithmetic (`Fraction`, `BigFraction`)
//! - Arbitrary-precision decimal arithmetic (`BigDecimal`)
//! - Percentage calculations and scientific notation
//! - Integer and real root extraction (isqrt, icbrt, nth-root)
//! - Logarithms (ln, log10, log2, log_b)
//! - Vector and Matrix linear algebra (norms, cross, dot, det, inverse, Ax=b, eigenvalues)
//! - Univariate & bivariate descriptive statistics (mean, median, mode, variance, stddev, skewness, covariance, correlation, regression)
//! - Probability distributions (Binomial, Poisson, Gaussian CDF/PDF, combinations, permutations, Bayes' theorem)
//! - Polynomial algebra and equation solving (linear, quadratic, cubic, derivatives, integrals)
//! - Numerical analysis (Bisection, Newton-Raphson, Secant, Simpson's quadrature, RK4 ODE solver)
//! - Physical unit conversion across 10 dimensions
//! - Multilingual math & science terminology (Kannada, English, symbols).

pub mod algebra;
pub mod decimal;
pub mod elementary;
pub mod fraction;
pub mod integer;
pub mod linear_algebra;
pub mod multilingual;
pub mod numerical;
pub mod probability;
pub mod statistics;
pub mod units;

pub use algebra::{ComplexRoot, EquationSolver, Polynomial};
pub use decimal::BigDecimal;
pub use elementary::{Logarithms, Percentage, Roots};
pub use fraction::{BigFraction, Fraction};
pub use integer::{checked_gcd_i64, checked_lcm_i64, BigInt, Sign};
pub use linear_algebra::{Matrix, Vector};
pub use multilingual::{MultilingualEngine, MultilingualTerm};
pub use numerical::NumericalMethods;
pub use probability::Probability;
pub use statistics::{DescriptiveStats, Statistics};
pub use units::{Dimension, UnitConverter};

use serde_json::{json, Value};

/// Central high-level computational dispatcher for TARA Brain and Tool calls.
pub struct ComputationEngine;

impl ComputationEngine {
    /// Execute an exact computation given an operation name and parameters JSON.
    pub fn execute(operation: &str, params: &Value) -> Result<Value, String> {
        match operation {
            "bigint_add" => {
                let a = params
                    .get("a")
                    .and_then(Value::as_str)
                    .ok_or("missing 'a'")?;
                let b = params
                    .get("b")
                    .and_then(Value::as_str)
                    .ok_or("missing 'b'")?;
                let big_a = BigInt::parse(a).map_err(|e| e.to_string())?;
                let big_b = BigInt::parse(b).map_err(|e| e.to_string())?;
                let res = big_a.checked_add(&big_b);
                Ok(json!({ "result": res.to_string(), "operation": "bigint_add" }))
            }
            "bigint_sub" => {
                let a = params
                    .get("a")
                    .and_then(Value::as_str)
                    .ok_or("missing 'a'")?;
                let b = params
                    .get("b")
                    .and_then(Value::as_str)
                    .ok_or("missing 'b'")?;
                let big_a = BigInt::parse(a).map_err(|e| e.to_string())?;
                let big_b = BigInt::parse(b).map_err(|e| e.to_string())?;
                let res = big_a.checked_sub(&big_b);
                Ok(json!({ "result": res.to_string(), "operation": "bigint_sub" }))
            }
            "bigint_mul" => {
                let a = params
                    .get("a")
                    .and_then(Value::as_str)
                    .ok_or("missing 'a'")?;
                let b = params
                    .get("b")
                    .and_then(Value::as_str)
                    .ok_or("missing 'b'")?;
                let big_a = BigInt::parse(a).map_err(|e| e.to_string())?;
                let big_b = BigInt::parse(b).map_err(|e| e.to_string())?;
                let res = big_a.checked_mul(&big_b);
                Ok(json!({ "result": res.to_string(), "operation": "bigint_mul" }))
            }
            "bigint_div_rem" => {
                let a = params
                    .get("a")
                    .and_then(Value::as_str)
                    .ok_or("missing 'a'")?;
                let b = params
                    .get("b")
                    .and_then(Value::as_str)
                    .ok_or("missing 'b'")?;
                let big_a = BigInt::parse(a).map_err(|e| e.to_string())?;
                let big_b = BigInt::parse(b).map_err(|e| e.to_string())?;
                let (q, r) = big_a.div_rem(&big_b).map_err(|e| e.to_string())?;
                Ok(json!({ "quotient": q.to_string(), "remainder": r.to_string() }))
            }
            "gcd" => {
                let a = params
                    .get("a")
                    .and_then(Value::as_str)
                    .ok_or("missing 'a'")?;
                let b = params
                    .get("b")
                    .and_then(Value::as_str)
                    .ok_or("missing 'b'")?;
                let big_a = BigInt::parse(a).map_err(|e| e.to_string())?;
                let big_b = BigInt::parse(b).map_err(|e| e.to_string())?;
                let res = big_a.gcd(&big_b);
                Ok(json!({ "gcd": res.to_string() }))
            }
            "lcm" => {
                let a = params
                    .get("a")
                    .and_then(Value::as_str)
                    .ok_or("missing 'a'")?;
                let b = params
                    .get("b")
                    .and_then(Value::as_str)
                    .ok_or("missing 'b'")?;
                let big_a = BigInt::parse(a).map_err(|e| e.to_string())?;
                let big_b = BigInt::parse(b).map_err(|e| e.to_string())?;
                let res = big_a.lcm(&big_b).map_err(|e| e.to_string())?;
                Ok(json!({ "lcm": res.to_string() }))
            }
            "mod_pow" => {
                let base = params
                    .get("base")
                    .and_then(Value::as_str)
                    .ok_or("missing 'base'")?;
                let exp = params
                    .get("exp")
                    .and_then(Value::as_str)
                    .ok_or("missing 'exp'")?;
                let modulus = params
                    .get("modulus")
                    .and_then(Value::as_str)
                    .ok_or("missing 'modulus'")?;
                let b = BigInt::parse(base).map_err(|e| e.to_string())?;
                let e = BigInt::parse(exp).map_err(|e| e.to_string())?;
                let m = BigInt::parse(modulus).map_err(|e| e.to_string())?;
                let res = b.mod_pow(&e, &m).map_err(|err| err.to_string())?;
                Ok(json!({ "result": res.to_string() }))
            }
            "fraction_op" => {
                let op = params.get("op").and_then(Value::as_str).unwrap_or("add");
                let f1_str = params
                    .get("f1")
                    .and_then(Value::as_str)
                    .ok_or("missing 'f1'")?;
                let f2_str = params
                    .get("f2")
                    .and_then(Value::as_str)
                    .ok_or("missing 'f2'")?;
                let f1 = Fraction::parse(f1_str).map_err(|e| e.to_string())?;
                let f2 = Fraction::parse(f2_str).map_err(|e| e.to_string())?;
                let res = match op {
                    "add" => f1.checked_add(&f2),
                    "sub" => f1.checked_sub(&f2),
                    "mul" => f1.checked_mul(&f2),
                    "div" => f1.checked_div(&f2),
                    _ => return Err(format!("unsupported fraction operation '{}'", op)),
                }
                .map_err(|e| e.to_string())?;
                Ok(json!({ "result": res.to_string(), "decimal": res.to_f64() }))
            }
            "decimal_op" => {
                let op = params.get("op").and_then(Value::as_str).unwrap_or("add");
                let d1_str = params
                    .get("d1")
                    .and_then(Value::as_str)
                    .ok_or("missing 'd1'")?;
                let d2_str = params
                    .get("d2")
                    .and_then(Value::as_str)
                    .ok_or("missing 'd2'")?;
                let d1 = BigDecimal::parse(d1_str).map_err(|e| e.to_string())?;
                let d2 = BigDecimal::parse(d2_str).map_err(|e| e.to_string())?;
                let prec = params
                    .get("precision")
                    .and_then(Value::as_u64)
                    .unwrap_or(12) as usize;
                let res = match op {
                    "add" => d1.add(&d2),
                    "sub" => d1.sub(&d2),
                    "mul" => d1.mul(&d2),
                    "div" => d1
                        .div_with_precision(&d2, prec)
                        .map_err(|e| e.to_string())?,
                    _ => return Err(format!("unsupported decimal operation '{}'", op)),
                };
                Ok(json!({ "result": res.to_string() }))
            }
            "percentage" => {
                let mode = params.get("mode").and_then(Value::as_str).unwrap_or("of");
                match mode {
                    "of" => {
                        let percent = params
                            .get("percent")
                            .and_then(Value::as_f64)
                            .ok_or("missing 'percent'")?;
                        let total = params
                            .get("total")
                            .and_then(Value::as_f64)
                            .ok_or("missing 'total'")?;
                        Ok(json!({ "result": Percentage::of(percent, total) }))
                    }
                    "change" => {
                        let old_val = params
                            .get("old")
                            .and_then(Value::as_f64)
                            .ok_or("missing 'old'")?;
                        let new_val = params
                            .get("new")
                            .and_then(Value::as_f64)
                            .ok_or("missing 'new'")?;
                        let chg =
                            Percentage::change(old_val, new_val).map_err(|e| e.to_string())?;
                        Ok(json!({ "percentage_change": chg }))
                    }
                    _ => Err(format!("unsupported percentage mode '{}'", mode)),
                }
            }
            "statistics" => {
                let data_arr = params
                    .get("data")
                    .and_then(Value::as_array)
                    .ok_or("missing 'data' array")?;
                let nums: Vec<f64> = data_arr.iter().filter_map(Value::as_f64).collect();
                let summary = Statistics::describe(&nums).map_err(|e| e.to_string())?;
                Ok(json!({
                    "count": summary.count,
                    "mean": summary.mean,
                    "median": summary.median,
                    "mode": summary.mode,
                    "variance_sample": summary.variance_sample,
                    "std_dev_sample": summary.std_dev_sample,
                    "min": summary.min,
                    "max": summary.max,
                    "range": summary.range,
                    "iqr": summary.iqr
                }))
            }
            "matrix_mult" => {
                let r1 = params
                    .get("r1")
                    .and_then(Value::as_u64)
                    .ok_or("missing 'r1'")? as usize;
                let c1 = params
                    .get("c1")
                    .and_then(Value::as_u64)
                    .ok_or("missing 'c1'")? as usize;
                let d1: Vec<f64> = params
                    .get("d1")
                    .and_then(Value::as_array)
                    .ok_or("missing 'd1'")?
                    .iter()
                    .filter_map(Value::as_f64)
                    .collect();

                let r2 = params
                    .get("r2")
                    .and_then(Value::as_u64)
                    .ok_or("missing 'r2'")? as usize;
                let c2 = params
                    .get("c2")
                    .and_then(Value::as_u64)
                    .ok_or("missing 'c2'")? as usize;
                let d2: Vec<f64> = params
                    .get("d2")
                    .and_then(Value::as_array)
                    .ok_or("missing 'd2'")?
                    .iter()
                    .filter_map(Value::as_f64)
                    .collect();

                let m1 = Matrix::new(r1, c1, d1).map_err(|e| format!("{:?}", e))?;
                let m2 = Matrix::new(r2, c2, d2).map_err(|e| format!("{:?}", e))?;
                let prod = m1.mul(&m2).map_err(|e| format!("{:?}", e))?;
                Ok(json!({ "rows": prod.rows, "cols": prod.cols, "data": prod.data }))
            }
            "solve_quadratic" => {
                let a = params
                    .get("a")
                    .and_then(Value::as_f64)
                    .ok_or("missing 'a'")?;
                let b = params
                    .get("b")
                    .and_then(Value::as_f64)
                    .ok_or("missing 'b'")?;
                let c = params
                    .get("c")
                    .and_then(Value::as_f64)
                    .ok_or("missing 'c'")?;
                let (r1, r2) =
                    EquationSolver::solve_quadratic(a, b, c).map_err(|e| format!("{:?}", e))?;
                Ok(json!({ "root1": r1.to_string(), "root2": r2.to_string() }))
            }
            "unit_convert" => {
                let val = params
                    .get("val")
                    .and_then(Value::as_f64)
                    .ok_or("missing 'val'")?;
                let from = params
                    .get("from")
                    .and_then(Value::as_str)
                    .ok_or("missing 'from'")?;
                let to = params
                    .get("to")
                    .and_then(Value::as_str)
                    .ok_or("missing 'to'")?;
                let dim_str = params
                    .get("dimension")
                    .and_then(Value::as_str)
                    .ok_or("missing 'dimension'")?;
                let dim = match dim_str.to_lowercase().as_str() {
                    "length" => Dimension::Length,
                    "mass" => Dimension::Mass,
                    "time" => Dimension::Time,
                    "temperature" => Dimension::Temperature,
                    "speed" => Dimension::Speed,
                    "force" => Dimension::Force,
                    "pressure" => Dimension::Pressure,
                    "energy" => Dimension::Energy,
                    "power" => Dimension::Power,
                    "data" => Dimension::Data,
                    other => return Err(format!("unknown dimension '{}'", other)),
                };
                let converted =
                    UnitConverter::convert(val, from, to, dim).map_err(|e| e.to_string())?;
                Ok(
                    json!({ "value": val, "from": from, "converted": converted, "to": to, "dimension": dim_str }),
                )
            }
            "multilingual_lookup" => {
                let term = params
                    .get("term")
                    .and_then(Value::as_str)
                    .ok_or("missing 'term'")?;
                if let Some(entry) = MultilingualEngine::lookup_english(term) {
                    return Ok(json!({
                        "english": entry.english,
                        "kannada": entry.kannada,
                        "transliteration": entry.kannada_transliteration,
                        "symbol": entry.symbol,
                        "category": entry.category,
                        "definition_en": entry.definition_en,
                        "definition_kn": entry.definition_kn,
                    }));
                }
                if let Some(entry) = MultilingualEngine::lookup_kannada(term) {
                    return Ok(json!({
                        "english": entry.english,
                        "kannada": entry.kannada,
                        "transliteration": entry.kannada_transliteration,
                        "symbol": entry.symbol,
                        "category": entry.category,
                        "definition_en": entry.definition_en,
                        "definition_kn": entry.definition_kn,
                    }));
                }
                Err(format!(
                    "term '{}' not found in multilingual dictionary",
                    term
                ))
            }
            _ => Err(format!(
                "unrecognized computation operation '{}'",
                operation
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bigint_arithmetic_and_primes() {
        let a = BigInt::parse("987654321987654321987654321").unwrap();
        let b = BigInt::parse("123456789123456789123456789").unwrap();
        let sum = a.checked_add(&b);
        assert_eq!(sum.to_string(), "1111111111111111111111111110");

        let diff = a.checked_sub(&b);
        assert_eq!(diff.to_string(), "864197532864197532864197532");

        let prod = a.checked_mul(&BigInt::from_i64(2));
        assert_eq!(prod.to_string(), "1975308643975308643975308642");

        let (q, r) = a.div_rem(&b).unwrap();
        assert_eq!(q.to_string(), "8");
        assert_eq!(r.to_string(), "9000000009000000009");

        // GCD and LCM
        let g = BigInt::from_i64(48).gcd(&BigInt::from_i64(18));
        assert_eq!(g.to_string(), "6");
        let l = BigInt::from_i64(12).lcm(&BigInt::from_i64(18)).unwrap();
        assert_eq!(l.to_string(), "36");

        // Modular exponentiation: 7^256 mod 13 = 9
        let m_pow = BigInt::from_i64(7)
            .mod_pow(&BigInt::from_i64(256), &BigInt::from_i64(13))
            .unwrap();
        assert_eq!(m_pow.to_string(), "9");

        // Primality test
        assert!(BigInt::from_i64(104729).is_prime(20));
        assert!(!BigInt::from_i64(104730).is_prime(20));

        // Factorial
        assert_eq!(BigInt::factorial(10).to_string(), "3628800");
    }

    #[test]
    fn test_fractions_and_decimals() {
        let f1 = Fraction::new(1, 3).unwrap();
        let f2 = Fraction::new(1, 6).unwrap();
        let sum = f1.checked_add(&f2).unwrap();
        assert_eq!(sum, Fraction::new(1, 2).unwrap());

        let bf1 = BigFraction::new(BigInt::from_i64(3), BigInt::from_i64(4)).unwrap();
        let bf2 = BigFraction::new(BigInt::from_i64(2), BigInt::from_i64(5)).unwrap();
        let bf_prod = bf1.mul(&bf2).unwrap();
        assert_eq!(
            bf_prod,
            BigFraction::new(BigInt::from_i64(3), BigInt::from_i64(10)).unwrap()
        );

        let d1 = BigDecimal::parse("1.25").unwrap();
        let d2 = BigDecimal::parse("2.50").unwrap();
        let d_sum = d1.add(&d2);
        assert!((d_sum.to_f64() - 3.75).abs() < 1e-4);

        let d4 = BigDecimal::parse("4.0").unwrap();
        let sqrt_d4 = d4.sqrt(6).unwrap();
        assert!((sqrt_d4.to_f64() - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_roots_and_logarithms() {
        assert_eq!(
            Roots::isqrt_bigint(&BigInt::from_i64(100)).unwrap(),
            BigInt::from_i64(10)
        );
        assert_eq!(
            Roots::isqrt_bigint(&BigInt::from_i64(99)).unwrap(),
            BigInt::from_i64(9)
        );
        assert_eq!(Roots::icbrt_i64(27), 3);
        assert_eq!(Roots::icbrt_i64(-27), -3);

        let ln_e = Logarithms::ln(std::f64::consts::E).unwrap();
        assert!((ln_e - 1.0).abs() < 1e-6);
        assert!((Logarithms::log10(1000.0).unwrap() - 3.0).abs() < 1e-6);
        assert!((Logarithms::log2(64.0).unwrap() - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_linear_algebra() {
        let v1 = Vector::new(vec![1.0, 2.0, 3.0]);
        let v2 = Vector::new(vec![4.0, 5.0, 6.0]);
        assert_eq!(v1.dot(&v2).unwrap(), 32.0);

        let cross = v1.cross(&v2).unwrap();
        assert_eq!(cross.data, vec![-3.0, 6.0, -3.0]);

        // Determinant of 2x2
        let m = Matrix::new(2, 2, vec![2.0, 3.0, 1.0, 4.0]).unwrap();
        let det = m.determinant().unwrap();
        assert!((det - 5.0).abs() < 1e-7);

        // Inverse of 2x2
        let inv = m.inverse().unwrap();
        let prod = m.mul(&inv).unwrap();
        assert!((prod.get(0, 0) - 1.0).abs() < 1e-6);
        assert!(prod.get(0, 1).abs() < 1e-6);
        assert!(prod.get(1, 0).abs() < 1e-6);
        assert!((prod.get(1, 1) - 1.0).abs() < 1e-6);

        // Solve Ax = b
        let b = Vector::new(vec![13.0, 14.0]);
        let x = m.solve(&b).unwrap();
        // 2(2) + 3(3) = 4 + 9 = 13
        // 1(2) + 4(3) = 2 + 12 = 14
        assert!((x.data[0] - 2.0).abs() < 1e-6);
        assert!((x.data[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_statistics_and_probability() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let stats = Statistics::describe(&data).unwrap();
        assert_eq!(stats.count, 8);
        assert!((stats.mean - 5.0).abs() < 1e-7);
        assert_eq!(stats.mode, Some(4.0));

        // Permutations and Combinations
        assert_eq!(
            Probability::permutations(5, 2).unwrap(),
            BigInt::from_i64(20)
        );
        assert_eq!(
            Probability::combinations(5, 2).unwrap(),
            BigInt::from_i64(10)
        );

        // Binomial PMF: 4 heads in 4 fair flips = (1/2)^4 = 0.0625
        let b_pmf = Probability::binomial_pmf(4, 4, 0.5).unwrap();
        assert!((b_pmf - 0.0625).abs() < 1e-6);

        // Bayes update: Prior 0.01, Sensitivity 0.95, False Positive 0.10
        // P(D|+) = 0.01*0.95 / (0.01*0.95 + 0.99*0.10) = 0.0095 / (0.0095 + 0.099) = 0.0095 / 0.1085 ~= 0.087557
        let post = Probability::bayes_partition(&[0.01, 0.99], &[0.95, 0.10], 0).unwrap();
        assert!((post - 0.087557).abs() < 1e-4);
    }

    #[test]
    fn test_polynomial_and_equations() {
        // p(x) = 2x^2 + 3x + 1
        let p = Polynomial::new(vec![1.0, 3.0, 2.0]);
        assert_eq!(p.evaluate(2.0), 15.0);

        let deriv = p.derivative();
        // d/dx = 3 + 4x
        assert_eq!(deriv.evaluate(2.0), 11.0);

        // Solve quadratic: x^2 - 5x + 6 = 0 -> roots 3 and 2
        let (r1, r2) = EquationSolver::solve_quadratic(1.0, -5.0, 6.0).unwrap();
        match (r1, r2) {
            (ComplexRoot::Real(val1), ComplexRoot::Real(val2)) => {
                assert!((val1 - 3.0).abs() < 1e-6 || (val1 - 2.0).abs() < 1e-6);
                assert!((val2 - 3.0).abs() < 1e-6 || (val2 - 2.0).abs() < 1e-6);
            }
            _ => panic!("expected real roots"),
        }

        // Complex roots: x^2 + 1 = 0 -> +- i
        let (c1, c2) = EquationSolver::solve_quadratic(1.0, 0.0, 1.0).unwrap();
        match (c1, c2) {
            (
                ComplexRoot::Complex { real: r1, imag: i1 },
                ComplexRoot::Complex { real: r2, imag: i2 },
            ) => {
                assert_eq!(r1, 0.0);
                assert_eq!(r2, 0.0);
                assert!((i1.abs() - 1.0).abs() < 1e-6);
                assert!((i2.abs() - 1.0).abs() < 1e-6);
            }
            _ => panic!("expected complex roots"),
        }
    }

    #[test]
    fn test_numerical_methods() {
        // Root of x^2 - 2 = 0 in [1, 2]
        let root = NumericalMethods::bisection(|x| x * x - 2.0, 1.0, 2.0, 1e-6, 100).unwrap();
        assert!((root - std::f64::consts::SQRT_2).abs() < 1e-5);

        // Simpson's 1/3 of x^2 from 0 to 3 = 9.0
        let integral = NumericalMethods::simpsons_one_third(|x| x * x, 0.0, 3.0, 100).unwrap();
        assert!((integral - 9.0).abs() < 1e-5);

        // RK4: dy/dt = y, y(0)=1 -> y(1) = e
        let rk = NumericalMethods::rk4_integrate(|_t, y| y, 0.0, 1.0, 1.0, 100).unwrap();
        assert!((rk.last().unwrap().1 - std::f64::consts::E).abs() < 1e-4);
    }

    #[test]
    fn test_unit_converter() {
        let km_to_m = UnitConverter::convert(5.0, "kilometer", "meter", Dimension::Length).unwrap();
        assert_eq!(km_to_m, 5000.0);

        let kg_to_g = UnitConverter::convert(2.5, "kilogram", "gram", Dimension::Mass).unwrap();
        assert_eq!(kg_to_g, 2500.0);

        let c_to_f =
            UnitConverter::convert(100.0, "celsius", "fahrenheit", Dimension::Temperature).unwrap();
        assert!((c_to_f - 212.0).abs() < 1e-4);

        let kwh_to_j =
            UnitConverter::convert(1.0, "kilowatt_hour", "joule", Dimension::Energy).unwrap();
        assert_eq!(kwh_to_j, 3_600_000.0);
    }

    #[test]
    fn test_multilingual_terminology() {
        let term = MultilingualEngine::lookup_kannada("ಗುಣಾಕಾರ").unwrap();
        assert_eq!(term.english, "Multiplication");

        let term_en = MultilingualEngine::lookup_english("acceleration").unwrap();
        assert_eq!(term_en.kannada, "ವೇಗೋತ್ಕರ್ಷ");
    }

    #[test]
    fn test_computation_engine_dispatcher() {
        let res =
            ComputationEngine::execute("bigint_add", &json!({ "a": "100", "b": "200" })).unwrap();
        assert_eq!(res["result"], "300");

        let gcd_res = ComputationEngine::execute("gcd", &json!({ "a": "54", "b": "24" })).unwrap();
        assert_eq!(gcd_res["gcd"], "6");

        let unit_res = ComputationEngine::execute(
            "unit_convert",
            &json!({
                "val": 10.0, "from": "meter", "to": "centimeter", "dimension": "length"
            }),
        )
        .unwrap();
        assert_eq!(unit_res["converted"], 1000.0);

        let multi_res =
            ComputationEngine::execute("multilingual_lookup", &json!({ "term": "acceleration" }))
                .unwrap();
        assert_eq!(multi_res["kannada"], "ವೇಗೋತ್ಕರ್ಷ");
    }
}
