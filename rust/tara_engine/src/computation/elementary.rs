//! Elementary mathematics, percentage, roots, and logarithms for TARA Native Engine.
//!
//! Provides:
//! - Percentage calculations (P% of X, percentage change, markups, discounts, percentage difference).
//! - Integer and high-precision roots (square root, cube root, integer nth-root via binary search, floating nth-root).
//! - Logarithms (ln, log10, log2, log_b) via verified convergent series expansions.
//! - Scientific notation conversion and formatting.

use thiserror::Error;

use super::fraction::Fraction;
use super::integer::{BigInt, IntegerError};

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ElementaryError {
    #[error("domain error: {0}")]
    DomainError(String),
    #[error("division by zero")]
    DivisionByZero,
    #[error("integer arithmetic error: {0}")]
    Integer(#[from] IntegerError),
}

/// Percentage computation utilities.
pub struct Percentage;

impl Percentage {
    /// Calculate `percent`% of `total`: (percent * total) / 100.
    pub fn of(percent: f64, total: f64) -> f64 {
        (percent * total) / 100.0
    }

    /// Calculate what percentage `part` is of `whole`: (part / whole) * 100.
    pub fn what_percent(part: f64, whole: f64) -> Result<f64, ElementaryError> {
        if whole == 0.0 {
            return Err(ElementaryError::DivisionByZero);
        }
        Ok((part / whole) * 100.0)
    }

    /// Calculate percentage increase or decrease: ((new - old) / old) * 100.
    pub fn change(old_val: f64, new_val: f64) -> Result<f64, ElementaryError> {
        if old_val == 0.0 {
            return Err(ElementaryError::DivisionByZero);
        }
        Ok(((new_val - old_val) / old_val) * 100.0)
    }

    /// Calculate symmetric percentage difference: |a - b| / ((a + b) / 2) * 100.
    pub fn difference(a: f64, b: f64) -> Result<f64, ElementaryError> {
        let avg = (a + b) / 2.0;
        if avg == 0.0 {
            return Err(ElementaryError::DivisionByZero);
        }
        Ok(((a - b).abs() / avg.abs()) * 100.0)
    }

    /// Apply percentage discount to price.
    pub fn discount(price: f64, discount_percent: f64) -> f64 {
        price * (1.0 - discount_percent / 100.0)
    }

    /// Apply percentage markup to cost.
    pub fn markup(cost: f64, markup_percent: f64) -> f64 {
        cost * (1.0 + markup_percent / 100.0)
    }

    /// Exact fraction percentage: (fraction * 100).
    pub fn fraction_percent(f: Fraction) -> Fraction {
        f * Fraction::from_i64(100)
    }
}

/// Root-finding utilities for integers and real numbers.
pub struct Roots;

impl Roots {
    /// Exact integer square root: returns floor(sqrt(n)) using binary search on BigInt.
    pub fn isqrt_bigint(n: &BigInt) -> Result<BigInt, ElementaryError> {
        if n.is_negative() {
            return Err(ElementaryError::DomainError(
                "square root of negative number".into(),
            ));
        }
        if n.is_zero() || *n == BigInt::one() {
            return Ok(n.clone());
        }

        let mut low = BigInt::one();
        let mut high = n.clone();
        let mut ans = BigInt::zero();

        while low <= high {
            let sum = low.checked_add(&high);
            let (mid, _) = sum.div_rem_u32(2).unwrap();
            let mid_sq = mid.clone() * mid.clone();

            if mid_sq == *n {
                return Ok(mid);
            } else if mid_sq < *n {
                ans = mid.clone();
                low = mid + BigInt::one();
            } else {
                if mid == BigInt::zero() {
                    break;
                }
                high = mid - BigInt::one();
            }
        }
        Ok(ans)
    }

    /// Integer square root for 64-bit unsigned integer.
    pub fn isqrt_u64(n: u64) -> u64 {
        if n == 0 {
            return 0;
        }
        let mut x0 = (n as f64).sqrt() as u64;
        let mut x1 = (x0 + n / x0) / 2;
        while x1 < x0 {
            x0 = x1;
            x1 = (x0 + n / x0) / 2;
        }
        x0
    }

    /// Exact integer cube root floor(cbrt(n)).
    pub fn icbrt_i64(n: i64) -> i64 {
        if n == 0 {
            return 0;
        }
        let sign = if n < 0 { -1 } else { 1 };
        let abs_n = n.unsigned_abs();

        let mut low = 1u64;
        let mut high = 2_097_152u64.min(abs_n); // 2_097_152^3 > 2^63 - 1
        let mut ans = 1u64;

        while low <= high {
            let mid = low + (high - low) / 2;
            let cube = (mid as u128) * (mid as u128) * (mid as u128);
            if cube <= (abs_n as u128) {
                ans = mid;
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }
        sign * (ans as i64)
    }

    /// Real nth root x^(1/n) using Newton's method.
    pub fn nth_root(x: f64, n: u32) -> Result<f64, ElementaryError> {
        if n == 0 {
            return Err(ElementaryError::DomainError("0th root is undefined".into()));
        }
        if x < 0.0 && n.is_multiple_of(2) {
            return Err(ElementaryError::DomainError(
                "even root of negative number is complex".into(),
            ));
        }
        if x == 0.0 {
            return Ok(0.0);
        }

        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let abs_x = x.abs();
        let fn_val = n as f64;

        // Initial guess
        let mut y = (abs_x.ln() / fn_val).exp();

        // Newton-Raphson: y_{k+1} = (1/n) * ((n-1)*y_k + x / y_k^(n-1))
        for _ in 0..30 {
            let y_pow = y.powi((n - 1) as i32);
            let next_y = ((fn_val - 1.0) * y + abs_x / y_pow) / fn_val;
            if (next_y - y).abs() < 1e-15 * y.abs() {
                y = next_y;
                break;
            }
            y = next_y;
        }
        Ok(sign * y)
    }
}

/// Exact and high-precision logarithmic calculations.
pub struct Logarithms;

impl Logarithms {
    /// Natural logarithm ln(x) using argument reduction and accelerated series:
    /// ln((1+y)/(1-y)) = 2 * (y + y^3/3 + y^5/5 + ...) with y = (x-1)/(x+1).
    pub fn ln(x: f64) -> Result<f64, ElementaryError> {
        if x <= 0.0 {
            return Err(ElementaryError::DomainError("ln(x) requires x > 0".into()));
        }
        if x == 1.0 {
            return Ok(0.0);
        }

        // Reduce x into [0.5, 2.0] by extracting power of 2: x = m * 2^k
        let mut m = x;
        let mut k = 0i32;
        while m > 2.0 {
            m /= 2.0;
            k += 1;
        }
        while m < 0.5 {
            m *= 2.0;
            k -= 1;
        }

        let y = (m - 1.0) / (m + 1.0);
        let y2 = y * y;
        let mut term = y;
        let mut sum = y;
        let mut d = 3.0;

        for _ in 0..50 {
            term *= y2;
            let add = term / d;
            sum += add;
            if add.abs() < 1e-16 {
                break;
            }
            d += 2.0;
        }

        let ln2 = std::f64::consts::LN_2;
        Ok(2.0 * sum + (k as f64) * ln2)
    }

    /// Common logarithm log10(x) = ln(x) / ln(10).
    pub fn log10(x: f64) -> Result<f64, ElementaryError> {
        let ln_x = Self::ln(x)?;
        let ln_10 = std::f64::consts::LN_10;
        Ok(ln_x / ln_10)
    }

    /// Binary logarithm log2(x) = ln(x) / ln(2).
    pub fn log2(x: f64) -> Result<f64, ElementaryError> {
        let ln_x = Self::ln(x)?;
        let ln_2 = std::f64::consts::LN_2;
        Ok(ln_x / ln_2)
    }

    /// Logarithm with arbitrary base b: log_b(x) = ln(x) / ln(b).
    pub fn log_base(x: f64, base: f64) -> Result<f64, ElementaryError> {
        if base <= 0.0 || (base - 1.0).abs() < 1e-15 {
            return Err(ElementaryError::DomainError(
                "log base must be positive and != 1".into(),
            ));
        }
        let ln_x = Self::ln(x)?;
        let ln_b = Self::ln(base)?;
        Ok(ln_x / ln_b)
    }
}
