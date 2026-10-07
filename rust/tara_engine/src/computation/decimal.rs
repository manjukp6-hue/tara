//! Exact high-precision decimal arithmetic for TARA Native Engine.
//!
//! Provides scaled arbitrary-precision decimal operations (`BigDecimal`) without
//! IEEE-754 binary floating-point representation drift or round-off errors.

use std::cmp::Ordering;
use std::fmt;
use thiserror::Error;

use super::integer::{BigInt, IntegerError};

#[derive(Debug, Error, PartialEq, Eq)]
pub enum DecimalError {
    #[error("division by zero")]
    DivisionByZero,
    #[error("decimal parse error: {0}")]
    ParseError(String),
    #[error("negative square root operand: {0}")]
    NegativeSquareRoot(String),
    #[error("integer arithmetic error: {0}")]
    Integer(#[from] IntegerError),
}

/// Arbitrary-precision decimal represented as `mantissa * 10^(-scale)`.
#[derive(Clone, PartialEq, Eq)]
pub struct BigDecimal {
    pub mantissa: BigInt,
    pub scale: usize,
}

impl BigDecimal {
    pub fn zero() -> Self {
        Self {
            mantissa: BigInt::zero(),
            scale: 0,
        }
    }

    pub fn one() -> Self {
        Self {
            mantissa: BigInt::one(),
            scale: 0,
        }
    }

    pub fn from_i64(val: i64) -> Self {
        Self {
            mantissa: BigInt::from_i64(val),
            scale: 0,
        }
    }

    pub fn new(mantissa: BigInt, scale: usize) -> Self {
        let mut d = Self { mantissa, scale };
        d.normalize();
        d
    }

    pub fn parse(s: &str) -> Result<Self, DecimalError> {
        let trimmed = s.trim();
        if trimmed.is_empty() {
            return Err(DecimalError::ParseError("empty string".into()));
        }

        // Handle scientific notation (e.g. 1.23e-4 or 5E+3)
        if let Some(e_idx) = trimmed.find(['e', 'E']) {
            let base_str = &trimmed[..e_idx];
            let exp_str = &trimmed[e_idx + 1..];
            let exp = exp_str
                .parse::<i32>()
                .map_err(|e| DecimalError::ParseError(format!("invalid exponent: {}", e)))?;
            let base_decimal = Self::parse_plain(base_str)?;
            if exp >= 0 {
                let factor = BigInt::from_i64(10).checked_pow(exp as u32);
                return Ok(Self::new(
                    base_decimal.mantissa * factor,
                    base_decimal.scale,
                ));
            } else {
                let add_scale = (-exp) as usize;
                return Ok(Self::new(
                    base_decimal.mantissa,
                    base_decimal.scale + add_scale,
                ));
            }
        }

        Self::parse_plain(trimmed)
    }

    fn parse_plain(trimmed: &str) -> Result<Self, DecimalError> {
        if let Some(dot_idx) = trimmed.find('.') {
            let int_part = &trimmed[..dot_idx];
            let frac_part = &trimmed[dot_idx + 1..];

            let sign = if int_part.starts_with('-') { "-" } else { "" };
            let clean_int = int_part.trim_start_matches('-').trim_start_matches('+');
            let combined = format!("{}{}{}", sign, clean_int, frac_part);
            let mantissa =
                BigInt::parse(&combined).map_err(|e| DecimalError::ParseError(e.to_string()))?;
            let scale = frac_part.len();
            Ok(Self::new(mantissa, scale))
        } else {
            let mantissa =
                BigInt::parse(trimmed).map_err(|e| DecimalError::ParseError(e.to_string()))?;
            Ok(Self { mantissa, scale: 0 })
        }
    }

    pub fn is_zero(&self) -> bool {
        self.mantissa.is_zero()
    }

    pub fn is_negative(&self) -> bool {
        self.mantissa.is_negative()
    }

    pub fn abs(&self) -> Self {
        Self {
            mantissa: self.mantissa.abs(),
            scale: self.scale,
        }
    }

    /// Strip redundant trailing fractional zeroes.
    pub fn normalize(&mut self) {
        if self.mantissa.is_zero() {
            self.scale = 0;
            return;
        }
        while self.scale > 0 {
            let (q, r) = self.mantissa.div_rem_u32(10).unwrap();
            if r == 0 {
                self.mantissa = q;
                self.scale -= 1;
            } else {
                break;
            }
        }
    }

    /// Match scales between two decimals by scaling up the smaller scale.
    fn match_scales(&self, other: &Self) -> (BigInt, BigInt, usize) {
        let max_scale = self.scale.max(other.scale);
        let m1 = if self.scale < max_scale {
            let diff = (max_scale - self.scale) as u32;
            self.mantissa.clone() * BigInt::from_i64(10).checked_pow(diff)
        } else {
            self.mantissa.clone()
        };
        let m2 = if other.scale < max_scale {
            let diff = (max_scale - other.scale) as u32;
            other.mantissa.clone() * BigInt::from_i64(10).checked_pow(diff)
        } else {
            other.mantissa.clone()
        };
        (m1, m2, max_scale)
    }

    pub fn add(&self, other: &Self) -> Self {
        let (m1, m2, scale) = self.match_scales(other);
        Self::new(m1 + m2, scale)
    }

    pub fn sub(&self, other: &Self) -> Self {
        let (m1, m2, scale) = self.match_scales(other);
        Self::new(m1 - m2, scale)
    }

    pub fn mul(&self, other: &Self) -> Self {
        let mantissa = self.mantissa.clone() * other.mantissa.clone();
        let scale = self.scale + other.scale;
        Self::new(mantissa, scale)
    }

    /// Divide with target fractional digit precision (e.g. 10 or 20 digits).
    pub fn div_with_precision(
        &self,
        other: &Self,
        target_precision: usize,
    ) -> Result<Self, DecimalError> {
        if other.is_zero() {
            return Err(DecimalError::DivisionByZero);
        }
        if self.is_zero() {
            return Ok(Self::zero());
        }

        // We want (self.m / 10^self.s) / (other.m / 10^other.s)
        // = (self.m * 10^(other.s + target_precision)) / other.m * 10^(-(self.s + target_precision))
        let extra_zeros = target_precision + other.scale;
        let scaled_numerator =
            self.mantissa.clone() * BigInt::from_i64(10).checked_pow(extra_zeros as u32);
        let (quotient, _) = scaled_numerator.div_rem(&other.mantissa)?;

        let final_scale = self.scale + target_precision;
        Ok(Self::new(quotient, final_scale))
    }

    pub fn div(&self, other: &Self) -> Result<Self, DecimalError> {
        self.div_with_precision(other, 12)
    }

    /// High-precision square root using Newton-Raphson method.
    pub fn sqrt(&self, precision: usize) -> Result<Self, DecimalError> {
        if self.is_negative() {
            return Err(DecimalError::NegativeSquareRoot(self.to_string()));
        }
        if self.is_zero() {
            return Ok(Self::zero());
        }

        // Initial estimate via floating point
        let approx_f64 = self.to_f64().sqrt();
        let mut x = Self::parse(&format!("{:.8}", approx_f64))?;
        let half = Self::parse("0.5")?;

        // 10 Newton iterations are more than sufficient for 10-20 decimals
        for _ in 0..15 {
            let div_val = self.div_with_precision(&x, precision + 4)?;
            x = (x.add(&div_val)).mul(&half);
        }

        // Truncate / round to requested precision
        x.round(precision)
    }

    pub fn round(&self, precision: usize) -> Result<Self, DecimalError> {
        if self.scale <= precision {
            return Ok(self.clone());
        }
        let diff = (self.scale - precision) as u32;
        let divisor = BigInt::from_i64(10).checked_pow(diff);
        let half_div = divisor.div_rem_u32(2)?.0;

        let is_neg = self.mantissa.is_negative();
        let abs_mantissa = self.mantissa.abs();
        let (q, r) = abs_mantissa.div_rem(&divisor)?;

        let rounded_q = if r.cmp_abs(&half_div) != Ordering::Less {
            q + BigInt::one()
        } else {
            q
        };

        let signed_m = if is_neg { -rounded_q } else { rounded_q };
        Ok(Self::new(signed_m, precision))
    }

    pub fn to_f64(&self) -> f64 {
        let s = self.to_string();
        s.parse::<f64>().unwrap_or(0.0)
    }
}

impl fmt::Display for BigDecimal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.mantissa.is_zero() {
            return write!(f, "0");
        }
        let is_neg = self.mantissa.is_negative();
        let abs_m = self.mantissa.abs().to_string();

        if is_neg {
            write!(f, "-")?;
        }

        if self.scale == 0 {
            write!(f, "{}", abs_m)
        } else if abs_m.len() <= self.scale {
            let zeros_needed = self.scale - abs_m.len();
            write!(f, "0.")?;
            for _ in 0..zeros_needed {
                write!(f, "0")?;
            }
            write!(f, "{}", abs_m)
        } else {
            let split = abs_m.len() - self.scale;
            write!(f, "{}.{}", &abs_m[..split], &abs_m[split..])
        }
    }
}

impl fmt::Debug for BigDecimal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BigDecimal({})", self)
    }
}

impl PartialOrd for BigDecimal {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BigDecimal {
    fn cmp(&self, other: &Self) -> Ordering {
        let (m1, m2, _) = self.match_scales(other);
        m1.cmp(&m2)
    }
}
