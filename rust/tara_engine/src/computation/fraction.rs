//! Exact rational number (fraction) arithmetic for TARA Native Engine.
//!
//! Provides exact fraction operations with automated simplification via GCD,
//! supporting both 64-bit fast fractions and arbitrary-precision BigFraction.

use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};
use thiserror::Error;

use super::integer::{checked_gcd_i64, BigInt, IntegerError};

#[derive(Debug, Error, PartialEq, Eq)]
pub enum FractionError {
    #[error("division by zero: denominator cannot be zero")]
    DenominatorZero,
    #[error("fraction parse error: {0}")]
    ParseError(String),
    #[error("integer arithmetic error: {0}")]
    Integer(#[from] IntegerError),
}

/// Exact fraction with 64-bit integer numerator and denominator.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Fraction {
    pub num: i64,
    pub den: i64,
}

impl Fraction {
    /// Create a simplified fraction num / den.
    pub fn new(mut num: i64, mut den: i64) -> Result<Self, FractionError> {
        if den == 0 {
            return Err(FractionError::DenominatorZero);
        }
        if den < 0 {
            num = -num;
            den = -den;
        }
        if num == 0 {
            return Ok(Self { num: 0, den: 1 });
        }
        let g = checked_gcd_i64(num, den);
        Ok(Self {
            num: num / g,
            den: den / g,
        })
    }

    pub fn from_i64(n: i64) -> Self {
        Self { num: n, den: 1 }
    }

    pub fn parse(s: &str) -> Result<Self, FractionError> {
        let trimmed = s.trim();
        if let Some(slash_idx) = trimmed.find('/') {
            let num_str = &trimmed[..slash_idx].trim();
            let den_str = &trimmed[slash_idx + 1..].trim();
            let num = num_str
                .parse::<i64>()
                .map_err(|e| FractionError::ParseError(e.to_string()))?;
            let den = den_str
                .parse::<i64>()
                .map_err(|e| FractionError::ParseError(e.to_string()))?;
            Self::new(num, den)
        } else {
            let n = trimmed
                .parse::<i64>()
                .map_err(|e| FractionError::ParseError(e.to_string()))?;
            Ok(Self::from_i64(n))
        }
    }

    pub fn is_zero(&self) -> bool {
        self.num == 0
    }

    pub fn is_integer(&self) -> bool {
        self.den == 1
    }

    pub fn reciprocal(&self) -> Result<Self, FractionError> {
        if self.num == 0 {
            return Err(FractionError::DenominatorZero);
        }
        Self::new(self.den, self.num)
    }

    pub fn to_f64(&self) -> f64 {
        self.num as f64 / self.den as f64
    }

    pub fn checked_add(&self, other: &Self) -> Result<Self, FractionError> {
        // a/b + c/d = (a*d + b*c) / (b*d)
        let g = checked_gcd_i64(self.den, other.den);
        let b_div_g = self.den / g;
        let d_div_g = other.den / g;

        let num = self
            .num
            .checked_mul(d_div_g)
            .and_then(|part1| {
                other
                    .num
                    .checked_mul(b_div_g)
                    .and_then(|part2| part1.checked_add(part2))
            })
            .ok_or_else(|| IntegerError::Overflow {
                operation: format!("{} + {}", self, other),
            })?;

        let den = self
            .den
            .checked_mul(d_div_g)
            .ok_or_else(|| IntegerError::Overflow {
                operation: format!("{} + {}", self, other),
            })?;

        Self::new(num, den)
    }

    pub fn checked_sub(&self, other: &Self) -> Result<Self, FractionError> {
        self.checked_add(&(-*other))
    }

    pub fn checked_mul(&self, other: &Self) -> Result<Self, FractionError> {
        // Simplify cross-terms to avoid intermediate overflow
        let g1 = checked_gcd_i64(self.num, other.den);
        let g2 = checked_gcd_i64(other.num, self.den);

        let num1 = self.num / g1;
        let den2 = other.den / g1;
        let num2 = other.num / g2;
        let den1 = self.den / g2;

        let num = num1
            .checked_mul(num2)
            .ok_or_else(|| IntegerError::Overflow {
                operation: format!("{} * {}", self, other),
            })?;
        let den = den1
            .checked_mul(den2)
            .ok_or_else(|| IntegerError::Overflow {
                operation: format!("{} * {}", self, other),
            })?;

        Self::new(num, den)
    }

    pub fn checked_div(&self, other: &Self) -> Result<Self, FractionError> {
        if other.num == 0 {
            return Err(FractionError::DenominatorZero);
        }
        self.checked_mul(&other.reciprocal()?)
    }

    pub fn pow(&self, exp: i32) -> Result<Self, FractionError> {
        if exp == 0 {
            return Ok(Self::from_i64(1));
        }
        if exp < 0 {
            return self.reciprocal()?.pow(-exp);
        }
        let uexp = exp as u32;
        let num = self
            .num
            .checked_pow(uexp)
            .ok_or_else(|| IntegerError::Overflow {
                operation: format!("{}^{}", self.num, exp),
            })?;
        let den = self
            .den
            .checked_pow(uexp)
            .ok_or_else(|| IntegerError::Overflow {
                operation: format!("{}^{}", self.den, exp),
            })?;
        Self::new(num, den)
    }
}

impl fmt::Display for Fraction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.den == 1 {
            write!(f, "{}", self.num)
        } else {
            write!(f, "{}/{}", self.num, self.den)
        }
    }
}

impl fmt::Debug for Fraction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Fraction({})", self)
    }
}

impl PartialOrd for Fraction {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Fraction {
    fn cmp(&self, other: &Self) -> Ordering {
        let left = self.num as i128 * other.den as i128;
        let right = other.num as i128 * self.den as i128;
        left.cmp(&right)
    }
}

impl Neg for Fraction {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self {
            num: -self.num,
            den: self.den,
        }
    }
}

impl Add for Fraction {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        self.checked_add(&rhs).expect("fraction addition overflow")
    }
}

impl Sub for Fraction {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        self.checked_sub(&rhs)
            .expect("fraction subtraction overflow")
    }
}

impl Mul for Fraction {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        self.checked_mul(&rhs)
            .expect("fraction multiplication overflow")
    }
}

impl Div for Fraction {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        self.checked_div(&rhs).expect("fraction division error")
    }
}

/// Arbitrary-precision rational number using BigInt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BigFraction {
    pub num: BigInt,
    pub den: BigInt,
}

impl BigFraction {
    pub fn new(mut num: BigInt, mut den: BigInt) -> Result<Self, FractionError> {
        if den.is_zero() {
            return Err(FractionError::DenominatorZero);
        }
        if den.is_negative() {
            num = -num;
            den = -den;
        }
        if num.is_zero() {
            return Ok(Self {
                num: BigInt::zero(),
                den: BigInt::one(),
            });
        }
        let g = num.gcd(&den);
        let simplified_num = (num.div_rem(&g)?).0;
        let simplified_den = (den.div_rem(&g)?).0;
        Ok(Self {
            num: simplified_num,
            den: simplified_den,
        })
    }

    pub fn from_i64(n: i64) -> Self {
        Self {
            num: BigInt::from_i64(n),
            den: BigInt::one(),
        }
    }

    pub fn add(&self, other: &Self) -> Result<Self, FractionError> {
        let num1 = self.num.checked_mul(&other.den);
        let num2 = other.num.checked_mul(&self.den);
        let num = num1.checked_add(&num2);
        let den = self.den.checked_mul(&other.den);
        Self::new(num, den)
    }

    pub fn sub(&self, other: &Self) -> Result<Self, FractionError> {
        self.add(&Self {
            num: -other.num.clone(),
            den: other.den.clone(),
        })
    }

    pub fn mul(&self, other: &Self) -> Result<Self, FractionError> {
        let num = self.num.checked_mul(&other.num);
        let den = self.den.checked_mul(&other.den);
        Self::new(num, den)
    }

    pub fn div(&self, other: &Self) -> Result<Self, FractionError> {
        if other.num.is_zero() {
            return Err(FractionError::DenominatorZero);
        }
        let num = self.num.checked_mul(&other.den);
        let den = self.den.checked_mul(&other.num);
        Self::new(num, den)
    }
}

impl fmt::Display for BigFraction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.den == BigInt::one() {
            write!(f, "{}", self.num)
        } else {
            write!(f, "{}/{}", self.num, self.den)
        }
    }
}
