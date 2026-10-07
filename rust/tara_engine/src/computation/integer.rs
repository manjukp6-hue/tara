//! Exact integer and arbitrary-precision integer arithmetic for TARA Native Engine.
//!
//! Provides:
//! - Arbitrary-precision integer (`BigInt`) supporting arbitrarily large integers without overflow.
//! - Basic operations: addition, subtraction, multiplication, division, modulo, power, negation.
//! - Number-theoretic functions: GCD, LCM, modular exponentiation, modular inverse, factorial, Miller-Rabin primality test.
//! - Exact 64/128-bit checked arithmetic helpers with overflow detection.

use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum IntegerError {
    #[error("division by zero")]
    DivisionByZero,
    #[error("negative exponent for integer power")]
    NegativeExponent,
    #[error("modular inverse does not exist: gcd({a}, {m}) != 1")]
    NoModularInverse { a: String, m: String },
    #[error("integer parse error: {0}")]
    ParseError(String),
    #[error("integer overflow detected in {operation}")]
    Overflow { operation: String },
}

/// Sign of a BigInt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sign {
    Positive,
    Zero,
    Negative,
}

/// Arbitrary-precision integer represented in base 10^9 (each digit in [0, 999_999_999]).
#[derive(Clone, PartialEq, Eq)]
pub struct BigInt {
    pub sign: Sign,
    /// Digits stored little-endian (least significant chunk at index 0).
    pub digits: Vec<u32>,
}

const BASE: u64 = 1_000_000_000;
const BASE_DIGITS: usize = 9;

impl BigInt {
    pub fn zero() -> Self {
        Self {
            sign: Sign::Zero,
            digits: Vec::new(),
        }
    }

    pub fn one() -> Self {
        Self {
            sign: Sign::Positive,
            digits: vec![1],
        }
    }

    pub fn from_i64(mut val: i64) -> Self {
        if val == 0 {
            return Self::zero();
        }
        let sign = if val < 0 {
            val = val.wrapping_neg();
            Sign::Negative
        } else {
            Sign::Positive
        };
        let mut uval = val as u64;
        let mut digits = Vec::new();
        while uval > 0 {
            digits.push((uval % BASE) as u32);
            uval /= BASE;
        }
        Self { sign, digits }
    }

    pub fn from_i128(mut val: i128) -> Self {
        if val == 0 {
            return Self::zero();
        }
        let sign = if val < 0 {
            val = val.wrapping_neg();
            Sign::Negative
        } else {
            Sign::Positive
        };
        let mut uval = val as u128;
        let mut digits = Vec::new();
        let base128 = BASE as u128;
        while uval > 0 {
            digits.push((uval % base128) as u32);
            uval /= base128;
        }
        Self { sign, digits }
    }

    pub fn parse(s: &str) -> Result<Self, IntegerError> {
        let trimmed = s.trim();
        if trimmed.is_empty() {
            return Err(IntegerError::ParseError("empty input string".into()));
        }

        let (sign, digits_str) = if let Some(stripped) = trimmed.strip_prefix('-') {
            (Sign::Negative, stripped)
        } else if let Some(stripped) = trimmed.strip_prefix('+') {
            (Sign::Positive, stripped)
        } else {
            (Sign::Positive, trimmed)
        };

        let digits_str = digits_str.trim_start_matches('0');
        if digits_str.is_empty() {
            return Ok(Self::zero());
        }

        for ch in digits_str.chars() {
            if !ch.is_ascii_digit() {
                return Err(IntegerError::ParseError(format!(
                    "invalid character: {}",
                    ch
                )));
            }
        }

        let mut chunks = Vec::new();
        let len = digits_str.len();
        let mut end = len;
        while end > 0 {
            let start = end.saturating_sub(BASE_DIGITS);
            let chunk = digits_str[start..end]
                .parse::<u32>()
                .map_err(|e| IntegerError::ParseError(e.to_string()))?;
            chunks.push(chunk);
            end = start;
        }

        let mut res = Self {
            sign,
            digits: chunks,
        };
        res.normalize();
        Ok(res)
    }

    pub fn is_zero(&self) -> bool {
        self.sign == Sign::Zero || self.digits.is_empty()
    }

    pub fn is_negative(&self) -> bool {
        self.sign == Sign::Negative
    }

    pub fn is_positive(&self) -> bool {
        self.sign == Sign::Positive
    }

    pub fn is_even(&self) -> bool {
        if self.is_zero() {
            true
        } else {
            self.digits[0].is_multiple_of(2)
        }
    }

    pub fn is_odd(&self) -> bool {
        !self.is_even()
    }

    pub fn abs(&self) -> Self {
        if self.is_zero() {
            Self::zero()
        } else {
            Self {
                sign: Sign::Positive,
                digits: self.digits.clone(),
            }
        }
    }

    fn normalize(&mut self) {
        while let Some(&0) = self.digits.last() {
            self.digits.pop();
        }
        if self.digits.is_empty() {
            self.sign = Sign::Zero;
        }
    }

    /// Compare absolute magnitudes only.
    pub fn cmp_abs(&self, other: &Self) -> Ordering {
        if self.digits.len() != other.digits.len() {
            return self.digits.len().cmp(&other.digits.len());
        }
        for (a, b) in self.digits.iter().rev().zip(other.digits.iter().rev()) {
            if a != b {
                return a.cmp(b);
            }
        }
        Ordering::Equal
    }

    /// Unsigned addition helper: returns |self| + |other|.
    fn add_abs(&self, other: &Self) -> Self {
        let max_len = self.digits.len().max(other.digits.len());
        let mut digits = Vec::with_capacity(max_len + 1);
        let mut carry = 0u64;

        for i in 0..max_len {
            let a = self.digits.get(i).copied().unwrap_or(0) as u64;
            let b = other.digits.get(i).copied().unwrap_or(0) as u64;
            let sum = a + b + carry;
            digits.push((sum % BASE) as u32);
            carry = sum / BASE;
        }
        if carry > 0 {
            digits.push(carry as u32);
        }
        Self {
            sign: Sign::Positive,
            digits,
        }
    }

    /// Unsigned subtraction helper: assumes |self| >= |other|, returns |self| - |other|.
    fn sub_abs(&self, other: &Self) -> Self {
        let mut digits = Vec::with_capacity(self.digits.len());
        let mut borrow = 0i64;

        for i in 0..self.digits.len() {
            let a = self.digits[i] as i64;
            let b = other.digits.get(i).copied().unwrap_or(0) as i64;
            let diff = a - b - borrow;
            if diff < 0 {
                digits.push((diff + BASE as i64) as u32);
                borrow = 1;
            } else {
                digits.push(diff as u32);
                borrow = 0;
            }
        }
        let mut res = Self {
            sign: Sign::Positive,
            digits,
        };
        res.normalize();
        res
    }

    pub fn checked_add(&self, other: &Self) -> Self {
        if self.is_zero() {
            return other.clone();
        }
        if other.is_zero() {
            return self.clone();
        }

        match (self.sign, other.sign) {
            (Sign::Positive, Sign::Positive) => self.add_abs(other),
            (Sign::Negative, Sign::Negative) => {
                let mut res = self.add_abs(other);
                res.sign = Sign::Negative;
                res
            }
            (Sign::Positive, Sign::Negative) => match self.cmp_abs(other) {
                Ordering::Greater => self.sub_abs(other),
                Ordering::Less => {
                    let mut res = other.sub_abs(self);
                    res.sign = Sign::Negative;
                    res
                }
                Ordering::Equal => Self::zero(),
            },
            (Sign::Negative, Sign::Positive) => match self.cmp_abs(other) {
                Ordering::Greater => {
                    let mut res = self.sub_abs(other);
                    res.sign = Sign::Negative;
                    res
                }
                Ordering::Less => other.sub_abs(self),
                Ordering::Equal => Self::zero(),
            },
            _ => Self::zero(),
        }
    }

    pub fn checked_sub(&self, other: &Self) -> Self {
        self.checked_add(&(-other.clone()))
    }

    pub fn checked_mul(&self, other: &Self) -> Self {
        if self.is_zero() || other.is_zero() {
            return Self::zero();
        }

        let n = self.digits.len();
        let m = other.digits.len();
        let mut result_digits = vec![0u64; n + m];

        for i in 0..n {
            let mut carry = 0u64;
            let a = self.digits[i] as u64;
            for j in 0..m {
                let b = other.digits[j] as u64;
                let cur = result_digits[i + j] + a * b + carry;
                result_digits[i + j] = cur % BASE;
                carry = cur / BASE;
            }
            if carry > 0 {
                result_digits[i + m] += carry;
            }
        }

        let sign = if self.sign == other.sign {
            Sign::Positive
        } else {
            Sign::Negative
        };

        let digits: Vec<u32> = result_digits.into_iter().map(|d| d as u32).collect();
        let mut res = Self { sign, digits };
        res.normalize();
        res
    }

    /// Multiply BigInt by single u32 scalar.
    pub fn mul_u32(&self, scalar: u32) -> Self {
        if self.is_zero() || scalar == 0 {
            return Self::zero();
        }
        let mut digits = Vec::with_capacity(self.digits.len() + 1);
        let mut carry = 0u64;
        let s = scalar as u64;
        for &d in &self.digits {
            let prod = (d as u64) * s + carry;
            digits.push((prod % BASE) as u32);
            carry = prod / BASE;
        }
        if carry > 0 {
            digits.push(carry as u32);
        }
        let mut res = Self {
            sign: self.sign,
            digits,
        };
        res.normalize();
        res
    }

    /// Divide BigInt by positive u32 scalar, returning (quotient, remainder).
    pub fn div_rem_u32(&self, divisor: u32) -> Result<(Self, u32), IntegerError> {
        if divisor == 0 {
            return Err(IntegerError::DivisionByZero);
        }
        if self.is_zero() {
            return Ok((Self::zero(), 0));
        }

        let mut quotient_digits = vec![0u32; self.digits.len()];
        let mut rem = 0u64;
        let d = divisor as u64;

        for i in (0..self.digits.len()).rev() {
            let cur = rem * BASE + self.digits[i] as u64;
            quotient_digits[i] = (cur / d) as u32;
            rem = cur % d;
        }

        let mut q = Self {
            sign: self.sign,
            digits: quotient_digits,
        };
        q.normalize();
        Ok((q, rem as u32))
    }

    /// Unsigned division returning (quotient, remainder).
    pub fn div_rem(&self, other: &Self) -> Result<(Self, Self), IntegerError> {
        if other.is_zero() {
            return Err(IntegerError::DivisionByZero);
        }
        if self.is_zero() {
            return Ok((Self::zero(), Self::zero()));
        }

        let cmp = self.cmp_abs(other);
        if cmp == Ordering::Less {
            return Ok((Self::zero(), self.clone()));
        }
        if cmp == Ordering::Equal {
            let sign = if self.sign == other.sign {
                Sign::Positive
            } else {
                Sign::Negative
            };
            return Ok((
                Self {
                    sign,
                    digits: vec![1],
                },
                Self::zero(),
            ));
        }

        // Fast path for small divisor
        if other.digits.len() == 1 {
            let (q, r) = self.div_rem_u32(other.digits[0])?;
            let sign = if self.sign == other.sign {
                Sign::Positive
            } else {
                Sign::Negative
            };
            let mut q_signed = q;
            q_signed.sign = sign;
            let mut r_signed = BigInt::from_i64(r as i64);
            if self.is_negative() && !r_signed.is_zero() {
                r_signed.sign = Sign::Negative;
            }
            return Ok((q_signed, r_signed));
        }

        // Binary search bit-by-bit division for arbitrary sizes
        let mut quotient = Self::zero();
        let mut current_rem = Self::zero();

        // Convert self to binary representation for shift-and-subtract division
        let abs_other = other.abs();
        let num_bits = self.bit_length();

        for bit in (0..num_bits).rev() {
            current_rem = current_rem.shift_left_bits(1);
            if self.get_bit(bit) {
                current_rem = current_rem.checked_add(&Self::one());
            }
            if current_rem.cmp_abs(&abs_other) != Ordering::Less {
                current_rem = current_rem.sub_abs(&abs_other);
                quotient.set_bit(bit);
            }
        }

        let q_sign = if self.sign == other.sign {
            Sign::Positive
        } else {
            Sign::Negative
        };
        let mut final_q = quotient;
        final_q.sign = q_sign;
        final_q.normalize();

        let mut final_r = current_rem;
        if self.is_negative() && !final_r.is_zero() {
            final_r.sign = Sign::Negative;
        }
        final_r.normalize();

        Ok((final_q, final_r))
    }

    pub fn bit_length(&self) -> usize {
        if self.is_zero() {
            return 0;
        }
        let mut temp = self.abs();
        let mut bits = 0;
        while !temp.is_zero() {
            let (q, _) = temp.div_rem_u32(2).unwrap();
            temp = q;
            bits += 1;
        }
        bits
    }

    pub fn get_bit(&self, bit: usize) -> bool {
        let mut temp = self.abs();
        for _ in 0..bit {
            let (q, _) = temp.div_rem_u32(2).unwrap();
            temp = q;
            if temp.is_zero() {
                return false;
            }
        }
        let (_, rem) = temp.div_rem_u32(2).unwrap();
        rem == 1
    }

    pub fn set_bit(&mut self, bit: usize) {
        let mut pow2 = Self::one();
        for _ in 0..bit {
            pow2 = pow2.mul_u32(2);
        }
        *self = self.checked_add(&pow2);
    }

    fn shift_left_bits(&self, shift: usize) -> Self {
        let mut res = self.clone();
        for _ in 0..shift {
            res = res.mul_u32(2);
        }
        res
    }

    pub fn checked_pow(&self, exp: u32) -> Self {
        if exp == 0 {
            return Self::one();
        }
        if self.is_zero() {
            return Self::zero();
        }
        let mut base = self.clone();
        let mut result = Self::one();
        let mut e = exp;

        while e > 0 {
            if e % 2 == 1 {
                result = result.checked_mul(&base);
            }
            base = base.checked_mul(&base);
            e /= 2;
        }
        result
    }

    /// Modular exponentiation: (self^exp) mod modulus.
    pub fn mod_pow(&self, exp: &Self, modulus: &Self) -> Result<Self, IntegerError> {
        if modulus.is_zero() {
            return Err(IntegerError::DivisionByZero);
        }
        if exp.is_negative() {
            return Err(IntegerError::NegativeExponent);
        }
        let mod_abs = modulus.abs();
        if mod_abs == Self::one() {
            return Ok(Self::zero());
        }

        let mut base = match self.div_rem(&mod_abs) {
            Ok((_, r)) => {
                if r.is_negative() {
                    r.checked_add(&mod_abs)
                } else {
                    r
                }
            }
            Err(e) => return Err(e),
        };

        let mut result = Self::one();
        let mut e = exp.clone();

        while !e.is_zero() {
            if e.is_odd() {
                result = (result.checked_mul(&base)).div_rem(&mod_abs)?.1;
            }
            base = (base.checked_mul(&base)).div_rem(&mod_abs)?.1;
            let (next_e, _) = e.div_rem_u32(2)?;
            e = next_e;
        }

        Ok(result)
    }

    /// Greatest common divisor (Euclidean algorithm).
    pub fn gcd(&self, other: &Self) -> Self {
        let mut a = self.abs();
        let mut b = other.abs();
        while !b.is_zero() {
            let (_, r) = a.div_rem(&b).unwrap();
            a = b;
            b = r;
        }
        a
    }

    /// Least common multiple: |a * b| / gcd(a, b).
    pub fn lcm(&self, other: &Self) -> Result<Self, IntegerError> {
        if self.is_zero() || other.is_zero() {
            return Ok(Self::zero());
        }
        let g = self.gcd(other);
        let (prod_div_g, _) = self.abs().div_rem(&g)?;
        Ok(prod_div_g.checked_mul(&other.abs()))
    }

    /// Extended Euclidean Algorithm returning (gcd, x, y) such that a*x + b*y = gcd(a, b).
    pub fn extended_gcd(a: &Self, b: &Self) -> (Self, Self, Self) {
        if b.is_zero() {
            return (a.abs(), Self::one(), Self::zero());
        }

        let mut old_r = a.clone();
        let mut r = b.clone();
        let mut old_s = Self::one();
        let mut s = Self::zero();
        let mut old_t = Self::zero();
        let mut t = Self::one();

        while !r.is_zero() {
            let (q, rem) = old_r.div_rem(&r).unwrap();
            old_r = r;
            r = rem;

            let next_s = old_s.checked_sub(&q.checked_mul(&s));
            old_s = s;
            s = next_s;

            let next_t = old_t.checked_sub(&q.checked_mul(&t));
            old_t = t;
            t = next_t;
        }

        (old_r, old_s, old_t)
    }

    /// Modular inverse x such that (self * x) = 1 (mod modulus).
    pub fn mod_inverse(&self, modulus: &Self) -> Result<Self, IntegerError> {
        let (g, x, _) = Self::extended_gcd(self, modulus);
        if g != Self::one() {
            return Err(IntegerError::NoModularInverse {
                a: self.to_string(),
                m: modulus.to_string(),
            });
        }
        let (_, r) = x.div_rem(modulus)?;
        if r.is_negative() {
            Ok(r.checked_add(modulus))
        } else {
            Ok(r)
        }
    }

    /// Factorial n! for n >= 0.
    pub fn factorial(n: u32) -> Self {
        let mut res = Self::one();
        for i in 2..=n {
            res = res.mul_u32(i);
        }
        res
    }

    /// Miller-Rabin probabilistic primality test with deterministic bases up to large numbers.
    pub fn is_prime(&self, rounds: usize) -> bool {
        if self.cmp_abs(&BigInt::from_i64(2)) == Ordering::Less {
            return false;
        }
        if *self == BigInt::from_i64(2) || *self == BigInt::from_i64(3) {
            return true;
        }
        if self.is_even() {
            return false;
        }

        // Write self - 1 as 2^s * d with d odd
        let n_minus_1 = self.checked_sub(&Self::one());
        let mut d = n_minus_1.clone();
        let mut s = 0usize;
        while d.is_even() {
            let (q, _) = d.div_rem_u32(2).unwrap();
            d = q;
            s += 1;
        }

        let bases: &[u32] = &[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
        let num_rounds = rounds.max(1).min(bases.len());

        for &base_u32 in &bases[..num_rounds] {
            let a = BigInt::from_i64(base_u32 as i64);
            if a.cmp_abs(&n_minus_1) != Ordering::Less {
                break;
            }

            let mut x = a.mod_pow(&d, self).unwrap_or_else(|_| Self::zero());
            if x == Self::one() || x == n_minus_1 {
                continue;
            }

            let mut composite = true;
            for _ in 0..s - 1 {
                x = x
                    .mod_pow(&BigInt::from_i64(2), self)
                    .unwrap_or_else(|_| Self::zero());
                if x == n_minus_1 {
                    composite = false;
                    break;
                }
            }
            if composite {
                return false;
            }
        }
        true
    }
}

impl fmt::Display for BigInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }
        if self.sign == Sign::Negative {
            write!(f, "-")?;
        }
        let last_idx = self.digits.len() - 1;
        write!(f, "{}", self.digits[last_idx])?;
        for i in (0..last_idx).rev() {
            write!(f, "{:09}", self.digits[i])?;
        }
        Ok(())
    }
}

impl fmt::Debug for BigInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BigInt({})", self)
    }
}

impl PartialOrd for BigInt {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BigInt {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.sign != other.sign {
            return match (self.sign, other.sign) {
                (Sign::Positive, _) => Ordering::Greater,
                (Sign::Negative, _) => Ordering::Less,
                (Sign::Zero, Sign::Negative) => Ordering::Greater,
                (Sign::Zero, Sign::Positive) => Ordering::Less,
                _ => Ordering::Equal,
            };
        }
        match self.sign {
            Sign::Zero => Ordering::Equal,
            Sign::Positive => self.cmp_abs(other),
            Sign::Negative => other.cmp_abs(self),
        }
    }
}

impl Neg for BigInt {
    type Output = Self;
    fn neg(mut self) -> Self::Output {
        self.sign = match self.sign {
            Sign::Positive => Sign::Negative,
            Sign::Negative => Sign::Positive,
            Sign::Zero => Sign::Zero,
        };
        self
    }
}

impl Add for BigInt {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        self.checked_add(&rhs)
    }
}

impl Sub for BigInt {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        self.checked_sub(&rhs)
    }
}

impl Mul for BigInt {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        self.checked_mul(&rhs)
    }
}

impl Div for BigInt {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        self.div_rem(&rhs)
            .expect("division by zero in BigInt::div")
            .0
    }
}

impl Rem for BigInt {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self::Output {
        self.div_rem(&rhs)
            .expect("division by zero in BigInt::rem")
            .1
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Checked Exact Arithmetic Helpers (64-bit & 128-bit)
// ─────────────────────────────────────────────────────────────────────────────

pub fn checked_add_i64(a: i64, b: i64) -> Result<i64, IntegerError> {
    a.checked_add(b).ok_or_else(|| IntegerError::Overflow {
        operation: format!("{} + {}", a, b),
    })
}

pub fn checked_sub_i64(a: i64, b: i64) -> Result<i64, IntegerError> {
    a.checked_sub(b).ok_or_else(|| IntegerError::Overflow {
        operation: format!("{} - {}", a, b),
    })
}

pub fn checked_mul_i64(a: i64, b: i64) -> Result<i64, IntegerError> {
    a.checked_mul(b).ok_or_else(|| IntegerError::Overflow {
        operation: format!("{} * {}", a, b),
    })
}

pub fn checked_div_i64(a: i64, b: i64) -> Result<i64, IntegerError> {
    if b == 0 {
        return Err(IntegerError::DivisionByZero);
    }
    a.checked_div(b).ok_or_else(|| IntegerError::Overflow {
        operation: format!("{} / {}", a, b),
    })
}

pub fn checked_rem_i64(a: i64, b: i64) -> Result<i64, IntegerError> {
    if b == 0 {
        return Err(IntegerError::DivisionByZero);
    }
    a.checked_rem(b).ok_or_else(|| IntegerError::Overflow {
        operation: format!("{} % {}", a, b),
    })
}

pub fn checked_gcd_i64(mut a: i64, mut b: i64) -> i64 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}

pub fn checked_lcm_i64(a: i64, b: i64) -> Result<i64, IntegerError> {
    if a == 0 || b == 0 {
        return Ok(0);
    }
    let g = checked_gcd_i64(a, b);
    let div = a.checked_div(g).ok_or_else(|| IntegerError::Overflow {
        operation: format!("lcm({}, {})", a, b),
    })?;
    div.checked_mul(b)
        .map(|v| v.abs())
        .ok_or_else(|| IntegerError::Overflow {
            operation: format!("lcm({}, {})", a, b),
        })
}
