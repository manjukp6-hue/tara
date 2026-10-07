//! Trigonometry and hyperbolic function computations for MathEngine.

use super::constants::PI;
use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum TrigError {
    #[error("tangent / secant undefined at odd multiples of pi/2")]
    UndefinedTanOrSec,
    #[error("cotangent / cosecant undefined at multiples of pi")]
    UndefinedCotOrCsc,
    #[error("input {val} out of domain [-1.0, 1.0] for {func}")]
    DomainError { func: &'static str, val: f64 },
}

pub struct Trigonometry;

impl Trigonometry {
    pub fn deg_to_rad(deg: f64) -> f64 {
        deg * (PI / 180.0)
    }

    pub fn rad_to_deg(rad: f64) -> f64 {
        rad * (180.0 / PI)
    }

    pub fn sin(rad: f64) -> f64 {
        rad.sin()
    }

    pub fn cos(rad: f64) -> f64 {
        rad.cos()
    }

    pub fn tan(rad: f64) -> Result<f64, TrigError> {
        let cos_val = rad.cos();
        if cos_val.abs() < 1e-15 {
            return Err(TrigError::UndefinedTanOrSec);
        }
        Ok(rad.tan())
    }

    pub fn cot(rad: f64) -> Result<f64, TrigError> {
        let sin_val = rad.sin();
        if sin_val.abs() < 1e-15 {
            return Err(TrigError::UndefinedCotOrCsc);
        }
        Ok(rad.cos() / sin_val)
    }

    pub fn sec(rad: f64) -> Result<f64, TrigError> {
        let cos_val = rad.cos();
        if cos_val.abs() < 1e-15 {
            return Err(TrigError::UndefinedTanOrSec);
        }
        Ok(1.0 / cos_val)
    }

    pub fn csc(rad: f64) -> Result<f64, TrigError> {
        let sin_val = rad.sin();
        if sin_val.abs() < 1e-15 {
            return Err(TrigError::UndefinedCotOrCsc);
        }
        Ok(1.0 / sin_val)
    }

    pub fn asin(x: f64) -> Result<f64, TrigError> {
        if !(-1.0..=1.0).contains(&x) {
            return Err(TrigError::DomainError {
                func: "asin",
                val: x,
            });
        }
        Ok(x.asin())
    }

    pub fn acos(x: f64) -> Result<f64, TrigError> {
        if !(-1.0..=1.0).contains(&x) {
            return Err(TrigError::DomainError {
                func: "acos",
                val: x,
            });
        }
        Ok(x.acos())
    }

    pub fn atan(x: f64) -> f64 {
        x.atan()
    }

    pub fn atan2(y: f64, x: f64) -> f64 {
        y.atan2(x)
    }

    // ── Hyperbolic functions ─────────────────────────────────────────────────

    pub fn sinh(x: f64) -> f64 {
        x.sinh()
    }

    pub fn cosh(x: f64) -> f64 {
        x.cosh()
    }

    pub fn tanh(x: f64) -> f64 {
        x.tanh()
    }

    /// Verifies Pythagorean identity: sin^2(x) + cos^2(x) == 1.
    pub fn verify_pythagorean_identity(rad: f64) -> (f64, bool) {
        let s = rad.sin();
        let c = rad.cos();
        let val = s * s + c * c;
        let is_valid = (val - 1.0).abs() < 1e-14;
        (val, is_valid)
    }
}
