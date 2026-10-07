//! Polynomial algebra and equation solving for TARA Native Engine.
//!
//! Provides:
//! - Polynomial arithmetic: addition, subtraction, multiplication, Horner evaluation,
//!   symbolic differentiation, and indefinite integration.
//! - Equation solving:
//!   * Linear: ax + b = 0
//!   * Quadratic: ax^2 + bx + c = 0 (exact real and complex conjugate roots)
//!   * Cubic: ax^3 + bx^2 + cx + d = 0 via Cardano's analytical method.

use std::fmt;
use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum AlgebraError {
    #[error("division by zero or leading coefficient is zero")]
    LeadingCoefficientZero,
    #[error("no solution exists for equation")]
    NoSolution,
    #[error("infinite solutions exist (identity equation)")]
    InfiniteSolutions,
}

/// Real or complex root representation.
#[derive(Debug, Clone, PartialEq)]
pub enum ComplexRoot {
    Real(f64),
    Complex { real: f64, imag: f64 },
}

impl fmt::Display for ComplexRoot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ComplexRoot::Real(r) => write!(f, "{:.6}", r),
            ComplexRoot::Complex { real, imag } => {
                if *imag >= 0.0 {
                    write!(f, "{:.6} + {:.6}i", real, imag)
                } else {
                    write!(f, "{:.6} - {:.6}i", real, -imag)
                }
            }
        }
    }
}

/// Polynomial representation: c_0 + c_1*x + c_2*x^2 + ... + c_n*x^n.
#[derive(Clone, PartialEq, Debug)]
pub struct Polynomial {
    /// Coefficients in ascending order of degree (c_0 at index 0).
    pub coeffs: Vec<f64>,
}

impl Polynomial {
    pub fn new(coeffs: Vec<f64>) -> Self {
        let mut p = Self { coeffs };
        p.normalize();
        p
    }

    pub fn zero() -> Self {
        Self { coeffs: vec![0.0] }
    }

    pub fn degree(&self) -> usize {
        if self.coeffs.len() <= 1 {
            0
        } else {
            self.coeffs.len() - 1
        }
    }

    pub fn normalize(&mut self) {
        while self.coeffs.len() > 1
            && self
                .coeffs
                .last()
                .map(|&c| c.abs() < 1e-15)
                .unwrap_or(false)
        {
            self.coeffs.pop();
        }
        if self.coeffs.is_empty() {
            self.coeffs.push(0.0);
        }
    }

    /// Evaluate polynomial at x using Horner's rule.
    pub fn evaluate(&self, x: f64) -> f64 {
        let mut result = 0.0;
        for &c in self.coeffs.iter().rev() {
            result = result * x + c;
        }
        result
    }

    pub fn add(&self, other: &Self) -> Self {
        let max_len = self.coeffs.len().max(other.coeffs.len());
        let mut out = Vec::with_capacity(max_len);
        for i in 0..max_len {
            let a = self.coeffs.get(i).copied().unwrap_or(0.0);
            let b = other.coeffs.get(i).copied().unwrap_or(0.0);
            out.push(a + b);
        }
        Self::new(out)
    }

    pub fn sub(&self, other: &Self) -> Self {
        let max_len = self.coeffs.len().max(other.coeffs.len());
        let mut out = Vec::with_capacity(max_len);
        for i in 0..max_len {
            let a = self.coeffs.get(i).copied().unwrap_or(0.0);
            let b = other.coeffs.get(i).copied().unwrap_or(0.0);
            out.push(a - b);
        }
        Self::new(out)
    }

    pub fn mul(&self, other: &Self) -> Self {
        let out_len = self.coeffs.len() + other.coeffs.len() - 1;
        let mut out = vec![0.0; out_len];
        for (i, &a) in self.coeffs.iter().enumerate() {
            for (j, &b) in other.coeffs.iter().enumerate() {
                out[i + j] += a * b;
            }
        }
        Self::new(out)
    }

    /// First symbolic derivative dP/dx.
    pub fn derivative(&self) -> Self {
        if self.coeffs.len() <= 1 {
            return Self::zero();
        }
        let mut out = Vec::with_capacity(self.coeffs.len() - 1);
        for i in 1..self.coeffs.len() {
            out.push(self.coeffs[i] * i as f64);
        }
        Self::new(out)
    }

    /// Indefinite integral indefinite_integral(P dx) with constant C.
    pub fn integral(&self, constant: f64) -> Self {
        let mut out = Vec::with_capacity(self.coeffs.len() + 1);
        out.push(constant);
        for (i, &c) in self.coeffs.iter().enumerate() {
            out.push(c / (i + 1) as f64);
        }
        Self::new(out)
    }

    /// Definite integral between bounds [a, b].
    pub fn integrate(&self, a: f64, b: f64) -> f64 {
        let antiderivative = self.integral(0.0);
        antiderivative.evaluate(b) - antiderivative.evaluate(a)
    }
}

/// Analytical equation solvers.
pub struct EquationSolver;

impl EquationSolver {
    /// Solve linear equation: ax + b = 0.
    pub fn solve_linear(a: f64, b: f64) -> Result<f64, AlgebraError> {
        if a.abs() < 1e-15 {
            if b.abs() < 1e-15 {
                return Err(AlgebraError::InfiniteSolutions);
            } else {
                return Err(AlgebraError::NoSolution);
            }
        }
        Ok(-b / a)
    }

    /// Solve quadratic equation: ax^2 + bx + c = 0.
    pub fn solve_quadratic(
        a: f64,
        b: f64,
        c: f64,
    ) -> Result<(ComplexRoot, ComplexRoot), AlgebraError> {
        if a.abs() < 1e-15 {
            let root = Self::solve_linear(b, c)?;
            return Ok((ComplexRoot::Real(root), ComplexRoot::Real(root)));
        }

        let disc = b * b - 4.0 * a * c;
        if disc > 1e-14 {
            let sqrt_disc = disc.sqrt();
            let r1 = (-b + sqrt_disc) / (2.0 * a);
            let r2 = (-b - sqrt_disc) / (2.0 * a);
            Ok((ComplexRoot::Real(r1), ComplexRoot::Real(r2)))
        } else if disc.abs() <= 1e-14 {
            let r = -b / (2.0 * a);
            Ok((ComplexRoot::Real(r), ComplexRoot::Real(r)))
        } else {
            let real = -b / (2.0 * a);
            let imag = (-disc).sqrt() / (2.0 * a).abs();
            Ok((
                ComplexRoot::Complex { real, imag },
                ComplexRoot::Complex { real, imag: -imag },
            ))
        }
    }

    /// Solve depressed cubic y^3 + py + q = 0 via Cardano's formula,
    /// then map back to ax^3 + bx^2 + cx + d = 0 via x = y - b/(3a).
    pub fn solve_cubic(a: f64, b: f64, c: f64, d: f64) -> Result<Vec<ComplexRoot>, AlgebraError> {
        if a.abs() < 1e-15 {
            let (r1, r2) = Self::solve_quadratic(b, c, d)?;
            return Ok(vec![r1, r2]);
        }

        let shift = b / (3.0 * a);
        let p = (3.0 * a * c - b * b) / (3.0 * a * a);
        let q = (2.0 * b.powi(3) - 9.0 * a * b * c + 27.0 * a * a * d) / (27.0 * a.powi(3));

        let delta = (q / 2.0).powi(2) + (p / 3.0).powi(3);

        if delta > 1e-12 {
            // One real root, two complex conjugate roots
            let sqrt_delta = delta.sqrt();
            let u_cube = -q / 2.0 + sqrt_delta;
            let v_cube = -q / 2.0 - sqrt_delta;

            let u = if u_cube >= 0.0 {
                u_cube.cbrt()
            } else {
                -(-u_cube).cbrt()
            };
            let v = if v_cube >= 0.0 {
                v_cube.cbrt()
            } else {
                -(-v_cube).cbrt()
            };

            let y1 = u + v;
            let real_part = -(u + v) / 2.0;
            let imag_part = ((3.0f64).sqrt() / 2.0) * (u - v).abs();

            Ok(vec![
                ComplexRoot::Real(y1 - shift),
                ComplexRoot::Complex {
                    real: real_part - shift,
                    imag: imag_part,
                },
                ComplexRoot::Complex {
                    real: real_part - shift,
                    imag: -imag_part,
                },
            ])
        } else if delta.abs() <= 1e-12 {
            // All roots real, at least two are equal
            let u = if q <= 0.0 {
                (-q / 2.0).cbrt()
            } else {
                -((q / 2.0).cbrt())
            };
            let y1 = 2.0 * u;
            let y2 = -u;
            Ok(vec![
                ComplexRoot::Real(y1 - shift),
                ComplexRoot::Real(y2 - shift),
                ComplexRoot::Real(y2 - shift),
            ])
        } else {
            // Three distinct real roots (casus irreducibilis)
            let r = (-p.powi(3) / 27.0).sqrt();
            let phi = (-q / (2.0 * r)).clamp(-1.0, 1.0).acos();
            let pi = std::f64::consts::PI;

            let y1 = 2.0 * (-p / 3.0).sqrt() * (phi / 3.0).cos();
            let y2 = 2.0 * (-p / 3.0).sqrt() * ((phi + 2.0 * pi) / 3.0).cos();
            let y3 = 2.0 * (-p / 3.0).sqrt() * ((phi + 4.0 * pi) / 3.0).cos();

            Ok(vec![
                ComplexRoot::Real(y1 - shift),
                ComplexRoot::Real(y2 - shift),
                ComplexRoot::Real(y3 - shift),
            ])
        }
    }
}

impl fmt::Display for Polynomial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        for (power, &c) in self.coeffs.iter().enumerate() {
            if c.abs() < 1e-15 {
                continue;
            }
            if !first {
                if c >= 0.0 {
                    write!(f, " + ")?;
                } else {
                    write!(f, " - ")?;
                }
            } else if c < 0.0 {
                write!(f, "-")?;
            }
            first = false;

            let abs_c = c.abs();
            if power == 0 {
                write!(f, "{:.4}", abs_c)?;
            } else if power == 1 {
                if (abs_c - 1.0).abs() < 1e-15 {
                    write!(f, "x")?;
                } else {
                    write!(f, "{:.4}x", abs_c)?;
                }
            } else {
                if (abs_c - 1.0).abs() < 1e-15 {
                    write!(f, "x^{}", power)?;
                } else {
                    write!(f, "{:.4}x^{}", abs_c, power)?;
                }
            }
        }
        if first {
            write!(f, "0")?;
        }
        Ok(())
    }
}
