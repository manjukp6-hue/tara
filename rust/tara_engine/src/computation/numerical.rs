//! Numerical methods: root finding, differentiation, integration, and ODE solving.
//!
//! Provides:
//! - Root finding: Bisection, Newton-Raphson, and Secant methods.
//! - Numerical differentiation: Central difference with adaptive step size.
//! - Numerical quadrature (integration): Trapezoidal rule, Simpson's 1/3 and 3/8 rules.
//! - Initial Value Problems: 4th-order Runge-Kutta (RK4) integrator for differential equations.

use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum NumericalError {
    #[error("root finding failed to converge within {max_iter} iterations")]
    MaxIterationsExceeded { max_iter: usize },
    #[error("interval [a, b] does not bracket a root (f(a)*f(b) > 0)")]
    IntervalNotBracketing,
    #[error("derivative evaluated to zero during Newton iteration")]
    ZeroDerivative,
    #[error("invalid step size or integration interval")]
    InvalidInterval,
    #[error("division by zero")]
    DivisionByZero,
}

pub struct NumericalMethods;

impl NumericalMethods {
    /// Bisection method to find root of f(x) = 0 in [a, b].
    pub fn bisection<F>(
        f: F,
        mut a: f64,
        mut b: f64,
        tol: f64,
        max_iter: usize,
    ) -> Result<f64, NumericalError>
    where
        F: Fn(f64) -> f64,
    {
        let mut fa = f(a);
        if fa * f(b) > 0.0 {
            return Err(NumericalError::IntervalNotBracketing);
        }
        if fa.abs() < tol {
            return Ok(a);
        }
        if f(b).abs() < tol {
            return Ok(b);
        }

        for _ in 0..max_iter {
            let mid = a + (b - a) / 2.0;
            let fmid = f(mid);

            if fmid.abs() < tol || (b - a) / 2.0 < tol {
                return Ok(mid);
            }

            if fa * fmid < 0.0 {
                b = mid;
            } else {
                a = mid;
                fa = fmid;
            }
        }

        Err(NumericalError::MaxIterationsExceeded { max_iter })
    }

    /// Newton-Raphson method to find root of f(x) = 0 using analytical or numerical derivative.
    pub fn newton_raphson<F, DF>(
        f: F,
        df: DF,
        mut x0: f64,
        tol: f64,
        max_iter: usize,
    ) -> Result<f64, NumericalError>
    where
        F: Fn(f64) -> f64,
        DF: Fn(f64) -> f64,
    {
        for _ in 0..max_iter {
            let fx = f(x0);
            if fx.abs() < tol {
                return Ok(x0);
            }
            let dfx = df(x0);
            if dfx.abs() < 1e-15 {
                return Err(NumericalError::ZeroDerivative);
            }
            let x1 = x0 - fx / dfx;
            if (x1 - x0).abs() < tol {
                return Ok(x1);
            }
            x0 = x1;
        }
        Err(NumericalError::MaxIterationsExceeded { max_iter })
    }

    /// Secant method (derivative-free root finding).
    pub fn secant<F>(
        f: F,
        mut x0: f64,
        mut x1: f64,
        tol: f64,
        max_iter: usize,
    ) -> Result<f64, NumericalError>
    where
        F: Fn(f64) -> f64,
    {
        let mut f0 = f(x0);
        let mut f1 = f(x1);

        for _ in 0..max_iter {
            if f1.abs() < tol {
                return Ok(x1);
            }
            let denom = f1 - f0;
            if denom.abs() < 1e-15 {
                return Err(NumericalError::DivisionByZero);
            }
            let x2 = x1 - f1 * (x1 - x0) / denom;
            if (x2 - x1).abs() < tol {
                return Ok(x2);
            }
            x0 = x1;
            f0 = f1;
            x1 = x2;
            f1 = f(x1);
        }
        Err(NumericalError::MaxIterationsExceeded { max_iter })
    }

    /// Central difference numerical derivative f'(x) with optimal step size h.
    pub fn derivative<F>(f: F, x: f64) -> f64
    where
        F: Fn(f64) -> f64,
    {
        let eps = f64::EPSILON;
        let h = (eps.cbrt() * (1.0 + x.abs())).max(1e-8);
        (f(x + h) - f(x - h)) / (2.0 * h)
    }

    /// Second derivative f''(x) via central second difference.
    pub fn second_derivative<F>(f: F, x: f64) -> f64
    where
        F: Fn(f64) -> f64,
    {
        let h = 1e-4 * (1.0 + x.abs());
        (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h)
    }

    /// Numerical integration via composite Trapezoidal rule with n panels.
    pub fn trapezoidal<F>(f: F, a: f64, b: f64, n: usize) -> Result<f64, NumericalError>
    where
        F: Fn(f64) -> f64,
    {
        if n == 0 || a >= b {
            return Err(NumericalError::InvalidInterval);
        }
        let h = (b - a) / (n as f64);
        let mut sum = 0.5 * (f(a) + f(b));
        for i in 1..n {
            sum += f(a + (i as f64) * h);
        }
        Ok(sum * h)
    }

    /// Numerical integration via composite Simpson's 1/3 rule (n must be even).
    pub fn simpsons_one_third<F>(f: F, a: f64, b: f64, n: usize) -> Result<f64, NumericalError>
    where
        F: Fn(f64) -> f64,
    {
        if n == 0 || !n.is_multiple_of(2) || a >= b {
            return Err(NumericalError::InvalidInterval);
        }
        let h = (b - a) / (n as f64);
        let mut sum = f(a) + f(b);

        for i in 1..n {
            let x = a + (i as f64) * h;
            if i % 2 == 1 {
                sum += 4.0 * f(x);
            } else {
                sum += 2.0 * f(x);
            }
        }
        Ok(sum * h / 3.0)
    }

    /// Runge-Kutta 4th-Order (RK4) single step for ODE: dy/dt = f(t, y).
    pub fn rk4_step<F>(f: &F, t: f64, y: f64, h: f64) -> f64
    where
        F: Fn(f64, f64) -> f64,
    {
        let k1 = f(t, y);
        let k2 = f(t + 0.5 * h, y + 0.5 * h * k1);
        let k3 = f(t + 0.5 * h, y + 0.5 * h * k2);
        let k4 = f(t + h, y + h * k3);
        y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    }

    /// Integrate ODE dy/dt = f(t, y) from t0 to t_end using RK4 across n steps.
    pub fn rk4_integrate<F>(
        f: F,
        t0: f64,
        y0: f64,
        t_end: f64,
        steps: usize,
    ) -> Result<Vec<(f64, f64)>, NumericalError>
    where
        F: Fn(f64, f64) -> f64,
    {
        if steps == 0 || t0 >= t_end {
            return Err(NumericalError::InvalidInterval);
        }
        let h = (t_end - t0) / steps as f64;
        let mut trajectory = Vec::with_capacity(steps + 1);
        let mut t = t0;
        let mut y = y0;
        trajectory.push((t, y));

        for _ in 0..steps {
            y = Self::rk4_step(&f, t, y, h);
            t += h;
            trajectory.push((t, y));
        }
        Ok(trajectory)
    }
}
