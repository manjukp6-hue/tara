//! Symbolic expression representation, differentiation, and simplification for MathEngine.

use std::collections::HashMap;
use std::fmt;
use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum SymbolicError {
    #[error("unbound variable: {0}")]
    UnboundVariable(String),
    #[error("division by zero in expression evaluation")]
    DivisionByZero,
}

/// Symbolic mathematical expression tree.
#[derive(Clone, PartialEq, Debug)]
pub enum Expr {
    Num(f64),
    Var(String),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Pow(Box<Expr>, Box<Expr>),
    Neg(Box<Expr>),
}

impl Expr {
    pub fn num(n: f64) -> Self {
        Expr::Num(n)
    }

    pub fn var(name: &str) -> Self {
        Expr::Var(name.to_string())
    }

    pub fn make_add(a: Expr, b: Expr) -> Self {
        Expr::Add(Box::new(a), Box::new(b))
    }

    pub fn make_sub(a: Expr, b: Expr) -> Self {
        Expr::Sub(Box::new(a), Box::new(b))
    }

    pub fn make_mul(a: Expr, b: Expr) -> Self {
        Expr::Mul(Box::new(a), Box::new(b))
    }

    pub fn make_div(a: Expr, b: Expr) -> Self {
        Expr::Div(Box::new(a), Box::new(b))
    }

    pub fn pow(a: Expr, b: Expr) -> Self {
        Expr::Pow(Box::new(a), Box::new(b))
    }

    /// Evaluate expression with variable bindings.
    pub fn eval(&self, env: &HashMap<&str, f64>) -> Result<f64, SymbolicError> {
        match self {
            Expr::Num(n) => Ok(*n),
            Expr::Var(v) => env
                .get(v.as_str())
                .copied()
                .ok_or_else(|| SymbolicError::UnboundVariable(v.clone())),
            Expr::Add(a, b) => Ok(a.eval(env)? + b.eval(env)?),
            Expr::Sub(a, b) => Ok(a.eval(env)? - b.eval(env)?),
            Expr::Mul(a, b) => Ok(a.eval(env)? * b.eval(env)?),
            Expr::Div(a, b) => {
                let den = b.eval(env)?;
                if den.abs() < 1e-15 {
                    return Err(SymbolicError::DivisionByZero);
                }
                Ok(a.eval(env)? / den)
            }
            Expr::Pow(a, b) => Ok(a.eval(env)?.powf(b.eval(env)?)),
            Expr::Neg(a) => Ok(-a.eval(env)?),
        }
    }

    /// Symbolic differentiation with respect to variable `var_name`.
    pub fn diff(&self, var_name: &str) -> Expr {
        match self {
            Expr::Num(_) => Expr::num(0.0),
            Expr::Var(v) => {
                if v == var_name {
                    Expr::num(1.0)
                } else {
                    Expr::num(0.0)
                }
            }
            Expr::Add(a, b) => Expr::make_add(a.diff(var_name), b.diff(var_name)),
            Expr::Sub(a, b) => Expr::make_sub(a.diff(var_name), b.diff(var_name)),
            Expr::Mul(a, b) => {
                // Product rule: d(u*v) = u'*v + u*v'
                Expr::make_add(
                    Expr::make_mul(a.diff(var_name), (**b).clone()),
                    Expr::make_mul((**a).clone(), b.diff(var_name)),
                )
            }
            Expr::Div(a, b) => {
                // Quotient rule: d(u/v) = (u'*v - u*v') / v^2
                Expr::make_div(
                    Expr::make_sub(
                        Expr::make_mul(a.diff(var_name), (**b).clone()),
                        Expr::make_mul((**a).clone(), b.diff(var_name)),
                    ),
                    Expr::pow((**b).clone(), Expr::num(2.0)),
                )
            }
            Expr::Pow(base, exp) => {
                // Power rule for base(x)^c: c * base(x)^(c-1) * base'(x)
                if let Expr::Num(c) = **exp {
                    Expr::make_mul(
                        Expr::make_mul(
                            Expr::num(c),
                            Expr::pow((**base).clone(), Expr::num(c - 1.0)),
                        ),
                        base.diff(var_name),
                    )
                } else {
                    // General d(u^v): u^v * (v' * ln(u) + v * u' / u)
                    Expr::make_mul(
                        self.clone(),
                        Expr::make_add(
                            Expr::make_mul(exp.diff(var_name), Expr::var(&format!("ln({})", base))),
                            Expr::make_div(
                                Expr::make_mul((**exp).clone(), base.diff(var_name)),
                                (**base).clone(),
                            ),
                        ),
                    )
                }
            }
            Expr::Neg(a) => Expr::Neg(Box::new(a.diff(var_name))),
        }
    }

    /// Algebraic simplification (constant folding and identities).
    pub fn simplify(&self) -> Expr {
        match self {
            Expr::Add(a, b) => {
                let sa = a.simplify();
                let sb = b.simplify();
                match (&sa, &sb) {
                    (Expr::Num(x), Expr::Num(y)) => Expr::num(x + y),
                    (Expr::Num(0.0), other) => other.clone(),
                    (other, Expr::Num(0.0)) => other.clone(),
                    _ => Expr::make_add(sa, sb),
                }
            }
            Expr::Sub(a, b) => {
                let sa = a.simplify();
                let sb = b.simplify();
                match (&sa, &sb) {
                    (Expr::Num(x), Expr::Num(y)) => Expr::num(x - y),
                    (other, Expr::Num(0.0)) => other.clone(),
                    _ if sa == sb => Expr::num(0.0),
                    _ => Expr::make_sub(sa, sb),
                }
            }
            Expr::Mul(a, b) => {
                let sa = a.simplify();
                let sb = b.simplify();
                match (&sa, &sb) {
                    (Expr::Num(x), Expr::Num(y)) => Expr::num(x * y),
                    (Expr::Num(0.0), _) | (_, Expr::Num(0.0)) => Expr::num(0.0),
                    (Expr::Num(1.0), other) => other.clone(),
                    (other, Expr::Num(1.0)) => other.clone(),
                    _ => Expr::make_mul(sa, sb),
                }
            }
            Expr::Div(a, b) => {
                let sa = a.simplify();
                let sb = b.simplify();
                match (&sa, &sb) {
                    (Expr::Num(x), Expr::Num(y)) if y.abs() > 1e-15 => Expr::num(x / y),
                    (Expr::Num(0.0), _) => Expr::num(0.0),
                    (other, Expr::Num(1.0)) => other.clone(),
                    _ => Expr::make_div(sa, sb),
                }
            }
            Expr::Pow(a, b) => {
                let sa = a.simplify();
                let sb = b.simplify();
                match (&sa, &sb) {
                    (Expr::Num(x), Expr::Num(y)) => Expr::num(x.powf(*y)),
                    (_, Expr::Num(0.0)) => Expr::num(1.0),
                    (other, Expr::Num(1.0)) => other.clone(),
                    _ => Expr::pow(sa, sb),
                }
            }
            Expr::Neg(a) => {
                let sa = a.simplify();
                if let Expr::Num(x) = sa {
                    Expr::num(-x)
                } else {
                    Expr::Neg(Box::new(sa))
                }
            }
            _ => self.clone(),
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Num(n) => write!(f, "{}", n),
            Expr::Var(v) => write!(f, "{}", v),
            Expr::Add(a, b) => write!(f, "({} + {})", a, b),
            Expr::Sub(a, b) => write!(f, "({} - {})", a, b),
            Expr::Mul(a, b) => write!(f, "({} * {})", a, b),
            Expr::Div(a, b) => write!(f, "({} / {})", a, b),
            Expr::Pow(a, b) => write!(f, "({} ^ {})", a, b),
            Expr::Neg(a) => write!(f, "(-{})", a),
        }
    }
}

impl std::ops::Add for Expr {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Expr::Add(Box::new(self), Box::new(rhs))
    }
}

impl std::ops::Sub for Expr {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Expr::Sub(Box::new(self), Box::new(rhs))
    }
}

impl std::ops::Mul for Expr {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Expr::Mul(Box::new(self), Box::new(rhs))
    }
}

impl std::ops::Div for Expr {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        Expr::Div(Box::new(self), Box::new(rhs))
    }
}
