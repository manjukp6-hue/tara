//! Probability and combinatorics computation engine for TARA Native Engine.
//!
//! Provides:
//! - Exact combinatorics: Permutations P(n, k), Combinations C(n, k), Factorials.
//! - Discrete distributions: Binomial PMF & CDF, Poisson PMF & CDF.
//! - Continuous distributions: Normal (Gaussian) PDF & CDF via error function erf(z).
//! - Bayesian probability updating: Bayes' Theorem with arbitrary prior and likelihoods.

use thiserror::Error;

use super::integer::BigInt;

#[derive(Debug, Error, PartialEq)]
pub enum ProbError {
    #[error("domain error: {0}")]
    DomainError(String),
    #[error("probability value out of bounds [0.0, 1.0]: {0}")]
    InvalidProbability(f64),
    #[error("division by zero")]
    DivisionByZero,
}

/// Probability and combinatorics calculations.
pub struct Probability;

impl Probability {
    /// Exact permutations P(n, k) = n! / (n - k)!.
    pub fn permutations(n: u64, k: u64) -> Result<BigInt, ProbError> {
        if k > n {
            return Ok(BigInt::zero());
        }
        let mut result = BigInt::one();
        for i in (n - k + 1)..=n {
            result = result.checked_mul(&BigInt::from_i64(i as i64));
        }
        Ok(result)
    }

    /// Exact combinations C(n, k) = n! / (k! * (n - k)!).
    pub fn combinations(n: u64, k: u64) -> Result<BigInt, ProbError> {
        if k > n {
            return Ok(BigInt::zero());
        }
        let k = k.min(n - k); // Symmetry C(n, k) = C(n, n-k)
        let mut num = BigInt::one();
        let mut den = BigInt::one();

        for i in 1..=k {
            num = num.checked_mul(&BigInt::from_i64((n - i + 1) as i64));
            den = den.checked_mul(&BigInt::from_i64(i as i64));
        }

        let (res, _) = num.div_rem(&den).map_err(|_| ProbError::DivisionByZero)?;
        Ok(res)
    }

    /// Combinations as f64 (for float distribution calculations).
    pub fn combinations_f64(n: u32, k: u32) -> f64 {
        if k > n {
            return 0.0;
        }
        let k = k.min(n - k);
        let mut c = 1.0;
        for i in 1..=k {
            c = c * (n - i + 1) as f64 / i as f64;
        }
        c
    }

    /// Binomial PMF P(X = k) = C(n, k) * p^k * (1-p)^(n-k).
    pub fn binomial_pmf(n: u32, k: u32, p: f64) -> Result<f64, ProbError> {
        if !(0.0..=1.0).contains(&p) {
            return Err(ProbError::InvalidProbability(p));
        }
        if k > n {
            return Ok(0.0);
        }
        let c = Self::combinations_f64(n, k);
        let prob = c * p.powi(k as i32) * (1.0 - p).powi((n - k) as i32);
        Ok(prob)
    }

    /// Binomial CDF P(X <= k) = sum_{i=0..k} P(X = i).
    pub fn binomial_cdf(n: u32, k: u32, p: f64) -> Result<f64, ProbError> {
        if !(0.0..=1.0).contains(&p) {
            return Err(ProbError::InvalidProbability(p));
        }
        let mut sum = 0.0;
        let limit = k.min(n);
        for i in 0..=limit {
            sum += Self::binomial_pmf(n, i, p)?;
        }
        Ok(sum.min(1.0))
    }

    /// Poisson PMF P(X = k) = (lambda^k * e^(-lambda)) / k!.
    pub fn poisson_pmf(lambda: f64, k: u32) -> Result<f64, ProbError> {
        if lambda <= 0.0 {
            return Err(ProbError::DomainError("lambda must be positive".into()));
        }
        let mut log_fact = 0.0;
        for i in 1..=k {
            log_fact += (i as f64).ln();
        }
        // Use log space to prevent overflow for large k or lambda
        let log_p = (k as f64) * lambda.ln() - lambda - log_fact;
        Ok(log_p.exp())
    }

    /// Poisson CDF P(X <= k).
    pub fn poisson_cdf(lambda: f64, k: u32) -> Result<f64, ProbError> {
        if lambda <= 0.0 {
            return Err(ProbError::DomainError("lambda must be positive".into()));
        }
        let mut sum = 0.0;
        for i in 0..=k {
            sum += Self::poisson_pmf(lambda, i)?;
        }
        Ok(sum.min(1.0))
    }

    /// Standard Normal (Gaussian) Probability Density Function (PDF).
    pub fn normal_pdf(x: f64, mean: f64, std_dev: f64) -> Result<f64, ProbError> {
        if std_dev <= 0.0 {
            return Err(ProbError::DomainError(
                "standard deviation must be > 0".into(),
            ));
        }
        let z = (x - mean) / std_dev;
        let pi = std::f64::consts::PI;
        let coeff = 1.0 / (std_dev * (2.0 * pi).sqrt());
        Ok(coeff * (-0.5 * z * z).exp())
    }

    /// Error function erf(x) via Abramowitz & Stegun high-precision rational approximation.
    pub fn erf(x: f64) -> f64 {
        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let abs_x = x.abs();

        // A&S formula 7.1.26
        let p = 0.3275911;
        let t = 1.0 / (1.0 + p * abs_x);
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;

        let poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))));
        sign * (1.0 - poly * (-abs_x * abs_x).exp())
    }

    /// Normal Cumulative Distribution Function (CDF) Phi(x).
    pub fn normal_cdf(x: f64, mean: f64, std_dev: f64) -> Result<f64, ProbError> {
        if std_dev <= 0.0 {
            return Err(ProbError::DomainError(
                "standard deviation must be > 0".into(),
            ));
        }
        let z = (x - mean) / (std_dev * std::f64::consts::SQRT_2);
        Ok(0.5 * (1.0 + Self::erf(z)))
    }

    /// Bayes' Theorem: P(A|B) = (P(B|A) * P(A)) / P(B).
    pub fn bayes_theorem(p_b_given_a: f64, p_a: f64, p_b: f64) -> Result<f64, ProbError> {
        if !(0.0..=1.0).contains(&p_b_given_a) {
            return Err(ProbError::InvalidProbability(p_b_given_a));
        }
        if !(0.0..=1.0).contains(&p_a) {
            return Err(ProbError::InvalidProbability(p_a));
        }
        if p_b <= 0.0 || p_b > 1.0 {
            return Err(ProbError::DomainError("P(B) must be in (0, 1]".into()));
        }
        let posterior = (p_b_given_a * p_a) / p_b;
        Ok(posterior.clamp(0.0, 1.0))
    }

    /// Bayes' Theorem with Total Probability partition:
    /// P(A_i|B) = (P(B|A_i) * P(A_i)) / sum_j (P(B|A_j) * P(A_j)).
    pub fn bayes_partition(
        priors: &[f64],
        likelihoods: &[f64],
        target_idx: usize,
    ) -> Result<f64, ProbError> {
        if priors.len() != likelihoods.len() {
            return Err(ProbError::DomainError(
                "priors and likelihoods must have matching length".into(),
            ));
        }
        if target_idx >= priors.len() {
            return Err(ProbError::DomainError("target index out of bounds".into()));
        }

        let mut total_p_b = 0.0;
        for (&p, &l) in priors.iter().zip(likelihoods.iter()) {
            if !(0.0..=1.0).contains(&p) {
                return Err(ProbError::InvalidProbability(p));
            }
            if !(0.0..=1.0).contains(&l) {
                return Err(ProbError::InvalidProbability(l));
            }
            total_p_b += p * l;
        }

        if total_p_b <= 0.0 {
            return Err(ProbError::DivisionByZero);
        }

        let posterior = (likelihoods[target_idx] * priors[target_idx]) / total_p_b;
        Ok(posterior.clamp(0.0, 1.0))
    }
}
