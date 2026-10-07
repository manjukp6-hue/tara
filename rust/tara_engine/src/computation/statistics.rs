//! Descriptive and bivariate statistical computation engine for TARA Native Engine.
//!
//! Provides:
//! - Univariate statistics: mean, median, mode, variance (sample & population),
//!   standard deviation, skewness, quantiles, quartiles, IQR, min, max, sum.
//! - Bivariate statistics: covariance, Pearson correlation coefficient (r),
//!   ordinary least squares (OLS) linear regression (slope, intercept, R^2).

use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum StatsError {
    #[error("dataset is empty")]
    EmptyDataset,
    #[error("insufficient data points (requires at least {min_needed}, got {actual})")]
    InsufficientData { min_needed: usize, actual: usize },
    #[error("paired datasets have unequal lengths ({len1} vs {len2})")]
    UnequalLengths { len1: usize, len2: usize },
    #[error("variance or standard deviation is zero (constant values)")]
    ZeroVariance,
}

/// Univariate descriptive statistics summary.
#[derive(Debug, Clone, PartialEq)]
pub struct DescriptiveStats {
    pub count: usize,
    pub sum: f64,
    pub mean: f64,
    pub median: f64,
    pub mode: Option<f64>,
    pub min: f64,
    pub max: f64,
    pub range: f64,
    pub variance_pop: f64,
    pub variance_sample: f64,
    pub std_dev_pop: f64,
    pub std_dev_sample: f64,
    pub skewness: f64,
    pub q1: f64,
    pub q3: f64,
    pub iqr: f64,
}

/// Statistics computation engine.
pub struct Statistics;

impl Statistics {
    pub fn sum(data: &[f64]) -> f64 {
        data.iter().sum()
    }

    pub fn mean(data: &[f64]) -> Result<f64, StatsError> {
        if data.is_empty() {
            return Err(StatsError::EmptyDataset);
        }
        Ok(Self::sum(data) / data.len() as f64)
    }

    pub fn median(data: &[f64]) -> Result<f64, StatsError> {
        if data.is_empty() {
            return Err(StatsError::EmptyDataset);
        }
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = sorted.len();
        if n % 2 == 1 {
            Ok(sorted[n / 2])
        } else {
            Ok((sorted[n / 2 - 1] + sorted[n / 2]) / 2.0)
        }
    }

    pub fn mode(data: &[f64]) -> Result<Option<f64>, StatsError> {
        if data.is_empty() {
            return Err(StatsError::EmptyDataset);
        }
        let mut counts: HashMap<i64, (usize, f64)> = HashMap::new();
        for &v in data {
            // Quantize to fixed precision key for float hashing
            let key = (v * 1_000_000.0).round() as i64;
            let entry = counts.entry(key).or_insert((0, v));
            entry.0 += 1;
        }

        let mut max_freq = 0;
        let mut best_val = None;
        for (freq, val) in counts.values() {
            if *freq > max_freq {
                max_freq = *freq;
                best_val = Some(*val);
            }
        }
        if max_freq > 1 {
            Ok(best_val)
        } else {
            Ok(None)
        }
    }

    pub fn variance_pop(data: &[f64]) -> Result<f64, StatsError> {
        if data.is_empty() {
            return Err(StatsError::EmptyDataset);
        }
        let m = Self::mean(data)?;
        let sum_sq_diff: f64 = data.iter().map(|&x| (x - m) * (x - m)).sum();
        Ok(sum_sq_diff / data.len() as f64)
    }

    pub fn variance_sample(data: &[f64]) -> Result<f64, StatsError> {
        if data.len() < 2 {
            return Err(StatsError::InsufficientData {
                min_needed: 2,
                actual: data.len(),
            });
        }
        let m = Self::mean(data)?;
        let sum_sq_diff: f64 = data.iter().map(|&x| (x - m) * (x - m)).sum();
        Ok(sum_sq_diff / (data.len() - 1) as f64)
    }

    pub fn std_dev_pop(data: &[f64]) -> Result<f64, StatsError> {
        Ok(Self::variance_pop(data)?.sqrt())
    }

    pub fn std_dev_sample(data: &[f64]) -> Result<f64, StatsError> {
        Ok(Self::variance_sample(data)?.sqrt())
    }

    pub fn quantile(data: &[f64], q: f64) -> Result<f64, StatsError> {
        if data.is_empty() {
            return Err(StatsError::EmptyDataset);
        }
        let q = q.clamp(0.0, 1.0);
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let index = q * (sorted.len() - 1) as f64;
        let lower = index.floor() as usize;
        let upper = index.ceil() as usize;
        let weight = index - lower as f64;

        Ok(sorted[lower] * (1.0 - weight) + sorted[upper] * weight)
    }

    pub fn describe(data: &[f64]) -> Result<DescriptiveStats, StatsError> {
        if data.is_empty() {
            return Err(StatsError::EmptyDataset);
        }
        let count = data.len();
        let sum = Self::sum(data);
        let mean = sum / count as f64;
        let median = Self::median(data)?;
        let mode = Self::mode(data)?;

        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for &x in data {
            if x < min {
                min = x;
            }
            if x > max {
                max = x;
            }
        }
        let range = max - min;

        let var_pop = Self::variance_pop(data)?;
        let std_pop = var_pop.sqrt();

        let (var_sample, std_sample) = if count > 1 {
            let vs = Self::variance_sample(data)?;
            (vs, vs.sqrt())
        } else {
            (0.0, 0.0)
        };

        // Skewness: m3 / s^3
        let skewness = if std_pop > 1e-15 {
            let m3: f64 = data.iter().map(|&x| (x - mean).powi(3)).sum::<f64>() / count as f64;
            m3 / std_pop.powi(3)
        } else {
            0.0
        };

        let q1 = Self::quantile(data, 0.25)?;
        let q3 = Self::quantile(data, 0.75)?;
        let iqr = q3 - q1;

        Ok(DescriptiveStats {
            count,
            sum,
            mean,
            median,
            mode,
            min,
            max,
            range,
            variance_pop: var_pop,
            variance_sample: var_sample,
            std_dev_pop: std_pop,
            std_dev_sample: std_sample,
            skewness,
            q1,
            q3,
            iqr,
        })
    }

    /// Sample covariance between paired datasets x and y.
    pub fn covariance(x: &[f64], y: &[f64]) -> Result<f64, StatsError> {
        if x.len() != y.len() {
            return Err(StatsError::UnequalLengths {
                len1: x.len(),
                len2: y.len(),
            });
        }
        if x.len() < 2 {
            return Err(StatsError::InsufficientData {
                min_needed: 2,
                actual: x.len(),
            });
        }
        let mean_x = Self::mean(x)?;
        let mean_y = Self::mean(y)?;

        let cov_sum: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
            .sum();

        Ok(cov_sum / (x.len() - 1) as f64)
    }

    /// Pearson correlation coefficient r in [-1.0, 1.0].
    pub fn correlation(x: &[f64], y: &[f64]) -> Result<f64, StatsError> {
        let cov = Self::covariance(x, y)?;
        let sx = Self::std_dev_sample(x)?;
        let sy = Self::std_dev_sample(y)?;

        if sx < 1e-15 || sy < 1e-15 {
            return Err(StatsError::ZeroVariance);
        }
        let r = cov / (sx * sy);
        Ok(r.clamp(-1.0, 1.0))
    }

    /// Linear regression y = slope * x + intercept.
    pub fn linear_regression(x: &[f64], y: &[f64]) -> Result<(f64, f64, f64), StatsError> {
        let cov = Self::covariance(x, y)?;
        let var_x = Self::variance_sample(x)?;
        if var_x < 1e-15 {
            return Err(StatsError::ZeroVariance);
        }
        let slope = cov / var_x;
        let mean_x = Self::mean(x)?;
        let mean_y = Self::mean(y)?;
        let intercept = mean_y - slope * mean_x;
        let r = Self::correlation(x, y)?;
        let r_squared = r * r;
        Ok((slope, intercept, r_squared))
    }
}
