//! Linear algebra engine: Vectors and Matrices for TARA Native Engine.
//!
//! Provides:
//! - N-dimensional vector operations: dot product, cross product, L1/L2/Linf norms,
//!   cosine similarity, angle, projection, unit normalization, distance.
//! - M x N matrix operations: addition, subtraction, matrix multiplication, transpose,
//!   trace, determinant, inverse via Gauss-Jordan elimination with partial pivoting,
//!   linear system solver (Ax = b), and eigenvalues/eigenvectors.

use std::fmt;
use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum LinAlgError {
    #[error("dimension mismatch: {expected:?} vs {actual:?}")]
    DimensionMismatch { expected: String, actual: String },
    #[error("matrix is singular and cannot be inverted (determinant is zero)")]
    SingularMatrix,
    #[error("non-square matrix for operation requiring square matrix: {rows}x{cols}")]
    NonSquareMatrix { rows: usize, cols: usize },
    #[error("cross product requires 3-dimensional vectors (got {dim})")]
    CrossProductDimError { dim: usize },
    #[error("vector has zero magnitude (cannot normalize)")]
    ZeroMagnitudeVector,
    #[error("linear system has no unique solution")]
    NoUniqueSolution,
}

/// N-dimensional numerical vector.
#[derive(Clone, PartialEq, Debug)]
pub struct Vector {
    pub data: Vec<f64>,
}

impl Vector {
    pub fn new(data: Vec<f64>) -> Self {
        Self { data }
    }

    pub fn zeros(dim: usize) -> Self {
        Self {
            data: vec![0.0; dim],
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn add(&self, other: &Self) -> Result<Self, LinAlgError> {
        if self.len() != other.len() {
            return Err(LinAlgError::DimensionMismatch {
                expected: format!("{}", self.len()),
                actual: format!("{}", other.len()),
            });
        }
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Ok(Self { data })
    }

    pub fn sub(&self, other: &Self) -> Result<Self, LinAlgError> {
        if self.len() != other.len() {
            return Err(LinAlgError::DimensionMismatch {
                expected: format!("{}", self.len()),
                actual: format!("{}", other.len()),
            });
        }
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a - b)
            .collect();
        Ok(Self { data })
    }

    pub fn scale(&self, scalar: f64) -> Self {
        let data = self.data.iter().map(|v| v * scalar).collect();
        Self { data }
    }

    /// Dot product (inner product) self . other.
    pub fn dot(&self, other: &Self) -> Result<f64, LinAlgError> {
        if self.len() != other.len() {
            return Err(LinAlgError::DimensionMismatch {
                expected: format!("{}", self.len()),
                actual: format!("{}", other.len()),
            });
        }
        let sum = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum();
        Ok(sum)
    }

    /// 3D Cross product: self x other.
    pub fn cross(&self, other: &Self) -> Result<Self, LinAlgError> {
        if self.len() != 3 {
            return Err(LinAlgError::CrossProductDimError { dim: self.len() });
        }
        if other.len() != 3 {
            return Err(LinAlgError::CrossProductDimError { dim: other.len() });
        }
        let a = &self.data;
        let b = &other.data;
        let c0 = a[1] * b[2] - a[2] * b[1];
        let c1 = a[2] * b[0] - a[0] * b[2];
        let c2 = a[0] * b[1] - a[1] * b[0];
        Ok(Self::new(vec![c0, c1, c2]))
    }

    /// Euclidean L2 norm ||v||_2.
    pub fn norm_l2(&self) -> f64 {
        self.data.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    /// Manhattan L1 norm ||v||_1.
    pub fn norm_l1(&self) -> f64 {
        self.data.iter().map(|v| v.abs()).sum::<f64>()
    }

    /// Chebyshev infinity norm ||v||_inf.
    pub fn norm_inf(&self) -> f64 {
        self.data.iter().map(|v| v.abs()).fold(0.0, f64::max)
    }

    /// Normalized unit vector v / ||v||.
    pub fn normalize(&self) -> Result<Self, LinAlgError> {
        let norm = self.norm_l2();
        if norm < 1e-15 {
            return Err(LinAlgError::ZeroMagnitudeVector);
        }
        Ok(self.scale(1.0 / norm))
    }

    /// Euclidean distance between two vectors.
    pub fn distance(&self, other: &Self) -> Result<f64, LinAlgError> {
        Ok(self.sub(other)?.norm_l2())
    }

    /// Angle between two vectors in radians.
    pub fn angle(&self, other: &Self) -> Result<f64, LinAlgError> {
        let dot = self.dot(other)?;
        let n1 = self.norm_l2();
        let n2 = other.norm_l2();
        if n1 < 1e-15 || n2 < 1e-15 {
            return Err(LinAlgError::ZeroMagnitudeVector);
        }
        let cos_theta = (dot / (n1 * n2)).clamp(-1.0, 1.0);
        Ok(cos_theta.acos())
    }

    /// Vector projection of self onto other: proj_other(self).
    pub fn project_onto(&self, other: &Self) -> Result<Self, LinAlgError> {
        let n2 = other.norm_l2();
        if n2 < 1e-15 {
            return Err(LinAlgError::ZeroMagnitudeVector);
        }
        let dot = self.dot(other)?;
        let scalar = dot / (n2 * n2);
        Ok(other.scale(scalar))
    }
}

/// M x N Matrix represented as row-major 2D numerical array.
#[derive(Clone, PartialEq, Debug)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self, LinAlgError> {
        if data.len() != rows * cols {
            return Err(LinAlgError::DimensionMismatch {
                expected: format!("{} elements", rows * cols),
                actual: format!("{} elements", data.len()),
            });
        }
        Ok(Self { rows, cols, data })
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn identity(n: usize) -> Self {
        let mut mat = Self::zeros(n, n);
        for i in 0..n {
            mat.data[i * n + i] = 1.0;
        }
        mat
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    #[inline]
    pub fn set(&mut self, r: usize, c: usize, val: f64) {
        self.data[r * self.cols + c] = val;
    }

    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    pub fn add(&self, other: &Self) -> Result<Self, LinAlgError> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(LinAlgError::DimensionMismatch {
                expected: format!("{}x{}", self.rows, self.cols),
                actual: format!("{}x{}", other.rows, other.cols),
            });
        }
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Ok(Self {
            rows: self.rows,
            cols: self.cols,
            data,
        })
    }

    pub fn sub(&self, other: &Self) -> Result<Self, LinAlgError> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(LinAlgError::DimensionMismatch {
                expected: format!("{}x{}", self.rows, self.cols),
                actual: format!("{}x{}", other.rows, other.cols),
            });
        }
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a - b)
            .collect();
        Ok(Self {
            rows: self.rows,
            cols: self.cols,
            data,
        })
    }

    pub fn scale(&self, scalar: f64) -> Self {
        let data = self.data.iter().map(|v| v * scalar).collect();
        Self {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }

    /// Matrix multiplication C = A * B.
    pub fn mul(&self, other: &Self) -> Result<Self, LinAlgError> {
        if self.cols != other.rows {
            return Err(LinAlgError::DimensionMismatch {
                expected: format!("columns={} matching rows={}", self.cols, other.rows),
                actual: format!("columns={} vs rows={}", self.cols, other.rows),
            });
        }
        let mut result = Self::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a_ik = self.get(i, k);
                for j in 0..other.cols {
                    let cur = result.get(i, j);
                    result.set(i, j, cur + a_ik * other.get(k, j));
                }
            }
        }
        Ok(result)
    }

    /// Multiply matrix by vector y = A * x.
    pub fn mul_vector(&self, vec: &Vector) -> Result<Vector, LinAlgError> {
        if self.cols != vec.len() {
            return Err(LinAlgError::DimensionMismatch {
                expected: format!("columns={}", self.cols),
                actual: format!("vector_dim={}", vec.len()),
            });
        }
        let mut out = Vec::with_capacity(self.rows);
        for i in 0..self.rows {
            let mut sum = 0.0;
            for j in 0..self.cols {
                sum += self.get(i, j) * vec.data[j];
            }
            out.push(sum);
        }
        Ok(Vector::new(out))
    }

    /// Matrix transpose A^T.
    pub fn transpose(&self) -> Self {
        let mut result = Self::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(j, i, self.get(i, j));
            }
        }
        result
    }

    /// Trace of square matrix sum(A_ii).
    pub fn trace(&self) -> Result<f64, LinAlgError> {
        if !self.is_square() {
            return Err(LinAlgError::NonSquareMatrix {
                rows: self.rows,
                cols: self.cols,
            });
        }
        let mut sum = 0.0;
        for i in 0..self.rows {
            sum += self.get(i, i);
        }
        Ok(sum)
    }

    /// Determinant via Gaussian elimination with partial pivoting.
    pub fn determinant(&self) -> Result<f64, LinAlgError> {
        if !self.is_square() {
            return Err(LinAlgError::NonSquareMatrix {
                rows: self.rows,
                cols: self.cols,
            });
        }
        let n = self.rows;
        if n == 1 {
            return Ok(self.get(0, 0));
        }
        if n == 2 {
            return Ok(self.get(0, 0) * self.get(1, 1) - self.get(0, 1) * self.get(1, 0));
        }

        let mut a = self.clone();
        let mut det = 1.0;

        for i in 0..n {
            // Find pivot
            let mut pivot_row = i;
            let mut max_val = a.get(i, i).abs();
            for k in (i + 1)..n {
                let val = a.get(k, i).abs();
                if val > max_val {
                    max_val = val;
                    pivot_row = k;
                }
            }

            if max_val < 1e-15 {
                return Ok(0.0);
            }

            if pivot_row != i {
                // Swap rows
                for col in 0..n {
                    let tmp = a.get(i, col);
                    a.set(i, col, a.get(pivot_row, col));
                    a.set(pivot_row, col, tmp);
                }
                det = -det;
            }

            let pivot = a.get(i, i);
            det *= pivot;

            for row in (i + 1)..n {
                let factor = a.get(row, i) / pivot;
                for col in i..n {
                    let val = a.get(row, col) - factor * a.get(i, col);
                    a.set(row, col, val);
                }
            }
        }
        Ok(det)
    }

    /// Matrix inverse A^(-1) via Gauss-Jordan elimination with partial pivoting.
    pub fn inverse(&self) -> Result<Self, LinAlgError> {
        if !self.is_square() {
            return Err(LinAlgError::NonSquareMatrix {
                rows: self.rows,
                cols: self.cols,
            });
        }
        let n = self.rows;
        let mut aug = Self::zeros(n, 2 * n);

        for i in 0..n {
            for j in 0..n {
                aug.set(i, j, self.get(i, j));
            }
            aug.set(i, n + i, 1.0);
        }

        for i in 0..n {
            // Find pivot
            let mut pivot_row = i;
            let mut max_val = aug.get(i, i).abs();
            for k in (i + 1)..n {
                let val = aug.get(k, i).abs();
                if val > max_val {
                    max_val = val;
                    pivot_row = k;
                }
            }

            if max_val < 1e-14 {
                return Err(LinAlgError::SingularMatrix);
            }

            if pivot_row != i {
                for col in 0..(2 * n) {
                    let tmp = aug.get(i, col);
                    aug.set(i, col, aug.get(pivot_row, col));
                    aug.set(pivot_row, col, tmp);
                }
            }

            let pivot = aug.get(i, i);
            for col in 0..(2 * n) {
                let val = aug.get(i, col) / pivot;
                aug.set(i, col, val);
            }

            for row in 0..n {
                if row != i {
                    let factor = aug.get(row, i);
                    for col in 0..(2 * n) {
                        let val = aug.get(row, col) - factor * aug.get(i, col);
                        aug.set(row, col, val);
                    }
                }
            }
        }

        let mut inv = Self::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                inv.set(i, j, aug.get(i, n + j));
            }
        }
        Ok(inv)
    }

    /// Solve linear system A * x = b.
    pub fn solve(&self, b: &Vector) -> Result<Vector, LinAlgError> {
        let inv = self.inverse()?;
        inv.mul_vector(b)
    }

    /// Compute eigenvalues of 2x2 matrix: det(A - lambda*I) = 0.
    pub fn eigenvalues_2x2(&self) -> Result<(f64, f64), LinAlgError> {
        if self.rows != 2 || self.cols != 2 {
            return Err(LinAlgError::DimensionMismatch {
                expected: "2x2".into(),
                actual: format!("{}x{}", self.rows, self.cols),
            });
        }
        let tr = self.trace()?;
        let det = self.determinant()?;
        let disc = tr * tr - 4.0 * det;
        if disc < 0.0 {
            return Err(LinAlgError::DimensionMismatch {
                expected: "real eigenvalues".into(),
                actual: "complex eigenvalues".into(),
            });
        }
        let sqrt_disc = disc.sqrt();
        let lambda1 = (tr + sqrt_disc) / 2.0;
        let lambda2 = (tr - sqrt_disc) / 2.0;
        Ok((lambda1, lambda2))
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "[{}x{} Matrix]:", self.rows, self.cols)?;
        for i in 0..self.rows {
            write!(f, "  [")?;
            for j in 0..self.cols {
                write!(f, "{:8.4}", self.get(i, j))?;
                if j + 1 < self.cols {
                    write!(f, ", ")?;
                }
            }
            writeln!(f, "]")?;
        }
        Ok(())
    }
}
