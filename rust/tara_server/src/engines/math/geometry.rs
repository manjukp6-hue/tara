//! Geometric and coordinate geometry computations for MathEngine.

use super::constants::PI;
use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum GeometryError {
    #[error("geometric dimension must be non-negative: {param} = {val}")]
    NegativeDimension { param: String, val: f64 },
    #[error("triangle inequality violated: sides ({a}, {b}, {c}) cannot form a triangle")]
    TriangleInequalityViolated { a: f64, b: f64, c: f64 },
    #[error("division by zero or vertical line (undefined slope)")]
    UndefinedSlope,
}

pub struct Geometry;

impl Geometry {
    // ── 2D Shapes ────────────────────────────────────────────────────────────

    pub fn circle_perimeter(radius: f64) -> Result<f64, GeometryError> {
        if radius < 0.0 {
            return Err(GeometryError::NegativeDimension {
                param: "radius".into(),
                val: radius,
            });
        }
        Ok(2.0 * PI * radius)
    }

    pub fn circle_area(radius: f64) -> Result<f64, GeometryError> {
        if radius < 0.0 {
            return Err(GeometryError::NegativeDimension {
                param: "radius".into(),
                val: radius,
            });
        }
        Ok(PI * radius * radius)
    }

    pub fn rectangle_perimeter(width: f64, height: f64) -> Result<f64, GeometryError> {
        if width < 0.0 || height < 0.0 {
            return Err(GeometryError::NegativeDimension {
                param: "width/height".into(),
                val: width.min(height),
            });
        }
        Ok(2.0 * (width + height))
    }

    pub fn rectangle_area(width: f64, height: f64) -> Result<f64, GeometryError> {
        if width < 0.0 || height < 0.0 {
            return Err(GeometryError::NegativeDimension {
                param: "width/height".into(),
                val: width.min(height),
            });
        }
        Ok(width * height)
    }

    /// Triangle area via Heron's formula given three side lengths.
    pub fn triangle_area_heron(a: f64, b: f64, c: f64) -> Result<f64, GeometryError> {
        if a <= 0.0 || b <= 0.0 || c <= 0.0 {
            return Err(GeometryError::NegativeDimension {
                param: "side".into(),
                val: a.min(b).min(c),
            });
        }
        if a + b <= c || a + c <= b || b + c <= a {
            return Err(GeometryError::TriangleInequalityViolated { a, b, c });
        }
        let s = (a + b + c) / 2.0;
        let area_sq = s * (s - a) * (s - b) * (s - c);
        Ok(area_sq.max(0.0).sqrt())
    }

    pub fn regular_polygon_area(n_sides: usize, side_length: f64) -> Result<f64, GeometryError> {
        if n_sides < 3 {
            return Err(GeometryError::NegativeDimension {
                param: "n_sides".into(),
                val: n_sides as f64,
            });
        }
        if side_length < 0.0 {
            return Err(GeometryError::NegativeDimension {
                param: "side_length".into(),
                val: side_length,
            });
        }
        let n = n_sides as f64;
        let s = side_length;
        let area = (n * s * s) / (4.0 * (PI / n).tan());
        Ok(area)
    }

    // ── 3D Solids ────────────────────────────────────────────────────────────

    pub fn sphere_volume(radius: f64) -> Result<f64, GeometryError> {
        if radius < 0.0 {
            return Err(GeometryError::NegativeDimension {
                param: "radius".into(),
                val: radius,
            });
        }
        Ok((4.0 / 3.0) * PI * radius.powi(3))
    }

    pub fn sphere_surface_area(radius: f64) -> Result<f64, GeometryError> {
        if radius < 0.0 {
            return Err(GeometryError::NegativeDimension {
                param: "radius".into(),
                val: radius,
            });
        }
        Ok(4.0 * PI * radius * radius)
    }

    pub fn cylinder_volume(radius: f64, height: f64) -> Result<f64, GeometryError> {
        if radius < 0.0 || height < 0.0 {
            return Err(GeometryError::NegativeDimension {
                param: "radius/height".into(),
                val: radius.min(height),
            });
        }
        Ok(PI * radius * radius * height)
    }

    pub fn cylinder_surface_area(radius: f64, height: f64) -> Result<f64, GeometryError> {
        if radius < 0.0 || height < 0.0 {
            return Err(GeometryError::NegativeDimension {
                param: "radius/height".into(),
                val: radius.min(height),
            });
        }
        // 2*pi*r^2 + 2*pi*r*h
        Ok(2.0 * PI * radius * (radius + height))
    }

    pub fn cone_volume(radius: f64, height: f64) -> Result<f64, GeometryError> {
        if radius < 0.0 || height < 0.0 {
            return Err(GeometryError::NegativeDimension {
                param: "radius/height".into(),
                val: radius.min(height),
            });
        }
        Ok((1.0 / 3.0) * PI * radius * radius * height)
    }

    pub fn cone_surface_area(radius: f64, height: f64) -> Result<f64, GeometryError> {
        if radius < 0.0 || height < 0.0 {
            return Err(GeometryError::NegativeDimension {
                param: "radius/height".into(),
                val: radius.min(height),
            });
        }
        let slant_height = (radius * radius + height * height).sqrt();
        Ok(PI * radius * (radius + slant_height))
    }

    // ── Coordinate Geometry ──────────────────────────────────────────────────

    pub fn distance_2d(x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
        let dx = x2 - x1;
        let dy = y2 - y1;
        (dx * dx + dy * dy).sqrt()
    }

    pub fn midpoint_2d(x1: f64, y1: f64, x2: f64, y2: f64) -> (f64, f64) {
        ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    }

    pub fn slope_2d(x1: f64, y1: f64, x2: f64, y2: f64) -> Result<f64, GeometryError> {
        let dx = x2 - x1;
        if dx.abs() < 1e-15 {
            return Err(GeometryError::UndefinedSlope);
        }
        Ok((y2 - y1) / dx)
    }
}
