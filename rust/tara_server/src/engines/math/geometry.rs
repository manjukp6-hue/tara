//! Geometric and coordinate geometry computations for MathEngine.

use super::constants::PI;
use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum GeometryError {
    #[error("geometric dimension must be finite and non-negative: {param} = {val}")]
    NegativeDimension { param: String, val: f64 },
    #[error("triangle inequality violated: sides ({a}, {b}, {c}) cannot form a triangle")]
    TriangleInequalityViolated { a: f64, b: f64, c: f64 },
    #[error("division by zero or vertical line (undefined slope)")]
    UndefinedSlope,
}

#[inline]
fn require_non_negative_finite(param: &str, val: f64) -> Result<(), GeometryError> {
    if !val.is_finite() || val < 0.0 {
        return Err(GeometryError::NegativeDimension {
            param: param.into(),
            val,
        });
    }
    Ok(())
}

pub struct Geometry;

impl Geometry {
    // ── 2D Shapes ────────────────────────────────────────────────────────────

    pub fn circle_perimeter(radius: f64) -> Result<f64, GeometryError> {
        require_non_negative_finite("radius", radius)?;
        Ok(2.0 * PI * radius)
    }

    pub fn circle_area(radius: f64) -> Result<f64, GeometryError> {
        require_non_negative_finite("radius", radius)?;
        Ok(PI * radius * radius)
    }

    pub fn rectangle_perimeter(width: f64, height: f64) -> Result<f64, GeometryError> {
        require_non_negative_finite("width", width)?;
        require_non_negative_finite("height", height)?;
        Ok(2.0 * (width + height))
    }

    pub fn rectangle_area(width: f64, height: f64) -> Result<f64, GeometryError> {
        require_non_negative_finite("width", width)?;
        require_non_negative_finite("height", height)?;
        Ok(width * height)
    }

    /// Triangle area via Heron's formula given three side lengths.
    pub fn triangle_area_heron(a: f64, b: f64, c: f64) -> Result<f64, GeometryError> {
        if !a.is_finite() || !b.is_finite() || !c.is_finite() || a <= 0.0 || b <= 0.0 || c <= 0.0 {
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
        require_non_negative_finite("side_length", side_length)?;
        let n = n_sides as f64;
        let s = side_length;
        let area = (n * s * s) / (4.0 * (PI / n).tan());
        Ok(area)
    }

    // ── 3D Solids ────────────────────────────────────────────────────────────

    pub fn sphere_volume(radius: f64) -> Result<f64, GeometryError> {
        require_non_negative_finite("radius", radius)?;
        Ok((4.0 / 3.0) * PI * radius.powi(3))
    }

    pub fn sphere_surface_area(radius: f64) -> Result<f64, GeometryError> {
        require_non_negative_finite("radius", radius)?;
        Ok(4.0 * PI * radius * radius)
    }

    pub fn cylinder_volume(radius: f64, height: f64) -> Result<f64, GeometryError> {
        require_non_negative_finite("radius", radius)?;
        require_non_negative_finite("height", height)?;
        Ok(PI * radius * radius * height)
    }

    pub fn cylinder_surface_area(radius: f64, height: f64) -> Result<f64, GeometryError> {
        require_non_negative_finite("radius", radius)?;
        require_non_negative_finite("height", height)?;
        Ok(2.0 * PI * radius * (radius + height))
    }

    pub fn cone_volume(radius: f64, height: f64) -> Result<f64, GeometryError> {
        require_non_negative_finite("radius", radius)?;
        require_non_negative_finite("height", height)?;
        Ok((1.0 / 3.0) * PI * radius * radius * height)
    }

    pub fn cone_surface_area(radius: f64, height: f64) -> Result<f64, GeometryError> {
        require_non_negative_finite("radius", radius)?;
        require_non_negative_finite("height", height)?;
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
        if !x1.is_finite() || !y1.is_finite() || !x2.is_finite() || !y2.is_finite() {
            return Err(GeometryError::UndefinedSlope);
        }
        let dx = x2 - x1;
        if dx.abs() < 1e-15 {
            return Err(GeometryError::UndefinedSlope);
        }
        Ok((y2 - y1) / dx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_finite_and_non_negative_validation() {
        assert!(Geometry::circle_perimeter(-1.0).is_err());
        assert!(Geometry::circle_perimeter(f64::NAN).is_err());
        assert!(Geometry::circle_perimeter(f64::INFINITY).is_err());
        assert!(Geometry::circle_perimeter(5.0).is_ok());

        assert!(Geometry::rectangle_perimeter(10.0, f64::NAN).is_err());
        assert!(Geometry::rectangle_perimeter(10.0, f64::INFINITY).is_err());
        assert!(Geometry::rectangle_perimeter(10.0, 5.0).is_ok());

        assert!(Geometry::sphere_volume(f64::NAN).is_err());
        assert!(Geometry::cylinder_volume(2.0, -1.0).is_err());
        assert!(Geometry::cone_surface_area(-2.0, 5.0).is_err());
    }

    #[test]
    fn test_triangle_heron_and_slope() {
        // Valid 3-4-5 right triangle: area is 6.0
        let area = Geometry::triangle_area_heron(3.0, 4.0, 5.0).unwrap();
        assert!((area - 6.0).abs() < 1e-10);

        // Degenerate / invalid triangle
        assert!(Geometry::triangle_area_heron(1.0, 2.0, 10.0).is_err());
        assert!(Geometry::triangle_area_heron(f64::NAN, 4.0, 5.0).is_err());

        // Slope 2D
        let slope = Geometry::slope_2d(0.0, 0.0, 2.0, 4.0).unwrap();
        assert!((slope - 2.0).abs() < 1e-10);

        // Vertical line
        assert!(Geometry::slope_2d(1.0, 0.0, 1.0, 5.0).is_err());

        // NaN slope
        assert!(Geometry::slope_2d(f64::NAN, 0.0, 1.0, 5.0).is_err());
    }
}
