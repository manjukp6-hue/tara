//! Astronomy and astrophysics computation engine for ScienceEngine.

use super::constants::{GRAVITATIONAL_CONSTANT, HUBBLE_CONSTANT, SPEED_OF_LIGHT, STEFAN_BOLTZMANN};
use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum AstronomyError {
    #[error("mass or radius must be positive: {param} = {val}")]
    InvalidStellarParameter { param: String, val: f64 },
    #[error("division by zero")]
    DivisionByZero,
}

pub struct Astronomy;

impl Astronomy {
    /// Escape velocity from celestial body: v_esc = sqrt(2 * G * M / R).
    pub fn escape_velocity(mass_kg: f64, radius_m: f64) -> Result<f64, AstronomyError> {
        if mass_kg <= 0.0 || radius_m <= 0.0 {
            return Err(AstronomyError::InvalidStellarParameter {
                param: "mass/radius".into(),
                val: mass_kg.min(radius_m),
            });
        }
        let v2 = 2.0 * GRAVITATIONAL_CONSTANT * mass_kg / radius_m;
        Ok(v2.sqrt())
    }

    /// Circular orbital velocity: v_orb = sqrt(G * M / R).
    pub fn orbital_velocity(
        central_mass_kg: f64,
        orbital_radius_m: f64,
    ) -> Result<f64, AstronomyError> {
        if central_mass_kg <= 0.0 || orbital_radius_m <= 0.0 {
            return Err(AstronomyError::InvalidStellarParameter {
                param: "mass/radius".into(),
                val: central_mass_kg.min(orbital_radius_m),
            });
        }
        let v2 = GRAVITATIONAL_CONSTANT * central_mass_kg / orbital_radius_m;
        Ok(v2.sqrt())
    }

    /// Kepler's Third Law orbital period: T = 2 * pi * sqrt(a^3 / (G * (M + m))).
    pub fn orbital_period(
        semi_major_axis_m: f64,
        m1_kg: f64,
        m2_kg: f64,
    ) -> Result<f64, AstronomyError> {
        let total_mass = m1_kg + m2_kg;
        if semi_major_axis_m <= 0.0 || total_mass <= 0.0 {
            return Err(AstronomyError::InvalidStellarParameter {
                param: "semi_major_axis/mass".into(),
                val: semi_major_axis_m.min(total_mass),
            });
        }
        let a3 = semi_major_axis_m.powi(3);
        let period =
            2.0 * std::f64::consts::PI * (a3 / (GRAVITATIONAL_CONSTANT * total_mass)).sqrt();
        Ok(period)
    }

    /// Schwarzschild radius (event horizon radius) of black hole: R_s = 2 * G * M / c^2.
    pub fn schwarzschild_radius(mass_kg: f64) -> Result<f64, AstronomyError> {
        if mass_kg <= 0.0 {
            return Err(AstronomyError::InvalidStellarParameter {
                param: "mass".into(),
                val: mass_kg,
            });
        }
        let c2 = SPEED_OF_LIGHT * SPEED_OF_LIGHT;
        Ok(2.0 * GRAVITATIONAL_CONSTANT * mass_kg / c2)
    }

    /// Stellar luminosity using Stefan-Boltzmann law: L = 4 * pi * R^2 * sigma * T^4.
    pub fn stellar_luminosity(radius_m: f64, effective_temp_k: f64) -> Result<f64, AstronomyError> {
        if radius_m <= 0.0 || effective_temp_k <= 0.0 {
            return Err(AstronomyError::InvalidStellarParameter {
                param: "radius/temperature".into(),
                val: radius_m.min(effective_temp_k),
            });
        }
        let surface_area = 4.0 * std::f64::consts::PI * radius_m * radius_m;
        Ok(surface_area * STEFAN_BOLTZMANN * effective_temp_k.powi(4))
    }

    /// Hubble's Law recession velocity: v = H0 * d (with H0 in km/s/Mpc, d in Mpc).
    pub fn hubble_recession_velocity(distance_mpc: f64) -> f64 {
        HUBBLE_CONSTANT * distance_mpc
    }
}
