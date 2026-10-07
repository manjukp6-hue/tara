//! Earth, atmospheric, and planetary geophysics computations for ScienceEngine.

use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum EarthScienceError {
    #[error("depth or height out of range: {val} km")]
    InvalidAltitude { val: f64 },
    #[error("division by zero")]
    DivisionByZero,
}

pub struct EarthScience;

impl EarthScience {
    /// Atmospheric pressure at altitude h (m) using barometric formula:
    /// P(h) = P0 * exp(-h / H), where H ~= 8400 m scale height.
    pub fn barometric_pressure(altitude_m: f64) -> Result<f64, EarthScienceError> {
        if altitude_m < -500.0 || altitude_m > 100_000.0 {
            return Err(EarthScienceError::InvalidAltitude {
                val: altitude_m / 1000.0,
            });
        }
        let p0 = 101325.0; // Sea level Pa
        let h_scale = 8400.0; // Scale height meters
        Ok(p0 * (-altitude_m / h_scale).exp())
    }

    /// Earth geothermal gradient temperature at depth z (meters):
    /// T(z) = T_surface + (gradient_K_per_km * z / 1000).
    /// Typical gradient: 25 K/km.
    pub fn geothermal_temperature(depth_m: f64, surface_temp_c: f64, gradient_per_km: f64) -> f64 {
        surface_temp_c + gradient_per_km * (depth_m / 1000.0)
    }

    /// Seismic wave arrival time difference (S - P wave) to estimate epicenter distance:
    /// distance = (t_S - t_P) / (1/v_S - 1/v_P).
    /// Standard continental crust: v_P ~= 6.0 km/s, v_S ~= 3.5 km/s.
    pub fn seismic_epicenter_distance(
        delta_t_seconds: f64,
        v_p_kms: f64,
        v_s_kms: f64,
    ) -> Result<f64, EarthScienceError> {
        if v_p_kms <= v_s_kms || v_s_kms <= 0.0 {
            return Err(EarthScienceError::DivisionByZero);
        }
        let denom = (1.0 / v_s_kms) - (1.0 / v_p_kms);
        Ok(delta_t_seconds / denom)
    }

    /// Coriolis acceleration magnitude: a_c = 2 * omega * v * sin(latitude_rad).
    /// Earth angular velocity omega = 7.2921159e-5 rad/s.
    pub fn coriolis_acceleration(velocity_mps: f64, latitude_deg: f64) -> f64 {
        let omega = 7.2921159e-5;
        let lat_rad = latitude_deg.to_radians();
        2.0 * omega * velocity_mps * lat_rad.sin()
    }
}
