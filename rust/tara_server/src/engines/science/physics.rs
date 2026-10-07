//! Physics computation engine for ScienceEngine.

use super::constants::{GRAVITATIONAL_CONSTANT, SPEED_OF_LIGHT, STANDARD_GRAVITY};
use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum PhysicsError {
    #[error("division by zero or zero denominator: {0}")]
    DivisionByZero(String),
    #[error("negative mass or invalid physical quantity: {param} = {val}")]
    InvalidPhysicalQuantity { param: String, val: f64 },
    #[error("velocity exceeds or equals speed of light: v = {val} m/s (c = 299792458 m/s)")]
    SuperluminalVelocity { val: f64 },
}

pub struct Physics;

impl Physics {
    // ── Kinematics ────────────────────────────────────────────────────────────

    /// Average speed: v = distance / time.
    pub fn velocity(distance: f64, time: f64) -> Result<f64, PhysicsError> {
        if time <= 0.0 {
            return Err(PhysicsError::DivisionByZero("time must be > 0".into()));
        }
        Ok(distance / time)
    }

    /// Acceleration: a = (v - u) / t.
    pub fn acceleration(initial_v: f64, final_v: f64, time: f64) -> Result<f64, PhysicsError> {
        if time <= 0.0 {
            return Err(PhysicsError::DivisionByZero("time must be > 0".into()));
        }
        Ok((final_v - initial_v) / time)
    }

    /// Displacement under constant acceleration: s = u*t + 0.5*a*t^2.
    pub fn displacement(initial_v: f64, acceleration: f64, time: f64) -> f64 {
        initial_v * time + 0.5 * acceleration * time * time
    }

    /// Final velocity squared: v^2 = u^2 + 2*a*s => v = sqrt(u^2 + 2as).
    pub fn final_velocity_kinematics(
        initial_v: f64,
        acceleration: f64,
        displacement: f64,
    ) -> Result<f64, PhysicsError> {
        let v2 = initial_v * initial_v + 2.0 * acceleration * displacement;
        if v2 < 0.0 {
            return Err(PhysicsError::InvalidPhysicalQuantity {
                param: "v^2".into(),
                val: v2,
            });
        }
        Ok(v2.sqrt())
    }

    // ── Dynamics & Forces ────────────────────────────────────────────────────

    /// Newton's second law: F = m * a.
    pub fn force(mass: f64, acceleration: f64) -> Result<f64, PhysicsError> {
        if mass < 0.0 {
            return Err(PhysicsError::InvalidPhysicalQuantity {
                param: "mass".into(),
                val: mass,
            });
        }
        Ok(mass * acceleration)
    }

    /// Momentum: p = m * v.
    pub fn momentum(mass: f64, velocity: f64) -> Result<f64, PhysicsError> {
        if mass < 0.0 {
            return Err(PhysicsError::InvalidPhysicalQuantity {
                param: "mass".into(),
                val: mass,
            });
        }
        Ok(mass * velocity)
    }

    /// Newton's law of universal gravitation: F = G * m1 * m2 / r^2.
    pub fn gravitational_force(m1: f64, m2: f64, distance: f64) -> Result<f64, PhysicsError> {
        if m1 < 0.0 || m2 < 0.0 {
            return Err(PhysicsError::InvalidPhysicalQuantity {
                param: "mass".into(),
                val: m1.min(m2),
            });
        }
        if distance <= 0.0 {
            return Err(PhysicsError::DivisionByZero("distance must be > 0".into()));
        }
        Ok((GRAVITATIONAL_CONSTANT * m1 * m2) / (distance * distance))
    }

    // ── Work, Energy & Power ─────────────────────────────────────────────────

    /// Work done: W = F * d * cos(theta).
    pub fn work(force: f64, distance: f64, angle_rad: f64) -> f64 {
        force * distance * angle_rad.cos()
    }

    /// Kinetic energy: K = 0.5 * m * v^2.
    pub fn kinetic_energy(mass: f64, velocity: f64) -> Result<f64, PhysicsError> {
        if mass < 0.0 {
            return Err(PhysicsError::InvalidPhysicalQuantity {
                param: "mass".into(),
                val: mass,
            });
        }
        Ok(0.5 * mass * velocity * velocity)
    }

    /// Gravitational potential energy near Earth: U = m * g * h.
    pub fn gravitational_potential_energy(mass: f64, height: f64) -> Result<f64, PhysicsError> {
        if mass < 0.0 {
            return Err(PhysicsError::InvalidPhysicalQuantity {
                param: "mass".into(),
                val: mass,
            });
        }
        Ok(mass * STANDARD_GRAVITY * height)
    }

    /// Power: P = Work / time.
    pub fn power(work: f64, time: f64) -> Result<f64, PhysicsError> {
        if time <= 0.0 {
            return Err(PhysicsError::DivisionByZero("time must be > 0".into()));
        }
        Ok(work / time)
    }

    // ── Relativity ───────────────────────────────────────────────────────────

    /// Mass-energy equivalence: E = m * c^2.
    pub fn mass_energy_equivalence(mass: f64) -> Result<f64, PhysicsError> {
        if mass < 0.0 {
            return Err(PhysicsError::InvalidPhysicalQuantity {
                param: "mass".into(),
                val: mass,
            });
        }
        Ok(mass * SPEED_OF_LIGHT * SPEED_OF_LIGHT)
    }

    /// Relativistic Lorentz factor: gamma = 1 / sqrt(1 - (v/c)^2).
    pub fn lorentz_factor(velocity: f64) -> Result<f64, PhysicsError> {
        let beta = velocity.abs() / SPEED_OF_LIGHT;
        if beta >= 1.0 {
            return Err(PhysicsError::SuperluminalVelocity { val: velocity });
        }
        Ok(1.0 / (1.0 - beta * beta).sqrt())
    }

    // ── Thermodynamics ───────────────────────────────────────────────────────

    /// Heat transfer: Q = m * c_specific * delta_T.
    pub fn heat_transfer(
        mass: f64,
        specific_heat: f64,
        delta_temp: f64,
    ) -> Result<f64, PhysicsError> {
        if mass < 0.0 || specific_heat < 0.0 {
            return Err(PhysicsError::InvalidPhysicalQuantity {
                param: "mass/specific_heat".into(),
                val: mass.min(specific_heat),
            });
        }
        Ok(mass * specific_heat * delta_temp)
    }

    // ── Fluid Mechanics ──────────────────────────────────────────────────────

    /// Hydrostatic pressure: P = rho * g * h.
    pub fn hydrostatic_pressure(density: f64, depth: f64) -> Result<f64, PhysicsError> {
        if density < 0.0 || depth < 0.0 {
            return Err(PhysicsError::InvalidPhysicalQuantity {
                param: "density/depth".into(),
                val: density.min(depth),
            });
        }
        Ok(density * STANDARD_GRAVITY * depth)
    }

    /// Archimedes' buoyant force: F_b = rho * V * g.
    pub fn buoyant_force(fluid_density: f64, displaced_volume: f64) -> Result<f64, PhysicsError> {
        if fluid_density < 0.0 || displaced_volume < 0.0 {
            return Err(PhysicsError::InvalidPhysicalQuantity {
                param: "density/volume".into(),
                val: fluid_density.min(displaced_volume),
            });
        }
        Ok(fluid_density * displaced_volume * STANDARD_GRAVITY)
    }

    // ── Electricity & Circuits ────────────────────────────────────────────────

    /// Ohm's Law voltage: V = I * R.
    pub fn voltage_ohms_law(current: f64, resistance: f64) -> f64 {
        current * resistance
    }

    /// Electrical power: P = V * I.
    pub fn electrical_power(voltage: f64, current: f64) -> f64 {
        voltage * current
    }

    /// Coulomb's Law electric force: F = k * q1 * q2 / r^2.
    pub fn coulomb_force(q1: f64, q2: f64, distance: f64) -> Result<f64, PhysicsError> {
        if distance <= 0.0 {
            return Err(PhysicsError::DivisionByZero("distance must be > 0".into()));
        }
        let k_e = 8.9875517923e9; // 1 / (4*pi*epsilon_0)
        Ok((k_e * q1 * q2) / (distance * distance))
    }

    // ── Waves ────────────────────────────────────────────────────────────────

    /// Wave speed: v = f * lambda.
    pub fn wave_speed(frequency: f64, wavelength: f64) -> Result<f64, PhysicsError> {
        if frequency < 0.0 || wavelength < 0.0 {
            return Err(PhysicsError::InvalidPhysicalQuantity {
                param: "frequency/wavelength".into(),
                val: frequency.min(wavelength),
            });
        }
        Ok(frequency * wavelength)
    }
}
