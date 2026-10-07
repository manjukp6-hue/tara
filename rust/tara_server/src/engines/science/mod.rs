//! ScienceEngine: Production Native Scientific Specialist Engine for TARA.
//!
//! Provides verifiable calculations across physics, chemistry, biology,
//! earth sciences, astronomy, and formal scientific reasoning.

pub mod astronomy;
pub mod biology;
pub mod chemistry;
pub mod constants;
pub mod earth_science;
pub mod physics;
pub mod scientific_reasoning;

pub use astronomy::{Astronomy, AstronomyError};
pub use biology::{Biology, BiologyError};
pub use chemistry::{ChemicalElement, Chemistry, ChemistryError};
pub use constants::*;
pub use earth_science::{EarthScience, EarthScienceError};
pub use physics::{Physics, PhysicsError};
pub use scientific_reasoning::{
    EpistemicStatus, ScientificInferenceRecord, ScientificReasoningEngine, ScientificVariable,
};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScienceEvaluationResult {
    pub success: bool,
    pub domain: String,
    pub discipline: String,
    pub formula_applied: String,
    pub input_parameters: Value,
    pub calculated_result: Value,
    pub explanation: String,
    pub execution_time_us: u64,
}

pub struct ScienceEngine;

impl Default for ScienceEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl ScienceEngine {
    pub fn new() -> Self {
        Self
    }

    /// Evaluates structured scientific requests with exact physics and chemistry formulas.
    pub fn evaluate(
        &self,
        discipline: &str,
        operation: &str,
        params: &Value,
    ) -> Result<ScienceEvaluationResult, String> {
        let start = std::time::Instant::now();
        let (formula, result) = match discipline {
            "physics" => match operation {
                "force" => {
                    let m = params
                        .get("m")
                        .or_else(|| params.get("mass"))
                        .and_then(Value::as_f64)
                        .ok_or("missing 'm' (mass in kg)")?;
                    let a = params
                        .get("a")
                        .or_else(|| params.get("acceleration"))
                        .and_then(Value::as_f64)
                        .ok_or("missing 'a' (acceleration in m/s^2)")?;
                    let f = Physics::force(m, a).map_err(|e| format!("{:?}", e))?;
                    (
                        "F = m * a",
                        json!({ "force_newtons": f, "mass_kg": m, "acceleration_mps2": a }),
                    )
                }
                "velocity" => {
                    let d = params
                        .get("d")
                        .or_else(|| params.get("distance"))
                        .and_then(Value::as_f64)
                        .ok_or("missing 'd' (distance in m)")?;
                    let t = params
                        .get("t")
                        .or_else(|| params.get("time"))
                        .and_then(Value::as_f64)
                        .ok_or("missing 't' (time in s)")?;
                    let v = Physics::velocity(d, t).map_err(|e| format!("{:?}", e))?;
                    (
                        "v = d / t",
                        json!({ "velocity_mps": v, "distance_m": d, "time_s": t }),
                    )
                }
                "kinetic_energy" => {
                    let m = params
                        .get("m")
                        .and_then(Value::as_f64)
                        .ok_or("missing 'm'")?;
                    let v = params
                        .get("v")
                        .and_then(Value::as_f64)
                        .ok_or("missing 'v'")?;
                    let ke = Physics::kinetic_energy(m, v).map_err(|e| format!("{:?}", e))?;
                    (
                        "K = 0.5 * m * v^2",
                        json!({ "kinetic_energy_joules": ke, "mass_kg": m, "velocity_mps": v }),
                    )
                }
                "power" => {
                    let w = params
                        .get("w")
                        .or_else(|| params.get("work"))
                        .and_then(Value::as_f64)
                        .ok_or("missing 'w' (work in J)")?;
                    let t = params
                        .get("t")
                        .or_else(|| params.get("time"))
                        .and_then(Value::as_f64)
                        .ok_or("missing 't' (time in s)")?;
                    let p = Physics::power(w, t).map_err(|e| format!("{:?}", e))?;
                    (
                        "P = W / t",
                        json!({ "power_watts": p, "work_joules": w, "time_s": t }),
                    )
                }
                "mass_energy" => {
                    let m = params
                        .get("m")
                        .and_then(Value::as_f64)
                        .ok_or("missing 'm' (mass in kg)")?;
                    let e =
                        Physics::mass_energy_equivalence(m).map_err(|err| format!("{:?}", err))?;
                    (
                        "E = m * c^2",
                        json!({ "energy_joules": e, "mass_kg": m, "speed_of_light_mps": SPEED_OF_LIGHT }),
                    )
                }
                "gravitational_force" => {
                    let m1 = params
                        .get("m1")
                        .and_then(Value::as_f64)
                        .ok_or("missing 'm1'")?;
                    let m2 = params
                        .get("m2")
                        .and_then(Value::as_f64)
                        .ok_or("missing 'm2'")?;
                    let r = params
                        .get("r")
                        .and_then(Value::as_f64)
                        .ok_or("missing 'r'")?;
                    let f =
                        Physics::gravitational_force(m1, m2, r).map_err(|e| format!("{:?}", e))?;
                    (
                        "F = G * m1 * m2 / r^2",
                        json!({ "force_newtons": f, "m1_kg": m1, "m2_kg": m2, "distance_m": r }),
                    )
                }
                "projectile" | "projectile_drag" => {
                    let v0 = params
                        .get("v0")
                        .or_else(|| params.get("velocity"))
                        .and_then(Value::as_f64)
                        .ok_or("missing 'v0' (m/s)")?;
                    let angle = params
                        .get("angle_deg")
                        .or_else(|| params.get("angle"))
                        .and_then(Value::as_f64)
                        .unwrap_or(45.0);
                    let mass = params
                        .get("mass_kg")
                        .or_else(|| params.get("mass"))
                        .or_else(|| params.get("m"))
                        .and_then(Value::as_f64)
                        .unwrap_or(1.0);
                    let cd = params
                        .get("drag_coeff")
                        .and_then(Value::as_f64)
                        .unwrap_or(0.47);
                    let area = params
                        .get("area_m2")
                        .and_then(Value::as_f64)
                        .unwrap_or(0.01);
                    let rho = params
                        .get("air_density")
                        .and_then(Value::as_f64)
                        .unwrap_or(1.225);
                    let dt = params.get("dt").and_then(Value::as_f64).unwrap_or(0.01);

                    let rad = angle.to_radians();
                    let mut x = 0.0;
                    let mut y = 0.0;
                    let mut vx = v0 * rad.cos();
                    let mut vy = v0 * rad.sin();
                    let g = 9.80665;
                    let k = 0.5 * rho * cd * area / mass;

                    let mut t = 0.0;
                    let mut max_height = 0.0;
                    let mut trajectory_len = 0;

                    while y >= 0.0 && t < 1000.0 {
                        trajectory_len += 1;
                        if y > max_height {
                            max_height = y;
                        }
                        let v = (vx * vx + vy * vy).sqrt();
                        let ax = -k * v * vx;
                        let ay = -g - k * v * vy;
                        vx += ax * dt;
                        vy += ay * dt;
                        x += vx * dt;
                        y += vy * dt;
                        t += dt;
                    }

                    (
                        "m*d^2r/dt^2 = -m*g*j - 0.5*rho*Cd*A*v*v_vec",
                        json!({
                            "initial_speed_mps": v0,
                            "angle_degrees": angle,
                            "flight_time_seconds": t,
                            "range_meters": x,
                            "max_height_meters": max_height,
                            "final_impact_speed_mps": (vx * vx + vy * vy).sqrt(),
                            "simulated_steps": trajectory_len
                        }),
                    )
                }
                "damped_oscillator" | "harmonic_oscillator" => {
                    let m = params
                        .get("m")
                        .or_else(|| params.get("mass"))
                        .and_then(Value::as_f64)
                        .unwrap_or(1.0);
                    let c = params
                        .get("c")
                        .or_else(|| params.get("damping"))
                        .and_then(Value::as_f64)
                        .unwrap_or(0.5);
                    let k = params
                        .get("k")
                        .or_else(|| params.get("spring_constant"))
                        .and_then(Value::as_f64)
                        .unwrap_or(10.0);
                    let x0 = params.get("x0").and_then(Value::as_f64).unwrap_or(1.0);
                    let v0 = params.get("v0").and_then(Value::as_f64).unwrap_or(0.0);
                    let t_max = params.get("t_max").and_then(Value::as_f64).unwrap_or(5.0);
                    let dt = params.get("dt").and_then(Value::as_f64).unwrap_or(0.01);

                    if m <= 0.0 || k <= 0.0 || dt <= 0.0 {
                        return Err("mass, spring constant, and time step must be positive".into());
                    }

                    let omega_0 = (k / m).sqrt();
                    let damping_ratio = c / (2.0 * (m * k).sqrt());
                    let regime = if (damping_ratio - 1.0).abs() < 1e-4 {
                        "critically_damped"
                    } else if damping_ratio < 1.0 {
                        "underdamped"
                    } else {
                        "overdamped"
                    };

                    let mut t = 0.0;
                    let mut x = x0;
                    let mut v = v0;
                    let mut steps = 0;

                    while t <= t_max {
                        steps += 1;
                        let a = -(c * v + k * x) / m;
                        v += a * dt;
                        x += v * dt;
                        t += dt;
                    }

                    (
                        "m*x'' + c*x' + k*x = 0",
                        json!({
                            "natural_frequency_rad_s": omega_0,
                            "damping_ratio": damping_ratio,
                            "regime": regime,
                            "initial_displacement": x0,
                            "steps": steps,
                            "final_displacement": x,
                            "final_velocity": v
                        }),
                    )
                }
                _ => return Err(format!("unknown physics operation '{}'", operation)),
            },

            "chemistry" => match operation {
                "element_lookup" => {
                    let sym = params
                        .get("symbol")
                        .and_then(Value::as_str)
                        .ok_or("missing 'symbol'")?;
                    let el = Chemistry::get_element(sym)
                        .ok_or_else(|| format!("element '{}' not found", sym))?;
                    (
                        "Periodic Table Lookup",
                        json!({
                            "symbol": el.symbol,
                            "name": el.name,
                            "atomic_number": el.atomic_number,
                            "atomic_mass_amu": el.atomic_mass,
                            "period": el.period,
                            "group": el.group
                        }),
                    )
                }
                "molar_mass" => {
                    let formula = params
                        .get("formula")
                        .and_then(Value::as_str)
                        .ok_or("missing 'formula'")?;
                    let mass = Chemistry::molar_mass(formula).map_err(|e| format!("{:?}", e))?;
                    (
                        "Molar Mass Calculation",
                        json!({ "formula": formula, "molar_mass_g_per_mol": mass }),
                    )
                }
                "ideal_gas_pressure" => {
                    let n = params
                        .get("n")
                        .and_then(Value::as_f64)
                        .ok_or("missing 'n' (moles)")?;
                    let t = params
                        .get("t")
                        .and_then(Value::as_f64)
                        .ok_or("missing 't' (temperature K)")?;
                    let v = params
                        .get("v")
                        .and_then(Value::as_f64)
                        .ok_or("missing 'v' (volume m^3)")?;
                    let p =
                        Chemistry::ideal_gas_pressure(n, t, v).map_err(|e| format!("{:?}", e))?;
                    (
                        "PV = nRT => P = nRT/V",
                        json!({ "pressure_pascals": p, "moles": n, "temp_k": t, "volume_m3": v }),
                    )
                }
                "ph" => {
                    let h_conc = params
                        .get("h_concentration")
                        .and_then(Value::as_f64)
                        .ok_or("missing 'h_concentration'")?;
                    let ph_val = Chemistry::ph(h_conc).map_err(|e| format!("{:?}", e))?;
                    (
                        "pH = -log10[H+]",
                        json!({ "ph": ph_val, "h_concentration_m": h_conc }),
                    )
                }
                _ => return Err(format!("unknown chemistry operation '{}'", operation)),
            },

            "biology" => match operation {
                "transcribe" => {
                    let dna = params
                        .get("dna")
                        .and_then(Value::as_str)
                        .ok_or("missing 'dna'")?;
                    let rna =
                        Biology::transcribe_dna_to_mrna(dna).map_err(|e| format!("{:?}", e))?;
                    let comp = Biology::dna_complement(dna).map_err(|e| format!("{:?}", e))?;
                    (
                        "Central Dogma: DNA -> RNA Transcription",
                        json!({ "dna_coding": dna, "mrna": rna, "dna_complement": comp }),
                    )
                }
                "translate" => {
                    let mrna = params
                        .get("mrna")
                        .and_then(Value::as_str)
                        .ok_or("missing 'mrna'")?;
                    let peptides = Biology::translate_mrna(mrna).map_err(|e| format!("{:?}", e))?;
                    (
                        "Genetic Code Translation: mRNA -> Polypeptide",
                        json!({ "mrna": mrna, "amino_acids": peptides }),
                    )
                }
                _ => return Err(format!("unknown biology operation '{}'", operation)),
            },

            "astronomy" => match operation {
                "escape_velocity" => {
                    let m = params
                        .get("mass")
                        .and_then(Value::as_f64)
                        .ok_or("missing 'mass' (kg)")?;
                    let r = params
                        .get("radius")
                        .and_then(Value::as_f64)
                        .ok_or("missing 'radius' (m)")?;
                    let v_esc = Astronomy::escape_velocity(m, r).map_err(|e| format!("{:?}", e))?;
                    (
                        "v_esc = sqrt(2*G*M / R)",
                        json!({ "escape_velocity_mps": v_esc, "mass_kg": m, "radius_m": r }),
                    )
                }
                "schwarzschild_radius" => {
                    let m = params
                        .get("mass")
                        .and_then(Value::as_f64)
                        .ok_or("missing 'mass' (kg)")?;
                    let rs = Astronomy::schwarzschild_radius(m).map_err(|e| format!("{:?}", e))?;
                    (
                        "R_s = 2*G*M / c^2",
                        json!({ "schwarzschild_radius_m": rs, "mass_kg": m }),
                    )
                }
                _ => return Err(format!("unknown astronomy operation '{}'", operation)),
            },

            _ => {
                return Err(format!(
                    "unsupported scientific discipline '{}'",
                    discipline
                ))
            }
        };

        let elapsed = start.elapsed().as_micros() as u64;
        Ok(ScienceEvaluationResult {
            success: true,
            domain: "science".into(),
            discipline: discipline.into(),
            formula_applied: formula.into(),
            input_parameters: params.clone(),
            calculated_result: result,
            explanation: format!(
                "Scientific formula '{}' evaluated deterministically in {} µs",
                formula, elapsed
            ),
            execution_time_us: elapsed,
        })
    }

    /// Natural language query interpreter for standard science queries.
    pub fn solve_query(&self, query: &str) -> Option<ScienceEvaluationResult> {
        let q = query.trim().to_lowercase();

        // Pattern: calculate force for m=... and a=...
        if q.contains("force") && (q.contains("m=") || q.contains("mass")) {
            let m_val = extract_param_f64(&q, &["m=", "mass="]);
            let a_val = extract_param_f64(&q, &["a=", "accel=", "acceleration="]);
            if let (Some(m), Some(a)) = (m_val, a_val) {
                return self
                    .evaluate("physics", "force", &json!({ "m": m, "a": a }))
                    .ok();
            }
        }

        // Pattern: molar mass of <formula>
        if q.contains("molar mass") {
            let words: Vec<&str> = q.split_whitespace().collect();
            if let Some(pos) = words.iter().position(|&w| w == "of") {
                if pos + 1 < words.len() {
                    let formula = words[pos + 1]
                        .trim_matches(|c: char| !c.is_alphanumeric())
                        .to_uppercase();
                    return self
                        .evaluate("chemistry", "molar_mass", &json!({ "formula": formula }))
                        .ok();
                }
            }
        }

        // Pattern: escape velocity
        if q.contains("escape velocity") && (q.contains("earth") || q.contains("m=")) {
            let (m, r) = if q.contains("earth") {
                (EARTH_MASS, EARTH_RADIUS)
            } else {
                let m = extract_param_f64(&q, &["m=", "mass="])?;
                let r = extract_param_f64(&q, &["r=", "radius="])?;
                (m, r)
            };
            return self
                .evaluate(
                    "astronomy",
                    "escape_velocity",
                    &json!({ "mass": m, "radius": r }),
                )
                .ok();
        }

        // Pattern: energy for mass or E = mc^2
        if (q.contains("e=mc^2") || q.contains("e = mc^2") || q.contains("mass energy"))
            && q.contains("m=")
        {
            let m = extract_param_f64(&q, &["m=", "mass="])?;
            return self
                .evaluate("physics", "mass_energy", &json!({ "m": m }))
                .ok();
        }

        None
    }
}

fn extract_param_f64(text: &str, keys: &[&str]) -> Option<f64> {
    for &key in keys {
        if let Some(idx) = text.find(key) {
            let tail = &text[idx + key.len()..];
            let num_str: String = tail
                .chars()
                .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-')
                .collect();
            if let Ok(v) = num_str.parse::<f64>() {
                return Some(v);
            }
        }
    }
    None
}
