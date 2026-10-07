//! Physical dimension and unit conversion engine for TARA Native Engine.
//!
//! Provides verified, high-precision unit conversions across:
//! - Length (m, km, cm, mm, in, ft, yd, mi, nautical_mile, light_year, au, parsec)
//! - Mass (kg, g, mg, lb, oz, metric_ton, amu)
//! - Time (s, ms, min, hr, day, week, year)
//! - Temperature (Celsius, Fahrenheit, Kelvin, Rankine)
//! - Speed (m/s, km/h, mph, knot, ft/s, c)
//! - Force (N, kN, dyn, lbf)
//! - Pressure (Pa, kPa, MPa, bar, atm, psi, torr)
//! - Energy (J, kJ, MJ, cal, kcal, Wh, kWh, eV, BTU)
//! - Power (W, kW, MW, hp)
//! - Digital Data (B, KB, MB, GB, TB, KiB, MiB, GiB, TiB)

use std::collections::HashMap;
use std::sync::OnceLock;
use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum UnitError {
    #[error("unknown or unsupported unit '{unit}' for dimension '{dimension}'")]
    UnknownUnit { unit: String, dimension: String },
    #[error("unsupported physical dimension '{0}'")]
    UnsupportedDimension(String),
    #[error("temperature below absolute zero ({val} {unit})")]
    BelowAbsoluteZero { val: f64, unit: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dimension {
    Length,
    Mass,
    Time,
    Temperature,
    Speed,
    Force,
    Pressure,
    Energy,
    Power,
    Data,
}

pub struct UnitConverter;

struct UnitRegistry {
    length: HashMap<&'static str, f64>,
    mass: HashMap<&'static str, f64>,
    time: HashMap<&'static str, f64>,
    speed: HashMap<&'static str, f64>,
    force: HashMap<&'static str, f64>,
    pressure: HashMap<&'static str, f64>,
    energy: HashMap<&'static str, f64>,
    power: HashMap<&'static str, f64>,
    data: HashMap<&'static str, f64>,
}

static REGISTRY: OnceLock<UnitRegistry> = OnceLock::new();

fn get_registry() -> &'static UnitRegistry {
    REGISTRY.get_or_init(|| {
        let mut length = HashMap::new();
        length.insert("m", 1.0);
        length.insert("meter", 1.0);
        length.insert("meters", 1.0);
        length.insert("km", 1000.0);
        length.insert("kilometer", 1000.0);
        length.insert("cm", 0.01);
        length.insert("centimeter", 0.01);
        length.insert("mm", 0.001);
        length.insert("millimeter", 0.001);
        length.insert("um", 1e-6);
        length.insert("micrometer", 1e-6);
        length.insert("nm", 1e-9);
        length.insert("nanometer", 1e-9);
        length.insert("in", 0.0254);
        length.insert("inch", 0.0254);
        length.insert("inches", 0.0254);
        length.insert("ft", 0.3048);
        length.insert("foot", 0.3048);
        length.insert("feet", 0.3048);
        length.insert("yd", 0.9144);
        length.insert("yard", 0.9144);
        length.insert("mi", 1609.344);
        length.insert("mile", 1609.344);
        length.insert("nautical_mile", 1852.0);
        length.insert("au", 1.495978707e11);
        length.insert("light_year", 9.4607304725808e15);
        length.insert("parsec", 3.08567758149137e16);

        let mut mass = HashMap::new();
        mass.insert("kg", 1.0);
        mass.insert("kilogram", 1.0);
        mass.insert("g", 0.001);
        mass.insert("gram", 0.001);
        mass.insert("mg", 1e-6);
        mass.insert("milligram", 1e-6);
        mass.insert("ug", 1e-9);
        mass.insert("microgram", 1e-9);
        mass.insert("lb", 0.45359237);
        mass.insert("pound", 0.45359237);
        mass.insert("oz", 0.028349523125);
        mass.insert("ounce", 0.028349523125);
        mass.insert("ton", 1000.0);
        mass.insert("metric_ton", 1000.0);
        mass.insert("amu", 1.66053906660e-27);

        let mut time = HashMap::new();
        time.insert("s", 1.0);
        time.insert("second", 1.0);
        time.insert("seconds", 1.0);
        time.insert("ms", 0.001);
        time.insert("millisecond", 0.001);
        time.insert("us", 1e-6);
        time.insert("microsecond", 1e-6);
        time.insert("ns", 1e-9);
        time.insert("nanosecond", 1e-9);
        time.insert("min", 60.0);
        time.insert("minute", 60.0);
        time.insert("minutes", 60.0);
        time.insert("h", 3600.0);
        time.insert("hr", 3600.0);
        time.insert("hour", 3600.0);
        time.insert("hours", 3600.0);
        time.insert("day", 86400.0);
        time.insert("days", 86400.0);
        time.insert("week", 604800.0);
        time.insert("year", 31557600.0); // 365.25 days

        let mut speed = HashMap::new();
        speed.insert("m/s", 1.0);
        speed.insert("km/h", 1.0 / 3.6);
        speed.insert("mph", 0.44704);
        speed.insert("knot", 0.514444);
        speed.insert("ft/s", 0.3048);
        speed.insert("c", 299792458.0);

        let mut force = HashMap::new();
        force.insert("n", 1.0);
        force.insert("newton", 1.0);
        force.insert("kn", 1000.0);
        force.insert("dyn", 1e-5);
        force.insert("lbf", 4.4482216152605);

        let mut pressure = HashMap::new();
        pressure.insert("pa", 1.0);
        pressure.insert("pascal", 1.0);
        pressure.insert("kpa", 1000.0);
        pressure.insert("mpa", 1e6);
        pressure.insert("bar", 100000.0);
        pressure.insert("mbar", 100.0);
        pressure.insert("atm", 101325.0);
        pressure.insert("psi", 6894.757293168);
        pressure.insert("torr", 133.322368421);
        pressure.insert("mmhg", 133.322387415);

        let mut energy = HashMap::new();
        energy.insert("j", 1.0);
        energy.insert("joule", 1.0);
        energy.insert("joules", 1.0);
        energy.insert("kj", 1000.0);
        energy.insert("mj", 1e6);
        energy.insert("cal", 4.184);
        energy.insert("calorie", 4.184);
        energy.insert("calories", 4.184);
        energy.insert("kcal", 4184.0);
        energy.insert("wh", 3600.0);
        energy.insert("kwh", 3.6e6);
        energy.insert("kilowatt_hour", 3.6e6);
        energy.insert("kilowatt-hour", 3.6e6);
        energy.insert("ev", 1.602176634e-19);
        energy.insert("btu", 1055.05585262);

        let mut power = HashMap::new();
        power.insert("w", 1.0);
        power.insert("watt", 1.0);
        power.insert("watts", 1.0);
        power.insert("kw", 1000.0);
        power.insert("kilowatt", 1000.0);
        power.insert("mw", 1e6);
        power.insert("megawatt", 1e6);
        power.insert("hp", 745.69987158227);

        let mut data = HashMap::new();
        data.insert("b", 1.0);
        data.insert("byte", 1.0);
        data.insert("bytes", 1.0);
        data.insert("kb", 1000.0);
        data.insert("mb", 1e6);
        data.insert("gb", 1e9);
        data.insert("tb", 1e12);
        data.insert("pb", 1e15);
        data.insert("kib", 1024.0);
        data.insert("mib", 1048576.0);
        data.insert("gib", 1073741824.0);
        data.insert("tib", 1099511627776.0);

        UnitRegistry {
            length,
            mass,
            time,
            speed,
            force,
            pressure,
            energy,
            power,
            data,
        }
    })
}

impl UnitConverter {
    /// Convert value from `from_unit` to `to_unit` under the specified `dimension`.
    pub fn convert(
        val: f64,
        from_unit: &str,
        to_unit: &str,
        dim: Dimension,
    ) -> Result<f64, UnitError> {
        let from_clean = from_unit.trim().to_lowercase();
        let to_clean = to_unit.trim().to_lowercase();

        if from_clean == to_clean {
            return Ok(val);
        }

        if dim == Dimension::Temperature {
            return Self::convert_temperature(val, &from_clean, &to_clean);
        }

        let reg = get_registry();
        let table = match dim {
            Dimension::Length => &reg.length,
            Dimension::Mass => &reg.mass,
            Dimension::Time => &reg.time,
            Dimension::Speed => &reg.speed,
            Dimension::Force => &reg.force,
            Dimension::Pressure => &reg.pressure,
            Dimension::Energy => &reg.energy,
            Dimension::Power => &reg.power,
            Dimension::Data => &reg.data,
            Dimension::Temperature => unreachable!(),
        };

        let from_factor =
            table
                .get(from_clean.as_str())
                .copied()
                .ok_or_else(|| UnitError::UnknownUnit {
                    unit: from_unit.to_string(),
                    dimension: format!("{:?}", dim),
                })?;

        let to_factor =
            table
                .get(to_clean.as_str())
                .copied()
                .ok_or_else(|| UnitError::UnknownUnit {
                    unit: to_unit.to_string(),
                    dimension: format!("{:?}", dim),
                })?;

        let base_val = val * from_factor;
        Ok(base_val / to_factor)
    }

    /// High-accuracy temperature conversion handling affine scales.
    pub fn convert_temperature(val: f64, from: &str, to: &str) -> Result<f64, UnitError> {
        // Convert to Kelvin first
        let kelvin = match from {
            "c" | "celsius" => val + 273.15,
            "f" | "fahrenheit" => (val - 32.0) * (5.0 / 9.0) + 273.15,
            "k" | "kelvin" => val,
            "r" | "rankine" => val * (5.0 / 9.0),
            _ => {
                return Err(UnitError::UnknownUnit {
                    unit: from.to_string(),
                    dimension: "Temperature".into(),
                })
            }
        };

        if kelvin < -1e-6 {
            return Err(UnitError::BelowAbsoluteZero {
                val,
                unit: from.to_string(),
            });
        }

        // Convert Kelvin to destination unit
        let out = match to {
            "c" | "celsius" => kelvin - 273.15,
            "f" | "fahrenheit" => (kelvin - 273.15) * (9.0 / 5.0) + 32.0,
            "k" | "kelvin" => kelvin,
            "r" | "rankine" => kelvin * (9.0 / 5.0),
            _ => {
                return Err(UnitError::UnknownUnit {
                    unit: to.to_string(),
                    dimension: "Temperature".into(),
                })
            }
        };
        Ok(out)
    }
}
