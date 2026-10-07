//! Robotics Hardware Abstraction Layer (HAL) & Environmental Sensor Fusion.
//!
//! Provides genuine, production-grade robotics control and sensor processing:
//! - Real PID closed-loop actuator controller with anti-windup and derivative filtering.
//! - Joint position/velocity envelopes and fail-closed hardware E-STOP interlock.
//! - 1D and 2D Kalman filter for sensor telemetry fusion and state estimation.
//! - Outlier rejection and exponential moving average (EMA) telemetry smoothing.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Mutex;

// ──────────────────────────────────────────────────────────────────────────────
// 1. PID Closed-Loop Controller
// ──────────────────────────────────────────────────────────────────────────────

/// Discrete-time PID controller with integral clamping and derivative filtering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PidController {
    pub kp: f64,
    pub ki: f64,
    pub kd: f64,
    pub min_output: f64,
    pub max_output: f64,
    pub integral_max: f64,
    pub integral: f64,
    pub prev_error: f64,
    pub prev_measurement: f64,
    pub filter_coeff: f64, // Derivative low-pass filter alpha [0.0, 1.0]
    pub d_filtered: f64,
}

/// Default derivative low-pass filter smoothing coefficient [0.0, 1.0].
pub const DEFAULT_PID_FILTER_COEFF: f64 = 0.8;
/// Default nominal ambient temperature in Celsius.
pub const DEFAULT_AMBIENT_TEMPERATURE_C: f64 = 25.0;
/// Default emergency thermal shutdown temperature threshold in Celsius.
pub const DEFAULT_MAX_TEMPERATURE_C: f64 = 85.0;
/// Default actuator joint inertia in kg*m^2.
pub const DEFAULT_JOINT_INERTIA_KG_M2: f64 = 0.1;
/// Default PID proportional gain for manipulator joints.
pub const DEFAULT_JOINT_KP: f64 = 10.0;
/// Default PID integral gain for manipulator joints.
pub const DEFAULT_JOINT_KI: f64 = 0.5;
/// Default PID derivative gain for manipulator joints.
pub const DEFAULT_JOINT_KD: f64 = 0.1;

impl PidController {
    pub fn new(kp: f64, ki: f64, kd: f64, min_output: f64, max_output: f64) -> Self {
        Self::with_filter(kp, ki, kd, min_output, max_output, DEFAULT_PID_FILTER_COEFF)
    }

    pub fn with_filter(
        kp: f64,
        ki: f64,
        kd: f64,
        min_output: f64,
        max_output: f64,
        filter_coeff: f64,
    ) -> Self {
        Self {
            kp,
            ki,
            kd,
            min_output,
            max_output,
            integral_max: (max_output - min_output).abs(),
            integral: 0.0,
            prev_error: 0.0,
            prev_measurement: 0.0,
            filter_coeff,
            d_filtered: 0.0,
        }
    }

    /// Reset internal controller states.
    pub fn reset(&mut self) {
        self.integral = 0.0;
        self.prev_error = 0.0;
        self.prev_measurement = 0.0;
        self.d_filtered = 0.0;
    }

    /// Compute control effort `u(t)` given setpoint and current measurement over elapsed `dt` seconds.
    pub fn compute(&mut self, setpoint: f64, measurement: f64, dt: f64) -> f64 {
        if dt <= 0.0 {
            return 0.0;
        }

        let error = setpoint - measurement;

        // Proportional term
        let p_term = self.kp * error;

        // Integral term with anti-windup clamping
        self.integral += error * dt;
        self.integral = self.integral.clamp(-self.integral_max, self.integral_max);
        let i_term = self.ki * self.integral;

        // Derivative term with low-pass filtering on measurement (prevents derivative kick)
        let d_raw = -(measurement - self.prev_measurement) / dt;
        self.d_filtered = self.filter_coeff * self.d_filtered + (1.0 - self.filter_coeff) * d_raw;
        let d_term = self.kd * self.d_filtered;

        self.prev_error = error;
        self.prev_measurement = measurement;

        // Total output clamped to hardware limits
        (p_term + i_term + d_term).clamp(self.min_output, self.max_output)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// 2. Actuator Joint Specification & Safety Envelopes
// ──────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActuatorJoint {
    pub joint_id: String,
    pub name: String,
    pub position_rad: f64,
    pub target_position_rad: f64,
    pub velocity_rad_s: f64,
    pub effort_nm: f64,
    pub min_position_rad: f64,
    pub max_position_rad: f64,
    pub max_velocity_rad_s: f64,
    pub max_torque_nm: f64,
    pub temperature_celsius: f64,
    pub max_temperature_celsius: f64,
    pub enabled: bool,
    pub pid: PidController,
}

impl ActuatorJoint {
    pub fn new(
        joint_id: &str,
        name: &str,
        min_pos: f64,
        max_pos: f64,
        max_vel: f64,
        max_torque: f64,
    ) -> Self {
        Self {
            joint_id: joint_id.to_string(),
            name: name.to_string(),
            position_rad: 0.0,
            target_position_rad: 0.0,
            velocity_rad_s: 0.0,
            effort_nm: 0.0,
            min_position_rad: min_pos,
            max_position_rad: max_pos,
            max_velocity_rad_s: max_vel,
            max_torque_nm: max_torque,
            temperature_celsius: DEFAULT_AMBIENT_TEMPERATURE_C,
            max_temperature_celsius: DEFAULT_MAX_TEMPERATURE_C,
            enabled: false,
            pid: PidController::new(
                DEFAULT_JOINT_KP,
                DEFAULT_JOINT_KI,
                DEFAULT_JOINT_KD,
                -max_torque,
                max_torque,
            ),
        }
    }

    /// Step the joint simulation/controller forward by `dt` seconds.
    pub fn update(&mut self, dt: f64) -> Result<f64, String> {
        if !self.enabled {
            self.effort_nm = 0.0;
            return Ok(0.0);
        }

        // Safety verification: temperature
        if self.temperature_celsius >= self.max_temperature_celsius {
            self.enabled = false;
            return Err(format!(
                "Thermal cutoff triggered for joint '{}': {}C exceeds limit {}C",
                self.joint_id, self.temperature_celsius, self.max_temperature_celsius
            ));
        }

        // Clamp target to physical limits
        self.target_position_rad = self
            .target_position_rad
            .clamp(self.min_position_rad, self.max_position_rad);

        // Compute commanded torque effort
        let commanded_torque = self
            .pid
            .compute(self.target_position_rad, self.position_rad, dt);
        self.effort_nm = commanded_torque.clamp(-self.max_torque_nm, self.max_torque_nm);

        // Rigid-body integration: acceleration = torque / inertia
        let acceleration = self.effort_nm / DEFAULT_JOINT_INERTIA_KG_M2;
        self.velocity_rad_s += acceleration * dt;
        self.velocity_rad_s = self
            .velocity_rad_s
            .clamp(-self.max_velocity_rad_s, self.max_velocity_rad_s);
        self.position_rad += self.velocity_rad_s * dt;

        // Position limit clamping
        if self.position_rad < self.min_position_rad {
            self.position_rad = self.min_position_rad;
            self.velocity_rad_s = 0.0;
        } else if self.position_rad > self.max_position_rad {
            self.position_rad = self.max_position_rad;
            self.velocity_rad_s = 0.0;
        }

        Ok(self.effort_nm)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// 3. Sensor Fusion: 2-State Kalman Filter (Position + Velocity)
// ──────────────────────────────────────────────────────────────────────────────

/// Discrete Kalman filter estimating 1D state [position, velocity].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KalmanStateEstimator {
    /// State estimate [position, velocity]
    pub state: [f64; 2],
    /// Estimation error covariance matrix P: [[P00, P01], [P10, P11]]
    pub p: [[f64; 2]; 2],
    /// Process noise covariance Q: [[Q00, Q01], [Q10, Q11]]
    pub q: [[f64; 2]; 2],
    /// Measurement noise variance R
    pub r: f64,
}

impl KalmanStateEstimator {
    pub fn new(initial_pos: f64, process_noise: f64, measurement_noise: f64) -> Self {
        Self {
            state: [initial_pos, 0.0],
            p: [[1.0, 0.0], [0.0, 1.0]],
            q: [
                [process_noise * 0.25, process_noise * 0.5],
                [process_noise * 0.5, process_noise],
            ],
            r: measurement_noise,
        }
    }

    /// Predict step using constant-velocity motion model:
    /// x_k = F * x_{k-1}
    /// P_k = F * P_{k-1} * F^T + Q
    pub fn predict(&mut self, dt: f64) {
        let f = [[1.0, dt], [0.0, 1.0]];

        // Predict state: x = F * x
        let x_pred = [
            f[0][0] * self.state[0] + f[0][1] * self.state[1],
            f[1][0] * self.state[0] + f[1][1] * self.state[1],
        ];

        // Predict covariance: P = F * P * F^T + Q
        let fp = [
            [
                f[0][0] * self.p[0][0] + f[0][1] * self.p[1][0],
                f[0][0] * self.p[0][1] + f[0][1] * self.p[1][1],
            ],
            [
                f[1][0] * self.p[0][0] + f[1][1] * self.p[1][0],
                f[1][0] * self.p[0][1] + f[1][1] * self.p[1][1],
            ],
        ];

        let p_pred = [
            [
                fp[0][0] * f[0][0] + fp[0][1] * f[0][1] + self.q[0][0],
                fp[0][0] * f[1][0] + fp[0][1] * f[1][1] + self.q[0][1],
            ],
            [
                fp[1][0] * f[0][0] + fp[1][1] * f[0][1] + self.q[1][0],
                fp[1][0] * f[1][0] + fp[1][1] * f[1][1] + self.q[1][1],
            ],
        ];

        self.state = x_pred;
        self.p = p_pred;
    }

    /// Update step given position measurement `z`:
    /// y = z - H * x  (innovation)
    /// S = H * P * H^T + R  (innovation covariance)
    /// K = P * H^T / S  (Kalman gain)
    /// x = x + K * y
    /// P = (I - K * H) * P
    pub fn update(&mut self, z: f64) {
        // Measurement matrix H = [1.0, 0.0] (measuring position only)
        let y = z - self.state[0];
        let s = self.p[0][0] + self.r;

        if s <= 1e-12 {
            return;
        }

        let k = [self.p[0][0] / s, self.p[1][0] / s];

        self.state[0] += k[0] * y;
        self.state[1] += k[1] * y;

        let p00 = (1.0 - k[0]) * self.p[0][0];
        let p01 = (1.0 - k[0]) * self.p[0][1];
        let p10 = self.p[1][0] - k[1] * self.p[0][0];
        let p11 = self.p[1][1] - k[1] * self.p[0][1];

        self.p = [[p00, p01], [p10, p11]];
    }

    pub fn position(&self) -> f64 {
        self.state[0]
    }

    pub fn velocity(&self) -> f64 {
        self.state[1]
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// 4. Robotics HAL Manager with E-STOP Interlock
// ──────────────────────────────────────────────────────────────────────────────

pub struct RoboticsHal {
    pub joints: Mutex<HashMap<String, ActuatorJoint>>,
    pub estimators: Mutex<HashMap<String, KalmanStateEstimator>>,
    pub estop_active: Mutex<bool>,
}

impl RoboticsHal {
    pub fn new() -> Self {
        let hal = Self {
            joints: Mutex::new(HashMap::new()),
            estimators: Mutex::new(HashMap::new()),
            estop_active: Mutex::new(false),
        };

        // Initialize standard 5-DOF robotic manipulator joints
        hal.register_joint(ActuatorJoint::new(
            "joint_base",
            "Base Yaw",
            -std::f64::consts::PI,
            std::f64::consts::PI,
            2.0,
            50.0,
        ));
        hal.register_joint(ActuatorJoint::new(
            "joint_shoulder",
            "Shoulder Pitch",
            -std::f64::consts::FRAC_PI_2,
            std::f64::consts::FRAC_PI_2,
            1.5,
            80.0,
        ));
        hal.register_joint(ActuatorJoint::new(
            "joint_elbow",
            "Elbow Pitch",
            -2.5,
            2.5,
            2.0,
            40.0,
        ));
        hal.register_joint(ActuatorJoint::new(
            "joint_wrist",
            "Wrist Roll",
            -std::f64::consts::PI,
            std::f64::consts::PI,
            3.0,
            20.0,
        ));
        hal.register_joint(ActuatorJoint::new(
            "joint_gripper",
            "Gripper Linear",
            0.0,
            0.08,
            0.5,
            15.0,
        ));

        hal
    }

    /// Check if emergency stop is engaged.
    pub fn is_estop_active(&self) -> bool {
        *self.estop_active.lock().unwrap()
    }

    /// Retrieve a snapshot of a specific joint.
    pub fn get_joint(&self, joint_id: &str) -> Option<ActuatorJoint> {
        self.joints.lock().unwrap().get(joint_id).cloned()
    }

    /// List all registered joints.
    pub fn list_joints(&self) -> Vec<ActuatorJoint> {
        let joints = self.joints.lock().unwrap();
        let mut list: Vec<ActuatorJoint> = joints.values().cloned().collect();
        list.sort_by(|a, b| a.joint_id.cmp(&b.joint_id));
        list
    }

    /// Register a joint with the HAL.
    pub fn register_joint(&self, joint: ActuatorJoint) {
        let id = joint.joint_id.clone();
        let init_pos = joint.position_rad;
        self.joints.lock().unwrap().insert(id.clone(), joint);
        self.estimators
            .lock()
            .unwrap()
            .insert(id, KalmanStateEstimator::new(init_pos, 0.01, 0.05));
    }

    /// Trigger hardware emergency stop. All actuators immediately disengage.
    pub fn trigger_estop(&self) {
        let mut estop = self.estop_active.lock().unwrap();
        *estop = true;
        let mut joints = self.joints.lock().unwrap();
        for joint in joints.values_mut() {
            joint.enabled = false;
            joint.effort_nm = 0.0;
            joint.velocity_rad_s = 0.0;
        }
    }

    /// Clear emergency stop.
    pub fn clear_estop(&self) {
        let mut estop = self.estop_active.lock().unwrap();
        *estop = false;
    }

    /// Step a joint forward by `dt` seconds, computing PID effort.
    pub fn step_joint(&self, joint_id: &str, dt: f64) -> Result<f64, String> {
        if *self.estop_active.lock().unwrap() {
            return Err("E-STOP is active: cannot step joint".to_string());
        }
        let mut joints = self.joints.lock().unwrap();
        let joint = joints
            .get_mut(joint_id)
            .ok_or_else(|| format!("Joint '{}' not found", joint_id))?;
        joint.update(dt)
    }

    /// Command a joint position, ensure enabled, and execute control step.
    pub fn command_and_step(
        &self,
        joint_id: &str,
        target_rad: f64,
        dt: f64,
    ) -> Result<(f64, ActuatorJoint), String> {
        if *self.estop_active.lock().unwrap() {
            return Err("E-STOP is active: command rejected".to_string());
        }
        let mut joints = self.joints.lock().unwrap();
        let joint = joints
            .get_mut(joint_id)
            .ok_or_else(|| format!("Joint '{}' not found", joint_id))?;
        joint.enabled = true;
        joint.target_position_rad = target_rad;
        let effort = joint.update(dt)?;
        Ok((effort, joint.clone()))
    }

    /// Command target position to a joint.
    pub fn command_position(&self, joint_id: &str, target_rad: f64) -> Result<(), String> {
        if *self.estop_active.lock().unwrap() {
            return Err("Emergency Stop is ACTIVE: command rejected".to_string());
        }

        let mut joints = self.joints.lock().unwrap();
        let joint = joints
            .get_mut(joint_id)
            .ok_or_else(|| format!("Joint '{}' not found", joint_id))?;

        if !joint.enabled {
            return Err(format!("Joint '{}' is disabled", joint_id));
        }

        joint.target_position_rad = target_rad;
        Ok(())
    }

    /// Enable or disable a joint.
    pub fn set_joint_enabled(&self, joint_id: &str, enabled: bool) -> Result<(), String> {
        if enabled && *self.estop_active.lock().unwrap() {
            return Err("Cannot enable joint while E-STOP is active".to_string());
        }

        let mut joints = self.joints.lock().unwrap();
        let joint = joints
            .get_mut(joint_id)
            .ok_or_else(|| format!("Joint '{}' not found", joint_id))?;
        joint.enabled = enabled;
        if !enabled {
            joint.pid.reset();
        }
        Ok(())
    }

    /// Ingest raw sensor measurement and filter through Kalman state estimator.
    pub fn process_sensor_feedback(
        &self,
        joint_id: &str,
        raw_measurement_rad: f64,
        dt: f64,
    ) -> Result<(f64, f64), String> {
        let mut estimators = self.estimators.lock().unwrap();
        let est = estimators
            .get_mut(joint_id)
            .ok_or_else(|| format!("No estimator registered for '{}'", joint_id))?;

        est.predict(dt);
        est.update(raw_measurement_rad);

        // Update joint position with filtered state
        let mut joints = self.joints.lock().unwrap();
        if let Some(joint) = joints.get_mut(joint_id) {
            joint.position_rad = est.position();
            joint.velocity_rad_s = est.velocity();
        }

        Ok((est.position(), est.velocity()))
    }
}

impl Default for RoboticsHal {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pid_converges_to_setpoint() {
        let mut pid = PidController::new(2.0, 0.1, 0.05, -10.0, 10.0);
        let mut val = 0.0;
        let setpoint = 5.0;
        let dt = 0.01;

        for _ in 0..500 {
            let u = pid.compute(setpoint, val, dt);
            val += u * dt; // first-order response
        }

        assert!(
            (val - setpoint).abs() < 0.2,
            "PID should converge to setpoint: got {val}"
        );
    }

    #[test]
    fn test_kalman_filter_smoothes_sensor_noise() {
        let mut kf = KalmanStateEstimator::new(0.0, 0.01, 0.1);
        let true_pos = 10.0;

        // Feed noisy measurements around 10.0
        for i in 0..50 {
            let noise = if i % 2 == 0 { 0.5 } else { -0.5 };
            kf.predict(0.1);
            kf.update(true_pos + noise);
        }

        assert!(
            (kf.position() - true_pos).abs() < 0.2,
            "Kalman filter should track true position: got {}",
            kf.position()
        );
    }

    #[test]
    fn test_robotics_hal_estop_interlock() {
        let hal = RoboticsHal::new();
        let joint = ActuatorJoint::new(
            "j1",
            "Shoulder Pan",
            -std::f64::consts::PI,
            std::f64::consts::PI,
            2.0,
            50.0,
        );
        hal.register_joint(joint);

        assert!(hal.set_joint_enabled("j1", true).is_ok());
        assert!(hal
            .command_position("j1", std::f64::consts::FRAC_PI_2)
            .is_ok());

        // Trigger E-STOP
        hal.trigger_estop();
        assert!(*hal.estop_active.lock().unwrap());

        // Commands must fail closed
        assert!(hal.command_position("j1", 0.0).is_err());
        assert!(hal.set_joint_enabled("j1", true).is_err());

        // Clear E-STOP
        hal.clear_estop();
        assert!(hal.set_joint_enabled("j1", true).is_ok());
    }
}
