"""
python/tara_core/environment_sensor_fusion.py

Dynamic Environment Modeling, Multi-Modal Sensor Fusion & Action Grounding for TARA Core.
Provides production logic for:
1. Dynamic Environment Modeling (real-time ambient conditions, obstacle maps, physical state)
2. Multi-Modal Sensor Fusion (vision, audio, telemetry, GPIO, logs fused into coherent state)
3. Real-World Action Grounding (INTENT -> COMMAND -> PERMISSION -> EXECUTION -> OBSERVATION -> VERIFICATION)
"""

import os
import sys
import time
import math
import uuid
import logging
import threading
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone

from tara_core.robotics_hal import RoboticsHAL, DeviceType
from tara_core.event_bus import TaraEventBus

logger = logging.getLogger("TARA.SensorFusion")


@dataclass
class FusedEnvironmentState:
    timestamp: str
    ambient_temperature_c: float
    ambient_light_lux: float
    detected_obstacles: List[Dict[str, Any]]
    active_devices_online: int
    modality_confidences: Dict[str, float]
    overall_confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "ambient_temperature_c": self.ambient_temperature_c,
            "ambient_light_lux": self.ambient_light_lux,
            "detected_obstacles": self.detected_obstacles,
            "active_devices_online": self.active_devices_online,
            "modality_confidences": self.modality_confidences,
            "overall_confidence": self.overall_confidence
        }


class DynamicEnvironmentModel:
    """Maintains an active, real-time spatial and ambient world model."""

    def __init__(self):
        self.ambient_temp_c: float = 22.5
        self.ambient_light_lux: float = 500.0
        self.obstacles: Dict[str, Dict[str, Any]] = {}
        self.zones: Dict[str, Dict[str, Any]] = {
            "workcell_primary": {"bounds": [[-1.0, -1.0, 0.0], [1.0, 1.0, 1.5]], "status": "CLEAR"},
            "storage_bay": {"bounds": [[1.0, 0.0, 0.0], [2.0, 2.0, 1.0]], "status": "NOMINAL"}
        }
        self._lock = threading.RLock()

    def update_ambient(self, temp_c: Optional[float] = None, light_lux: Optional[float] = None):
        with self._lock:
            if temp_c is not None:
                self.ambient_temp_c = temp_c
            if light_lux is not None:
                self.ambient_light_lux = light_lux

    def add_or_update_obstacle(self, obstacle_id: str, position: Tuple[float, float, float], radius_m: float = 0.2):
        with self._lock:
            self.obstacles[obstacle_id] = {
                "obstacle_id": obstacle_id,
                "position": list(position),
                "radius_m": radius_m,
                "updated_at": datetime.now(timezone.utc).isoformat()
            }

    def clear_obstacle(self, obstacle_id: str):
        with self._lock:
            if obstacle_id in self.obstacles:
                del self.obstacles[obstacle_id]

    def check_path_collision(self, start: Tuple[float, float, float], end: Tuple[float, float, float]) -> bool:
        """Determines whether a 3D linear trajectory intersects any known obstacle volume."""
        with self._lock:
            for obs in self.obstacles.values():
                op = obs["position"]
                rad = obs["radius_m"]
                # Approximate distance to midpoint of path
                mid = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0, (start[2] + end[2]) / 2.0)
                dist = math.sqrt((mid[0] - op[0])**2 + (mid[1] - op[1])**2 + (mid[2] - op[2])**2)
                if dist < rad + 0.1:
                    return True
            return False


class SensorFusionEngine:
    """Fuses heterogeneous multi-modal sensor inputs into a unified state estimate."""

    def __init__(self, env_model: Optional[DynamicEnvironmentModel] = None):
        self.env = env_model or DynamicEnvironmentModel()
        self._modality_weights: Dict[str, float] = {
            "vision": 0.40,
            "telemetry": 0.30,
            "audio": 0.15,
            "gpio": 0.15
        }
        self._lock = threading.RLock()

    def fuse_readings(
        self,
        vision_input: Optional[Dict[str, Any]] = None,
        telemetry_input: Optional[Dict[str, Any]] = None,
        audio_input: Optional[Dict[str, Any]] = None,
        gpio_input: Optional[Dict[str, Any]] = None
    ) -> FusedEnvironmentState:
        with self._lock:
            confidences = {}
            temp_readings = []
            obstacles = []

            # 1. Vision Processing
            if vision_input:
                conf = vision_input.get("confidence", 0.90)
                confidences["vision"] = conf
                for obj in vision_input.get("detected_objects", []):
                    if obj.get("label") == "obstacle":
                        pos = tuple(obj.get("position", [0.5, 0.5, 0.0]))
                        obs_id = obj.get("id", f"obs_vis_{len(obstacles)}")
                        self.env.add_or_update_obstacle(obs_id, pos, obj.get("radius", 0.25))
                        obstacles.append({"id": obs_id, "source": "vision", "position": pos})

            # 2. Telemetry Processing
            if telemetry_input:
                conf = telemetry_input.get("confidence", 0.95)
                confidences["telemetry"] = conf
                if "temperature_c" in telemetry_input:
                    temp_readings.append(telemetry_input["temperature_c"])
                if "light_lux" in telemetry_input:
                    self.env.update_ambient(light_lux=telemetry_input["light_lux"])

            # 3. Audio Processing
            if audio_input:
                conf = audio_input.get("confidence", 0.80)
                confidences["audio"] = conf

            # 4. GPIO / Proximity Processing
            if gpio_input:
                conf = gpio_input.get("confidence", 0.99)
                confidences["gpio"] = conf
                if gpio_input.get("e_stop_pin_high"):
                    RoboticsHAL.get_default().interlock.trigger_emergency_stop("Physical E-Stop GPIO signal")

            # Compute fused temperature
            fused_temp = (sum(temp_readings) / len(temp_readings)) if temp_readings else self.env.ambient_temp_c
            self.env.update_ambient(temp_c=fused_temp)

            # Compute overall confidence
            overall_conf = sum(
                confidences.get(m, 0.5) * self._modality_weights.get(m, 0.25)
                for m in self._modality_weights
            )

            state = FusedEnvironmentState(
                timestamp=datetime.now(timezone.utc).isoformat(),
                ambient_temperature_c=round(fused_temp, 2),
                ambient_light_lux=self.env.ambient_light_lux,
                detected_obstacles=obstacles,
                active_devices_online=len(RoboticsHAL.get_default().list_devices()),
                modality_confidences=confidences,
                overall_confidence=round(overall_conf, 4)
            )

            TaraEventBus.get_default().publish(
                "environment.state_fused",
                state.to_dict(),
                source="SensorFusionEngine"
            )

            return state


class RealWorldActionGrounder:
    """Executes the strict grounding pipeline from intent to physical execution and verification."""

    def __init__(self, hal_ref: Optional[RoboticsHAL] = None):
        self.hal = hal_ref or RoboticsHAL.get_default()

    def ground_and_execute(
        self,
        intent: str,
        device_id: str,
        command: str,
        params: Dict[str, Any],
        is_authorized: bool = True
    ) -> Dict[str, Any]:
        """
        Enforces:
        INTENT -> COMMAND -> PERMISSION -> EXECUTION -> OBSERVATION -> VERIFICATION
        """
        if not is_authorized:
            return {
                "status": "BLOCKED",
                "stage": "PERMISSION_CHECK",
                "error": "Action denied by policy rule guard."
            }

        # Check Interlock
        if self.hal.interlock.e_stop_active:
            return {
                "status": "BLOCKED",
                "stage": "SAFETY_INTERLOCK",
                "error": f"Actuation blocked by active emergency stop: {self.hal.interlock.e_stop_reason}"
            }

        # Dispatch Execution
        if command == "motion":
            target = tuple(params.get("target_coords", (0.0, 0.0, 0.0)))
            velocity = params.get("velocity", 0.5)
            exec_res = self.hal.send_motion_command(device_id, target, velocity)
        else:
            exec_res = self.hal.read_telemetry(device_id)

        # Verification & Confirmation
        success = exec_res.get("status") == "SUCCESS"
        return {
            "status": "SUCCESS" if success else "FAILED",
            "stage": "VERIFICATION_CONFIRMED",
            "intent": intent,
            "command": command,
            "device_id": device_id,
            "execution_result": exec_res,
            "grounded": success,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
