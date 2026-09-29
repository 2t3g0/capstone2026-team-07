"""PC display watchdog: no ROS, camera, or flight-control dependencies."""
from copy import deepcopy
import math


def unknown(reason):
    return {"connected":False,"report":{"assessment":"UNKNOWN","reason":reason},"sensor":{},
            "camera_ok":False,"px4_pose_received":False}


class RemoteDisplayLease:
    def __init__(self):
        self.value = unknown("waiting_for_jetson")
        self.expires = self.sensor_expires = self.report_expires = float("-inf")

    def receive(self, value, *, nonce, started, received):
        def duration(item):
            if type(item) not in (int,float) or not math.isfinite(item) or not 0 <= item <= .5:
                raise ValueError("invalid_display_lifetime")
            return float(item)
        try:
            if not math.isfinite(started) or not math.isfinite(received) or received < started:
                raise ValueError("invalid_client_clock")
            if (value["schema_version"] != 1 or value["mode"] != "OBSERVE_ONLY"
                    or value["flight_commands_enabled"] is not False or value["nonce"] != nonce):
                raise ValueError("invalid_observer_response")
            report = value["report"]
            if (report["assessment"] not in {"CLEAR","CLIMB_REQUIRED","BLOCKED","UNKNOWN"}
                    or report["flight_commands_enabled"] is not False or report["mode"] != "OBSERVE_ONLY"):
                raise ValueError("invalid_observer_report")
            if not isinstance(report.get("reason"),str) or len(report["reason"]) > 2048:
                raise ValueError("invalid_reason")
            if type(value.get("camera_ok")) is not bool or type(value.get("px4_pose_received")) is not bool:
                raise ValueError("invalid_sensor_status")
            if not isinstance(value.get("sensor"),dict):
                raise ValueError("invalid_sensor_metrics")
            for name in ("front_near_m","front_median_m","upper_roi_near_m","valid_fraction"):
                item=value["sensor"].get(name)
                if item is not None and (type(item) not in (int,float) or not math.isfinite(item) or item < 0):
                    raise ValueError("invalid_sensor_metric:"+name)
                if name == "valid_fraction" and item is not None and item > 1:
                    raise ValueError("invalid_valid_fraction")
            rtt = received-started
            self.report_expires = received+max(0.,duration(report["valid_for_s"])-rtt)
            self.sensor_expires = received+max(0.,duration(value["sensor_valid_for_s"])-rtt)
            self.expires = received+max(0.,1.0-rtt)
            self.value = deepcopy(value)
            self.value["connected"] = True
            self.value["rtt_ms"] = rtt*1000
        except (KeyError, TypeError, ValueError) as exc:
            self.fail(str(exc))

    def fail(self, reason):
        self.value = unknown(reason)
        self.expires = self.sensor_expires = self.report_expires = float("-inf")

    def current(self, now):
        if not self.value.get("connected",False):
            return deepcopy(self.value)
        if not math.isfinite(now) or now >= self.expires:
            return unknown("connection_lost_or_status_expired")
        value = deepcopy(self.value)
        if now >= self.report_expires and value["report"]["assessment"] != "UNKNOWN":
            value["report"] = {"assessment":"UNKNOWN","reason":"observation_expired"}
        if now >= self.sensor_expires:
            value["sensor"] = {}
            value["camera_ok"] = False
        return value
