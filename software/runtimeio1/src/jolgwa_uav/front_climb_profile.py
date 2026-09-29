"""Dedicated, approved 2m-front / 3m-pass profile; generic policy unchanged."""
from dataclasses import replace
PROFILE = "FRONT_DETECT_2M_PASS_3M_V1"
BUILD_ID = "low-speed-runtimeio1-20260929"
PREFIX = "scenario_front_detect_2m_pass_3m_v1:"
def configure(config):
    return replace(config, scenario_demo=True, front_climb_demo=True,
        trigger_distance_m=2.0, release_distance_m=4.5,
        minimum_standoff_m=1.0, max_dynamic_trigger_distance_m=2.0)
