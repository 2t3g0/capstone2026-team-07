"""Startup-only source isolation for explicitly labelled SITL incident replay."""
import re
from dataclasses import dataclass
from typing import Mapping
import uuid


@dataclass(frozen=True)
class Phase1InputProfile:
    topic: str
    separate: bool
    source: str
    evidence_topic: str
    replay_frame_id: str = ""


def phase1_input_profile(*, color_topic: str, phase1_topic: str = "",
                        mode: str = "camera", evidence_topic: str = "",
                        replay_frame_id: str = "",
                        environment: Mapping[str, str]) -> Phase1InputProfile:
    if mode not in ("camera", "sitl_replay"):
        raise ValueError("unknown Phase1 input mode")
    if not isinstance(replay_frame_id, str):
        raise ValueError("Phase1 replay frame ID must be a string")
    if mode == "camera" and replay_frame_id:
        raise ValueError("camera mode must not set a replay frame ID")
    selected = phase1_topic or color_topic
    separate = selected != color_topic
    if mode == "camera" and separate:
        raise ValueError("separate Phase1 input requires labelled sitl_replay mode")
    if mode == "sitl_replay" or evidence_topic:
        if (environment.get("ROS_DOMAIN_ID") != "148"
                or environment.get("ROS_LOCALHOST_ONLY") != "1"):
            raise ValueError("trial input/evidence requires ROS domain148 localhost-only")
    topic_pattern = r"/jolgwa/trial/[A-Za-z_][A-Za-z0-9_]*(?:/[A-Za-z_][A-Za-z0-9_]*)*"
    if mode == "sitl_replay":
        if not separate or not re.fullmatch(topic_pattern, selected):
            raise ValueError("SITL replay must use a separate /jolgwa/trial topic")
        prefix = "jolgwa_sitl_replay/"
        token = replay_frame_id[len(prefix):]
        try:
            valid_id = replay_frame_id.startswith(prefix) and str(uuid.UUID(token)) == token
        except ValueError:
            valid_id = False
        if not valid_id:
            raise ValueError("SITL replay requires jolgwa_sitl_replay/<canonical UUID> frame ID")
    if evidence_topic and (not re.fullmatch(topic_pattern, evidence_topic)
                           or evidence_topic in (color_topic, selected)):
        raise ValueError("trial evidence must use a separate /jolgwa/trial topic")
    return Phase1InputProfile(selected, separate,
        "jetson-phase1-sitl-replay" if mode == "sitl_replay" else "jetson-phase1-d435i",
        evidence_topic, replay_frame_id)
