"""Small, observe-first front end for existing field tools.

This is not a new flight controller. Offline evidence, a file collector, the
receive-only radio viewer, and the event rehearsal are executable adapters.
Flight launch remains explicitly blocked until its transport/ownership path is
connected; emitting a ROS profile is never reported as a successful flight.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.parse import urlsplit
import uuid

ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = ROOT / "config/field_experiments.json"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _save(path: Path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="ground", help="1..6 or ground/jetson-sitl/route-observe/route-avoid/event-only/integrated")
    parser.add_argument("--mode", choices=("observe", "simulate", "control"), default="observe")
    parser.add_argument("--target", choices=("unspecified", "sim", "real"), default="unspecified")
    parser.add_argument("--confirm-target", choices=("sim", "real"))
    parser.add_argument("--execute", action="store_true", help="Execute the selected non-flight adapter; never creates approval")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/field_experiments")
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--sample-period-s", type=float, default=0.2)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--status-file", type=Path, help="Read an existing observer JSON; never opens USB")
    action.add_argument("--radio", action="store_true", help="Receive QGC-forwarded loopback UDP; no COM or transmit")
    action.add_argument("--rehearse-event", action="store_true")
    action.add_argument("--evaluate", action="store_true", help="Evaluate closed flight evidence; does not fly")
    parser.add_argument("--listen-port", type=int, default=14551)
    parser.add_argument("--backend-url", help="Explicit compute endpoint for generated SIM bridge plan; no fallback")
    parser.add_argument("--backend-manifest", type=Path, help="Recorded compute evidence, not proof of current health")
    parser.add_argument("--event-source", choices=("video", "d435"), default="video")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--serial")
    parser.add_argument("--phase1-root", type=Path)
    parser.add_argument("--device", default="0", help="Phase1 rehearsal device, e.g. 0 or cpu")
    parser.add_argument("--session-label", choices=("positive", "negative"), default="positive")
    parser.add_argument("--expected-event")
    parser.add_argument("--camera-owner-confirmed", action="store_true")
    parser.add_argument("--observations", type=Path)
    parser.add_argument("--capture-metadata", type=Path)
    parser.add_argument("--api-log", type=Path)
    parser.add_argument("--route-context", type=Path)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--obstacle-evaluation", type=Path)
    parser.add_argument("--mission-id")
    args = parser.parse_args(argv)
    if not math.isfinite(args.duration_s) or not 0.1 <= args.duration_s <= 3600:
        parser.error("duration-s must be finite and 0.1..3600")
    if not math.isfinite(args.sample_period_s) or not 0.02 <= args.sample_period_s <= 10:
        parser.error("sample-period-s must be finite and 0.02..10")
    if not 1024 <= args.listen_port <= 65535:
        parser.error("listen-port must be 1024..65535")
    return args


def select_stage(value):
    profiles = _load(PROFILE_PATH)
    for stage in profiles["stages"]:
        if str(value) in (str(stage["id"]), stage["name"]):
            return profiles, stage
    raise ValueError("unknown stage; use 1..6 or a listed stage name")


def _backend(args):
    declared = None
    if args.backend_manifest:
        raw = _load(args.backend_manifest)
        declared = raw.get("metadata", raw)
        if not isinstance(declared, dict):
            raise ValueError("backend manifest must contain an object")
    endpoint = args.backend_url or (declared or {}).get("backend_url")
    if args.backend_url and (declared or {}).get("backend_url") not in (None, args.backend_url):
        raise ValueError("backend URL differs from manifest; no fallback is permitted")
    if endpoint:
        parsed = urlsplit(endpoint)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("backend URL must be an explicit HTTP(S) endpoint without credentials/query")
    return {"endpoint": endpoint, "declared_compute_location": (declared or {}).get("compute_location"),
            "manifest": str(args.backend_manifest.resolve()) if args.backend_manifest else None,
            "current_backend_verified": False, "actual_jetson_used": "UNVERIFIED"}


def stage_ros_parameters(stage):
    """Locked profile: changing a stage cannot enable vehicle outputs."""
    selector = stage["ros_experiment_stage"]
    manager = {"simulation_only": True, "allow_real_hardware": False}
    controller = {"simulation_only": True, "allow_real_hardware": False, "enable_px4_commands": False}
    event = {"simulation_only": True, "allow_real_hardware": False, "camera_backend": "ros_compressed",
             "request_source": "gazebo-d435-event-node",
             "gimbal_backend": "fixed", "gimbal_commands_enabled": False,
             "allow_local_capture_without_control": False, "capture_duration_s": 5.0, "capture_fps": 25}
    if selector is not None:
        from jolgwa_uav.experiment_stage_policy import ros_stage_parameters
        role_parameters = ros_stage_parameters(selector)
        manager.update(role_parameters["mission_manager"])
        controller.update(role_parameters["px4_offboard_controller"])
        event.update(role_parameters["event_response"])
    else:
        event.update(enabled=False, recording_enabled=False)
    return {"mission_manager": {"ros__parameters": manager},
            "px4_offboard_controller": {"ros__parameters": controller},
            "event_response": {"ros__parameters": event}}


def build_plan(args, stage):
    backend = _backend(args)
    flight_blockers = []
    if args.mode == "control":
        flight_blockers.append("physical USB owner to ROS approved-route/event/Home control transport is not integrated; control is not ready")
    if stage["id"] == 2:
        flight_blockers.append("real Jetson compute identity, Gazebo camera/depth bridge and native SITL ownership must be connected; endpoint label alone is not proof")
    elif stage["id"] in (3, 4):
        flight_blockers.append("existing one-approval baseline helper requires event/Home policy; stage 3/4 lifecycle handoff is not connected")
    elif stage["id"] in (5, 6):
        flight_blockers.append("use the existing owned SIM lifecycle with a real dashboard approval; this runner does not claim or replace its owner")
    elif args.mode != "observe":
        flight_blockers.append("ground stage has no flight launcher")
    command = None
    if stage["ros_experiment_stage"] is not None:
        command = ["ros2", "launch", "jolgwa_ros", "patrol_stack.launch.py",
                   f"experiment_stage:={stage['id']}", "simulation_only:=true",
                   "allow_real_hardware:=false", "enable_px4_commands:=false",
                   "enable_operator_gateway:=false", "enable_gimbal_commands:=false",
                   "event_camera_backend:=ros_compressed", "event_gimbal_backend:=fixed",
                   "allow_local_event_capture:=false",
                   f"enable_event_response_node:={'true' if stage['id'] in (2, 5, 6) else 'false'}"]
        if backend["endpoint"]:
            command += ["enable_jetson_d435i_bridge:=true", f"jetson_url:={backend['endpoint']}"]
        if stage["id"] == 2:
            if not backend["endpoint"]:
                command = None
                flight_blockers.append("stage 2 requires --backend-url or an explicit backend_url in --backend-manifest")
            elif backend["declared_compute_location"] == "pc_local" or urlsplit(backend["endpoint"]).hostname in ("127.0.0.1", "::1", "localhost"):
                command = None
                flight_blockers.append("stage 2 requires a real Jetson endpoint, not a PC-local compute endpoint")
    return {"schema_version": 1, "stage": stage, "requested_mode": args.mode,
            "target": args.target, "execute_requested": args.execute,
            "backend": backend, "flight_ready": False, "flight_blockers": flight_blockers,
            "locked_ros_launch_command": command,
            "ros_profile_warning": "Locked SIM wiring only: no approval, no ARM, no flight. Existing owner must not be duplicated.",
            "available_executable_adapters": ["existing_status_file", "receive_only_radio", "ground_event_rehearsal", "closed_event_flight_evaluation"],
            "approval_generated": False, "physical_usb_opened_by_runner": False}


def _collect_status(args, out):
    path = args.status_file.resolve(strict=True)
    deadline = time.monotonic() + args.duration_s
    samples = errors = 0
    hashes = set()
    with (out / "observer_samples.jsonl").open("x", encoding="utf-8", buffering=1) as log:
        while True:
            try:
                data = path.read_bytes()
                payload = json.loads(data.decode("utf-8-sig"))
                if not isinstance(payload, dict):
                    raise ValueError("observer status must be a JSON object")
                digest = hashlib.sha256(data).hexdigest()
                hashes.add(digest)
                record = {"receipt_monotonic_s": time.monotonic(), "file_mtime_ns": path.stat().st_mtime_ns,
                          "source_sha256": digest, "payload": payload}
                samples += 1
            except (OSError, ValueError, UnicodeError) as exc:
                errors += 1
                record = {"receipt_monotonic_s": time.monotonic(), "read_error": str(exc)}
            log.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            if time.monotonic() >= deadline:
                break
            time.sleep(min(args.sample_period_s, max(0, deadline - time.monotonic())))
    return {"status": "UNVERIFIED" if samples else "BLOCKED", "adapter": "existing_status_file",
            "samples": samples, "distinct_payloads": len(hashes), "read_errors": errors,
            "reason": "Collected source records, not proof of freshness/clearance/flight success; unchanged files remain historical",
            "source": str(path)}


def _run_child(command, out, *, timeout):
    with (out / "adapter.log").open("x", encoding="utf-8") as log:
        options = {"start_new_session": True} if os.name == "posix" else {
            "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, **options)
        try:
            return child.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            # Only the process tree created by this adapter is signalled.
            # The child owns ffmpeg finalization; forced timeout is not PASS.
            if child.poll() is None:
                try:
                    if os.name == "posix":
                        os.killpg(child.pid, signal.SIGINT)
                    else:
                        child.send_signal(signal.CTRL_BREAK_EVENT)
                    child.wait(timeout=5)
                except (OSError, subprocess.SubprocessError):
                    pass
            if os.name == "posix":
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            elif child.poll() is None:
                subprocess.run(["taskkill.exe", "/PID", str(child.pid), "/T", "/F"],
                               stdout=log, stderr=subprocess.STDOUT, timeout=5, check=False)
            if child.poll() is None:
                try:
                    child.wait(timeout=5)
                except subprocess.SubprocessError:
                    pass
            raise RuntimeError("adapter interrupted/timed out; owned cleanup attempted but complete descendant cleanup is UNVERIFIED")


def execute_adapter(args, stage, out, plan):
    if args.mode == "control":
        return {"status": "BLOCKED", "reason": plan["flight_blockers"][0]}
    if args.confirm_target and args.confirm_target != args.target:
        return {"status": "BLOCKED", "reason": "confirmed target differs from requested target"}
    if args.status_file:
        return _collect_status(args, out)
    if args.radio:
        if args.mode != "observe":
            return {"status": "BLOCKED", "reason": "receive-only radio adapter requires --mode observe"}
        command = [sys.executable, str(ROOT / "scripts/observer_radio_viewer.py"), "--headless",
                   "--listen-port", str(args.listen_port), "--duration", str(args.duration_s),
                   "--log", str(out / "radio_samples.jsonl")]
        code = _run_child(command, out, timeout=args.duration_s + 15)
        return {"status": "UNVERIFIED" if code == 0 else "BLOCKED", "adapter": "receive_only_radio", "exit_code": code,
                "reason": "Radio receipt is historical, latency-unverified; viewer exit is not a flight PASS"}
    if args.rehearse_event:
        if stage["id"] != 1 or args.mode != "observe":
            return {"status": "BLOCKED", "reason": "event rehearsal is ground/observe only; not an event-flight substitute"}
        if not args.phase1_root or (args.event_source == "video" and not args.video):
            return {"status": "BLOCKED", "reason": "rehearsal requires --phase1-root and --video for video input"}
        if not 10 <= args.duration_s <= 600:
            return {"status": "BLOCKED", "reason": "event rehearsal duration must be 10..600 seconds"}
        if (args.session_label == "positive") != bool(args.expected_event):
            return {"status": "BLOCKED", "reason": "positive requires expected-event; negative must omit it"}
        if args.event_source == "d435" and not args.camera_owner_confirmed:
            return {"status": "BLOCKED", "reason": "another camera owner must not be duplicated; --camera-owner-confirmed is required"}
        command = [sys.executable, str(ROOT / "scripts/run_event_rehearsal.py"), "--source", args.event_source,
                   "--phase1-root", str(args.phase1_root), "--output-dir", str(out / "event_rehearsal"),
                   "--session-label", args.session_label, "--duration-s", str(args.duration_s),
                   "--device", args.device, "--execute"]
        for flag, value in (("--video", args.video), ("--serial", args.serial), ("--expected-event", args.expected_event)):
            if value is not None:
                command += [flag, str(value)]
        if args.camera_owner_confirmed:
            command.append("--camera-owner-confirmed")
        code = _run_child(command, out, timeout=args.duration_s + 180)
        return {"status": "UNVERIFIED" if code == 0 else "BLOCKED", "adapter": "ground_event_rehearsal", "exit_code": code,
                "reason": "Read the rehearsal result for model/capture checks; this adapter does not prove airborne event response"}
    if args.evaluate:
        if stage["id"] in (3, 4):
            if not all((args.observations, args.mission_id, args.route_context, args.provenance)):
                return {"status": "BLOCKED", "reason": "route evaluation requires observations, mission-id, route-context and provenance"}
            from jolgwa_uav.field_route_evidence import evaluate_route_flight
            rows = [json.loads(line) for line in args.observations.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
            value = evaluate_route_flight(rows, mission_id=args.mission_id, stage=stage["id"],
                                          route_context=_load(args.route_context), provenance=_load(args.provenance))
            _save(out / "route_evaluation.json", value)
            status = value.get("status", "UNVERIFIED")
            return {"status": status if status in ("PASS", "BLOCKED", "UNVERIFIED") else "UNVERIFIED",
                    "adapter": "closed_route_flight_evaluation", "evaluation": str(out / "route_evaluation.json"),
                    "reason": "Result applies only to the supplied recorded mission; no new flight ran"}
        if stage["id"] not in (5, 6):
            return {"status": "BLOCKED", "reason": "flight evidence evaluators support stages 3..6"}
        required = (args.observations, args.capture_metadata, args.api_log, args.mission_id)
        if not all(required):
            return {"status": "BLOCKED", "reason": "evaluation requires observations, capture-metadata, api-log and mission-id"}
        command = [sys.executable, str(ROOT / "scripts/evaluate_event_flight.py"), "--stage", str(stage["id"]),
                   "--observations", str(args.observations), "--capture-metadata", str(args.capture_metadata),
                   "--api-log", str(args.api_log), "--mission-id", args.mission_id, "--output", str(out / "event_evaluation.json")]
        for flag, value in (("--route-context", args.route_context), ("--provenance", args.provenance),
                            ("--obstacle-evaluation", args.obstacle_evaluation)):
            if value:
                command += [flag, str(value)]
        code = _run_child(command, out, timeout=120)
        result_path = out / "event_evaluation.json"
        if not result_path.is_file():
            return {"status": "BLOCKED", "reason": "evaluator did not write its result", "exit_code": code}
        result = _load(result_path)
        status = result.get("status", "UNVERIFIED")
        if code != 0 and status == "PASS":
            status = "BLOCKED"
        return {"status": status if status in ("PASS", "BLOCKED", "UNVERIFIED") else "UNVERIFIED",
                "adapter": "closed_event_flight_evaluation", "exit_code": code,
                "reason": "Result applies only to the supplied recorded mission; no new flight ran", "evaluation": str(result_path)}
    return {"status": "BLOCKED", "reason": "; ".join(plan["flight_blockers"]) or "select --status-file, --radio, --rehearse-event or --evaluate"}


def main(argv=None):
    args = parse_args(argv)
    try:
        profiles, stage = select_stage(args.stage)
        plan = build_plan(args, stage)
    except (ValueError, OSError, KeyError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}, ensure_ascii=False))
        return 2
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.output_root.resolve() / f"{tag}_{stage['name']}_{uuid.uuid4().hex[:12]}"
    out.mkdir(parents=True, exist_ok=False)
    if plan["locked_ros_launch_command"] and stage["ros_experiment_stage"] is not None:
        plan["locked_ros_launch_command"].append("experiment_parameter_file:=" + str(out / "stage.params.yaml"))
    _save(out / "plan.json", plan)
    _save(out / "profile.json", {"defaults": profiles["defaults"], "stage": stage})
    # JSON is valid YAML 1.2 and accepted as a ROS parameter YAML file.
    _save(out / "stage.params.yaml", stage_ros_parameters(stage))
    result = {"status": "UNVERIFIED", "reason": "plan only; --execute was not supplied"}
    try:
        if args.execute:
            result = execute_adapter(args, stage, out, plan)
    except KeyboardInterrupt:
        result = {"status": "BLOCKED", "reason": "operator interrupted collection; no runner flight command was emitted"}
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        result = {"status": "BLOCKED", "reason": type(exc).__name__ + ": " + str(exc)}
    result.update({"schema_version": 1, "stage": stage["id"], "stage_name": stage["name"],
                   "mode": args.mode, "target": args.target, "session_dir": str(out),
                   "new_flight_executed": False, "physical_flight_ready": False,
                   "runner_flight_commands_emitted": 0, "approval_generated": False})
    _save(out / "result.json", result)
    print(json.dumps(result, ensure_ascii=False))
    return 2 if result["status"] == "BLOCKED" else 0
