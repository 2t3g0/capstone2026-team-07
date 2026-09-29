from __future__ import annotations

import argparse
import json
from pathlib import Path

from .gemini import GeminiPlanner
from .models import MissionPlan
from .zones import ZoneCatalog


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Jolgwa UAV mission tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="run the local mission planner API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=9293)

    plan = subparsers.add_parser("plan", help="plan a text patrol command")
    plan.add_argument("text")
    plan.add_argument("--thinking-level", choices=("low", "medium", "high"), default="medium")

    audio = subparsers.add_parser("plan-audio", help="plan a command from an audio file")
    audio.add_argument("path", type=Path)
    audio.add_argument("--mime-type", default="audio/wav")

    listen = subparsers.add_parser("listen", help="record a spoken patrol command")
    listen.add_argument("--seconds", type=float, default=5.0)
    listen.add_argument("--device")
    listen.add_argument(
        "--thinking-level", choices=("low", "medium", "high"), default="medium"
    )

    validate = subparsers.add_parser("validate", help="validate a mission JSON file")
    validate.add_argument("path", type=Path)
    validate.add_argument("--zones", type=Path, default=Path("config/zones.yaml"))
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "serve":
        import uvicorn

        uvicorn.run("jolgwa_uav.service:app", host=args.host, port=args.port)
        return

    if args.command == "plan":
        result = GeminiPlanner(thinking_level=args.thinking_level).plan_text(args.text)
    elif args.command == "plan-audio":
        result = GeminiPlanner().plan_audio(args.path.read_bytes(), args.mime_type)
    elif args.command == "listen":
        from .voice import record_fixed_duration

        recording = record_fixed_duration(args.seconds, device=args.device)
        result = GeminiPlanner(thinking_level=args.thinking_level).plan_audio(
            recording.wav_bytes,
            "audio/wav",
        )
    else:
        result = MissionPlan.model_validate_json(args.path.read_text(encoding="utf-8"))
        ZoneCatalog.load(args.zones).build_route(result)
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
