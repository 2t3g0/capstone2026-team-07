from __future__ import annotations

import json
import os
import sys
import time
from contextlib import asynccontextmanager
from ipaddress import ip_address
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .gemini import GeminiPlanner, PlannerError
from .landmark_routes import LandmarkRouteResolver
from .landmark_search import LandmarkSearchIndex
from .models import (
    AvailableLandmark,
    AvailableRoute,
    MissionPlan,
    PlanStatus,
    PlanningContext,
)
from .operator_hub import OperatorHub
from .operations_observer import OperationsObserver
from .route_store import (
    RouteNotFound,
    RouteRevisionConflict,
    RouteStore,
    RouteUpdate,
    RouteWrite,
)


class TextPlanRequest(BaseModel):
    command: str = Field(min_length=1, max_length=4000)
    context: PlanningContext | None = None


class GeminiKeyUpdate(BaseModel):
    key: str = Field(min_length=8, max_length=512)


def _persist_user_gemini_api_key(value: str) -> None:
    if sys.platform != "win32":
        return

    import ctypes
    import winreg

    with winreg.OpenKey(
        winreg.HKEY_CURRENT_USER,
        "Environment",
        0,
        winreg.KEY_SET_VALUE,
    ) as environment_key:
        winreg.SetValueEx(
            environment_key,
            "GEMINI_API_KEY",
            0,
            winreg.REG_SZ,
            value,
        )

    # Tell newly launched Windows processes that the user environment changed.
    hwnd_broadcast = 0xFFFF
    wm_settingchange = 0x001A
    smto_abortifhung = 0x0002
    result = ctypes.c_ulong()
    ctypes.windll.user32.SendMessageTimeoutW(
        hwnd_broadcast,
        wm_settingchange,
        0,
        "Environment",
        smto_abortifhung,
        2000,
        ctypes.byref(result),
    )


def _require_loopback(request: Request) -> None:
    host = request.client.host if request.client else ""
    try:
        is_loopback = ip_address(host).is_loopback
    except ValueError:
        is_loopback = host == "testclient"
    if not is_loopback:
        raise HTTPException(
            status_code=403,
            detail="Gemini API key settings are available only on this PC.",
        )


def create_app(
    planner: GeminiPlanner | None = None,
    routes: RouteStore | None = None,
    landmark_routes: LandmarkRouteResolver | None = None,
    operations: OperationsObserver | None = None,
    api_key_writer: Callable[[str], None] | None = None,
) -> FastAPI:
    project_root = Path(__file__).resolve().parents[2]
    operations_observer = operations or OperationsObserver.from_environment(project_root)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            operations_observer.close()

    app = FastAPI(
        title="Jolgwa UAV Mission Planner",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["Content-Type"],
    )
    app.state.operations_observer = operations_observer
    operator_hub = OperatorHub(operations_observer)
    app.state.operator_hub = operator_hub
    route_store = routes or RouteStore(
        Path(
            os.getenv(
                "JOLGWA_ROAD_CATALOG",
                project_root / "config" / "routes" / "pnu_roads.json",
            )
        ),
        Path(
            os.getenv(
                "JOLGWA_CUSTOM_ROUTES",
                project_root / "config" / "routes" / "custom_routes.json",
            )
        ),
    )
    app.state.route_store = route_store
    landmark_resolver = landmark_routes
    landmark_catalog_path: Path | None = None
    if landmark_resolver is None and routes is None:
        configured_catalog = os.getenv("JOLGWA_LANDMARK_CATALOG")
        if configured_catalog:
            landmark_catalog_path = Path(configured_catalog)
        else:
            candidates = (
                project_root / "config" / "routes" / "pnu_landmarks.enriched.json",
                project_root / "config" / "routes" / "pnu_landmarks.generated.json",
                project_root / "config" / "routes" / "pnu_landmarks.json",
            )
            landmark_catalog_path = next(path for path in candidates if path.exists())
        landmark_resolver = LandmarkRouteResolver(
            route_store.base_catalog_path,
            landmark_catalog_path,
        )
    app.state.landmark_route_resolver = landmark_resolver
    landmark_summaries = landmark_resolver.summaries() if landmark_resolver else []
    if landmark_catalog_path is not None:
        catalog_payload = json.loads(landmark_catalog_path.read_text(encoding="utf-8"))
        landmark_summaries.extend(
            AvailableLandmark.model_validate(
                {
                    field: item[field]
                    for field in AvailableLandmark.model_fields
                    if field in item
                }
            )
            for item in catalog_payload.get("reference_only_landmarks", [])
        )
    landmark_by_id = {item.id: item for item in landmark_summaries}
    routeable_landmark_ids = {
        item.id for item in landmark_summaries if item.routeable
    }
    landmark_search = (
        LandmarkSearchIndex(item.model_dump(mode="json") for item in landmark_summaries)
        if landmark_summaries
        else None
    )
    app.state.landmark_search = landmark_search
    active_planner = planner
    managed_planner = planner is None
    persist_api_key = api_key_writer or _persist_user_gemini_api_key

    def get_planner() -> GeminiPlanner:
        nonlocal active_planner
        if active_planner is None:
            active_planner = GeminiPlanner(
                model=os.getenv("GEMINI_MODEL", "gemini-3.7-flash"),
                thinking_level=os.getenv("GEMINI_THINKING_LEVEL", "medium"),
            )
        return active_planner

    def planning_context(
        source: PlanningContext | None,
        *,
        command: str | None = None,
    ) -> PlanningContext:
        context = source or PlanningContext()
        routes = [
            AvailableRoute.model_validate(route)
            for route in route_store.route_summaries()
        ]
        landmarks = landmark_summaries
        if command is not None and landmark_search is not None:
            retrieval = landmark_search.search(command, limit=24)
            landmarks = [
                landmark_by_id[candidate.landmark_id]
                for candidate in retrieval.candidates
            ]
        return context.model_copy(
            update={
                "available_routes": routes,
                "available_landmarks": landmarks,
            }
        )

    def landmark_clarification(
        plan: MissionPlan,
        invalid_ids: list[str],
        *,
        message: str | None = None,
    ) -> MissionPlan:
        return MissionPlan.model_validate(
            {
                "request_purpose": plan.request_purpose,
                "status": PlanStatus.NEED_CLARIFICATION,
                "missing_fields": ["landmark_route"],
                "unsupported_values": invalid_ids,
                "message": message
                or (
                    "요청한 장소를 부산대 랜드마크 후보에서 확정하지 못했습니다. "
                    "지도에 표시된 정식 명칭이나 건물 번호를 확인해 주세요."
                ),
            }
        )

    def resolve_route(
        plan: MissionPlan,
        *,
        allowed_landmark_ids: set[str] | None = None,
    ) -> MissionPlan:
        if plan.status is PlanStatus.OK and plan.landmark_route:
            if landmark_resolver is None:
                raise ValueError("landmark route resolver is unavailable")
            if allowed_landmark_ids is not None:
                invalid_ids = sorted(
                    {
                        step.landmark_id
                        for step in plan.landmark_route
                        if step.landmark_id not in allowed_landmark_ids
                    }
                )
                if invalid_ids:
                    return landmark_clarification(plan, invalid_ids)
            unavailable_ids = sorted(
                {
                    step.landmark_id
                    for step in plan.landmark_route
                    if step.landmark_id not in routeable_landmark_ids
                }
            )
            if unavailable_ids:
                return landmark_clarification(
                    plan,
                    unavailable_ids,
                    message=(
                        "장소는 확인했지만 검증된 건물 외곽선 좌표가 없어 아직 비행 "
                        "경로를 만들 수 없습니다. 다른 건물을 선택하거나 지도에서 경로를 "
                        "직접 지정해 주세요."
                    ),
                )
            payload = plan.model_dump(mode="json")
            payload.update(landmark_resolver.generate(plan.landmark_route))
            return MissionPlan.model_validate(payload)
        if plan.status is not PlanStatus.OK or plan.route_id is None:
            return plan
        route = route_store.get(plan.route_id)
        payload = plan.model_dump(mode="json")
        payload.update(
            route_revision=route["revision"],
            route_name=route["name"],
            route_waypoints_enu=route["waypoints_enu"],
            route_frame=route["route_frame"],
            simulation_only=route["simulation_only"],
        )
        return MissionPlan.model_validate(payload)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": os.getenv("GEMINI_MODEL", "gemini-3.7-flash")}

    @app.get("/health/control")
    async def control_health() -> dict[str, object]:
        return await operator_hub.control_snapshot()

    @app.get("/health/operations")
    def operations_health() -> dict[str, object]:
        return operations_observer.summary()

    @app.get("/v1/settings/gemini-key")
    def gemini_key_status(request: Request) -> dict[str, bool]:
        _require_loopback(request)
        return {"configured": bool(os.getenv("GEMINI_API_KEY", "").strip())}

    @app.put("/v1/settings/gemini-key")
    def update_gemini_key(
        update: GeminiKeyUpdate,
        request: Request,
    ) -> dict[str, bool]:
        nonlocal active_planner
        _require_loopback(request)
        value = update.key.strip()
        if len(value) < 8:
            raise HTTPException(status_code=422, detail="Gemini API key is too short.")
        try:
            persist_api_key(value)
        except OSError as exc:
            raise HTTPException(
                status_code=500,
                detail="Gemini API key could not be saved to the Windows user profile.",
            ) from exc
        os.environ["GEMINI_API_KEY"] = value
        if managed_planner:
            active_planner = None
        return {"configured": True}

    @app.post("/v1/plan/text", response_model=MissionPlan)
    def plan_text(request: TextPlanRequest) -> MissionPlan:
        request_id = operations_observer.new_request_id()
        total_started_at = time.perf_counter()
        stage = "llm"
        operations_observer.planning_started(
            request_id,
            input_mode="text",
            command=request.command,
        )
        try:
            llm_started_at = time.perf_counter()
            context = planning_context(request.context, command=request.command)
            plan = get_planner().plan_text(request.command, context)
            llm_duration_ms = (time.perf_counter() - llm_started_at) * 1_000
            stage = "route_resolution"
            route_started_at = time.perf_counter()
            resolved = resolve_route(
                plan,
                allowed_landmark_ids={item.id for item in context.available_landmarks},
            )
            route_duration_ms = (time.perf_counter() - route_started_at) * 1_000
            operations_observer.planning_completed(
                request_id,
                resolved,
                llm_duration_ms=llm_duration_ms,
                route_duration_ms=route_duration_ms,
                total_duration_ms=(time.perf_counter() - total_started_at) * 1_000,
            )
            return resolved
        except RouteNotFound as exc:
            operations_observer.planning_failed(
                request_id,
                stage=stage,
                duration_ms=(time.perf_counter() - total_started_at) * 1_000,
                error=exc,
            )
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (PlannerError, ValueError) as exc:
            operations_observer.planning_failed(
                request_id,
                stage=stage,
                duration_ms=(time.perf_counter() - total_started_at) * 1_000,
                error=exc,
            )
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post("/v1/plan/audio", response_model=MissionPlan)
    async def plan_audio(
        audio: UploadFile = File(...),
        context_json: str | None = Form(default=None),
    ) -> MissionPlan:
        request_id = operations_observer.new_request_id()
        total_started_at = time.perf_counter()
        stage = "audio_upload"
        try:
            context = (
                PlanningContext.model_validate_json(context_json)
                if context_json
                else None
            )
            data = await audio.read()
            operations_observer.planning_started(
                request_id,
                input_mode="audio",
                audio_bytes=len(data),
                mime_type=audio.content_type or "audio/wav",
            )
            stage = "llm"
            llm_started_at = time.perf_counter()
            enriched_context = planning_context(context)
            plan = get_planner().plan_audio(
                data, audio.content_type or "audio/wav", enriched_context
            )
            llm_duration_ms = (time.perf_counter() - llm_started_at) * 1_000
            stage = "route_resolution"
            route_started_at = time.perf_counter()
            resolved = resolve_route(
                plan,
                allowed_landmark_ids={
                    item.id for item in enriched_context.available_landmarks
                },
            )
            route_duration_ms = (time.perf_counter() - route_started_at) * 1_000
            operations_observer.planning_completed(
                request_id,
                resolved,
                llm_duration_ms=llm_duration_ms,
                route_duration_ms=route_duration_ms,
                total_duration_ms=(time.perf_counter() - total_started_at) * 1_000,
            )
            return resolved
        except RouteNotFound as exc:
            operations_observer.planning_failed(
                request_id,
                stage=stage,
                duration_ms=(time.perf_counter() - total_started_at) * 1_000,
                error=exc,
            )
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (PlannerError, ValueError) as exc:
            operations_observer.planning_failed(
                request_id,
                stage=stage,
                duration_ms=(time.perf_counter() - total_started_at) * 1_000,
                error=exc,
            )
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.get("/v1/routes")
    def list_routes() -> dict[str, object]:
        return route_store.catalog()

    @app.post("/v1/routes", status_code=201)
    def create_route(request: RouteWrite) -> dict[str, object]:
        try:
            return route_store.create(request)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.put("/v1/routes/{route_id}")
    def update_route(route_id: str, request: RouteUpdate) -> dict[str, object]:
        try:
            return route_store.update(route_id, request)
        except RouteNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RouteRevisionConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.delete("/v1/routes/{route_id}", status_code=204)
    def delete_route(route_id: str, revision: int) -> Response:
        try:
            route_store.delete(route_id, revision)
        except RouteNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RouteRevisionConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return Response(status_code=204)

    @app.websocket("/ws/operator")
    async def operator_websocket(websocket: WebSocket) -> None:
        await operator_hub.operator_session(websocket)

    @app.websocket("/ws/ros")
    async def ros_websocket(websocket: WebSocket) -> None:
        await operator_hub.ros_session(websocket)

    return app


app = create_app()
