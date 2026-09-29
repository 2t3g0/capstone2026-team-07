from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .gemini import GeminiPlanner, PlannerError
from .models import AvailableRoute, MissionPlan, PlanStatus, PlanningContext
from .operator_hub import OperatorHub
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


def create_app(
    planner: GeminiPlanner | None = None,
    routes: RouteStore | None = None,
) -> FastAPI:
    app = FastAPI(title="Jolgwa UAV Mission Planner", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["Content-Type"],
    )
    operator_hub = OperatorHub()
    app.state.operator_hub = operator_hub
    project_root = Path(__file__).resolve().parents[2]
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
    active_planner = planner

    def get_planner() -> GeminiPlanner:
        nonlocal active_planner
        if active_planner is None:
            active_planner = GeminiPlanner(
                model=os.getenv("GEMINI_MODEL", "gemini-3.7-flash"),
                thinking_level=os.getenv("GEMINI_THINKING_LEVEL", "medium"),
            )
        return active_planner

    def planning_context(source: PlanningContext | None) -> PlanningContext:
        context = source or PlanningContext()
        routes = [
            AvailableRoute.model_validate(route)
            for route in route_store.route_summaries()
        ]
        return context.model_copy(update={"available_routes": routes})

    def resolve_route(plan: MissionPlan) -> MissionPlan:
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

    @app.post("/v1/plan/text", response_model=MissionPlan)
    def plan_text(request: TextPlanRequest) -> MissionPlan:
        try:
            plan = get_planner().plan_text(
                request.command, planning_context(request.context)
            )
            return resolve_route(plan)
        except RouteNotFound as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (PlannerError, ValueError) as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post("/v1/plan/audio", response_model=MissionPlan)
    async def plan_audio(
        audio: UploadFile = File(...),
        context_json: str | None = Form(default=None),
    ) -> MissionPlan:
        try:
            context = (
                PlanningContext.model_validate_json(context_json)
                if context_json
                else None
            )
            data = await audio.read()
            plan = get_planner().plan_audio(
                data, audio.content_type or "audio/wav", planning_context(context)
            )
            return resolve_route(plan)
        except RouteNotFound as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (PlannerError, ValueError) as exc:
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
