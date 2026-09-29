"""Dedicated RGB incident diagnostics. No depth, flight or control endpoints.

This process owns one temporal-model session; restart it to change clients.
Client clock values are only ordered, never subtracted from server clocks.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
import math
import os
import threading
import time
import uuid

import cv2
from fastapi import FastAPI, HTTPException, Query, Request
import numpy as np
from starlette.concurrency import run_in_threadpool

from .phase1_inference import PHASE1_EVENT_TYPES, Phase1JetsonRuntime
from .incident_coverage import incident_coverage, incident_event_is_covered


MAX_JPEG_BYTES = 8 * 1024 * 1024


def create_incident_observe_app(runtime=None, *, clock=time.monotonic,
                                max_frame_age_s=.5, max_result_age_s=2.):
    if not (math.isfinite(max_frame_age_s) and 0 < max_frame_age_s <= 2
            and math.isfinite(max_result_age_s) and max_frame_age_s <= max_result_age_s <= 5):
        raise ValueError('Invalid bounded incident diagnostic age limits')
    warmup_needed = runtime is None
    if runtime is None:
        runtime = Phase1JetsonRuntime(
            os.environ.get('JOLGWA_PHASE1_ROOT', '/home/jetson/jolgwa/phase1-demo-local'),
            device=os.environ.get('JOLGWA_PHASE1_DEVICE', '0'))
    @asynccontextmanager
    async def lifespan(_app):
        # Finish loading before accepting live observations; startup frames must
        # not consume a real incident while initialization exceeds the budget.
        if warmup_needed:
            await run_in_threadpool(runtime.warmup)
        yield
    app = FastAPI(title='Jolgwa Incident Observe Only', docs_url=None, redoc_url=None,
                  openapi_url=None, lifespan=lifespan)
    app.state.runtime = runtime
    app.state.session = None
    app.state.last_stamp = 0
    lock = threading.Lock()

    def envelope(stamp, **fields):
        return dict(observation_only=True, purpose='incident_observe',
                    physical_fc_commands=0, frame_timestamp_ns=stamp, **fields)

    @app.get('/health')
    def health():
        return envelope(0, status='ok', phase1=runtime.health(),
                        max_frame_age_s=max_frame_age_s,
                        max_result_age_s=max_result_age_s,
                        session_bound=app.state.session is not None)

    @app.post('/v1/incident-observe')
    async def observe(request: Request, frame_timestamp_ns: int = Query(gt=0),
                      frame_age_s: float = Query(ge=0)):
        started = clock()
        if not math.isfinite(frame_age_s) or frame_age_s > max_frame_age_s:
            raise HTTPException(422, 'incident_frame_stale_or_invalid')
        try:
            session = str(uuid.UUID(request.headers.get('X-Jolgwa-Incident-Session', '')))
        except ValueError as exc:
            raise HTTPException(403, 'incident_session_required') from exc
        if not lock.acquire(blocking=False):
            raise HTTPException(409, 'incident_observer_busy')
        try:
            if app.state.session not in (None, session):
                raise HTTPException(409, 'incident_session_conflict_restart_backend')
            if frame_timestamp_ns <= app.state.last_stamp:
                raise HTTPException(409, 'incident_duplicate_or_regressing_frame')
            payload = bytearray()
            async for chunk in request.stream():
                payload.extend(chunk)
                if len(payload) > MAX_JPEG_BYTES:
                    raise HTTPException(413, 'incident_jpeg_too_large')
            if not payload or payload[:2] != b'\xff\xd8':
                raise HTTPException(422, 'incident_jpeg_required')
            image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.size == 0 or image.shape[0]*image.shape[1] > 4096*2160:
                raise HTTPException(422, 'incident_jpeg_invalid_dimensions')
            if frame_age_s+clock()-started > max_frame_age_s:
                raise HTTPException(422, 'incident_frame_expired_before_inference')
            app.state.session, app.state.last_stamp = session, frame_timestamp_ns
            try:
                result = await run_in_threadpool(runtime.evaluate_bgr, image)
            except Exception as exc:
                # Source was consumed, including on failure: it must not be replayed.
                return envelope(frame_timestamp_ns, status='UNKNOWN',
                                reason='incident_inference_failed', error_type=type(exc).__name__,
                                events=[], elapsed_s=clock()-started)
            elapsed = clock()-started
            if frame_age_s+elapsed > max_result_age_s:
                return envelope(frame_timestamp_ns, status='UNKNOWN',
                                reason='incident_result_expired', events=[], elapsed_s=elapsed)
            coverage, coverage_complete = incident_coverage(
                result.modules, result.detections, getattr(result, 'temporal_window_ready', False))
            events = [dict(event) for event in result.events
                      if event.get('state') == 'CONFIRMED'
                      and event.get('event_type') in PHASE1_EVENT_TYPES
                      and incident_event_is_covered(event.get('event_type'), coverage)]
            # A valid fire event is not lost because an unrelated temporal
            # module is warming up. The overall coverage is still UNKNOWN;
            # consumers must validate each positive event's own dependencies.
            return envelope(frame_timestamp_ns,
                            status='OBSERVED' if coverage_complete else 'UNKNOWN',
                            reason='diagnostic_only' if coverage_complete else 'phase1_coverage_incomplete',
                            elapsed_s=elapsed, events=events,
                            phase1=dict(enabled=True, inference_ms=result.inference_ms,
                                        detections=list(result.detections), modules=result.modules,
                                        temporal_window_ready=getattr(result, 'temporal_window_ready', False),
                                        coverage=coverage, coverage_complete=coverage_complete))
        finally:
            lock.release()

    return app


def main():
    import argparse
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8776)
    args = parser.parse_args()
    uvicorn.run('jolgwa_uav.incident_observe_service:create_incident_observe_app',
                factory=True, host=args.host, port=args.port, workers=1)


if __name__ == '__main__':
    main()
