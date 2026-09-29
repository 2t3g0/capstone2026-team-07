from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence


def normalized_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        bbox = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not (
        0.0 <= bbox[0] < bbox[2] <= 1.0
        and 0.0 <= bbox[1] < bbox[3] <= 1.0
    ):
        return None
    return bbox


def select_target_bbox(
    event: dict[str, Any],
    detections: Iterable[dict[str, Any]],
    *,
    previous_bbox: Sequence[float] | None = None,
) -> tuple[float, float, float, float] | None:
    """Match Phase 1 live detections to a confirmed event."""

    candidates = []
    for detection in detections:
        bbox = normalized_bbox(detection.get("bbox"))
        if bbox is not None:
            candidates.append((detection, bbox))
    if not candidates:
        return None

    wanted_ids = {
        int(value)
        for value in event.get("track_ids", [])
        if str(value).lstrip("-").isdigit() and int(value) >= 0
    }
    if wanted_ids:
        matched = [
            bbox
            for detection, bbox in candidates
            if int(detection.get("track_id", -1)) in wanted_ids
        ]
        if matched:
            return _union(matched)

    if event.get("event_type") == "FIRE_SMOKE":
        fire = [
            (detection, bbox)
            for detection, bbox in candidates
            if str(detection.get("class_name", "")).lower()
            in {"fire", "smoke", "fire_smoke"}
        ]
        if fire:
            candidates = fire

    reference = normalized_bbox(previous_bbox or event.get("bbox"))
    if reference is None:
        return max(
            candidates,
            key=lambda item: float(item[0].get("confidence", 0.0)),
        )[1]
    return max(candidates, key=lambda item: _iou(reference, item[1]))[1]


class JsonlTail:
    """Incrementally read complete JSON objects and survive truncation."""

    def __init__(self, path: str | Path, *, start_at_end: bool = False) -> None:
        self.path = Path(path)
        self.position = (
            self.path.stat().st_size
            if start_at_end and self.path.exists()
            else 0
        )
        self._partial = ""

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        size = self.path.stat().st_size
        if size < self.position:
            self.position = 0
            self._partial = ""
        with self.path.open("r", encoding="utf-8") as stream:
            stream.seek(self.position)
            chunk = stream.read()
            self.position = stream.tell()
        text = self._partial + chunk
        lines = text.splitlines(keepends=True)
        self._partial = ""
        if lines and not lines[-1].endswith(("\n", "\r")):
            self._partial = lines.pop()
        result = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                result.append(value)
        return result


def latest_phase1_run(root: str | Path) -> Path | None:
    directory = Path(root)
    if not directory.exists():
        return None
    candidates = [
        path
        for path in directory.iterdir()
        if path.is_dir() and (path / "events.jsonl").exists()
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def _union(
    boxes: Sequence[tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _iou(
    first: Sequence[float], second: Sequence[float]
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(
        0.0, first[3] - first[1]
    )
    second_area = max(0.0, second[2] - second[0]) * max(
        0.0, second[3] - second[1]
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0
