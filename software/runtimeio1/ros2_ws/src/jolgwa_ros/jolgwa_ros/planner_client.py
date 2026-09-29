import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class PlannerClientError(RuntimeError):
    pass


class PlannerClient:
    def __init__(self, endpoint: str, timeout_s: float = 30.0) -> None:
        self.endpoint = endpoint
        self.timeout_s = timeout_s

    def plan_text(self, command: str) -> dict:
        payload = json.dumps({"command": command}).encode("utf-8")
        request = Request(
            self.endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise PlannerClientError(
                "planner returned HTTP {}: {}".format(exc.code, detail)
            ) from exc
        except (URLError, TimeoutError) as exc:
            raise PlannerClientError("planner request failed: {}".format(exc)) from exc

        try:
            result = json.loads(body)
        except json.JSONDecodeError as exc:
            raise PlannerClientError("planner returned invalid JSON") from exc
        if not isinstance(result, dict) or "status" not in result:
            raise PlannerClientError("planner response has no mission status")
        return result
