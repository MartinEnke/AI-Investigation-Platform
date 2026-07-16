"""Local JSON implementation of the service-health evidence tool."""

import json
from pathlib import Path

from ai_investigation.tools.protocols import JsonRecord


class JsonServiceHealthTool:
    def __init__(self, fixture_path: Path) -> None:
        with fixture_path.open(encoding="utf-8") as fixture:
            data = json.load(fixture)
        if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
            raise ValueError(f"Expected a list of objects in {fixture_path}")
        self._health_records: list[JsonRecord] = data

    def get_service_health(self, service: str) -> JsonRecord | None:
        return next(
            (item for item in self._health_records if item.get("service") == service),
            None,
        )

