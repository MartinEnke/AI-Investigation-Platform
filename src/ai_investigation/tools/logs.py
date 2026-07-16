"""Local JSON implementation of the log evidence tool."""

import json
from pathlib import Path

from ai_investigation.tools.protocols import JsonRecord


class JsonLogTool:
    def __init__(self, fixture_path: Path) -> None:
        with fixture_path.open(encoding="utf-8") as fixture:
            data = json.load(fixture)
        if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
            raise ValueError(f"Expected a list of objects in {fixture_path}")
        self._logs: list[JsonRecord] = data

    def get_logs(self, deployment_id: str) -> list[JsonRecord]:
        return [item for item in self._logs if item.get("deployment_id") == deployment_id]

