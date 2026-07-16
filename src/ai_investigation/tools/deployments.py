"""Local JSON implementation of the deployment evidence tool."""

import json
from pathlib import Path

from ai_investigation.tools.protocols import JsonRecord


class JsonDeploymentTool:
    def __init__(self, fixture_path: Path) -> None:
        self._deployments = _load_records(fixture_path)

    def get_deployment(self, deployment_id: str) -> JsonRecord | None:
        return next(
            (item for item in self._deployments if item.get("id") == deployment_id),
            None,
        )


def _load_records(path: Path) -> list[JsonRecord]:
    with path.open(encoding="utf-8") as fixture:
        data = json.load(fixture)
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError(f"Expected a list of objects in {path}")
    return data

