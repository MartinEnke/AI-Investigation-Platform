"""Offline-testable prompt diagnostic; live execution is explicit and isolated."""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Literal, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv

from ai_investigation.evaluation.loader import DIAGNOSIS_IDS, load_scenarios
from ai_investigation.evaluation.models import EvaluationScenario
from ai_investigation.evidence import CollectedEvidence, EvidenceCollector
from ai_investigation.groq_model import (
    DEFAULT_GROQ_MODEL,
    GROQ_CHAT_COMPLETIONS_URL,
    GROQ_USER_AGENT,
)
from ai_investigation.investigator import request_from_question
from ai_investigation.tools import JsonDeploymentTool, JsonLogTool, JsonServiceHealthTool
from ai_investigation.uncertainty_investigator import (
    CANDIDATE_SEMANTICS_PROMPT_SELECTION,
    CANDIDATE_SEMANTICS_PROMPT_VERSION,
    build_uncertainty_prompt,
    parse_candidate_semantics_proposal,
    serialize_uncertainty_evidence,
)

Variant = Literal["a", "b", "c", "d"]
VARIANTS: tuple[Variant, ...] = ("a", "b", "c", "d")
VARIANT_IDENTIFIERS = {
    "a": "plain-classification",
    "b": "forced-diagnosis-comparison",
    "c": "minimal-json",
    "d": CANDIDATE_SEMANTICS_PROMPT_VERSION,
}


class DiagnosticModel(Protocol):
    def generate(self, prompt: str, *, json_mode: bool) -> str:
        """Generate diagnostic text without invoking investigation or policy."""


@dataclass(frozen=True, slots=True)
class DiagnosticResult:
    scenario_id: str
    variant: Variant
    prompt_identifier: str
    raw_response: str
    supported_diagnoses: tuple[str, ...]
    rejected_diagnoses: tuple[str, ...]
    not_relevant_diagnoses: tuple[str, ...]
    parse_status: str


class GroqDiagnosticModel:
    """Diagnostic-only Groq transport supporting plain text and JSON output."""

    def __init__(self, api_key: str, model: str, timeout_seconds: float = 30.0) -> None:
        if not api_key:
            raise ValueError("GROQ_API_KEY is required for the live diagnostic.")
        if not model:
            raise ValueError("A Groq model name is required.")
        self._api_key = api_key
        self._model = model
        self._timeout_seconds = timeout_seconds

    def generate(self, prompt: str, *, json_mode: bool) -> str:
        payload: dict[str, object] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        request = Request(
            GROQ_CHAT_COMPLETIONS_URL,
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": GROQ_USER_AGENT,
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:  # noqa: S310
                raw = response.read()
        except HTTPError as error:
            detail = _error_body(error)
            raise RuntimeError(
                f"Groq diagnostic failed with HTTP {error.code}"
                + (f": {detail}" if detail else ".")
            ) from error
        except (URLError, OSError) as error:
            raise RuntimeError("Groq diagnostic request failed.") from error
        try:
            response_value = json.loads(raw.decode("utf-8"))
            content = response_value["choices"][0]["message"]["content"]
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
            raise RuntimeError("Groq diagnostic returned an invalid API response.") from error
        if not isinstance(content, str):
            raise RuntimeError("Groq diagnostic returned non-text content.")
        return content


def run_diagnostic(
    scenarios: tuple[EvaluationScenario, ...],
    scenario_id: str,
    collector: EvidenceCollector,
    model: DiagnosticModel,
    variants: tuple[Variant, ...] = VARIANTS,
) -> tuple[DiagnosticResult, ...]:
    scenario = next((item for item in scenarios if item.id == scenario_id), None)
    if scenario is None:
        raise ValueError(f"Unknown scenario ID: {scenario_id}.")
    collected = collector.collect(request_from_question(scenario.question))
    payload = serialize_uncertainty_evidence(collected)
    results = []
    for variant in variants:
        prompt = build_variant_prompt(variant, collected, payload)
        raw = model.generate(prompt, json_mode=variant in ("c", "d"))
        results.append(parse_variant_result(scenario_id, variant, raw))
    return tuple(results)


def build_variant_prompt(
    variant: Variant,
    collected: CollectedEvidence,
    evidence_payload: dict[str, object] | None = None,
) -> str:
    payload = evidence_payload or serialize_uncertainty_evidence(collected)
    evidence = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    diagnoses = ", ".join(sorted(DIAGNOSIS_IDS))
    shared = (
        "Supported candidate: a diagnosis with direct positive evidence that could independently "
        "explain the failure. Rejected hypothesis: a diagnosis considered during reasoning but "
        "lacking direct support, contradicted by evidence, only representing a symptom, or weaker "
        "than another supported cause. Do not classify every considered diagnosis as supported. "
        "Absence of contradiction is not support. Shared vocabulary is not support. "
        f"Allowed diagnoses: {diagnoses}. Use only exact evidence IDs shown below."
    )
    if variant == "a":
        return (
            "Diagnostic variant A: plain classification. No JSON. "
            + shared
            + " Return exactly these sections: SUPPORTED CANDIDATES, REJECTED HYPOTHESES, "
            "EXPLANATION.\n\nEvidence:\n"
            + evidence
        )
    if variant == "b":
        return (
            "Diagnostic variant B: forced comparison. No JSON. "
            + shared
            + " For every allowed diagnosis, output exactly one classification: SUPPORTED, "
            "REJECTED, or NOT RELEVANT, followed by a one-sentence reason and exact evidence IDs. "
            "Do not omit any diagnosis.\n\nEvidence:\n"
            + evidence
        )
    if variant == "c":
        contract = {
            "supported_candidates": [
                {
                    "diagnosis_id": "...",
                    "evidence_references": ["E1"],
                    "reason": "...",
                }
            ],
            "rejected_hypotheses": [
                {
                    "diagnosis_id": "...",
                    "evidence_references": ["E2"],
                    "reason": "...",
                }
            ],
        }
        return (
            "Diagnostic variant C: minimal JSON. "
            + shared
            + " Return only this JSON shape, without confidence, evidence strength, flags, source "
            f"metadata, or policy concepts: {json.dumps(contract, separators=(',', ':'))}"
            "\n\nEvidence:\n"
            + evidence
        )
    if variant == "d":
        return build_uncertainty_prompt(
            collected, CANDIDATE_SEMANTICS_PROMPT_SELECTION
        )
    raise ValueError(f"Unknown diagnostic variant: {variant}.")


def parse_variant_result(
    scenario_id: str, variant: Variant, raw_response: str
) -> DiagnosticResult:
    try:
        if variant == "a":
            supported, rejected = _parse_section_response(raw_response)
            not_relevant: tuple[str, ...] = ()
        elif variant == "b":
            supported, rejected, not_relevant = _parse_forced_response(raw_response)
        elif variant == "c":
            supported, rejected = _parse_minimal_json(raw_response)
            not_relevant = ()
        else:
            proposal = parse_candidate_semantics_proposal(raw_response)
            supported = tuple(item.diagnosis_id for item in proposal.candidates)
            rejected = tuple(item.diagnosis_id for item in proposal.rejected_hypotheses)
            not_relevant = ()
        status = "parsed"
    except ValueError as error:
        supported = rejected = not_relevant = ()
        status = f"parse_error: {error}"
    return DiagnosticResult(
        scenario_id=scenario_id,
        variant=variant,
        prompt_identifier=VARIANT_IDENTIFIERS[variant],
        raw_response=raw_response,
        supported_diagnoses=supported,
        rejected_diagnoses=rejected,
        not_relevant_diagnoses=not_relevant,
        parse_status=status,
    )


def render_diagnostic(results: tuple[DiagnosticResult, ...]) -> str:
    lines: list[str] = ["Uncertainty Prompt Diagnostic", "=============================", ""]
    for result in results:
        lines.extend(
            (
                f"Scenario: {result.scenario_id}",
                f"Variant: {result.variant.upper()}",
                f"Prompt identifier: {result.prompt_identifier}",
                "Raw response:",
                result.raw_response,
                "Parsed supported diagnoses: "
                + (", ".join(result.supported_diagnoses) or "none"),
                "Parsed rejected diagnoses: "
                + (", ".join(result.rejected_diagnoses) or "none"),
                "Parsed not-relevant diagnoses: "
                + (", ".join(result.not_relevant_diagnoses) or "none"),
                f"Parse status: {result.parse_status}",
                "",
            )
        )
    lines.extend(("Comparison", "----------"))
    for result in results:
        lines.append(
            f"Variant {result.variant.upper()}: supported={len(result.supported_diagnoses)}, "
            f"rejected={len(result.rejected_diagnoses)}, "
            f"not-relevant={len(result.not_relevant_diagnoses)}"
        )
    return "\n".join(lines) + "\n"


def main(
    argv: Sequence[str] | None = None,
    *,
    model: DiagnosticModel | None = None,
) -> int:
    load_dotenv(Path.cwd() / ".env", override=False)
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Diagnose uncertainty prompt semantics.")
    parser.add_argument("--provider", choices=("groq",), default="groq")
    parser.add_argument(
        "--model",
        default=os.environ.get("AI_INVESTIGATION_MODEL") or DEFAULT_GROQ_MODEL,
    )
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=root / "tests" / "fixtures" / "evaluation_scenarios_m10.json",
    )
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--variant", choices=(*VARIANTS, "all"), default="all")
    parser.add_argument("--fixtures", type=Path, default=root / "tests" / "fixtures")
    args = parser.parse_args(argv)

    selected_model = model or GroqDiagnosticModel(
        os.environ.get("GROQ_API_KEY", ""), args.model
    )
    collector = EvidenceCollector(
        JsonDeploymentTool(args.fixtures / "deployments.json"),
        JsonLogTool(args.fixtures / "logs.json"),
        JsonServiceHealthTool(args.fixtures / "service_health.json"),
    )
    selected_variants = VARIANTS if args.variant == "all" else (args.variant,)
    results = run_diagnostic(
        load_scenarios(args.scenarios),
        args.scenario_id,
        collector,
        selected_model,
        selected_variants,
    )
    print(render_diagnostic(results), end="")
    return 0


def _parse_section_response(raw: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    upper = raw.upper()
    headings = ("SUPPORTED CANDIDATES", "REJECTED HYPOTHESES", "EXPLANATION")
    positions = tuple(upper.find(heading) for heading in headings)
    if any(position < 0 for position in positions) or positions != tuple(sorted(positions)):
        raise ValueError("Required plain-text sections were not found in order.")
    supported_text = raw[positions[0] : positions[1]]
    rejected_text = raw[positions[1] : positions[2]]
    return _diagnoses_in(supported_text), _diagnoses_in(rejected_text)


def _parse_forced_response(
    raw: str,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    classifications: dict[str, str] = {}
    for line in raw.splitlines():
        normalized = line.casefold()
        for diagnosis in DIAGNOSIS_IDS:
            if diagnosis in normalized:
                remainder = normalized.split(diagnosis, 1)[1].lstrip(" :-—|\t")
                for label in ("not relevant", "supported", "rejected"):
                    if remainder.startswith(label):
                        classifications[diagnosis] = label
                        break
    if set(classifications) != DIAGNOSIS_IDS:
        raise ValueError("Every diagnosis must have exactly one recognizable classification.")
    return (
        tuple(sorted(key for key, value in classifications.items() if value == "supported")),
        tuple(sorted(key for key, value in classifications.items() if value == "rejected")),
        tuple(sorted(key for key, value in classifications.items() if value == "not relevant")),
    )


def _parse_minimal_json(raw: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("Minimal response is not valid JSON.") from error
    if not isinstance(value, dict) or set(value) != {
        "supported_candidates",
        "rejected_hypotheses",
    }:
        raise ValueError("Minimal response has invalid top-level fields.")
    return (
        _parse_minimal_items(value["supported_candidates"], "supported_candidates"),
        _parse_minimal_items(value["rejected_hypotheses"], "rejected_hypotheses"),
    )


def _parse_minimal_items(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list.")
    diagnoses = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {
            "diagnosis_id",
            "evidence_references",
            "reason",
        }:
            raise ValueError(f"{field} contains an invalid item.")
        diagnosis = item["diagnosis_id"]
        references = item["evidence_references"]
        reason = item["reason"]
        if diagnosis not in DIAGNOSIS_IDS:
            raise ValueError(f"{field} contains an unsupported diagnosis.")
        if not isinstance(references, list) or not all(
            isinstance(reference, str)
            and re.fullmatch(r"E[1-9][0-9]*", reference) is not None
            for reference in references
        ):
            raise ValueError(f"{field} contains invalid evidence references.")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"{field} contains an invalid reason.")
        diagnoses.append(diagnosis)
    if len(diagnoses) != len(set(diagnoses)):
        raise ValueError(f"{field} contains duplicate diagnoses.")
    return tuple(diagnoses)


def _diagnoses_in(text: str) -> tuple[str, ...]:
    lowered = text.casefold()
    return tuple(sorted(diagnosis for diagnosis in DIAGNOSIS_IDS if diagnosis in lowered))


def _error_body(error: HTTPError) -> str | None:
    try:
        body = error.read()
    except Exception:
        return None
    return body.decode("utf-8", errors="replace").strip() or None


if __name__ == "__main__":
    raise SystemExit(main())
