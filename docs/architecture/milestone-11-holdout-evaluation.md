# Milestone 11: Generalization and Holdout Evaluation

## Purpose

Prompt v3 was developed from failures observed on the Milestone 10 benchmark. Its improvement on
that same set cannot distinguish general reasoning gains from benchmark-specific adaptation.
Milestone 11 introduces a frozen holdout to measure that distinction without changing either
investigator.

## Frozen experimental boundary

The holdout was created after prompt v3. Before its first real-provider evaluation, the following
components are frozen:

- deterministic diagnosis rules;
- LLM investigator and prompts v1, v2, and v3;
- LLM-facing evidence serialization;
- structured response schema;
- reference and source-coverage validators;
- diagnosis identifiers;
- providers and execution flow;
- evaluation, comparison, and regression semantics.

No holdout case is included by the Milestone 10 scenario file, and no existing scenario was edited.

## Holdout design

`tests/fixtures/evaluation_scenarios_holdout_m11.json` contains 20 scenarios:

- 10 supported-diagnosis cases;
- 10 required-abstention cases;
- robustness dimensions overlapping both groups.

The supported cases cover all five current diagnosis identifiers with new language and evidence
arrangements. Abstention cases cover unsupported TLS, network-policy, disk, permission, dependency,
and unrelated configuration failures as well as incomplete, contradictory, and merely symptomatic
evidence.

Each scenario can carry ordered `robustness_categories`. The holdout represents:

- `wording_variation`;
- `evidence_reordering`;
- `distractor`;
- `missing_evidence`;
- `conflicting_evidence`;
- `unsupported_cause`;
- `supported_generalization`;
- `duplicate_evidence`.

The metadata is descriptive and does not affect investigation or scoring.

## Supplementary error reporting

Evaluation retains all existing metrics and adds one deterministic primary error category for a
failed run where structured fields support classification:

- `false_diagnosis`;
- `unnecessary_abstention`;
- `wrong_diagnosis`;
- `invalid_evidence_reference`;
- `missing_required_source`;
- `provider_failure`;
- `invalid_structured_response`;
- `not_evaluated`.

Provider and validation failures take precedence over semantic categories. Categories are derived
from execution status, expected outcome, actual diagnosis or abstention, and structured validation
errors—not answer prose and not another model. They supplement rather than replace current metrics.

## Experiment identity and planned comparison

Existing experiment metadata already persists the scenario source path and ordered scenario IDs.
Experiment inspection now displays that source, which distinguishes Milestone 10 development runs
from Milestone 11 holdout runs without a schema migration.

The first holdout comparison will use:

1. the frozen deterministic investigator;
2. the Groq investigator using `llama-3.3-70b-versatile` and frozen prompt v3.

No real-provider holdout result is documented before that explicit run. The comparison will help
decide whether the evidence supports keeping the current deterministic boundary, expanding the LLM
role, or evaluating a narrow hybrid policy. It does not establish a need for agents, LangGraph, or
multi-agent orchestration.

## Provider request pacing

Live holdout runs may use explicit pacing to remain within an external provider's token-per-minute
limit:

```bash
python -m ai_investigation.evaluate \
  --investigator llm \
  --prompt-version v3 \
  --scenarios tests/fixtures/evaluation_scenarios_holdout_m11.json \
  --request-delay-seconds 4 \
  --save-experiment \
  --tag milestone-11-holdout
```

Pacing is evaluation infrastructure, not an investigator policy. No delay occurs before the first
provider request or for scenarios resolved locally without a model call. The configured value is
stored in the existing experiment configuration and shown during inspection when non-zero. It is
not compared as a quality metric and does not alter prompts, semantics, validators, or provider
failure handling.
