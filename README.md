# AI Investigation Platform

![Python 3.13+](https://img.shields.io/badge/Python-3.13%2B-3776AB)
![Tests](https://img.shields.io/badge/tests-166%20passing-2E8B57)
![Evaluation scenarios](https://img.shields.io/badge/evaluations-16%20scenarios-2E8B57)
![Milestone 10](https://img.shields.io/badge/milestone-10-6C63FF)
![Architecture](https://img.shields.io/badge/architecture-deterministic--first-555555)

Exploring how reliable AI systems are engineered—through deterministic evidence collection, explicit evaluation, progressive orchestration, and explainable investigations.

**What turns an AI demo into a reliable AI system?**

Most AI projects demonstrate what a language model can do. Far fewer explore how AI systems should be engineered: how evidence is collected, how uncertainty is represented, how decisions are evaluated, and how more capable components can be introduced without losing control of the system around them.

After building several AI-powered applications, I became increasingly interested in what happens beyond the model call. This repository is a practical study of that engineering layer. It begins with a deliberately small deterministic baseline so that every later architectural decision can be compared against behavior that is understandable, reproducible, and tested.

The investigation domain is the vehicle. Reliable AI engineering is the destination.

## Why an Investigation Platform?

Deployment-failure investigation is a compact but useful systems problem. It is:

- small enough to understand end to end;
- complex enough to require evidence from multiple sources;
- suitable for competing root-cause hypotheses;
- dependent on uncertainty, missing evidence, and abstention;
- testable through controlled synthetic scenarios;
- a practical foundation for studying future orchestration and probabilistic reasoning.

An investigation cannot be reliable merely because it produces a plausible answer. It must show which evidence was available, distinguish a supported conclusion from an unsupported one, detect conflicting explanations, and remain stable as the architecture evolves.

That makes this domain a useful miniature of a broader AI system.

## Current Implementation

Milestone 10 adds a controlled probabilistic investigator as an experimental alternative to the
deterministic baseline. It changes the reasoning mechanism while preserving evidence collection,
validation, evaluation, experiment tracking, and scenario comparison.

It currently includes:

- fixture-backed deployment, error-log, and service-health evidence collection;
- typed, immutable, slotted domain models;
- protocol-based tool contracts with local JSON implementations;
- deterministic evidence filtering and ordering;
- evaluation of every supported diagnosis rule;
- conclusive results only when exactly one rule matches;
- explicit abstention when no rule matches;
- explicit conflict abstention when multiple rules match;
- deterministic confidence and ordered limitations;
- an immutable, machine-readable decision trace;
- shared deterministic evidence collection for both reasoning paths;
- strict structured LLM parsing and evidence-reference validation;
- a common investigator contract over already-collected evidence;
- isolated Gemini and Groq structured-model adapters;
- a versioned, deterministic LLM prompt and strict response schema;
- a command-line interface;
- structured scenario results, aggregate metrics, and text or JSON evaluation reports;
- local experiment metadata, stage events, timing, persistence, inspection, and comparison;
- typed scenario changes, failure categories, multidimensional deltas, and recommendations;
- 166 passing tests (plus one opt-in provider test skipped by default);
- 11 established regression scenarios and 5 Milestone 10 generalization scenarios.

The supported diagnosis rules are intentionally narrow:

1. Health-check timeout
2. Missing required environment variable
3. Database migration failure

The deterministic CLI, tests, and evaluation make no network or model calls. Real Gemini and Groq
execution is optional and credential-gated. There are no agents, graph frameworks, retries,
provider routing, or hidden probabilistic fallbacks.

## Architecture

The design keeps evidence access, interpretation, trace construction, validation, and public
presentation separate enough to test each responsibility directly.

```text
CLI
 │
 v
Evidence Collection ──> CollectedEvidence
                       ├──> Deployment Tool
                       ├──> Log Tool
                       └──> Service Health Tool
                            │
              ┌─────────────┴─────────────┐
              v                           v
Deterministic Diagnosis             LLM Interpretation
       │                                  │
       v                                  v
Decision Trace                    Parsing + Reference Validation
       │                                  │
       └──────────────> Investigation Result
```

The tool layer exposes small protocols rather than binding investigation logic to JSON files. The current adapters are local and read-only; diagnosis rules depend only on their evidence context.

## Investigation Flow

Every supported rule is evaluated. The system never silently selects the first matching rule.

```text
Question ──> Parse request ──> Collect ordered evidence
                                      │
                                      v
                              Evaluate all rules
                                      │
                                      v
                              Build decision trace
                                      │
              ┌───────────────────────┼───────────────────────┐
              v                       v                       v
         single match              no match            multiple matches
         report cause              abstain              conflict abstention
```

Error logs are restricted to the requested deployment and records whose level is exactly `error`. All relevant error records are inspected in fixture order and remain understandable in the public evidence sequence.

## Structured Decision Trace

Milestone 5 adds explanation without generating a reasoning narrative.

The public `InvestigationResult` can include an immutable `DecisionTrace` containing:

- every evaluated rule in declaration order;
- every named condition in rule-defined order;
- whether each condition was true or false;
- the ordered identifiers of matching rules;
- one aggregate decision outcome:
  - `single_match`
  - `no_match`
  - `multiple_matches`

Rules compute each predicate once. The same boolean facts determine the match and populate the trace, preventing a separate explanation layer from drifting away from diagnosis behavior.

This structure supports reproducibility, auditing, direct tests, and eventual comparison with a probabilistic investigator. It does not duplicate the public answer, expose raw logs as metadata, or generate free-form internal reasoning.

## Experimentation and Evaluation

The repository includes a reusable evaluation framework in addition to unit tests. Its purpose is
to protect deterministic behavior and measure probabilistic behavior across the same controlled
scenarios.

> The question is not only whether the code runs, but whether the system still behaves as intended
> after architectural changes.

The 11 controlled synthetic scenarios cover:

- each supported diagnosis;
- alternative and case-insensitive supported evidence;
- a diagnostic record appearing after an unrelated error;
- missing log or service-health evidence;
- complete evidence that matches no supported diagnosis;
- a generic database error without migration context;
- conflicting supported patterns;
- stable ordered evidence sources across multiple logs.

Each scenario declares a stable expected diagnosis category or abstention. Every run records
execution status, semantic correctness, abstention behavior, evidence-source coverage, structural
and reference validity where applicable, confidence, errors, and local monotonic-clock latency.
Reports retain these individual dimensions instead of reducing them to one score.

```text
Benchmark Scenario
        │
        v
EvidenceCollector
        │
        v
CollectedEvidence
        │
        v
Investigator ──> Validation ──> Evaluation
                                      │
                                      v
                               Scenario Result
                                      │
                                      v
                               Aggregate Report
```

The original regression schema still compares root cause, conclusive status, and ordered evidence
sources. Scenarios may also opt into exact checks for:

- confidence;
- ordered limitations;
- decision outcome;
- ordered matched-rule identifiers.

Optional expectations keep older scenarios valid while allowing important cases—especially
conflicts—to make stronger assertions. Semantic correctness is evaluated deterministically from
stable diagnosis IDs; no model judges another model. Agreement is reported independently because
two investigators can agree and still be wrong. Model confidence is self-reported and
uncalibrated.

Milestone 6 used that baseline for one unchanged `gemini-2.5-flash` comparison run. The complete
scenario-level results and limitations are preserved in the
[Milestone 6 benchmark report](docs/experiments/milestone-06-gemini-benchmark.md).

### Controlled Probabilistic Investigator

Milestone 10 introduces the LLM path now because exact deterministic rules do not generalize
automatically to unfamiliar language or multi-step causal evidence. The deterministic investigator
remains intact: probabilistic reasoning must demonstrate measured improvements without adding
unsupported conclusions, unsafe failures to abstain, or invalid evidence references.

Both implementations satisfy one application-facing contract over the same immutable
`CollectedEvidence`. The LLM receives only the question, deployment identifier, numbered evidence,
supported decision IDs, and response schema. It never receives benchmark expectations,
deterministic output, or comparison results.

The prompt is pure, deterministic, and identified as `llm-investigator-v1`. Structured output is
validated again in application code even when a provider offers JSON constraints. Unknown
diagnoses, contradictory fields, invalid confidence, malformed JSON, and nonexistent references
remain explicit failures. No retries are used because raw provider reliability is itself an
experimental result.

Groq is a single small transport adapter behind the existing `StructuredModel` protocol, not a
provider platform. The extended 16-scenario set adds semantic rephrasing, multi-step causal
synthesis, unsupported evidence, conflicting evidence, and distractors without rewriting the
original 11 cases. The deterministic path scores 14/16 on semantic correctness in that extended
set; no real Groq benchmark result is claimed here.

## Local Experiment Tracking

Evaluation answers whether behavior met expectations. Observability records how that evaluation
was executed: its configuration, stage lifecycle, timing, provider outcome, and stored artifacts.
Milestone 8 keeps those concerns local and optional.

```text
Evaluation Configuration
        │
        v
Experiment Run Context
        │
        v
Scenario Execution
        ├── Evidence Collection
        ├── Investigation
        ├── Validation
        └── Evaluation
        │
        v
Structured Events ──> Experiment Record ──> Local Experiment Store
                                                │
                                                v
                                        List / Inspect / Compare
```

A saved run creates:

```text
experiments/runs/<experiment-id>/
├── experiment.json
├── report.txt
└── events.jsonl
```

`experiment.json` composes the existing evaluation report with reproducibility metadata and a
timing summary. `events.jsonl` contains small sequenced lifecycle events; it does not persist API
keys, prompts, raw provider responses, or complete evidence payloads. Generated run directories
are ignored by Git.

Durations use a monotonic clock. Evidence collection, deterministic interpretation, model
interpretation, evaluation, scenario total, and experiment total are measured around existing
boundaries. Model parsing and reference validation currently occur within the LLM investigator,
so their lifecycle is recorded but their duration is not presented as a separately measurable
value.

## Scenario-Level Regression Analysis

Aggregate accuracy cannot reveal which behavior changed. Two experiments can both score 10/11
while fixing one scenario and breaking another. Milestone 9 therefore matches stored results by
stable scenario ID and classifies each shared case as improved, regressed, unchanged correct,
unchanged incorrect, or not comparable.

Comparison preserves semantic correctness, abstention behavior, structural and evidence-reference
validity, execution status, investigator agreement, and latency as separate dimensions. Agreement
never substitutes for correctness. One deterministic primary failure category distinguishes wrong
diagnoses, failed or unnecessary abstention, provider failures, invalid responses, invalid
references, and other structural failures.

The recommendation policy is deliberately narrow: any semantic scenario regression produces a
warning, and improvements never cancel regressions. Optional dimensions and latency remain visible
trade-offs rather than arbitrary acceptance weights. The regression gate exits with status `1`
only for semantic regressions; missing or malformed experiments remain distinct errors.

## Engineering Principles

- Evidence before conclusions
- Deterministic where possible
- Explicit uncertainty over fabricated certainty
- Evaluation before optimization
- Prefer explicit workflows over hidden reasoning
- Every abstraction must earn its place
- Introduce complexity only when justified
- Keep tools replaceable and business logic independent

## Engineering Decisions

### Why deterministic first?

A known baseline makes future probabilistic behavior measurable. Before asking whether a more capable system is better, the project needs stable outcomes, explicit failure cases, and repeatable evaluation.

### Why explicit abstention?

Available evidence can support one diagnosis, support none, or support several conflicting diagnoses. Treating those states as equivalent would hide important uncertainty. The investigator therefore returns no root cause when its evidence cannot justify exactly one supported conclusion.

### Why structured decision traces?

Stable identifiers and boolean condition results are easier to test, compare, and audit than generated explanation narratives. They expose decision facts without making presentation prose part of rule evaluation.

### Why protocol-based tools?

The investigation logic depends on small evidence-source contracts. Local fixtures can later be replaced by APIs, CI/CD systems, databases, cloud platforms, or MCP servers without requiring the diagnosis rules to know where evidence originated.

### Why not simply ask an LLM?

Evidence collection and evidence interpretation are different responsibilities. A model may later help interpret evidence, but it should not silently become the evidence source or erase the distinction between observed facts and generated conclusions.

### Why progressive architecture?

Frameworks solve real coordination problems, but they also add operational and conceptual cost. This project introduces an abstraction only when the current implementation demonstrates a need for it, keeping each milestone reviewable and its behavior measurable.

## Example Investigation

Question:

```text
Why did deployment deploy-1042 fail?
```

Deterministic CLI output:

```text
Investigation
-------------
Question: Why did deployment deploy-1042 fail?
Investigator: deterministic

Collected Evidence
------------------
1. [deployment] deploy-1042 has status failed and failed stage health_check.
2. [logs] Health check timed out after 30 seconds.
3. [service_health] checkout-api was unhealthy: Readiness endpoint returned HTTP 503.

Deterministic Conclusion
------------------------
Answer: Deployment deploy-1042 failed during its health check. The deployment health check timed out because the target service was unhealthy.
Diagnosis: The deployment health check timed out because the target service was unhealthy.
Confidence: 1.00
Decision outcome: single_match
Matched rules: health_check_timeout

Deterministic Evidence References
---------------------------------
1, 2, 3

Deterministic Validation
------------------------
Deterministic rule evaluation completed.
```

The CLI also renders the ordered rule and condition evaluations. The same trace remains available programmatically:

```text
outcome: single_match
matched_rule_ids:
  - health_check_timeout

selected rule conditions:
  deployment_status_is_failed: true
  failed_stage_is_health_check: true
  error_log_reason_is_timeout: true
  service_health_is_unhealthy: true
```

Limitations are empty for this supported diagnosis. Inconclusive results render their applicable limitations through the existing CLI.

## Repository Structure

```text
.
├── AGENTS.md
├── README.md
├── pyproject.toml
├── docs/
│   ├── architecture/
│   │   ├── milestone-06-deterministic-vs-llm.md
│   │   └── milestone-10-controlled-probabilistic-investigator.md
│   └── experiments/
│       ├── milestone-06-gemini-benchmark.md
│       ├── milestone-7-deterministic-baseline.md
│       ├── milestone-8-local-observability.md
│       └── milestone-9-scenario-regression-analysis.md
├── src/
│   └── ai_investigation/
│       ├── __init__.py
│       ├── cli.py
│       ├── diagnosis.py
│       ├── evaluate.py
│       ├── experiments.py
│       ├── evidence.py
│       ├── gemini_model.py
│       ├── groq_model.py
│       ├── investigator.py
│       ├── investigators.py
│       ├── llm_investigator.py
│       ├── models.py
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── comparison.py
│       │   ├── framework.py
│       │   ├── loader.py
│       │   ├── models.py
│       │   ├── runner.py
│       │   └── tracking.py
│       └── tools/
│           ├── __init__.py
│           ├── deployments.py
│           ├── logs.py
│           ├── protocols.py
│           └── service_health.py
└── tests/
    ├── conftest.py
    ├── test_comparison.py
    ├── test_cli.py
    ├── test_diagnosis.py
    ├── test_evaluation.py
    ├── test_evidence.py
    ├── test_experiment_evaluation.py
    ├── test_experiment_tracking.py
    ├── test_gemini_model.py
    ├── test_groq_integration.py
    ├── test_groq_model.py
    ├── test_investigator.py
    ├── test_llm_investigator.py
    ├── test_milestone10.py
    ├── test_tools.py
    └── fixtures/
        ├── deployments.json
        ├── evaluation_scenarios.json
        ├── evaluation_scenarios_m10.json
        ├── logs.json
        └── service_health.json
```

## Setup and Usage

Python 3.13 or newer is required.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Run an investigation:

```bash
python -m ai_investigation.cli "Why did deployment deploy-1042 fail?"
```

## Run an Interactive Investigation

Select the deterministic path directly:

```bash
python -m ai_investigation.cli \
  --investigator deterministic \
  "Why did deployment deploy-1042 fail?"
```

For Gemini, install the optional experiment dependency and provide the API key through the
existing environment convention:

```bash
python -m pip install -e ".[experiment]"
export GEMINI_API_KEY="your-api-key"
python -m ai_investigation.cli \
  --investigator gemini \
  "Why did deployment deploy-1042 fail?"
```

Run both reasoning paths over one shared collection of evidence:

```bash
python -m ai_investigation.cli \
  --investigator both \
  "Why did deployment deploy-1042 fail?"
```

Omit the question to choose an investigator and enter a question interactively:

```bash
python -m ai_investigation.cli
```

The current tools read synthetic local fixtures. In `both` mode, the deterministic and Gemini
paths receive the exact same `CollectedEvidence`; the model does not call tools itself. A
structurally valid response and valid evidence references do not establish semantic correctness,
and model-reported confidence is currently uncalibrated. Provider, parsing, and reference-validation
failures are displayed without automatic retries.

Run all tests:

```bash
pytest
```

Run the controlled evaluation suite:

```bash
python -m ai_investigation.evaluation.runner \
  tests/fixtures/evaluation_scenarios.json \
  --fixtures tests/fixtures
```

Run the extended deterministic benchmark:

```bash
python -m ai_investigation.evaluate \
  --investigator deterministic \
  --scenarios tests/fixtures/evaluation_scenarios_m10.json
```

Run the same extended benchmark through Groq explicitly:

```bash
export GROQ_API_KEY="your-api-key"
export AI_INVESTIGATION_PROVIDER=groq
export AI_INVESTIGATION_MODEL=llama-3.3-70b-versatile
python -m ai_investigation.evaluate \
  --investigator llm \
  --provider groq \
  --model llama-3.3-70b-versatile \
  --scenarios tests/fixtures/evaluation_scenarios_m10.json \
  --save-experiment
```

The real-provider smoke test is skipped by default and makes at most one request when enabled:

```bash
RUN_LLM_INTEGRATION_TESTS=1 pytest -m llm_integration
```

It consumes provider quota. Ordinary tests use fake structured models and require neither the
Groq SDK nor an API key. Compare a stored deterministic run with a stored LLM run using the same
existing command and regression policy:

```bash
python -m ai_investigation.experiments compare \
  <deterministic-experiment-id> <llm-experiment-id> \
  --fail-on-regression
```

Run the Milestone 7 deterministic evaluation with a human-readable report:

```bash
python -m ai_investigation.evaluate --investigator deterministic --format text
```

Write a stable machine-readable report:

```bash
python -m ai_investigation.evaluate \
  --investigator deterministic \
  --format json \
  --output reports/deterministic-baseline.json
```

An explicitly selected `both` run shares one evidence collection per scenario:

```bash
GEMINI_API_KEY="your-api-key" python -m ai_investigation.evaluate \
  --investigator both \
  --format text
```

The safe default is deterministic and never constructs Gemini. Gemini and `both` evaluations are
opt-in real-provider experiments; automated tests inject fake models and make no provider calls.
The initial deterministic result is documented in the
[Milestone 7 baseline report](docs/experiments/milestone-7-deterministic-baseline.md).

Save a deterministic experiment locally:

```bash
python -m ai_investigation.evaluate \
  --investigator deterministic \
  --save-experiment \
  --tag baseline \
  --notes "Milestone 8 deterministic baseline"
```

Inspect experiment history:

```bash
python -m ai_investigation.experiments list
python -m ai_investigation.experiments show <experiment-id>
python -m ai_investigation.experiments compare <before-id> <after-id>
```

Machine-readable comparison and the semantic regression gate use the same typed comparison model:

```bash
python -m ai_investigation.experiments compare \
  <baseline-id> <candidate-id> \
  --json

python -m ai_investigation.experiments compare \
  <baseline-id> <candidate-id> \
  --fail-on-regression
```

Comparison keeps correctness, agreement, validation, provider failures, latency, regressions, and
improvements separate. Metrics without compatible denominators are reported as not comparable.
External observability platforms remain deferred until local tracking demonstrates a concrete
need. See the [Milestone 8 observability note](docs/experiments/milestone-8-local-observability.md).
The comparison policy and failure precedence are documented in the
[Milestone 9 engineering note](docs/experiments/milestone-9-scenario-regression-analysis.md).

## Possible Architectural Evolution

The roadmap is problem-driven, not framework-driven.

```text
Completed
  deterministic investigation foundation
        │
        v
  multi-rule diagnosis and conflict handling
        │
        v
  controlled synthetic evaluation
        │
        v
  structured deterministic decision trace
        │
        v
  controlled Gemini baseline comparison
        │
        v
  local experiment tracking and execution observability
        │
        v
  scenario-level comparison and semantic regression gate
        │
        v
  controlled probabilistic investigator and Groq experiment path

Possible directions
  ├── graph orchestration if branching complexity justifies it
  ├── external evidence integrations
  ├── MCP if interoperability creates concrete value
  ├── tracing and operational observability
  └── optional web interface
```

None of these technologies is assumed to be necessary. Each would need to solve a demonstrated limitation and preserve measurable behavior against the current baseline.

## Current Limitations

The project currently:

- uses controlled local synthetic fixtures;
- covers a deliberately small deployment-failure domain;
- supports three deterministic diagnosis patterns;
- has no production evidence integrations;
- has experimental Gemini and Groq adapters rather than a production probabilistic investigator;
- has no retry, fallback, or semantic-validation policy for model output;
- does not measure performance against real incident data.

These constraints are intentional. They keep the baseline inspectable and make its claims honest. The evaluation suite measures behavior on its declared synthetic scenarios, not general incident-diagnosis accuracy.

## Project Philosophy

This repository is not trying to build the largest AI agent or assemble the longest list of frameworks. It documents the engineering path from deterministic software toward increasingly capable AI systems while keeping evidence, uncertainty, evaluation, and explanation visible.

The objective is not to showcase frameworks. The objective is to understand when they become necessary—and how to introduce them without sacrificing reliability.
