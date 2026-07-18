# AI Investigation Platform

![Python 3.13+](https://img.shields.io/badge/Python-3.13%2B-3776AB)
![Tests](https://img.shields.io/badge/tests-53%20passing-2E8B57)
![Evaluation scenarios](https://img.shields.io/badge/evaluations-11%20passing-2E8B57)
![Milestone 5](https://img.shields.io/badge/milestone-5-6C63FF)
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

Milestone 5 provides a complete deterministic deployment-failure investigation pipeline.

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
- a command-line interface;
- 53 passing tests;
- 11 passing synthetic evaluation scenarios.

The supported diagnosis rules are intentionally narrow:

1. Health-check timeout
2. Missing required environment variable
3. Database migration failure

The system makes no network calls and no LLM calls. There are no agents, graph frameworks, external services, or probabilistic scores hidden behind the deterministic behavior.

## Architecture

The design keeps evidence access, diagnosis, trace construction, and public presentation separate enough to test each responsibility directly.

```text
CLI
 │
 v
Investigator ──> Evidence Collection
                       ├──> Deployment Tool
                       ├──> Log Tool
                       └──> Service Health Tool
 │
 v
Deterministic Diagnosis ──> Decision Trace ──> Investigation Result
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

## Evaluation Framework

The repository includes a dedicated deterministic evaluation runner in addition to unit tests. Its purpose is to protect system behavior across architectural changes.

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

The evaluation schema always compares root cause, conclusive status, and ordered evidence sources. Scenarios may also opt into exact checks for:

- confidence;
- ordered limitations;
- decision outcome;
- ordered matched-rule identifiers.

Optional expectations keep older scenarios valid while allowing important cases—especially conflicts—to make stronger assertions. Together, the scenarios form a regression baseline for measuring any future probabilistic component against the existing deterministic system.

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

Actual CLI output:

```text
Deployment deploy-1042 failed during its health check. The deployment health check timed out because the target service was unhealthy.
Root cause: The deployment health check timed out because the target service was unhealthy.
Confidence: 100%
Evidence:
  1. [deployment] deploy-1042 has status failed and failed stage health_check.
  2. [logs] Health check timed out after 30 seconds.
  3. [service_health] checkout-api was unhealthy: Readiness endpoint returned HTTP 503.
```

The CLI intentionally renders the concise investigation result, not the full decision trace. The same investigation exposes this trace programmatically:

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
├── src/
│   └── ai_investigation/
│       ├── __init__.py
│       ├── cli.py
│       ├── diagnosis.py
│       ├── investigator.py
│       ├── models.py
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── loader.py
│       │   ├── models.py
│       │   └── runner.py
│       └── tools/
│           ├── __init__.py
│           ├── deployments.py
│           ├── logs.py
│           ├── protocols.py
│           └── service_health.py
└── tests/
    ├── conftest.py
    ├── test_diagnosis.py
    ├── test_evaluation.py
    ├── test_investigator.py
    ├── test_tools.py
    └── fixtures/
        ├── deployments.json
        ├── evaluation_scenarios.json
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

Possible directions
  ├── controlled LLM baseline comparison
  ├── probabilistic investigation beside the deterministic baseline
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
- has no probabilistic AI component;
- does not measure performance against real incident data.

These constraints are intentional. They keep the baseline inspectable and make its claims honest. The evaluation suite measures behavior on its declared synthetic scenarios, not general incident-diagnosis accuracy.

## Project Philosophy

This repository is not trying to build the largest AI agent or assemble the longest list of frameworks. It documents the engineering path from deterministic software toward increasingly capable AI systems while keeping evidence, uncertainty, evaluation, and explanation visible.

The objective is not to showcase frameworks. The objective is to understand when they become necessary—and how to introduce them without sacrificing reliability.
