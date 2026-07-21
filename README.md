# AI Investigation Platform

![Python 3.13+](https://img.shields.io/badge/Python-3.13%2B-3776AB)
![Tests](https://img.shields.io/badge/tests-285%20passing-2E8B57)
![Holdout scenarios](https://img.shields.io/badge/holdout-20%20scenarios-2E8B57)
![Milestone 12.3](https://img.shields.io/badge/milestone-12.3-6C63FF)
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
- 285 passing tests (plus one opt-in provider test skipped by default);
- 11 established regression scenarios and 5 Milestone 10 generalization scenarios.
- a separate frozen 20-scenario Milestone 11 holdout benchmark.
- an explicit immutable uncertainty model and deterministic decision policy;
- a separate `llm-policy` experiment path in which the model proposes candidates but policy owns
  the final outcome.

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

Milestone 10.1 begins with explicit prompt versions. `llm-investigator-v1` preserves the
Milestone 10 behavior and remains the default. `llm-investigator-v2` presents each evidence item as
a self-contained record with its exact existing numeric ID, type, source, observation, and factual
content, then applies stricter grounding and abstention instructions. Structured output is validated
again in application code even when a provider offers JSON constraints. Unknown
diagnoses, contradictory fields, invalid confidence, malformed JSON, and nonexistent references
remain explicit failures. No retries are used because raw provider reliability is itself an
experimental result.

This first 10.1 experiment changes no architecture, provider adapter, deterministic rule, benchmark
case, or evaluation policy. Few-shot examples are intentionally deferred so their effect can be
measured separately.

`llm-investigator-v3` is a narrow controlled iteration over the same evidence representation. It
targets complete required-source references and nearest-label overdiagnosis through prompt text
only.

Groq is a single small transport adapter behind the existing `StructuredModel` protocol, not a
provider platform. The extended 16-scenario set adds semantic rephrasing, multi-step causal
synthesis, unsupported evidence, conflicting evidence, and distractors without rewriting the
original 11 cases. The deterministic path scores 14/16 on semantic correctness in that extended
set.

## Controlled Prompt Engineering Experiments

The repository treats prompts as versioned engineering artifacts rather than mutable strings tuned
until one example looks convincing. Each revision is persisted in experiment metadata and run
against the same benchmark before its effect is assessed.

### Prompt v1

The initial structured investigator established the probabilistic baseline:

- diagnosis accuracy: 3/8;
- abstention accuracy: 2/8;
- evidence-reference validity: 6/15.

It frequently selected unsupported diagnoses, abstained poorly, and returned weakly grounded or
invalid evidence references.

### Prompt v2

The second experiment introduced explicit prompt versioning, self-contained evidence serialization,
and stronger grounding and abstention instructions:

- diagnosis accuracy: 5/8;
- abstention accuracy: 4/8;
- evidence-reference validity: 10/15.

The measurable improvement showed that clearer evidence presentation mattered. Remaining failures
mostly omitted required evidence sources, while nearest-label reasoning still produced unsupported
diagnoses.

### Prompt v3

The third experiment added explicit source-coverage requirements, causal diagnosis boundaries,
stronger abstention rules, required deployment context, and conditional service-health coverage.
It states directly that similarity is not causal evidence:

- diagnosis accuracy: 7/8;
- abstention accuracy: 6/8;
- evidence-reference validity: 14/15.

Structured-response validity remained 15/15. Almost all grounding failures were removed; one
invalid-reference failure and two semantic edge cases remained. The result is materially better,
not perfect.

| Prompt | Diagnosis | Abstention | Evidence References |
| ------ | --------- | ---------- | ------------------- |
| v1     | 3/8       | 2/8        | 6/15                |
| v2     | 5/8       | 4/8        | 10/15               |
| v3     | 7/8       | 6/8        | 14/15               |

Every run used identical scenarios, evaluator, validators, provider, and model. Only the prompt
changed. That isolation makes the progression a controlled engineering experiment rather than
anecdotal prompt optimization.

### Why deterministic validation still exists

Prompt improvements do not remove the need for deterministic verification. The validation layer:

- rejects unsupported diagnosis identifiers and malformed decision combinations;
- verifies that cited evidence IDs exist;
- enforces required evidence-source coverage;
- catches edge cases that an improved model still misses;
- keeps probabilistic interpretation separate from deterministic verification.

Semantic evaluation separately exposes supported-but-incorrect diagnoses that structural and
reference validation cannot prove or disprove.

This is not compensation for a bad prompt. It is an intentional reliability boundary: model
behavior can improve while the system continues to enforce stable, machine-readable invariants.

Reliable AI systems improve through controlled experimentation and measurement rather than
intuition. Prompt revisions in this repository are versioned, benchmarked, reproducible, and
compared with earlier experiments. Successful and unsuccessful iterations are both documented
because both reveal how the system behaves.

## Generalization and Holdout Evaluation

Milestone 10.1 improved prompt v3 using failures observed on the original development benchmark.
That makes another run on the same scenarios insufficient evidence of generalization. Milestone 11
therefore adds a separate 20-scenario holdout created after v3 was frozen.

The holdout contains 10 supported diagnoses and 10 required abstentions. Its cases vary wording,
formatting, evidence order, distractors, missing evidence, conflicting signals, duplicate records,
and unsupported causes. Stable robustness categories make those dimensions inspectable without
changing diagnosis or scoring semantics.

The experimental rules are strict:

- prompt v3 is frozen;
- deterministic reasoning is frozen;
- response and evidence-reference validators are frozen;
- the original Milestone 10 scenarios remain unchanged;
- neither investigator may change before the first holdout comparison.

The planned comparison runs the deterministic investigator and the Groq-backed LLM investigator
with prompt v3 over exactly the same holdout. No Groq holdout result is published yet. The outcome
will inform whether the next step should retain the current deterministic boundary, expand the
model's role, or test a narrowly defined hybrid decision policy. It does not yet justify graph or
multi-agent orchestration.

Evaluation reports now supplement existing metrics with deterministic error categories such as
false diagnosis, unnecessary abstention, wrong diagnosis, missing required source, invalid
reference, invalid response, provider failure, and not evaluated. Stored experiments already record
the scenario source, and experiment inspection now surfaces it explicitly so development and
holdout runs cannot be confused.

## Uncertainty and Decision Policy

Milestone 11 exposed complementary behavior: deterministic reasoning abstained safely but missed
four supported holdout diagnoses, while LLM v3 found every supported diagnosis but produced two
false diagnoses instead of abstaining. Milestone 12 treats this as a decision-policy problem rather
than immediately changing the prompt or combining investigators.

The immutable domain layer represents supported candidates, supporting and contradicting
evidence, source completeness, evidence strength, unsupported signals, conflicts, and insufficient
evidence. A pure deterministic policy then returns one of three outcomes:

- `diagnosis` for exactly one sufficiently supported, complete candidate;
- `abstention` when evidence cannot justify a supported cause;
- `needs_review` when plausible supported candidates or material evidence remain unresolved.

Evidence strength is deliberately distinct from reported confidence, and confidence does not drive
the policy. Milestone 12.3 connects this layer through a new, separately selectable `llm-policy`
path while leaving the deterministic and frozen LLM v3 paths unchanged:

```text
Collected Evidence
       │
       v
LLM uncertainty proposal
       │
       v
Typed uncertainty adapter
       │
       v
Deterministic decision policy
       │
       v
diagnosis / abstention / needs_review
       │
       v
Public InvestigationResult
```

The versioned `v4-uncertainty` prompt asks for zero, one, or multiple supported candidates,
per-candidate supporting and contradicting references, evidence strength, reported confidence, and
cross-candidate uncertainty flags. It explicitly does not ask the model to choose the final
outcome. Evidence is exposed through exact request-local IDs such as `E1`, `E2`, and `E3`.
Application code validates those identities and derives source coverage before applying the
existing fixed policy.

The model does not declare exact missing source types. Required sources come from diagnosis rules,
available sources come from collected evidence, and deterministic code computes
`missing_required_sources = required_sources - available_sources`. This keeps semantic uncertainty
with the model while evidence identity and availability remain application-owned facts.

The first valid live policy run then exposed a different semantic boundary: the model placed nearly
every diagnosis it considered into the candidate list. Because policy correctly treats every
candidate as independently supported, it returned review for 13 scenarios. The separately
versioned `v4-uncertainty-candidate-semantics` contract now distinguishes policy-eligible
`supported_candidates` from explainability-only `rejected_hypotheses`. Rejected hypotheses retain
typed reasons and valid evidence IDs but never enter candidate counts, conflict state, or policy.
This is a controlled proposal-contract correction; no improved benchmark result is claimed before
a new live run.

Milestone 12.4 hardens that contract by removing model-authored conflict state. The application now
derives conflict exactly from the supported-candidate count, while legacy schema-v3 responses remain
readable. Source-difference reporting also uses unique source categories, preventing one category
from appearing as both referenced and missing. Neither change alters policy or evaluation scoring.

`needs_review` is reserved for plausible competing supported causes or material unresolved
contradiction. The public result remains backward compatible by representing review as an
abstention while retaining the typed policy outcome, reason, and candidate labels in the new path's
structured evaluation result. This is still a single-pass experiment: it adds no agent loop,
retry, routing, or human-review UI.

Prompt v3 remains frozen as the Milestone 11 comparison baseline. The untested hypothesis is that
the policy-controlled path may retain semantic generalization while reducing unsafe diagnoses in
ambiguous cases; no score is claimed before a live evaluation. The design and full decision table
are documented in the
[Milestone 12 architecture note](docs/architecture/milestone-12-uncertainty-decision-policy.md).

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
│   │   ├── milestone-10-controlled-probabilistic-investigator.md
│   │   ├── milestone-11-holdout-evaluation.md
│   │   └── milestone-12-uncertainty-decision-policy.md
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
│       ├── decision_policy.py
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
    ├── test_decision_policy.py
    ├── test_evaluation.py
    ├── test_evidence.py
    ├── test_experiment_evaluation.py
    ├── test_experiment_tracking.py
    ├── test_gemini_model.py
    ├── test_groq_integration.py
    ├── test_groq_model.py
    ├── test_holdout_m11.py
    ├── test_investigator.py
    ├── test_llm_investigator.py
    ├── test_milestone10.py
    ├── test_tools.py
    └── fixtures/
        ├── deployments.json
        ├── evaluation_scenarios.json
        ├── evaluation_scenarios_m10.json
        ├── evaluation_scenarios_holdout_m11.json
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

Run the frozen deterministic holdout:

```bash
python -m ai_investigation.evaluate \
  --investigator deterministic \
  --scenarios tests/fixtures/evaluation_scenarios_holdout_m11.json \
  --save-experiment \
  --tag milestone-11-holdout
```

Run the same holdout with frozen prompt v3 only when a real-provider experiment is intended:

```bash
python -m ai_investigation.evaluate \
  --investigator llm \
  --prompt-version v3 \
  --scenarios tests/fixtures/evaluation_scenarios_holdout_m11.json \
  --request-delay-seconds 4 \
  --save-experiment \
  --tag milestone-11-holdout
```

Request pacing is an evaluation reliability control for external provider rate limits. It sleeps
only between provider-backed LLM scenario requests, is recorded with the experiment, and does not
change prompts, evidence, validation, or investigator reasoning. Its default is zero; deterministic
evaluation never sleeps.

Run the same extended benchmark through Groq explicitly:

```bash
# .env
GROQ_API_KEY=your-api-key
AI_INVESTIGATION_PROVIDER=groq
AI_INVESTIGATION_MODEL=llama-3.3-70b-versatile

python -m ai_investigation.evaluate \
  --investigator llm \
  --provider groq \
  --model llama-3.3-70b-versatile \
  --prompt-version v3 \
  --scenarios tests/fixtures/evaluation_scenarios_m10.json \
  --save-experiment
```

The evaluation CLI loads `.env` from the current directory. Explicit `--provider` and `--model`
arguments override those environment defaults. Credentials are never stored in experiment
metadata.

Run the policy-controlled uncertainty path over the original 16-scenario benchmark:

```bash
python -m ai_investigation.evaluate \
  --investigator llm-policy \
  --provider groq \
  --model llama-3.3-70b-versatile \
  --prompt-version v4-uncertainty \
  --scenarios tests/fixtures/evaluation_scenarios_m10.json \
  --request-delay-seconds 4 \
  --save-experiment \
  --tag milestone-12-policy-original
```

Rerun the candidate-semantics correction as a separate experiment:

```bash
python -m ai_investigation.evaluate \
  --investigator llm-policy \
  --provider groq \
  --model llama-3.3-70b-versatile \
  --prompt-version v4-uncertainty-candidate-semantics \
  --scenarios tests/fixtures/evaluation_scenarios_m10.json \
  --request-delay-seconds 4 \
  --save-experiment \
  --tag milestone-12-candidate-semantics
```

Run it over the frozen 20-scenario holdout without changing v3 or the holdout data:

```bash
python -m ai_investigation.evaluate \
  --investigator llm-policy \
  --provider groq \
  --model llama-3.3-70b-versatile \
  --prompt-version v4-uncertainty \
  --scenarios tests/fixtures/evaluation_scenarios_holdout_m11.json \
  --request-delay-seconds 4 \
  --save-experiment \
  --tag milestone-12-policy-holdout
```

Experiments from this mode record the investigator, provider, model, uncertainty prompt version,
response-schema version, and deterministic policy version. Reports add policy outcome, reason, and
candidate diagnoses per scenario plus diagnosis, abstention, review, single-candidate, and
multi-candidate counts. Existing deterministic and LLM report fields retain their meanings.

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
        │
        v
  frozen holdout generalization evaluation
        │
        v
  explicit uncertainty model and deterministic decision policy
        │
        v
  policy-controlled LLM uncertainty experiment path

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
