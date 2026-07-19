# Milestone 6: Deterministic vs. LLM-Assisted Investigation

## Status

Approved for phased implementation.

## Purpose

Milestone 6 introduces the smallest controlled comparison between deterministic diagnosis and
LLM-assisted interpretation. The deployment investigation domain remains a compact environment for
studying reliable AI system design; the LLM path does not replace the deterministic investigator.

The engineering question is:

> What changes when identical evidence is interpreted by deterministic rules versus an LLM?

The deterministic implementation remains the regression baseline. Evidence acquisition is shared
so that the comparison isolates the reasoning mechanism as far as the current architecture permits.

## Existing Baseline

Before Milestone 6, the project provides:

- protocol-backed deployment, log, and service-health tools;
- local deterministic JSON evidence sources;
- three deterministic diagnosis rules;
- explicit abstention for no matches and conflicting matches;
- ordered public evidence;
- immutable structured decision traces;
- exact deterministic tests and synthetic evaluation scenarios;
- no model calls, network calls, or orchestration framework.

The public `DeploymentFailureInvestigator` currently collects evidence and evaluates deterministic
rules in one workflow. Milestone 6 makes the collection boundary explicit without changing its
observable behavior.

## Milestone Boundary

Milestone 6 includes:

- deterministic evidence collection extracted into `EvidenceCollector`;
- an immutable `CollectedEvidence` input shared by both interpretation paths;
- deterministic interpretation of pre-collected evidence;
- a provider-independent LLM core with strict structured parsing and reference validation;
- deterministic fake-model tests;
- one isolated Google Gemini adapter;
- an opt-in comparison experiment over the existing scenarios.

Milestone 6 does not include:

- new diagnosis rules;
- changes to deterministic outcomes, confidence, limitations, evidence ordering, or traces;
- model tool calls or independent evidence retrieval;
- deterministic-rule fallback for model conclusions;
- free-form model answers as the primary contract;
- provider routing, retries, fallback models, or a provider registry;
- LangGraph, MCP, LiteLLM, or generic agent orchestration;
- persistence, background jobs, tracing platforms, or UI work.

## Approved Flow

```text
Question
   │
   v
Parse InvestigationRequest
   │
   v
Deterministic EvidenceCollector
   │
   v
CollectedEvidence
   │
   ├────────────────────────────────────┐
   │                                    │
   v                                    v
DeploymentFailureInvestigator       LLMInvestigator
deterministic rules                 structured model response
DecisionTrace                       parsing and validation
   │                                    │
   v                                    v
InvestigationResult                success or execution failure
   │                                    │
   └──────────────────┬─────────────────┘
                      v
              Comparison report
```

Evidence is collected once per comparison scenario. The identical `CollectedEvidence` instance is
passed to both paths. The LLM path receives no evidence tools.

## Shared Evidence Boundary

`CollectedEvidence` is an immutable, slotted model containing:

- the `InvestigationRequest`;
- the deployment record, when available;
- all relevant error-log records in source order;
- the service-health record, when available;
- ordered public `Evidence` values;
- ordered evidence-availability limitations.

The structured records are retained because deterministic rules use fields that are not present in
public summaries, such as the log `reason`. The future LLM prompt will serialize those same records
so neither reasoning path receives a richer collected input.

`EvidenceCollector` owns only evidence acquisition and normalization. It does not perform diagnosis.
It preserves the existing rules for missing IDs, unknown deployments, error-level filtering,
service lookup, summary wording, and ordering.

The existing investigator constructor and `investigate()` method remain unchanged. Internally,
`investigate()` collects evidence and delegates to deterministic interpretation of that collected
input. A pre-collected entry point supports the comparison without a second tool pass.

No shared `Reasoner` hierarchy is introduced. The deterministic and LLM paths have materially
different execution contracts.

## LLM Structured Contract

The provider-independent response contains exactly:

```json
{
  "outcome": "diagnosis",
  "diagnosis_id": "missing_environment_variable",
  "confidence": 0.22,
  "evidence_references": [1, 2],
  "abstention_reason": null
}
```

Allowed diagnosis identifiers are:

- `health_check_timeout`
- `missing_environment_variable`
- `database_migration_failure`

The outcome is either `diagnosis` or `abstain`. An abstention has a null diagnosis identifier and
one of these reasons:

- `insufficient_evidence`
- `conflicting_evidence`
- `low_confidence`

Confidence is a non-boolean number between `0.0` and `1.0`. It is experimental output, not an
application policy. A valid diagnosis with low confidence remains a diagnosis.

Evidence references are unique, one-based positions in the ordered public evidence tuple. They are
local to one investigation and are neither persistent nor globally unique.

The model does not supply public answer text, root-cause prose, arbitrary limitations, tool calls,
or a deterministic `DecisionTrace`. Known diagnosis identifiers map to application-owned result
wording.

## LLM Execution Outcomes

A genuine diagnosis or model-selected abstention is a successful interpretation and contains a
valid `InvestigationResult`.

Execution failures do not fabricate an `InvestigationResult`. The provider-independent LLM core
uses a discriminated success/failure outcome:

- success: valid decision plus `InvestigationResult`;
- `not_evaluated`: the request could not reach model interpretation;
- `invalid_response`: malformed or schema-invalid output;
- `invalid_references`: reference integrity or source coverage failed;
- `refused`: the provider explicitly blocked or filtered the request or candidate;
- `provider_failure`: authentication, quota, transport, SDK, or unavailable-response failure.

Wrong but structurally valid diagnoses remain successful executions and are marked incorrect by
experiment evaluation. Deterministic rules must not repair or override them.

## Evidence-Reference Validation

The prompt includes the request, complete collected records, and numbered public evidence.

Application validation checks:

- integer, non-boolean references;
- one-based bounds;
- uniqueness;
- at least one reference for a diagnosis;
- internal consistency between outcome fields;
- required referenced source categories.

Required source coverage is:

| Diagnosis | Required sources |
|---|---|
| Health-check timeout | deployment, logs, service health |
| Missing environment variable | deployment, logs |
| Database migration failure | deployment, logs |

`reference_validity` means cited positions exist and required source categories are present. It does
not mean the cited evidence logically proves the diagnosis. Diagnosis correctness is evaluated
separately against scenario expectations.

## Gemini Provider

The selected provider is Google Gemini. The isolated adapter is:

```text
src/ai_investigation/gemini_model.py
GeminiStructuredModel
```

The first configured model is `gemini-2.5-flash`. The adapter uses the official Google Gen AI
Python SDK package, `google-genai`, installed only through an optional experiment dependency.

The adapter:

- receives an explicit API key and model name;
- creates the Gemini client only for real execution;
- makes one synchronous structured-generation call;
- requests `application/json` with the approved JSON response schema;
- returns raw response text to the provider-independent parser;
- detects documented prompt blocking or filtered candidates;
- normalizes provider failures and refusals;
- performs no retries or fallback calls.

Provider-side schema enforcement does not replace application parsing, diagnosis validation,
confidence validation, field-combination validation, reference integrity, or source-coverage checks.

Configuration uses `GEMINI_API_KEY`. Credentials are not read during package import, pytest,
deterministic evaluation, or normal deterministic CLI use. The model name is explicit in the
experiment configuration and report.

## Evaluation and Reporting

The existing deterministic evaluation runner and exact expectations remain unchanged.

The LLM comparison runner reuses the existing scenario data but remains a separate opt-in command.
For each scenario it reports:

- scenario identifier;
- expected diagnosis or abstention;
- deterministic result;
- LLM result;
- structured-response validity;
- reference validity;
- model-reported confidence;
- execution status;
- concise validation errors.

Results are classified as:

- correct diagnosis;
- wrong diagnosis;
- correct abstention;
- incorrect abstention;
- malformed response;
- invalid references;
- refusal;
- provider failure;
- not evaluated.

The summary counts these categories. It does not calculate a weighted score, claim statistical
significance, or persist benchmark results. Optional provider usage metadata may be displayed only
if it does not affect core interfaces.

## Implementation Phases

### Phase 1: Shared Evidence Boundary

- implement `CollectedEvidence`;
- implement `EvidenceCollector`;
- allow deterministic interpretation of pre-collected evidence;
- prove existing deterministic behavior remains unchanged;
- run all tests and all 11 deterministic scenarios;
- stop for review.

### Phase 2: Provider-Independent LLM Core

- implement `StructuredModel`;
- implement the structured decision and success/failure outcomes;
- build the prompt;
- parse strictly;
- validate evidence references and source coverage;
- convert valid decisions into canonical results;
- add fake-model tests;
- make no real Gemini call;
- stop for review.

### Phase 3: Gemini Adapter

- implement `GeminiStructuredModel`;
- add the optional bounded `google-genai` dependency;
- configure `GEMINI_API_KEY` and explicit model selection;
- make one synchronous structured-generation call;
- normalize blocking, refusal, and provider failure;
- stop for review.

### Phase 4: Controlled Real-Model Smoke Test

Run `gemini-2.5-flash` against exactly:

1. one clear supported diagnosis;
2. one unsupported case expecting abstention;
3. one conflicting case expecting abstention.

Report raw structured decisions, validated outcomes, and failures. Do not tune the prompt merely to
force these cases to pass.

### Phase 5: Full Comparison Experiment

- run all 11 scenarios;
- produce the approved readable report;
- preserve all deterministic expectations.

## Design Decisions and Rationale

- **Deterministic baseline remains authoritative for regression.** Probabilistic observations do not
  weaken exact tests.
- **Evidence is shared; reasoning is independent.** This isolates the comparison without giving the
  model tool access.
- **No shared reasoner abstraction.** The two paths do not yet have equivalent contracts.
- **Structured response over generated prose.** Known identifiers and positional references make
  validity testable.
- **Execution failure is not abstention.** Invalid calls and responses do not receive fabricated
  investigation results.
- **Confidence is observed, not thresholded.** The application records the model's value unchanged.
- **Reference validity is not semantic proof.** Correctness remains a separate evaluation dimension.
- **One concrete provider.** A provider registry is unnecessary.
- **No retries.** Provider failure remains visible in the experiment.
- **No orchestration framework.** The approved flow is direct and sequential.

## Acceptance Criteria

Milestone 6 is complete when:

- evidence is collected once into immutable `CollectedEvidence`;
- the same collected input reaches both reasoning paths;
- deterministic behavior and public APIs remain compatible;
- the model has no evidence-tool access;
- fake-model tests are fast, deterministic, and credential-free;
- malformed, refused, unavailable, and reference-invalid executions remain distinguishable;
- confidence is preserved without automatic policy;
- the optional Gemini adapter is isolated from standard installation and execution;
- the three-scenario smoke test completes before the full experiment;
- all 11 scenarios receive a readable comparison report;
- no deterministic expectation is changed based on Gemini output.
