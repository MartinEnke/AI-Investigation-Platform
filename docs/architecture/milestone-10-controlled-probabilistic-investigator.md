# Milestone 10: Controlled Probabilistic Investigator

## Purpose

Milestone 10 tests whether probabilistic interpretation can generalize beyond deliberately narrow
deterministic rules without weakening evidence collection, validation, evaluation, or regression
control. The LLM path is an experimental alternative, not a replacement.

## Architecture

Evidence is collected once by `EvidenceCollector`. Investigators receive the same immutable
`CollectedEvidence` and cannot call tools.

```text
InvestigationRequest
        |
        v
EvidenceCollector
        |
        v
CollectedEvidence
        |
        +--> DeterministicInvestigatorAdapter --> InvestigationResult + DecisionTrace
        |
        +--> LLMInvestigatorAdapter
                |
                +--> versioned prompt --> StructuredModel --> strict validation
                                              |
                                              v
                                  InvestigationResult or explicit failure
```

`InvestigatorIdentity` records mode, provider, model, and prompt version without changing diagnosis
semantics. `InvestigatorExecution` normalizes successful results and execution/validation status at
the application boundary. The established `DeploymentFailureInvestigator` and `LLMInvestigator`
remain independently usable.

The existing `StructuredModel` protocol is the provider boundary. It is intentionally limited to
stable provider/model metadata and one synchronous structured generation call. No registry,
routing, retry, fallback, or generic provider framework was added.

## Prompt and response contract

`llm-investigator-v1` serializes only:

- the question and parsed deployment identifier;
- ordered collected evidence and one-based references;
- collection limitations;
- supported diagnosis and abstention identifiers;
- the strict response JSON schema.

It excludes expected answers, scenario labels, deterministic output, and comparison data. The
response schema is `llm-decision-v1`. Application validation checks exact fields, diagnosis IDs,
field combinations, finite confidence in `[0.0, 1.0]`, reference existence, uniqueness, and required
source coverage. Provider-side JSON enforcement never replaces these checks.

Valid decisions convert to the existing `InvestigationResult`. Provider failure, refusal, malformed
response, and invalid references remain distinct outcomes and do not fabricate an investigation
result.

## Groq adapter

`GroqStructuredModel` performs one synchronous request to Groq's chat-completions API using JSON
object mode. The initial model is `llama-3.3-70b-versatile`. Configuration is read only when the
explicit `llm` mode constructs the adapter:

- `GROQ_API_KEY`
- `AI_INVESTIGATION_PROVIDER=groq`
- `AI_INVESTIGATION_MODEL=llama-3.3-70b-versatile`

Authentication, rate-limit/quota, timeout, malformed provider envelope, and generic transport
errors are normalized without retries. Deterministic imports, tests, and commands do not require
credentials or a provider dependency.

## Evaluation design

The original 11-scenario regression set is unchanged. `evaluation_scenarios_m10.json` composes it
with five controlled cases:

1. semantically equivalent missing database configuration;
2. migration blocked through multi-step database contention;
3. failed deployment with no supported cause;
4. contradictory database-configuration evidence;
5. relevant missing-variable evidence among distractors.

This creates a 16-scenario comparison set without copying the original fixtures. The deterministic
baseline remains 11/11 on its established set and is semantically correct on 14/16 extended cases;
the two misses are the intended generalization cases. A Groq result is not documented until an
explicit real-provider experiment is run.

Experiment metadata persists provider, model, prompt version, and response-schema version for LLM
runs. Old schema-version-1 records that lack the new optional metadata remain readable. Existing
scenario comparison and the semantic regression gate require no investigator-specific branches.

## Verification and limitations

Normal tests use deterministic fake providers. The opt-in `llm_integration` test makes at most one
Groq request and checks structure and reference integrity rather than exact wording.

This milestone does not add agent behavior, tool calling, retries, fallback models, prompt tuning,
confidence calibration, or an LLM judge. The two new diagnosis categories define evaluation labels
and canonical result conversion for the experimental path; they do not expand deterministic rule
behavior.
