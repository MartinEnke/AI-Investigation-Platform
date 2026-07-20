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
the two misses are the intended generalization cases. Probabilistic results are interpreted only
within this controlled synthetic benchmark.

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

## Milestone 10.1: Grounding and Abstention

The first controlled 10.1 change adds explicit `v1` and `v2` prompt selection without changing the
investigation architecture. `v1` remains the default and preserves the Milestone 10 prompt.
`v2` keeps the existing structured response schema and numbered-reference validation while making
each LLM-facing evidence item self-contained: exact ID, type, source, observation, and original
factual content.

The `v2` instructions tighten grounding, diagnosis boundaries, and abstention behavior. Providers,
benchmark scenarios, deterministic reasoning, evaluation, comparison, and regression policies are
unchanged. New experiments persist the stable identifier `llm-investigator-v2`; older experiment
records without prompt metadata remain readable. Few-shot examples are intentionally deferred to a
separate experiment.

`v3` preserves the v2 evidence representation and every existing architectural boundary. Its only
change is concise prompt guidance requiring deployment and cause-establishing log references,
conditional service-health coverage, and abstention when a required source is absent. It also
draws explicit causal boundaries around generic database errors and timeout symptoms, and prohibits
mapping an unsupported cause to the nearest supported label. Benchmark fixtures, validators,
schemas, providers, scoring, and deterministic behavior remain unchanged.

## Completed prompt experiment cycle

The three Groq prompt experiments used the same 16 scenarios, evaluator, deterministic validators,
provider, and `llama-3.3-70b-versatile` model. Only the versioned prompt changed.

| Prompt | Diagnosis accuracy | Abstention accuracy | Evidence-reference validity |
| ------ | ------------------ | ------------------- | --------------------------- |
| v1     | 3/8                | 2/8                 | 6/15                        |
| v2     | 5/8                | 4/8                 | 10/15                       |
| v3     | 7/8                | 6/8                 | 14/15                       |

The v1 run exposed frequent unsupported diagnoses, weak abstention, and poor grounding. V2's
self-contained evidence representation and explicit grounding rules improved every measured
dimension, but several diagnoses still omitted required source categories and symptom similarity
still encouraged nearest-label overdiagnosis. V3 made source coverage and causal boundaries
explicit. Structured-response validity remained 15/15, almost all grounding failures disappeared,
and the remaining failures narrowed to one invalid reference and two semantic edge cases.

These results do not establish general Groq or model performance. They describe three reproducible
runs of this concrete system and benchmark configuration.

### Deterministic validation as a reliability layer

The experiment improved model behavior but did not make deterministic validation redundant.
Reference existence, required source coverage, diagnosis identifiers, response structure, and field
combinations remain deterministic invariants. They prevent unsupported outputs from silently
becoming valid investigations and expose failures that prompt changes alone do not eliminate.

The validator is therefore not a patch for an inadequate prompt. It deliberately separates
probabilistic interpretation from deterministic verification. Treating each prompt as a versioned,
benchmarked artifact makes successful and unsuccessful iterations inspectable rather than relying
on intuition or selectively chosen examples.
