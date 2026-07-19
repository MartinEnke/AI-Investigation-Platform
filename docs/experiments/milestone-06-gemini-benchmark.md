# Milestone 6 Gemini Benchmark

## Experiment Status

- Run date: 2026-07-19
- Model: `gemini-2.5-flash`
- Scenarios: 11 controlled synthetic cases
- Prompt: unchanged Milestone 6 prompt
- Schema and validation: unchanged Milestone 6 implementation
- Retries: none
- Deterministic baseline: 11/11 scenarios passed

This report preserves one concrete benchmark run. It evaluates this repository's evidence,
prompt, schema, validation policy, provider adapter, and `gemini-2.5-flash` response behavior in
that run. It is not a general evaluation of Gemini 2.5 Flash.

## Comparison Architecture

```text
Question
   │
   v
EvidenceCollector ──> CollectedEvidence
                            │
              ┌─────────────┴─────────────┐
              v                           v
   Deterministic rules             Gemini interpretation
   + DecisionTrace                 + structured response
              │                           │
              v                           v
   InvestigationResult             parsing and reference validation
              │                           │
              └─────────────┬─────────────┘
                            v
                    scenario comparison
```

Evidence was collected once. Both paths received the same collected records and ordered public
evidence. Gemini could not call evidence tools or retrieve additional information. Deterministic
rules did not repair, approve, or override Gemini conclusions.

## Validation Dimensions

The experiment keeps three concepts separate:

1. **Structural validity** — the response is valid JSON and satisfies the application response
   contract, including diagnosis ID, confidence, and field-combination checks.
2. **Reference validity** — cited one-based evidence positions exist, are unique, and cover the
   source categories required for the selected diagnosis.
3. **Semantic correctness** — the validated diagnosis or abstention matches the scenario
   expectation.

Valid references do not establish that evidence logically entails a diagnosis. Provider failure
is also an end-to-end outcome: no retry or fallback hides it.

## Scenario Results

| Scenario | Expected | Raw Gemini decision | Validation | Semantic outcome |
|---|---|---|---|---|
| `supported-health-check-timeout` | `health_check_timeout` | `{"outcome":"diagnosis","diagnosis_id":"health_check_timeout","confidence":0.9,"evidence_references":[1,2,3],"abstention_reason":null}` | Passed | Correct diagnosis |
| `unknown-deployment` | Abstain | No call | `not_evaluated`: deployment not found | Not evaluated |
| `missing-log-evidence` | Abstain | `{"outcome":"diagnosis","diagnosis_id":"health_check_timeout","confidence":0.9,"evidence_references":[1,2],"abstention_reason":null}` | Invalid references: missing `logs` | No valid result |
| `missing-service-health-evidence` | Abstain | `{"outcome":"diagnosis","diagnosis_id":"health_check_timeout","confidence":0.9,"evidence_references":[1,2],"abstention_reason":null}` | Invalid references: missing `service_health` | No valid result |
| `unsupported-complete-evidence` | Abstain | `{"outcome":"abstain","diagnosis_id":null,"confidence":0.9,"evidence_references":[1,2],"abstention_reason":"insufficient_evidence"}` | Passed | Correct abstention |
| `supported-missing-environment-variable` | `missing_environment_variable` | `{"outcome":"diagnosis","diagnosis_id":"missing_environment_variable","confidence":1.0,"evidence_references":[2],"abstention_reason":null}` | Invalid references: missing `deployment` | Correct ID, no valid result |
| `supported-database-migration-failure` | `database_migration_failure` | `{"outcome":"diagnosis","diagnosis_id":"database_migration_failure","confidence":0.95,"evidence_references":[1,2],"abstention_reason":null}` | Passed | Correct diagnosis |
| `supported-environment-variable-variant` | `missing_environment_variable` | No response | `provider_failure` | Not scored |
| `supported-diagnosis-after-irrelevant-error` | `missing_environment_variable` | No response | `provider_failure` | Not scored |
| `unsupported-generic-database-error` | Abstain | `{"outcome":"diagnosis","diagnosis_id":"database_migration_failure","confidence":0.9,"evidence_references":[1,2],"abstention_reason":null}` | Passed | False-positive diagnosis |
| `conflicting-supported-patterns` | Conflict abstention | No response | `provider_failure` | Not scored |

No failed scenario was rerun. No prompt, rule, schema, or validation behavior was changed in response
to these results.

## Aggregate Results

| Metric | Deterministic | Gemini |
|---|---:|---:|
| Total scenarios | 11 | 11 |
| Exact benchmark successes | 11 | 3 |
| Correct diagnoses | 5 | 2 |
| Correct abstentions | 6 | 1 |
| Accepted false-positive diagnoses | 0 | 1 |
| Wrong supported diagnoses | 0 | 0 |
| Invalid references or source coverage | 0 | 3 |
| Malformed responses | 0 | 0 |
| Refusals | 0 | 0 |
| Provider failures | 0 | 3 |
| Not evaluated | 1 expected early outcome | 1 |

Of 10 model-eligible scenarios:

- 3/10 produced semantically correct end-to-end outcomes;
- 4/10 produced valid `InvestigationResult` values;
- 7/10 returned structurally valid JSON;
- 4/7 structurally valid responses also had valid references;
- 3/4 valid results were semantically correct.

Five scenarios expected a supported diagnosis. Gemini produced two correct valid diagnoses, one
correct diagnosis ID with invalid references, and two provider failures: end-to-end diagnosis
accuracy was 2/5.

Five model-eligible scenarios expected abstention. Gemini produced one correct abstention, one
accepted false-positive diagnosis, two diagnosis attempts rejected for invalid references, and one
provider failure: end-to-end abstention accuracy was 1/5.

## Confidence Observations

The seven structured responses reported:

```text
0.90, 0.90, 0.90, 0.90, 1.00, 0.95, 0.90
```

Their mean was approximately 0.92. Correct results reported 0.90, 0.90, and 0.95. The accepted
false positive reported 0.90. Reference-invalid responses reported 0.90, 0.90, and 1.00.

Confidence did not separate correct, incorrect, and invalid outcomes. The highest confidence in
the run belonged to a response rejected for incomplete evidence references. This single run
provides no evidence that model-reported confidence is calibrated.

## Conclusions

The architecture behaved as intended:

- Gemini consistently used the structured response shape when it returned a response.
- Application parsing and reference validation kept malformed or incompletely supported outputs
  distinct from valid investigations.
- Reference validation rejected three conclusions that would otherwise have appeared usable.
- Reference validity alone could not detect the semantically unsupported database-migration false
  positive.
- Provider failures remained visible because the system used no retries or fallbacks.
- Semantically wrong behavior was measured rather than silently repaired by deterministic rules.

The main engineering conclusion is:

> In a narrow, known problem space, deterministic reasoning remained more reliable, while the LLM
> path demonstrated flexible structured interpretation but required independent validation and
> showed weak abstention, uncalibrated confidence, semantic false positives, and provider
> instability.

These findings describe one unchanged run of this concrete system configuration. They should be
used as a regression and architecture reference, not as a universal model-performance claim.
