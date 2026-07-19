# Milestone 8 Local Experiment Observability

## Purpose

Milestone 8 makes evaluation executions inspectable and comparable without adding an external
observability service. Investigation behavior is unchanged. Instrumentation surrounds the existing
evidence collection, investigation, validation-outcome, and evaluation boundaries.

## Local Lifecycle

An explicitly saved evaluation creates one filesystem-safe experiment ID and one directory:

```text
experiments/runs/<experiment-id>/
├── experiment.json   # metadata, EvaluationReport, timing summary, and events
├── report.txt        # existing human-readable evaluation report
└── events.jsonl      # one ordered structured event per line
```

Artifacts are written as UTF-8 through temporary files and atomic replacement. Existing experiment
directories are never overwritten. Machine-generated runs are ignored by Git; this document is the
curated repository record.

## Event Lifecycle

A deterministic scenario emits events for:

1. experiment start;
2. scenario start;
3. evidence collection start and completion;
4. deterministic investigation start and completion;
5. scenario evaluation;
6. scenario completion;
7. experiment completion.

Model runs additionally record model investigation and validation completion. Provider and
validation failures remain explicit scenario outcomes and can still be persisted.

## Timing Definitions

- `evidence_collection_ms`: calls to the fixture-backed evidence collector.
- `deterministic_investigation_ms`: deterministic interpretation of collected evidence.
- `model_investigation_ms`: the complete LLM investigator call, including provider execution and
  its internal parsing and reference validation.
- `validation_ms`: separately measurable validation time; currently unavailable because validation
  is intentionally encapsulated by the LLM investigator.
- `evaluation_ms`: conversion of public outcomes into scenario evaluation dimensions.
- `scenario_total_ms`: the complete scenario execution, including shared evidence collection.
- `experiment_total_ms`: the complete benchmark execution.

Durations come from a monotonic clock. Wall-clock UTC timestamps identify events but are never used
to calculate durations.

## Deterministic Baseline

The 11 controlled scenarios continue to produce:

| Metric | Result |
|---|---:|
| Diagnosis accuracy | 5/5 |
| Abstention accuracy | 6/6 |
| Evidence-reference integrity | 11/11 |
| Provider failures | 0 |
| Semantic failures | 0 |

Local latency values vary by machine and run, so no generated timing is presented as a universal
performance claim.

## Reproducibility and Privacy

Metadata includes the scenario source and IDs, investigator mode, provider/model when applicable,
Git revision when available, Python and platform identifiers, safe CLI configuration, tags, and
notes. Git absence does not fail a run. Configuration keys that indicate secrets, tokens, or API
keys are excluded, and provider credentials are never serialized.

## Limitations

- The store is a local directory, not a transactional database.
- Atomic writes protect completed individual artifacts but do not implement crash recovery for a
  partially created experiment directory.
- Comparison is descriptive and performs no statistical significance testing.
- Validation lifecycle is observable, but its duration is not separable from model investigation
  without changing the current investigator contract.
- No new real-provider comparison was run for this milestone. Fake-model tests verify provider
  metadata, failures, persistence, and comparison without network access.

External tracing platforms, dashboards, retries, and distributed telemetry remain intentionally
deferred. The local system is sufficient to reveal which additional infrastructure problems are
real rather than hypothetical.
