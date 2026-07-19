# Milestone 7 Deterministic Baseline

## Experiment Status

- Run date: 2026-07-19
- Investigator: deterministic rules
- Benchmark size: 11 controlled synthetic scenarios
- Provider calls: none
- Result: all semantic expectations satisfied

## Architecture

Each scenario was parsed and collected once through `EvidenceCollector`. The resulting
`CollectedEvidence` was passed to the deterministic investigator, then evaluated from its public
result and decision trace. Evaluation did not call tools or reproduce diagnosis logic.

## Metrics

| Metric | Result |
|---|---:|
| Completed runs | 11/11 |
| Diagnosis accuracy | 5/5 |
| Abstention accuracy | 6/6 |
| Evidence-reference integrity | 11/11 |
| Semantic failures | 0 |
| Provider failures | 0 |

Local interpretation latency averaged approximately 0.013 ms in this run. This timing excludes
fixture collection, is environment-specific, and is not a performance benchmark.

## Observations

- All five supported-diagnosis scenarios produced the expected stable diagnosis identifier.
- All six unsupported, incomplete, unknown, or conflicting scenarios produced the expected
  abstention behavior.
- Duplicate log-source entries remained visible and ordered where scenarios contained multiple
  relevant records.
- Structural-response validity is not applicable to deterministic reasoning.
- No Gemini adapter, API credential, or network call was required.

## Known Limitations

The benchmark uses local synthetic fixtures and three deliberately narrow diagnosis families. Its
11/11 result describes behavior on this declared dataset, not general deployment-investigation
accuracy. Deterministic confidence values are policy outputs; this report does not claim that they
are statistically calibrated.

Fake-model tests cover provider failures, invalid structured responses, invalid references,
semantic errors, and agreement analysis. They are validation of the framework, not real-provider
experiment results. The earlier concrete Gemini run remains documented separately in the
[Milestone 6 benchmark](milestone-06-gemini-benchmark.md).
