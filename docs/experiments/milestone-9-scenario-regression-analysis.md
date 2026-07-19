# Milestone 9 Scenario Regression Analysis

## Purpose

Aggregate metrics are useful summaries, but they are insufficient replacement decisions. Equal
accuracy can conceal an improvement in one scenario and a regression in another. Milestone 9
compares the stored scenario results that produced those aggregates.

## Comparison Semantics

Scenarios are matched by stable scenario ID rather than list position. When an experiment contains
multiple investigator results for one scenario, the investigator suffix disambiguates those
results. Unmatched scenarios remain visible and are classified as not comparable.

The primary change classification reuses `semantic_correctness_status` from evaluation:

| Baseline | Candidate | Classification |
|---|---|---|
| incorrect | correct | improved |
| correct | incorrect | regressed |
| correct | correct | unchanged correct |
| incorrect | incorrect | unchanged incorrect |
| unavailable | any | not comparable |

Agreement is retained as an independent dimension. It cannot make an incorrect result correct.

## Failure Attribution

Each failed result receives at most one primary category. Precedence is explicit:

1. provider failure;
2. invalid structured response;
3. invalid evidence reference;
4. other structural validation failure;
5. failed to abstain;
6. unnecessary abstention;
7. wrong supported diagnosis;
8. unknown or incomplete failure.

Execution and validation failures take precedence because no valid semantic result exists.
Incorrect abstention behavior takes precedence over a generic wrong diagnosis because it identifies
the more useful system-level failure.

## Regression Gate

```bash
python -m ai_investigation.experiments compare \
  <baseline-id> <candidate-id> \
  --fail-on-regression
```

Exit status `1` means at least one comparable scenario transitioned from semantically correct to
incorrect. Status `0` means no gated semantic regression was found. Missing or corrupt experiments
return the existing non-regression error status `2`.

The gate does not currently fail on latency, provider changes, optional validation dimensions, or
not-comparable scenarios. Those remain visible for engineering review. This narrow policy forms a
clear bridge to future CI usage without adding CI configuration in this milestone.

## Reports

Text output presents aggregate deltas, scenario classifications, regressions, improvements,
failure-category counts, unmatched scenarios, and a deterministic recommendation. `--json`
serializes the same immutable comparison model with stable ordering and enum values.

No provider cost is compared because experiments do not currently store cost data. No LLM judge,
statistical significance claim, or external evaluation platform is used.

## Limitations

- Old Milestone 8 records remain comparable because their stored scenario fields are sufficient.
- Deterministic trace predicates are not stored in `ScenarioRunResult`, so trace-level changes are
  not fabricated; that dimension remains unavailable for persisted comparison.
- Comparisons are descriptive and local. They do not claim statistical generalization beyond the
  controlled scenario set.
- External observability and evaluation platforms remain deferred until local comparison exposes a
  concrete scaling limitation.
