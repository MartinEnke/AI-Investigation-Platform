# Milestone 12: Uncertainty Model and Deterministic Decision Policy

## Motivation

The frozen Milestone 11 holdout exposed complementary failure modes:

- the deterministic investigator diagnosed 6/10 supported cases and abstained correctly on 10/10,
  producing four unnecessary abstentions;
- the Groq investigator with prompt v3 diagnosed 10/10 supported cases and abstained correctly on
  8/10, producing two false diagnoses through failure to abstain.

These failures are not only language-understanding problems. They expose a missing shared vocabulary
for candidate support, contradiction, incomplete evidence, ambiguity, and review requirements.

> Reasoning proposes candidates.  
> Policy decides whether the evidence justifies diagnosis, abstention, or review.

Milestone 12.1 defines that vocabulary. Milestone 12.2 defines a pure policy over it. Milestone
12.3 connects a new LLM uncertainty proposal to that policy as a separately evaluable path. The
deterministic investigator and the existing LLM v1-v3 path remain unchanged.

## Milestone 12.3 integration

```text
Evidence
  ↓
LLM uncertainty proposal
  ↓
Typed uncertainty adapter
  ↓
Deterministic decision policy
  ↓
Diagnosis / abstention / needs review
  ↓
Public investigation result
```

The `llm-policy` investigator uses the versioned `v4-uncertainty` prompt. The model may propose
zero, one, or multiple supported candidates and reports, for each candidate:

- supporting and contradicting evidence references;
- weak, moderate, or strong evidence;
- self-reported confidence retained only as metadata.

It also reports unsupported signals, material unsupported signals, candidate conflict,
semantic insufficiency, and a short summary. The prompt explicitly
forbids selecting the final diagnosis, hiding a competing candidate, using confidence in place of
evidence, or mapping an unsupported cause to the nearest supported label.

Each LLM-facing evidence item receives a deterministic capability-like identifier (`E1`, `E2`,
`E3`, and so on). The structured response accepts only this syntax. Parsing rejects unknown labels,
duplicate candidates, invalid strengths or confidence, malformed or duplicate references, and any
reference used as both supporting and contradicting evidence. The adapter separately rejects a
syntactically valid ID that is not present in the current request.

The first live run exposed an ambiguous boundary: the response asked the model to declare
`missing_required_sources`, but the model sometimes returned the sources it understood to be
required even when they were available. Exact comparison against deterministic state therefore
stopped otherwise valid proposals before policy evaluation. The corrected contract removes that
redundant model-authored field entirely.

Source terminology is now explicit:

- `required_sources` are the source types application policy requires for a diagnosis;
- `available_sources` are the source types present in `CollectedEvidence`;
- `missing_required_sources` is the deterministic set difference
  `required_sources - available_sources`.

Only application code computes that difference. A model may still propose a candidate when a
source is absent; the adapter marks the candidate incomplete and the unchanged policy abstains with
`missing_required_evidence`. The model's general `insufficient_evidence` signal remains independent
semantic input. Evidence identity and availability belong to deterministic code; semantic
uncertainty belongs to the model.

## Milestone 12.3b: candidate semantics correction

The first working live uncertainty evaluation was stored as
`20260721T070544Z-llm-policy-de20a86-e376eb`. Its integration and validation boundaries worked, but
its proposal semantics did not:

| Dimension | Result |
| --- | ---: |
| Diagnosis accuracy | 0/8 |
| Abstention accuracy | 8/8 |
| Structured responses | 15/15 |
| Evidence references | 15/15 |
| Policy needs review | 13 |
| Single-candidate assessments | 1 |
| Multi-candidate assessments | 14 |

The model returned diagnoses as candidates when they had merely been considered. The deterministic
policy interpreted each candidate according to its documented contract—as an independently
supported cause—and safely produced `needs_review` rather than choosing among them.

### Considered hypothesis versus supported candidate

A considered hypothesis is part of semantic reasoning. A supported candidate is eligible for the
decision policy because direct, diagnosis-specific evidence exists and the cause could independently
explain the failure. Absence of contradiction, shared vocabulary, symptom similarity, or general
plausibility does not make a hypothesis supported.

The new `llm-investigator-v4-uncertainty-candidate-semantics` prompt and
`llm-uncertainty-proposal-v3` schema therefore expose two distinct collections:

- `supported_candidates` contains only policy-eligible causes;
- `rejected_hypotheses` preserves considered but unsupported alternatives with a typed reason,
  concise summary, evidence references, and metadata-only confidence.

Rejection reasons include no direct support, shared vocabulary only, downstream symptom,
contradiction, missing required evidence, a weaker explanation than a supported cause, and absence
from the available supported diagnosis set. Rejected hypotheses are validated for identity and
structure but never enter `UncertaintyAssessment.candidates`, candidate counts, conflict state, or
policy input.

`conflicting_supported_candidates` now refers only to genuine ambiguity among two or more supported
candidates. Zero or one supported candidate cannot declare that conflict. A supported candidate and
several rejected hypotheses is not ambiguous policy input.

### Why policy remains unchanged

The policy behaved correctly for the proposal it received. Weakening the multi-candidate review rule
or selecting the strongest model-ranked candidate would hide a proposal-quality defect and return
decision control to probabilistic ranking. Separating rejected hypotheses preserves transparency
without manufacturing ambiguity.

This correction is a new, explicitly versioned experiment. The prior prompt and schema remain
identifiable for the baseline run, and no performance improvement is claimed until another live
evaluation is recorded.

## Prompt-semantics diagnostic

The candidate-semantics live run still produced no rejected hypotheses and classified most
considered diagnoses as supported. Before selecting another production correction, the repository
therefore includes a small diagnostic command that isolates four possible influences while holding
the scenario and collected evidence constant:

1. **Variant A — plain classification:** no JSON contract; asks for supported candidates, rejected
   hypotheses, and an explanation.
2. **Variant B — forced comparison:** no JSON contract; requires every supported diagnosis label to
   be classified as supported, rejected, or not relevant.
3. **Variant C — minimal JSON:** contains only supported and rejected collections, evidence IDs, and
   short reasons.
4. **Variant D — production control:** uses the current production candidate-semantics prompt and
   response schema unchanged.

Every variant receives the same `CollectedEvidence` and request-local `E<number>` identifiers. The
diagnostic prints the prompt identifier, raw provider response, parsed diagnosis collections, parse
status, and compact counts. It does not instantiate an investigator, invoke decision policy,
calculate benchmark accuracy, emit experiment events, or save a normal experiment record.

Interpretation is deliberately deferred until raw results are inspected:

- if plain text separates hypotheses but JSON does not, investigate schema or structured-output
  interaction;
- if A, B, and C separate hypotheses but D does not, investigate production prompt complexity or
  conflicting instructions;
- if forced Variant B still marks many diagnoses supported, investigate diagnosis eligibility and
  evidence interpretation;
- if the clear scenario separates correctly but ambiguous evidence proliferates, investigate the
  uncertainty threshold or causal distinction;
- if every variant behaves correctly, verify live prompt routing and experiment metadata before
  changing wording.

This diagnostic is observational. It does not modify production behavior, and no architectural or
prompt correction should be selected before comparing both the clear supported scenario and an
ambiguous or unsupported scenario.

## Diagnostic-versus-production parity investigation

The production benchmark reported zero rejected hypotheses, while diagnostic Variant D populated
them correctly for both a clear supported case and conflicting evidence. This established that the
model could learn and emit the distinction. Tracing both paths verified a prompt-routing mismatch,
not a candidate-mapping or serialization defect.

The diagnostic Variant D path is:

```text
diagnostic CLI
→ load scenario
→ EvidenceCollector
→ build_variant_prompt("d")
→ v4-uncertainty-candidate-semantics prompt / schema v3
→ diagnostic Groq JSON request
→ parse_candidate_semantics_proposal
→ diagnostic rendering
```

It intentionally stops before adaptation and policy. The production path is:

```text
evaluation CLI
→ resolve requested prompt selection
→ load scenario
→ EvidenceCollector
→ LLMPolicyInvestigator
→ selected uncertainty prompt
→ structured provider request
→ selected proposal parser
→ proposal_to_uncertainty
→ deterministic decision policy
→ ScenarioRunResult
→ EvaluationReport
→ optional ExperimentRecord
```

The benchmark command requested `--prompt-version v4-uncertainty`. That historical selection still
resolves to `llm-investigator-v4-uncertainty-contract-v2` and
`llm-uncertainty-proposal-v2`, whose response contract has a legacy `candidates` list and no
`rejected_hypotheses` field. Diagnostic Variant D explicitly resolves
`v4-uncertainty-candidate-semantics` to
`llm-investigator-v4-uncertainty-candidate-semantics` and
`llm-uncertainty-proposal-v3`. The old alias was intentionally preserved for experiment
reproducibility, so the two commands did not run the same prompt.

An end-to-end same-response test now proves that schema-v3 rejected hypotheses survive unchanged
through the production parser, investigator result, evaluator projection, JSON serialization,
stored experiment record, terminal scenario rendering, and aggregate counters. Only supported
candidates enter `UncertaintyAssessment` and policy. Therefore neither candidate mapping nor result
propagation caused the observed zero count.

Text evaluation output now displays the requested selection, resolved prompt identifier, and
resolved schema. Saved experiment metadata already stores the requested selection in configuration
and the resolved prompt and schema in typed metadata. `--debug-uncertainty` additionally writes raw
and parsed per-scenario tracing to standard error; ordinary execution does not expose raw provider
responses.

## Milestone 12.4: contract hardening

The next live candidate-semantics benchmark reached the intended reasoning behavior but produced
five structural failures. The recurring error was disagreement over
`conflicting_supported_candidates`. This value was model-authored even though it is exactly
derivable from the complete supported-candidate collection.

The schema audit found only one perfectly derived semantic-contract field:

```text
conflicting_supported_candidates = len(supported_candidates) > 1
```

The active `v4-uncertainty-candidate-semantics` selection now resolves to
`llm-investigator-v4-uncertainty-candidate-semantics-contract-v2` and
`llm-uncertainty-proposal-v4`. Schema v4 removes the conflict field. Parsing always derives it from
the supported list. For compatibility, the parser also accepts a legacy schema-v3 response that
still contains the field, ignores its value, and derives the authoritative result. Stored
experiments retain their original prompt and schema metadata unchanged.

No other uncertainty field was removable without losing semantic input:

- unsupported-signal presence and materiality are model observations;
- insufficiency is a semantic assessment independent of deterministic source coverage;
- evidence strength, confidence, candidate eligibility, rejection reasons, and reasoning summary
  are not derivable from collection structure alone.

This hardening does not change candidate semantics or policy. Two supported candidates still derive
conflict and produce `needs_review`; zero or one cannot produce conflict.

The source-reporting inconsistency was separate and limited to evaluator projection. Expected
evidence sources may contain repeated categories because multiple log records are expected. The
previous multiset subtraction could therefore report `logs` as both referenced and missing, while
terminal rendering summarized referenced sources by category. Scenario source differences are now
computed as ordered unique categories. This changes neither evidence-reference validation nor
diagnosis/abstention scoring, but guarantees that missing sources are unique and disjoint from
referenced sources.

The unchanged decision policy remains authoritative. A successful diagnosis, abstention, or review
is converted into the existing public `InvestigationResult`. Because that public model has no
native review outcome, `needs_review` is exposed as an abstention; the new evaluation result retains
the typed policy outcome, reason, and candidate labels. `needs_review` is reserved for multiple
plausible supported causes or material unresolved contradiction, not for ordinary insufficient
evidence.

Experiment metadata records `llm-policy`, provider, model,
`llm-investigator-v4-uncertainty-contract-v2`, the
uncertainty response-schema version, and `deterministic-decision-policy-v1`. Scenario reports expose
policy outcome, policy reason, and candidates; aggregates count diagnoses, abstentions, reviews,
single-candidate assessments, and multi-candidate assessments. Existing evaluation accuracy and
comparison semantics are unchanged: a review is an abstention for current scoring.

## Domain model

The new structures are immutable and slotted:

- `CandidateDiagnosis` identifies a supported cause, supporting references, contradicting
  references, and explicit evidence strength;
- `CandidateAssessment` records reference validity, required-source completeness, and optional
  reported confidence;
- `UncertaintyAssessment` groups candidates and cross-candidate uncertainty facts;
- `InvestigationDecision` records `diagnosis`, `abstention`, or `needs_review` with a typed reason;
- `DecisionOutcome`, `DecisionReason`, and `EvidenceStrength` provide finite machine-readable
  vocabularies.

Evidence strength is not confidence. Strength represents the quality and completeness of evidence
available to policy. Confidence is an observed belief value and is retained only for analysis. The
policy never reads confidence when selecting an outcome; a confidence of 0.05 and 0.99 therefore
produce the same decision for otherwise identical evidence.

## Deterministic policy

Policy precedence is explicit:

| Assessment | Outcome | Reason |
| --- | --- | --- |
| Invalid or empty candidate references | Abstention | `invalid_candidate_evidence` |
| Required sources missing | Abstention | `missing_required_evidence` |
| Assessment explicitly insufficient | Abstention | `insufficient_evidence` |
| No candidates, no unsupported signals | Abstention | `no_supported_candidate` |
| Unsupported signals only | Abstention | `unsupported_signals_only` |
| Multiple supported candidates | Needs review | `conflicting_supported_candidates` |
| Material contradiction against one candidate | Needs review | `unresolved_contradictory_evidence` |
| One weak candidate | Abstention | `insufficient_evidence` |
| One moderate or strong complete candidate | Diagnosis | `single_supported_candidate` |

Unsupported signals that are present but immaterial do not block an otherwise valid single
candidate. Material unsupported or contradictory evidence does. Multiple candidates are never
silently reduced by declaration order or reported confidence.

## Validation rules

Construction rejects:

- unsupported diagnosis identifiers;
- non-positive, duplicate, or overlapping supporting and contradicting references;
- duplicate candidate diagnoses;
- missing-source states that do not identify the missing sources;
- a conflict flag without at least two candidates;
- material unsupported signals not marked as present;
- diagnosis outcomes without an assessed supported diagnosis;
- diagnoses attached to abstention or review outcomes;
- `requires_review` values inconsistent with the outcome.

These checks validate policy input structure. They do not replace existing evidence-reference or
source-coverage validators.

## Why v3 remains frozen

Prompt v3 is the frozen Milestone 11 comparison baseline. Changing it after observing holdout
failures would optimize against the holdout and weaken its value as a generalization test. The
uncertainty prompt is a new experimental output contract, not a replacement or revision of v3.

## Why the LLM does not choose the result

The LLM contributes semantic generalization and hypothesis generation. Deterministic code validates
the proposal and controls whether the evidence justifies an automatic diagnosis, safe abstention,
or review. Confidence remains observable but cannot override weak, incomplete, or conflicting
evidence.

## Why not hybrid orchestration yet?

Combining deterministic and probabilistic investigators before they share an uncertainty contract
would hide coupling inside orchestration code. A common assessment and policy boundary must be
reviewed first so later combinations have explicit inputs and responsibilities.

## Why not LangGraph?

The current need is one side-effect-free function over immutable data. It has no dynamic routing,
stateful workflow, tool loop, or distributed execution requirement. An orchestration framework
would add machinery without solving a demonstrated problem.

## Experiment hypothesis and boundary

The hypothesis is deliberately prospective:

> The new path may retain the LLM's semantic generalization while reducing false diagnoses in
> ambiguous cases.

No benchmark improvement is claimed before the live experiment. The path remains a single provider
call followed by pure validation, adaptation, policy, and result conversion. Multiple investigators,
retries, dynamic routing, LangGraph, and a human-review interface remain outside this milestone.
