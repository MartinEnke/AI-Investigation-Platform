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

Milestone 12.1 defines that vocabulary. Milestone 12.2 defines a pure policy over it. Neither is
connected to a current investigator yet, so all existing behavior and benchmark results remain
unchanged.

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

## Why not prompt v4?

Prompt tuning remains frozen. Changing the prompt after observing holdout failures would optimize
against the holdout and weaken its value as a generalization test. Milestone 12 instead makes the
decision responsibility explicit before any new model output is designed.

## Why not hybrid orchestration yet?

Combining deterministic and probabilistic investigators before they share an uncertainty contract
would hide coupling inside orchestration code. A common assessment and policy boundary must be
reviewed first so later combinations have explicit inputs and responsibilities.

## Why not LangGraph?

The current need is one side-effect-free function over immutable data. It has no dynamic routing,
stateful workflow, tool loop, or distributed execution requirement. An orchestration framework
would add machinery without solving a demonstrated problem.

## Future integration

Later milestones may:

1. expose structured uncertainty signals from the LLM;
2. adapt investigator results into `UncertaintyAssessment`;
3. evaluate decision correctness separately from reasoning correctness;
4. compare or combine investigators through the shared decision model.

Those steps will require independent review and evaluation. This milestone makes no benchmark
improvement claim.
