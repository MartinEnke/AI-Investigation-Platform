# AI Investigation Platform

A small, deterministic proof of concept that explains deployment failures from local JSON evidence.
Milestone 3 supports health-check timeouts, missing required environment variables, and database
migration failures while explicitly abstaining on unsupported or conflicting evidence.

## Setup

Python 3.13 or newer is required.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

## Usage

Run an investigation from the repository root:

```bash
python -m ai_investigation.cli "Why did deployment deploy-1042 fail?"
```

Run all tests with:

```bash
pytest
```

Run the deterministic evaluation scenarios with:

```bash
python -m ai_investigation.evaluation.runner \
  tests/fixtures/evaluation_scenarios.json \
  --fixtures tests/fixtures
```

The investigation reads deployment, error-log, and service-health fixtures in order, then evaluates
a fixed tuple of three plain deterministic rules. Every rule is evaluated; zero matches are
inconclusive and multiple matches produce an explicit conflict abstention. It makes no network or
model calls.
