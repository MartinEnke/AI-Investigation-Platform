# AI Investigation Platform

A small, deterministic proof of concept that explains deployment failures from local JSON evidence.

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

The first milestone uses only read-only JSON fixtures and deterministic rules. It makes no network or model calls.

