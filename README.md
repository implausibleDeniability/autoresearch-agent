# PII Autoresearch

An experiment in autonomous LLM research: can an AI agent improve a system that extracts people—names, surnames, and email addresses—from documents?

The project provides a labeled dataset, an immutable evaluator, and an agent-editable PII
extraction implementation. The agent runs experiments to maximize a recall-weighted entity F-score
while keeping cost at or below $1.50 per million original source-document tokens, recording each
hypothesis and result.

Inspired by [Karpathy's Autoresearch](https://github.com/karpathy/autoresearch), this project applies a similar autonomous experimentation loop to improving LLM prompts rather than training neural networks.

## Method

Development uses a small visible development set. A two-document debug set supports inexpensive
pipeline checks. A separate test split can be added later.

## Solution

`solution.extract_pii(text: str) -> list[PIIItem]` extracts people and their PII from one document.
The initial solution preserves Beagle's multi-pass detection, structuring, validation, and
deterministic merge flow, while leaving out SpaCy, Azure services, storage, caching, progress
reporting, and cross-document deduplication.

The extractor calls the pinned `gpt-4o-2024-08-06` and `gpt-4o-mini-2024-07-18` models. Non-empty
input therefore requires network access and an `OPENAI_API_KEY`, and each run incurs OpenAI API
costs. Empty or whitespace-only input returns an empty list without creating an API client.

Install the Python 3.12 environment with `uv sync`. After exporting `OPENAI_API_KEY`, call the
extractor directly:

```python
from solution import extract_pii

people = extract_pii("John Smith can be reached at john@example.com.")
```

Evaluate the solution on one complete dataset split:

```bash
uv run python -m src.evaluation.cli --dataset debug
uv run python -m src.evaluation.cli --dataset dev-5k
uv run python -m src.evaluation.cli --dataset dev-50k
```

Use `debug` to check that an experiment works, `dev-5k` for routine quality comparisons, and
`dev-50k` for broader validation.

`uv run pytest` runs the offline suite. `uv run pytest -m live` runs the paid synthetic and visible
development-data checks.

## Evaluation and cost

The evaluation CLI runs a baseline or editable candidate solution and reports quality plus immediate
API cost:

```bash
uv run python -m src.evaluation.cli --dataset dev-5k
```

The primary quality objective is the entity F-score with recall weighted five times as strongly as
precision:

```
weighted entity F-score = 6 * precision * recall / (5 * precision + recall)
```

Entity precision and recall, people-level precision, recall, and F-score, and document accuracy are
reported as diagnostics. Experiment selection is driven primarily by the weighted entity F-score,
and only candidates costing at most $1.50 per million original source-document tokens are eligible.
For effectively equal objective scores, prefer the simpler solution, then the less expensive one.

Before every paid evaluation, estimate both total API cost and normalized cost from the expected
model calls and the dataset's original source-token count. Do not start an experiment estimated to
exceed the $1.50 normalized limit.

The evaluator also applies a default absolute guard of $0.08 to every run. Larger intentional runs
require an explicit limit in cents, for example:

```bash
uv run python -m src.evaluation.cli --dataset dev-50k --cents-limit 12
```

An override changes only the absolute guard; it does not relax the $1.50 normalized limit. Larger
datasets can cost more than $0.08 in total while remaining acceptable per million source tokens, so
broader validation may need an override after both estimates have been checked. Use the smallest
justified override and continue to prefer inexpensive development runs.

The evaluator keeps the real `OPENAI_API_KEY` in its own process. It gives the solution subprocess a
short-lived token and redirects the OpenAI SDK through an evaluator-owned localhost meter. The meter
supports Chat Completions and Responses requests using the pinned GPT-4o and GPT-4o mini models,
including structured outputs, local function calling, prompt caching, retries, concurrency, and
streaming. Missing usage, unknown pricing, unsupported models, and unsupported billable endpoints
fail the evaluation instead of reporting an incomplete cost.

Cost is reported as total USD and USD per million original source-document tokens. The denominator
uses `o200k_base` and counts each source document once, independent of solution chunking or repeated
context. The price table is evaluator-owned and currently uses standard API pricing for the supported
models.

## Data

- `data/raw/industry-document-baseline`: the complete, unchanged source dataset.
- `data/debug`: one PII-positive and one PII-negative document totaling less than 1,000 tokens.
- `data/dev-5k`: 16 complete documents, 4,361 words, 74 labeled people, and 198 labeled PII values.
- `data/dev-50k`: 20 complete documents, 49,953 words, 93 labeled people, and 230 labeled PII values.

`data/raw` is archival source data and should not be exposed to the research agent.
