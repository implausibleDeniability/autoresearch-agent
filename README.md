# PII Autoresearch

An experiment in autonomous LLM research: can an AI agent improve a system that extracts people—names, surnames, and email addresses—from documents?

The project will provide a labeled dataset, an immutable evaluator, and an agent-editable PII extraction implementation. The agent will run experiments and optimize for extraction quality and cost, while recording each hypothesis and result.

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
uv run python -m src.evaluation.cli --dataset dev-19k
uv run python -m src.evaluation.cli --dataset dev-87k
```

Use `debug` for inexpensive pipeline checks and `dev-19k` for most experiments. Because `dev-87k`
costs several times more, use it for occasional checks and final measurement.

Write a detailed error inventory during the same evaluation with:

```bash
uv run python -m src.evaluation.cli --dataset dev-19k --diagnostics diagnostics.json
```

The ignored `diagnostics.json` file contains schema-v1 aggregate and per-field metrics, raw
predictions, raw ground truth, person and field-value matches, false positives, false negatives, and
literal source occurrences. Occurrence offsets are zero-based Python character indexes with an
end-exclusive `end`; an empty `occurrences` list means the raw value was not found literally in the
source. This is expected for some normalized or fuzzy matches. The file contains labeled PII and
source context, is overwritten on each run, has owner-only permissions, and must not be committed.
Other output paths are supported but are not ignored automatically.
Each value stores at most 20 source snippets; `occurrence_count` preserves the total and
`occurrences_truncated` reports whether additional occurrences were omitted.

The top-level shape is:

```json
{
  "schema_version": 1,
  "dataset": "dev-19k",
  "metrics": {"true_positive": 0, "false_positive": 0, "false_negative": 0},
  "field_metrics": {"email": {"true_positive": 0, "false_positive": 0, "false_negative": 0}},
  "documents": []
}
```

`uv run pytest` runs the offline suite. `uv run pytest -m live` runs the paid synthetic and visible
development-data checks.

## Evaluation and cost

The evaluation CLI runs the current editable solution and reports wall-clock duration, immediate API
cost, and one quality metric: F-score with recall weighted five times as heavily as precision
(`beta² = 5`):

```bash
uv run python -m src.evaluation.cli --dataset dev-19k
```

Runs should target no more than $1.50 per million source tokens, but only the 8-cent total-cost guard
is enforced (overridable with `--cents-limit`), allowing modest overruns to return useful results. Once
the guard is exceeded, new requests are rejected, though in-flight requests may add some overshoot.

The evaluator keeps the real `OPENAI_API_KEY` in its own process. It gives the solution subprocess a
short-lived token and redirects the OpenAI SDK through an evaluator-owned localhost meter. The meter
supports Chat Completions and Responses requests using the pinned GPT-4o and GPT-4o mini models,
including structured outputs, local function calling, prompt caching, retries, concurrency, and
streaming. Missing usage, unknown pricing, unsupported models, and unsupported billable endpoints
fail the evaluation instead of reporting an incomplete cost.

Cost is reported as total USD and USD per million original source-document tokens. The denominator
uses `o200k_base` and counts each source document once, independent of solution chunking or repeated
context. The price table is evaluator-owned and currently uses standard API pricing for the supported
models. The CLI prints `f_score` as the optimization metric and precision and recall as diagnostics
for experiment logs.

Autonomous research stops after 20 evaluations or $0.50 in cumulative API cost. Each run logs its
dataset and observed cost; a crash that hides its cost is charged the pre-run estimate.

## Data

- `data/raw/industry-document-baseline`: the complete, unchanged source dataset.
- `data/debug`: one PII-positive and one PII-negative document totaling less than 1,000 tokens.
- `data/dev-19k`: 17 complete documents, 19,346 tokens, 11,162 words, 74 labeled people, and 198
  labeled PII values.
- `data/dev-87k`: 20 complete documents, 87,454 tokens, 49,953 words, 93 labeled people, and 230
  labeled PII values.

`data/raw` is archival source data and should not be exposed to the research agent.
