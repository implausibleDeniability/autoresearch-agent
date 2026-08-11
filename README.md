# PII Autoresearch

An experiment in autonomous LLM research: can an AI agent improve a system that extracts people—names, surnames, and email addresses—from documents?

The project will provide a labeled dataset, an immutable evaluator, and an agent-editable PII extraction implementation. The agent will run experiments and optimize for extraction quality and cost, while recording each hypothesis and result.

Inspired by [Karpathy's Autoresearch](https://github.com/karpathy/autoresearch), this project applies a similar autonomous experimentation loop to improving LLM prompts rather than training neural networks.

## Method

Development uses the visible `dev-19k` and `dev-87k` datasets. A two-document `debug` dataset
supports inexpensive pipeline checks. Dataset names beginning with `test-` are reserved for a blind
final evaluation after development has ended.

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

Diagnostics are development-only. The evaluator rejects `--diagnostics` for every dataset whose
name starts with `test-`.

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

Development allows 20 evaluations. Reserve enough of the $0.50 cumulative API budget for the final
test. Charge a crash without an observed cost at its pre-run estimate.

### Blind final evaluation

Researchers must not access `data/test-*` files or detailed test results. The complete dataset name
is supplied for the final evaluation rather than hard-coded.

After development is complete, commit the chosen `solution.py` and leave it unchanged. Run the one
final blind evaluation by passing that current commit explicitly:

```bash
uv run python -m src.evaluation.cli \
  --dataset test-<provided-name> \
  --frozen-commit "$(git rev-parse HEAD)"
```

The evaluator checks the commit and `solution.py` before and after the run. It prints only aggregate
`f_score`, precision, recall, API cost, and duration. This final evaluation is outside the 20-run
allowance, but its spend counts toward the $0.50 budget. Success ends the research run. Never tune
against the result; changing the solution invalidates it.

## Data

- `data/raw/industry-document-baseline`: the complete, unchanged source dataset.
- `data/debug`: one PII-positive and one PII-negative document totaling less than 1,000 tokens.
- `data/dev-19k`: 17 complete documents, 19,346 tokens, 11,162 words, 73 labeled people, and 196
  labeled PII values.
- `data/dev-87k`: 20 complete documents, 87,454 tokens, 49,953 words, 92 labeled people, and 228
  labeled PII values.
- `data/test-*`: blind final-evaluation data.

`data/raw` and `data/test-*` must not be exposed to the research agent.
