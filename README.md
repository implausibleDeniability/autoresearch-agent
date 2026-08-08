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
uv run python evaluator.py --dataset debug
uv run python evaluator.py --dataset dev-5k
uv run python evaluator.py --dataset dev-50k
```

Use `debug` to check that an experiment works, `dev-5k` for routine quality comparisons, and
`dev-50k` for broader validation.

`uv run pytest` runs the offline suite. `uv run pytest -m live` runs the paid synthetic and visible
development-data checks.

## Data

- `data/raw/industry-document-baseline`: the complete, unchanged source dataset.
- `data/debug`: one PII-positive and one PII-negative document totaling less than 1,000 tokens.
- `data/dev-5k`: 16 complete documents, 4,361 words, 74 labeled people, and 198 labeled PII values.
- `data/dev-50k`: 20 complete documents, 49,953 words, 93 labeled people, and 230 labeled PII values.

`data/raw` is archival source data and should not be exposed to the research agent.
