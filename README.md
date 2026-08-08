# PII Autoresearch

An experiment in autonomous LLM research: can an AI agent improve a system that extracts people—names, surnames, and email addresses—from documents?

The project will provide a labeled dataset, an immutable evaluator, and an agent-editable PII extraction implementation. The agent will run experiments and optimize for extraction quality and cost, while recording each hypothesis and result.

## Method

Development uses a visible development set. Final evaluation uses a separate hidden test set that the agent cannot access during research.

## Baseline extractor

`baseline_solution.extract_pii(text: str) -> list[PIIItem]` extracts people and their PII from one
document. It preserves Beagle's multi-pass detection, structuring, validation, and deterministic
merge flow, while leaving out SpaCy, Azure services, storage, caching, progress reporting, and
cross-document deduplication.

The extractor calls the pinned `gpt-4o-2024-08-06` and `gpt-4o-mini-2024-07-18` models. Non-empty
input therefore requires network access and an `OPENAI_API_KEY`, and each run incurs OpenAI API
costs. Empty or whitespace-only input returns an empty list without creating an API client.

Install the Python 3.12 environment with `uv sync`. After exporting `OPENAI_API_KEY`, call the
extractor directly:

```python
from baseline_solution import extract_pii

people = extract_pii("John Smith can be reached at john@example.com.")
```

`uv run pytest` runs the offline suite. `uv run pytest -m live` runs the paid synthetic and visible
development-data checks.

## Data

- `data/raw/industry-document-baseline`: the complete, unchanged source dataset.
- `data/dev`: four representative documents available during research.
- `data/test`: the remaining 17 documents reserved for final evaluation.

Only `data/dev` should be exposed to the research agent. Access control for `data/raw` and `data/test` will be enforced by the research runner.
