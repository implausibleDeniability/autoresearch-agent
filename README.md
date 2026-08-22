# PII Autoresearch

An experiment in autonomous LLM research: can an AI agent improve a system that extracts people—names, surnames, and email addresses—from documents?

The project will provide a labeled dataset, an immutable evaluator, and an agent-editable PII extraction implementation. The agent will run experiments and optimize for extraction quality and cost, while recording each hypothesis and result.

Inspired by [Karpathy's Autoresearch](https://github.com/karpathy/autoresearch), this project applies a similar autonomous experimentation loop to improving LLM prompts rather than training neural networks.

## Method

Development uses the visible `dev-19k`, `dev-87k`, and `dev-205k` datasets. A two-document `debug`
dataset supports inexpensive pipeline checks. Dataset names beginning with `test-` are reserved for
a blind final evaluation after development has ended.

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
uv run python -m src.evaluation.cli --dataset dev-205k
```

Solution processing has a three-minute deadline. Use `--timeout SECONDS` for a shorter deadline;
values above 180 seconds are rejected.

Evaluations process up to 50 documents concurrently by default. Lower the limit to reduce
simultaneous worker and API load:

```bash
uv run python -m src.evaluation.cli --dataset dev-205k --max-concurrent-documents 20
```

Each concurrent development document uses an isolated metering credential, preserving exact
per-document API usage, cost, latency, failures, and partial results.

Use `debug` for inexpensive pipeline checks, `dev-19k` for cheap hypothesis tests, and `dev-87k`
for moderate-cost validation. The larger `dev-205k` is likely more representative for measuring
quality, but costs more to evaluate; use it for generality checks and final measurement.

Inspect a development dataset's scale before planning paid runs:

```bash
uv run python -m src.evaluation.cli --dataset dev-205k --describe-dataset
```

This free, read-only mode reports the document count and aggregate `o200k_base` source-token
distribution. It does not require API credentials, inspect labels, run the solution, or count as an
evaluation. The evaluator rejects it for every `test-*` dataset.

Write a detailed error inventory during the same evaluation with:

```bash
uv run python -m src.evaluation.cli --dataset dev-19k --diagnostics diagnostics.json
```

The ignored `diagnostics.json` file contains schema-v3 run state, per-document execution and cost,
metrics, raw predictions and ground truth, matching ledgers, errors, and evaluator-compatible source
evidence. Source evidence uses the
evaluator's value normalization and fuzzy threshold, but it does not reproduce person pairing,
uniqueness, or one-to-one assignment. A zero evidence count means no accepted source span was found,
not that the evaluator must reject the value.

Raw evidence preserves case-sensitive substrings. Normalized evidence can come from same-width
substrings or whitespace-delimited token windows. Fuzzy evidence also considers punctuation-delimited
subspans such as `Name:Jonh,` → `Jonh`. Overlaps prefer raw, then normalized evidence; fuzzy selection
maximizes the number of non-overlapping spans. Offsets index the original Python string and `end` is
exclusive. Each value stores at most 20 spans while counts describe the full selected set.
Ground-truth values also expose accepted OCR variants. `source_value` identifies the accepted form
used for evidence, or is `null` when none matched.

Fuzzy work is bounded so malformed or unusually long model output cannot stall diagnostics. When
any accepted form's fuzzy search is incomplete, `fuzzy_search_complete` is false. Literal counts
remain complete, but the fuzzy count is unavailable; do not interpret a zero fuzzy count as absence
of evaluator-compatible evidence.

Earlier schema versions use different value or occurrence semantics and must be regenerated.
Consumers should reject unknown `schema_version` values. The file contains labeled PII and source
context, is overwritten on each run, has owner-only permissions, and must not be committed. Other
paths are not ignored automatically. The CLI reports serialization time as
`diagnostics_duration_seconds`.

Diagnostics are development-only. The evaluator rejects `--diagnostics` for every dataset whose
name starts with `test-`.

The top-level shape is:

```json
{
  "schema_version": 3,
  "source_matching_policy": {
    "version": 1,
    "normalization": "lower_strip_trailing_period_v1",
    "similarity_algorithm": "difflib_sequence_matcher_autojunk_false",
    "similarity_threshold": 0.65,
    "minimum_fuzzy_length": 3,
    "fuzzy_work_budget": 50000000,
    "candidate_enumeration_budget": 200000,
    "comparison_orientation": "prediction_values_value_to_span_ground_truth_values_span_to_value",
    "candidate_boundaries": "literal_substrings_and_punctuation_delimited_token_windows",
    "overlap_policy": "raw_then_normalized_then_maximum_cardinality_v1"
  },
  "dataset": "dev-19k",
  "lifecycle_status": "terminal",
  "result_status": "complete",
  "cost_status": "complete",
  "coverage": {"total": 17, "completed": 17, "failed": 0, "not_attempted": 0},
  "metrics": {"true_positive": 0, "false_positive": 0, "false_negative": 0},
  "field_metrics": {"email": {"true_positive": 0, "false_positive": 0, "false_negative": 0}},
  "documents": [],
  "document_results": []
}
```

The evaluator writes an initial `running` checkpoint, replaces it after each completion batch, and
writes the terminal state after cost finalization. Each document ends as `completed`, `failed`, or
`not_attempted`. Failed and unattempted execution records contain no predictions or ground truth.

Every prediction, ground-truth value, false positive, and false negative uses this shape:

```json
{
  "person_index": 0,
  "value_index": 0,
  "value": "john@example.com",
  "variants": [],
  "source_value": "john@example.com",
  "source_evidence_count": 1,
  "raw_occurrence_count": 0,
  "normalized_occurrence_count": 1,
  "fuzzy_occurrence_count": 0,
  "fuzzy_search_complete": true,
  "source_evidence_truncated": false,
  "source_evidence": [{
    "start": 12,
    "end": 28,
    "match_kind": "normalized_exact",
    "similarity": 1.0,
    "context_start": 0,
    "context_end": 50,
    "context": "Contact JOHN@EXAMPLE.COM for details."
  }]
}
```

List false negatives with their evidence counts:

```bash
jq '.documents[] | .document_id as $document_id | .field_results[] | .field as $field | .false_negatives[] | {document_id: $document_id, field: $field, value, source_evidence_count, raw_occurrence_count, normalized_occurrence_count, fuzzy_occurrence_count}' diagnostics.json
```

Add `select(.fuzzy_occurrence_count > 0)` or `select(.source_evidence_count == 0)` before the final
object to isolate fuzzy-supported or absent values.

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
is enforced (overridable with `--cents-limit`). Before forwarding a request, the meter reserves its
maximum possible cost against the limit. Requests wait when their reservation would exceed the
remaining budget. One request may still exceed the limit so the run can return useful results;
further requests are rejected.

The evaluator keeps the real `OPENAI_API_KEY` in its own process. It gives the solution subprocess a
short-lived token and redirects the OpenAI SDK through an evaluator-owned localhost meter. The meter
supports Chat Completions and Responses requests using the pinned GPT-4o and GPT-4o mini models,
including structured outputs, local function calling, prompt caching, retries, concurrency, and
streaming. Missing usage, unknown pricing, unsupported models, and unsupported billable endpoints
mark cost accounting incomplete without discarding observed spend.

Cost is reported as total USD and USD per million original source-document tokens. The denominator
uses `o200k_base` and counts each source document once, independent of solution chunking or repeated
context. The price table is evaluator-owned and currently uses standard API pricing for the supported
models. The CLI prints `f_score` as the optimization metric and precision and recall as diagnostics
for experiment logs.

Development allows 40 evaluations. Reserve enough of the $0.50 cumulative API budget for the final
test. Charge a crash without an observed cost at its pre-run estimate.

Development output includes `result_status=complete|partial`, `score_is_final=true|false`, coverage,
a termination category, and `cost_status=complete|incomplete`. Complete runs exit 0 and use the normal
metric keys. Partial runs exit 2 and use `partial_*` metrics over completed documents only. These
metrics guide diagnosis but are never final scores. `document_results_json` reports each document's
status, source and API tokens, observed spend, latency, and safe failure category.

When metering is incomplete, budget accounting uses the larger of observed spend and the pre-run
estimate.

### Blind final evaluation

Researchers may discover the complete blind dataset name by listing directories matching
`data/test-*`, but must not inspect their contents or detailed test results without explicit
permission.

After development is complete, commit the chosen `solution.py` and leave it unchanged. Run the one
final blind evaluation by passing that current commit explicitly:

```bash
uv run python -m src.evaluation.cli \
  --dataset test-<discovered-name> \
  --frozen-commit "$(git rev-parse HEAD)"
```

The evaluator checks the commit and `solution.py` before and after the run. It prints only aggregate
`f_score`, precision, recall, API cost, and duration. This final evaluation is outside the 40-run
allowance, but its spend counts toward the $0.50 budget. Success ends the research run. Never tune
against the result; changing the solution invalidates it.

## Data

- `data/raw/industry-document-baseline`: the complete, unchanged source dataset.
- `data/debug`: one PII-positive and one PII-negative document totaling less than 1,000 tokens.
- `data/dev-19k`: 17 complete documents, 19,346 tokens, 11,162 words, 74 labeled people, and 203
  labeled PII values.
- `data/dev-87k`: 20 complete documents, 87,454 tokens, 49,953 words, 93 labeled people, and 238
  labeled PII values.
- `data/dev-205k`: 122 complete documents, 204,153 tokens, 98,127 words, 535 labeled people, and
  1,329 labeled PII values.
- `data/test-*`: blind final-evaluation data.

`data/raw` must not be exposed to the research agent. The agent may list `data/test-*` directory
names but must not inspect their contents without explicit permission.

### Ground-truth labels

Field lists contain distinct logical PII values. A plain string has one accepted form. OCR
ambiguity or a supported alias can instead use one canonical value with source-visible variants:

```json
{
  "canonical": "kenny.shannon@epa.gov",
  "variants": ["kenny.shannon@8pa.gov"]
}
```

The canonical value and its variants count as one label; any accepted transcription can match it
once. Variants never represent separate phone numbers, email addresses, or other distinct values.

Labels retain explicit partial names and person-associated contact details, including fax, direct,
and secondary numbers. Professional mailing addresses and profile locations count when the document
associates them with a person. Shared venues and organization addresses do not. Annotators must
inspect the rendered page when layout determines the association. Name-component boundaries must
remain consistent across dataset splits.
