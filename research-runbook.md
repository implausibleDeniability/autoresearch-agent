# Research runbook

This is the technical companion to `program.md`. Follow it for setup, evaluation, logging, candidate management, and finalization. `program.md` remains authoritative for research goals, decisions, and guardrails.

## Setup

1. Read the repository and user instructions for workspace location and naming.
2. In the primary repository, check out `main` and pull the latest remote `main`. Stop if the pull fails.
3. Create `autoresearch/<tag>` from updated `main`, using a filesystem-safe day-and-time tag such as `aug-8-11-52`. When the instructions require a worktree, create and enter it with `git worktree add -b autoresearch/<tag> <path> main`. If workspace placement is unspecified, ask before choosing it.
4. For a new worktree, copy only required ignored local files such as `.env`, preserve their permissions, and confirm they remain ignored.
5. Create `results.tsv` with only the header defined below, `research.md` with a short ranked experiment portfolio, and the ignored `diagnostics/` directory. Do not commit these working files.
6. Review `baseline-results.tsv` to understand ordinary baseline variation, cost, and runtime before spending the first evaluation.
7. Inspect dataset scale with `uv run python -m src.evaluation.cli --dataset dev-87k --describe-dataset`. This command is read-only, requires no API credentials, and does not count as an evaluation.
8. Discover the blind dataset name by listing only directory names matching `data/test-*`. Do not inspect their contents.
9. Before modifying `solution.py`, evaluate the unchanged solution once on `dev-202k` with a new path under `diagnostics/`, following **Before each development evaluation** and **Run a development evaluation** below. Require a complete result and record it as the current baseline `keep` with fresh diagnostics.

## Working files

| File | Purpose |
| --- | --- |
| `results.tsv` | Development evaluation results and cumulative spend |
| `research.md` | Ranked hypothesis portfolio and external research findings |
| `REQUESTS.md` | Information and requests for the human supervisor |
| `run.log` | Output from the latest evaluation |
| `diagnostics/<evaluation>-<commit>-<dataset>.json` | Diagnostics from development evaluations |

Use a new path for every development evaluation, named `diagnostics/<evaluation>-<commit>-<dataset>.json`. The evaluator rejects existing paths. Never request diagnostics for a blind dataset.

## Before each development evaluation

1. Confirm that another evaluation fits within the 40-evaluation limit. For a live evaluation, also confirm that it fits within the remaining $0.50 API budget while preserving enough budget for the final evaluation.
2. Before a live evaluation, estimate both total API spend and normalized cost. The target is at most $1.50 per million source-document tokens.
3. Choose the dataset and concurrency appropriate for the hypothesis. The evaluator processes up to 50 documents concurrently by default; use `--max-concurrent-documents N` to reduce simultaneous worker and API load.

## Cost and evaluator behavior

Cost is measured in USD per million source-document tokens:

```text
cost = total actual USD cost of all model calls / total tokens in the original source documents * 1,000,000
```

The denominator counts each original document once and excludes system prompts, instructions, repeated context, and generated tokens. Those tokens still contribute to actual API charges in the numerator. The evaluator defines how source-document tokens are counted and prints cost immediately after each run.

The evaluator supports Chat Completions and Responses with the allowed models, including structured outputs, local function calling, prompt caching, retries, concurrency, and streaming. Successful responses that cannot be priced make cost accounting incomplete rather than counting as free. Provider-hosted tools and other billable endpoints are unavailable without an evaluator pricing rule.

The default 8-cent meter is a guard rather than the normalized-cost target. Override the default limit with `--cents-limit` only when the estimated live run fits the total research budget. A request may exceed the remaining per-run limit so the evaluator can return useful results; further requests are then rejected.

### Cached replay

Prefer `--cache` for paired development evaluations whose OpenAI requests should remain identical, such as changes to post-processing, parsing, or thresholds. Exact replay removes model variability, isolating the code change at zero API cost. A cache miss fails without a live call; omit the flag when requests change or when measuring model variation. Cached replay is unavailable for blind evaluations.

Cached evaluations count toward the 40-evaluation limit but spend no API budget. For every cached result, including partial, crash, or missing-status results, record `budget_cost_usd=0`. Cached cost fields cannot support claims about solution cost. Log a cache miss as `crash`; it still counts toward the evaluation limit.

## Run a development evaluation

Run the evaluator in one of these modes:

```bash
# Live evaluation
set -a; source .env; set +a; test -n "$OPENAI_API_KEY" && uv run python -m src.evaluation.cli --dataset dev-19k --diagnostics diagnostics/001-a1b2c3d-dev-19k.json > run.log 2>&1

# Cached replay
uv run python -m src.evaluation.cli --dataset dev-19k --diagnostics diagnostics/002-a1b2c3d-dev-19k.json --cache > run.log 2>&1
```

Substitute the actual evaluation number, commit, and visible dataset. `--timeout` may set a shorter deadline but cannot exceed 180 seconds.

Read the run status before interpreting metrics:

```bash
grep -E '^(result_status|score_is_final|termination_category|evaluation_mode|cache_hits|cache_misses|cache_errors|cache_writes|cache_write_errors|openai_live_requests|documents_completed|documents_failed|documents_not_attempted|cost_status|api_cost_usd|observed_api_cost_usd|f_score|partial_f_score|precision|partial_precision|recall|partial_recall|cost_usd_per_million_source_tokens|partial_cost_usd_per_million_completed_source_tokens|document_results_json)=' run.log
```

### Complete result

When `result_status=complete` and `score_is_final=true`, use the normal score fields to evaluate the candidate. Use cost fields for candidate comparisons only when `evaluation_mode=live`.

### Partial result

When `result_status=partial` and `score_is_final=false`:

- count the attempt toward the evaluation limit;
- use completed-document metrics and diagnostics only to form hypotheses;
- do not rank, keep, discard, or promote the candidate from this run;
- record `crash` with zero score, precision, recall, and normalized-cost fields;
- preserve coverage, the failed document ID, its source, prompt, completion and total token counts, observed cost, latency, and failure category in the diagnostic file;
- record the failure category and diagnostic path in `finding`;
- for a live evaluation with `cost_status=incomplete`, charge the larger of `observed_api_cost_usd` and the pre-run estimate.

### Missing status

If `result_status` is missing, treat the attempt as an evaluator or protocol crash and inspect the final 50 lines of `run.log`. For a live evaluation, charge the pre-run estimate unless a trustworthy observed subtotal is available.

## Diagnostics and failures

Before another paid run, inspect the diagnostic file from the current evaluation. Inventory every false negative and false positive by document ID, field, person index, and exact value; use source evidence to inspect the highest-impact errors. Record concrete examples and the diagnostic path in `research.md`, then name an observed error class in the next hypothesis and summarize the conclusion in `results.tsv` `finding`. Never propose or evaluate a new experiment from aggregate metrics alone.

If a run crashes, preserve and inspect any partial diagnostics. Fix and rerun an easy implementation failure; log `crash` and move on when the idea is fundamentally broken. Never assume a live crash cost $0: charge the observed cost when complete, or the larger of observed cost and the pre-run estimate when metering is incomplete.

## Logging results

Use tab-separated `results.tsv`; commas in descriptions make CSV unsuitable. Start with this header:

```tsv
commit	score	precision	recall	cost	status	description	dataset	budget_cost_usd	finding	evaluation_mode
```

The columns are:

1. seven-character commit hash;
2. recall-weighted F-score, or `0.000000` for crashes;
3. precision, or `0.000000` for crashes;
4. recall, or `0.000000` for crashes;
5. USD per million source-document tokens, or `0.000000` for crashes;
6. `keep`, `discard`, `inconclusive`, or `crash`;
7. short description of what the experiment tried;
8. `debug`, `dev-19k`, `dev-87k`, or `dev-202k`;
9. API cost charged to the total budget: `0` for cached evaluations; for live evaluations with incomplete metering, use the larger of the observed subtotal and pre-run estimate;
10. short evidence-backed conclusion from the result, including the diagnostic path when useful. Every `keep` row on `dev-202k` must have a non-empty finding;
11. `live` or `cached`. Cached zeroes describe replay spend and cannot support solution-cost comparisons.

Record every development evaluation, including repetitions and crashes, as a separate row. Row order determines the experiment number. After deciding, assign the same status to successful repetitions of the same commit and dataset. Recompute cumulative spend after each evaluation. Report the blind result separately.

## Candidate decisions

Commit every changed candidate before evaluating it and record its seven-character commit hash.

A complete result on a smaller development dataset may justify `discard` or `inconclusive`, but never `keep`. Before marking a candidate `keep` or replacing the incumbent, evaluate that exact commit on `dev-202k`. Require `result_status=complete` and `score_is_final=true`; partial results cannot support promotion.

When a candidate does not replace the incumbent, return to the incumbent without deleting its recorded evaluations.

## Final evaluation

1. Commit the selected `solution.py`.
2. Confirm `git diff --quiet HEAD -- solution.py` succeeds.
3. Save the full output of `git rev-parse HEAD` and do not modify the solution afterward.
4. Estimate the blind-evaluation spend and confirm it fits the remaining budget.
5. If it fits, run one evaluation on the discovered blind dataset without diagnostics:

```bash
set -a; source .env; set +a; test -n "$OPENAI_API_KEY" && uv run python -m src.evaluation.cli --dataset 'test-<discovered-name>' --frozen-commit '<full-frozen-commit>' > run.log 2>&1
```

The evaluator verifies the frozen commit before and after the run. Leave its aggregate output in `run.log`, charge the cost, and stop experimenting.

Whether or not the blind evaluation ran, invoke `$report-autoresearch-progress` and use its output as the final report. Invoke `$generate-autoresearch-trajectory` only when the user requests a trajectory.
