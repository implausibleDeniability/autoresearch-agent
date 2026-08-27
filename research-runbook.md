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
9. Before modifying `solution.py`, evaluate the unchanged solution on `dev-202k` with seeds `0` through `4`, using a new path under `diagnostics/` for each run. Follow **Before each development evaluation** and **Run a development evaluation** below. Require five complete results and record them as the current baseline `keep` panel.

## Working files

| File | Purpose |
| --- | --- |
| `results.tsv` | Development evaluation results and cumulative spend |
| `research.md` | Ranked hypothesis portfolio and external research findings |
| `REQUESTS.md` | Information and requests for the human supervisor |
| `run.log` | Output from the latest evaluation |
| `diagnostics/<evaluation>-<commit>-<dataset>-seed<seed>.json` | Diagnostics from development evaluations |

Use a new path for every development evaluation, named `diagnostics/<evaluation>-<commit>-<dataset>-seed<seed>.json`. The evaluator rejects existing paths. Never request diagnostics for a blind dataset.

## Before each development evaluation

1. Confirm that another evaluation fits within the 40-evaluation limit. For `--cache-fill` or `--fresh`, also confirm that possible live requests fit within the remaining $0.50 API budget while preserving enough budget for the final evaluation.
2. Before a mode that can call OpenAI, estimate total API spend. Estimate normalized cost when the result can be comparable. The target is at most $1.50 per million source-document tokens.
3. Choose the dataset and concurrency appropriate for the hypothesis. The evaluator processes up to 50 documents concurrently by default; use `--max-concurrent-documents N` to reduce simultaneous worker and API load.
4. Choose the hypothesis before the cache mode. Cache availability must not raise an experiment's research priority.
5. Choose the seed before the run. Use seeds `0` through `4` for the baseline and every comparable five-run `dev-202k` candidate, pairing results by seed without additions, omissions, or substitutions.

## Cost and evaluator behavior

Cost is measured in USD per million source-document tokens:

```text
cost = total actual USD cost of all model calls / total tokens in the original source documents * 1,000,000
```

The denominator counts each original document once and excludes system prompts, instructions, repeated context, and generated tokens. Those tokens still contribute to actual API charges in the numerator. The evaluator defines how source-document tokens are counted and prints cost immediately after each run.

The evaluator supports Chat Completions and Responses with the allowed models, including structured outputs, local function calling, prompt caching, retries, concurrency, and streaming. Successful responses that cannot be priced make cost accounting incomplete rather than counting as free. Provider-hosted tools and other billable endpoints are unavailable without an evaluator pricing rule.

The default 8-cent meter is a guard rather than the normalized-cost target. Override the default limit with `--cents-limit` only when the estimated live run fits the total research budget. A request may exceed the remaining per-run limit so the evaluator can return useful results; further requests are then rejected.

### Evaluation modes

| Mode | Cache behavior | OpenAI behavior | Cost evidence |
| --- | --- | --- | --- |
| `--cache-fill` | Replay exact hits; save successful misses | Call and charge on misses | Comparable only when `cache_hits=0` |
| `--fresh` | Bypass reads and writes | Call and charge every request | Comparable; use only for response-variability hypotheses |
| `--cache` | Strict exact replay | Never call OpenAI; fail on a miss | Never comparable |

Development evaluations default to cache-fill if a flag is omitted, but research commands must include the flag explicitly. Cache-fill hits work without credentials; a miss without `OPENAI_API_KEY` fails before a live call. Strict cache provides deterministic response replay, not a guarantee that arbitrary solution code has no other source of nondeterminism or network access. Development cache modes are unavailable for blind evaluations.

The evaluator passes `--seed N` to `solution.py` as `EVALUATION_SEED=N`. The reference solution includes it in every OpenAI request, so the cache stores one response per exact request and seed. Filling baseline seeds `0` through `4` lets downstream-only candidates replay the matching five response sets. Prompt, model, schema, or request changes still miss and fill normally.

Every mode counts toward the 40-evaluation limit. Charge `budget_cost_usd=0` for strict cache. For cache-fill and fresh, charge actual live spend; apply the existing incomplete-metering rule after a failure. A strict miss is a `crash` and still counts.

## Run a development evaluation

Run the evaluator in one of these modes:

```bash
# Normal development evaluation: replay hits and fill misses
set -a; source .env; set +a; test -n "$OPENAI_API_KEY" && uv run python -m src.evaluation.cli --dataset dev-19k --seed 0 --diagnostics diagnostics/001-a1b2c3d-dev-19k-seed0.json --cache-fill > run.log 2>&1

# Fresh response-variability experiment; record the reason in research.md first
set -a; source .env; set +a; test -n "$OPENAI_API_KEY" && uv run python -m src.evaluation.cli --dataset dev-19k --seed 0 --diagnostics diagnostics/002-a1b2c3d-dev-19k-seed0.json --fresh > run.log 2>&1

# Strict response replay with no OpenAI call
uv run python -m src.evaluation.cli --dataset dev-19k --seed 0 --diagnostics diagnostics/003-a1b2c3d-dev-19k-seed0.json --cache > run.log 2>&1
```

Substitute the actual evaluation number, commit, and visible dataset. `--timeout` may set a shorter deadline but cannot exceed 180 seconds.

Read the run status before interpreting metrics:

```bash
grep -E '^(result_status|score_is_final|termination_category|evaluation_mode|evaluation_seed|cache_hits|cache_misses|cache_errors|cache_writes|cache_write_errors|openai_live_requests|documents_completed|documents_failed|documents_not_attempted|cost_status|cost_is_comparable|api_cost_usd|observed_api_cost_usd|f_score|partial_f_score|precision|partial_precision|recall|partial_recall|cost_usd_per_million_source_tokens|partial_cost_usd_per_million_completed_source_tokens|document_results_json)=' run.log
```

### Complete result

When `result_status=complete` and `score_is_final=true`, use the normal score fields to evaluate the candidate. Use normalized cost for candidate comparisons only when `cost_is_comparable=true`. Complete hybrid cache-fill results may support quality conclusions, but not cost conclusions.

### Partial result

When `result_status=partial` and `score_is_final=false`:

- count the attempt toward the evaluation limit;
- use completed-document metrics and diagnostics only to form hypotheses;
- do not rank, keep, discard, or promote the candidate from this run;
- record `crash` with zero score, precision, recall, and normalized-cost fields;
- preserve coverage, the failed document ID, its source, prompt, completion and total token counts, observed cost, latency, and failure category in the diagnostic file;
- record the failure category and diagnostic path in `finding`;
- for an API-capable evaluation with `cost_status=incomplete`, charge the larger of `observed_api_cost_usd` and the pre-run estimate.

### Missing status

If `result_status` is missing, treat the attempt as an evaluator or protocol crash and inspect the final 50 lines of `run.log`. For an API-capable evaluation, charge the pre-run estimate unless a trustworthy observed subtotal is available.

## Diagnostics and failures

Before another paid run, inspect the diagnostic file from the current evaluation. Inventory every false negative and false positive by document ID, field, person index, and exact value; use source evidence to inspect the highest-impact errors. Record concrete examples and the diagnostic path in `research.md`, then name an observed error class in the next hypothesis and summarize the conclusion in `results.tsv` `finding`. Never propose or evaluate a new experiment from aggregate metrics alone.

If a run crashes, preserve and inspect any partial diagnostics. Fix and rerun an easy implementation failure; log `crash` and move on when the idea is fundamentally broken. Never assume an API-capable crash cost $0: charge the observed cost when complete, or the larger of observed cost and the pre-run estimate when metering is incomplete.

## Logging results

Use tab-separated `results.tsv`; commas in descriptions make CSV unsuitable. Start with this header:

```tsv
commit	score	precision	recall	cost	status	description	dataset	budget_cost_usd	finding	evaluation_mode	cost_is_comparable	evaluation_seed
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
9. API cost charged to the total budget: `0` for strict cache; actual live spend for cache-fill or fresh; for incomplete metering, use the larger of the observed subtotal and pre-run estimate;
10. short evidence-backed conclusion from the result, including the diagnostic path when useful. Every `keep` row on `dev-202k` must have a non-empty finding;
11. `cache-fill`, `fresh`, or `cache`;
12. `true` only when the numeric normalized cost supports comparison: fresh or zero-hit cache-fill. Otherwise `false` and record cost as `0.000000`.
13. non-negative development evaluation seed.

Record every development evaluation, including repetitions and crashes, as a separate row. Row order determines the experiment number. Record the cache mode and seed in `research.md`; for `--fresh`, explain why fresh responses are necessary. After deciding, assign the same status to successful repetitions of the same commit, dataset, and seed panel. Recompute cumulative spend after each evaluation. Report the blind result separately.

## Candidate decisions

Commit every changed candidate before evaluating it and record its seven-character commit hash.

A complete result on a smaller development dataset may justify `discard` or `inconclusive`, but never `keep`. Before marking a candidate `keep` or replacing the incumbent, evaluate that exact commit on `dev-202k` with the baseline seed panel `0` through `4`. Compare the paired per-seed results and their medians. Require five complete, final results; partial results cannot support promotion.

If cost could change a candidate decision and no comparable saved run exists, mark the cost conclusion inconclusive. Do not use `--fresh` merely to obtain cost evidence; it remains reserved for hypotheses that explicitly measure model-response variability.

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
