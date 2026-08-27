# Research runbook

This is the technical companion to `program.md`. Follow it for setup, evaluation, logging, candidate management, and finalization. `program.md` remains authoritative for research goals, decisions, and guardrails.

## Setup

1. Read the repository and user instructions for workspace location and naming.
2. In the primary repository, check out `main` and pull the latest remote `main`. Stop if the pull fails.
3. Create `autoresearch/<tag>` from updated `main`, using a filesystem-safe day-and-time tag such as `aug-8-11-52`. When the instructions require a worktree, create and enter it with `git worktree add -b autoresearch/<tag> <path> main`. If workspace placement is unspecified, ask before choosing it.
4. For a new worktree, copy only required ignored local files such as `.env`, preserve their permissions, and confirm they remain ignored. If the primary repository contains `.cache-local/`, copy it without inspection to the new worktree as `.openai-response-cache/`, preserve its permissions, and confirm both paths remain ignored.
5. Create `results.tsv` with only the header defined below, `research.md` with a short ranked experiment portfolio, and the ignored `diagnostics/` directory. Do not commit these working files.
6. Review `baseline-results.tsv` to understand ordinary baseline variation, cost, and runtime before spending the first evaluation.
7. Inspect dataset scale with `uv run python -m src.evaluation.cli --dataset dev-87k --describe-dataset`. This command is read-only, requires no API credentials, and does not count as an evaluation.
8. Discover the blind dataset name by listing only directory names matching `data/test-*`. Do not inspect their contents.

## Working files

| File | Purpose |
| --- | --- |
| `results.tsv` | Development evaluation results and cumulative spend |
| `research.md` | Ranked hypothesis portfolio and external research findings |
| `REQUESTS.md` | Information and requests for the human supervisor |
| `run.log` | Output from the latest evaluation |
| `diagnostics/<bank>/<evaluation>-<commit>-<dataset>-seed<seed>.json` | Rich development diagnostics |
| `diagnostics/<bank>/<evaluation>-<commit>-<dataset>-seed<seed>.evidence.json` | PII-free comparison evidence written automatically |

Keep each exact incumbent or candidate panel in its own directory. Use a new diagnostic path for every development evaluation; the evaluator derives and preflights its `.evidence.json` sidecar. Never request diagnostics for a blind dataset.

## Development evaluation protocol

### Evaluation modes

| Mode | Cache behavior | OpenAI behavior | Cost evidence |
| --- | --- | --- | --- |
| `--cache-fill` | Replay exact hits; save successful misses | Call and charge on misses | Comparable only when `cache_hits=0` |
| `--fresh` | Bypass reads and writes | Call and charge every request | Comparable; use only for response-variability hypotheses |
| `--cache` | Strict exact replay | Never call OpenAI; fail on a miss | Never comparable |

Development evaluations default to cache-fill if a flag is omitted, but research commands must include the flag explicitly. Cache-fill hits work without credentials; a miss without `OPENAI_API_KEY` fails before a live call. Strict cache provides deterministic response replay, not a guarantee that arbitrary solution code has no other source of nondeterminism or network access. Development cache modes are unavailable for blind evaluations.

For `--fresh`, record in `research.md` why the hypothesis requires fresh responses.

### Paired seed panels

The evaluator passes `--seed N` to `solution.py` as `EVALUATION_SEED=N`. The reference solution includes it in every OpenAI request, so the cache stores one response per exact request and seed. Prompt, model, schema, or request changes miss and fill normally.

Use seeds `0` through `2` for the incumbent control bank. A response-changing candidate starts at seed `0`; run seed `1` and then seed `2` only when `pii-compare` recommends the next fixed seed. The comparator always matches the candidate prefix to the same incumbent seeds while requiring the complete three-seed control bank. Never pool different candidate commits or add targeted repeats.

For a change strictly downstream of model responses, `--change-type fixed-replay` permits a formal look after seed `0`. When distinct cached banks already exist, it also accepts the fixed prefixes `0,1` and `0,1,2`. Every paired seed must use `--cache`, replay every request, and have identical document-scoped request and response receipts across arms. Banks must be distinct across seeds; rerunning one bank adds no evidence. A caller's classification alone cannot activate this exception, and do not make paid calls solely to create more replay banks.

### Cost accounting and evaluator behavior

Cost is measured in USD per million source-document tokens:

```text
cost = total actual USD cost of all model calls / total tokens in the original source documents * 1,000,000
```

The denominator counts each original document once and excludes system prompts, instructions, repeated context, and generated tokens. Those tokens still contribute to actual API charges in the numerator. The evaluator defines how source-document tokens are counted and prints cost immediately after each run.

The evaluator supports Chat Completions and Responses with the allowed models, including structured outputs, local function calling, prompt caching, retries, concurrency, and streaming. Successful responses that cannot be priced make cost accounting incomplete rather than counting as free. Provider-hosted tools and other billable endpoints are unavailable without an evaluator pricing rule.

The default 8-cent meter is a guard rather than the normalized-cost target. Override the default limit with `--cents-limit` only when the estimated live run fits the total research budget. A request may exceed the remaining per-run limit so the evaluator can return useful results; further requests are then rejected.

Every mode counts toward the 40-evaluation limit. Charge `budget_cost_usd=0` for strict cache. For cache-fill and fresh, charge actual live spend; apply the existing incomplete-metering rule after a failure. A strict miss is a `crash` and still counts.

### Before each run

1. Confirm that another evaluation fits within the 40-evaluation limit. For `--cache-fill` or `--fresh`, also confirm that possible live requests fit within the remaining $0.50 API budget while preserving enough budget for the final evaluation.
2. Before a mode that can call OpenAI, estimate total API spend. Estimate normalized cost when the result can be comparable. The target is at most $1.50 per million source-document tokens.
3. Choose the dataset and concurrency appropriate for the hypothesis. The evaluator processes up to 50 documents concurrently by default; use `--max-concurrent-documents N` to reduce simultaneous worker and API load.
4. Select the evaluation mode and seed under the protocol above.

### Commands

Run the evaluator in one of these modes:

```bash
# Normal development evaluation: replay hits and fill misses
set -a; source .env; set +a; uv run python -m src.evaluation.cli --dataset dev-19k --seed 0 --diagnostics diagnostics/candidate/001-a1b2c3d-dev-19k-seed0.json --cache-fill > run.log 2>&1

# Fresh response-variability experiment; record the reason in research.md first
set -a; source .env; set +a; test -n "$OPENAI_API_KEY" && uv run python -m src.evaluation.cli --dataset dev-19k --seed 0 --diagnostics diagnostics/candidate/002-a1b2c3d-dev-19k-seed0.json --fresh > run.log 2>&1

# Strict response replay with no OpenAI call
uv run python -m src.evaluation.cli --dataset dev-19k --seed 0 --diagnostics diagnostics/candidate/003-a1b2c3d-dev-19k-seed0.json --cache > run.log 2>&1
```

Substitute the actual evaluation number, commit, and visible dataset. `--timeout` may set a shorter deadline but cannot exceed 180 seconds.

### Result interpretation

Read the run status before interpreting metrics:

```bash
grep -E '^(result_status|score_is_final|termination_category|evaluation_mode|evaluation_seed|cache_hits|cache_misses|cache_errors|cache_writes|cache_write_errors|openai_live_requests|documents_completed|documents_failed|documents_not_attempted|cost_status|cost_is_comparable|api_cost_usd|observed_api_cost_usd|f_score|partial_f_score|precision|partial_precision|recall|partial_recall|cost_usd_per_million_source_tokens|partial_cost_usd_per_million_completed_source_tokens|document_results_json)=' run.log
```

#### Complete result

When `result_status=complete` and `score_is_final=true`, use the normal score fields to evaluate the candidate. Use normalized cost for candidate comparisons only when `cost_is_comparable=true`. Complete hybrid cache-fill results may support quality conclusions, but not cost conclusions.

#### Partial result

When `result_status=partial` and `score_is_final=false`:

- count the attempt toward the evaluation limit;
- use completed-document metrics and diagnostics only to form hypotheses;
- do not rank, keep, discard, or promote the candidate from this run;
- record `crash` with zero score, precision, recall, and normalized-cost fields;
- preserve coverage, the failed document ID, its source, prompt, completion and total token counts, observed cost, latency, and failure category in the diagnostic file;
- record the failure category and diagnostic path in `finding`;
- for an API-capable evaluation with `cost_status=incomplete`, charge the larger of `observed_api_cost_usd` and the pre-run estimate.

#### Missing status

If `result_status` is missing, treat the attempt as an evaluator or protocol crash and inspect the final 50 lines of `run.log`. For an API-capable evaluation, charge the pre-run estimate unless a trustworthy observed subtotal is available.

## Initial baseline

Before modifying `solution.py`, evaluate the unchanged solution on `dev-202k` with seeds `0` through `2`. Follow the development evaluation protocol and store the three runs in one incumbent bank directory. Require three complete, final results and record them as the current baseline `keep` panel.

Use `--cache-fill` for this panel. Restored hits are valid baseline evidence; only misses require live requests. Do not use `--fresh` merely to refresh a compatible cached baseline.

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

Record every development evaluation, including repetitions and crashes, as a separate row. Row order determines the experiment number. After deciding, assign the same status to successful repetitions of the same commit, dataset, and seed panel. Recompute cumulative spend after each evaluation. Report the blind result separately.

## Paired comparison decisions

Commit every changed candidate before evaluating it. Comparison evidence binds every run to the full commit, immutable solution snapshot, scoring contract, dataset, and response bank; filenames are not provenance.

A complete result on a smaller development dataset may justify `discard` or `inconclusive`, but never `keep`. Start a response-changing `dev-202k` candidate at seed `0`, then run:

```bash
uv run pii-compare \
  --incumbent diagnostics/incumbent-bank \
  --candidate diagnostics/candidate-bank
```

JSON is the authoritative agent interface. Exit `0` means valid evidence and any scientific outcome; `2` means invalid evidence or usage; `1` means an internal failure. `run_again` names the next fixed seed. `stop_for_futility` is an early heuristic, not a formal rejection. At seed three, promotion requires one-sided `p < 0.05`; a gain of at least 1 percentage point is promotion-eligible, while a gain from 0.3 to 1 point requires mechanistic review. Formal rejection occurs only when the one-sided 80% upper confidence bound is below +0.3 points. Otherwise postpone the result as inconclusive.

For a strict downstream replay, compare the same fixed seed prefix in each arm with `--change-type fixed-replay`. One paired bank is sufficient; use additional distinct cached banks when available. If receipt identity or provenance fails, treat the result as response-changing or regenerate exact evidence; never override the gate.

### Migration and recovery

Diagnostics created before comparison-evidence schema v1 are unsupported because they lack immutable provenance and response receipts. Do not backfill or infer those fields from filenames. Generate a new incumbent panel and evaluate candidates under the same evaluator, dataset, runtime, and scoring contract.

On the first qualifying real three-seed bank, record both the legacy score summary and the comparator decision in `research.md` before making that bank the control. This shadow record checks the operational migration without weakening the comparator's locked decision. It is required once; it does not authorize extra targeted runs.

Common failures are recoverable without changing thresholds:

| Error code or outcome | Recovery |
| --- | --- |
| `invalid_usage` | Run `pii-compare --help`; provide exactly one directory or exact-file form per arm. |
| `unreadable_evidence` | Use regular, complete `.evidence.json` files; reject symlinks, FIFOs, malformed JSON, and oversized files. |
| `mixed_commits` or `mixed_solution_snapshots` | Split banks by exact candidate commit. |
| `mixed_scoring_contract`, `mixed_dataset`, or `mixed_runtime` | Re-evaluate both arms under one environment. |
| `fixed_replay_mismatch` | Treat the change as response-changing or regenerate a strict identical replay. |
| `stop_for_futility` | Restore the incumbent; this is an early heuristic stop, not a formal rejection. |
| `inconclusive` | Preserve and postpone the candidate; do not add targeted repeats. |
| `internal_error` | Rerun with `--debug`, preserve the evidence, and report the local traceback. |

**Accept:** Promote the exact candidate commit. All candidate runs become its initial response bank. Before using it as the control for another AI-changing hypothesis, obtain at least three complete runs total. Three baseline runs cost only about a 5% sensitivity loss versus five.

**Reject:** Restore the incumbent. Preserve the result and diagnostics, but do not combine its runs with later candidates.

**Inconclusive/postponed:** Preserve the commit and evidence, but do not promote it. Revisit only when:

- a related hypothesis provides stronger mechanistic evidence;
- several compatible changes can produce a larger expected effect;
- evaluation variance has fallen;
- or the change has separate strategic value.

Any modification to the candidate between repetitions creates a **new hypothesis**. Its results cannot be pooled with the previous version.

If cost could change a candidate decision and no comparable saved run exists, leave the cost conclusion inconclusive. Missing cost evidence does not justify a fresh run.

When a candidate does not replace the incumbent, return to the incumbent without deleting its recorded evaluations. The comparator never modifies Git state or launches another evaluation.

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
