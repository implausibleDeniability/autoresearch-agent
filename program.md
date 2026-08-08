# autoresearch-prompting

This is an experiment to have the LLM do its own research on prompting other LLMs.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. mar5). The branch autoresearch/<tag> must not already exist — this is a fresh run.
2. **Create the branch**: git checkout -b autoresearch/<tag> from current main.
3. **Initialize results.tsv**: Create results.tsv with just the header row. The baseline will be recorded after the first run.
4. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

You conduct the research to maximize the recall-weighted entity F-score of the PII-extracting
system, subject to a fixed cost limit of $1.50 per million original source-document tokens. Entity
recall is weighted five times as strongly as entity precision:

```
weighted entity F-score = 6 * precision * recall / (5 * precision + recall)
```

This entity score is the primary experiment objective. People-level precision, recall, and F-score
remain useful diagnostics, but do not replace the entity objective when comparing candidates.

Your PII-extracting system lives in the file `solution.py`. You can change the prompt, add and remove LLM steps, use hard-coded heuristics and algorithms in addition to LLM calls to reduce the cost while preserving the quality. You're not limited to just those three -- you can get creative and make other improvements as well.

The goal is to get the highest weighted entity F-score without exceeding the normalized cost limit.
Lower cost is useful for cautious experimentation and as a secondary consideration, but it is not a
substitute for the primary quality objective once candidates are within the fixed limit.

Cost is measured in USD per million source-document tokens:

```
cost = total actual USD cost of all model calls / total tokens in the original source documents * 1,000,000
```

The denominator counts each original document once and excludes system prompts, instructions, repeated context, and generated tokens. Those tokens still affect the numerator through their actual API charges. The evaluator defines how source-document tokens are counted.

The evaluation script runs for a **fixed time budget of 5 minutes** (wall clock evaluation time, excluding startup/compilation). Use `uv run python -m src.evaluation.cli --dataset debug` to debug the pipeline cheaply. Use `uv run python -m src.evaluation.cli --dataset dev-5k` for routine quality decisions and `uv run python -m src.evaluation.cli --dataset dev-50k` for broader validation.

The evaluator measures API usage outside `solution.py` and prints cost immediately after the run. It supports Chat Completions and Responses with the allowed models, including structured outputs, local function calling, prompt caching, retries, concurrency, and streaming. A successful API response that cannot be priced invalidates the experiment instead of counting as zero cost. Provider-hosted tools or other billable endpoints are unavailable until the evaluator has an explicit pricing rule for them.

Before every paid evaluation, estimate both:

1. total API cost in USD, using the expected model calls, input and output tokens, cache behavior,
   retries, and evaluator-owned prices; and
2. normalized cost in USD per million original source-document tokens, using the evaluator's source
   token count and the cost formula in this section.

Do not start an experiment whose estimated normalized cost exceeds $1.50 per million original
source-document tokens. Use conservative estimates and reduce the run or candidate cost when the
bound is uncertain.

Every evaluator run also has a default absolute spending guard of $0.08. A larger intentional run
requires an explicit `--cents-limit <cents>` override. Use the smallest justified override only after
confirming that the estimated normalized cost remains at or below $1.50. A larger dataset can have
an acceptable normalized cost while exceeding $0.08 in total, so broader validation may require an
override even when the candidate satisfies the research cost limit.

**Simplicity criterion**: All else being equal, simpler is better. A tiny weighted entity F-score
improvement that adds ugly complexity may not be worth it. Conversely, removing something and
getting equal or better quality within the cost limit is a great outcome — that's a simplification
win. When evaluating whether to keep a change, weigh the complexity cost against the improvement
magnitude. A change of about zero that makes the implementation much simpler? Keep. Never trade
away the fixed cost limit for quality.


## Restrictions
- You can NOT use models from the gpt-5 and later family, and other LLM providers, such as Google or Anthropic. Only use gpt-4o and gpt-4o-mini from OpenAI.
- You can NOT access files in `data/raw` in any way. These are archival source data: don't read them and don't write scripts, searches, or Git commands that interact with them.
- You may modify and commit only `solution.py` as the experiment implementation.
- You may create or update `results.tsv`, `REQUESTS.md`, and `run.log` only for experiment logging and communication. Do not commit these files. Do not intentionally modify any other repository files.


## Logging results

Log every considered experiment to `results.tsv` (tab-separated, NOT comma-separated — commas
break in descriptions). Record both cost estimates before a paid run, then complete the row with
actual results afterward. Also log a candidate that is not run because its estimate is over the
normalized limit.

The TSV has a header row and 13 columns:

```
commit	entity_f_score	entity_precision	entity_recall	people_precision	people_recall	people_f_score	estimated_api_cost_usd	estimated_cost_usd_per_million	api_cost_usd	cost_usd_per_million	status	description
```

1. git commit hash (short, 7 chars)
2. weighted entity F-score, the primary objective — use 0.000000 for crashes
3. entity precision
4. entity recall
5. people precision, as a diagnostic
6. people recall, as a diagnostic
7. people F-score, as a diagnostic
8. estimated total API cost in USD recorded before the run
9. estimated USD per million original source-document tokens recorded before the run
10. actual total API cost in USD — use 0.000000 when no paid request completed
11. actual USD per million original source-document tokens — use 0.000000 for crashes
12. status: `keep`, `discard`, `crash`, or `over-limit`
13. short text description of what this experiment tried

Example:

```tsv
commit	entity_f_score	entity_precision	entity_recall	people_precision	people_recall	people_f_score	estimated_api_cost_usd	estimated_cost_usd_per_million	api_cost_usd	cost_usd_per_million	status	description
a1b2c3d	0.840000	0.700000	0.875000	0.780000	0.820000	0.813043	0.006000	1.350000	0.005900	1.330000	keep	baseline
b2c3d4e	0.860000	0.710000	0.900000	0.790000	0.830000	0.823333	0.006200	1.420000	0.006100	1.398762	keep	shorten extraction prompt
c3d4e5f	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.007000	1.600000	0.000000	0.000000	over-limit	verification estimate exceeded normalized limit; not run
d4e5f6g	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.006000	1.380000	0.000000	0.000000	crash	invalid structured output schema
```

## Extra communication bugs and required packages:
In the file REQUESTS.md write things that the human supervisor should know about the research.
- If you found the bug or inaccuracy in the evaluation code — write it there.
- If you figure out that there's some external package that can improve the quality significantly — write there the request to add this package. I will read it, and maybe approve and add it to the dependencies so that you could use.
- If you otherwise see a way to reduce the cost but you'll need to change the restrictions and do something that's currently not allowed — write in the document, too. I will read it, and maybe approve and change the restrictions.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5`).

Run at most 10 experiments, counting the baseline, crashes, and reruns.

1. Look at the git state: the current branch/commit we're on
2. Before every paid run, estimate and record total and normalized cost. For the first experiment,
   evaluate the current `solution.py` as the baseline only if the normalized estimate complies and
   the total estimate fits either the default guard or a justified override. If its normalized
   estimate exceeds $1.50, do not run it; first reduce the candidate's expected cost. For later
   experiments, tune `solution.py` with an experimental idea by directly hacking the code.
3. If `solution.py` changed, git commit.
4. If the change could break execution, first run
   `uv run python -m src.evaluation.cli --dataset debug > run.log 2>&1`. Then run the routine quality
   evaluation with
   `uv run python -m src.evaluation.cli --dataset dev-5k > run.log 2>&1` (redirect everything — do
   NOT use tee or let output flood your context). Use `dev-50k` only when its normalized estimate
   stays within the fixed limit. If an estimated total cost exceeds the default $0.08 guard, append
   an intentional `--cents-limit <cents>` override after recording both estimates; a normalized
   estimate over $1.50 cannot be overridden.
5. Read out the results: `grep -E '^(objective_score|people_(precision|recall|f_score)|entity_(precision|recall|f_score)|document_accuracy|api_cost_usd|cost_usd_per_million_source_tokens)=' run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. A run whose actual normalized cost exceeds $1.50 is ineligible and must never become the
   incumbent. Among eligible runs, compare experiments primarily by weighted entity F-score. Keep a
   candidate that materially improves that score. For effectively equal scores, prefer the simpler
   implementation; if complexity is also equal, prefer lower normalized cost. Use people-level
   metrics to diagnose behavior, not to override the entity objective.
9. If a candidate is over the cost limit or does not beat the incumbent under those selection rules,
   git reset back to the incumbent.
10. Stop after 10 experiments and summarize the results for the user.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 7 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**DO NOT PAUSE BETWEEN EXPERIMENTS**: Once the experiment loop has begun (after the initial setup), do not pause to ask the human if you should continue. Do not ask "should I keep going?" or "is this a good stopping point?". Continue autonomously until 10 experiments have been attempted, the human interrupts you, or an unrecoverable failure prevents further evaluation. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, and try more radical architectural changes.
