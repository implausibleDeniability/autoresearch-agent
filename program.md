# autoresearch-prompting

This is an experiment to have the LLM do its own research on prompting other LLMs.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. mar5). The branch autoresearch/<tag> must not already exist — this is a fresh run.
2. **Create the branch**: git checkout -b autoresearch/<tag> from current main.
3. **Initialize the logs**: Create `results.tsv` with just the header row and `research.md` with a short ranked experiment portfolio. Neither file is committed.
4. **Read the saved baseline**: Review `baseline-results.tsv` to understand ordinary baseline variation before spending the first run.
5. **Preflight the evaluator**: Before any counted run, verify the worktree, dependencies, credentials, and command are ready. The evaluator does not load `.env`, and every invocation counts as a run even when it exits before an API request.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Maximize PII extraction quality and reduce cost, targeting no more than $1.50 per million source-document tokens. Measure quality with F-score (`beta² = 5`).

Your PII-extracting system lives in the file `solution.py`. You can change the prompt, add and remove LLM steps, use hard-coded heuristics and algorithms in addition to LLM calls to reduce the cost while preserving the quality. You're not limited to just those three -- you can get creative and make other improvements as well.

F-score is the primary objective. Cost is secondary until the solution reaches the $1.50-per-million target; once it does, do not sacrifice meaningful quality for further savings.

Cost is measured in USD per million source-document tokens:

```
cost = total actual USD cost of all model calls / total tokens in the original source documents * 1,000,000
```

The denominator counts each original document once and excludes system prompts, instructions, repeated context, and generated tokens. Those tokens still affect the numerator through their actual API charges. The evaluator defines how source-document tokens are counted.

The evaluation worker has a **3-minute wall-clock limit**. Use `debug` for inexpensive pipeline checks and `dev-19k` for most experiments. `dev-87k` costs several times more, so use it occasionally and for final development validation. These three datasets are development datasets: you may inspect their diagnostics and tune against them.

The evaluator measures API usage outside `solution.py` and prints cost immediately after the run. It supports Chat Completions and Responses with the allowed models, including structured outputs, local function calling, prompt caching, retries, concurrency, and streaming. A successful API response that cannot be priced invalidates the experiment instead of counting as zero cost. Provider-hosted tools or other billable endpoints are unavailable until the evaluator has an explicit pricing rule for them.

Target no more than $1.50 per million source tokens—about 2.9 cents on `dev-19k`—and estimate cost before each run. The meter enforces only an 8-cent total limit, overridable with `--cents-limit`, so modest target overruns still return useful results.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.1% cost reduction that adds 20 lines of hacky code? Probably not worth it. A 0.1% cost reduction from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

### Quality objective

The research goal is at least 99% recall and 95% precision while meeting the cost and runtime constraints. F-score remains the primary comparison metric between experiments, but improvements that do not create a credible path toward both quality targets are not sufficient by themselves.

After the baseline, quantify the gap to the targets and the approximate reduction in errors required. Use this to judge the ambition of proposed experiments.

### Research trajectory

Periodically assess whether the current line of research can plausibly reach the quality targets within the remaining budget. Consider the size and consistency of improvements, the remaining errors, and evidence of a local ceiling.

Incremental experiments are valuable when they generate knowledge or advance a credible path. Do not continue making minor variations merely because they produce small improvements. When the observed trajectory is inadequate, reconsider the problem and broaden the search.

### Search breadth

Explore the solution space rather than only variations of the current implementation. Maintain multiple competing hypotheses, including both refinements and materially different approaches, and choose among them using expected impact, evidence, cost, runtime, and complexity.

No architecture, number of steps, balance of model and non-model methods, library, or technique is preferred in advance. The researcher owns these decisions and may invent, combine, revise, or abandon approaches as evidence develops.

If most completed experiments belong to one narrow family, explicitly justify why continuing that family has higher expected value than exploring alternatives. Before the run ends, record the major directions considered, which were tested, which were deferred, and why.

### External research

Before the first non-baseline experiment, consult relevant primary sources, official documentation, established techniques, available libraries, and prior research. Return to external research when unfamiliar failures or weak progress make it likely to improve the experiment portfolio.

Treat findings as possibilities, not prescribed solutions. Record promising findings in `research.md` with their source, relevance, uncertainty, and cost or runtime implications. If one requires an unavailable package, evaluator capability, or permission, record it in `REQUESTS.md`.

### Experiment portfolio

Maintain a short, ranked portfolio of promising research hypotheses in `research.md`. For each hypothesis, record:

- the observed problem or opportunity;
- why the hypothesis might improve the result;
- its main uncertainty;
- its estimated evaluation cost;
- what evidence would support or reject it.

Before each evaluation, compare the available hypotheses and choose the experiment with the highest expected research value, considering potential quality improvement, information gained, cost, runtime, complexity, and remaining runs.

The portfolio is a decision aid, not a fixed plan. Add, revise, combine, reorder, or discard hypotheses whenever new evidence changes their value. Do not continue an existing line merely because it is already underway.

If a lower-ranked experiment is selected, briefly record why it became the best use of the next run.

### Evaluation confidence

Model-backed evaluations may vary even when the implementation is unchanged. Treat each result as evidence, not exact truth.

`baseline-results.tsv` contains five identical baseline evaluations with metrics, error counts, costs, and durations. Use their observed range to distinguish ordinary variation from meaningful changes, then run one fresh baseline for the current research. Keep model and sampling settings consistent across comparable experiments. If a seed is used, choose it as a reproducibility setting and do not optimize it for score.

Use additional evaluations only when uncertainty could change a research decision. A clearly inferior candidate may be discarded after one run. A candidate that appears competitive with the incumbent should receive enough confirmation to determine whether the improvement is credible within the remaining run and cost budgets.

Allow experiments to be marked `inconclusive`. Do not force a keep-or-discard decision when the observed difference is too small or unstable to support one.

Record repeated evaluations as separate results. After deciding, give all successful repetitions of the same commit and dataset the same status. Base claims on their combined evidence, not the most favorable run.

Use the saved baseline results and other non-paid evidence when they can resolve uncertainty without another evaluation.

### Blind final evaluation

Every `test-*` dataset is blind. Its complete name is supplied for the final evaluation rather than hard-coded.

Do not access `data/test-*` files or detailed test results, including through code or side effects. Never pass `--diagnostics` with a blind dataset.

After development, commit the final `solution.py` and pass that commit with `--frozen-commit`. The evaluator verifies it before and after the run. The final evaluation is outside the 20-run allowance, but its spend counts toward the $0.50 budget. It returns only aggregate score, precision, recall, API cost, and duration. Success ends the run; never tune against the result, and any solution change invalidates it.


## Restrictions
- You can NOT use models from the gpt-5 and later family, and other LLM providers, such as Google or Anthropic. Only use gpt-4o and gpt-4o-mini from OpenAI.
- You can NOT execute `solution.py`, call `extract_pii` directly, run live tests, or make OpenAI API requests outside the evaluation CLI. Every paid model call must run through `uv run python -m src.evaluation.cli` so the evaluator can meter and limit its cost.
- You can NOT access files in `data/raw` in any way. These are archival source data: don't read them and don't write scripts, searches, or Git commands that interact with them.
- You can NOT access `data/test-*` except through the final evaluator invocation. Do not use commands, code, or side effects to inspect it.
- You may modify and commit only `solution.py` as the experiment implementation.
- You may create or update `results.tsv`, `research.md`, `REQUESTS.md`, `run.log`, and `diagnostics.json` only for experiment logging and communication. Do not commit these files. `diagnostics.json` contains labeled PII and source context: keep it local, overwrite it on each development run, and do not copy its contents into committed files. Never create blind-test diagnostics. Do not intentionally modify any other repository files.


## Logging results

Log development experiments to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions). Report the restricted blind result separately.

Use this 9-column header:

```
commit	score	precision	recall	cost	status	description	dataset	budget_cost_usd
```

1. git commit hash (short, 7 chars)
2. recall-weighted F-score — use 0.000000 for crashes
3. precision — use 0.000000 for crashes
4. recall — use 0.000000 for crashes
5. USD per million source-document tokens achieved — use 0.000000 for crashes
6. status: `keep`, `discard`, `inconclusive`, or `crash`
7. short text description of what this experiment tried
8. dataset: `debug`, `dev-19k`, or `dev-87k`
9. API cost charged to the budget; use the pre-run estimate if a crash hides the observed cost

Example:

```tsv
commit	score	precision	recall	cost	status	description	dataset	budget_cost_usd
a1b2c3d	0.940000	0.960000	0.936000	1.420000	keep	baseline	dev-19k	0.027467
d4e5f6g	0.000000	0.000000	0.000000	0.000000	crash	invalid structured output schema	debug	0.000500
```

## Extra communication bugs and required packages:
In the file REQUESTS.md write things that the human supervisor should know about the research.
- If you found the bug or inaccuracy in the evaluation code — write it there.
- If you figure out that there's some external package that can improve the quality significantly — write there the request to add this package. I will read it, and maybe approve and add it to the dependencies so that you could use.
- If you otherwise see a way to reduce the cost but you'll need to change the restrictions and do something that's currently not allowed — write in the document, too. I will read it, and maybe approve and change the restrictions.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5`).

Stop development after 20 evaluations or when only the budget reserved for the final test remains. Baselines, debug checks, crashes, and reruns count toward 20. The final test does not, but its spend counts toward $0.50.

1. Look at the git state: the current branch/commit we're on
2. For the first experiment, evaluate the current `solution.py` as the baseline. For later experiments, tune `solution.py` with an experimental idea by directly hacking the code.
3. If `solution.py` changed, git commit.
4. Check the run and spending limits. Estimate total and normalized cost, targeting $1.50 per million source tokens.
5. Load `.env` and run the evaluator in the same shell invocation: `set -a; source .env; set +a; test -n "$OPENAI_API_KEY" && uv run python -m src.evaluation.cli --dataset dev-19k --diagnostics diagnostics.json > run.log 2>&1`, substituting another allowed dataset when appropriate.
6. Read out the results: `grep -E '^(f_score|precision|recall|true_positive|false_positive|false_negative|api_cost_usd|cost_usd_per_million_source_tokens)=' run.log`
7. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
8. Before another paid run, inspect `diagnostics.json`. Inventory false negatives and false positives by field and document, inspect the highest-impact documents, and name at least one observed error class in the next experiment's hypothesis and `results.tsv` description. Never optimize aggregate metrics without this error inventory.
9. Record the result in `results.tsv` and recompute cumulative spend. Do not commit the file.
10. Keep the baseline. Prefer higher F-score and credible progress toward both quality targets, while using cost as the secondary objective until the solution reaches $1.50 per million source tokens. Once below the cost target, do not accept a meaningful quality regression merely to save more money. Treat runs above the target as useful evidence rather than automatic crashes.
11. If a candidate is competitive and uncertainty could change the decision, repeat it within the remaining limits and evaluate the combined evidence.
12. Mark the candidate `keep`, `discard`, or `inconclusive`. A candidate replaces the incumbent only when the evidence justifies it; otherwise return to the incumbent without losing the recorded result.
13. When development ends, summarize its results and select the final solution without blind-test evidence. Report the best `dev-87k` score separately.
14. Commit `solution.py`, confirm that `git diff --quiet HEAD -- solution.py` succeeds, and save the full `git rev-parse HEAD` output. Do not modify the solution afterward.
15. If its estimated spend fits the remaining budget, run one evaluation with the supplied blind dataset name and no diagnostics: `set -a; source .env; set +a; test -n "$OPENAI_API_KEY" && uv run python -m src.evaluation.cli --dataset 'test-<provided-name>' --frozen-commit '<full-frozen-commit>' > run.log 2>&1`.
16. Report only `f_score`, `precision`, `recall`, `api_cost_usd`, and `duration_seconds`; charge the cost and stop. Any later solution change invalidates the result.

The idea is that you are a completely autonomous researcher trying things out. Advance the branch when the evidence supports a candidate, otherwise return to the incumbent and keep exploring. If you feel like you're getting stuck, reconsider the research direction rather than repeatedly tuning the same idea.

**Timeout**: Each experiment has a 3-minute wall-clock limit from worker launch, including startup and evaluation. The evaluator terminates the worker at the limit; treat the timeout as a failure, discard the experiment, and revert it.

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**DO NOT PAUSE BETWEEN EXPERIMENTS**: Once the experiment loop has begun (after the initial setup), do not pause to ask the human if you should continue. Do not ask "should I keep going?" or "is this a good stopping point?". Continue autonomously until 20 development experiments have been attempted, only the final-test budget remains, the human interrupts you, or an unrecoverable failure prevents further evaluation. If you run out of ideas during development, think harder — read papers referenced in the code, re-read the in-scope development files for new angles, try combining previous near-misses, and try more radical architectural changes.
