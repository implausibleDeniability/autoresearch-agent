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

You conduct the research to minimize the cost of the PII-extracting system while preserving the desired precision and recall. These should be given to you when you start running the research.

Your PII-extracting system lives in the file `solution.py`. You can change the prompt, add and remove LLM steps, use hard-coded heuristics and algorithms in addition to LLM calls to reduce the cost while preserving the quality. You're not limited to just those three -- you can get creative and make other improvements as well.

The goal is to get the smallest possible cost, such that the precision and recall on the development sets are not worse than the passed thresholds.

Cost is measured in USD per million source-document tokens:

```
cost = total actual USD cost of all model calls / total tokens in the original source documents * 1,000,000
```

The denominator counts each original document once and excludes system prompts, instructions, repeated context, and generated tokens. Those tokens still affect the numerator through their actual API charges. The evaluator defines how source-document tokens are counted.

The evaluation script runs for a **fixed time budget of 5 minutes** (wall clock evaluation time, excluding startup/compilation). Use `uv run python evaluator.py --dataset debug` to debug the pipeline cheaply. Use `uv run python evaluator.py --dataset dev-5k` for routine quality decisions and `uv run python evaluator.py --dataset dev-50k` for broader validation.

The evaluator measures API usage outside `solution.py` and prints cost immediately after the run. It supports Chat Completions and Responses with the allowed models, including structured outputs, local function calling, prompt caching, retries, concurrency, and streaming. A successful API response that cannot be priced invalidates the experiment instead of counting as zero cost. Provider-hosted tools or other billable endpoints are unavailable until the evaluator has an explicit pricing rule for them.

Every individual evaluation must cost no more than $0.05. Do not start a run when its cost cannot
be bounded below that limit.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.1% cost reduction that adds 20 lines of hacky code? Probably not worth it. A 0.1% cost reduction from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.


## Restrictions
- You can NOT use models from the gpt-5 and later family, and other LLM providers, such as Google or Anthropic. Only use gpt-4o and gpt-4o-mini from OpenAI.
- You can NOT access files in `data/raw` in any way. These are archival source data: don't read them and don't write scripts, searches, or Git commands that interact with them.
- You may modify and commit only `solution.py` as the experiment implementation.
- You may create or update `results.tsv`, `REQUESTS.md`, and `run.log` only for experiment logging and communication. Do not commit these files. Do not intentionally modify any other repository files.


## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 4 columns:

```
commit	cost	status	description
```

1. git commit hash (short, 7 chars)
2. USD per million source-document tokens achieved (e.g. 10.22) — use 0.000000 for crashes
3. status: `keep`, `discard`, or `crash`
4. short text description of what this experiment tried

Example:

```tsv
commit	cost	status	description
a1b2c3d	10.220000	keep	baseline
b2c3d4e	9.930000	keep	shorten extraction prompt
c3d4e5f	10.050000	discard	add a verification call
d4e5f6g	0.000000	crash	invalid structured output schema
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
2. For the first experiment, evaluate the current `solution.py` as the baseline. For later experiments, tune `solution.py` with an experimental idea by directly hacking the code.
3. If `solution.py` changed, git commit.
4. If the change could break execution, first run `uv run python evaluator.py --dataset debug > run.log 2>&1`. Then run the routine quality evaluation with `uv run python evaluator.py --dataset dev-5k > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context). Use `dev-50k` only when its expected cost also stays within the per-run limit.
5. Read out the results: `grep -E '^(people|entity)_(precision|recall|f1)=|^(document_accuracy|api_cost_usd|cost_usd_per_million_source_tokens)=' run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. Keep the baseline. For later experiments, keep the commit only if precision and recall meet their passed thresholds and cost is lower than the incumbent. At equal cost and quality, keep the change only if the implementation is simpler.
9. If either quality threshold fails, or cost is equal or worse without a simplification win, git reset back to the incumbent.
10. Stop after 10 experiments and summarize the results for the user.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 7 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**DO NOT PAUSE BETWEEN EXPERIMENTS**: Once the experiment loop has begun (after the initial setup), do not pause to ask the human if you should continue. Do not ask "should I keep going?" or "is this a good stopping point?". Continue autonomously until 10 experiments have been attempted, the human interrupts you, or an unrecoverable failure prevents further evaluation. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, and try more radical architectural changes.
