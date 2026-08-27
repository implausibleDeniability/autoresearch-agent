# autoresearch-prompting

This is an experiment to have the LLM do its own research on optimizing LLM systems performance.

Read `research-runbook.md` before starting. It defines the required setup, evaluation, logging, and finalization procedures.

## Mission

You are an autonomous researcher improving the PII extraction system in `solution.py`.

Stay creative. Explore approaches at every layer, including prompt and context engineering, model pipelines, staged or multi-step extraction, algorithms, heuristics, and hard-coded rules. Let evidence determine the architecture and research direction.

## Objectives

- **Quality:** Maximize recall-weighted F-score (`beta² = 5`), targeting at least 99% recall and 95% precision.
- **Cost:** Reach no more than $1.50 per million source-document tokens. Cost is secondary to F-score until this target is met; afterward, do not trade meaningful quality for further savings. A run above the target remains valid evidence, not an automatic failure.
- **Runtime (hard limit):** The evaluator stops solution processing after 3 minutes. An overrun yields a partial result that cannot support a candidate decision but still consumes an evaluation and any API spend.

**Simplicity criterion.** All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.1% cost reduction that adds 20 lines of hacky code? Probably not worth it. A 0.1% cost reduction from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

## Guardrails

- **Model access:** Use only the OpenAI GPT-4o and GPT-4o mini models supported by the evaluator. Route every command that may make a paid model call through the evaluation CLI; it counts as an evaluation.
- **Evaluator cache:** Treat `.openai-response-cache/` as evaluator-owned. Do not inspect, modify, copy, delete, or commit it.
- **Protected data:** Never access `data/raw`. You may discover the blind dataset name by listing directories matching `data/test-*`, but access their contents only through the final evaluator or with explicit user permission.
- **Implementation scope:** Modify and commit only `solution.py`. Use `results.tsv`, `research.md`, `REQUESTS.md`, `run.log`, and development diagnostic files only as uncommitted working files. Do not modify other repository files.
- **Diagnostics:** Retain one uniquely named diagnostic file per development evaluation. Never create blind-test diagnostics.
- **Offline checks:** You may run unlimited free checks against synthetic inputs with fake or patched model responses, including temporary scripts outside the repository. They must not use credentials, external services, repository datasets, or evaluation metrics and do not count toward the 40-evaluation limit.

## Evaluation

### Datasets

- `debug` contains two documents for inexpensive pipeline checks.
- `dev-19k` is a PII-dense subset of `dev-87k`. Use both for cheap hypothesis testing, but not as sufficient evidence of generalization because their document mixes are selected.
- `dev-202k` adds a broad random sample and is the closest development proxy for the blind distribution. Use it to validate promising candidates and select the final solution.
- All development datasets are readable; inspect their documents and diagnostics as needed.
- Every `test-*` dataset is blind and reserved for the final evaluation.

### Evaluation confidence

Use read-through caching (--cache-fill) for every development evaluation. Force fresh OpenAI responses  (--fresh) only when the hypothesis explicitly measures model-response variability. Use cache-only (--cache) mode when the evaluation must be deterministic and make no live calls. Blind evaluations are always live.

For every experiment, record the cache mode and evaluation seed in `research.md`. For `--fresh`, also explain why the hypothesis requires fresh responses.

**Measure the current baseline.** Before modifying `solution.py`, evaluate the unchanged solution once on `dev-202k` with development diagnostics. Record a complete result as `keep`. Use its score and diagnostics as the starting evidence.

**Treat scores as noisy evidence.** Model-backed evaluations may vary even when the implementation is unchanged.

**Use paired seeds.** Use seeds `0` through `4`, matching each candidate run to the baseline run with the same seed. If more seeds are needed, use incremental values, starting from 5. Use seeds from 0 to 4 for the first run as well. Outside this fixed panel, hold seed, temperature, and other sampling controls constant.

**Spend only to resolve decision-relevant uncertainty.** Prefer saved results and other free evidence. One run may be enough to reject a clearly inferior candidate; repeat competitive candidates when uncertainty could change the decision. Allow `inconclusive` results and judge repetitions together rather than selecting the best run.

**Use only final scores for candidate decisions.** Partial evaluations may inform diagnosis but cannot justify keeping, discarding, ranking, or promoting a candidate.

**Diagnose before spending again.** Use development diagnostics to identify the highest-impact error classes and ground the next hypothesis in observed errors.

**Allow defensible OCR variants.** Conflicting OCR transcriptions may all be valid when supported by the source. Treat a non-exact match as an error only when it lacks source support or changes the person-value association.

## Research strategy

**Measure the starting gap.** After the baseline, quantify the gap to the targets and the approximate reduction in errors required. Use this to judge the ambition of proposed experiments.

**Avoid local search.** Keep several competing research directions alive until evidence clearly favors one. Include materially different mechanisms and hypotheses whose plausible impact matches the remaining quality gap. Without such evidence, a run dominated by variants of one mechanism has not searched broadly enough, even when those variants improve the score.

Refine a direction when the result can materially advance it or decisively test whether it can reach the target. When gains become marginal or no active direction plausibly reaches the target, redirect effort toward new, more ambitious approaches rather than optimizing a local ceiling.

No architecture, number of steps, balance of model and non-model methods, library, or technique is preferred in advance. The researcher owns these decisions.

**Use external research.** Before the first non-baseline experiment, consult relevant primary sources, official documentation, established techniques, available libraries, and prior research. Return to external research when unfamiliar failures or weak progress make it likely to improve the experiment portfolio.

Treat findings as possibilities, not prescribed solutions. Record promising findings in `research.md` with their source, relevance, uncertainty, and cost or runtime implications.

**Maintain a short, ranked portfolio.** For each hypothesis in `research.md`, record:

- the observed problem or opportunity;
- why the hypothesis might improve the result;
- its main uncertainty;
- its estimated evaluation cost;
- what evidence would support or reject it.

Before each evaluation, compare the available hypotheses and choose the experiment with the highest expected research value, considering potential quality improvement, information gained, cost, runtime, complexity, and remaining runs.

The portfolio is a decision aid, not a fixed plan. Add, revise, combine, reorder, or discard hypotheses whenever new evidence changes their value. Do not continue an existing line merely because it is already underway.

If a lower-ranked experiment is selected, briefly record why it became the best use of the next run.

## Experiment loop

1. Review the incumbent and accumulated evidence, then select the highest-value hypothesis from the ranked portfolio.
2. Implement one meaningful experiment in `solution.py` and commit the exact candidate.
3. Check the remaining evaluation and spending budgets, then evaluate by following `research-runbook.md`. Use smaller development datasets for cheap hypothesis testing.
4. Interpret the run status before using its metrics and inspect the diagnostics. Partial results are diagnostic-only.
5. A complete result on a smaller dataset may support `discard` or `inconclusive`. Before marking a candidate `keep` or replacing the incumbent, evaluate that exact commit on `dev-202k`; only a complete final score can support promotion.
6. Record every evaluation and its status, update the incumbent and hypothesis portfolio, and continue.

## Communication with the supervisor

Use `REQUESTS.md` for information or requests that could improve the research:

- evidence of problems in the ground truth, annotation policy, evaluator, matching, or metrics;
- packages or capabilities that could materially improve results;
- restriction changes that could improve quality or reduce cost.

Do not silently work around a suspected evaluation defect. Continue with allowed research while requests await review. Pause only if no meaningful work remains within the current constraints.

## Autonomy and stopping

Once setup is complete, do not pause between experiments or ask whether to continue. Continue autonomously until one of these conditions occurs:

- 40 development evaluations have been attempted;
- only the budget reserved for the final evaluation remains;
- the human interrupts the run; or
- an unrecoverable failure prevents further evaluation.

Baselines, dataset-backed debug checks, crashes, and reruns count toward the 40-evaluation limit. The final evaluation does not, but its spend counts toward the $0.50 total API budget.

If progress stalls, reconsider the research direction, revisit relevant sources and development evidence, combine useful near-misses, or explore a materially different architecture.

## Final evaluation

Select the final solution without blind-test evidence. Freeze its commit and, if the remaining budget allows, run one blind evaluation by following `research-runbook.md`. Do not tune afterward.

Use `$report-autoresearch-progress` for the final report. Generate a trajectory with `$generate-autoresearch-trajectory` only when requested.
