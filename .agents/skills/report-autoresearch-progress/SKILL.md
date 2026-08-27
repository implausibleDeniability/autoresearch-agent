---
name: report-autoresearch-progress
description: Report current PII autoresearch progress, spending, latest results by dataset, research findings, and the next hypothesis. Use whenever the user asks about autoresearch progress, status, completed runs, budget usage, spending, experiment results, metric improvements, tested hypotheses, or research learnings.
---

# Report Autoresearch Progress

Read existing experiment artifacts only. Do not run evaluations or make paid model calls.

1. If the current `autoresearch/*` worktree contains `results.tsv`, use it. Otherwise, locate the active worktrees and use the one named by the user or whose `results.tsv` was modified most recently.
2. Read that worktree's `program.md` for the run and cumulative budget limits, plus `research.md` for hypothesis intent and planned work. Read `baseline-results.tsv` from the repository when it contains data rows. From `run.log`, extract only lines matching `^(result_status|evaluation_mode|evaluation_seed|cost_is_comparable|cache_hits|cache_misses|openai_live_requests|f_score|precision|recall|api_cost_usd|duration_seconds)=`.
3. Accept the supported nine- through thirteen-column results headers; the latter append `finding`, `evaluation_mode`, `cost_is_comparable`, and `evaluation_seed`. Treat legacy rows as live. Count every data row in `results.tsv` as a completed run, including crashes, debug checks, and reruns.
4. Sum `budget_cost_usd` for budget usage. Add `api_cost_usd` when `run.log` contains a valid blind result because the final evaluation spends budget but is not logged in `results.tsv`. Calculate the percentage from unrounded values, then round it to the nearest whole percent.
5. Build the results table in this order: `dev-19k`, `dev-87k`, `dev-202k`, then the discovered `test-*` dataset. Discover the blind name only from existing artifacts or by listing `data/test-*` directory names. Keep every row even when the dataset was not evaluated in the current run.
6. For each development dataset, scan `results.tsv` from last row to first and use the latest `keep`, `discard`, or `inconclusive` row. A crash is not a quality measurement. For repeated `dev-202k` candidates, compare medians from the same seed panel and summarize paired differences by seed; do not count duplicate cached replays as independent evidence. Show `Experiment N — description`, where `N` is the row's one-based experiment number excluding the header. Do not show its commit or status.
7. Treat `run.log` as a blind result only when it contains `f_score`, `precision`, `recall`, `api_cost_usd`, and `duration_seconds` but no `result_status`; this is the frozen evaluator's aggregate output shape. Label the experiment `Frozen final evaluation`. Never inspect blind dataset contents or seek detailed results.
8. Show development `cost` as normalized USD per million source tokens only when `cost_is_comparable=true`; otherwise show `—`. Show `budget_cost_usd` as actual run spend: zero for strict cache, paid misses for cache-fill, and all calls for fresh. Infer legacy live rows as comparable and cached rows as non-comparable. The blind evaluator does not expose normalized cost, so show `—` for it and use `api_cost_usd` as run spend. For a dataset without a completed measurement, show `Not evaluated in this run` and `—` for every metric and cost.
9. Use `dev-19k` as the comparison dataset for the research summary when present. Otherwise, use the most frequently evaluated non-debug development dataset.
10. Build tested-hypothesis groups from result rows on the comparison dataset in first-seen order, including valid `keep` and `discard` rows. Use the median for repeated valid rows from the same commit and matched seed panel. Combine different commits only when result descriptions or `research.md` clearly identify them as retries of the same hypothesis. Preserve crashes as outcomes, but never treat their zero metrics as measured performance. Use `description` for experiment intent and `finding` for the evidence-backed conclusion when present; fall back to `research.md`, then inspect the relevant Git log or diff only when those are unclear.
11. Compare each tested hypothesis with the immediately preceding distinct evaluated code state on the same dataset. Compare with the matching saved baseline only when no preceding state exists, and say so. Never infer a numeric improvement from research notes or a cost improvement from a row where `cost_is_comparable=false`.
12. Classify a hypothesis as **Worked** when the measured result supports its intended effect without breaking a stated constraint, **Mixed** when gains have a material trade-off, **Failed** when it regresses, crashes, or violates a constraint, and **Inconclusive** when repeated evidence conflicts or no valid comparison exists.
13. Select two to five decision-relevant summary bullets. Prioritize the strongest success, meaningful trade-offs, failures, and uncertain results instead of listing every run. Add at most one **Next** bullet for the most promising explicitly documented untested hypothesis. Omit categories without evidence and never invent a hypothesis.

Return the two progress bullets, the results table, then the research-summary heading and bullets. Do not add separate best-result or best-cost bullets.

```markdown
- **Completed runs:** X out of Y
- **Budget:** $X.XX out of $Y.YY (P%)

| Dataset | Mode | Experiment | F-score | Precision | Recall | Cost / 1M source tokens | Run spend |
|---|---|---|---:|---:|---:|---:|---:|
| dev-19k | cache | Experiment 12 — Hardcoded normalization | 0.940000 | 0.960 | 0.936 | — | $0.0000 |
| dev-87k | — | Not evaluated in this run | — | — | — | — | — |
| dev-202k | cache-fill | Experiment 17 — Larger-sample validation | 0.920000 | 0.950 | 0.914 | $1.39 | $0.2849 |
| test-example | fresh | Frozen final evaluation | 0.910000 | 0.940 | 0.904 | — | $0.0581 |

**Research summary**

- **Worked — Hypothesis:** +D.DDDDDD F-score versus the preceding evaluated state. Concise supporting evidence and any material trade-off.
- **Mixed — Hypothesis:** Measured gains and regressions, with the relevant metric deltas.
- **Failed — Hypothesis:** Measured regression, crash, or violated constraint; include cost when relevant.
- **Inconclusive — Hypothesis:** Why the available results do not support a conclusion.
- **Next — Hypothesis:** Why this documented, untested idea is the best next experiment.
```

The table values are examples, not defaults. Use only the applicable summary bullet patterns; do not emit placeholders or empty categories. Preserve six decimal places for F-score values and deltas, three decimal places for precision and recall values and deltas, two decimal places for normalized costs and cost deltas, and four decimal places for run spend. Always show an explicit `+` for nonnegative deltas and `-` for negative deltas; place the sign before `$` for negative cost deltas.
