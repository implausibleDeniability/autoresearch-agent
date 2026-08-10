---
name: report-autoresearch-progress
description: Report the current PII autoresearch experiment progress, spending, and best result in the required four-bullet format. Use whenever the user asks about autoresearch progress, status, completed runs, budget usage, spending, the best experiment, or metric improvements.
---

# Report Autoresearch Progress

Read existing experiment artifacts only. Do not run evaluations or make paid model calls.

1. Locate the active `autoresearch/*` Git worktree and read its `results.tsv`. If several exist, use the one whose `results.tsv` was modified most recently unless the user names a run.
2. Read that worktree's `program.md` for the run and cumulative budget limits. Read `baseline-results.tsv` from the repository when it contains data rows.
3. Count every data row in `results.tsv` as a completed run, including crashes, debug checks, and reruns.
4. Sum `budget_cost_usd` for budget usage. Calculate the percentage from unrounded values, then round it to the nearest whole percent.
5. Use `dev-19k` as the comparison dataset when present. Otherwise, use the most frequently evaluated non-debug dataset.
6. Group `keep` rows on that dataset by commit. Average each group's score, precision, recall, and cost, then select the group with the highest average F-score.
7. Average the saved baseline rows for the same dataset. If none exist, use the first non-crash row in the current run.
8. Calculate metric and cost deltas as best minus baseline. Use the average `cost` column, which is USD per million source tokens, for the best-result cost.

Return exactly these four Markdown bullets with no heading, explanation, dataset name, or trailing note. Bold each label, plus the best F-score and best-result cost:

```markdown
- **Completed runs:** X out of Y
- **Budget:** $X.XX out of $Y.YY (P%)
- **Best result:** **X.XXXXXX F-score** (+D.XXXXXX), Y.YYYYYY precision (+D.YYYYYY), Z.ZZZZZZ recall (+D.ZZZZZZ)
- **Best result cost:** **$X.XX** (+$D.XX)
```

Preserve six decimal places for metrics and metric deltas. Preserve two decimal places for dollar values and cost deltas. Always show an explicit `+` for nonnegative deltas and `-` for negative deltas; place the sign before `$` for negative cost deltas.
