---
name: report-autoresearch-progress
description: Report the current PII autoresearch experiment progress, spending, best result, research findings, and next hypothesis. Use whenever the user asks about autoresearch progress, status, completed runs, budget usage, spending, the best experiment, metric improvements, tested hypotheses, or research learnings.
---

# Report Autoresearch Progress

Read existing experiment artifacts only. Do not run evaluations or make paid model calls.

1. Locate the active `autoresearch/*` Git worktree and read its `results.tsv`. If several exist, use the one whose `results.tsv` was modified most recently unless the user names a run.
2. Read that worktree's `program.md` for the run and cumulative budget limits, plus `research.md` for hypothesis intent and planned work. Read `baseline-results.tsv` from the repository when it contains data rows.
3. Count every data row in `results.tsv` as a completed run, including crashes, debug checks, and reruns.
4. Sum `budget_cost_usd` for budget usage. Calculate the percentage from unrounded values, then round it to the nearest whole percent.
5. Use `dev-19k` as the comparison dataset when present. Otherwise, use the most frequently evaluated non-debug dataset.
6. Group `keep` rows on that dataset by commit. Average each group's score, precision, recall, and cost, then select the group with the highest average F-score.
7. Average the saved baseline rows for the same dataset. If none exist, use the first non-crash row in the current run.
8. Calculate metric and cost deltas as best minus baseline. Use the average `cost` column, which is USD per million source tokens, for the best-result cost.
9. Build tested-hypothesis groups from result rows on the comparison dataset in first-seen order, including valid `keep` and `discard` rows. Average repeated valid rows for the same commit. Combine different commits only when result descriptions or `research.md` clearly identify them as retries of the same hypothesis. Preserve crashes as outcomes, but never treat their zero metrics as measured performance. Use result descriptions and `research.md` to identify intent; inspect the relevant Git log or diff only when those are unclear.
10. Compare each tested hypothesis with the immediately preceding distinct evaluated code state on the same dataset. Compare with the baseline only when no preceding state exists, and say so. Never infer a numeric improvement from research notes.
11. Classify a hypothesis as **Worked** when the measured result supports its intended effect without breaking a stated constraint, **Mixed** when gains have a material trade-off, **Failed** when it regresses, crashes, or violates a constraint, and **Inconclusive** when repeated evidence conflicts or no valid comparison exists.
12. Select two to five decision-relevant summary bullets. Prioritize the strongest success, meaningful trade-offs, failures, and uncertain results instead of listing every run. Add at most one **Next** bullet for the most promising explicitly documented untested hypothesis. Omit categories without evidence and never invent a hypothesis.

Return the four metric bullets first with no heading or dataset name. Bold each label, plus the best F-score and best-result cost. Then add the research-summary heading and bullets:

```markdown
- **Completed runs:** X out of Y
- **Budget:** $X.XX out of $Y.YY (P%)
- **Best result:** **X.XXXXXX F-score** (+D.XXXXXX), Y.YYY precision (+D.YYY), Z.ZZZ recall (+D.ZZZ)
- **Best result cost:** **$X.XX** (+$D.XX)

**Research summary**

- **Worked — Hypothesis:** +D.DDDDDD F-score versus the preceding evaluated state. Concise supporting evidence and any material trade-off.
- **Mixed — Hypothesis:** Measured gains and regressions, with the relevant metric deltas.
- **Failed — Hypothesis:** Measured regression, crash, or violated constraint; include cost when relevant.
- **Inconclusive — Hypothesis:** Why the available results do not support a conclusion.
- **Next — Hypothesis:** Why this documented, untested idea is the best next experiment.
```

Use only the applicable summary bullet patterns; do not emit placeholders or empty categories. Preserve six decimal places for F-score values and deltas, three decimal places for precision and recall values and deltas, and two decimal places for dollar values and cost deltas. Always show an explicit `+` for nonnegative deltas and `-` for negative deltas; place the sign before `$` for negative cost deltas.
