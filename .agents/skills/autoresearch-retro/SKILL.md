---
name: autoresearch-retro
description: Review a completed PII autoresearch run and identify gaps in its research design, constraints, evaluator, data, environment, and retained context. Use when the user asks for an autoresearch retrospective, lessons, limitations, or improvements that could help future runs perform better on dev-87k and unseen test data.
---

# Autoresearch Retro

Review existing artifacts only. Do not run evaluations, make paid model calls, or inspect restricted data.

Locate the requested autoresearch worktree, or the most recently active one. Read its `program.md`, the tracked files and relevant logs under `workspace/`, and Git history and diffs for those research files and `solution.py`. For a legacy run without `workspace/results.tsv`, use the corresponding root-level artifacts. For a legacy run without a portfolio file, use the portfolio recorded in `research.md`. Read `baseline-results.tsv` when useful.

Do retrospection on this research. Answer these questions sequentially internally, then give the top ten gaps or problems in the research design and suggestions for improving them:

1. What was inconvenient in the research?
2. What was missing?
3. Which information should be preserved for the next research run?
4. Was money the limiting factor?
5. Was the number of runs the limiting factor?
6. Did you try to build solutions that include a few model calls? If not, why?
7. Did you try chunking?
8. Did you try prompt tuning?
9. Which methods did you try?
10. Did you try structured-output schema tuning?
11. Did you try including reasoning in the prompt?
12. Did you check external sources about PII extraction or prompt engineering for solution ideas?
13. Did you try adding Python packages that could help?
14. Did you try extracting emails with regex parsing as a potential improvement?
15. Did you try making a list of common names and surnames and matching it against the text?

Base every answer on the artifacts. Treat unrecorded work as unknown.

Do not suggest specific research hypotheses, such as adding chunking or trying a multi-call architecture. Explain whether the research limitations, research design, evaluator interface, data, environment, retained information, or anything else should change so autoresearch can find a better solution on both `dev-87k` and the unseen test set.

Return only the ten ranked gaps or problems. For each, give concise evidence and an actionable improvement.
