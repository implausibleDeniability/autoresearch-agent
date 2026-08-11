# Changelog

## [0.4.0.0] - 2026-08-12

### Added

- Checkpoint development evaluations after every document with per-document status, token usage, observed cost, and latency.
- Report usable partial metrics over completed documents when later documents fail.

### Changed

- Preserve observed metered spend and explicitly distinguish complete from incomplete cost accounting.
- Give partial development runs distinct metric keys and exit status so automation cannot mistake diagnostics for a final score.
- Upgrade diagnostics to schema v2 with running and terminal lifecycle checkpoints.

## [0.3.0.0] - 2026-08-12

### Added

- Support dynamically named `test-*` datasets for one blind final evaluation against an explicitly frozen solution commit.

### Changed

- Limit successful blind-test output to aggregate score, precision, recall, API cost, and duration.
- Reserve the blind final evaluation outside the 20 development-run allowance while charging its spend to the monetary budget.

### Security

- Reject blind-test diagnostics and reject test evaluation unless `solution.py` matches the current supplied commit before and after the run.

## [0.2.0.0] - 2026-08-11

### Added

- Write an optional schema-versioned error inventory with raw predictions, ground truth, evaluator match decisions, per-field errors, and bounded source evidence from the same evaluation run.

### Changed

- Require autoresearch to inspect false positives and false negatives before spending another capped evaluation run.
- Validate diagnostic output before metered work and write the ignored local artifact atomically with owner-only permissions.

## [0.1.1.0] - 2026-08-10

### Added

- Stop evaluations after an 8-cent default API spending limit, with `--cents-limit` available for intentional larger runs.

### Changed

- Optimize one recall-weighted F-score (`beta² = 5`), report precision and recall, and target $1.50 per million source-document tokens.

### Removed

- Remove document accuracy and person-level quality metrics from evaluation output.

## [0.1.0.0] - 2026-08-08

### Added

- Measure OpenAI usage and report API cost immediately after each evaluation without exposing the real API key to candidate code.
- Support metered Chat Completions and Responses requests, including structured output, local function tools, streaming, retries, caching, and concurrency.
- Fail evaluations when usage, model pricing, service tier, or hosted-tool charges cannot be accounted for completely.
- Normalize cost by original source-document tokens and document the autonomous prompting experiment workflow.

### Changed

- Explain the project's relationship to Karpathy's Autoresearch and provide an editable `solution.py` candidate entry point.
