# Changelog

## [0.7.0.1] - 2026-08-22

### Fixed

- Restore the three-minute evaluation limit now that documents run concurrently.

## [0.7.0.0] - 2026-08-22

### Added

- Evaluate up to 50 documents concurrently by default, with a CLI option to change the limit.

### Changed

- Preserve exact per-document metering and partial recovery with isolated concurrent workers.
- Reuse completed source-evidence diagnostics across checkpoints.

### Fixed

- Bound concurrent spending by reserving each request's maximum cost before forwarding it.
- Preserve a terminal partial result when a document worker cannot start.

## [0.6.0.0] - 2026-08-13

### Added

- Preserve completed-document scores, token usage, cost, and latency when a later development document fails.
- Checkpoint schema-v2 diagnostics before work, after each document, and after cost finalization.

### Changed

- Mark partial scores as non-final and expose document coverage, safe failure categories, and incomplete-cost status.
- Keep blind final evaluation concurrent while development runs use document boundaries for accurate attribution.

## [0.5.0.1] - 2026-08-13

### Changed

- Let autoresearch discover the blind `test-*` dataset name while keeping its contents inaccessible without explicit permission.

## [0.5.0.0] - 2026-08-13

### Added

- Report case-insensitive, normalized, and fuzzy source evidence for diagnostic values using the evaluator's matching rules.
- Mark bounded fuzzy searches as incomplete when diagnostic safety limits stop candidate enumeration.

### Changed

- Upgrade diagnostic output to schema version 2 with evidence-kind counts, raw source spans, truncation flags, and matching-policy metadata.

### Fixed

- Keep prediction and ground-truth evidence comparisons aligned with the evaluator's direction-sensitive fuzzy matching.
- Bound fuzzy evidence work and select non-overlapping occurrences without quadratic memory or overlap scans.

## [0.3.1.0] - 2026-08-12

### Fixed

- Score a person with multiple email addresses as one person by consolidating the duplicate Justin Schwab development labels.
- Keep the documented development-set person and PII-value totals aligned with the corrected labels.

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
