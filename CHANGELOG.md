# Changelog

## [0.11.0.0] - 2026-08-23

### Added

- Add a worktree-local exact-response replay cache for repeatable development evaluations.
- Report evaluation mode and cache activity, and preserve mode in experiment logs.

### Changed

- Allow development evaluations to use strict cache-only replay without API credentials or network fallback.
- Reject cache access for blind, dataset-description, and worker modes.

## [0.9.0.0] - 2026-08-22

### Added

- Mark names recoverable only from an email local-part as optional ground-truth values.
- Expose matched and unmatched optional values in a neutral diagnostics ledger.

### Changed

- Upgrade diagnostics to schema version 4 and scored runs to evaluator contract version 2.
- Treat historical contract-1 baseline scores as incomparable with new results.

### Fixed

- Stop counting omitted optional names as false negatives without hiding incorrect name predictions.
- Exclude optional names from person matching and match them only by normalized exact values or
  explicit variants.

## [0.8.0.0] - 2026-08-22

### Added

- Represent OCR transcriptions and supported aliases as one canonical ground-truth value with
  accepted variants.

### Changed

- Preserve accepted label variants and the source-visible matched form in schema-v3 diagnostics.

### Fixed

- Add omitted partial names, person-associated phone numbers, emails, and locations to development
  labels.
- Remove test labels absent from the evaluated text and align compound-surname and alias labels
  across splits.

## [0.7.0.1] - 2026-08-22

### Fixed

- Remove the former ten-minute override; `--timeout` now accepts at most 180 seconds.

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
