# Changelog

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
