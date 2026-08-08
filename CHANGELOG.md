# Changelog

## [0.1.0.0] - 2026-08-08

### Added

- Measure OpenAI usage and report API cost immediately after each evaluation without exposing the real API key to candidate code.
- Support metered Chat Completions and Responses requests, including structured output, local function tools, streaming, retries, caching, and concurrency.
- Fail evaluations when usage, model pricing, service tier, or hosted-tool charges cannot be accounted for completely.
- Normalize cost by original source-document tokens and document the autonomous prompting experiment workflow.

### Changed

- Explain the project's relationship to Karpathy's Autoresearch and provide an editable `solution.py` candidate entry point.
- Track quality with one recall-weighted PII F-score (`beta² = 5`).
- Enforce a $1.50 normalized cost ceiling and an evaluator-owned 8-cent default absolute run ceiling with an intentional CLI override.
