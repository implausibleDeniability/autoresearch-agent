# PII Autoresearch

An experiment in autonomous LLM research: can an AI agent improve a system that extracts people—names, surnames, and email addresses—from documents?

The project will provide a labeled dataset, an immutable evaluator, and an agent-editable PII extraction implementation. The agent will run experiments and optimize for extraction quality and cost, while recording each hypothesis and result.

Inspired by [Karpathy's Autoresearch](https://github.com/karpathy/autoresearch), this project applies a similar autonomous experimentation loop to improving LLM prompts rather than training neural networks.

## Method

Development uses a visible development set. Final evaluation uses a separate hidden test set that the agent cannot access during research.

## Data

- `data/raw/industry-document-baseline`: the complete, unchanged source dataset.
- `data/dev`: four representative documents available during research.
- `data/test`: the remaining 17 documents reserved for final evaluation.

Only `data/dev` should be exposed to the research agent. Access control for `data/raw` and `data/test` will be enforced by the research runner.
