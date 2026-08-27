# Autoresearch Agent

Can an AI agent improve another AI system's quality by doing the research itself? This repository tests that hypothesis on a PII extraction system that finds people's names, emails, phones in documents.

Inspired by [Karpathy's Autoresearch](https://github.com/karpathy/autoresearch), it applies an
autonomous experimentation loop to optimizing an LLM system rather than training a neural network.

![PII extraction research trajectory](research-trajectory.svg)

## How it works

The agent starts with a baseline PII extraction solution, labeled development datasets, and a fixed
evaluator. For each experiment, it:

1. Proposes a hypothesis.
2. Changes the solution to test it.
3. Evaluates the solution's quality and cost.
4. Records the result and what it learned.
5. Keeps the change when the evidence supports it; otherwise, it returns to the previous solution.

The agent repeats this process within the run and spending limits. At the end, it selects the best
solution without seeing the test data and evaluates it once on the blind final dataset. The main
metric is an F-score that weighs recall five times as heavily as precision.

**Cost tracking**: Paid model calls pass through an evaluator-owned proxy that meters spending and
enforces per-evaluation limits, while the agent tracks the cumulative run budget.

## Careful, cost-efficient research

LLM evaluations are noisy and expensive, so the agent does not treat every score change as a real
improvement. It uses document-paired linearized testing to make comparisons more sensitive while
requiring fewer model runs, then applies conservative rules to decide when to continue, stop, or
leave a result inconclusive.

## Framework self-improvement

You can improve the autoresearch framework using feedback from the research agent. The agent records
that feedback in `workspace/REQUESTS.md` as the research proceeds. You can also run the `$autoresearch-retro`
skill after a completed run to review the journal, portfolio, experiment artifacts, and their Git
history; identify what limited the agent; and recommend changes to the research design, evaluator,
data, environment, and repository. This creates a feedback loop that improves the research process.

## Quick start

You need Python 3.12, [uv](https://docs.astral.sh/uv/), Codex, and an OpenAI API key.

```bash
git clone https://github.com/implausibleDeniability/autoresearch-agent.git
cd autoresearch-agent
uv sync
```

Create a local `.env` file:

```dotenv
OPENAI_API_KEY=your-api-key
```

Open the repository in Codex and run:

```text
$run-autoresearch
```

The skill creates a separate research worktree, runs experiments within the configured limits, and
returns a final report.

To validate or run the evaluator directly:

```bash
# No credentials or API calls
uv run pii-eval --dataset dev-202k --execution-mode threaded \
  --max-concurrent-documents 150 --max-upstream-requests 150 --fresh --preflight

# Fast live evaluation; the explicit liability limit permits full parallel admission
set -a; source .env; set +a
mkdir -p diagnostics
uv run pii-eval --dataset dev-202k --seed 0 --execution-mode threaded \
  --max-concurrent-documents 150 --max-upstream-requests 150 \
  --max-inflight-liability-cents 100 --fresh --diagnostics-dir diagnostics
```

The default `isolated` mode launches one killable process per document. `threaded` imports the
solution once in one killable process, then processes documents concurrently. Use threaded mode
only when the solution and its dependencies are thread-safe. Blind evaluations use the fixed
threaded topology; their liability and upstream-concurrency flags remain enforced.
