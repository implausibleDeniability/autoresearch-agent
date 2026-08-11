# Commit Discipline

- Always squash pull requests when merging them into `main`.
- When making changes directly on `main`, stage and commit every file that belongs to the change before finishing. Do not include unrelated changes.
- When working in a separate worktree, committing is optional unless the user explicitly requests it.
- When the user asks for a correction to a previous related change, make the correction and squash it into the original commit. Do not create a series of follow-up commits for iterations on the same change.

# Concise Writing

- Apply [$write-concisely](/Users/maxon/.codex/skills/write-concisely/SKILL.md) when modifying `AGENTS.md`, `README.md`, `program.md`, skills, or other reader-facing ground-truth or critical files. This rule does not apply to code.
- Apply the skill only to text you add or modify. Do not rewrite existing text solely to make it more concise.

# Cost Discipline

- Both code-development agents and autoresearch agents must be careful about spending money.
- Prefer cheaper models when they can complete the task effectively.
- If an execution could cost $0.10 or more, first try to reduce its cost.
- If an execution could cost $1.00 or more, obtain explicit human approval before running it.

# CPU Discipline

- Never run more than two CPU-heavy processes concurrently on this machine, including processes started by subagents.
- Before starting CPU-heavy work, count active local and subagent processes. Queue or stop work to keep the total at two or fewer.

# Worktree Workflow

- When the user requests a separate worktree, do not make implementation changes in the main workspace.
- In the main workspace, check out `main` and pull the latest remote `main` before creating the feature branch. Do not create the branch or worktree unless the pull succeeds.
- Create the feature branch from the updated `main`.
- Choose a filesystem-safe branch name and create its worktree at `/Users/maxon/Documents/code/aaMachineLearning/projects/pii-autoresearch-worktrees/<branch-name>`, using the branch name as the feature folder name.
- Before beginning work in the new worktree, identify development-critical files that are intentionally excluded from Git, such as `.env` and other local configuration or credential files required by the project. Copy each required file from the main workspace into the equivalent location in the new worktree, preserve restrictive permissions, and confirm that it remains ignored by Git. Do not copy unrelated ignored files or commit secrets.
- Perform all implementation and validation inside the feature worktree. Leave the main workspace on `main` and unchanged after the worktree is created.
