# Commit Discipline

- When making changes directly on `main`, stage and commit every file that belongs to the change before finishing. Do not include unrelated changes.
- When working in a separate worktree, committing is optional unless the user explicitly requests it.
- When the user asks for a correction to a previous related change, make the correction and squash it into the original commit. Do not create a series of follow-up commits for iterations on the same change.

# Worktree Workflow

- When the user requests a separate worktree, do not make implementation changes in the main workspace.
- In the main workspace, check out `main` and pull the latest remote `main` before creating the feature branch. Do not create the branch or worktree unless the pull succeeds.
- Create the feature branch from the updated `main`.
- Choose a filesystem-safe branch name and create its worktree at `/Users/maxon/Documents/code/aaMachineLearning/projects/pii-autoresearch-worktrees/<branch-name>`, using the branch name as the feature folder name.
- Perform all implementation and validation inside the feature worktree. Leave the main workspace on `main` and unchanged after the worktree is created.
