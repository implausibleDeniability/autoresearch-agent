---
name: generate-autoresearch-trajectory
description: Generate a polished SVG of Dev dataset progress and the optional final Test dataset result. Use when the user asks for a research trajectory, progress chart, experiment graph, or README-ready research image.
---

# Generate Autoresearch Trajectory

Generate the image only. Never edit README or another documentation file.

1. Use a run path supplied by the user. Otherwise use the current directory when it contains `results.tsv`.
2. If the current directory has no results, inspect `git worktree list` and find autoresearch worktrees containing `results.tsv`. Use the only match. If several match, list their paths and ask the user to choose; never guess.
3. Run the generator from the selected worktree:

   ```bash
   uv run python .agents/skills/generate-autoresearch-trajectory/scripts/generate_trajectory.py
   ```

   The zero-argument command reads `results.tsv`, uses adjacent `run.log` when present, and writes `research-trajectory.svg`. Use `--results`, `--run-log`, `--no-run-log`, `--output`, or `--title` when the user specifies an override. Use `--help` for the full interface.
4. Inspect the SVG at native size and around 800 px wide. Check for clipping, ellipses, overlaps, optical misalignment, uneven row rhythm, and unbalanced margins. Correct and repeat until the image is clean; if correction requires repository edits outside the request, report the issue instead of presenting the image as ready.
5. When the host supports local image previews, show the result. Return its absolute path, the generated summary, the selected experiment list, and an explicit statement that README was not changed.

The generator is offline and standard-library-only. Do not run evaluations, access project datasets, use the network, or inspect unrecognized `run.log` content.
