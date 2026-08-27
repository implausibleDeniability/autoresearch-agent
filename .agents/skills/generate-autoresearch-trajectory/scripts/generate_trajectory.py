import argparse
import os
import sys
import tempfile
from pathlib import Path

from trajectory_data import TrajectoryError, load_trajectory
from trajectory_layout import format_delta, format_percentage
from trajectory_svg import render_svg

DEFAULT_TITLE = "PII extraction research trajectory"


def main(arguments: list[str] | None = None) -> int:
    parser = _argument_parser()
    options = parser.parse_args(arguments)
    results_path = Path(options.results)
    output_path = (
        Path(options.output) if options.output else results_path.with_name("research-trajectory.svg")
    )
    try:
        run_log_path = _resolve_run_log(results_path, options.run_log, options.no_run_log)
        trajectory = load_trajectory(results_path, run_log_path=run_log_path)
        svg = render_svg(trajectory, heading=options.title)
        _write_atomic(output_path, svg)
    except (TrajectoryError, OSError, UnicodeError) as error:
        print(f"trajectory: {error}", file=sys.stderr)
        return 2
    _print_summary(output_path, trajectory)
    return 0


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an autoresearch trajectory SVG.")
    parser.add_argument(
        "--results",
        default="workspace/results.tsv",
        help="experiment TSV (default: workspace/results.tsv)",
    )
    blind_input = parser.add_mutually_exclusive_group()
    blind_input.add_argument(
        "--run-log",
        default=None,
        help="Test dataset run log; omitted logs are inferred beside results.tsv when present",
    )
    blind_input.add_argument(
        "--no-run-log",
        action="store_true",
        help="ignore any run.log beside results.tsv and render development results only",
    )
    parser.add_argument("--output", default=None, help="output SVG (default: beside results.tsv)")
    parser.add_argument("--title", default=DEFAULT_TITLE, help="SVG heading")
    return parser


def _resolve_run_log(results_path: Path, requested: str | None, ignore: bool) -> Path | None:
    if ignore:
        return None
    if requested is not None:
        path = Path(requested)
        if not path.is_file():
            raise TrajectoryError(
                f"explicit run log does not exist: {path}; provide an existing file or use --no-run-log"
            )
        return path
    inferred = results_path.with_name("run.log")
    return inferred if inferred.is_file() else None


def _write_atomic(path: Path, svg: str) -> None:
    if path.suffix.lower() != ".svg":
        raise TrajectoryError(f"output must use the .svg extension: {path}")
    parent = path.parent
    if not parent.is_dir():
        raise TrajectoryError(f"output directory does not exist: {parent}")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(svg)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except OSError:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _print_summary(path: Path, trajectory) -> None:
    print(f"Generated: {path.resolve()}")
    print(f"Development experiments: {trajectory.experiment_count}")
    print(f"Dev dataset points: {len(trajectory.states)}")
    blind_status = format_percentage(trajectory.blind.score) if trajectory.blind else "not present"
    print(f"Test dataset: {blind_status}")
    print("Selected milestones:")
    if not trajectory.milestones:
        print("  none")
        return
    for state in trajectory.milestones:
        print(f"  Experiment {state.experiment} · {format_delta(state.delta or 0)} · {state.description}")


if __name__ == "__main__":
    raise SystemExit(main())
