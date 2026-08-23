from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from trajectory_data import Experiment, IncumbentState, Trajectory, validate_xml_text
from trajectory_labels import draw_labels, separate_annotations
from trajectory_layout import format_delta, format_percentage

DISCARDED_COLOR = "#cccccc"
KEPT_COLOR = "#2ecc71"
RUNNING_BEST_COLOR = "#27ae60"
TEST_COLOR = "#e67e22"


def save_plot(path: Path, *, trajectory: Trajectory, heading: str) -> None:
    figure = create_figure(trajectory, heading=heading)
    metadata = {"Title": heading, "Description": accessible_description(trajectory)}
    try:
        figure.savefig(path, dpi=150, bbox_inches="tight", metadata=metadata)
    finally:
        figure.clear()


def create_figure(trajectory: Trajectory, *, heading: str) -> Figure:
    validate_xml_text(heading, context="title")
    figure = Figure(figsize=(16, 8))
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    visible_results = _visible_results(trajectory)
    _draw_discarded(axes, visible_results)
    _draw_kept(axes, trajectory.states)
    _draw_running_best(axes, trajectory)
    annotations = draw_labels(axes, trajectory.states)
    _draw_test_result(axes, trajectory)
    _style_axes(axes, trajectory=trajectory, heading=heading)
    figure.tight_layout()
    separate_annotations(figure, axes=axes, annotations=annotations)
    return figure


def _visible_results(trajectory: Trajectory) -> tuple[Experiment, ...]:
    baseline = trajectory.states[0].score
    return tuple(result for result in trajectory.representative_results if result.score >= baseline - 0.0005)


def _draw_discarded(axes: Axes, results: tuple[Experiment, ...]) -> None:
    discarded = tuple(result for result in results if result.status != "keep")
    axes.scatter(
        [result.number - 1 for result in discarded],
        [result.score for result in discarded],
        c=DISCARDED_COLOR,
        s=12,
        alpha=0.5,
        zorder=2,
        label="Discarded",
    )


def _draw_kept(axes: Axes, states: tuple[IncumbentState, ...]) -> None:
    axes.scatter(
        [state.experiment - 1 for state in states],
        [state.score for state in states],
        c=KEPT_COLOR,
        s=50,
        zorder=4,
        label="Kept",
        edgecolors="black",
        linewidths=0.5,
    )


def _draw_running_best(axes: Axes, trajectory: Trajectory) -> None:
    scores = _running_best_scores(trajectory.states)
    axes.step(
        [state.experiment - 1 for state in trajectory.states],
        scores,
        where="post",
        color=RUNNING_BEST_COLOR,
        linewidth=2,
        alpha=0.7,
        zorder=3,
        label="Running best",
    )


def _running_best_scores(states: tuple[IncumbentState, ...]) -> list[float]:
    scores = []
    best = states[0].score
    for state in states:
        best = max(best, state.score)
        scores.append(best)
    return scores


def _draw_test_result(axes: Axes, trajectory: Trajectory) -> None:
    if not trajectory.blind:
        return
    x = trajectory.experiment_count + 1
    axes.scatter(
        [x],
        [trajectory.blind.score],
        marker="D",
        c=TEST_COLOR,
        s=55,
        zorder=4,
        edgecolors="black",
        linewidths=0.5,
        label="Test result",
    )
    axes.annotate(
        f"Test {format_percentage(trajectory.blind.score)}",
        (x, trajectory.blind.score),
        textcoords="offset points",
        xytext=(0, 9),
        fontsize=8,
        color=TEST_COLOR,
        ha="center",
        va="bottom",
    )


def _style_axes(axes: Axes, *, trajectory: Trajectory, heading: str) -> None:
    axes.set_xlabel("Experiment #", fontsize=12)
    axes.set_ylabel("Representative Dev F-score (higher is better)", fontsize=12)
    axes.set_title(
        f"{heading}: {trajectory.experiment_count} Experiments, {len(trajectory.states)} Kept Improvements",
        fontsize=14,
    )
    axes.legend(loc="upper left", bbox_to_anchor=(1.005, 1), borderaxespad=0, fontsize=9)
    axes.grid(True, alpha=0.2)
    axes.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
    axes.set_xlim(-1, trajectory.experiment_count + 3)
    axes.set_ylim(*_y_limits(trajectory))


def _y_limits(trajectory: Trajectory) -> tuple[float, float]:
    baseline = trajectory.states[0].score
    best = max(state.score for state in trajectory.states)
    values = [baseline, best]
    if trajectory.blind:
        values.append(trajectory.blind.score)
    span = max(values) - min(values)
    margin = max(0.01, span * 0.15)
    return max(0, min(values) - margin), min(1, max(values) + margin)


def accessible_description(trajectory: Trajectory) -> str:
    baseline = format_percentage(trajectory.states[0].score)
    final = format_percentage(max(state.score for state in trajectory.states))
    blind = f" Test dataset: {format_percentage(trajectory.blind.score)}." if trajectory.blind else ""
    milestones = "; ".join(
        f"experiment {state.experiment}, {state.description}, {format_delta(state.delta or 0)}, "
        f"finding: {state.finding}"
        for state in trajectory.milestones
    )
    suffix = f" Kept improvements: {milestones}." if milestones else " No kept improvement yet."
    return f"Representative Dev progress from {baseline} to {final}.{blind}{suffix}"
