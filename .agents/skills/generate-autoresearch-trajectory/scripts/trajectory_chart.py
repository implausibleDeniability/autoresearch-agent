import xml.etree.ElementTree as ET

from svg_primitives import (
    BACKGROUND,
    BLUE,
    GRID,
    MONO,
    ORANGE,
    SECONDARY,
    add_diamond,
    add_text,
    format_number,
)
from trajectory_data import Trajectory
from trajectory_layout import format_percentage, scale, x_ticks, y_domain


def draw_chart(
    root: ET.Element,
    trajectory: Trajectory,
    *,
    left: float,
    right: float,
    top: float,
    bottom: float,
    final_divider: float,
    blind_x: float,
) -> None:
    scores = [state.score for state in trajectory.states]
    if trajectory.blind:
        scores.append(trajectory.blind.score)
    domain_low, domain_high, ticks = y_domain(scores)
    _draw_axes(root, trajectory, left, right, top, bottom, ticks, domain_low, domain_high)
    _draw_incumbent(root, trajectory, left, right, top, bottom, domain_low, domain_high)
    _draw_milestone_markers(root, trajectory, left, right, top, bottom, domain_low, domain_high)
    if trajectory.blind:
        _draw_blind(root, trajectory, top, bottom, final_divider, blind_x, domain_low, domain_high)


def _draw_axes(
    root: ET.Element,
    trajectory: Trajectory,
    left: float,
    right: float,
    top: float,
    bottom: float,
    ticks: tuple[float, ...],
    domain_low: float,
    domain_high: float,
) -> None:
    for tick in ticks:
        y = _score_y(tick, domain_low, domain_high, top, bottom)
        ET.SubElement(
            root,
            "line",
            {
                "x1": str(left),
                "y1": format_number(y),
                "x2": "772",
                "y2": format_number(y),
                "stroke": GRID,
                "stroke-width": "1",
            },
        )
        add_text(
            root,
            left - 14,
            y + 5,
            format_percentage(tick),
            size=14,
            fill=SECONDARY,
            anchor="end",
            family=MONO,
        )
    for experiment in x_ticks(trajectory.experiment_count):
        x = _experiment_x(experiment, trajectory.experiment_count, left, right)
        ET.SubElement(
            root,
            "line",
            {
                "x1": format_number(x),
                "y1": str(bottom),
                "x2": format_number(x),
                "y2": str(bottom + 6),
                "stroke": GRID,
            },
        )
        add_text(root, x, bottom + 28, str(experiment), size=14, fill=SECONDARY, anchor="middle", family=MONO)
    add_text(root, (left + right) / 2, bottom + 58, "Experiment", size=15, fill=SECONDARY, anchor="middle")
    add_text(
        root,
        24,
        (top + bottom) / 2,
        "F-score (β² = 5)",
        size=14,
        fill=SECONDARY,
        anchor="middle",
        transform=f"rotate(-90 24 {(top + bottom) / 2})",
    )


def _draw_incumbent(
    root: ET.Element,
    trajectory: Trajectory,
    left: float,
    right: float,
    top: float,
    bottom: float,
    domain_low: float,
    domain_high: float,
) -> None:
    path = _step_path(trajectory, left, right, top, bottom, domain_low, domain_high)
    ET.SubElement(
        root,
        "path",
        {
            "d": path,
            "fill": "none",
            "stroke": BLUE,
            "stroke-width": "3",
            "stroke-linecap": "round",
            "stroke-linejoin": "round",
            "data-series": "dev-dataset",
        },
    )
    for state in trajectory.states:
        x = _experiment_x(state.experiment, trajectory.experiment_count, left, right)
        y = _score_y(state.score, domain_low, domain_high, top, bottom)
        ET.SubElement(
            root,
            "circle",
            {
                "cx": format_number(x),
                "cy": format_number(y),
                "r": "5",
                "fill": BACKGROUND,
                "stroke": BLUE,
                "stroke-width": "3",
                "data-experiment": str(state.experiment),
                "data-score": f"{state.score:.6f}",
            },
        )


def _step_path(
    trajectory: Trajectory,
    left: float,
    right: float,
    top: float,
    bottom: float,
    domain_low: float,
    domain_high: float,
) -> str:
    first_y = _score_y(trajectory.states[0].score, domain_low, domain_high, top, bottom)
    commands = [f"M {format_number(left)} {format_number(first_y)}"]
    for state in trajectory.states[1:]:
        x = _experiment_x(state.experiment, trajectory.experiment_count, left, right)
        y = _score_y(state.score, domain_low, domain_high, top, bottom)
        commands.extend((f"H {format_number(x)}", f"V {format_number(y)}"))
    commands.append(f"H {format_number(right)}")
    return " ".join(commands)


def _draw_milestone_markers(
    root: ET.Element,
    trajectory: Trajectory,
    left: float,
    right: float,
    top: float,
    bottom: float,
    domain_low: float,
    domain_high: float,
) -> None:
    positions = [
        (
            _experiment_x(state.experiment, trajectory.experiment_count, left, right),
            _score_y(state.score, domain_low, domain_high, top, bottom),
        )
        for state in trajectory.milestones
    ]
    if any(right_x - left_x < 36 for (left_x, _), (right_x, _) in zip(positions, positions[1:])):
        return
    for index, (x, y) in enumerate(positions, start=1):
        offset = -18
        if y + offset < top + 4:
            offset = 26
        if y + offset > bottom - 4:
            offset = -18
        label_y = y + offset
        ET.SubElement(
            root,
            "line",
            {
                "x1": format_number(x),
                "y1": format_number(y),
                "x2": format_number(x),
                "y2": format_number(label_y),
                "stroke": BLUE,
                "stroke-width": "1",
                "data-series": "milestone-label",
            },
        )
        add_text(
            root,
            x,
            label_y + 5,
            f"{index:02d}",
            size=13,
            fill=BLUE,
            weight=700,
            anchor="middle",
            family=MONO,
        )


def _draw_blind(
    root: ET.Element,
    trajectory: Trajectory,
    top: float,
    bottom: float,
    divider: float,
    blind_x: float,
    domain_low: float,
    domain_high: float,
) -> None:
    ET.SubElement(
        root,
        "line",
        {
            "x1": str(divider),
            "y1": str(top - 8),
            "x2": str(divider),
            "y2": str(bottom + 4),
            "stroke": GRID,
            "stroke-width": "1",
            "stroke-dasharray": "5 7",
        },
    )
    add_text(root, blind_x, top - 18, "TEST", size=13, fill=SECONDARY, anchor="middle", family=MONO)
    blind_y = _score_y(trajectory.blind.score, domain_low, domain_high, top, bottom)
    add_diamond(root, blind_x, blind_y, 8, ORANGE, data_score=f"{trajectory.blind.score:.6f}")
    label_y = blind_y - 18 if blind_y > top + 44 else blind_y + 30
    add_text(
        root,
        blind_x,
        label_y,
        format_percentage(trajectory.blind.score),
        size=14,
        fill=ORANGE,
        weight=650,
        anchor="middle",
        family=MONO,
    )


def _experiment_x(experiment: int, count: int, left: float, right: float) -> float:
    return scale(experiment, source_min=1, source_max=max(1, count), target_min=left, target_max=right)


def _score_y(score: float, low: float, high: float, top: float, bottom: float) -> float:
    return scale(score, source_min=low, source_max=high, target_min=bottom, target_max=top)
