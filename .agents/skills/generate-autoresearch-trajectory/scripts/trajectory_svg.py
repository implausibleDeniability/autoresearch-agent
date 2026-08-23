import math
import xml.etree.ElementTree as ET

from svg_primitives import (
    BACKGROUND,
    BLUE,
    MONO,
    ORANGE,
    PRIMARY,
    SECONDARY,
    add_diamond,
    add_multiline,
    add_text,
)
from trajectory_chart import draw_chart
from trajectory_data import Trajectory, validate_xml_text
from trajectory_layout import format_delta, format_percentage, wrap_text
from trajectory_milestones import draw_milestone_list, layout_milestones

WIDTH = 1200
MIN_HEIGHT = 720
BASE_CHART_BOTTOM = 570
CANVAS_BOTTOM_PADDING = 40
CHART_LIST_OFFSET = 64


def render_svg(trajectory: Trajectory, *, heading: str) -> str:
    validate_xml_text(heading, context="title")
    heading_lines = wrap_text(heading, width=42, max_lines=2)
    chart_top = 164 + (len(heading_lines) - 1) * 28
    milestone_layout = layout_milestones(trajectory.milestones, top=chart_top)
    chart_bottom = max(BASE_CHART_BOTTOM, milestone_layout.bottom - CHART_LIST_OFFSET)
    height = _canvas_height(milestone_layout.bottom)
    root = ET.Element(
        "svg",
        {
            "xmlns": "http://www.w3.org/2000/svg",
            "viewBox": f"0 0 {WIDTH} {height}",
            "role": "img",
            "aria-labelledby": "trajectory-title trajectory-desc",
        },
    )
    title = ET.SubElement(root, "title", {"id": "trajectory-title"})
    title.text = heading
    description = ET.SubElement(root, "desc", {"id": "trajectory-desc"})
    description.text = _accessible_description(trajectory)
    ET.SubElement(root, "rect", {"width": str(WIDTH), "height": str(height), "fill": BACKGROUND})

    add_multiline(root, 56, 56, heading_lines, size=32, fill=PRIMARY, weight=650, line_height=34)
    subtitle_y = 96 + (len(heading_lines) - 1) * 30
    add_text(
        root,
        56,
        subtitle_y,
        f"Dev dataset across {trajectory.experiment_count} experiments",
        size=15,
        fill=SECONDARY,
    )
    _draw_summary(root, trajectory)

    chart_left = 88
    development_right = 690 if trajectory.blind else 772
    _draw_legend(root, 64, chart_top - 30, trajectory.blind is not None)
    draw_chart(
        root,
        trajectory,
        left=chart_left,
        right=development_right,
        top=chart_top,
        bottom=chart_bottom,
        final_divider=724,
        blind_x=756,
    )
    draw_milestone_list(root, milestone_layout, top=chart_top)

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True, short_empty_elements=True) + "\n"


def _draw_summary(root: ET.Element, trajectory: Trajectory) -> None:
    baseline = trajectory.states[0].score
    final = trajectory.states[-1].score
    add_text(
        root,
        1144,
        55,
        f"{format_percentage(baseline)}  →  {format_percentage(final)}",
        size=24,
        fill=PRIMARY,
        weight=650,
        anchor="end",
        family=MONO,
    )
    add_text(
        root,
        1144,
        84,
        f"{format_delta(final - baseline)} on Dev dataset",
        size=14,
        fill=BLUE,
        anchor="end",
    )


def _draw_legend(root: ET.Element, x: float, y: float, has_blind: bool) -> None:
    ET.SubElement(
        root,
        "line",
        {
            "x1": str(x),
            "y1": str(y),
            "x2": str(x + 28),
            "y2": str(y),
            "stroke": BLUE,
            "stroke-width": "3",
        },
    )
    add_text(root, x + 38, y + 5, "Dev dataset", size=14, fill=SECONDARY)
    if has_blind:
        add_diamond(root, x + 145, y, 6, ORANGE)
        add_text(root, x + 161, y + 5, "Test dataset", size=14, fill=SECONDARY)


def _canvas_height(milestone_bottom: float) -> int:
    required = milestone_bottom + CANVAS_BOTTOM_PADDING
    return max(MIN_HEIGHT, math.ceil(required / 8) * 8)


def _accessible_description(trajectory: Trajectory) -> str:
    baseline = format_percentage(trajectory.states[0].score)
    final = format_percentage(trajectory.states[-1].score)
    blind = f" Test dataset: {format_percentage(trajectory.blind.score)}." if trajectory.blind else ""
    milestone_parts = []
    for state in trajectory.milestones:
        evidence = f", finding: {state.finding}" if state.finding != state.description else ""
        milestone_parts.append(
            f"experiment {state.experiment}, {state.description}, "
            f"{format_delta(state.delta or 0)}{evidence}"
        )
    milestones = "; ".join(milestone_parts)
    suffix = f" Accepted improvements: {milestones}." if milestones else " No positive accepted improvements."
    return (
        f"Dev dataset trajectory across {trajectory.experiment_count} experiments, "
        f"from {baseline} to {final}.{blind}{suffix}"
    )
