import xml.etree.ElementTree as ET
from dataclasses import dataclass

from svg_primitives import BLUE, MONO, PRIMARY, SECONDARY, add_multiline, add_text
from trajectory_data import IncumbentState
from trajectory_layout import format_delta, wrap_text_complete

LIST_X = 836
CONTENT_X = 884
NUMBER_SIZE = 26
META_SIZE = 14
TITLE_SIZE = 17
FINDING_SIZE = 15
TITLE_WIDTH = 31
FINDING_WIDTH = 36
TITLE_BASELINE = 36
TITLE_LINE_HEIGHT = 18
FINDING_GAP = 22
FINDING_LINE_HEIGHT = 17
ROW_BOTTOM_PADDING = 12
ROW_GAP = 14
TEXT_ASCENT_RATIO = 0.82


@dataclass(frozen=True)
class MilestoneRow:
    index: int
    state: IncumbentState
    title_lines: tuple[str, ...]
    finding_lines: tuple[str, ...]
    top: float
    bottom: float


@dataclass(frozen=True)
class MilestoneLayout:
    rows: tuple[MilestoneRow, ...]
    bottom: float


def layout_milestones(milestones: tuple[IncumbentState, ...], *, top: float) -> MilestoneLayout:
    rows = []
    row_top = top
    for index, state in enumerate(milestones, start=1):
        row = _layout_row(index=index, state=state, top=row_top)
        rows.append(row)
        row_top = row.bottom + ROW_GAP
    bottom = rows[-1].bottom if rows else top + 28
    return MilestoneLayout(tuple(rows), bottom)


def draw_milestone_list(root: ET.Element, layout: MilestoneLayout, *, top: float) -> None:
    group = ET.SubElement(root, "g", {"data-section": "milestones"})
    _draw_list_heading(group, top=top)
    if not layout.rows:
        _draw_empty_state(group, top=top)
        return
    for row in layout.rows:
        _draw_row(group, row)


def _draw_list_heading(root: ET.Element, *, top: float) -> None:
    add_text(
        root,
        LIST_X,
        top - 28,
        "WHAT WORKED",
        size=15,
        fill=SECONDARY,
        weight=650,
        letter_spacing="1.8",
    )


def _draw_empty_state(root: ET.Element, *, top: float) -> None:
    add_text(root, LIST_X, top + 28, "No accepted improvement yet", size=16, fill=SECONDARY)


def _layout_row(*, index: int, state: IncumbentState, top: float) -> MilestoneRow:
    title_lines = wrap_text_complete(state.description, width=TITLE_WIDTH)
    finding_lines = wrap_text_complete(state.finding, width=FINDING_WIDTH)
    title_last = top + TITLE_BASELINE + (len(title_lines) - 1) * TITLE_LINE_HEIGHT
    finding_last = title_last + FINDING_GAP + (len(finding_lines) - 1) * FINDING_LINE_HEIGHT
    return MilestoneRow(
        index=index,
        state=state,
        title_lines=title_lines,
        finding_lines=finding_lines,
        top=top,
        bottom=finding_last + ROW_BOTTOM_PADDING,
    )


def _draw_row(root: ET.Element, row: MilestoneRow) -> None:
    group = _row_group(root, row)
    _draw_number(group, row)
    _draw_metadata(group, row)
    _draw_title(group, row)
    _draw_finding(group, row)


def _row_group(root: ET.Element, row: MilestoneRow) -> ET.Element:
    return ET.SubElement(
        root,
        "g",
        {
            "data-row": str(row.index),
            "data-top": str(row.top),
            "data-bottom": str(row.bottom),
        },
    )


def _draw_number(root: ET.Element, row: MilestoneRow) -> None:
    number = add_text(
        root,
        LIST_X,
        row.top + round(NUMBER_SIZE * TEXT_ASCENT_RATIO),
        f"{row.index:02d}",
        size=NUMBER_SIZE,
        fill=BLUE,
        weight=650,
        family=MONO,
    )
    number.set("data-role", "milestone-number")


def _draw_metadata(root: ET.Element, row: MilestoneRow) -> None:
    metadata = add_text(
        root,
        CONTENT_X,
        row.top + round(META_SIZE * TEXT_ASCENT_RATIO),
        f"EXP {row.state.experiment:02d}  ·  {format_delta(row.state.delta or 0)}",
        size=META_SIZE,
        fill=SECONDARY,
        weight=600,
        family=MONO,
    )
    metadata.set("data-role", "milestone-meta")


def _draw_title(root: ET.Element, row: MilestoneRow) -> None:
    add_multiline(
        root,
        CONTENT_X,
        row.top + TITLE_BASELINE,
        row.title_lines,
        size=TITLE_SIZE,
        fill=PRIMARY,
        weight=600,
        line_height=TITLE_LINE_HEIGHT,
    )


def _draw_finding(root: ET.Element, row: MilestoneRow) -> None:
    title_last = row.top + TITLE_BASELINE + (len(row.title_lines) - 1) * TITLE_LINE_HEIGHT
    add_multiline(
        root,
        CONTENT_X,
        title_last + FINDING_GAP,
        row.finding_lines,
        size=FINDING_SIZE,
        fill=SECONDARY,
        line_height=FINDING_LINE_HEIGHT,
    )
