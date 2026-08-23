from matplotlib.axes import Axes
from matplotlib.backend_bases import RendererBase
from matplotlib.figure import Figure
from matplotlib.text import Annotation
from matplotlib.transforms import Bbox

from trajectory_data import IncumbentState

LABEL_COLOR = "#1a7a3a"
LABEL_Y_OFFSETS = (6, 24, 42, 60, 84, 108, 132, -12, -30, -48, -72, -96, -120, -144)
LABEL_PLACEMENTS = ((6, 6, "left", "bottom"), (-6, -6, "right", "top")) + tuple(
    (x_offset, y_offset, alignment, "bottom")
    for x_offset, alignment in ((6, "left"), (-6, "right"))
    for y_offset in LABEL_Y_OFFSETS
)


def draw_labels(axes: Axes, states: tuple[IncumbentState, ...]) -> tuple[Annotation, ...]:
    annotations = []
    for index, state in enumerate(states):
        label = "baseline" if index == 0 else short_label(state.description)
        annotations.append(
            axes.annotate(
                label,
                (state.experiment - 1, state.score),
                textcoords="offset points",
                xytext=LABEL_PLACEMENTS[0][:2],
                fontsize=8,
                color=LABEL_COLOR,
                alpha=0.9,
                rotation=30,
                ha="left",
                va="bottom",
            )
        )
    return tuple(annotations)


def short_label(description: str) -> str:
    return description if len(description) <= 45 else description[:42] + "..."


def separate_annotations(
    figure: Figure,
    *,
    axes: Axes,
    annotations: tuple[Annotation, ...],
) -> None:
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    axes_box = axes.get_window_extent(renderer)
    placed = _static_obstacles(axes, annotations=annotations, renderer=renderer)
    for annotation in annotations:
        box = _place_annotation(annotation, renderer=renderer, axes_box=axes_box, placed=placed)
        placed.append(box)
    figure.canvas.draw()


def _static_obstacles(
    axes: Axes,
    *,
    annotations: tuple[Annotation, ...],
    renderer: RendererBase,
) -> list[Bbox]:
    boxes = [
        text.get_window_extent(renderer).expanded(1.04, 1.12)
        for text in axes.texts
        if text not in annotations
    ]
    legend = axes.get_legend()
    if legend is not None:
        boxes.append(legend.get_window_extent(renderer).expanded(1.02, 1.04))
    return boxes


def _place_annotation(
    annotation: Annotation,
    *,
    renderer: RendererBase,
    axes_box: Bbox,
    placed: list[Bbox],
) -> Bbox:
    best: tuple[tuple[bool, float, float], Bbox, tuple[int, int], str, str] | None = None
    for x_offset, y_offset, alignment, vertical_alignment in LABEL_PLACEMENTS:
        offset = (x_offset, y_offset)
        annotation.set_position(offset)
        annotation.set_ha(alignment)
        annotation.set_va(vertical_alignment)
        box = annotation.get_window_extent(renderer).expanded(1.02, 1.08)
        overflow = _overflow_area(box, axes_box)
        overlap = sum(_intersection_area(box, other) for other in placed)
        score = (overlap > 0, overlap, overflow)
        if best is None or score < best[0]:
            best = (score, box, offset, alignment, vertical_alignment)
    _, box, offset, alignment, vertical_alignment = best
    annotation.set_position(offset)
    annotation.set_ha(alignment)
    annotation.set_va(vertical_alignment)
    return box


def _intersection_area(first: Bbox, second: Bbox) -> float:
    width = max(0, min(first.x1, second.x1) - max(first.x0, second.x0))
    height = max(0, min(first.y1, second.y1) - max(first.y0, second.y0))
    return width * height


def _overflow_area(box: Bbox, boundary: Bbox) -> float:
    horizontal = max(0, boundary.x0 - box.x0) + max(0, box.x1 - boundary.x1)
    vertical = max(0, boundary.y0 - box.y0) + max(0, box.y1 - boundary.y1)
    return horizontal * box.height + vertical * box.width
