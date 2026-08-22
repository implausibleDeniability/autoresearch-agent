import xml.etree.ElementTree as ET

BACKGROUND = "#0B0F19"
PRIMARY = "#EDF3FF"
SECONDARY = "#AAB6CC"
GRID = "#273149"
BLUE = "#5AA7FF"
ORANGE = "#FFB55A"
SANS = "IBM Plex Sans, Segoe UI, sans-serif"
MONO = "IBM Plex Mono, Consolas, monospace"


def add_multiline(
    root: ET.Element,
    x: float,
    y: float,
    lines: tuple[str, ...],
    *,
    size: int,
    fill: str,
    weight: int = 400,
    line_height: int,
) -> None:
    text = add_text(root, x, y, "", size=size, fill=fill, weight=weight)
    for index, line in enumerate(lines):
        span = ET.SubElement(
            text,
            "tspan",
            {"x": format_number(x), "dy": "0" if index == 0 else str(line_height)},
        )
        span.text = line


def add_text(
    root: ET.Element,
    x: float,
    y: float,
    value: str,
    *,
    size: int,
    fill: str,
    weight: int = 400,
    anchor: str | None = None,
    family: str = SANS,
    transform: str | None = None,
    letter_spacing: str | None = None,
) -> ET.Element:
    attributes = {
        "x": format_number(x),
        "y": format_number(y),
        "fill": fill,
        "font-family": family,
        "font-size": str(size),
        "font-weight": str(weight),
    }
    if anchor:
        attributes["text-anchor"] = anchor
    if transform:
        attributes["transform"] = transform
    if letter_spacing:
        attributes["letter-spacing"] = letter_spacing
    text = ET.SubElement(root, "text", attributes)
    text.text = value
    return text


def add_diamond(
    root: ET.Element,
    x: float,
    y: float,
    radius: float,
    fill: str,
    **attributes: str,
) -> None:
    points = " ".join(
        (
            f"{format_number(x)},{format_number(y - radius)}",
            f"{format_number(x + radius)},{format_number(y)}",
            f"{format_number(x)},{format_number(y + radius)}",
            f"{format_number(x - radius)},{format_number(y)}",
        )
    )
    values = {"points": points, "fill": fill}
    values.update({key.replace("_", "-"): value for key, value in attributes.items()})
    ET.SubElement(root, "polygon", values)


def format_number(value: float) -> str:
    rounded = round(value, 2)
    return str(int(rounded)) if rounded.is_integer() else f"{rounded:.2f}".rstrip("0").rstrip(".")
