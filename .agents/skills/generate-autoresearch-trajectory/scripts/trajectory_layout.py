import math


def wrap_text(text: str, *, width: int, max_lines: int) -> tuple[str, ...]:
    words = " ".join(text.split()).split(" ")
    lines: list[str] = []
    current = ""
    tail_omitted = False
    for word in words:
        if not word:
            continue
        if len(word) > width:
            word = word[: width - 1] + "…"
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= width:
            current = candidate
            continue
        lines.append(current)
        current = word
        if len(lines) == max_lines:
            tail_omitted = True
            current = ""
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    elif current:
        tail_omitted = True
    if not lines:
        return ("",)
    if tail_omitted and not lines[-1].endswith("…"):
        lines[-1] = lines[-1][: max(1, width - 1)].rstrip() + "…"
    return tuple(lines)


def y_domain(values: list[float]) -> tuple[float, float, tuple[float, ...]]:
    lower = min(values)
    upper = max(values)
    span = upper - lower
    padding = max(0.01, span * 0.10)
    raw_lower = max(0.0, lower - padding)
    raw_upper = min(1.0, upper + padding)
    step = _nice_number(max(raw_upper - raw_lower, 0.01) / 6)
    domain_lower = max(0.0, math.floor(raw_lower / step) * step)
    domain_upper = min(1.0, math.ceil(raw_upper / step) * step)
    if math.isclose(domain_lower, domain_upper):
        domain_lower = max(0.0, domain_lower - step)
        domain_upper = min(1.0, domain_upper + step)
    ticks = []
    tick = domain_lower
    while tick <= domain_upper + step / 10:
        ticks.append(round(tick, 10))
        tick += step
    return domain_lower, domain_upper, tuple(ticks)


def x_ticks(experiment_count: int) -> tuple[int, ...]:
    if experiment_count <= 7:
        return tuple(range(1, experiment_count + 1))
    raw_step = (experiment_count - 1) / 5
    magnitude = 10 ** math.floor(math.log10(raw_step))
    normalized = raw_step / magnitude
    multiplier = 1 if normalized <= 1 else 2 if normalized <= 2 else 5 if normalized <= 5 else 10
    step = max(1, int(multiplier * magnitude))
    ticks = [1]
    tick = step
    while tick < experiment_count:
        if tick > 1:
            ticks.append(tick)
        tick += step
    if ticks[-1] != experiment_count:
        ticks.append(experiment_count)
    return tuple(ticks)


def scale(
    value: float, *, source_min: float, source_max: float, target_min: float, target_max: float
) -> float:
    if math.isclose(source_min, source_max):
        return (target_min + target_max) / 2
    ratio = (value - source_min) / (source_max - source_min)
    return target_min + ratio * (target_max - target_min)


def format_percentage(value: float) -> str:
    return f"{value * 100:.1f}%"


def format_delta(value: float) -> str:
    return f"{value * 100:+.1f} pp"


def _nice_number(value: float) -> float:
    magnitude = 10 ** math.floor(math.log10(value))
    normalized = value / magnitude
    if normalized <= 1:
        multiplier = 1
    elif normalized <= 2:
        multiplier = 2
    elif normalized <= 3:
        multiplier = 2.5
    elif normalized <= 5:
        multiplier = 5
    else:
        multiplier = 10
    return multiplier * magnitude
