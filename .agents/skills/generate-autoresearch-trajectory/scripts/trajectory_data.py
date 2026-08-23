import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

LEGACY_FIELDS = (
    "commit",
    "score",
    "precision",
    "recall",
    "cost",
    "status",
    "description",
    "dataset",
    "budget_cost_usd",
)
CURRENT_FIELDS = LEGACY_FIELDS + ("finding",)
REPLAY_FIELDS = CURRENT_FIELDS + ("evaluation_mode",)
EVALUATION_MODES = {"live", "cached"}
STATUSES = {"keep", "discard", "inconclusive", "crash"}
DATASETS = {"debug", "dev-19k", "dev-87k", "dev-205k"}
BLIND_FIELDS = ("f_score", "precision", "recall", "api_cost_usd", "duration_seconds")


class TrajectoryError(ValueError):
    pass


@dataclass(frozen=True)
class Experiment:
    number: int
    commit: str
    score: float
    precision: float
    recall: float
    cost: float
    status: str
    description: str
    dataset: str
    budget_cost_usd: float
    finding: str
    evaluation_mode: str = "live"


@dataclass(frozen=True)
class IncumbentState:
    experiment: int
    commit: str
    score: float
    description: str
    finding: str
    delta: float | None


@dataclass(frozen=True)
class BlindResult:
    score: float
    precision: float
    recall: float
    api_cost_usd: float
    duration_seconds: float


@dataclass(frozen=True)
class Trajectory:
    experiment_count: int
    states: tuple[IncumbentState, ...]
    milestones: tuple[IncumbentState, ...]
    blind: BlindResult | None


def load_trajectory(
    results_path: Path,
    *,
    run_log_path: Path | None = None,
    max_milestones: int = 7,
) -> Trajectory:
    experiments = read_experiments(results_path)
    states = build_incumbent_states(experiments)
    if not states:
        raise TrajectoryError(
            f"{results_path}: no accepted dev-205k result; run a complete dev-205k baseline "
            "and mark it keep"
        )
    baseline = experiments[0]
    if baseline.status != "keep" or baseline.dataset != "dev-205k":
        raise TrajectoryError(f"{results_path}: experiment 1 must be an accepted dev-205k baseline")
    candidates = [state for state in states[1:] if state.delta is not None and state.delta > 0]
    ranked = sorted(candidates, key=lambda state: (-state.delta, state.experiment))[:max_milestones]
    milestones = tuple(sorted(ranked, key=lambda state: state.experiment))
    blind = read_blind_result(run_log_path) if run_log_path is not None else None
    return Trajectory(len(experiments), tuple(states), milestones, blind)


def read_experiments(path: Path) -> list[Experiment]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
            fields = tuple(reader.fieldnames or ())
            if fields not in {LEGACY_FIELDS, CURRENT_FIELDS, REPLAY_FIELDS}:
                raise TrajectoryError(f"{path}: expected the exact 9-, 10-, or 11-column supported header")
            experiments = [
                _parse_experiment(
                    path,
                    number,
                    row,
                    has_finding=fields in {CURRENT_FIELDS, REPLAY_FIELDS},
                    has_evaluation_mode=fields == REPLAY_FIELDS,
                )
                for number, row in enumerate(reader, start=1)
            ]
    except OSError as error:
        raise TrajectoryError(f"cannot read {path}: {error.strerror or error}") from error
    except UnicodeError as error:
        raise TrajectoryError(f"cannot read {path}: invalid UTF-8") from error
    except csv.Error as error:
        raise TrajectoryError(f"{path}: malformed TSV: {error}") from error
    if not experiments:
        raise TrajectoryError(f"{path}: no experiment rows")
    return experiments


def _parse_experiment(
    path: Path,
    number: int,
    row: dict[str | None, str | list[str] | None],
    has_finding: bool,
    has_evaluation_mode: bool,
) -> Experiment:
    if None in row or any(value is None for value in row.values()):
        raise TrajectoryError(f"{path}: row {number}: column count does not match the header")
    values = {key: str(value).strip() for key, value in row.items() if key is not None}
    for field in ("commit", "status", "description", "dataset"):
        if not values[field]:
            raise TrajectoryError(f"{path}: row {number}: {field} must not be empty")
    if values["status"] not in STATUSES:
        raise TrajectoryError(f"{path}: row {number}: unsupported status")
    if values["dataset"] not in DATASETS:
        raise TrajectoryError(f"{path}: row {number}: unsupported dataset")
    for field in ("commit", "description", "finding"):
        _validate_xml_text(path, number, field, values.get(field, ""))
    score = _parse_number(path, number, "score", values["score"], minimum=0, maximum=1)
    precision = _parse_number(path, number, "precision", values["precision"], minimum=0, maximum=1)
    recall = _parse_number(path, number, "recall", values["recall"], minimum=0, maximum=1)
    cost = _parse_number(path, number, "cost", values["cost"], minimum=0)
    budget = _parse_number(path, number, "budget_cost_usd", values["budget_cost_usd"], minimum=0)
    finding = values.get("finding", "")
    evaluation_mode = values.get("evaluation_mode", "live")
    if has_evaluation_mode and evaluation_mode not in EVALUATION_MODES:
        raise TrajectoryError(f"{path}: row {number}: unsupported evaluation_mode")
    if has_finding and values["status"] == "keep" and values["dataset"] == "dev-205k" and not finding:
        raise TrajectoryError(f"{path}: row {number}: accepted dev-205k finding must not be empty")
    return Experiment(
        number,
        values["commit"],
        score,
        precision,
        recall,
        cost,
        values["status"],
        values["description"],
        values["dataset"],
        budget,
        finding,
        evaluation_mode,
    )


def _parse_number(
    path: Path,
    row_number: int,
    field: str,
    value: str,
    *,
    minimum: float,
    maximum: float | None = None,
) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise TrajectoryError(f"{path}: row {row_number}: {field} must be a number") from error
    if not math.isfinite(number) or number < minimum or (maximum is not None and number > maximum):
        bounds = f"between {minimum:g} and {maximum:g}" if maximum is not None else f"at least {minimum:g}"
        raise TrajectoryError(f"{path}: row {row_number}: {field} must be finite and {bounds}")
    return number


def _validate_xml_text(path: Path, row_number: int, field: str, value: str) -> None:
    validate_xml_text(value, context=f"{path}: row {row_number}: {field}")


def validate_xml_text(value: str, *, context: str) -> None:
    for character in value:
        codepoint = ord(character)
        if not (
            codepoint in {0x9, 0xA, 0xD}
            or 0x20 <= codepoint <= 0xD7FF
            or 0xE000 <= codepoint <= 0xFFFD
            or 0x10000 <= codepoint <= 0x10FFFF
        ):
            raise TrajectoryError(f"{context} contains an illegal XML character")


def build_incumbent_states(experiments: list[Experiment]) -> list[IncumbentState]:
    accepted = [row for row in experiments if row.status == "keep" and row.dataset == "dev-205k"]
    episodes: list[list[Experiment]] = []
    for row in accepted:
        if episodes and episodes[-1][-1].commit == row.commit:
            episodes[-1].append(row)
        else:
            episodes.append([row])
    states: list[IncumbentState] = []
    for episode in episodes:
        final = episode[-1]
        score = statistics.median(row.score for row in episode)
        delta = None if not states else score - states[-1].score
        states.append(
            IncumbentState(
                final.number,
                final.commit,
                score,
                final.description,
                final.finding or final.description,
                delta,
            )
        )
    return states


def read_blind_result(path: Path) -> BlindResult | None:
    try:
        handle = path.open(encoding="utf-8", errors="strict")
    except (OSError, UnicodeError) as error:
        detail = getattr(error, "strerror", None) or error
        raise TrajectoryError(f"cannot read {path}: {detail}") from error
    recognized = set(BLIND_FIELDS) | {"result_status"}
    values: dict[str, str] = {}
    try:
        with handle:
            for line in handle:
                key, separator, value = line.rstrip("\r\n").partition("=")
                if not separator or key not in recognized:
                    continue
                if key in values:
                    raise TrajectoryError(f"{path}: duplicate {key} field")
                values[key] = value.strip()
    except UnicodeError as error:
        raise TrajectoryError(f"cannot read {path}: invalid UTF-8") from error
    if "result_status" in values or not values:
        return None
    missing = [field for field in BLIND_FIELDS if field not in values]
    if missing:
        raise TrajectoryError(f"{path}: partial Test dataset result; missing {', '.join(missing)}")
    parsed = {
        field: _parse_number(path, 0, field, values[field], minimum=0, maximum=1)
        for field in ("f_score", "precision", "recall")
    }
    parsed["api_cost_usd"] = _parse_number(path, 0, "api_cost_usd", values["api_cost_usd"], minimum=0)
    parsed["duration_seconds"] = _parse_number(
        path, 0, "duration_seconds", values["duration_seconds"], minimum=0
    )
    return BlindResult(
        parsed["f_score"],
        parsed["precision"],
        parsed["recall"],
        parsed["api_cost_usd"],
        parsed["duration_seconds"],
    )
