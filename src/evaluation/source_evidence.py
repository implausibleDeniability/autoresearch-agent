from bisect import bisect_right
from dataclasses import dataclass
from typing import Dict, List, Literal, Sequence, Tuple


class SourceMatchKind:
    RAW_EXACT = "raw_exact"
    NORMALIZED_EXACT = "normalized_exact"
    FUZZY = "fuzzy"

    @classmethod
    def all(cls) -> Tuple[str, ...]:
        return cls.RAW_EXACT, cls.NORMALIZED_EXACT, cls.FUZZY


SourceMatchKindLiteral = Literal["raw_exact", "normalized_exact", "fuzzy"]


@dataclass(frozen=True)
class SourceEvidence:
    start: int
    end: int
    match_kind: SourceMatchKindLiteral
    similarity: float


def select_source_evidence(candidates: Sequence[SourceEvidence]) -> Tuple[SourceEvidence, ...]:
    deduplicated = _deduplicate(candidates)
    raw = _select_maximum(_with_kind(deduplicated, SourceMatchKind.RAW_EXACT))
    normalized = _select_without_overlaps(
        _with_kind(deduplicated, SourceMatchKind.NORMALIZED_EXACT),
        reserved=raw,
    )
    fuzzy = _select_without_overlaps(
        _with_kind(deduplicated, SourceMatchKind.FUZZY),
        reserved=raw + normalized,
    )
    return tuple(sorted(raw + normalized + fuzzy, key=lambda evidence: (evidence.start, evidence.end)))


def _deduplicate(candidates: Sequence[SourceEvidence]) -> Tuple[SourceEvidence, ...]:
    best_by_coordinates: Dict[Tuple[int, int], SourceEvidence] = {}
    for candidate in candidates:
        coordinates = candidate.start, candidate.end
        current = best_by_coordinates.get(coordinates)
        if current is None or _candidate_key(candidate) > _candidate_key(current):
            best_by_coordinates[coordinates] = candidate
    return tuple(best_by_coordinates.values())


def _candidate_key(evidence: SourceEvidence) -> Tuple[int, float]:
    kind_priority = len(SourceMatchKind.all()) - SourceMatchKind.all().index(evidence.match_kind)
    return kind_priority, evidence.similarity


def _with_kind(
    candidates: Sequence[SourceEvidence], match_kind: SourceMatchKindLiteral
) -> Tuple[SourceEvidence, ...]:
    return tuple(candidate for candidate in candidates if candidate.match_kind == match_kind)


def _select_without_overlaps(
    candidates: Sequence[SourceEvidence], *, reserved: Sequence[SourceEvidence]
) -> Tuple[SourceEvidence, ...]:
    reserved_by_end = sorted(reserved, key=lambda evidence: (evidence.end, evidence.start))
    endpoints = [evidence.end for evidence in reserved_by_end]
    available = tuple(
        candidate for candidate in candidates if not _overlaps_reserved(candidate, reserved_by_end, endpoints)
    )
    return _select_maximum(available)


def _overlaps_reserved(
    candidate: SourceEvidence,
    reserved_by_end: Sequence[SourceEvidence],
    endpoints: Sequence[int],
) -> bool:
    index = bisect_right(endpoints, candidate.start)
    return index < len(reserved_by_end) and reserved_by_end[index].start < candidate.end


def _select_maximum(candidates: Sequence[SourceEvidence]) -> Tuple[SourceEvidence, ...]:
    ordered = sorted(candidates, key=lambda evidence: (evidence.end, evidence.start))
    endpoints = [evidence.end for evidence in ordered]
    scores = [(0, 0.0, 0, 0, 0)]
    parents = [0]
    included = [False]
    for index, candidate in enumerate(ordered):
        predecessor = bisect_right(endpoints, candidate.start, hi=index) - 1
        predecessor_state = predecessor + 1
        included_score = _add_score(scores[predecessor_state], _candidate_score(candidate))
        excluded_score = scores[index]
        choose_included = included_score > excluded_score
        scores.append(included_score if choose_included else excluded_score)
        parents.append(predecessor_state if choose_included else index)
        included.append(choose_included)
    return _reconstruct_selection(ordered, parents=parents, included=included)


def _candidate_score(evidence: SourceEvidence) -> Tuple[int, float, int, int, int]:
    return (
        1,
        evidence.similarity,
        -(evidence.end - evidence.start),
        -evidence.start,
        -evidence.end,
    )


def _add_score(
    first: Tuple[int, float, int, int, int], second: Tuple[int, float, int, int, int]
) -> Tuple[int, float, int, int, int]:
    return tuple(left + right for left, right in zip(first, second))


def _reconstruct_selection(
    ordered: Sequence[SourceEvidence], *, parents: Sequence[int], included: Sequence[bool]
) -> Tuple[SourceEvidence, ...]:
    selected: List[SourceEvidence] = []
    state = len(ordered)
    while state:
        if included[state]:
            selected.append(ordered[state - 1])
        state = parents[state]
    return tuple(reversed(selected))


def _overlap(first: SourceEvidence, second: SourceEvidence) -> bool:
    return first.start < second.end and second.start < first.end
