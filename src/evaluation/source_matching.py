import re
from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

from src.evaluation.matching import (
    MATCH,
    MINIMUM_FUZZY_LENGTH,
    SIMILARITY_THRESHOLD,
    compare_values,
    normalize_value,
    similarity_length_bounds,
)
from src.evaluation.source_candidate_windows import CandidateWindowBudgetExceeded, candidate_windows
from src.evaluation.source_evidence import SourceEvidence, SourceMatchKind, select_source_evidence

SOURCE_MATCHING_POLICY_VERSION = 1
CANDIDATE_BOUNDARIES = "literal_substrings_and_punctuation_delimited_token_windows"
OVERLAP_POLICY = "raw_then_normalized_then_maximum_cardinality_v1"
FUZZY_WORK_BUDGET = 50_000_000
CANDIDATE_ENUMERATION_BUDGET = 200_000


class SourceValueRole:
    PREDICTION = "prediction"
    GROUND_TRUTH = "ground_truth"


SourceValueRoleLiteral = Literal["prediction", "ground_truth"]


@dataclass(frozen=True)
class SourceMatchResult:
    evidence: Tuple[SourceEvidence, ...]
    fuzzy_search_complete: bool


def source_matching_policy() -> Dict[str, object]:
    return {
        "version": SOURCE_MATCHING_POLICY_VERSION,
        "normalization": "lower_strip_trailing_period_v1",
        "similarity_algorithm": "difflib_sequence_matcher_autojunk_false",
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "minimum_fuzzy_length": MINIMUM_FUZZY_LENGTH,
        "fuzzy_work_budget": FUZZY_WORK_BUDGET,
        "candidate_enumeration_budget": CANDIDATE_ENUMERATION_BUDGET,
        "comparison_orientation": "prediction_values_value_to_span_ground_truth_values_span_to_value",
        "candidate_boundaries": CANDIDATE_BOUNDARIES,
        "overlap_policy": OVERLAP_POLICY,
    }


class SourceTextMatcher:
    def __init__(self, text: str) -> None:
        self._text = text
        self._tokens = tuple(match.span() for match in re.finditer(r"\S+", text))
        self._cache: Dict[Tuple[str, str], SourceMatchResult] = {}

    def find(self, value: str, *, role: SourceValueRoleLiteral) -> SourceMatchResult:
        cache_key = role, value
        if cache_key not in self._cache:
            fuzzy_candidates, fuzzy_search_complete = self._token_candidates(value, role=role)
            candidates = (
                self._raw_candidates(value) + self._normalized_literal_candidates(value) + fuzzy_candidates
            )
            self._cache[cache_key] = SourceMatchResult(
                evidence=select_source_evidence(candidates),
                fuzzy_search_complete=fuzzy_search_complete,
            )
        return self._cache[cache_key]

    def _raw_candidates(self, value: str) -> List[SourceEvidence]:
        if not value:
            return []
        candidates = []
        start = self._text.find(value)
        while start >= 0:
            candidates.append(_raw_evidence(start, value=value))
            start = self._text.find(value, start + len(value))
        return candidates

    def _normalized_literal_candidates(self, value: str) -> List[SourceEvidence]:
        normalized_value = normalize_value(value)
        if not normalized_value:
            return []
        candidates = []
        for match in re.finditer(re.escape(value), self._text, flags=re.IGNORECASE):
            start, end = match.span()
            candidate = self._text[start:end]
            if candidate != value and normalize_value(candidate) == normalized_value:
                candidates.append(_normalized_evidence(start, value=value))
        return candidates

    def _token_candidates(
        self, value: str, *, role: SourceValueRoleLiteral
    ) -> Tuple[List[SourceEvidence], bool]:
        normalized_value = normalize_value(value)
        if not normalized_value:
            return [], True
        minimum_length, maximum_length = similarity_length_bounds(value)
        if minimum_length > len(normalize_value(self._text)):
            return [], True
        maximum_windows = FUZZY_WORK_BUDGET // max(1, len(normalized_value)) ** 2
        windows = self._candidate_windows(
            minimum_length=minimum_length,
            maximum_length=maximum_length,
            maximum_windows=maximum_windows,
        )
        if windows is None:
            return [], False
        candidates = []
        for start, end in windows:
            evidence = _matching_evidence(
                self._text[start:end],
                value=value,
                role=role,
                start=start,
                end=end,
            )
            if evidence:
                candidates.append(evidence)
        return candidates, True

    def _candidate_windows(
        self, *, minimum_length: int, maximum_length: int, maximum_windows: int
    ) -> List[Tuple[int, int]] | None:
        windows: Dict[Tuple[int, int], None] = {}
        try:
            for span in candidate_windows(
                self._text,
                tokens=self._tokens,
                minimum_length=minimum_length,
                maximum_length=maximum_length,
                maximum_examined=CANDIDATE_ENUMERATION_BUDGET,
            ):
                windows[span] = None
                if len(windows) > maximum_windows:
                    return None
        except CandidateWindowBudgetExceeded:
            return None
        return list(windows)


def _raw_evidence(start: int, *, value: str) -> SourceEvidence:
    return SourceEvidence(
        start=start,
        end=start + len(value),
        match_kind=SourceMatchKind.RAW_EXACT,
        similarity=1.0,
    )


def _normalized_evidence(start: int, *, value: str) -> SourceEvidence:
    return SourceEvidence(
        start=start,
        end=start + len(value),
        match_kind=SourceMatchKind.NORMALIZED_EXACT,
        similarity=1.0,
    )


def _matching_evidence(
    candidate: str,
    *,
    value: str,
    role: SourceValueRoleLiteral,
    start: int,
    end: int,
) -> SourceEvidence | None:
    if role == SourceValueRole.PREDICTION:
        comparison = compare_values(value, ground_truth=candidate)
    else:
        comparison = compare_values(candidate, ground_truth=value)
    if comparison.result != MATCH:
        return None
    match_kind = SourceMatchKind.NORMALIZED_EXACT if comparison.normalized_exact else SourceMatchKind.FUZZY
    return SourceEvidence(start=start, end=end, match_kind=match_kind, similarity=comparison.similarity)
