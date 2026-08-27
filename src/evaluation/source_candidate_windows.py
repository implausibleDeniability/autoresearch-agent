import unicodedata
from bisect import bisect_left
from typing import Dict, Iterator, List, Sequence, Tuple

TextSpan = Tuple[int, int]
WindowCacheKey = Tuple[int, int, int, int]


class CandidateWindowBudgetExceeded(RuntimeError):
    pass


class CandidateWindowIndex:
    def __init__(self, text: str, *, tokens: Sequence[TextSpan]) -> None:
        boundary_punctuation = tuple(_is_boundary_punctuation(character) for character in text)
        trim_characters = tuple(
            character.isspace() or boundary for character, boundary in zip(text, boundary_punctuation)
        )
        whitespace = tuple(character.isspace() for character in text)
        self._text = text
        self._tokens = tokens
        self._punctuation_indexes = tuple(
            index for index, boundary in enumerate(boundary_punctuation) if boundary
        )
        self._next_trimmed_content = _next_content_indexes(trim_characters)
        self._previous_trimmed_content = _previous_content_ends(trim_characters)
        self._next_nonspace = _next_content_indexes(whitespace)
        self._previous_nonspace = _previous_content_ends(whitespace)
        self._lower_length_prefix = _lower_length_prefix(text)
        self._cache: Dict[WindowCacheKey, List[TextSpan] | None] = {}

    def find(
        self,
        *,
        minimum_length: int,
        maximum_length: int,
        maximum_windows: int,
        maximum_examined: int,
    ) -> List[TextSpan] | None:
        key = minimum_length, maximum_length, maximum_windows, maximum_examined
        if key not in self._cache:
            self._cache[key] = self._collect(
                minimum_length=minimum_length,
                maximum_length=maximum_length,
                maximum_windows=maximum_windows,
                maximum_examined=maximum_examined,
            )
        return self._cache[key]

    def normalized_length(self, start: int, *, end: int) -> int:
        left = min(self._next_nonspace[start], end)
        right = max(left, self._previous_nonspace[end])
        while right > left and self._text[right - 1] == ".":
            right -= 1
        return self._lower_length_prefix[right] - self._lower_length_prefix[left]

    def _collect(
        self,
        *,
        minimum_length: int,
        maximum_length: int,
        maximum_windows: int,
        maximum_examined: int,
    ) -> List[TextSpan] | None:
        windows: Dict[TextSpan, None] = {}
        try:
            for span in self._candidate_windows(
                minimum_length=minimum_length,
                maximum_length=maximum_length,
                maximum_examined=maximum_examined,
            ):
                windows[span] = None
                if len(windows) > maximum_windows:
                    return None
        except CandidateWindowBudgetExceeded:
            return None
        return list(windows)

    def _candidate_windows(
        self,
        *,
        minimum_length: int,
        maximum_length: int,
        maximum_examined: int,
    ) -> Iterator[TextSpan]:
        examined = 0
        for start_index, (start, first_token_end) in enumerate(self._tokens):
            growth_start = self._growth_start(start, end=first_token_end)
            for end_index in range(start_index, len(self._tokens)):
                examined += 1
                if examined > maximum_examined:
                    raise CandidateWindowBudgetExceeded
                end = self._tokens[end_index][1]
                _, growth_end = self._trim_edges(growth_start, end=end)
                if self.normalized_length(growth_start, end=growth_end) > maximum_length:
                    break
                yield from self._eligible_spans(
                    start,
                    end=end,
                    minimum_length=minimum_length,
                    maximum_length=maximum_length,
                )

    def _eligible_spans(
        self,
        start: int,
        *,
        end: int,
        minimum_length: int,
        maximum_length: int,
    ) -> Iterator[TextSpan]:
        for left, right in self._candidate_spans(start, end=end):
            length = self.normalized_length(left, end=right)
            if minimum_length <= length <= maximum_length:
                yield left, right

    def _candidate_spans(self, start: int, *, end: int) -> Tuple[TextSpan, ...]:
        spans = [(start, end)]
        trimmed = self._trim_edges(start, end=end)
        if trimmed != (start, end):
            spans.append(trimmed)
        spans.extend(self._punctuation_segments(start, end=end))
        return tuple(dict.fromkeys(span for span in spans if span[0] < span[1]))

    def _punctuation_segments(self, start: int, *, end: int) -> List[TextSpan]:
        spans = []
        segment_start = start
        punctuation_index = bisect_left(self._punctuation_indexes, start)
        for punctuation in self._punctuation_indexes[punctuation_index:]:
            if punctuation >= end:
                break
            spans.append(self._trim_edges(segment_start, end=punctuation))
            segment_start = punctuation + 1
        spans.append(self._trim_edges(segment_start, end=end))
        return spans

    def _trim_edges(self, start: int, *, end: int) -> TextSpan:
        left = min(self._next_trimmed_content[start], end)
        return left, max(left, self._previous_trimmed_content[end])

    def _growth_start(self, start: int, *, end: int) -> int:
        left, right = self._trim_edges(start, end=end)
        punctuation_index = bisect_left(self._punctuation_indexes, right) - 1
        if punctuation_index >= 0 and self._punctuation_indexes[punctuation_index] >= left:
            return self._punctuation_indexes[punctuation_index] + 1
        return left


def _next_content_indexes(excluded: Sequence[bool]) -> Tuple[int, ...]:
    indexes = [len(excluded)] * (len(excluded) + 1)
    next_index = len(excluded)
    for index in range(len(excluded) - 1, -1, -1):
        if not excluded[index]:
            next_index = index
        indexes[index] = next_index
    return tuple(indexes)


def _previous_content_ends(excluded: Sequence[bool]) -> Tuple[int, ...]:
    indexes = [0] * (len(excluded) + 1)
    previous_end = 0
    for end in range(1, len(excluded) + 1):
        if not excluded[end - 1]:
            previous_end = end
        indexes[end] = previous_end
    return tuple(indexes)


def _lower_length_prefix(text: str) -> Tuple[int, ...]:
    lengths = [0]
    for character in text:
        lengths.append(lengths[-1] + len(character.lower()))
    return tuple(lengths)


def _is_boundary_punctuation(character: str) -> bool:
    return character in ":;," or unicodedata.category(character) in {"Ps", "Pe", "Pi", "Pf"}
