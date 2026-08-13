import unicodedata
from typing import Iterator, Sequence, Tuple

from src.evaluation.matching import normalize_value

TextSpan = Tuple[int, int]


class CandidateWindowBudgetExceeded(RuntimeError):
    pass


def candidate_windows(
    text: str,
    *,
    tokens: Sequence[TextSpan],
    minimum_length: int,
    maximum_length: int,
    maximum_examined: int,
) -> Iterator[TextSpan]:
    examined = 0
    for start_index, (start, first_token_end) in enumerate(tokens):
        growth_start = _growth_start(text, start=start, end=first_token_end)
        for end_index in range(start_index, len(tokens)):
            examined += 1
            if examined > maximum_examined:
                raise CandidateWindowBudgetExceeded
            end = tokens[end_index][1]
            _, growth_end = _trim_edges(text, start=growth_start, end=end)
            if len(normalize_value(text[growth_start:growth_end])) > maximum_length:
                break
            spans = _candidate_spans(text, start=start, end=end)
            lengths = [len(normalize_value(text[left:right])) for left, right in spans]
            for span, length in zip(spans, lengths):
                if minimum_length <= length <= maximum_length:
                    yield span


def _candidate_spans(text: str, *, start: int, end: int) -> Tuple[TextSpan, ...]:
    spans = [(start, end)]
    trimmed = _trim_edges(text, start=start, end=end)
    if trimmed != (start, end):
        spans.append(trimmed)
    segment_start = start
    for index in range(start, end):
        if _is_boundary_punctuation(text[index]):
            spans.append(_trim_edges(text, start=segment_start, end=index))
            segment_start = index + 1
    spans.append(_trim_edges(text, start=segment_start, end=end))
    return tuple(dict.fromkeys(span for span in spans if span[0] < span[1]))


def _trim_edges(text: str, *, start: int, end: int) -> TextSpan:
    while start < end and (text[start].isspace() or _is_boundary_punctuation(text[start])):
        start += 1
    while end > start and (text[end - 1].isspace() or _is_boundary_punctuation(text[end - 1])):
        end -= 1
    return start, end


def _growth_start(text: str, *, start: int, end: int) -> int:
    trimmed_start, trimmed_end = _trim_edges(text, start=start, end=end)
    for index in range(trimmed_start, trimmed_end):
        if _is_boundary_punctuation(text[index]):
            trimmed_start = index + 1
    return trimmed_start


def _is_boundary_punctuation(character: str) -> bool:
    return character in ":;," or unicodedata.category(character) in {"Ps", "Pe", "Pi", "Pf"}
