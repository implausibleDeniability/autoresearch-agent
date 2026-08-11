import time

from src.evaluation.matching import match_values, similarity_length_bounds
from src.evaluation.source_evidence import SourceEvidence, SourceMatchKind, select_source_evidence
from src.evaluation.source_matching import SourceTextMatcher, SourceValueRole

PREDICTION = SourceValueRole.PREDICTION
GROUND_TRUTH = SourceValueRole.GROUND_TRUTH


def test_source_evidence_distinguishes_raw_normalized_and_fuzzy_matches():
    text = "Christine CHRISTINE Chris"

    evidence = SourceTextMatcher(text).find("Christine", role=GROUND_TRUTH).evidence

    assert [(item.start, item.end, item.match_kind) for item in evidence] == [
        (0, 9, SourceMatchKind.RAW_EXACT),
        (10, 19, SourceMatchKind.NORMALIZED_EXACT),
        (20, 25, SourceMatchKind.FUZZY),
    ]


def test_raw_evidence_preserves_literal_inside_surrounding_punctuation():
    text = "Email (john@example.com), today."

    evidence = SourceTextMatcher(text).find("john@example.com", role=GROUND_TRUTH).evidence

    assert evidence == (
        SourceEvidence(start=7, end=23, match_kind=SourceMatchKind.RAW_EXACT, similarity=1.0),
    )
    assert text[evidence[0].start : evidence[0].end] == "john@example.com"


def test_normalized_evidence_preserves_case_only_literal_inside_punctuation():
    text = "Email (JOHN@EXAMPLE.COM), today."

    evidence = SourceTextMatcher(text).find("john@example.com", role=GROUND_TRUTH).evidence

    assert evidence == (
        SourceEvidence(
            start=7,
            end=23,
            match_kind=SourceMatchKind.NORMALIZED_EXACT,
            similarity=1.0,
        ),
    )


def test_fuzzy_evidence_spans_whitespace_separated_phone_tokens():
    text = "Call 513 558 7949 now"

    evidence = SourceTextMatcher(text).find("513-558-7949", role=GROUND_TRUTH).evidence

    assert len(evidence) == 1
    assert text[evidence[0].start : evidence[0].end] == "513 558 7949"
    assert evidence[0].match_kind == SourceMatchKind.FUZZY


def test_short_values_keep_explicit_raw_substring_evidence_without_fuzzy_matches():
    evidence = SourceTextMatcher("Alice Al").find("Al", role=GROUND_TRUTH).evidence

    assert [(item.start, item.end, item.match_kind) for item in evidence] == [
        (0, 2, SourceMatchKind.RAW_EXACT),
        (6, 8, SourceMatchKind.RAW_EXACT),
    ]


def test_unicode_fuzzy_evidence_uses_original_source_offsets():
    text = "İstanbul office"

    evidence = SourceTextMatcher(text).find("istanbul", role=GROUND_TRUTH).evidence

    assert len(evidence) == 1
    assert text[evidence[0].start : evidence[0].end] == "İstanbul"
    assert evidence[0].match_kind == SourceMatchKind.FUZZY


def test_fuzzy_overlap_selection_maximizes_distinct_evidence_count():
    candidates = (
        SourceEvidence(start=0, end=10, match_kind=SourceMatchKind.FUZZY, similarity=0.99),
        SourceEvidence(start=0, end=4, match_kind=SourceMatchKind.FUZZY, similarity=0.70),
        SourceEvidence(start=5, end=9, match_kind=SourceMatchKind.FUZZY, similarity=0.70),
    )

    selected = select_source_evidence(candidates)

    assert [(item.start, item.end) for item in selected] == [(0, 4), (5, 9)]


def test_source_matcher_caches_each_raw_value_without_merging_variants():
    matcher = SourceTextMatcher("Smith smith Smith.")

    first = matcher.find("Smith", role=PREDICTION)

    assert matcher.find("Smith", role=PREDICTION) is first
    assert matcher.find("Smith", role=GROUND_TRUTH) is not first
    assert matcher.find("smith", role=PREDICTION) is not first
    assert matcher.find("Smith.", role=PREDICTION) is not first


def test_similarity_length_bounds_include_threshold_edge_lengths():
    assert similarity_length_bounds("x" * 20) == (10, 41)


def test_exact_value_assignment_remains_reserved_before_fuzzy_assignment():
    matches = match_values(("Jon", "John"), ground_truth=("John",))

    assert matches == {1: 0}


def test_repetitive_source_matching_finishes_within_a_generous_bound():
    matcher = SourceTextMatcher("noise " * 5_000)

    started_at = time.monotonic()
    match_result = matcher.find("john@example.com", role=GROUND_TRUTH)

    assert match_result.evidence == ()
    assert time.monotonic() - started_at < 3


def test_fuzzy_evidence_preserves_evaluator_comparison_direction():
    matcher = SourceTextMatcher("bca")

    prediction_result = matcher.find("aba", role=PREDICTION)
    ground_truth_result = matcher.find("aba", role=GROUND_TRUTH)

    assert prediction_result.evidence == ()
    assert ground_truth_result.evidence[0].match_kind == SourceMatchKind.FUZZY


def test_fuzzy_evidence_trims_surrounding_punctuation():
    texts = ("Name: (Jonh), confirmed", "Name:Jonh,", "Name:(Jonh)")

    for text in texts:
        evidence = SourceTextMatcher(text).find("John", role=GROUND_TRUTH).evidence

        assert len(evidence) == 1
        assert text[evidence[0].start : evidence[0].end] == "Jonh"
        assert evidence[0].match_kind == SourceMatchKind.FUZZY


def test_fuzzy_evidence_keeps_normalization_safe_trailing_period_candidates():
    text = "Jonh....."

    match_result = SourceTextMatcher(text).find("John", role=GROUND_TRUTH)

    assert match_result.fuzzy_search_complete is True
    assert text[match_result.evidence[0].start : match_result.evidence[0].end] == text


def test_oversized_fuzzy_search_is_skipped_explicitly_and_quickly():
    matcher = SourceTextMatcher("x " * 1_000)
    value = " ".join("y" for _ in range(100))

    started_at = time.monotonic()
    match_result = matcher.find(value, role=PREDICTION)

    assert match_result.evidence == ()
    assert match_result.fuzzy_search_complete is False
    assert time.monotonic() - started_at < 1


def test_dense_fuzzy_candidate_search_stops_at_the_work_budget():
    matcher = SourceTextMatcher("x " * 1_000)
    value = " ".join("y" for _ in range(30))

    started_at = time.monotonic()
    match_result = matcher.find(value, role=PREDICTION)

    assert match_result.evidence == ()
    assert match_result.fuzzy_search_complete is False
    assert time.monotonic() - started_at < 1


def test_impossibly_long_value_skips_candidate_enumeration_quickly():
    matcher = SourceTextMatcher("a " * 1_000)

    started_at = time.monotonic()
    match_result = matcher.find("b" * 5_000, role=PREDICTION)

    assert match_result.evidence == ()
    assert match_result.fuzzy_search_complete is True
    assert time.monotonic() - started_at < 1


def test_repetitive_raw_evidence_selection_scales_linearly_in_memory():
    matcher = SourceTextMatcher("a" * 10_000)

    started_at = time.monotonic()
    match_result = matcher.find("a", role=GROUND_TRUTH)

    assert len(match_result.evidence) == 10_000
    assert time.monotonic() - started_at < 1


def test_raw_and_normalized_overlap_filtering_avoids_quadratic_scans():
    matcher = SourceTextMatcher("aA" * 10_000)

    started_at = time.monotonic()
    match_result = matcher.find("a", role=GROUND_TRUTH)

    assert len(match_result.evidence) == 20_000
    assert time.monotonic() - started_at < 1
