import pytest
import tiktoken
from openai.types.chat import ChatCompletion

from solution import (
    INNER_CHUNK_SIZE,
    INNER_SEPARATORS,
    OUTPUT_END_TAG,
    OUTPUT_START_TAG,
    STRUCTURING_MODEL,
    _StructuredPerson,
    _completion_text,
    _drop_structured_candidates,
    _merge_people,
    _normalize_people,
    _parse_boolean_tag,
    _parse_candidate_response,
    _split_text,
    extract_pii,
)
from pii_item import PIIItem


def test_empty_text_returns_without_openai_credentials():
    assert extract_pii(" \n\t") == []


def test_non_string_text_fails_fast():
    with pytest.raises(TypeError, match="text must be str, got NoneType"):
        extract_pii(None)


def test_candidate_response_parses_tokens_and_empty_output():
    response = f"analysis\n{OUTPUT_START_TAG}\nJohn; john@example.com\n{OUTPUT_END_TAG}"

    assert _parse_candidate_response(response, stage="test") == ["John", "john@example.com"]
    assert _parse_candidate_response(f"{OUTPUT_START_TAG}{OUTPUT_END_TAG}", stage="test") == []


def test_candidate_response_requires_both_tags():
    with pytest.raises(RuntimeError, match="missing <personal_information_tokens>"):
        _parse_candidate_response("John", stage="candidate test")


@pytest.mark.parametrize(
    ("value", "expected"),
    [("true", True), ("TRUE", True), ("false", False)],
)
def test_boolean_response_parsing(value: str, expected: bool):
    response = f"<describes_person>{value}</describes_person>"

    assert _parse_boolean_tag(response, tag="describes_person", stage="validator test") is expected


def test_boolean_response_rejects_unknown_value():
    response = "<describes_person>maybe</describes_person>"

    with pytest.raises(RuntimeError, match="invalid describes_person: 'maybe'"):
        _parse_boolean_tag(response, tag="describes_person", stage="validator test")


def test_completion_text_returns_content_and_rejects_empty_or_refused_messages():
    assert _completion_text(_completion(content="answer"), stage="test") == "answer"

    with pytest.raises(RuntimeError, match="refused the request: unsafe"):
        _completion_text(_completion(refusal="unsafe"), stage="test")
    with pytest.raises(RuntimeError, match="returned empty content"):
        _completion_text(_completion(), stage="test")


def test_token_chunking_respects_limit():
    encoding = tiktoken.encoding_for_model(STRUCTURING_MODEL)
    text = " ".join(f"word{index}" for index in range(250))

    chunks = _split_text(
        text,
        chunk_size=INNER_CHUNK_SIZE,
        separators=INNER_SEPARATORS,
        encoding=encoding,
    )

    assert len(chunks) > 1
    assert max(len(encoding.encode(chunk)) for chunk in chunks) <= INNER_CHUNK_SIZE


def test_normalization_maps_every_supported_field():
    person = _StructuredPerson(
        first_name="MICHAEL",
        middle_name="O.",
        last_name="MCCLELLAN",
        age=" 72 ",
        birthdate="1954-01-01",
        phone="513-558-7949",
        email="ROGER.O.MCCLELLAN@ATT.NET",
        social_network_identifier="@roger",
        location="Cincinnati",
        ssn="123-45-6789",
    )
    text = "Michael O. McClellan, 72, lives in Cincinnati. Email ROGER.O.MCCLELLAN@ATT.NET."

    result = _normalize_people([person], text=text)

    assert result == [
        PIIItem(
            first_name=("Michael",),
            middle_name=("O",),
            last_name=("Mcclellan",),
            age=("72",),
            birthdate=("1954-01-01",),
            phone=("513-558-7949",),
            email=("roger.o.mcclellan@att.net",),
            social_network_identifier=("@roger",),
            location=("Cincinnati",),
            ssn=("123-45-6789",),
        )
    ]


def test_normalization_splits_middle_initial_and_drops_invalid_email():
    person = _StructuredPerson(first_name="Robert A.", last_name="Smith", email="not-an-email")

    result = _normalize_people([person], text="Robert A. Smith uses not-an-email internally.")

    assert result == [PIIItem(first_name=("Robert",), middle_name=("A",), last_name=("Smith",))]


def test_normalization_splits_two_initials():
    person = _StructuredPerson(first_name="JS", last_name="Smith")

    result = _normalize_people([person], text="J S Smith")

    assert result == [PIIItem(first_name=("J",), middle_name=("S",), last_name=("Smith",))]


def test_normalization_drops_unnamed_and_hallucinated_people():
    people = [
        _StructuredPerson(phone="555-0100"),
        _StructuredPerson(first_name="Invented", last_name="Person"),
    ]

    assert _normalize_people(people, text="No people are named here.") == []


def test_structured_values_are_removed_from_second_pass_candidates():
    candidates = ["John", "Smith", "john@example.com", "unassigned"]
    people = [_StructuredPerson(first_name="John", last_name="Smith", email="john@example.com")]

    assert _drop_structured_candidates(candidates, people=people) == ["unassigned"]


def test_compatible_people_merge_in_stable_order_and_keep_longer_values():
    people = [
        PIIItem(first_name=("John",), phone=("555-0100",)),
        PIIItem(first_name=("John",), last_name=("Smith",), location=("NY",)),
        PIIItem(last_name=("Smith",), email=("john@example.com",), location=("New York",)),
    ]

    result = _merge_people(people)

    assert result == [
        PIIItem(
            first_name=("John",),
            last_name=("Smith",),
            phone=("555-0100",),
            email=("john@example.com",),
            location=("NY", "New York"),
        )
    ]


def test_conflicting_core_fields_remain_separate():
    people = [
        PIIItem(first_name=("John",), last_name=("Smith",)),
        PIIItem(first_name=("John",), last_name=("Jones",)),
    ]

    assert _merge_people(people) == people


def test_merge_deduplicates_case_insensitively():
    people = [
        PIIItem(first_name=("John",), email=("JOHN@example.com",)),
        PIIItem(first_name=("john",), email=("john@example.com",)),
    ]

    assert _merge_people(people) == [PIIItem(first_name=("John",), email=("JOHN@example.com",))]


def test_merge_prefers_longer_containing_value():
    people = [
        PIIItem(first_name=("John",), location=("York",)),
        PIIItem(first_name=("John",), location=("New York",)),
    ]

    assert _merge_people(people) == [PIIItem(first_name=("John",), location=("New York",))]


def _completion(*, content: str = "", refusal: str = "") -> ChatCompletion:
    return ChatCompletion.model_validate(
        {
            "id": "test",
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "content": content or None,
                        "refusal": refusal or None,
                        "role": "assistant",
                    },
                }
            ],
            "created": 0,
            "model": "test",
            "object": "chat.completion",
        }
    )
