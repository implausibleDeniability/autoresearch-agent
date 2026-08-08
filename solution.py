import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import List, Optional, Sequence, Tuple

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import APIError, OpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, ConfigDict, Field

from src.evaluation.models import PIIItem, PIIValues

DETECTION_MODEL = "gpt-4o-mini-2024-07-18"
STRUCTURING_MODEL = "gpt-4o-2024-08-06"
VALIDATION_MODEL = "gpt-4o-mini-2024-07-18"
TOP_P = 0.7
OUTER_CHUNK_SIZE = 500
INNER_CHUNK_SIZE = 100
STRUCTURING_PASSES = 2
MAX_WORKERS = 4
DETECTION_ATTEMPTS = 2
MIN_SUBSTRING_LENGTH = 4
OUTPUT_START_TAG = "<personal_information_tokens>"
OUTPUT_END_TAG = "</personal_information_tokens>"

OUTER_SEPARATORS = ("\n\n", "\n", ". ", "! ", "? ", " ", "")
INNER_SEPARATORS = (".\n", "!\n", "?\n", ". ", "! ", "? ", " ", "")
NAME_FIELDS = ("first_name", "middle_name", "last_name")


class MalformedModelResponseError(RuntimeError):
    pass


DETECTION_PROMPT = f"""
You will be given a small text. Analyze it in two steps.

<text>
{{text}}
</text>

First, identify tokens that are personal information. Supported types are first name, middle name,
last name, age, birthdate, phone, email, social-network identifier, location, and social security
number. Treat an entire email address as one token.

Second, write every personal-information token separated by a semicolon and a space inside these
tags:

{OUTPUT_START_TAG}
[Token 1]; [Token 2]; [Token 3]
{OUTPUT_END_TAG}

If the text contains no personal-information tokens, leave the tagged section empty. Include no
personal-information token that is absent from the text.
"""

STRUCTURING_PROMPT = """
Extract people and their personally identifiable information from the supplied text. The known
personal-information tokens are hints extracted from the same text. Use context to assign tokens to
the correct person. A person can be identified by a first name, last name, email, or a combination.

Supported fields are first_name, middle_name, last_name, age, birthdate, phone, email,
social_network_identifier, location, and ssn. Do not invent values. Preserve values as written in
the text. If an entity cannot be assigned confidently, omit it.

<known_personal_information>
{known_personal_information}
</known_personal_information>

<text>
{text}
</text>

Explain how you identified each person and associated the fields. Put the final grouping inside
<output> tags.
"""

STRUCTURING_FORMAT_PROMPT = """
Convert the people inside the <output> tags below into the required structured response. Return one
item per person. Use null for unsupported or absent fields. Do not add information that is absent
from the supplied output.

<analysis_output>
{analysis}
</analysis_output>
"""

VALIDATION_PROMPT = """
Validate whether the supplied PII describes a real person in the supplied context and whether the
name components are in the correct order.

<pii_info>
{pii_info}
</pii_info>

<context>
{context}
</context>

Return exactly:
<validation>
<describes_person>true or false</describes_person>
<has_mixed_names>true or false</has_mixed_names>
</validation>
"""


class _StructuredPerson(BaseModel):
    model_config = ConfigDict(extra="forbid")

    first_name: Optional[str] = None
    middle_name: Optional[str] = None
    last_name: Optional[str] = None
    age: Optional[str] = None
    birthdate: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    social_network_identifier: Optional[str] = None
    location: Optional[str] = None
    ssn: Optional[str] = None


class _StructuredPeople(BaseModel):
    model_config = ConfigDict(extra="forbid")

    people: List[_StructuredPerson] = Field(default_factory=list)


class _BaselineExtractor:
    def __init__(self) -> None:
        self._client = OpenAI(max_retries=2, timeout=300.0)
        self._encoding = tiktoken.encoding_for_model(STRUCTURING_MODEL)

    def extract(self, text: str) -> List[PIIItem]:
        chunks = _split_text(
            text, chunk_size=OUTER_CHUNK_SIZE, separators=OUTER_SEPARATORS, encoding=self._encoding
        )
        extracted = [person for chunk_people in self._extract_chunks(chunks) for person in chunk_people]
        return _merge_people(extracted)

    def _extract_chunks(self, chunks: Sequence[str]) -> List[List[PIIItem]]:
        indexed_chunks = list(enumerate(chunks))
        if len(indexed_chunks) == 1:
            return [self._extract_indexed_chunk(indexed_chunks[0])]
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(indexed_chunks))) as executor:
            return list(executor.map(self._extract_indexed_chunk, indexed_chunks))

    def _extract_indexed_chunk(self, indexed_chunk: Tuple[int, str]) -> List[PIIItem]:
        chunk_index, text = indexed_chunk
        return self._extract_chunk(text, chunk_index=chunk_index)

    def _extract_chunk(self, text: str, *, chunk_index: int) -> List[PIIItem]:
        candidates = self._detect_candidates(text, chunk_index=chunk_index)
        if not candidates:
            return []
        structured = self._structure_people(text, candidates=candidates, chunk_index=chunk_index)
        normalized = _normalize_people(structured, text=text)
        merged = _merge_people(normalized)
        return self._validate_people(merged, context=text, chunk_index=chunk_index)

    def _detect_candidates(self, text: str, *, chunk_index: int) -> List[str]:
        chunks = _split_text(
            text, chunk_size=INNER_CHUNK_SIZE, separators=INNER_SEPARATORS, encoding=self._encoding
        )
        candidates = []
        for inner_index, chunk in enumerate(chunks):
            if not _has_candidate_content(chunk):
                continue
            candidates.extend(
                self._detect_chunk_candidates(chunk, chunk_index=chunk_index, inner_index=inner_index)
            )
        return list(dict.fromkeys(candidates))

    def _detect_chunk_candidates(self, text: str, *, chunk_index: int, inner_index: int) -> List[str]:
        stage = f"candidate detection at outer chunk {chunk_index}, inner chunk {inner_index}"
        attempt = 0
        while True:
            response = self._complete(
                DETECTION_PROMPT.format(text=text), model=DETECTION_MODEL, stage=stage, max_tokens=1000
            )
            try:
                return _parse_candidate_response(response, stage=stage)
            except MalformedModelResponseError:
                attempt += 1
                if attempt == DETECTION_ATTEMPTS:
                    raise

    def _structure_people(
        self,
        text: str,
        *,
        candidates: Sequence[str],
        chunk_index: int,
    ) -> List[_StructuredPerson]:
        remaining = list(candidates)
        people = []
        for pass_index in range(STRUCTURING_PASSES):
            structured = self._run_structuring_pass(
                text,
                candidates=remaining,
                chunk_index=chunk_index,
                pass_index=pass_index,
            )
            people.extend(structured)
            remaining = _drop_structured_candidates(remaining, people=people)
        return people

    def _run_structuring_pass(
        self,
        text: str,
        *,
        candidates: Sequence[str],
        chunk_index: int,
        pass_index: int,
    ) -> List[_StructuredPerson]:
        stage = f"structuring analysis at outer chunk {chunk_index}, pass {pass_index}"
        prompt = STRUCTURING_PROMPT.format(known_personal_information="; ".join(candidates), text=text)
        analysis = self._complete(prompt, model=STRUCTURING_MODEL, stage=stage, max_tokens=4096)
        return self._parse_structured_people(analysis, chunk_index=chunk_index, pass_index=pass_index)

    def _parse_structured_people(
        self,
        analysis: str,
        *,
        chunk_index: int,
        pass_index: int,
    ) -> List[_StructuredPerson]:
        stage = f"structured response at outer chunk {chunk_index}, pass {pass_index}"
        try:
            completion = self._client.chat.completions.parse(
                model=STRUCTURING_MODEL,
                messages=[{"role": "user", "content": STRUCTURING_FORMAT_PROMPT.format(analysis=analysis)}],
                response_format=_StructuredPeople,
                max_completion_tokens=4096,
                top_p=TOP_P,
            )
        except APIError as error:
            raise RuntimeError(f"OpenAI {stage} failed") from error
        message = completion.choices[0].message
        if message.refusal:
            raise RuntimeError(f"OpenAI {stage} refused the request: {message.refusal}")
        if message.parsed is None:
            raise RuntimeError(f"OpenAI {stage} returned no parsed response")
        return message.parsed.people

    def _validate_people(
        self,
        people: Sequence[PIIItem],
        *,
        context: str,
        chunk_index: int,
    ) -> List[PIIItem]:
        return [
            person
            for person_index, person in enumerate(people)
            if self._validate_person(
                person,
                context=context,
                chunk_index=chunk_index,
                person_index=person_index,
            )
        ]

    def _validate_person(
        self,
        person: PIIItem,
        *,
        context: str,
        chunk_index: int,
        person_index: int,
    ) -> bool:
        stage = f"person validation at outer chunk {chunk_index}, person {person_index}"
        prompt = VALIDATION_PROMPT.format(pii_info=_format_validation_person(person), context=context)
        response = self._complete(prompt, model=VALIDATION_MODEL, stage=stage, max_tokens=1024)
        describes_person = _parse_boolean_tag(response, tag="describes_person", stage=stage)
        _parse_boolean_tag(response, tag="has_mixed_names", stage=stage)
        return describes_person

    def _complete(self, prompt: str, *, model: str, stage: str, max_tokens: int) -> str:
        try:
            completion = self._client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_tokens,
                top_p=TOP_P,
            )
        except APIError as error:
            raise RuntimeError(f"OpenAI {stage} failed with model {model}") from error
        return _completion_text(completion, stage=stage)


def extract_pii(text: str) -> List[PIIItem]:
    _validate_text(text)
    if not text.strip():
        return []
    return _BaselineExtractor().extract(text)


def _validate_text(text: str) -> None:
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")


def _split_text(
    text: str,
    *,
    chunk_size: int,
    separators: Sequence[str],
    encoding: tiktoken.Encoding,
) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        separators=list(separators),
        chunk_size=chunk_size,
        chunk_overlap=0,
        keep_separator=True,
        length_function=lambda value: len(encoding.encode(value)),
    )
    chunks = splitter.split_text(text)
    return chunks or [text]


def _completion_text(completion: ChatCompletion, *, stage: str) -> str:
    message = completion.choices[0].message
    if message.refusal:
        raise RuntimeError(f"OpenAI {stage} refused the request: {message.refusal}")
    if not message.content:
        raise RuntimeError(f"OpenAI {stage} returned empty content")
    return message.content


def _parse_candidate_response(response: str, *, stage: str) -> List[str]:
    content = _tag_content(response, start_tag=OUTPUT_START_TAG, end_tag=OUTPUT_END_TAG, stage=stage)
    return [token.strip() for token in content.split(";") if token.strip()]


def _parse_boolean_tag(response: str, *, tag: str, stage: str) -> bool:
    value = _tag_content(response, start_tag=f"<{tag}>", end_tag=f"</{tag}>", stage=stage).lower()
    if value not in {"true", "false"}:
        raise RuntimeError(f"OpenAI {stage} returned invalid {tag}: {value!r}")
    return value == "true"


def _tag_content(response: str, *, start_tag: str, end_tag: str, stage: str) -> str:
    start = response.find(start_tag)
    end = response.find(end_tag, start + len(start_tag))
    if start == -1 or end == -1:
        raise MalformedModelResponseError(f"OpenAI {stage} response is missing {start_tag} or {end_tag}")
    return response[start + len(start_tag) : end].strip()


def _has_candidate_content(text: str) -> bool:
    return any(character.isalnum() for character in text)


def _drop_structured_candidates(
    candidates: Sequence[str],
    *,
    people: Sequence[_StructuredPerson],
) -> List[str]:
    extracted = {value for person in people for value in person.model_dump().values() if value}
    return [candidate for candidate in candidates if candidate not in extracted]


def _normalize_people(people: Sequence[_StructuredPerson], *, text: str) -> List[PIIItem]:
    normalized = [_normalize_person(person) for person in people]
    return [
        person for person in normalized if _is_supported_person(person) and _is_grounded(person, text=text)
    ]


def _normalize_person(person: _StructuredPerson) -> PIIItem:
    values = {field_name: _clean_value(value) for field_name, value in person.model_dump().items()}
    first_name, middle_name = _split_first_name(values["first_name"], middle_name=values["middle_name"])
    first_name, middle_name, last_name = _normalize_name_parts(
        first_name,
        middle_name=middle_name,
        last_name=values["last_name"],
    )
    email = values["email"].lower() if values["email"] and "@" in values["email"] else None
    return PIIItem(
        first_name=_tuple_value(first_name),
        middle_name=_tuple_value(middle_name),
        last_name=_tuple_value(last_name),
        age=_tuple_value(values["age"]),
        birthdate=_tuple_value(values["birthdate"]),
        phone=_tuple_value(values["phone"]),
        email=_tuple_value(email),
        social_network_identifier=_tuple_value(values["social_network_identifier"]),
        location=_tuple_value(values["location"]),
        ssn=_tuple_value(values["ssn"]),
    )


def _clean_value(value: Optional[str]) -> Optional[str]:
    cleaned = value.strip() if value else ""
    return cleaned or None


def _split_first_name(
    first_name: Optional[str], *, middle_name: Optional[str]
) -> Tuple[Optional[str], Optional[str]]:
    if not first_name or middle_name:
        return first_name, middle_name
    match = re.fullmatch(r"([A-Za-z]{3,})\s+([A-Z])\.?", first_name)
    return (match.group(1), match.group(2)) if match else (first_name, middle_name)


def _normalize_name_parts(
    first_name: Optional[str],
    *,
    middle_name: Optional[str],
    last_name: Optional[str],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    first_name = _normalize_name(first_name)
    middle_name = _normalize_name(middle_name)
    last_name = _normalize_name(last_name)
    if first_name and not middle_name and len(first_name) == 2 and _is_initials(first_name):
        first_name, middle_name = first_name[0], first_name[1]
    return first_name, middle_name, last_name


def _normalize_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    name = name.rstrip(".") if _is_initials(name.rstrip(".")) else name
    if _is_initials(name):
        return name
    return (
        "-".join(part.capitalize() for part in name.split("-"))
        if " " not in name and name.isupper()
        else name
    )


def _is_initials(name: str) -> bool:
    return len(name) <= 2 and name.isupper()


def _tuple_value(value: Optional[str]) -> PIIValues:
    return (value,) if value else ()


def _is_supported_person(person: PIIItem) -> bool:
    return bool(person.first_name or person.last_name or person.email)


def _is_grounded(person: PIIItem, *, text: str) -> bool:
    lowered = text.lower()
    values = person.first_name + person.middle_name + person.last_name + person.email
    return all(value.lower() in lowered for value in values)


def _format_validation_person(person: PIIItem) -> str:
    values = asdict(person)
    return "\n".join(
        f"{field_name.replace('_', ' ')}: {', '.join(values[field_name])}"
        for field_name in NAME_FIELDS + ("email",)
        if values[field_name]
    )


def _merge_people(people: Sequence[PIIItem]) -> List[PIIItem]:
    merged = list(people)
    while _merge_first_compatible_pair(merged):
        pass
    return merged


def _merge_first_compatible_pair(people: List[PIIItem]) -> bool:
    for first_index, first in enumerate(people):
        for second_index in range(first_index + 1, len(people)):
            if _people_are_compatible(first, people[second_index]):
                people[first_index] = _merge_person(first, people.pop(second_index))
                return True
    return False


def _people_are_compatible(first: PIIItem, second: PIIItem) -> bool:
    field_pairs = list(zip(_core_values(first), _core_values(second)))
    has_exact_match = any(_values_match_exactly(left, right) for left, right in field_pairs)
    return has_exact_match and not any(_values_conflict(left, right) for left, right in field_pairs)


def _core_values(person: PIIItem) -> Tuple[PIIValues, PIIValues, PIIValues]:
    return person.first_name, person.last_name, person.email


def _values_match_exactly(first: Sequence[str], second: Sequence[str]) -> bool:
    return bool({_normalized(value) for value in first} & {_normalized(value) for value in second})


def _values_conflict(first: Sequence[str], second: Sequence[str]) -> bool:
    return bool(first and second) and not any(
        _values_are_compatible(left, right) for left in first for right in second
    )


def _values_are_compatible(first: str, second: str) -> bool:
    first_normalized = _normalized(first)
    second_normalized = _normalized(second)
    if first_normalized == second_normalized:
        return True
    shorter, longer = sorted((first_normalized, second_normalized), key=len)
    return len(shorter) >= MIN_SUBSTRING_LENGTH and shorter in longer


def _merge_person(first: PIIItem, second: PIIItem) -> PIIItem:
    first_values = asdict(first)
    second_values = asdict(second)
    merged = {
        field_name: _merge_values(first_values[field_name], second_values[field_name])
        for field_name in first_values
    }
    return PIIItem(**merged)


def _merge_values(first: Sequence[str], second: Sequence[str]) -> PIIValues:
    merged = list(first)
    for value in second:
        merged = _merge_value(merged, value=value)
    return tuple(merged)


def _merge_value(values: List[str], *, value: str) -> List[str]:
    for index, existing in enumerate(values):
        if _normalized(existing) == _normalized(value) or _normalized(value) in _normalized(existing):
            return values
        if _normalized(existing) in _normalized(value):
            values[index] = value
            return values
    values.append(value)
    return values


def _normalized(value: str) -> str:
    return value.lower().strip()
