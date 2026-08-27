import os
from typing import List

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from src.evaluation.models import PIIItem

MODEL = "gpt-4o-mini-2024-07-18"
DEFAULT_SEED = 42
SEED = int(os.environ.get("EVALUATION_SEED", str(DEFAULT_SEED)))
MAX_COMPLETION_TOKENS = 8192

PROMPT = """
Extract every real person mentioned in the document and group their personally identifiable
information. Include people in prose, email headers, signatures, and lists. Return each person once.

Use only values explicitly present in the document and preserve their spelling. Do not infer missing
values from an email address. Supported fields are first name, middle name, last name, age, birthdate,
phone, email, social-network identifier, location, and social security number. Leave a field empty
when it is absent. Do not return organizations or generic roles.

<document>
{text}
</document>
"""


class _Person(BaseModel):
    model_config = ConfigDict(extra="forbid")

    first_name: List[str] = Field(default_factory=list)
    middle_name: List[str] = Field(default_factory=list)
    last_name: List[str] = Field(default_factory=list)
    age: List[str] = Field(default_factory=list)
    birthdate: List[str] = Field(default_factory=list)
    phone: List[str] = Field(default_factory=list)
    email: List[str] = Field(default_factory=list)
    social_network_identifier: List[str] = Field(default_factory=list)
    location: List[str] = Field(default_factory=list)
    ssn: List[str] = Field(default_factory=list)

    def to_pii_item(self) -> PIIItem:
        return PIIItem(**{field: tuple(values) for field, values in self.model_dump().items()})


class _People(BaseModel):
    model_config = ConfigDict(extra="forbid")

    people: List[_Person] = Field(default_factory=list)


def extract_pii(text: str) -> List[PIIItem]:
    _validate_text(text)
    if not text.strip():
        return []
    completion = _extract_people(text)
    return [person.to_pii_item() for person in completion.people]


def _validate_text(text: str) -> None:
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")


def _has_candidate_content(text: str) -> bool:
    return any(character.isalnum() for character in text)


def _extract_people(text: str) -> _People:
    completion = OpenAI(max_retries=2, timeout=300.0).chat.completions.parse(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT.format(text=text)}],
        response_format=_People,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        seed=SEED,
        temperature=0,
    )
    message = completion.choices[0].message
    if message.refusal:
        raise RuntimeError(f"OpenAI refused PII extraction: {message.refusal}")
    if message.parsed is None:
        raise RuntimeError("OpenAI returned no parsed PII response")
    return message.parsed
