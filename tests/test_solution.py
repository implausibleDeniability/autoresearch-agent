import pytest

from solution import extract_pii


def test_empty_text_returns_without_openai_credentials():
    assert extract_pii(" \n\t") == []


def test_non_string_text_fails_fast():
    with pytest.raises(TypeError, match="text must be str, got NoneType"):
        extract_pii(None)
