import os
import subprocess
import sys

import pytest

from solution import extract_pii


def test_empty_text_returns_without_openai_credentials():
    assert extract_pii(" \n\t") == []


def test_non_string_text_fails_fast():
    with pytest.raises(TypeError, match="text must be str, got NoneType"):
        extract_pii(None)


@pytest.mark.parametrize(("environment_seed", "expected"), [(None, "42"), ("4", "4")])
def test_solution_uses_the_evaluator_seed_with_blind_fallback(environment_seed, expected):
    environment = dict(os.environ)
    if environment_seed is None:
        environment.pop("EVALUATION_SEED", None)
    else:
        environment["EVALUATION_SEED"] = environment_seed

    completed = subprocess.run(
        [sys.executable, "-c", "import solution; print(solution.SEED)"],
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )

    assert completed.stdout.strip() == expected
