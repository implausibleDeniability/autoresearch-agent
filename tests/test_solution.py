import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from solution import extract_pii
import solution


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


def test_openai_client_is_constructed_once_during_concurrent_first_use(monkeypatch):
    clients = []

    def construct_client(**_kwargs):
        time.sleep(0.01)
        client = object()
        clients.append(client)
        return client

    monkeypatch.setattr(solution, "_CLIENT", None)
    monkeypatch.setattr(solution, "OpenAI", construct_client)

    with ThreadPoolExecutor(max_workers=50) as executor:
        results = tuple(executor.map(lambda _index: solution._openai_client(), range(50)))

    assert len(clients) == 1
    assert all(result is clients[0] for result in results)
