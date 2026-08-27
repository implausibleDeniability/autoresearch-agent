import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Tuple
from urllib.parse import urlsplit, urlunsplit

SCORING_CONTRACT_VERSION = 1
SCORING_CONTRACT_PATHS = (
    Path("src/evaluation/results.py"),
    Path("src/evaluation/metrics.py"),
    Path("src/evaluation/trace.py"),
    Path("src/evaluation/matching.py"),
    Path("src/evaluation/models.py"),
)
RUNTIME_PACKAGES = ("httpx", "openai", "pydantic", "tiktoken")
SOURCE_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class FinalProvenance:
    repository_available: bool
    repository_commit: str
    repository_clean_start: bool
    repository_clean_end: bool
    solution_matches_head_start: bool
    solution_matches_snapshot_end: bool
    repository_commit_unchanged: bool
    solution_snapshot_fingerprint: str
    scoring_contract_fingerprint: str
    dataset_fingerprint: str
    runtime_fingerprint: str
    promotion_capable: bool
    invalidation_reasons: Tuple[str, ...]


@dataclass(frozen=True)
class EvidenceContext:
    repository_available: bool
    repository_commit: str
    repository_clean_start: bool
    solution_matches_head_start: bool
    solution_snapshot_fingerprint: str
    scoring_contract_fingerprint: str
    dataset_fingerprint: str
    runtime_fingerprint: str
    solution_module: str
    snapshot_directory: Path
    dataset: str
    started_monotonic: float

    @property
    def duration_seconds(self) -> float:
        return max(time.monotonic() - self.started_monotonic, 0.0)

    def running_provenance(self) -> FinalProvenance:
        return FinalProvenance(
            repository_available=self.repository_available,
            repository_commit=self.repository_commit,
            repository_clean_start=self.repository_clean_start,
            repository_clean_end=False,
            solution_matches_head_start=self.solution_matches_head_start,
            solution_matches_snapshot_end=False,
            repository_commit_unchanged=False,
            solution_snapshot_fingerprint=self.solution_snapshot_fingerprint,
            scoring_contract_fingerprint=self.scoring_contract_fingerprint,
            dataset_fingerprint=self.dataset_fingerprint,
            runtime_fingerprint=self.runtime_fingerprint,
            promotion_capable=False,
            invalidation_reasons=("run_not_terminal",),
        )

    def finalize(self) -> FinalProvenance:
        current_commit = _try_git_output("rev-parse", "HEAD")
        repository_clean_end = self.repository_available and _repository_is_clean()
        snapshot_path = self.snapshot_directory / f"{self.solution_module}.py"
        solution_matches_snapshot_end = _path_matches_fingerprint(
            Path("solution.py"), self.solution_snapshot_fingerprint
        ) and _path_matches_fingerprint(snapshot_path, self.solution_snapshot_fingerprint)
        scoring_unchanged = scoring_contract_fingerprint() == self.scoring_contract_fingerprint
        dataset_unchanged = dataset_fingerprint(self.dataset) == self.dataset_fingerprint
        commit_unchanged = bool(current_commit) and current_commit == self.repository_commit
        gates = {
            "repository_unavailable": self.repository_available,
            "repository_dirty_at_start": self.repository_clean_start,
            "solution_not_committed_at_start": self.solution_matches_head_start,
            "repository_dirty_at_end": repository_clean_end,
            "solution_changed_during_run": solution_matches_snapshot_end,
            "repository_commit_changed_during_run": commit_unchanged,
            "scoring_contract_changed_during_run": scoring_unchanged,
            "dataset_changed_during_run": dataset_unchanged,
        }
        invalidation_reasons = tuple(code for code, passed in gates.items() if not passed)
        return FinalProvenance(
            repository_available=self.repository_available,
            repository_commit=self.repository_commit,
            repository_clean_start=self.repository_clean_start,
            repository_clean_end=repository_clean_end,
            solution_matches_head_start=self.solution_matches_head_start,
            solution_matches_snapshot_end=solution_matches_snapshot_end,
            repository_commit_unchanged=commit_unchanged,
            solution_snapshot_fingerprint=self.solution_snapshot_fingerprint,
            scoring_contract_fingerprint=self.scoring_contract_fingerprint,
            dataset_fingerprint=self.dataset_fingerprint,
            runtime_fingerprint=self.runtime_fingerprint,
            promotion_capable=not invalidation_reasons,
            invalidation_reasons=invalidation_reasons,
        )


@contextmanager
def capture_evidence_context(
    *,
    run_id: str,
    dataset: str,
    upstream_base_url: str = "",
) -> Iterator[EvidenceContext]:
    solution_bytes = Path("solution.py").read_bytes()
    repository_commit = _try_git_output("rev-parse", "HEAD")
    repository_available = bool(repository_commit)
    snapshot_directory = Path(tempfile.mkdtemp(prefix="pii-evaluation-"))
    snapshot_directory.chmod(0o700)
    solution_module = f"evaluation_solution_{run_id}"
    snapshot_path = snapshot_directory / f"{solution_module}.py"
    try:
        descriptor = os.open(snapshot_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(solution_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        snapshot_path.chmod(0o400)
        yield EvidenceContext(
            repository_available=repository_available,
            repository_commit=repository_commit,
            repository_clean_start=repository_available and _repository_is_clean(),
            solution_matches_head_start=_head_solution_bytes() == solution_bytes,
            solution_snapshot_fingerprint=_sha256(solution_bytes),
            scoring_contract_fingerprint=scoring_contract_fingerprint(),
            dataset_fingerprint=dataset_fingerprint(dataset),
            runtime_fingerprint=runtime_fingerprint(upstream_base_url=upstream_base_url),
            solution_module=solution_module,
            snapshot_directory=snapshot_directory,
            dataset=dataset,
            started_monotonic=time.monotonic(),
        )
    finally:
        shutil.rmtree(snapshot_directory)


def scoring_contract_fingerprint() -> str:
    manifest = {
        "version": SCORING_CONTRACT_VERSION,
        "files": [
            {"path": str(path), "sha256": _sha256((SOURCE_ROOT / path).read_bytes())}
            for path in SCORING_CONTRACT_PATHS
        ],
    }
    return _canonical_fingerprint(manifest)


def dataset_fingerprint(dataset: str) -> str:
    text_directory = Path("data") / dataset / "texts"
    ground_truth_path = Path("data") / dataset / "ground_truth.json"
    labels = json.loads(ground_truth_path.read_bytes())
    text_paths = sorted(text_directory.glob("*.txt"))
    if not isinstance(labels, dict) or set(labels) != {path.stem for path in text_paths}:
        raise ValueError("dataset texts and visible labels do not contain the same document IDs")
    documents = [
        {
            "document_id": path.stem,
            "text_sha256": _sha256(path.read_bytes()),
            "labels_sha256": _canonical_fingerprint(labels[path.stem]),
        }
        for path in text_paths
    ]
    return _canonical_fingerprint({"dataset": dataset, "documents": documents})


def runtime_fingerprint(*, upstream_base_url: str = "") -> str:
    manifest = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "packages": {name: importlib.metadata.version(name) for name in RUNTIME_PACKAGES},
        "upstream_endpoint_sha256": _sha256(_normalized_endpoint(upstream_base_url).encode()),
    }
    return _canonical_fingerprint(manifest)


def _normalized_endpoint(upstream_base_url: str) -> str:
    if not upstream_base_url:
        return ""
    parsed = urlsplit(upstream_base_url.rstrip("/"))
    scheme = parsed.scheme.lower()
    hostname = (parsed.hostname or "").lower()
    if not scheme or not hostname:
        raise ValueError(f"invalid upstream base URL {upstream_base_url!r}")
    port = parsed.port
    default_port = (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    host = f"[{hostname}]" if ":" in hostname else hostname
    netloc = host if port is None or default_port else f"{host}:{port}"
    return urlunsplit((scheme, netloc, parsed.path.rstrip("/"), "", ""))


def snapshot_environment(context: EvidenceContext, source: Mapping[str, str]) -> dict[str, str]:
    environment = dict(source)
    existing = environment.get("PYTHONPATH", "")
    entries = [str(context.snapshot_directory)]
    if existing:
        entries.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(entries)
    return environment


def _repository_is_clean() -> bool:
    result = _try_git_output("status", "--porcelain", "--untracked-files=normal")
    return result == ""


def _head_solution_bytes() -> bytes | None:
    completed = subprocess.run(
        ["git", "show", "HEAD:solution.py"],
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        return None
    return completed.stdout


def _try_git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        return ""
    return completed.stdout.strip()


def _canonical_fingerprint(value: object) -> str:
    serialized = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    return _sha256(serialized)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _path_matches_fingerprint(path: Path, expected: str) -> bool:
    try:
        return _sha256(path.read_bytes()) == expected
    except OSError:
        return False
