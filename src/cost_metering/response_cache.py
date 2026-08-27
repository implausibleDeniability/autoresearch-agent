import base64
import errno
import hashlib
import json
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional
from urllib.parse import urlsplit, urlunsplit

CACHE_SCHEMA_VERSION = 1
SEMANTIC_HEADER_NAMES = (
    "Idempotency-Key",
    "OpenAI-Beta",
    "OpenAI-Organization",
    "OpenAI-Project",
)


class CacheEntryError(RuntimeError):
    pass


@dataclass(frozen=True)
class CachedResponse:
    status_code: int
    content_type: str
    content: bytes


class ResponseCache:
    def __init__(self, directory: Path, *, upstream_base_url: str) -> None:
        self._directory = directory
        self._upstream_base_url = _normalize_base_url(upstream_base_url)

    def get(
        self,
        *,
        path: str,
        body: bytes,
        headers: Mapping[str, str],
    ) -> Optional[CachedResponse]:
        entry_path = self._entry_path(path=path, body=body, headers=headers)
        if not entry_path.exists():
            return None
        try:
            serialized = entry_path.read_bytes()
            payload = json.loads(serialized)
            return _decode_response(payload)
        except (OSError, UnicodeError, ValueError, TypeError, KeyError) as error:
            raise CacheEntryError("response cache entry is malformed") from error

    def put(
        self,
        *,
        path: str,
        body: bytes,
        headers: Mapping[str, str],
        response: CachedResponse,
    ) -> bool:
        entry_path = self._entry_path(path=path, body=body, headers=headers)
        self._prepare_directory(entry_path.parent)
        if entry_path.exists():
            return False
        serialized = _encode_response(response)
        temporary_path = entry_path.with_name(f".{entry_path.name}.{secrets.token_hex(8)}.tmp")
        descriptor = os.open(temporary_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary_path, entry_path)
            except OSError as error:
                if error.errno == errno.EEXIST:
                    return False
                raise
            _fsync_directory(entry_path.parent)
            return True
        finally:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass

    def request_key(self, *, path: str, body: bytes, headers: Mapping[str, str]) -> str:
        return _request_digest(
            upstream_base_url=self._upstream_base_url,
            path=path,
            body=body,
            headers=headers,
        )

    def _entry_path(self, *, path: str, body: bytes, headers: Mapping[str, str]) -> Path:
        digest = self.request_key(path=path, body=body, headers=headers)
        return self._directory / str(CACHE_SCHEMA_VERSION) / digest[:2] / f"{digest}.json"

    def _prepare_directory(self, directory: Path) -> None:
        cache_root = self._directory
        version_directory = cache_root / str(CACHE_SCHEMA_VERSION)
        for path in (cache_root, version_directory, directory):
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
            path.chmod(0o700)


def _request_digest(
    *,
    upstream_base_url: str,
    path: str,
    body: bytes,
    headers: Mapping[str, str],
) -> str:
    try:
        request_body = json.loads(body)
    except (UnicodeError, ValueError) as error:
        raise ValueError("response cache request body must be valid JSON") from error
    normalized_headers = {name.lower(): value for name, value in headers.items()}
    semantic_headers = {
        name.lower(): normalized_headers[name.lower()]
        for name in SEMANTIC_HEADER_NAMES
        if name.lower() in normalized_headers
    }
    envelope = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "method": "POST",
        "upstream_base_url": upstream_base_url,
        "path": path,
        "headers": semantic_headers,
        "body": request_body,
    }
    canonical = json.dumps(
        envelope,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _encode_response(response: CachedResponse) -> bytes:
    checksum = hashlib.sha256(response.content).hexdigest()
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "status_code": response.status_code,
        "content_type": response.content_type,
        "content_sha256": checksum,
        "content_base64": base64.b64encode(response.content).decode("ascii"),
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()


def _decode_response(payload: object) -> CachedResponse:
    if not isinstance(payload, dict) or payload.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError("unsupported response cache schema")
    status_code = payload["status_code"]
    content_type = payload["content_type"]
    checksum = payload["content_sha256"]
    encoded = payload["content_base64"]
    if isinstance(status_code, bool) or not isinstance(status_code, int) or not 200 <= status_code < 300:
        raise ValueError("invalid response cache status")
    if not all(isinstance(value, str) for value in (content_type, checksum, encoded)):
        raise ValueError("invalid response cache metadata")
    content = base64.b64decode(encoded, validate=True)
    if hashlib.sha256(content).hexdigest() != checksum:
        raise ValueError("response cache checksum mismatch")
    return CachedResponse(status_code=status_code, content_type=content_type, content=content)


def _normalize_base_url(value: str) -> str:
    parsed = urlsplit(value.rstrip("/"))
    scheme = parsed.scheme.lower()
    hostname = (parsed.hostname or "").lower()
    if not scheme or not hostname:
        raise ValueError(f"invalid upstream base URL {value!r}")
    port = parsed.port
    default_port = (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    host = f"[{hostname}]" if ":" in hostname else hostname
    netloc = host if port is None or default_port else f"{host}:{port}"
    return urlunsplit((scheme, netloc, parsed.path.rstrip("/"), "", ""))


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
