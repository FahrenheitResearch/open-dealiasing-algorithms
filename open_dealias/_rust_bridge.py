from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
import os

_BACKEND_IMPORT_ERROR: Exception | None = None
_UNSET = object()


def _normalize_backend_policy(mode: str | None) -> str:
    normalized = "auto" if mode is None else str(mode).strip().lower() or "auto"
    if normalized not in {"auto", "native", "python"}:
        raise ValueError("backend policy must be one of: auto, native, python")
    return normalized


_DEFAULT_BACKEND_POLICY = _normalize_backend_policy(
    os.environ.get("OPEN_DEALIAS_BACKEND_MODE", os.environ.get("OPEN_DEALIAS_BACKEND", "auto"))
)

try:  # pragma: no cover - import path exercised only when the extension exists.
    from . import _rust as _native_backend
except Exception as exc:  # pragma: no cover - fallback when Rust has not been built.
    _native_backend = None
    _BACKEND_IMPORT_ERROR = exc

if _DEFAULT_BACKEND_POLICY == "native" and _native_backend is None:  # pragma: no cover - configuration error.
    raise RuntimeError("OPEN_DEALIAS_BACKEND=native was requested, but the Rust extension could not be imported") from _BACKEND_IMPORT_ERROR


_BACKEND_POLICY: ContextVar[str] = ContextVar(
    "open_dealias_backend_policy",
    default=_DEFAULT_BACKEND_POLICY,
)


def has_rust_backend() -> bool:
    return _native_backend is not None


def get_rust_backend():
    return _native_backend


def get_rust_backend_error() -> Exception | None:
    return _BACKEND_IMPORT_ERROR


def get_backend_policy() -> str:
    return _BACKEND_POLICY.get()


def set_backend_policy(mode: str) -> Token[str]:
    normalized = _normalize_backend_policy(mode)
    if normalized == "native" and _native_backend is None:
        raise RuntimeError("backend policy 'native' requested, but the Rust extension is unavailable") from _BACKEND_IMPORT_ERROR
    return _BACKEND_POLICY.set(normalized)


def reset_backend_policy(token: Token[str]) -> None:
    _BACKEND_POLICY.reset(token)


def resolve_rust_backend(local_backend=_UNSET):
    if get_backend_policy() == "python":
        return None
    if local_backend is _UNSET:
        return _native_backend
    return local_backend


@contextmanager
def backend_policy(mode: str):
    token = set_backend_policy(mode)
    try:
        yield
    finally:
        reset_backend_policy(token)
