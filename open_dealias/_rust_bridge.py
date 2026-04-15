from __future__ import annotations

try:  # pragma: no cover - import path exercised only when the extension exists.
    from . import _rust as _native_backend
except Exception:  # pragma: no cover - fallback when Rust has not been built.
    _native_backend = None


def has_rust_backend() -> bool:
    return _native_backend is not None


def get_rust_backend():
    return _native_backend
