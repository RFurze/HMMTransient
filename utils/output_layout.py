"""Helpers for case-oriented output layout."""

from __future__ import annotations

import os
from pathlib import Path


def infer_case_id(output_dir: str) -> str:
    """Infer case id from ``output_dir``.

    Supports the new layout ``.../<case_id>/run/exchange`` and falls back to
    the basename for legacy paths.
    """
    env_case_id = os.environ.get("HMM_CASE_ID")
    if env_case_id:
        return env_case_id

    p = Path(output_dir).resolve()
    parts = p.parts
    if len(parts) >= 3 and parts[-2:] == ("run", "exchange"):
        return parts[-3]
    return p.name