"""Reusable development-data access guards.

These guards are intentionally conservative. They do not prove that a process
never opened a forbidden file, but they reject forbidden paths and partitions at
the application boundary before event loading.
"""

from __future__ import annotations

from pathlib import Path


FORBIDDEN_PATH_TOKENS = (
    "label_pairs_test",
    "th232",
    "eu152",
)
ALLOWED_DEVELOPMENT_PARTITIONS = {"train", "validation"}


def assert_no_forbidden_path(path: Path | str) -> None:
    value = str(path).lower()
    for token in FORBIDDEN_PATH_TOKENS:
        if token in value:
            raise ValueError(f"Forbidden locked/external path for development audit: {path}")


def assert_development_csv(path: Path | str) -> None:
    """Reject test/external manifests and require a train/validation filename."""

    assert_no_forbidden_path(path)
    stem = Path(path).stem
    if stem not in {"label_pairs_train", "label_pairs_validation"}:
        raise ValueError(f"Development audit requires train/validation pair CSV: {path}")


def assert_development_partition(partition: str) -> None:
    if partition not in ALLOWED_DEVELOPMENT_PARTITIONS:
        raise ValueError(f"Forbidden development partition: {partition}")
