from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the ``--run-live-api`` opt-in for live-API tests.

    Tests marked ``@pytest.mark.live_api`` are skipped by default
    because they require network access and a working Anthropic
    auth context.  Pass ``--run-live-api`` (e.g.,
    ``pytest tests/integration --run-live-api``) to opt in.
    """
    parser.addoption(
        "--run-live-api",
        action="store_true",
        default=False,
        help=(
            "Run tests marked @pytest.mark.live_api "
            "(requires network + Anthropic credentials)."
        ),
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item],
) -> None:
    """Skip live_api tests unless ``--run-live-api`` is passed."""
    if config.getoption("--run-live-api"):
        return
    skip_marker = pytest.mark.skip(
        reason="needs --run-live-api opt-in")
    for item in items:
        if "live_api" in item.keywords:
            item.add_marker(skip_marker)


@pytest.fixture()
def git_repo(tmp_path: Path) -> Path:
    """Create a minimal git repo and return its path."""
    _init_git_repo(tmp_path)
    return tmp_path


def _init_git_repo(path: Path) -> str:
    """Create a minimal git repo and return the HEAD commit SHA."""
    subprocess.run(["git", "init", str(path)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@test.com"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test"],
        check=True,
        capture_output=True,
    )
    (path / "README.md").write_text("test")
    subprocess.run(
        ["git", "-C", str(path), "add", "."], check=True, capture_output=True
    )
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "init"],
        check=True,
        capture_output=True,
    )
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()
