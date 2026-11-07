"""Phase 0 smoke tests: the package imports and settings load safely."""

from pathlib import Path

import pytest

import cfp
from cfp.config import Settings


def test_package_imports() -> None:
    assert cfp.__version__


def test_settings_defaults_without_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Run from an empty directory so a developer's real .env is not picked up.
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("COMPANIES_HOUSE_API_KEY", raising=False)
    monkeypatch.delenv("CFP_DATA_DIR", raising=False)

    settings = Settings()

    assert settings.companies_house_api_key is None
    assert settings.raw_dir == Path("data") / "raw"
    assert settings.api_cache_dir == Path("data") / "raw" / "ch_api_cache"
    with pytest.raises(RuntimeError, match="COMPANIES_HOUSE_API_KEY"):
        settings.require_api_key()


def test_api_key_is_read_from_env_and_masked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COMPANIES_HOUSE_API_KEY", "not-a-real-key-123")

    settings = Settings()

    assert settings.require_api_key() == "not-a-real-key-123"
    assert "not-a-real-key-123" not in repr(settings)
    assert "not-a-real-key-123" not in str(settings.model_dump())
