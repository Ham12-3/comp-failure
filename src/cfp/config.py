"""Project settings, read from environment variables and an optional `.env` file.

Secrets never live in code. The Companies House key is read from
`COMPANIES_HOUSE_API_KEY` and stored as a `SecretStr`, so it is masked in
logs, reprs and tracebacks.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration in one place.

    Relative paths are resolved against the current working directory, which
    is the repo root when running locally and `/app` inside Docker.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    companies_house_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="COMPANIES_HOUSE_API_KEY",
        description="Companies House Public Data API key. Optional so tests and CI run without it.",
    )
    data_dir: Path = Field(default=Path("data"), validation_alias="CFP_DATA_DIR")
    reports_dir: Path = Field(default=Path("reports"), validation_alias="CFP_REPORTS_DIR")
    mlflow_tracking_uri: str = Field(
        default="file:./mlruns", validation_alias="MLFLOW_TRACKING_URI"
    )

    @property
    def raw_dir(self) -> Path:
        """Downloaded files exactly as received. Never edited."""
        return self.data_dir / "raw"

    @property
    def interim_dir(self) -> Path:
        """Parsed but not yet joined data (e.g. one Parquet file per accounts archive)."""
        return self.data_dir / "interim"

    @property
    def processed_dir(self) -> Path:
        """Model-ready tables: labels, features, splits."""
        return self.data_dir / "processed"

    @property
    def api_cache_dir(self) -> Path:
        """On-disk cache of Companies House API responses, so nothing is fetched twice."""
        return self.raw_dir / "ch_api_cache"

    def require_api_key(self) -> str:
        """Return the API key, or fail with a message saying how to set it."""
        if self.companies_house_api_key is None:
            raise RuntimeError(
                "COMPANIES_HOUSE_API_KEY is not set. Copy .env.example to .env and add your key "
                "from https://developer.company-information.service.gov.uk/"
            )
        return self.companies_house_api_key.get_secret_value()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings instance for application code. Tests build `Settings()` directly."""
    return Settings()
