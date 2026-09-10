from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """analytics-api settings, sourced from environment variables / .env."""

    database_url: str = "postgresql+psycopg://ledger:ledger@db:5432/ledger"
    app_host: str = "0.0.0.0"
    app_port: int = 8400

    gpu_enricher_url: str = "http://gpu-enricher:8300"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
