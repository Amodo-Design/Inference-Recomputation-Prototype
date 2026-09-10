from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """gpu-enricher settings, sourced from environment variables / .env."""

    database_url: str = "postgresql+psycopg://ledger:ledger@db:5432/ledger"
    app_host: str = "0.0.0.0"
    app_port: int = 8300

    # Poller runs only when prometheus_url is set (unset in docker-compose,
    # so dev never starts it); the window endpoint 503s without it too.
    prometheus_url: str | None = None
    gpu_poll_interval_seconds: float = 30.0
    gpu_enrich_delay_seconds: float = 10.0
    gpu_poll_batch: int = 100

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
