from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, sourced from environment variables / .env."""

    database_url: str = "postgresql+psycopg://ledger:ledger@db:5432/ledger"
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
