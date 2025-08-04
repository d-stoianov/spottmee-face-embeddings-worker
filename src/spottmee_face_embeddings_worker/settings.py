from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class Settings(BaseSettings):
    database_url: str
    redis_url: str

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parent.parent.parent / '.env',
        env_file_encoding='utf-8'
    )

settings = Settings()