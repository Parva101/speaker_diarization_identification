from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Configuration that may still be handy for CI / prod."""

    similarity_threshold: float = 0.6 # Fallback only
    temp_dir: Path | None = None

    class Config:
        env_file = ".env"

settings = Settings()
