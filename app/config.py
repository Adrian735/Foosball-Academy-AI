from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    foosball_database_url: str = "postgresql+psycopg2://foosball:foosball@localhost:5433/foosball"
    redis_url: str = "redis://localhost:6379/0"
    storage_dir: str = "./data/videos"
    max_video_duration_seconds: int = 60
    max_video_size_mb: int = 200

    class Config:
        env_file = ".env"


settings = Settings()
