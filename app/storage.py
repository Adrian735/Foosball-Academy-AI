import uuid
from pathlib import Path
from typing import Optional

from fastapi import UploadFile

from app.config import settings


class VideoStorage:
    """Local-disk video storage for the MVP; swap the implementation for S3/R2 later."""

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir or settings.storage_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, file: UploadFile) -> str:
        extension = Path(file.filename or "").suffix or ".mp4"
        filename = f"{uuid.uuid4()}{extension}"
        destination = self.base_dir / filename
        with destination.open("wb") as out:
            out.write(file.file.read())
        return str(destination)

    def save_bytes(self, content: bytes, extension: str = ".mp4") -> str:
        """Save raw video content and return its local path."""
        filename = f"{uuid.uuid4()}{extension}"
        destination = self.base_dir / filename
        destination.write_bytes(content)
        return str(destination)


storage = VideoStorage()
