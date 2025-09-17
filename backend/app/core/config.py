# File: app/core/config.py
"""
Application Configuration
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    # App
    APP_NAME: str = Field(default="EaP WarWatcher")
    ENV: Literal["dev", "staging", "prod"] = Field(default="dev")

    # API & Static
    API_PREFIX: str = Field(default="/api", description="Root API prefix")
    STATIC_URL: str = Field(default="/static", description="Static mount URL")

    # Database
    DATABASE_URL: str = Field(
        ...,
        description="postgresql+psycopg://user:pass@host:port/dbname",
    )

    # Storage
    STORAGE_ROOT: Path = Field(default=Path("storage"))

    # OpenCV
    OPENCV_NUM_THREADS: int | None = Field(
        default=None,
        description="If None -> auto: max(1, cpu_count//2)",
    )

    # OCR
    TESSERACT_PATH: str | None = Field(default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure storage dir
        self.STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
        # Make Tesseract visible
        if self.TESSERACT_PATH:
            os.environ["TESSERACT_PATH"] = self.TESSERACT_PATH


settings = Settings()
