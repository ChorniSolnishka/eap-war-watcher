"""
Screenshot crop model

This module regulates crops with nicknames and digits
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class ScreenshotCrop(Base):
    __tablename__ = "screenshot_crops"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    war_id: Mapped[int] = mapped_column(
        ForeignKey("wars.war_id", ondelete="CASCADE"), nullable=False
    )
    screen_id: Mapped[int] = mapped_column(
        ForeignKey("screenshots.screen_id", ondelete="CASCADE"), nullable=False
    )

    # row внутри скриншота (1..N)
    row_idx: Mapped[int] = mapped_column(Integer, nullable=False)

    # "left" | "mid" | "right"
    kind: Mapped[str] = mapped_column(String(8), nullable=False)

    # путь к png-файлу кропа
    path: Mapped[str] = mapped_column(String, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
