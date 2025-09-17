"""
Screenshot Model

This module regulates screenshots uploaded by user
"""

from sqlalchemy import ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class Screenshot(Base):
    __tablename__ = "screenshots"
    screen_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    war_id: Mapped[int] = mapped_column(
        ForeignKey("wars.war_id"), index=True, nullable=False
    )
    folder: Mapped[str] = mapped_column(String(1024), nullable=False)
