"""
Alliance Model

This module defines the Alliance database model representing game alliances.
Each alliance has a unique name and can participate in multiple wars.

"""

from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class Alliance(Base):
    """
    Alliance model representing a game alliance.

    Attributes:
        id: Primary key, auto-incrementing alliance identifier
        name: Unique alliance name (indexed for fast lookups)
    """

    __tablename__ = "alliances"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="Unique alliance identifier",
    )
    name: Mapped[str] = mapped_column(
        String(255), index=True, unique=True, comment="Unique alliance name"
    )
