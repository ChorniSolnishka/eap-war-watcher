"""
War Model

This module defines the War database model representing individual war instances
between alliances. Each war tracks the participating alliance and enemy alliance
name, along with sequence counters for attack and defense ordering.

"""

from sqlalchemy import ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class War(Base):
    """
    War model representing a single war instance between alliances.

    Attributes:
        war_id: Primary key, auto-incrementing war identifier
        alliance_id: Foreign key to the participating alliance
        war_enemy_name: Name of the enemy alliance
        attack_seq_counter: Global sequence counter for attacks in this war
        defence_seq_counter: Global sequence counter for defenses in this war
    """

    __tablename__ = "wars"

    war_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True, comment="Unique war identifier"
    )
    alliance_id: Mapped[int] = mapped_column(
        ForeignKey("alliances.id"),
        index=True,
        nullable=False,
        comment="Reference to participating alliance",
    )
    war_enemy_name: Mapped[str] = mapped_column(
        String(255), nullable=False, comment="Name of the enemy alliance"
    )

    # Global sequence counters for ordering events within the war
    attack_seq_counter: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Global attack sequence counter for this war",
    )
    defence_seq_counter: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Global defense sequence counter for this war",
    )
