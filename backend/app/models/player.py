"""
Player Model

This module defines the Player database model representing individual players
in wars. Each player has attack and defense statistics with sequence tracking
for proper event ordering.

"""

from sqlalchemy import ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class Player(Base):
    """
    Player model representing a single player in a war.

    This model tracks player statistics including:
    - Basic information (name, alliance, war)
    - Base HP for damage calculations
    - Attack statistics (up to 6 attacks per player)
    - Defense statistics (up to 30 defenses per player)
    - Sequence IDs for proper event ordering

    Attributes:
        gamer_id: Primary key, auto-incrementing player identifier
        name_folder: Player name or folder identifier
        alliance_id: Foreign key to the player's alliance
        war_id: Foreign key to the war this player participates in
        base_hp: Player's base hit points for damage calculations
        attack_N: Damage dealt in attack slot N (1-6)
        attack_N_sc_id: Screenshot crop ID for attack N
        attack_N_sec_id: Sequence ID for attack N ordering
        defence_N: Damage received in defense slot N (1-30)
        defence_N_sc_id: Screenshot crop ID for defense N
        defence_N_sec_id: Sequence ID for defense N ordering
    """

    __tablename__ = "players"

    # Primary key and basic information
    gamer_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="Unique player identifier",
    )
    name_folder: Mapped[str] = mapped_column(
        String(1024), nullable=False, comment="Player name or folder identifier"
    )
    alliance_id: Mapped[int] = mapped_column(
        ForeignKey("alliances.id"),
        index=True,
        nullable=False,
        comment="Reference to player's alliance",
    )
    war_id: Mapped[int] = mapped_column(
        ForeignKey("wars.war_id"),
        index=True,
        nullable=False,
        comment="Reference to the war this player participates in",
    )

    # Base HP for damage calculations
    base_hp: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Player's base hit points for damage calculations",
    )

    # Attack slots (1-6): damage, screenshot crop ID, and sequence ID
    attack_1: Mapped[int | None] = mapped_column(
        Integer, comment="Damage in attack slot 1"
    )
    attack_1_sc_id: Mapped[int | None] = mapped_column(
        Integer, comment="Screenshot crop ID for attack 1"
    )
    attack_1_sec_id: Mapped[int | None] = mapped_column(
        Integer, comment="Sequence ID for attack 1"
    )

    attack_2: Mapped[int | None] = mapped_column(
        Integer, comment="Damage in attack slot 2"
    )
    attack_2_sc_id: Mapped[int | None] = mapped_column(
        Integer, comment="Screenshot crop ID for attack 2"
    )
    attack_2_sec_id: Mapped[int | None] = mapped_column(
        Integer, comment="Sequence ID for attack 2"
    )

    attack_3: Mapped[int | None] = mapped_column(
        Integer, comment="Damage in attack slot 3"
    )
    attack_3_sc_id: Mapped[int | None] = mapped_column(
        Integer, comment="Screenshot crop ID for attack 3"
    )
    attack_3_sec_id: Mapped[int | None] = mapped_column(
        Integer, comment="Sequence ID for attack 3"
    )

    attack_4: Mapped[int | None] = mapped_column(
        Integer, comment="Damage in attack slot 4"
    )
    attack_4_sc_id: Mapped[int | None] = mapped_column(
        Integer, comment="Screenshot crop ID for attack 4"
    )
    attack_4_sec_id: Mapped[int | None] = mapped_column(
        Integer, comment="Sequence ID for attack 4"
    )

    attack_5: Mapped[int | None] = mapped_column(
        Integer, comment="Damage in attack slot 5"
    )
    attack_5_sc_id: Mapped[int | None] = mapped_column(
        Integer, comment="Screenshot crop ID for attack 5"
    )
    attack_5_sec_id: Mapped[int | None] = mapped_column(
        Integer, comment="Sequence ID for attack 5"
    )

    attack_6: Mapped[int | None] = mapped_column(
        Integer, comment="Damage in attack slot 6"
    )
    attack_6_sc_id: Mapped[int | None] = mapped_column(
        Integer, comment="Screenshot crop ID for attack 6"
    )
    attack_6_sec_id: Mapped[int | None] = mapped_column(
        Integer, comment="Sequence ID for attack 6"
    )

    # Defense slots (1-30): dynamically created fields
    # Each defense slot has: damage, screenshot crop ID, and sequence ID
    for i in range(1, 31):
        locals()[f"defence_{i}"] = mapped_column(
            Integer, nullable=True, comment=f"Damage received in defense slot {i}"
        )
        locals()[f"defence_{i}_sc_id"] = mapped_column(
            Integer, nullable=True, comment=f"Screenshot crop ID for defense {i}"
        )
        locals()[f"defence_{i}_sec_id"] = mapped_column(
            Integer, nullable=True, comment=f"Sequence ID for defense {i}"
        )
