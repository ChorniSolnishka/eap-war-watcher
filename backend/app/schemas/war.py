"""
War Schemas

This module defines Pydantic schemas for war-related API requests and responses.
These schemas provide data validation and serialization for the war endpoints.

"""

from pydantic import BaseModel, Field


class WarCreate(BaseModel):
    """
    Schema for creating a new war.

    Attributes:
        my_alliance: Name of the player's alliance
        enemy_alliance: Name of the enemy alliance
    """

    my_alliance: str = Field(
        ..., min_length=1, max_length=255, description="Name of the player's alliance"
    )
    enemy_alliance: str = Field(
        ..., min_length=1, max_length=255, description="Name of the enemy alliance"
    )


class WarOut(BaseModel):
    """
    Schema for war response data.

    Attributes:
        war_id: Unique identifier for the created war
        my_alliance_id: Database ID of the player's alliance
        enemy_alliance_id: Database ID of the enemy alliance
    """

    war_id: int = Field(..., description="Unique war identifier")
    my_alliance_id: int = Field(..., description="Player's alliance database ID")
    enemy_alliance_id: int = Field(..., description="Enemy alliance database ID")
