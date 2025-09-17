# File: app/db/session.py
"""
Database Session Management

This module provides database connection and session management functionality.
It configures SQLAlchemy engine and provides dependency injection for database sessions.

"""
import logging
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Base class for all database models."""

    pass


# Create database engine with connection pooling
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,  # Verify connections before use
    pool_recycle=3600,  # Recycle connections every hour
    echo=(settings.ENV == "dev"),  # Log SQL queries in development
)

# Create session factory
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,  # Manual control over flushing
    autocommit=False,  # Manual control over commits
    expire_on_commit=False,  # Keep objects accessible after commit
)


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that yields a DB session and guarantees cleanup.
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error("Database session error: %s", e)
        db.rollback()
        raise
    finally:
        db.close()


def create_tables() -> None:
    """Create all database tables."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error("Failed to create database tables: %s", e)
        raise


def drop_tables() -> None:
    """
    Drop all database tables.
    """
    try:
        Base.metadata.drop_all(bind=engine)
        logger.warning("All database tables dropped")
    except Exception as e:
        logger.error("Failed to drop database tables: %s", e)
        raise
