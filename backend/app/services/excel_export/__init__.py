"""
Excel export package for WarWatcher.

Public API is exposed from .api.
"""

from .api import build_war_report_xlsx

__all__ = ["build_war_report_xlsx"]
