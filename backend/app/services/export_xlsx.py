"""
Compatibility facade for Excel export.

build_war_report_xlsx needs to be imported from this module.
The implementation lives in app.services.excel_export.api.
"""

from __future__ import annotations

# re-export
from app.services.excel_export.api import build_war_report_xlsx

__all__ = ["build_war_report_xlsx"]
