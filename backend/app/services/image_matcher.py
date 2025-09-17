"""
Compatibility facade for text-oriented image matcher.

Keep importing `match_or_new` from this module.
Implementation lives in `app.services.imatch`.
"""

from __future__ import annotations

# Re-export public API unchanged
from app.services.imatch import image_hash64, match_or_new, verify_gray

__all__ = ["match_or_new", "verify_gray", "image_hash64"]
