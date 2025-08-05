"""Test utilities for live-crew.

This module provides pre-generated Pydantic generic types to avoid schema generation
issues when using Freezegun with datetime fields in generic models.

The schema generation happens at import time, before any time mocking occurs.
"""

from typing import Any

from live_crew.core.models import Event, Action

# Pre-generate common Event types to avoid Freezegun + Pydantic v2 schema issues
# These schemas are created at import time, before freezegun patches datetime
EventDict = Event[dict[str, Any]]
EventAny = Event[Any]

# Pre-generate common Action types
ActionDict = Action[dict[str, Any]]
ActionAny = Action[Any]

# Export for easy importing
__all__ = [
    "EventDict",
    "EventAny",
    "ActionDict",
    "ActionAny",
]
