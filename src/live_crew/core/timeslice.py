"""TimeSlice calculation utilities for live-crew.

Based on specification in .vibes/live_crew_spec.md section 2.2
Provides deterministic time slicing for event processing.
"""

from datetime import datetime


def slice_index(ts: datetime, epoch0: datetime, slice_ms: int = 500) -> int:
    """Return the zero-based slice index for timestamp ts.

    Args:
        ts: The timestamp to calculate slice index for
        epoch0: The epoch start time for the stream
        slice_ms: Time slice duration in milliseconds (defaults to 500ms)

    Returns:
        Zero-based slice index (can be negative if ts < epoch0)

    Note:
        Uses integer division to ensure deterministic results.
        For replay consistency, all times should be in the same timezone.
    """
    # Calculate milliseconds since epoch0
    delta_ms = (ts - epoch0).total_seconds() * 1000

    # Use integer division to get slice index
    return int(delta_ms // slice_ms)
