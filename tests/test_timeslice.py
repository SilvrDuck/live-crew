"""Tests for TimeSlice calculation utilities.

Based on specification in .vibes/live_crew_spec.md section 2.2
"""

from datetime import datetime, timezone, timedelta

from live_crew.core.timeslice import slice_index


class TestSliceIndex:
    """Test cases for slice_index function."""

    def test_slice_index_basic(self):
        """Test basic slice_index calculation."""
        epoch0 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        # Same timestamp should be slice 0
        assert slice_index(epoch0, epoch0) == 0

        # 500ms later should be slice 1
        ts_500ms = epoch0 + timedelta(milliseconds=500)
        assert slice_index(ts_500ms, epoch0) == 1

        # 1000ms later should be slice 2
        ts_1000ms = epoch0 + timedelta(milliseconds=1000)
        assert slice_index(ts_1000ms, epoch0) == 2

    def test_slice_index_fractional_slices(self):
        """Test slice_index handles fractional slices correctly."""
        epoch0 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        # 250ms (0.5 slices) should round down to slice 0
        ts_250ms = epoch0 + timedelta(milliseconds=250)
        assert slice_index(ts_250ms, epoch0) == 0

        # 499ms should still be slice 0
        ts_499ms = epoch0 + timedelta(milliseconds=499)
        assert slice_index(ts_499ms, epoch0) == 0

        # 750ms (1.5 slices) should round down to slice 1
        ts_750ms = epoch0 + timedelta(milliseconds=750)
        assert slice_index(ts_750ms, epoch0) == 1

    def test_slice_index_large_intervals(self):
        """Test slice_index with larger time intervals."""
        epoch0 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        # 5 seconds = 10 slices
        ts_5s = epoch0 + timedelta(seconds=5)
        assert slice_index(ts_5s, epoch0) == 10

        # 1 minute = 120 slices
        ts_1min = epoch0 + timedelta(minutes=1)
        assert slice_index(ts_1min, epoch0) == 120

        # 1 hour = 7200 slices
        ts_1hour = epoch0 + timedelta(hours=1)
        assert slice_index(ts_1hour, epoch0) == 7200

    def test_slice_index_microsecond_precision(self):
        """Test slice_index handles microsecond precision correctly."""
        epoch0 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        # Test with microseconds that should not affect slice calculation
        ts_with_microseconds = epoch0 + timedelta(milliseconds=500, microseconds=123)
        assert slice_index(ts_with_microseconds, epoch0) == 1

        # Test microseconds just under the next slice
        ts_499_999 = epoch0 + timedelta(milliseconds=499, microseconds=999)
        assert slice_index(ts_499_999, epoch0) == 0

    def test_slice_index_different_timezones(self):
        """Test slice_index works correctly with different timezones."""
        # Create epoch0 in UTC
        epoch0_utc = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        # Create timestamp in different timezone but same absolute time + 500ms
        pst = timezone(timedelta(hours=-8))
        ts_pst = datetime(
            2024, 1, 1, 4, 0, 0, 500000, tzinfo=pst
        )  # 4am PST = 12pm UTC + 500ms

        assert slice_index(ts_pst, epoch0_utc) == 1

    def test_slice_index_past_timestamps(self):
        """Test slice_index handles timestamps before epoch0."""
        epoch0 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        # 500ms before epoch0 should be slice -1
        ts_before = epoch0 - timedelta(milliseconds=500)
        assert slice_index(ts_before, epoch0) == -1

        # 1000ms before epoch0 should be slice -2
        ts_1s_before = epoch0 - timedelta(milliseconds=1000)
        assert slice_index(ts_1s_before, epoch0) == -2

    def test_slice_index_deterministic(self):
        """Test slice_index produces deterministic results."""
        epoch0 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        ts = epoch0 + timedelta(milliseconds=1337)

        # Multiple calls should return same result
        result1 = slice_index(ts, epoch0)
        result2 = slice_index(ts, epoch0)
        assert result1 == result2 == 2  # 1337ms / 500ms = 2.674 -> 2


class TestSliceConfigurable:
    """Test cases for configurable slice_ms parameter."""

    def test_slice_ms_default_value(self):
        """Test slice_index uses correct default slice_ms value."""
        epoch0 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        ts_500ms = epoch0 + timedelta(milliseconds=500)

        # Default behavior should use 500ms slices
        assert slice_index(ts_500ms, epoch0) == 1

    def test_slice_ms_custom_value(self):
        """Test slice_index with custom slice_ms parameter."""
        epoch0 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        # Test with 1000ms slices
        ts_500ms = epoch0 + timedelta(milliseconds=500)
        assert slice_index(ts_500ms, epoch0, slice_ms=1000) == 0

        ts_1000ms = epoch0 + timedelta(milliseconds=1000)
        assert slice_index(ts_1000ms, epoch0, slice_ms=1000) == 1

        # Test with 250ms slices
        ts_250ms = epoch0 + timedelta(milliseconds=250)
        assert slice_index(ts_250ms, epoch0, slice_ms=250) == 1


class TestSliceIndexEdgeCases:
    """Test edge cases for slice_index function."""

    def test_slice_index_exact_boundaries(self):
        """Test slice_index at exact slice boundaries."""
        epoch0 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        slice_ms = 500  # Using default value

        # Test exact multiples of slice_ms
        for i in range(10):
            ts = epoch0 + timedelta(milliseconds=i * slice_ms)
            assert slice_index(ts, epoch0, slice_ms=slice_ms) == i

    def test_slice_index_large_time_differences(self):
        """Test slice_index with very large time differences."""
        epoch0 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        slice_ms = 500  # Using default value

        # 1 year later (approximate)
        ts_1year = epoch0 + timedelta(days=365)
        expected_slices = int((365 * 24 * 60 * 60 * 1000) // slice_ms)
        assert slice_index(ts_1year, epoch0, slice_ms=slice_ms) == expected_slices

    def test_slice_index_naive_datetime_handling(self):
        """Test slice_index behavior with naive datetimes."""
        # Both timestamps naive (should work as long as they're consistent)
        epoch0_naive = datetime(2024, 1, 1, 12, 0, 0)
        ts_naive = datetime(2024, 1, 1, 12, 0, 0, 500000)  # +500ms

        assert slice_index(ts_naive, epoch0_naive) == 1
