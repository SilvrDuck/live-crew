"""Tests for string pattern validation.

Tests the enhanced validation that ensures kind and stream_id fields
match their documented patterns.
"""

import pytest
from datetime import datetime, timezone
from pydantic import ValidationError

from live_crew.core.models import Event, Action


@pytest.fixture
def fixed_timestamp():
    """Fixed timestamp for testing."""
    return datetime(2025, 7, 31, 12, 0, 0, tzinfo=timezone.utc)


class TestKindPatternValidation:
    """Test kind field pattern validation."""

    def test_valid_kind_patterns(self, fixed_timestamp):
        """Test valid kind patterns."""
        valid_kinds = [
            "goal_scored",
            "frame_update",
            "user_input",
            "test123",
            "TEST_EVENT",
            "simple",
            "a_b_c_1_2_3",
        ]

        for kind in valid_kinds:
            event = Event[str](
                ts=fixed_timestamp, kind=kind, stream_id="test_stream", payload="test"
            )
            assert event.kind == kind

    def test_invalid_kind_patterns(self, fixed_timestamp):
        """Test invalid kind patterns are rejected."""
        invalid_kinds = [
            "goal-scored",  # dash not allowed
            "goal scored",  # space not allowed
            "goal.scored",  # dot not allowed
            "goal@scored",  # special chars not allowed
            "goal/scored",  # slash not allowed
            "goal\\scored",  # backslash not allowed
            "goal:scored",  # colon not allowed
            "goal;scored",  # semicolon not allowed
            "goal,scored",  # comma not allowed
            "goal(scored)",  # parentheses not allowed
            "goal[scored]",  # brackets not allowed
            "goal{scored}",  # braces not allowed
            "goal|scored",  # pipe not allowed
            "goal+scored",  # plus not allowed
            "goal=scored",  # equals not allowed
            "goal%scored",  # percent not allowed
            "goal&scored",  # ampersand not allowed
            "goal*scored",  # asterisk not allowed
            "goal#scored",  # hash not allowed
            "goal!scored",  # exclamation not allowed
            "goal?scored",  # question mark not allowed
            "goal'scored",  # quote not allowed
            'goal"scored',  # double quote not allowed
        ]

        for kind in invalid_kinds:
            with pytest.raises(ValidationError) as exc_info:
                Event[str](
                    ts=fixed_timestamp,
                    kind=kind,
                    stream_id="test_stream",
                    payload="test",
                )
            assert "String should match pattern" in str(exc_info.value)

    def test_kind_whitespace_validation(self, fixed_timestamp):
        """Test kind whitespace validation."""
        invalid_kinds_with_whitespace = [
            " goal_scored",  # leading space
            "goal_scored ",  # trailing space
            "  goal_scored  ",  # leading and trailing spaces
            "\tgoal_scored",  # leading tab
            "goal_scored\t",  # trailing tab
            "\ngoal_scored",  # leading newline
            "goal_scored\n",  # trailing newline
        ]

        for kind in invalid_kinds_with_whitespace:
            with pytest.raises(ValidationError) as exc_info:
                Event[str](
                    ts=fixed_timestamp,
                    kind=kind,
                    stream_id="test_stream",
                    payload="test",
                )
            assert "String should match pattern" in str(exc_info.value)


class TestStreamIdPatternValidation:
    """Test stream_id field pattern validation."""

    def test_valid_stream_id_patterns(self, fixed_timestamp):
        """Test valid stream_id patterns."""
        valid_stream_ids = [
            "match42",
            "user-123",
            "test_stream",
            "stream-with-dashes",
            "stream_with_underscores",
            "MixedCase123",
            "a-b_c-d_1-2_3",
            "simple",
            "UPPER_CASE",
        ]

        for stream_id in valid_stream_ids:
            event = Event[str](
                ts=fixed_timestamp,
                kind="test_event",
                stream_id=stream_id,
                payload="test",
            )
            assert event.stream_id == stream_id

    def test_invalid_stream_id_patterns(self, fixed_timestamp):
        """Test invalid stream_id patterns are rejected."""
        invalid_stream_ids = [
            "stream.id",  # dot not allowed
            "stream id",  # space not allowed
            "stream@id",  # special chars not allowed
            "stream/id",  # slash not allowed
            "stream\\id",  # backslash not allowed
            "stream:id",  # colon not allowed
            "stream;id",  # semicolon not allowed
            "stream,id",  # comma not allowed
            "stream(id)",  # parentheses not allowed
            "stream[id]",  # brackets not allowed
            "stream{id}",  # braces not allowed
            "stream|id",  # pipe not allowed
            "stream+id",  # plus not allowed
            "stream=id",  # equals not allowed
            "stream%id",  # percent not allowed
            "stream&id",  # ampersand not allowed
            "stream*id",  # asterisk not allowed
            "stream#id",  # hash not allowed
            "stream!id",  # exclamation not allowed
            "stream?id",  # question mark not allowed
            "stream'id",  # quote not allowed
            'stream"id',  # double quote not allowed
        ]

        for stream_id in invalid_stream_ids:
            with pytest.raises(ValidationError) as exc_info:
                Event[str](
                    ts=fixed_timestamp,
                    kind="test_event",
                    stream_id=stream_id,
                    payload="test",
                )
            assert "String should match pattern" in str(exc_info.value)

    def test_stream_id_whitespace_validation(self, fixed_timestamp):
        """Test stream_id whitespace validation."""
        invalid_stream_ids_with_whitespace = [
            " match42",  # leading space
            "match42 ",  # trailing space
            "  match42  ",  # leading and trailing spaces
            "\tmatch42",  # leading tab
            "match42\t",  # trailing tab
            "\nmatch42",  # leading newline
            "match42\n",  # trailing newline
        ]

        for stream_id in invalid_stream_ids_with_whitespace:
            with pytest.raises(ValidationError) as exc_info:
                Event[str](
                    ts=fixed_timestamp,
                    kind="test_event",
                    stream_id=stream_id,
                    payload="test",
                )
            assert "String should match pattern" in str(exc_info.value)


class TestActionPatternValidation:
    """Test that Action also uses the same pattern validation."""

    def test_action_kind_validation(self, fixed_timestamp):
        """Test Action kind validation works the same as Event."""
        # Valid kind
        action = Action[str](
            ts=fixed_timestamp,
            kind="valid_action",
            stream_id="test_stream",
            payload="test",
        )
        assert action.kind == "valid_action"

        # Invalid kind
        with pytest.raises(ValidationError):
            Action[str](
                ts=fixed_timestamp,
                kind="invalid-action",  # dash not allowed in kind
                stream_id="test_stream",
                payload="test",
            )

    def test_action_stream_id_validation(self, fixed_timestamp):
        """Test Action stream_id validation works the same as Event."""
        # Valid stream_id
        action = Action[str](
            ts=fixed_timestamp,
            kind="test_action",
            stream_id="valid-stream_id",
            payload="test",
        )
        assert action.stream_id == "valid-stream_id"

        # Invalid stream_id
        with pytest.raises(ValidationError):
            Action[str](
                ts=fixed_timestamp,
                kind="test_action",
                stream_id="invalid.stream.id",  # dots not allowed
                payload="test",
            )


class TestEdgeCases:
    """Test edge cases for pattern validation."""

    def test_empty_strings(self, fixed_timestamp):
        """Test empty strings are rejected."""
        with pytest.raises(ValidationError):
            Event[str](
                ts=fixed_timestamp,
                kind="",  # Empty kind
                stream_id="test_stream",
                payload="test",
            )

        with pytest.raises(ValidationError):
            Event[str](
                ts=fixed_timestamp,
                kind="test_event",
                stream_id="",  # Empty stream_id
                payload="test",
            )

    def test_boundary_lengths(self, fixed_timestamp):
        """Test boundary length validation still works."""
        # Max length kind (50 chars)
        max_kind = "a" * 50
        event = Event[str](
            ts=fixed_timestamp, kind=max_kind, stream_id="test_stream", payload="test"
        )
        assert event.kind == max_kind

        # Exceeds max length kind (51 chars)
        with pytest.raises(ValidationError):
            Event[str](
                ts=fixed_timestamp,
                kind="a" * 51,
                stream_id="test_stream",
                payload="test",
            )

        # Max length stream_id (100 chars)
        max_stream_id = "a" * 100
        event = Event[str](
            ts=fixed_timestamp,
            kind="test_event",
            stream_id=max_stream_id,
            payload="test",
        )
        assert event.stream_id == max_stream_id

        # Exceeds max length stream_id (101 chars)
        with pytest.raises(ValidationError):
            Event[str](
                ts=fixed_timestamp,
                kind="test_event",
                stream_id="a" * 101,
                payload="test",
            )

    def test_unicode_characters(self, fixed_timestamp):
        """Test unicode characters are rejected."""
        invalid_kinds_with_unicode = [
            "goal_scored_🎯",  # emoji
            "événement",  # accented characters
            "событие",  # cyrillic
            "事件",  # chinese characters
            "イベント",  # japanese characters
        ]

        for kind in invalid_kinds_with_unicode:
            with pytest.raises(ValidationError) as exc_info:
                Event[str](
                    ts=fixed_timestamp,
                    kind=kind,
                    stream_id="test_stream",
                    payload="test",
                )
            assert "String should match pattern" in str(exc_info.value)
