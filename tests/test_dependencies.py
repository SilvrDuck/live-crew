"""Tests for dependency models.

Based on specification in .vibes/live_crew_spec.md section 2.4
"""

from pydantic import ValidationError
import pytest

from live_crew.core.dependencies import CrewDep, EventDep, Dependency


class TestCrewDep:
    """Test cases for CrewDep model."""

    def test_crew_dep_creation_valid(self):
        """Test creating a valid CrewDep."""
        dep = CrewDep(type="crew", crew="vision_crew")

        assert dep.type == "crew"
        assert dep.crew == "vision_crew"
        assert dep.offset == -1  # Default value

    def test_crew_dep_custom_offset(self):
        """Test CrewDep with custom offset."""
        dep = CrewDep(type="crew", crew="score_tracker", offset=-2)

        assert dep.type == "crew"
        assert dep.crew == "score_tracker"
        assert dep.offset == -2

    def test_crew_dep_validation_missing_crew(self):
        """Test validation fails when crew field is missing."""
        with pytest.raises(ValidationError) as exc_info:
            CrewDep(type="crew")  # type: ignore[call-arg]  # Intentionally missing crew
        assert "crew" in str(exc_info.value)

    def test_crew_dep_validation_wrong_type(self):
        """Test validation fails with wrong type literal."""
        with pytest.raises(ValidationError):
            CrewDep(type="event", crew="vision_crew")  # type: ignore[arg-type]  # Wrong type

    def test_crew_dep_immutable(self):
        """Test that CrewDep is immutable."""
        dep = CrewDep(type="crew", crew="test_crew")

        with pytest.raises(ValidationError):
            dep.crew = "modified_crew"


class TestEventDep:
    """Test cases for EventDep model."""

    def test_event_dep_creation_valid(self):
        """Test creating a valid EventDep."""
        dep = EventDep(type="event", event="goal_scored")

        assert dep.type == "event"
        assert dep.event == "goal_scored"
        assert dep.offset == 0  # Default value

    def test_event_dep_custom_offset(self):
        """Test EventDep with custom offset."""
        dep = EventDep(type="event", event="frame_embedding", offset=1)

        assert dep.type == "event"
        assert dep.event == "frame_embedding"
        assert dep.offset == 1

    def test_event_dep_validation_missing_event(self):
        """Test validation fails when event field is missing."""
        with pytest.raises(ValidationError) as exc_info:
            EventDep(type="event")  # type: ignore[call-arg]  # Intentionally missing event
        assert "event" in str(exc_info.value)

    def test_event_dep_validation_wrong_type(self):
        """Test validation fails with wrong type literal."""
        with pytest.raises(ValidationError):
            EventDep(type="crew", event="goal_scored")  # type: ignore[arg-type]  # Wrong type

    def test_event_dep_immutable(self):
        """Test that EventDep is immutable."""
        dep = EventDep(type="event", event="test_event")

        with pytest.raises(ValidationError):
            dep.event = "modified_event"


class TestDependency:
    """Test cases for Dependency discriminated union."""

    def test_dependency_crew_type(self):
        """Test Dependency with CrewDep."""
        crew_data = {"type": "crew", "crew": "vision_crew", "offset": -1}
        dep: Dependency = CrewDep(**crew_data)

        assert isinstance(dep, CrewDep)
        assert dep.type == "crew"
        assert dep.crew == "vision_crew"
        assert dep.offset == -1

    def test_dependency_event_type(self):
        """Test Dependency with EventDep."""
        event_data = {"type": "event", "event": "goal_scored", "offset": 0}
        dep: Dependency = EventDep(**event_data)

        assert isinstance(dep, EventDep)
        assert dep.type == "event"
        assert dep.event == "goal_scored"
        assert dep.offset == 0

    def test_dependency_json_serialization_crew(self):
        """Test Dependency JSON serialization for CrewDep."""
        dep = CrewDep(type="crew", crew="commentator", offset=-2)

        json_str = dep.model_dump_json()
        assert "crew" in json_str
        assert "commentator" in json_str
        assert "-2" in json_str

        # Test deserialization
        parsed = CrewDep.model_validate_json(json_str)
        assert parsed.crew == dep.crew
        assert parsed.offset == dep.offset

    def test_dependency_json_serialization_event(self):
        """Test Dependency JSON serialization for EventDep."""
        dep = EventDep(type="event", event="frame_update", offset=1)

        json_str = dep.model_dump_json()
        assert "event" in json_str
        assert "frame_update" in json_str
        assert '"offset":1' in json_str

        # Test deserialization
        parsed = EventDep.model_validate_json(json_str)
        assert parsed.event == dep.event
        assert parsed.offset == dep.offset

    def test_dependency_discriminator_validation(self):
        """Test that discriminator properly validates type field."""
        # This would typically be used when parsing from dict/JSON
        crew_dict = {"type": "crew", "crew": "test_crew"}
        event_dict = {"type": "event", "event": "test_event"}

        # Create instances directly (discriminator is mainly for parsing)
        crew_dep = CrewDep(**crew_dict)
        event_dep = EventDep(**event_dict)

        assert crew_dep.type == "crew"
        assert event_dep.type == "event"


class TestDependencyEdgeCases:
    """Test edge cases for dependency models."""

    def test_crew_dep_negative_offsets(self):
        """Test CrewDep with various negative offsets."""
        for offset in [-10, -5, -1]:
            dep = CrewDep(type="crew", crew="test_crew", offset=offset)
            assert dep.offset == offset

    def test_event_dep_positive_offsets(self):
        """Test EventDep with various positive offsets."""
        for offset in [0, 1, 5, 10]:
            dep = EventDep(type="event", event="test_event", offset=offset)
            assert dep.offset == offset

    def test_crew_dep_empty_crew_name(self):
        """Test validation of empty crew name."""
        with pytest.raises(ValidationError):
            CrewDep(type="crew", crew="")

    def test_event_dep_empty_event_name(self):
        """Test validation of empty event name."""
        with pytest.raises(ValidationError):
            EventDep(type="event", event="")

    def test_crew_dep_whitespace_crew_name(self):
        """Test validation of crew name with whitespace."""
        with pytest.raises(ValidationError):
            CrewDep(type="crew", crew="  vision_crew  ")

    def test_event_dep_whitespace_event_name(self):
        """Test validation of event name with whitespace."""
        with pytest.raises(ValidationError):
            EventDep(type="event", event="  goal_scored  ")


class TestDependencyUseCases:
    """Test real-world use cases for dependencies."""

    def test_commentator_dependency_pattern(self):
        """Test typical commentator crew dependency pattern."""
        # Commentator depends on vision crew from previous slice
        dep = CrewDep(type="crew", crew="vision_crew", offset=-1)

        assert dep.type == "crew"
        assert dep.crew == "vision_crew"
        assert dep.offset == -1

    def test_score_tracker_dependency_pattern(self):
        """Test typical score tracker event dependency pattern."""
        # Score tracker depends on goal_scored events in current slice
        dep = EventDep(type="event", event="goal_scored", offset=0)

        assert dep.type == "event"
        assert dep.event == "goal_scored"
        assert dep.offset == 0

    def test_overlay_future_dependency_pattern(self):
        """Test overlay waiting for future frame updates."""
        # Overlay might wait for next frame update
        dep = EventDep(type="event", event="frame_update", offset=1)

        assert dep.type == "event"
        assert dep.event == "frame_update"
        assert dep.offset == 1

    def test_multiple_dependencies_list(self):
        """Test creating list of mixed dependencies."""
        dependencies = [
            CrewDep(type="crew", crew="vision_crew", offset=-1),
            EventDep(type="event", event="goal_scored", offset=0),
            CrewDep(type="crew", crew="score_tracker", offset=-1),
        ]

        # Type checking works correctly
        crew_deps = [d for d in dependencies if isinstance(d, CrewDep)]
        event_deps = [d for d in dependencies if isinstance(d, EventDep)]

        assert len(crew_deps) == 2
        assert len(event_deps) == 1
        assert all(d.type == "crew" for d in crew_deps)
        assert all(d.type == "event" for d in event_deps)
