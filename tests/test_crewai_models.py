"""Tests for CrewAI integration models.

Comprehensive test coverage for Pydantic models used in CrewAI integration,
including validation, constraints, and error handling.
"""

import pytest
from pydantic import ValidationError

from live_crew.crewai_integration.models import (
    CrewRuntimeConfig,
    CrewOrchestrationConfig,
    CrewConfig,
)
from live_crew.core.dependencies import CrewDep, EventDep


class TestCrewRuntimeConfig:
    """Test cases for CrewRuntimeConfig model validation."""

    def test_crew_runtime_config_valid_minimal(self):
        """Test creating CrewRuntimeConfig with minimal valid data."""
        config = CrewRuntimeConfig(crew="test_crew", triggers=["user_input"])

        assert config.crew == "test_crew"
        assert config.triggers == ["user_input"]
        assert config.needs is None
        assert config.wait_policy == "none"
        assert config.timeout_ms == 5000
        assert config.slice_stride == 1

    def test_crew_runtime_config_valid_full(self):
        """Test creating CrewRuntimeConfig with all fields specified."""
        dependencies = [
            CrewDep(type="crew", crew="other_crew", offset=-1),
            EventDep(type="event", event="data_ready", offset=0),
        ]

        config = CrewRuntimeConfig(
            crew="analysis_crew",
            triggers=["data_received", "manual_trigger"],
            needs=dependencies,
            wait_policy="all",
            timeout_ms=30000,
            slice_stride=3,
        )

        assert config.crew == "analysis_crew"
        assert config.triggers == ["data_received", "manual_trigger"]
        assert config.needs == dependencies
        assert config.wait_policy == "all"
        assert config.timeout_ms == 30000
        assert config.slice_stride == 3

    def test_crew_runtime_config_empty_triggers_invalid(self):
        """Test that empty triggers list raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            CrewRuntimeConfig(
                crew="test_crew",
                triggers=[],  # Empty list should fail min_length=1
            )

        assert "at least 1 item" in str(exc_info.value).lower()

    def test_crew_runtime_config_invalid_trigger_pattern(self):
        """Test that invalid trigger patterns raise ValidationError."""
        invalid_triggers = [
            ["invalid-trigger"],  # Contains hyphen
            ["invalid trigger"],  # Contains space
            ["invalid@trigger"],  # Contains special char
            ["123invalid!"],  # Contains exclamation
            ["trigger.name"],  # Contains dot
        ]

        for triggers in invalid_triggers:
            with pytest.raises(ValidationError) as exc_info:
                CrewRuntimeConfig(crew="test_crew", triggers=triggers)

            assert "must match pattern" in str(exc_info.value)

    def test_crew_runtime_config_valid_trigger_patterns(self):
        """Test that valid trigger patterns are accepted."""
        valid_triggers = [
            ["simple_trigger"],
            ["trigger123"],
            ["UPPERCASE_TRIGGER"],
            ["mixed_Case123"],
            ["numbers_123_456"],
            ["single_underscore_"],
            ["multiple_underscores___"],
        ]

        for triggers in valid_triggers:
            config = CrewRuntimeConfig(crew="test_crew", triggers=triggers)
            assert config.triggers == triggers

    def test_crew_runtime_config_timeout_bounds(self):
        """Test timeout_ms boundary validation."""
        # Test minimum boundary (must be > 0)
        with pytest.raises(ValidationError):
            CrewRuntimeConfig(crew="test_crew", triggers=["test"], timeout_ms=0)

        with pytest.raises(ValidationError):
            CrewRuntimeConfig(crew="test_crew", triggers=["test"], timeout_ms=-1000)

        # Test maximum boundary (must be <= 300000)
        with pytest.raises(ValidationError):
            CrewRuntimeConfig(crew="test_crew", triggers=["test"], timeout_ms=300001)

        # Test valid boundaries
        config_min = CrewRuntimeConfig(
            crew="test_crew", triggers=["test"], timeout_ms=1
        )
        assert config_min.timeout_ms == 1

        config_max = CrewRuntimeConfig(
            crew="test_crew", triggers=["test"], timeout_ms=300000
        )
        assert config_max.timeout_ms == 300000

    def test_crew_runtime_config_slice_stride_validation(self):
        """Test slice_stride validation (must be >= 1)."""
        # Test invalid values
        with pytest.raises(ValidationError):
            CrewRuntimeConfig(crew="test_crew", triggers=["test"], slice_stride=0)

        with pytest.raises(ValidationError):
            CrewRuntimeConfig(crew="test_crew", triggers=["test"], slice_stride=-5)

        # Test valid values
        config = CrewRuntimeConfig(crew="test_crew", triggers=["test"], slice_stride=10)
        assert config.slice_stride == 10

    def test_crew_runtime_config_wait_policy_validation(self):
        """Test wait_policy literal validation."""
        valid_policies = ["any", "all", "none"]

        for policy in valid_policies:
            config = CrewRuntimeConfig(
                crew="test_crew", triggers=["test"], wait_policy=policy
            )
            assert config.wait_policy == policy

        # Test invalid policy
        with pytest.raises(ValidationError):
            CrewRuntimeConfig(
                crew="test_crew",
                triggers=["test"],
                wait_policy="invalid_policy",  # type: ignore  # Testing validation
            )

    def test_crew_runtime_config_missing_required_fields(self):
        """Test that missing required fields raise ValidationError."""
        # Missing crew
        with pytest.raises(ValidationError):
            CrewRuntimeConfig(triggers=["test"])  # type: ignore  # Testing validation

        # Missing triggers
        with pytest.raises(ValidationError):
            CrewRuntimeConfig(crew="test_crew")  # type: ignore  # Testing validation

    def test_crew_runtime_config_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            CrewRuntimeConfig(
                crew="test_crew",
                triggers=["test"],
                extra_field="not_allowed",  # type: ignore  # Testing validation
            )

    def test_crew_runtime_config_dependencies_mixed_types(self):
        """Test CrewRuntimeConfig with mixed dependency types."""
        dependencies = [
            CrewDep(type="crew", crew="crew1", offset=-1),
            EventDep(type="event", event="event1", offset=0),
            CrewDep(type="crew", crew="crew2", offset=-2),
        ]

        config = CrewRuntimeConfig(
            crew="test_crew", triggers=["test"], needs=dependencies
        )

        assert config.needs is not None
        assert len(config.needs) == 3
        assert config.needs[0].type == "crew"
        assert config.needs[1].type == "event"
        assert config.needs[2].type == "crew"


class TestCrewOrchestrationConfig:
    """Test cases for CrewOrchestrationConfig model validation."""

    def test_crew_orchestration_config_valid_minimal(self):
        """Test creating CrewOrchestrationConfig with minimal valid data."""
        crew_config = CrewConfig(path="/path/to/crew", runtime="crew.runtime.yaml")

        config = CrewOrchestrationConfig(crews=[crew_config])

        assert len(config.crews) == 1
        assert config.crews[0] == crew_config
        assert config.slice_ms is None

    def test_crew_orchestration_config_valid_full(self):
        """Test creating CrewOrchestrationConfig with all fields specified."""
        crew_configs = [
            CrewConfig(path="/crew1", runtime="runtime1.yaml"),
            CrewConfig(path="/crew2", runtime="runtime2.yaml"),
        ]

        config = CrewOrchestrationConfig(crews=crew_configs, slice_ms=1000)

        assert config.crews == crew_configs
        assert config.slice_ms == 1000

    def test_crew_orchestration_config_empty_crews_invalid(self):
        """Test that empty crews list raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            CrewOrchestrationConfig(crews=[])

        assert "at least 1 item" in str(exc_info.value).lower()

    def test_crew_orchestration_config_slice_ms_validation(self):
        """Test slice_ms boundary validation."""
        crew_config = CrewConfig(path="/crew", runtime="runtime.yaml")

        # Test invalid values
        with pytest.raises(ValidationError):
            CrewOrchestrationConfig(crews=[crew_config], slice_ms=0)

        with pytest.raises(ValidationError):
            CrewOrchestrationConfig(crews=[crew_config], slice_ms=-1000)

        # Test valid values
        config = CrewOrchestrationConfig(crews=[crew_config], slice_ms=500)
        assert config.slice_ms == 500

    def test_crew_orchestration_config_missing_crews(self):
        """Test that missing crews field raises ValidationError."""
        with pytest.raises(ValidationError):
            CrewOrchestrationConfig()  # type: ignore  # Testing validation


class TestCrewConfig:
    """Test cases for CrewConfig model validation."""

    def test_crew_config_valid(self):
        """Test creating valid CrewConfig."""
        config = CrewConfig(path="/path/to/crew/directory", runtime="crew.runtime.yaml")

        assert config.path == "/path/to/crew/directory"
        assert config.runtime == "crew.runtime.yaml"

    def test_crew_config_relative_runtime_path(self):
        """Test CrewConfig with relative runtime path."""
        config = CrewConfig(path="/crews/analysis", runtime="config/runtime.yaml")

        assert config.path == "/crews/analysis"
        assert config.runtime == "config/runtime.yaml"

    def test_crew_config_absolute_runtime_path(self):
        """Test CrewConfig with absolute runtime path."""
        config = CrewConfig(
            path="/crews/processing", runtime="/configs/global/runtime.yaml"
        )

        assert config.path == "/crews/processing"
        assert config.runtime == "/configs/global/runtime.yaml"

    def test_crew_config_missing_required_fields(self):
        """Test that missing required fields raise ValidationError."""
        # Missing path
        with pytest.raises(ValidationError):
            CrewConfig(runtime="runtime.yaml")  # type: ignore  # Testing validation

        # Missing runtime
        with pytest.raises(ValidationError):
            CrewConfig(path="/path/to/crew")  # type: ignore  # Testing validation

    def test_crew_config_empty_string_fields(self):
        """Test that empty string fields raise ValidationError."""
        with pytest.raises(ValidationError):
            CrewConfig(path="", runtime="runtime.yaml")

        with pytest.raises(ValidationError):
            CrewConfig(path="/valid/path", runtime="")

    def test_crew_config_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            CrewConfig(
                path="/path",
                runtime="runtime.yaml",
                extra_field="not_allowed",  # type: ignore  # Testing validation
            )


class TestComplexScenarios:
    """Test complex scenarios involving multiple model interactions."""

    def test_full_orchestration_config_complex(self):
        """Test complete orchestration configuration with multiple crews and complex dependencies."""
        # Create complex orchestration config
        crew1_config = CrewConfig(
            path="/crews/data_ingestion", runtime="ingestion.runtime.yaml"
        )

        crew2_config = CrewConfig(
            path="/crews/data_analysis", runtime="analysis.runtime.yaml"
        )

        crew3_config = CrewConfig(
            path="/crews/reporting", runtime="reporting.runtime.yaml"
        )

        orchestration = CrewOrchestrationConfig(
            crews=[crew1_config, crew2_config, crew3_config], slice_ms=2000
        )

        assert len(orchestration.crews) == 3
        assert orchestration.slice_ms == 2000

        # Verify each crew config
        assert orchestration.crews[0].path == "/crews/data_ingestion"
        assert orchestration.crews[1].path == "/crews/data_analysis"
        assert orchestration.crews[2].path == "/crews/reporting"

    def test_runtime_config_with_complex_dependencies(self):
        """Test CrewRuntimeConfig with complex dependency chains."""
        dependencies = [
            CrewDep(type="crew", crew="data_loader", offset=-1),
            CrewDep(type="crew", crew="validator", offset=-1),
            EventDep(type="event", event="external_trigger", offset=0),
            EventDep(type="event", event="scheduled_run", offset=-2),
        ]

        config = CrewRuntimeConfig(
            crew="complex_processor",
            triggers=["data_ready", "manual_start", "scheduled_trigger"],
            needs=dependencies,
            wait_policy="all",
            timeout_ms=120000,
            slice_stride=5,
        )

        assert config.crew == "complex_processor"
        assert len(config.triggers) == 3
        assert config.needs is not None
        assert len(config.needs) == 4
        assert config.wait_policy == "all"
        assert config.timeout_ms == 120000
        assert config.slice_stride == 5

        # Verify dependency types
        assert config.needs is not None
        crew_deps = [dep for dep in config.needs if dep.type == "crew"]
        event_deps = [dep for dep in config.needs if dep.type == "event"]

        assert len(crew_deps) == 2
        assert len(event_deps) == 2

    def test_edge_case_unicode_and_special_chars(self):
        """Test model behavior with Unicode and edge case string inputs."""
        # Test unicode in crew names (should fail validation due to pattern)
        with pytest.raises(ValidationError):
            CrewRuntimeConfig(
                crew="test_crew",
                triggers=["tëst_trigge̊r"],  # Contains unicode characters
            )

        # Test extremely long strings
        very_long_crew_name = "a" * 1000
        config = CrewRuntimeConfig(crew=very_long_crew_name, triggers=["test"])
        assert config.crew == very_long_crew_name

        # Test paths with various formats
        path_formats = [
            "/simple/path",
            "/path/with spaces/in it",
            "/path/with-dashes",
            "/path/with.dots.and.extensions",
            "relative/path",
            "C:\\Windows\\Path\\Style",
        ]

        for path in path_formats:
            config = CrewConfig(path=path, runtime="runtime.yaml")
            assert config.path == path

    def test_boundary_values_comprehensive(self):
        """Test comprehensive boundary value scenarios."""
        # Minimum timeout (1ms)
        config_min = CrewRuntimeConfig(
            crew="test_crew", triggers=["test"], timeout_ms=1
        )
        assert config_min.timeout_ms == 1

        # Maximum timeout (300000ms = 5 minutes)
        config_max = CrewRuntimeConfig(
            crew="test_crew", triggers=["test"], timeout_ms=300000
        )
        assert config_max.timeout_ms == 300000

        # Large slice stride
        config_stride = CrewRuntimeConfig(
            crew="test_crew", triggers=["test"], slice_stride=1000
        )
        assert config_stride.slice_stride == 1000

        # Large number of triggers
        many_triggers = [f"trigger_{i}" for i in range(100)]
        config_triggers = CrewRuntimeConfig(crew="test_crew", triggers=many_triggers)
        assert len(config_triggers.triggers) == 100
        assert config_triggers.triggers[99] == "trigger_99"
