"""Tests for CrewAI loader supporting both YAML and Python crew loading.

Comprehensive test coverage for CrewAILoader including YAML crew loading,
Python crew loading, configuration validation, and error handling.
"""

import pytest
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, mock_open


from live_crew.crewai_integration.loader import (
    CrewAILoader,
    CrewAILoadError,
    CrewAIConfigError,
)
from live_crew.crewai_integration.wrapper import CrewAIWrapper
from live_crew.crewai_integration.models import CrewRuntimeConfig


class TestCrewAILoaderYAMLCrew:
    """Test cases for YAML-based CrewAI crew loading."""

    @pytest.fixture
    def valid_runtime_config_dict(self):
        """Provide a valid runtime configuration dictionary."""
        return {
            "crew": "test_yaml_crew",
            "triggers": ["data_received", "manual_trigger"],
            "needs": [
                {"type": "crew", "crew": "data_loader", "offset": -1},
                {"type": "event", "event": "validation_complete", "offset": 0},
            ],
            "wait_policy": "all",
            "timeout_ms": 30000,
            "slice_stride": 2,
        }

    @pytest.fixture
    def mock_crew_class(self):
        """Create a mock CrewAI crew class with required attributes."""
        crew_class = Mock()
        crew_class.__name__ = "TestCrew"
        crew_class.agents_config = "mock_agents_config"
        crew_class.tasks_config = "mock_tasks_config"

        # Mock crew instance
        crew_instance = Mock()
        crew_instance.crew = Mock(return_value="mock_crew_object")
        crew_class.return_value = crew_instance

        return crew_class

    @pytest.fixture
    def mock_crew_module(self, mock_crew_class):
        """Create a mock crew module with a CrewAI crew class."""
        module = Mock()
        module.TestCrew = mock_crew_class
        # Configure dir() to return crew class names
        module.__dir__ = Mock(return_value=["TestCrew"])
        return module

    def test_load_yaml_crew_success(self, valid_runtime_config_dict, mock_crew_module):
        """Test successful YAML crew loading with all components working."""
        crew_path = Path("/test/crew/path")
        runtime_config_path = Path("/test/crew/path/runtime.yaml")

        with (
            patch.object(CrewAILoader, "_load_runtime_config") as mock_load_config,
            patch.object(CrewAILoader, "_load_crew_module") as mock_load_module,
            patch.object(
                CrewAILoader, "_instantiate_crew_from_module"
            ) as mock_instantiate,
        ):
            # Setup mocks
            mock_runtime_config = CrewRuntimeConfig(**valid_runtime_config_dict)
            mock_load_config.return_value = mock_runtime_config
            mock_load_module.return_value = mock_crew_module
            mock_instantiate.return_value = "mock_crewai_crew_object"

            # Load the crew
            wrapper = CrewAILoader.load_yaml_crew(crew_path, runtime_config_path)

            # Verify method calls
            mock_load_config.assert_called_once_with(runtime_config_path)
            mock_load_module.assert_called_once_with(crew_path)
            mock_instantiate.assert_called_once_with(mock_crew_module)

            # Verify wrapper properties
            assert isinstance(wrapper, CrewAIWrapper)
            assert wrapper.crew_id == "test_yaml_crew"
            assert wrapper.crewai_crew == "mock_crewai_crew_object"
            assert wrapper.triggers == ["data_received", "manual_trigger"]
            assert wrapper.timeout_ms == 30000

    def test_load_yaml_crew_config_load_failure(self):
        """Test YAML crew loading when runtime config loading fails."""
        crew_path = Path("/test/crew/path")
        runtime_config_path = Path("/test/crew/path/runtime.yaml")

        with patch.object(CrewAILoader, "_load_runtime_config") as mock_load_config:
            mock_load_config.side_effect = CrewAIConfigError("Invalid config")

            with pytest.raises(CrewAILoadError) as exc_info:
                CrewAILoader.load_yaml_crew(crew_path, runtime_config_path)

            error_msg = str(exc_info.value)
            assert "Failed to load YAML-configured CrewAI crew" in error_msg
            assert str(crew_path) in error_msg

    def test_load_yaml_crew_module_load_failure(self, valid_runtime_config_dict):
        """Test YAML crew loading when crew module loading fails."""
        crew_path = Path("/test/crew/path")
        runtime_config_path = Path("/test/crew/path/runtime.yaml")

        with (
            patch.object(CrewAILoader, "_load_runtime_config") as mock_load_config,
            patch.object(CrewAILoader, "_load_crew_module") as mock_load_module,
        ):
            mock_runtime_config = CrewRuntimeConfig(**valid_runtime_config_dict)
            mock_load_config.return_value = mock_runtime_config
            mock_load_module.side_effect = CrewAILoadError("Module load failed")

            with pytest.raises(CrewAILoadError) as exc_info:
                CrewAILoader.load_yaml_crew(crew_path, runtime_config_path)

            error_msg = str(exc_info.value)
            assert "Failed to load YAML-configured CrewAI crew" in error_msg

    def test_load_yaml_crew_instantiation_failure(
        self, valid_runtime_config_dict, mock_crew_module
    ):
        """Test YAML crew loading when crew instantiation fails."""
        crew_path = Path("/test/crew/path")
        runtime_config_path = Path("/test/crew/path/runtime.yaml")

        with (
            patch.object(CrewAILoader, "_load_runtime_config") as mock_load_config,
            patch.object(CrewAILoader, "_load_crew_module") as mock_load_module,
            patch.object(
                CrewAILoader, "_instantiate_crew_from_module"
            ) as mock_instantiate,
        ):
            mock_runtime_config = CrewRuntimeConfig(**valid_runtime_config_dict)
            mock_load_config.return_value = mock_runtime_config
            mock_load_module.return_value = mock_crew_module
            mock_instantiate.side_effect = CrewAILoadError("Instantiation failed")

            with pytest.raises(CrewAILoadError) as exc_info:
                CrewAILoader.load_yaml_crew(crew_path, runtime_config_path)

            error_msg = str(exc_info.value)
            assert "Failed to load YAML-configured CrewAI crew" in error_msg


class TestCrewAILoaderPythonCrew:
    """Test cases for Python-based CrewAI crew loading."""

    def test_load_python_crew_success_minimal(self):
        """Test successful Python crew loading with minimal configuration."""
        crew_id = "python_test_crew"
        mock_crewai_crew = Mock()
        runtime_config = {"triggers": ["python_event"], "timeout_ms": 10000}

        wrapper = CrewAILoader.load_python_crew(
            crew_id, mock_crewai_crew, runtime_config
        )

        assert isinstance(wrapper, CrewAIWrapper)
        assert wrapper.crew_id == crew_id
        assert wrapper.crewai_crew == mock_crewai_crew
        assert wrapper.triggers == ["python_event"]
        assert wrapper.timeout_ms == 10000

    def test_load_python_crew_success_full_config(self):
        """Test successful Python crew loading with full configuration."""
        crew_id = "complex_python_crew"
        mock_crewai_crew = Mock()
        runtime_config = {
            "triggers": ["event1", "event2", "event3"],
            "needs": [{"type": "crew", "crew": "dependency_crew", "offset": -1}],
            "wait_policy": "any",
            "timeout_ms": 60000,
            "slice_stride": 5,
        }

        wrapper = CrewAILoader.load_python_crew(
            crew_id, mock_crewai_crew, runtime_config
        )

        assert wrapper.crew_id == crew_id
        assert wrapper.triggers == ["event1", "event2", "event3"]
        assert wrapper.timeout_ms == 60000

    def test_load_python_crew_invalid_config(self):
        """Test Python crew loading with invalid runtime configuration."""
        crew_id = "invalid_crew"
        mock_crewai_crew = Mock()

        # Test various invalid configurations
        invalid_configs = [
            {},  # Missing triggers
            {"triggers": []},  # Empty triggers
            {"triggers": ["valid"], "timeout_ms": 0},  # Invalid timeout
            {"triggers": ["valid"], "wait_policy": "invalid"},  # Invalid wait policy
            {"triggers": ["invalid-trigger"]},  # Invalid trigger pattern
            {"triggers": ["valid"], "slice_stride": 0},  # Invalid slice stride
        ]

        for invalid_config in invalid_configs:
            with pytest.raises(CrewAIConfigError) as exc_info:
                CrewAILoader.load_python_crew(crew_id, mock_crewai_crew, invalid_config)

            error_msg = str(exc_info.value)
            assert f"Invalid runtime configuration for crew '{crew_id}'" in error_msg

    def test_load_python_crew_none_crew(self):
        """Test Python crew loading with None crew object."""
        crew_id = "none_crew"
        runtime_config = {"triggers": ["test"]}

        # Should not fail during loading, but may cause issues during execution
        wrapper = CrewAILoader.load_python_crew(crew_id, None, runtime_config)
        assert wrapper.crewai_crew is None

    def test_load_python_crew_generic_exception(self):
        """Test Python crew loading with unexpected exception during validation."""
        crew_id = "exception_crew"
        mock_crewai_crew = Mock()

        # Create a config that will cause validation to raise an unexpected exception
        runtime_config = {"triggers": ["test"]}

        with patch(
            "live_crew.crewai_integration.loader.CrewRuntimeConfig"
        ) as mock_config_class:
            mock_config_class.side_effect = RuntimeError("Unexpected error")

            with pytest.raises(CrewAILoadError) as exc_info:
                CrewAILoader.load_python_crew(crew_id, mock_crewai_crew, runtime_config)

            error_msg = str(exc_info.value)
            assert f"Failed to load Python-defined CrewAI crew '{crew_id}'" in error_msg

    def test_load_python_crew_config_merge(self):
        """Test that crew_id is correctly merged into runtime configuration."""
        crew_id = "merge_test_crew"
        mock_crewai_crew = Mock()
        runtime_config = {"triggers": ["test_trigger"], "timeout_ms": 15000}

        with patch(
            "live_crew.crewai_integration.loader.CrewRuntimeConfig"
        ) as mock_config_class:
            mock_config_instance = Mock()
            mock_config_instance.triggers = ["test_trigger"]
            mock_config_instance.timeout_ms = 15000
            mock_config_class.return_value = mock_config_instance

            CrewAILoader.load_python_crew(crew_id, mock_crewai_crew, runtime_config)

            # Verify that crew_id was added to config dict before validation
            mock_config_class.assert_called_once_with(
                crew="merge_test_crew", triggers=["test_trigger"], timeout_ms=15000
            )


class TestCrewAILoaderRuntimeConfig:
    """Test cases for runtime configuration loading and validation."""

    def test_load_runtime_config_success(self):
        """Test successful runtime configuration loading from YAML."""
        config_data = {
            "crew": "yaml_config_crew",
            "triggers": ["config_event"],
            "timeout_ms": 20000,
        }

        yaml_content = yaml.dump(config_data)

        with (
            patch("builtins.open", mock_open(read_data=yaml_content)),
            patch("pathlib.Path.exists", return_value=True),
        ):
            config_path = Path("/test/config.yaml")
            runtime_config = CrewAILoader._load_runtime_config(config_path)

            assert isinstance(runtime_config, CrewRuntimeConfig)
            assert runtime_config.crew == "yaml_config_crew"
            assert runtime_config.triggers == ["config_event"]
            assert runtime_config.timeout_ms == 20000

    def test_load_runtime_config_file_not_found(self):
        """Test runtime configuration loading when file doesn't exist."""
        config_path = Path("/nonexistent/config.yaml")

        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(CrewAIConfigError) as exc_info:
                CrewAILoader._load_runtime_config(config_path)

            error_msg = str(exc_info.value)
            assert "Runtime config file not found" in error_msg
            assert str(config_path) in error_msg

    def test_load_runtime_config_invalid_yaml(self):
        """Test runtime configuration loading with invalid YAML content."""
        invalid_yaml = """
        crew: test_crew
        triggers: [event1, event2
        # Missing closing bracket
        timeout_ms: 5000
        """

        with (
            patch("builtins.open", mock_open(read_data=invalid_yaml)),
            patch("pathlib.Path.exists", return_value=True),
        ):
            config_path = Path("/test/invalid.yaml")

            with pytest.raises(CrewAIConfigError) as exc_info:
                CrewAILoader._load_runtime_config(config_path)

            error_msg = str(exc_info.value)
            assert "Invalid YAML in runtime config" in error_msg

    def test_load_runtime_config_validation_failure(self):
        """Test runtime configuration loading with validation failures."""
        invalid_configs = [
            # Missing required fields
            {"crew": "test_crew"},  # Missing triggers
            {"triggers": ["test"]},  # Missing crew
            # Invalid field values
            {"crew": "test_crew", "triggers": []},  # Empty triggers
            {
                "crew": "test_crew",
                "triggers": ["test"],
                "timeout_ms": 0,
            },  # Invalid timeout
            {
                "crew": "test_crew",
                "triggers": ["invalid-trigger"],
            },  # Invalid trigger pattern
        ]

        for invalid_config in invalid_configs:
            yaml_content = yaml.dump(invalid_config)

            with (
                patch("builtins.open", mock_open(read_data=yaml_content)),
                patch("pathlib.Path.exists", return_value=True),
            ):
                config_path = Path("/test/invalid_config.yaml")

                with pytest.raises(CrewAIConfigError) as exc_info:
                    CrewAILoader._load_runtime_config(config_path)

                error_msg = str(exc_info.value)
                assert "Invalid runtime configuration" in error_msg
                assert str(config_path) in error_msg

    def test_load_runtime_config_complex_valid(self):
        """Test loading complex but valid runtime configuration."""
        complex_config = {
            "crew": "complex_crew",
            "triggers": ["event_a", "event_b", "event_c"],
            "needs": [
                {"type": "crew", "crew": "crew1", "offset": -1},
                {"type": "crew", "crew": "crew2", "offset": -2},
                {"type": "event", "event": "external_event", "offset": 0},
            ],
            "wait_policy": "all",
            "timeout_ms": 120000,
            "slice_stride": 10,
        }

        yaml_content = yaml.dump(complex_config)

        with (
            patch("builtins.open", mock_open(read_data=yaml_content)),
            patch("pathlib.Path.exists", return_value=True),
        ):
            config_path = Path("/test/complex.yaml")
            runtime_config = CrewAILoader._load_runtime_config(config_path)

            assert runtime_config.crew == "complex_crew"
            assert len(runtime_config.triggers) == 3
            assert runtime_config.needs is not None
            assert len(runtime_config.needs) == 3
            assert runtime_config.wait_policy == "all"
            assert runtime_config.timeout_ms == 120000
            assert runtime_config.slice_stride == 10

    def test_load_runtime_config_encoding_handling(self):
        """Test runtime configuration loading with different encodings and special characters."""
        config_with_unicode = {
            "crew": "unicode_crew_测试",
            "triggers": ["event_测试", "trigger_café"],
            "timeout_ms": 5000,
        }

        yaml_content = yaml.dump(config_with_unicode, allow_unicode=True)

        with (
            patch("builtins.open", mock_open(read_data=yaml_content)),
            patch("pathlib.Path.exists", return_value=True),
        ):
            config_path = Path("/test/unicode.yaml")

            # Should fail validation due to trigger pattern restrictions
            with pytest.raises(CrewAIConfigError) as exc_info:
                CrewAILoader._load_runtime_config(config_path)

            error_msg = str(exc_info.value)
            assert "must match pattern" in error_msg


class TestCrewAILoaderModuleHandling:
    """Test cases for crew module loading and instantiation."""

    def test_load_crew_module_success(self):
        """Test successful crew module loading."""
        crew_path = Path("/test/crew")

        # Mock the module loading process
        mock_spec = Mock()
        mock_loader = Mock()
        mock_spec.loader = mock_loader
        mock_module = Mock()

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("importlib.util.spec_from_file_location", return_value=mock_spec),
            patch("importlib.util.module_from_spec", return_value=mock_module),
        ):
            result = CrewAILoader._load_crew_module(crew_path)

            # Verify the module loading process
            mock_loader.exec_module.assert_called_once_with(mock_module)
            assert result == mock_module

    def test_load_crew_module_file_not_found(self):
        """Test crew module loading when crew.py doesn't exist."""
        crew_path = Path("/test/crew")

        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(CrewAILoadError) as exc_info:
                CrewAILoader._load_crew_module(crew_path)

            error_msg = str(exc_info.value)
            assert "crew.py not found" in error_msg
            assert str(crew_path) in error_msg

    def test_load_crew_module_spec_creation_failure(self):
        """Test crew module loading when spec creation fails."""
        crew_path = Path("/test/crew")

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("importlib.util.spec_from_file_location", return_value=None),
        ):
            with pytest.raises(CrewAILoadError) as exc_info:
                CrewAILoader._load_crew_module(crew_path)

            error_msg = str(exc_info.value)
            assert "Cannot load module spec" in error_msg

    def test_load_crew_module_no_loader(self):
        """Test crew module loading when spec has no loader."""
        crew_path = Path("/test/crew")

        mock_spec = Mock()
        mock_spec.loader = None

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("importlib.util.spec_from_file_location", return_value=mock_spec),
        ):
            with pytest.raises(CrewAILoadError) as exc_info:
                CrewAILoader._load_crew_module(crew_path)

            error_msg = str(exc_info.value)
            assert "Cannot load module spec" in error_msg

    def test_load_crew_module_execution_failure(self):
        """Test crew module loading when module execution fails."""
        crew_path = Path("/test/crew")

        mock_spec = Mock()
        mock_loader = Mock()
        mock_loader.exec_module.side_effect = ImportError("Import failed")
        mock_spec.loader = mock_loader
        mock_module = Mock()

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("importlib.util.spec_from_file_location", return_value=mock_spec),
            patch("importlib.util.module_from_spec", return_value=mock_module),
        ):
            with pytest.raises(CrewAILoadError) as exc_info:
                CrewAILoader._load_crew_module(crew_path)

            error_msg = str(exc_info.value)
            assert "Failed to load crew module" in error_msg

    def test_instantiate_crew_from_module_success(self):
        """Test successful crew instantiation from module."""
        # Create mock crew instance and actual crew
        mock_crew_instance = Mock()
        mock_actual_crew = Mock()
        mock_crew_instance.crew = Mock(return_value=mock_actual_crew)

        # Create mock crew class using a real class to pass isinstance check
        class TestCrewClass:
            agents_config = "mock_agents"
            tasks_config = "mock_tasks"

            def __new__(cls):
                # Return the mock instance when instantiated
                return mock_crew_instance

        mock_crew_class = TestCrewClass

        # Create a simple module object
        import types

        mock_module = types.ModuleType("test_module")
        mock_module.TestCrewClass = mock_crew_class
        mock_module.other_attr = "not_crew"

        result = CrewAILoader._instantiate_crew_from_module(mock_module)

        # Verify the crew method was called and result is correct
        mock_crew_instance.crew.assert_called_once()
        assert result == mock_actual_crew

    def test_instantiate_crew_from_module_no_crew_classes(self):
        """Test crew instantiation when no valid crew classes found."""

        # Create class without crew config attributes
        class NotACrewClass:
            pass  # Missing agents_config and tasks_config attributes

        # Create module with no valid crew classes
        import types

        mock_module = types.ModuleType("test_module")
        mock_module.NotACrewClass = NotACrewClass

        with pytest.raises(CrewAILoadError) as exc_info:
            CrewAILoader._instantiate_crew_from_module(mock_module)

        error_msg = str(exc_info.value)
        assert "No @CrewBase decorated class found" in error_msg

    def test_instantiate_crew_from_module_multiple_crew_classes(self):
        """Test crew instantiation when multiple crew classes found."""

        # Create multiple crew classes with required attributes
        class CrewClass1:
            agents_config = "config1"
            tasks_config = "config1"

        class CrewClass2:
            agents_config = "config2"
            tasks_config = "config2"

        # Create module with multiple crew classes
        import types

        mock_module = types.ModuleType("test_module")
        mock_module.CrewClass1 = CrewClass1
        mock_module.CrewClass2 = CrewClass2

        with pytest.raises(CrewAILoadError) as exc_info:
            CrewAILoader._instantiate_crew_from_module(mock_module)

        error_msg = str(exc_info.value)
        assert "Multiple crew classes found" in error_msg
        assert "CrewClass1" in error_msg
        assert "CrewClass2" in error_msg

    def test_instantiate_crew_from_module_no_crew_method(self):
        """Test crew instantiation when crew class has no crew method/attribute."""

        # Create instance without crew method/attribute
        class CrewInstance:
            pass  # No 'crew' attribute or method

        # Create crew class that returns instance without crew
        class CrewWithoutMethod:
            agents_config = "config"
            tasks_config = "config"

            def __new__(cls):
                return CrewInstance()

        # Create module with crew class
        import types

        mock_module = types.ModuleType("test_module")
        mock_module.CrewWithoutMethod = CrewWithoutMethod

        with pytest.raises(CrewAILoadError) as exc_info:
            CrewAILoader._instantiate_crew_from_module(mock_module)

        error_msg = str(exc_info.value)
        assert "does not have a 'crew' method or attribute" in error_msg

    def test_instantiate_crew_from_module_crew_as_attribute(self):
        """Test crew instantiation when crew is an attribute (not callable)."""

        mock_actual_crew = "test_crew_object"  # Use a simple string instead of Mock

        # Create instance with crew as non-callable attribute
        class CrewInstance:
            def __init__(self):
                self.crew = mock_actual_crew  # Not callable, just an attribute

        # Create crew class
        class CrewWithAttribute:
            agents_config = "config"
            tasks_config = "config"

            def __new__(cls):
                return CrewInstance()

        # Create module with crew class
        import types

        mock_module = types.ModuleType("test_module")
        mock_module.CrewWithAttribute = CrewWithAttribute

        result = CrewAILoader._instantiate_crew_from_module(mock_module)
        assert result == mock_actual_crew

    def test_instantiate_crew_from_module_instantiation_failure(self):
        """Test crew instantiation when class instantiation fails."""

        # Create crew class that raises error on instantiation
        class FailingCrew:
            agents_config = "config"
            tasks_config = "config"

            def __new__(cls):
                raise RuntimeError("Instantiation failed")

        # Create module with failing crew class
        import types

        mock_module = types.ModuleType("test_module")
        mock_module.FailingCrew = FailingCrew

        with pytest.raises(CrewAILoadError) as exc_info:
            CrewAILoader._instantiate_crew_from_module(mock_module)

        error_msg = str(exc_info.value)
        assert "Failed to instantiate CrewAI crew" in error_msg


class TestCrewAILoaderExceptions:
    """Test cases for CrewAI loader exceptions."""

    def test_crew_ai_load_error_basic(self):
        """Test basic CrewAILoadError creation."""
        error_msg = "Load operation failed"
        error = CrewAILoadError(error_msg)

        assert str(error) == error_msg
        assert isinstance(error, Exception)

    def test_crew_ai_config_error_basic(self):
        """Test basic CrewAIConfigError creation."""
        error_msg = "Configuration is invalid"
        error = CrewAIConfigError(error_msg)

        assert str(error) == error_msg
        assert isinstance(error, Exception)

    def test_exception_inheritance_chain(self):
        """Test that exceptions have proper inheritance chain."""
        load_error = CrewAILoadError("Load failed")
        config_error = CrewAIConfigError("Config failed")

        assert isinstance(load_error, Exception)
        assert isinstance(config_error, Exception)

        # Test exception cause chain
        original_error = ValueError("Original issue")
        chained_error = CrewAILoadError("Chained error")
        chained_error.__cause__ = original_error

        assert str(chained_error) == "Chained error"
        assert chained_error.__cause__ == original_error


class TestCrewAILoaderEdgeCases:
    """Test edge cases and boundary conditions for CrewAI loader."""

    def test_load_yaml_crew_with_special_paths(self):
        """Test YAML crew loading with special path formats."""
        special_paths = [
            (Path("/simple/path"), Path("/simple/path/runtime.yaml")),
            (Path("/path/with spaces"), Path("/path/with spaces/runtime.yaml")),
            (Path("/path-with-dashes"), Path("/path-with-dashes/runtime.yaml")),
            (Path("/path.with.dots"), Path("/path.with.dots/runtime.yaml")),
            (Path("relative/path"), Path("relative/path/runtime.yaml")),
        ]

        for crew_path, runtime_path in special_paths:
            config_data = {
                "crew": f"crew_{crew_path.name}",
                "triggers": ["test_event"],
                "timeout_ms": 5000,
            }

            with (
                patch.object(CrewAILoader, "_load_runtime_config") as mock_load_config,
                patch.object(CrewAILoader, "_load_crew_module") as mock_load_module,
                patch.object(
                    CrewAILoader, "_instantiate_crew_from_module"
                ) as mock_instantiate,
            ):
                mock_runtime_config = CrewRuntimeConfig(**config_data)
                mock_load_config.return_value = mock_runtime_config
                mock_load_module.return_value = Mock()
                mock_instantiate.return_value = Mock()

                # Should not raise exceptions for any valid path format
                wrapper = CrewAILoader.load_yaml_crew(crew_path, runtime_path)
                assert isinstance(wrapper, CrewAIWrapper)

    def test_load_python_crew_with_extreme_config_values(self):
        """Test Python crew loading with extreme configuration values."""
        crew_id = "extreme_crew"
        mock_crewai_crew = Mock()

        extreme_configs = [
            # Minimum values
            {"triggers": ["a"], "timeout_ms": 1, "slice_stride": 1},
            # Maximum values
            {"triggers": ["t"] * 100, "timeout_ms": 300000, "slice_stride": 1000000},
            # Very long strings
            {"triggers": ["a" * 1000], "timeout_ms": 5000},
            # Complex nested dependencies
            {
                "triggers": ["test"],
                "needs": [
                    {"type": "crew", "crew": f"dep_{i}", "offset": -i}
                    for i in range(1, 11)
                ],
            },
        ]

        for config in extreme_configs:
            wrapper = CrewAILoader.load_python_crew(crew_id, mock_crewai_crew, config)
            assert wrapper.crew_id == crew_id
            assert wrapper.crewai_crew == mock_crewai_crew

    def test_runtime_config_with_null_and_empty_values(self):
        """Test runtime configuration handling of null and empty values."""
        config_with_nulls = {
            "crew": "null_test_crew",
            "triggers": ["test"],
            "needs": None,  # Explicitly null
            "timeout_ms": 5000,
        }

        yaml_content = yaml.dump(config_with_nulls)

        with (
            patch("builtins.open", mock_open(read_data=yaml_content)),
            patch("pathlib.Path.exists", return_value=True),
        ):
            config_path = Path("/test/null_config.yaml")
            runtime_config = CrewAILoader._load_runtime_config(config_path)

            assert runtime_config.crew == "null_test_crew"
            assert runtime_config.needs is None
            assert runtime_config.wait_policy == "none"  # Default value

    def test_module_loading_with_complex_module_structure(self):
        """Test module loading behavior with complex module structures."""

        from unittest.mock import Mock

        mock_actual_crew = Mock()

        # Define a function
        def some_function():
            pass

        # Define a regular class without crew attributes
        class RegularClass:
            pass

        # Define a crew instance with callable crew method
        class CrewInstance:
            def __init__(self):
                self.crew = lambda: mock_actual_crew

        # Define the actual crew class
        class ActualCrewClass:
            agents_config = "agents"
            tasks_config = "tasks"

            def __new__(cls):
                return CrewInstance()

        # Create module with various attributes
        import types

        mock_module = types.ModuleType("test_module")
        mock_module.some_function = some_function
        mock_module.RegularClass = RegularClass
        mock_module.ActualCrewClass = ActualCrewClass
        mock_module.some_variable = "string_value"
        mock_module.some_number = 42

        result = CrewAILoader._instantiate_crew_from_module(mock_module)
        assert result == mock_actual_crew
