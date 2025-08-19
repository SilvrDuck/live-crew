"""CrewAI loader supporting both YAML-configured and Python-defined crews."""

import importlib.util
from pathlib import Path
from typing import Any, Dict
import yaml
from pydantic import ValidationError

from live_crew.crewai_integration.wrapper import CrewAIWrapper
from live_crew.crewai_integration.models import CrewRuntimeConfig


class CrewAILoader:
    """Loader for CrewAI crews supporting both YAML and Python definition patterns.

    This class provides static methods to load CrewAI crews from either:
    1. YAML configuration (CrewAI's @CrewBase pattern with agents.yaml, tasks.yaml, crew.yaml)
    2. Python definition (direct Crew instantiation with agents, tasks, and process)

    The loader abstracts the complexity of CrewAI crew initialization and provides
    a unified interface for live-crew's orchestration layer.
    """

    @staticmethod
    def load_yaml_crew(crew_path: Path, runtime_config_path: Path) -> CrewAIWrapper:
        """Load a CrewAI crew from YAML configuration files.

        This method loads a standard CrewAI crew that follows the @CrewBase pattern
        with separate YAML files for agents, tasks, and crew configuration. The
        runtime configuration specifies how the crew integrates with live-crew.

        Expected directory structure:
        crew_path/
        ├── agents.yaml          # CrewAI agent definitions
        ├── tasks.yaml           # CrewAI task definitions
        ├── crew.yaml           # CrewAI crew configuration
        ├── crew.py             # Python crew class with @CrewBase
        └── <crew>.runtime.yaml # live-crew orchestration config

        Args:
            crew_path: Path to directory containing CrewAI crew files
            runtime_config_path: Path to the runtime configuration YAML file

        Returns:
            CrewAIWrapper instance ready for live-crew orchestration

        Raises:
            CrewAILoadError: If crew files are missing or invalid
            CrewAIConfigError: If runtime configuration is invalid
        """
        try:
            # Load and validate runtime configuration using Pydantic
            runtime_config = CrewAILoader._load_runtime_config(runtime_config_path)

            # Find and load the CrewAI crew class
            crew_module = CrewAILoader._load_crew_module(crew_path)
            crewai_crew = CrewAILoader._instantiate_crew_from_module(crew_module)

            # Create wrapper with validated runtime configuration
            wrapper = CrewAIWrapper(
                crew_id=runtime_config.crew,
                crewai_crew=crewai_crew,
                triggers=runtime_config.triggers,
                timeout_ms=runtime_config.timeout_ms,
            )

            return wrapper

        except Exception as e:
            raise CrewAILoadError(
                f"Failed to load YAML-configured CrewAI crew from {crew_path}: {str(e)}"
            ) from e

    @staticmethod
    def load_python_crew(
        crew_id: str, crewai_crew: Any, runtime_config: Dict[str, Any]
    ) -> CrewAIWrapper:
        """Load a CrewAI crew from direct Python definition.

        This method wraps a CrewAI crew that was defined programmatically in Python
        (not from YAML configuration). The crew instance and runtime configuration
        are provided directly.

        Args:
            crew_id: Unique identifier for this crew
            crewai_crew: CrewAI Crew instance (already configured with agents/tasks)
            runtime_config: Dictionary with orchestration settings (triggers, timeout_ms, etc.)

        Returns:
            CrewAIWrapper instance ready for live-crew orchestration

        Raises:
            CrewAIConfigError: If runtime configuration is invalid
        """
        try:
            # Validate runtime configuration using Pydantic model
            # Add crew_id to config for validation
            config_dict = {"crew": crew_id, **runtime_config}
            validated_config = CrewRuntimeConfig(**config_dict)

            # Create wrapper with validated configuration
            wrapper = CrewAIWrapper(
                crew_id=crew_id,
                crewai_crew=crewai_crew,
                triggers=validated_config.triggers,
                timeout_ms=validated_config.timeout_ms,
            )

            return wrapper

        except ValidationError as e:
            raise CrewAIConfigError(
                f"Invalid runtime configuration for crew '{crew_id}': {e}"
            ) from e
        except Exception as e:
            raise CrewAILoadError(
                f"Failed to load Python-defined CrewAI crew '{crew_id}': {str(e)}"
            ) from e

    @staticmethod
    def _load_runtime_config(config_path: Path) -> CrewRuntimeConfig:
        """Load and validate runtime configuration from YAML file using Pydantic.

        Args:
            config_path: Path to the runtime configuration YAML file

        Returns:
            Validated CrewRuntimeConfig instance

        Raises:
            CrewAIConfigError: If configuration is missing or invalid
        """
        if not config_path.exists():
            raise CrewAIConfigError(f"Runtime config file not found: {config_path}")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                raw_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise CrewAIConfigError(f"Invalid YAML in runtime config: {e}") from e

        try:
            # Use Pydantic for validation
            return CrewRuntimeConfig(**raw_config)
        except ValidationError as e:
            raise CrewAIConfigError(
                f"Invalid runtime configuration in {config_path}: {e}"
            ) from e

    @staticmethod
    def _load_crew_module(crew_path: Path):
        """Load the Python module containing the CrewAI crew class.

        Args:
            crew_path: Path to directory containing the crew.py file

        Returns:
            Loaded Python module

        Raises:
            CrewAILoadError: If crew.py file is missing or cannot be loaded
        """
        crew_py_path = crew_path / "crew.py"
        if not crew_py_path.exists():
            raise CrewAILoadError(f"crew.py not found in {crew_path}")

        try:
            # Use absolute path and proper module name to preserve file metadata
            module_name = f"live_crew_dynamic_{crew_path.name}_{id(crew_path)}"
            spec = importlib.util.spec_from_file_location(module_name, crew_py_path)
            if spec is None or spec.loader is None:
                raise CrewAILoadError(f"Cannot load module spec from {crew_py_path}")

            crew_module = importlib.util.module_from_spec(spec)

            # Ensure the module has proper __file__ attribute for inspect.getfile()
            crew_module.__file__ = str(crew_py_path.absolute())

            # Import sys to add module to sys.modules for proper class resolution
            import sys

            sys.modules[module_name] = crew_module

            spec.loader.exec_module(crew_module)

            return crew_module

        except Exception as e:
            raise CrewAILoadError(
                f"Failed to load crew module from {crew_py_path}: {e}"
            ) from e

    @staticmethod
    def _instantiate_crew_from_module(crew_module) -> Any:
        """Find and instantiate the CrewAI crew from the loaded module.

        This method looks for a class decorated with @CrewBase and instantiates it
        to get the configured CrewAI Crew instance.

        Args:
            crew_module: The loaded Python module containing the crew class

        Returns:
            CrewAI Crew instance

        Raises:
            CrewAILoadError: If no suitable crew class is found or instantiation fails
        """
        try:
            # Look for classes in the module that are CrewAI crews
            crew_classes = []
            for name in dir(crew_module):
                obj = getattr(crew_module, name)
                # Check if it's a class type
                if not isinstance(obj, type):
                    continue

                # Check for @CrewBase decorated class by looking for is_crew_class attribute
                # This is set by the @CrewBase decorator in CrewAI
                if hasattr(obj, "is_crew_class") and getattr(
                    obj, "is_crew_class", False
                ):
                    crew_classes.append(obj)
                # Fallback: Check for traditional indicators (for non-@CrewBase classes)
                elif hasattr(obj, "agents_config") and hasattr(obj, "tasks_config"):
                    crew_classes.append(obj)

            if not crew_classes:
                raise CrewAILoadError(
                    "No @CrewBase decorated class found in crew module"
                )

            if len(crew_classes) > 1:
                raise CrewAILoadError(
                    f"Multiple crew classes found: {[cls.__name__ for cls in crew_classes]}. Expected exactly one."
                )

            # Instantiate the crew class
            crew_class = crew_classes[0]
            crew_instance = crew_class()

            # Get the actual CrewAI Crew object
            if hasattr(crew_instance, "crew"):
                if callable(crew_instance.crew):
                    return crew_instance.crew()
                else:
                    return crew_instance.crew
            else:
                raise CrewAILoadError(
                    "Crew class does not have a 'crew' method or attribute"
                )

        except Exception as e:
            raise CrewAILoadError(f"Failed to instantiate CrewAI crew: {e}") from e


class CrewAILoadError(Exception):
    """Exception raised when CrewAI crew loading fails."""

    pass


class CrewAIConfigError(Exception):
    """Exception raised when CrewAI configuration is invalid."""

    pass
