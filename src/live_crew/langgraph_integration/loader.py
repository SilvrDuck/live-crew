"""LangGraph loader supporting both Python-defined and YAML-configured graphs."""

import importlib.util
from pathlib import Path
from typing import Any, Dict
import yaml
from pydantic import ValidationError

from live_crew.langgraph_integration.wrapper import LangGraphWrapper
from live_crew.langgraph_integration.models import GraphRuntimeConfig


class LangGraphLoader:
    """Loader for LangGraph workflows supporting both Python and YAML configuration patterns.

    This class provides static methods to load LangGraph workflows from either:
    1. Python definition (direct CompiledGraph/StateGraph instantiation)
    2. YAML configuration (referencing Python modules with graph factory functions)

    The loader abstracts the complexity of LangGraph initialization and provides
    a unified interface for live-crew's orchestration layer.
    """

    @staticmethod
    def load_yaml_graph(graph_path: Path, runtime_config_path: Path) -> LangGraphWrapper:
        """Load a LangGraph workflow from YAML configuration files.

        This method loads a LangGraph workflow defined in a Python module, with
        orchestration configuration specified in YAML. The runtime configuration
        specifies how the graph integrates with live-crew.

        Expected directory structure:
        graph_path/
        ├── graph.py                  # Python module with LangGraph definition
        ├── nodes.py                  # (Optional) Node function definitions
        ├── state.py                  # (Optional) State schema definitions
        └── <graph>.runtime.yaml      # live-crew orchestration config

        Args:
            graph_path: Path to directory containing LangGraph workflow Python module
            runtime_config_path: Path to the runtime configuration YAML file

        Returns:
            LangGraphWrapper instance ready for live-crew orchestration

        Raises:
            LangGraphLoadError: If graph files are missing or invalid
            LangGraphConfigError: If runtime configuration is invalid
        """
        try:
            # Load and validate runtime configuration using Pydantic
            runtime_config = LangGraphLoader._load_runtime_config(runtime_config_path)

            # Find and load the LangGraph module
            graph_module = LangGraphLoader._load_graph_module(graph_path)

            # Instantiate the compiled graph from module
            langgraph_app = LangGraphLoader._instantiate_graph_from_module(
                graph_module, runtime_config
            )

            # Create wrapper with validated runtime configuration
            wrapper = LangGraphWrapper(
                graph_id=runtime_config.graph,
                langgraph_app=langgraph_app,
                triggers=runtime_config.triggers,
                timeout_ms=runtime_config.timeout_ms,
                checkpointing=runtime_config.checkpointing,
                thread_id_strategy=runtime_config.thread_id_strategy,
                interrupt_before=runtime_config.interrupt_before,
                interrupt_after=runtime_config.interrupt_after,
            )

            return wrapper

        except Exception as e:
            raise LangGraphLoadError(
                f"Failed to load YAML-configured LangGraph workflow from {graph_path}: {str(e)}"
            ) from e

    @staticmethod
    def load_python_graph(
        graph_id: str, langgraph_app: Any, runtime_config: Dict[str, Any]
    ) -> LangGraphWrapper:
        """Load a LangGraph workflow from direct Python definition.

        This method wraps a LangGraph workflow that was defined programmatically in Python
        (not from YAML configuration). The compiled graph instance and runtime configuration
        are provided directly.

        Args:
            graph_id: Unique identifier for this graph
            langgraph_app: Compiled LangGraph application instance
            runtime_config: Dictionary with orchestration settings (triggers, timeout_ms, etc.)

        Returns:
            LangGraphWrapper instance ready for live-crew orchestration

        Raises:
            LangGraphConfigError: If runtime configuration is invalid
        """
        try:
            # Validate runtime configuration using Pydantic model
            # Add graph_id to config for validation
            config_dict = {"graph": graph_id, **runtime_config}
            validated_config = GraphRuntimeConfig(**config_dict)

            # Create wrapper with validated configuration
            wrapper = LangGraphWrapper(
                graph_id=graph_id,
                langgraph_app=langgraph_app,
                triggers=validated_config.triggers,
                timeout_ms=validated_config.timeout_ms,
                checkpointing=validated_config.checkpointing,
                thread_id_strategy=validated_config.thread_id_strategy,
                interrupt_before=validated_config.interrupt_before,
                interrupt_after=validated_config.interrupt_after,
            )

            return wrapper

        except ValidationError as e:
            raise LangGraphConfigError(
                f"Invalid runtime configuration for graph '{graph_id}': {e}"
            ) from e
        except Exception as e:
            raise LangGraphLoadError(
                f"Failed to load Python-defined LangGraph workflow '{graph_id}': {str(e)}"
            ) from e

    @staticmethod
    def _load_runtime_config(config_path: Path) -> GraphRuntimeConfig:
        """Load and validate runtime configuration from YAML file using Pydantic.

        Args:
            config_path: Path to the runtime configuration YAML file

        Returns:
            Validated GraphRuntimeConfig instance

        Raises:
            LangGraphConfigError: If configuration is missing or invalid
        """
        if not config_path.exists():
            raise LangGraphConfigError(f"Runtime config file not found: {config_path}")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                raw_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise LangGraphConfigError(f"Invalid YAML in runtime config: {e}") from e

        try:
            # Use Pydantic for validation
            return GraphRuntimeConfig(**raw_config)
        except ValidationError as e:
            raise LangGraphConfigError(
                f"Invalid runtime configuration in {config_path}: {e}"
            ) from e

    @staticmethod
    def _load_graph_module(graph_path: Path):
        """Load the Python module containing the LangGraph workflow.

        Args:
            graph_path: Path to directory containing the graph.py file

        Returns:
            Loaded Python module

        Raises:
            LangGraphLoadError: If graph.py file is missing or cannot be loaded
        """
        graph_py_path = graph_path / "graph.py"
        if not graph_py_path.exists():
            raise LangGraphLoadError(f"graph.py not found in {graph_path}")

        try:
            # Use absolute path and proper module name to preserve file metadata
            module_name = f"live_crew_dynamic_graph_{graph_path.name}_{id(graph_path)}"
            spec = importlib.util.spec_from_file_location(module_name, graph_py_path)
            if spec is None or spec.loader is None:
                raise LangGraphLoadError(f"Cannot load module spec from {graph_py_path}")

            graph_module = importlib.util.module_from_spec(spec)

            # Ensure the module has proper __file__ attribute
            graph_module.__file__ = str(graph_py_path.absolute())

            # Import sys to add module to sys.modules for proper module resolution
            import sys

            sys.modules[module_name] = graph_module

            spec.loader.exec_module(graph_module)

            return graph_module

        except Exception as e:
            raise LangGraphLoadError(
                f"Failed to load graph module from {graph_py_path}: {e}"
            ) from e

    @staticmethod
    def _instantiate_graph_from_module(
        graph_module, runtime_config: GraphRuntimeConfig
    ) -> Any:
        """Find and instantiate the LangGraph workflow from the loaded module.

        This method looks for a factory function (default: create_graph) that returns
        a compiled LangGraph application.

        Args:
            graph_module: The loaded Python module containing the graph
            runtime_config: Runtime configuration with graph_factory name

        Returns:
            Compiled LangGraph application

        Raises:
            LangGraphLoadError: If no suitable graph factory is found or instantiation fails
        """
        try:
            # Determine factory function name from config or use default
            factory_name = getattr(runtime_config, "graph_factory", "create_graph")

            # Look for the factory function in the module
            if not hasattr(graph_module, factory_name):
                raise LangGraphLoadError(
                    f"Graph factory function '{factory_name}' not found in module. "
                    f"Expected a function that returns a compiled LangGraph app."
                )

            factory_func = getattr(graph_module, factory_name)

            # Call the factory function to get compiled graph
            if not callable(factory_func):
                raise LangGraphLoadError(
                    f"'{factory_name}' is not callable. Expected a function that returns compiled graph."
                )

            # Try calling with runtime config first (for configurable graphs)
            try:
                langgraph_app = factory_func(runtime_config)
            except TypeError:
                # Fall back to no-argument call
                langgraph_app = factory_func()

            # Verify we got a compiled LangGraph application
            if not hasattr(langgraph_app, "invoke"):
                raise LangGraphLoadError(
                    f"Factory function '{factory_name}' did not return a compiled LangGraph app. "
                    f"Expected object with 'invoke' method."
                )

            return langgraph_app

        except Exception as e:
            raise LangGraphLoadError(f"Failed to instantiate LangGraph workflow: {e}") from e


class LangGraphLoadError(Exception):
    """Exception raised when LangGraph workflow loading fails."""

    pass


class LangGraphConfigError(Exception):
    """Exception raised when LangGraph configuration is invalid."""

    pass
