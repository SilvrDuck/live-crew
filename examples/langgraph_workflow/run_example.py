"""Example demonstrating LangGraph workflow integration with live-crew.

This script shows how to integrate a LangGraph workflow with live-crew's
event-driven orchestration system using the Python-defined graph pattern.
"""

import asyncio
from pathlib import Path

from live_crew import Orchestrator, LangGraphLoader


async def main():
    """Run the LangGraph workflow example."""
    print("=" * 60)
    print("LangGraph Workflow Integration Example")
    print("=" * 60)
    print()

    # Get paths
    example_dir = Path(__file__).parent
    events_file = example_dir / "events.json"
    runtime_config_file = example_dir / "simple_workflow.runtime.yaml"

    print("📋 Configuration:")
    print(f"   Events file: {events_file}")
    print(f"   Runtime config: {runtime_config_file}")
    print()

    try:
        # Import the graph module
        import sys

        sys.path.insert(0, str(example_dir))
        from graph import create_graph

        # Load the LangGraph workflow using the loader
        print("🔧 Loading LangGraph workflow...")
        import yaml

        with open(runtime_config_file) as f:
            runtime_config = yaml.safe_load(f)

        # Create the compiled graph
        langgraph_app = create_graph()

        # Load using LangGraphLoader
        wrapper = LangGraphLoader.load_python_graph(
            graph_id="simple_workflow",
            langgraph_app=langgraph_app,
            runtime_config=runtime_config,
        )
        print(f"✅ Loaded graph: {wrapper.graph_id}")
        print(f"   Triggers: {wrapper.triggers}")
        print()

        # Create orchestrator
        print("🚀 Setting up orchestrator...")
        orchestrator = Orchestrator.from_file(str(events_file))

        # Register the LangGraph wrapper
        orchestrator.register_handler(wrapper, dependencies=[])
        print(f"✅ Registered LangGraph workflow handler")
        print()

        # Run the orchestration
        print("⚡ Processing events...")
        print("-" * 60)
        result = await orchestrator.run()
        print("-" * 60)
        print()

        # Display results
        print("📊 Results:")
        print(f"   Events processed: {result.total_events}")
        print(f"   Actions generated: {len(result.actions)}")
        print()

        if result.actions:
            print("📤 Generated Actions:")
            for i, action in enumerate(result.actions, 1):
                print(f"   {i}. Kind: {action.kind}")
                print(f"      Stream: {action.stream_id}")
                if isinstance(action.payload, dict):
                    print(f"      Payload: {action.payload}")
                print()

        print("✅ Example completed successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
