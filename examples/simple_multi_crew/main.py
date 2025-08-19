"""YAML-Driven Content Moderation Demo for live-crew.

This demo showcases:
- YAML-based crew configuration for content moderation
- Dependency management with analysis → action pipeline
- Event triggers for message processing
- Context sharing between crews via KV backend
- Multi-crew orchestration with real business logic

The analysis crew runs first to analyze content, then action crew takes moderation decisions.
"""

import asyncio
import os
from pathlib import Path

from live_crew.crewai_integration.orchestrator import CrewOrchestrator


async def main():
    """Run the YAML-driven content moderation demo."""

    # Disable timestamp validation for demo (using environment variable)
    os.environ["LIVE_CREW_EVENT_VALIDATION__TIMESTAMP_VALIDATION_ENABLED"] = "false"

    # Clear config cache to pick up env var
    from live_crew.config.settings import get_config

    get_config.cache_clear()

    # Load orchestrator from YAML configuration
    config_file = Path(__file__).parent / "orchestration.yaml"

    try:
        orchestrator = CrewOrchestrator.from_config(config_file)

    except Exception:
        return

    # Run orchestration with dependency resolution
    events_file = Path(__file__).parent / "events.json"
    print("🎯 Processing social media messages through moderation pipeline...")

    try:
        _ = await orchestrator.run(events_source=events_file)

    except Exception as e:
        print(f"❌ Orchestration failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
