#!/usr/bin/env python3
"""Hello World example for live-crew.

This self-contained example demonstrates:
- Orchestrator API usage
- Event handler decorators
- Processing events with dynamic timestamps
- Basic live-crew workflow

Run this example:
    python examples/hello_world/main.py
"""

import asyncio
from pathlib import Path

from live_crew import Orchestrator, event_handler, Action


@event_handler("user_signup")
def greet_user(event):
    """Handle user signup events with a personalized greeting."""
    return Action.create("greeting", f"Welcome {event.payload['name']}!")


@event_handler(["user_login", "profile_updated"])
def track_activity(event):
    """Track user activity events."""
    return Action.create(
        "activity_logged",
        {
            "event_type": event.kind,
            "user": event.payload.get("name", "unknown"),
            "timestamp": str(event.ts),
        },
    )


@event_handler()  # Handle all events
def audit_log(event):
    """Log all events for auditing purposes."""
    return Action.create(
        "audit_logged", {"original_event": event.kind, "stream": event.stream_id}
    )


async def main():
    """Run the hello world example."""
    print("🚀 Starting Hello World live-crew example...")

    # Use events file with recent timestamps
    events_file = Path(__file__).parent / "events.json"

    # Create orchestrator from events file
    orchestrator = Orchestrator.from_file(events_file)

    # Register event handlers
    orchestrator.register_handler(greet_user, ["user_signup"])
    orchestrator.register_handler(track_activity, ["user_login", "profile_updated"])
    orchestrator.register_handler(audit_log)  # All events

    # Run processing
    result = await orchestrator.run()

    # Show results
    print(f"✅ Processed {result.events_processed} events")
    print(f"🎯 Generated {result.actions_generated} actions")
    print(f"⏱️  Used {result.time_slices} time slices")

    print("\n📋 Processing completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
