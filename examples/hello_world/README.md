# Hello World Example

This is a complete, self-contained example showing basic live-crew usage patterns.

## What This Example Demonstrates

- **Orchestrator API**: Simplified orchestration setup
- **Event Handlers**: Using `@event_handler` decorators
- **Dynamic Events**: Events generated with current timestamps
- **Action Creation**: Using `Action.create()` helper methods
- **Configuration**: Example-friendly config with relaxed validation

## Running the Example

```bash
# From the project root
python examples/hello_world/main.py
```

## Expected Output

The example will:
1. Generate 6 events with current timestamps (user signups, logins, etc.)
2. Process events through registered handlers
3. Output actions to console (greetings, activity logs, audit logs)
4. Display processing statistics

## Files in This Example

- `main.py` - Main example script (runnable)
- `config.yaml` - Example configuration with timestamp validation disabled
- `events.json` - Generated dynamically with current timestamps
- `README.md` - This documentation

## Key Concepts Demonstrated

### Event Handlers
```python
@event_handler("user_signup")
def greet_user(event):
    return Action.create("greeting", f"Welcome {event.payload['name']}\!")
```

### Orchestrator Setup
```python
orchestrator = Orchestrator.from_file(events_file, config_file="config.yaml")
orchestrator.register_handler(greet_user, ["user_signup"])
```

### Dynamic Event Generation
Events are generated with current timestamps to avoid validation issues:
```python
events = create_hello_world_events()  # Always current timestamps
```

## Using This as a Project Starter

This example can be copied as a starting point for your own live-crew projects:

1. Copy the entire `examples/hello_world/` directory
2. Modify `main.py` with your event handlers
3. Update `config.yaml` for your requirements
4. Customize event generation or use static files
EOF < /dev/null
