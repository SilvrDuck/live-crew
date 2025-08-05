"""Command-line interface for live-crew.

Provides the main entry point for running live-crew processes
with configuration and file processing capabilities.
"""

import asyncio
import sys
from pathlib import Path
from typing import Annotated

import typer

from live_crew.backends.context import DictContextBackend
from live_crew.config.settings import load_config
from live_crew.crew.handlers import EchoEventHandler
from live_crew.crew.registry import SimpleCrewRegistry
from live_crew.scheduling.memory import MemoryScheduler
from live_crew.transports.console import ConsoleActionTransport
from live_crew.transports.file import FileEventTransport

app = typer.Typer(
    name="live-crew",
    help="Low-latency, slice-based orchestration for CrewAI crews over real-time event streams",
    no_args_is_help=True,
)


@app.command()
def process(
    input_file: Annotated[
        Path,
        typer.Argument(
            help="Input file containing events (JSON format)",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    config_file: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Configuration file (YAML format)",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Enable verbose logging",
        ),
    ] = False,
) -> None:
    """Process events from input file using live-crew orchestration.

    This command reads events from the specified input file, processes them
    through registered crews using time slicing, and outputs actions.

    Examples:
        live-crew process events.json
        live-crew process events.json --config live-config.yaml
        live-crew process events.json --verbose
    """
    try:
        # Load configuration
        config = load_config(config_file) if config_file else load_config()

        if verbose:
            typer.echo("📋 Configuration loaded:")
            typer.echo(f"   Slice duration: {config.slice_ms}ms")
            typer.echo(f"   Heartbeat: {config.heartbeat_s}s")
            typer.echo(f"   Backend: {config.kv_backend}")
            typer.echo(f"📂 Processing: {input_file}")

        # Run the async processing using recommended pattern
        asyncio.run(_process_events_async(input_file, config, verbose))

        if verbose:
            typer.echo("✅ Processing completed successfully")

    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


async def _process_events_async(input_file: Path, config, verbose: bool) -> None:
    """Async processing function for events."""
    # Initialize components
    event_transport = FileEventTransport(input_file)
    action_transport = ConsoleActionTransport()
    context_backend = DictContextBackend()
    crew_registry = SimpleCrewRegistry()

    # Create scheduler
    scheduler = MemoryScheduler(
        config=config,
        event_transport=event_transport,
        action_transport=action_transport,
        context_backend=context_backend,
        crew_registry=crew_registry,
    )

    # Register default echo crew for demonstration
    echo_handler = EchoEventHandler("default_echo")
    crew_registry.register_crew(echo_handler, [])

    if verbose:
        typer.echo("🔧 Scheduler initialized with MemoryScheduler")
        typer.echo(f"🎯 Registered crews: {len(crew_registry.list_crews())}")

    # Process events
    await scheduler.process_events()

    if verbose:
        processed_slices = len(scheduler._processed_slices)
        typer.echo(f"📊 Processed {processed_slices} time slices")


@app.command()
def config(
    show: Annotated[
        bool,
        typer.Option(
            "--show",
            help="Show current configuration",
        ),
    ] = False,
    validate: Annotated[
        Path | None,
        typer.Option(
            "--validate",
            help="Validate configuration file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
) -> None:
    """Manage live-crew configuration.

    Examples:
        live-crew config --show
        live-crew config --validate live-config.yaml
    """
    try:
        if validate:
            # Validate specific config file
            config_obj = load_config(validate)
            typer.echo(f"✅ Configuration file '{validate}' is valid")
            typer.echo(f"   Slice duration: {config_obj.slice_ms}ms")
            typer.echo(f"   Heartbeat: {config_obj.heartbeat_s}s")
            typer.echo(f"   Backend: {config_obj.kv_backend}")
        elif show:
            # Show current configuration (defaults + env vars)
            config_obj = load_config()
            typer.echo("📋 Current Configuration:")
            typer.echo(f"   Slice duration: {config_obj.slice_ms}ms")
            typer.echo(f"   Heartbeat: {config_obj.heartbeat_s}s")
            typer.echo(f"   Backend: {config_obj.kv_backend}")
            if config_obj.vector:
                typer.echo(f"   Vector store: {config_obj.vector}")
        else:
            typer.echo(
                "Use --show to display configuration or --validate to check a file"
            )

    except Exception as e:
        typer.echo(f"❌ Configuration error: {e}", err=True)
        sys.exit(1)


@app.command()
def version() -> None:
    """Show live-crew version information."""
    typer.echo("live-crew 0.1.0")
    typer.echo("Low-latency, slice-based orchestration for CrewAI crews")


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
