# Typer CLI Library Reference

## Overview

Typer is a library for building CLI applications that users will love using and developers will love creating. Based on Python type hints and inspired by FastAPI.

**Key Benefits:**
- Intuitive to write: Great editor support. Completion everywhere. Less time debugging
- Easy to use: It's easy to use for the final users. Automatic help, and automatic completion for all shells
- Short: Minimize code duplication. Multiple features from each parameter declaration
- Start simple: The simplest example adds only 2 lines of code to your app: 1 import, 1 function call
- Grow large: Grow in complexity as much as you want, create arbitrarily complex trees of commands and groups of subcommands

## Installation

```bash
# With optional dependencies (recommended)
uv add typer[all]

# Minimal installation
uv add typer

# Slim version (without rich and shellingham)
uv add typer-slim
```

## Core Concepts

### Basic Application Structure

```python
import typer

app = typer.Typer()

@app.command()
def hello(name: str):
    """Say hello to someone."""
    typer.echo(f"Hello {name}")

if __name__ == "__main__":
    app()
```

### Type Hints Usage

Typer relies heavily on Python type hints for automatic CLI parameter handling:

```python
# Use T | None in modern Python 3.10+
import typer

def main(
    name: str,
    count: int = 1,
    formal: bool = False,
    output_file: str | None = typer.Option(None, "--output", "-o")
):
    """Example with various parameter types."""
    greeting = "Good day" if formal else "Hello"
    message = f"{greeting} {name}! " * count

    if output_file:
        with open(output_file, "w") as f:
            f.write(message)
    else:
        typer.echo(message)
```

### Arguments vs Options

```python
import typer

def process(
    # Positional argument (required)
    input_file: str = typer.Argument(..., help="Input file to process"),

    # Option with flag
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),

    # Option with value
    output_format: str = typer.Option("json", "--format", "-f", help="Output format"),

    # Option with file validation
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Configuration file"
    )
):
    """Process files with various options."""
    pass
```

## Async Support (Current Best Practice)

**Status:** Typer does not have native async support, but there's a well-established workaround pattern.

### Recommended Async Pattern

```python
import asyncio
# Use T | None in modern Python 3.10+
from pathlib import Path
import typer

app = typer.Typer()

@app.command()
def process(
    input_file: Path = typer.Argument(..., help="Input file"),
    config: Path | None = typer.Option(None, "--config", "-c"),
    verbose: bool = typer.Option(False, "--verbose", "-v")
):
    """Process files asynchronously."""
    # Wrap async function call
    asyncio.run(_process_async(input_file, config, verbose))

async def _process_async(input_file: Path, config: Path | None, verbose: bool):
    """Actual async implementation."""
    if verbose:
        typer.echo(f"Processing {input_file}")

    # Your async logic here
    await some_async_operation()

    if verbose:
        typer.echo("Processing completed")

async def some_async_operation():
    """Example async operation."""
    await asyncio.sleep(1)  # Simulate async work

if __name__ == "__main__":
    app()
```

### Alternative Async Decorator Pattern

```python
import asyncio
from functools import wraps

def async_command(func):
    """Decorator to handle async functions in Typer commands."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))
    return wrapper

@app.command()
@async_command
async def async_hello(name: str):
    """Async command using decorator pattern."""
    await asyncio.sleep(1)  # Simulate async work
    typer.echo(f"Hello {name} (async)")
```

## Application Organization

### Single File Applications

```python
import typer

app = typer.Typer(
    name="myapp",
    help="My awesome CLI application",
    no_args_is_help=True  # Show help when no command provided
)

@app.command()
def command1():
    """First command."""
    pass

@app.command()
def command2():
    """Second command."""
    pass

if __name__ == "__main__":
    app()
```

### Multi-File Applications

```python
# main.py
import typer
from . import users, posts

app = typer.Typer()
app.add_typer(users.app, name="users")
app.add_typer(posts.app, name="posts")

if __name__ == "__main__":
    app()

# users.py
import typer

app = typer.Typer()

@app.command()
def create(name: str):
    """Create a user."""
    typer.echo(f"Creating user: {name}")

@app.command()
def delete(name: str):
    """Delete a user."""
    typer.echo(f"Deleting user: {name}")
```

## Error Handling and Exit Codes

```python
import sys
import typer

def risky_operation():
    """Operation that might fail."""
    try:
        # Your logic here
        result = perform_operation()
        typer.echo("✅ Operation completed successfully")
        return result
    except FileNotFoundError as e:
        typer.echo(f"❌ File not found: {e}", err=True)
        sys.exit(1)
    except ValueError as e:
        typer.echo(f"❌ Invalid input: {e}", err=True)
        sys.exit(2)
    except Exception as e:
        typer.echo(f"❌ Unexpected error: {e}", err=True)
        sys.exit(3)
```

## Rich Integration (Modern Output)

```python
import typer
from rich.console import Console
from rich.table import Table

console = Console()

def show_table():
    """Display data in a rich table."""
    table = Table(title="Results")
    table.add_column("Name", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Item 1", "Value 1")
    table.add_row("Item 2", "Value 2")

    console.print(table)
```

## Testing CLI Applications

```python
import typer
from typer.testing import CliRunner

app = typer.Typer()

@app.command()
def hello(name: str = "World"):
    typer.echo(f"Hello {name}")

def test_hello():
    runner = CliRunner()
    result = runner.invoke(app, ["hello", "Alice"])
    assert result.exit_code == 0
    assert "Hello Alice" in result.stdout
```

## Best Practices (2024-2025)

### 1. Type Annotations
- Always use type hints for better IDE support and validation
- Use `T | None` for optional parameters (modern Python 3.10+)
- Leverage `Path` type for file/directory parameters

### 2. Command Organization
- Use descriptive command names and help text
- Group related commands with `add_typer()`
- Set `no_args_is_help=True` for better UX

### 3. Error Handling
- Provide meaningful error messages
- Use appropriate exit codes (0 for success, non-zero for errors)
- Use `typer.echo(..., err=True)` for error output

### 4. Async Operations
- Use the `asyncio.run()` wrapper pattern for async code
- Consider creating a reusable async decorator
- Keep async logic separate from CLI parameter handling

### 5. File Operations
- Use `Path` type with validation options
- Validate file existence, permissions, and types at CLI level
- Provide clear feedback for file operations

### 6. Output Formatting
- Use Rich for enhanced output when available
- Provide `--verbose` flags for detailed output
- Use consistent emoji/icons for status messages (✅ ❌ 📋 etc.)

## Common Patterns

### Configuration File Support
```python
from pathlib import Path
import typer
import yaml

def load_config(config_path: Path | None = None) -> dict:
    """Load configuration from file."""
    if config_path and config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}

@app.command()
def process(
    input_file: Path,
    config: Path | None = typer.Option(None, "--config", "-c")
):
    """Process with optional config file."""
    config_dict = load_config(config)
    # Use config_dict...
```

### Progress Indication
```python
import typer
from rich.progress import Progress

def long_operation():
    """Show progress during long operations."""
    with Progress() as progress:
        task = progress.add_task("Processing...", total=100)
        for i in range(100):
            # Do work
            progress.update(task, advance=1)
```

## Current Limitations

1. **No Native Async Support**: Must use `asyncio.run()` wrapper pattern
2. **Limited Dependency Injection**: Unlike FastAPI, no built-in DI system
3. **Rich Integration**: While powerful, requires additional setup for advanced formatting

## Recent Updates (2024-2025)

- Continued maintenance and bug fixes
- Improved type hint support
- Better Rich integration
- Enhanced testing utilities

This reference should be updated as Typer evolves, particularly if native async support is added in future versions.
