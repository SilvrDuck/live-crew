# Pydantic-Settings Reference

## Key Concepts

Pydantic-settings provides type-safe configuration management with automatic validation and support for multiple configuration sources.

### Installation
```bash
pip install pydantic-settings
```

### Configuration Sources (Priority Order)
1. CLI arguments
2. Initialization arguments
3. Environment variables
4. Dotenv files (.env)
5. Secrets directory
6. Default field values

## Basic Usage

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    database_url: str
    debug: bool = False
    port: int = 8000

    model_config = SettingsConfigDict(
        env_prefix='MYAPP_',  # Environment variables prefixed with MYAPP_
        env_file='.env'       # Load from .env file
    )

# Usage
settings = Settings()  # Automatically loads from env vars, .env file, etc.
```

## YAML Configuration

### Basic YAML Setup
```python
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

class Settings(BaseSettings):
    model_config = SettingsConfigDict(yaml_file="config.yaml")

    host: str
    port: int
    debug: bool = False

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            YamlConfigSettingsSource(settings_cls),
            env_settings,  # Env vars can override YAML
            init_settings,
            dotenv_settings,
            file_secret_settings,
        )
```

### Nested Models with YAML
```python
from pydantic import BaseModel

class DatabaseSettings(BaseModel):
    host: str
    port: int
    username: str
    password: str

class RedisSettings(BaseModel):
    host: str
    port: int = 6379

class Settings(BaseSettings):
    model_config = SettingsConfigDict(yaml_file='config.yaml')

    app_name: str
    database: DatabaseSettings
    redis: RedisSettings

    @classmethod
    def settings_customise_sources(cls, ...):
        return (YamlConfigSettingsSource(settings_cls),)
```

Example config.yaml:
```yaml
app_name: "live-crew"
database:
  host: "localhost"
  port: 5432
  username: "user"
  password: "pass"
redis:
  host: "localhost"
  port: 6379
```

## Custom Sources

### Multiple YAML Files
```python
from pathlib import Path
from typing import Any
import yaml
from pydantic.utils import deep_update

class MultiYamlSource(PydanticBaseSettingsSource):
    def get_field_value(self, field_info, field_name: str) -> tuple[Any, str, bool]:
        # Load multiple YAML files in order
        config_data = {}
        config_files = [
            Path("global.yaml"),
            Path(f"{self.settings_cls.__name__.lower()}.yaml"),
            Path("local.yaml"),  # Override file
        ]

        for config_file in config_files:
            if config_file.exists():
                with open(config_file, 'r') as f:
                    file_data = yaml.safe_load(f)
                    if file_data:
                        config_data = deep_update(config_data, file_data)

        return config_data.get(field_name), str(config_file), False

    def prepare_field_value(self, field_name: str, field: FieldInfo, value: Any, value_is_complex: bool) -> Any:
        return value

    def __call__(self) -> dict[str, Any]:
        # Return complete config dictionary
        config_data = {}
        config_files = [Path("global.yaml"), Path("local.yaml")]

        for config_file in config_files:
            if config_file.exists():
                with open(config_file, 'r') as f:
                    file_data = yaml.safe_load(f)
                    if file_data:
                        config_data = deep_update(config_data, file_data)

        return config_data
```

### JSON/Custom Format Source
```python
import json
from typing import Any

class JsonConfigSource(PydanticBaseSettingsSource):
    def __init__(self, settings_cls: type[BaseSettings], json_file: str = None):
        super().__init__(settings_cls)
        self.json_file = json_file or "config.json"

    def __call__(self) -> dict[str, Any]:
        try:
            with open(self.json_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
```

## Environment Variable Patterns

### Complex Types
```python
from typing import List, Dict

class Settings(BaseSettings):
    # List from comma-separated string
    allowed_hosts: List[str] = []

    # Dict from JSON string
    feature_flags: Dict[str, bool] = {}

    model_config = SettingsConfigDict(
        env_prefix='APP_',
        # Parse complex types from env vars
        json_encoders={
            list: lambda v: ','.join(v),
            dict: lambda v: json.dumps(v)
        }
    )

# Environment variables:
# APP_ALLOWED_HOSTS=localhost,127.0.0.1,example.com
# APP_FEATURE_FLAGS='{"new_ui": true, "beta_feature": false}'
```

## Best Practices

### 1. Use Type Hints
```python
from typing import Optional
from pydantic import Field

class Settings(BaseSettings):
    database_url: str = Field(..., description="Database connection URL")
    debug: bool = Field(False, description="Enable debug mode")
    worker_count: Optional[int] = Field(None, ge=1, le=10)
```

### 2. Environment Prefixes
```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix='MYAPP_',
        case_sensitive=False,
        env_file='.env',
        env_file_encoding='utf-8',
    )
```

### 3. Secrets Management
```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        secrets_dir='/run/secrets',  # Docker secrets
        env_file='.env',
    )

    database_password: str  # Will look for /run/secrets/database_password
```

### 4. Validation
```python
from pydantic import validator

class Settings(BaseSettings):
    database_url: str

    @validator('database_url')
    def validate_database_url(cls, v):
        if not v.startswith(('postgresql://', 'mysql://')):
            raise ValueError('Database URL must be PostgreSQL or MySQL')
        return v
```

## Migration from Custom YAML Loading

### Before (Custom YAML)
```python
import yaml
from pathlib import Path

def load_config(config_path: Path = None):
    config_path = config_path or Path("config.yaml")
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    return data
```

### After (Pydantic-Settings)
```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        yaml_file='config.yaml',
        env_prefix='LIVE_CREW_',
    )

    slice_ms: int = 500
    heartbeat_s: int = 1

    @classmethod
    def settings_customise_sources(cls, ...):
        return (
            YamlConfigSettingsSource(settings_cls),
            env_settings,  # Allow env var overrides
            init_settings,
        )

# Usage - automatically validates and provides type safety
settings = Settings()
```

## Performance Benefits
- Type validation at startup
- Environment variable caching
- Lazy loading of configuration sources
- Built-in error handling and validation messages
- No need for manual YAML parsing code

## Common Patterns
- Use YAML for default configuration
- Allow environment variable overrides for deployment
- Validate configuration at application startup
- Use nested models for complex configuration sections
- Leverage Field() for documentation and validation constraints
