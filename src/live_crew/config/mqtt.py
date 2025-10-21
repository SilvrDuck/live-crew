"""MQTT configuration models for live-crew.

Provides Pydantic models for MQTT connection configuration
with validation and secure defaults.
"""

from pydantic import BaseModel, Field


class MQTTConfig(BaseModel):
    """MQTT broker connection configuration.

    Defines connection parameters for MQTT brokers with
    secure defaults and comprehensive validation.
    """

    hostname: str = Field(
        default="localhost",
        min_length=1,
        description="MQTT broker hostname or IP address",
    )

    port: int = Field(
        default=1883,
        ge=1,
        le=65535,
        description="MQTT broker port (1883 for unencrypted, 8883 for TLS)",
    )

    username: str | None = Field(
        default=None,
        description="MQTT username for authentication",
    )

    password: str | None = Field(
        default=None,
        description="MQTT password for authentication",
    )

    topic_prefix: str = Field(
        default="live",
        pattern=r"^[a-zA-Z0-9_/-]+$",
        min_length=1,
        max_length=100,
        description="Topic prefix for all MQTT topics",
    )

    qos: int = Field(
        default=1,
        ge=0,
        le=2,
        description="Quality of Service level: 0 (at most once), 1 (at least once), 2 (exactly once)",
    )

    keepalive: int = Field(
        default=60,
        ge=1,
        le=86400,
        description="Keepalive interval in seconds",
    )

    timeout: int = Field(
        default=10,
        ge=1,
        le=300,
        description="Connection timeout in seconds",
    )

    client_id: str | None = Field(
        default=None,
        pattern=r"^[a-zA-Z0-9_-]*$",
        max_length=23,
        description="MQTT client ID (auto-generated if not provided)",
    )

    clean_session: bool = Field(
        default=True,
        description="Whether to start with a clean session",
    )

    use_tls: bool = Field(
        default=False,
        description="Whether to use TLS/SSL encryption",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "hostname": "mqtt.example.com",
                    "port": 1883,
                    "username": "live-crew",
                    "password": "secure-password",
                    "topic_prefix": "live",
                    "qos": 1,
                },
                {
                    "hostname": "localhost",
                    "port": 8883,
                    "use_tls": True,
                    "topic_prefix": "production/live",
                    "qos": 2,
                },
            ]
        }
    }
