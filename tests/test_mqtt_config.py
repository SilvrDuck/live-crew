"""Tests for MQTT configuration models."""

import pytest
from pydantic import ValidationError

from live_crew.config.mqtt import MQTTConfig


class TestMQTTConfig:
    """Test suite for MQTT configuration model."""

    def test_default_values(self):
        """Test default configuration values."""
        config = MQTTConfig()

        assert config.hostname == "localhost"
        assert config.port == 1883
        assert config.username is None
        assert config.password is None
        assert config.topic_prefix == "live"
        assert config.qos == 1
        assert config.keepalive == 60
        assert config.timeout == 10
        assert config.client_id is None
        assert config.clean_session is True
        assert config.use_tls is False

    def test_custom_values(self):
        """Test configuration with custom values."""
        config = MQTTConfig(
            hostname="mqtt.example.com",
            port=8883,
            username="test_user",
            password="secure_pass",
            topic_prefix="production/live",
            qos=2,
            keepalive=120,
            timeout=30,
            client_id="client-001",
            clean_session=False,
            use_tls=True,
        )

        assert config.hostname == "mqtt.example.com"
        assert config.port == 8883
        assert config.username == "test_user"
        assert config.password == "secure_pass"
        assert config.topic_prefix == "production/live"
        assert config.qos == 2
        assert config.keepalive == 120
        assert config.timeout == 30
        assert config.client_id == "client-001"
        assert config.clean_session is False
        assert config.use_tls is True

    def test_port_validation(self):
        """Test port number validation."""
        # Valid ports
        config = MQTTConfig(port=1)
        assert config.port == 1

        config = MQTTConfig(port=65535)
        assert config.port == 65535

        # Invalid ports
        with pytest.raises(ValidationError):
            MQTTConfig(port=0)

        with pytest.raises(ValidationError):
            MQTTConfig(port=65536)

        with pytest.raises(ValidationError):
            MQTTConfig(port=-1)

    def test_qos_validation(self):
        """Test QoS level validation."""
        # Valid QoS levels
        for qos in [0, 1, 2]:
            config = MQTTConfig(qos=qos)
            assert config.qos == qos

        # Invalid QoS levels
        with pytest.raises(ValidationError):
            MQTTConfig(qos=-1)

        with pytest.raises(ValidationError):
            MQTTConfig(qos=3)

    def test_topic_prefix_validation(self):
        """Test topic prefix validation."""
        # Valid topic prefixes
        valid_prefixes = [
            "live",
            "production/live",
            "dev-env/live",
            "live_crew",
            "a/b/c/d",
        ]

        for prefix in valid_prefixes:
            config = MQTTConfig(topic_prefix=prefix)
            assert config.topic_prefix == prefix

        # Invalid topic prefixes (special MQTT characters)
        invalid_prefixes = [
            "live+",  # + is wildcard
            "live#",  # # is wildcard
            "live$sys",  # $ is reserved
            "",  # empty
        ]

        for prefix in invalid_prefixes:
            with pytest.raises(ValidationError):
                MQTTConfig(topic_prefix=prefix)

    def test_keepalive_validation(self):
        """Test keepalive interval validation."""
        # Valid keepalive
        config = MQTTConfig(keepalive=1)
        assert config.keepalive == 1

        config = MQTTConfig(keepalive=86400)
        assert config.keepalive == 86400

        # Invalid keepalive
        with pytest.raises(ValidationError):
            MQTTConfig(keepalive=0)

        with pytest.raises(ValidationError):
            MQTTConfig(keepalive=86401)

    def test_timeout_validation(self):
        """Test timeout validation."""
        # Valid timeout
        config = MQTTConfig(timeout=1)
        assert config.timeout == 1

        config = MQTTConfig(timeout=300)
        assert config.timeout == 300

        # Invalid timeout
        with pytest.raises(ValidationError):
            MQTTConfig(timeout=0)

        with pytest.raises(ValidationError):
            MQTTConfig(timeout=301)

    def test_client_id_validation(self):
        """Test client ID validation."""
        # Valid client IDs
        valid_ids = [
            "client-001",
            "test_client",
            "abc123",
            "a",
            None,  # None is valid (auto-generated)
        ]

        for client_id in valid_ids:
            config = MQTTConfig(client_id=client_id)
            assert config.client_id == client_id

        # Invalid client IDs
        with pytest.raises(ValidationError):
            # Too long (max 23 chars)
            MQTTConfig(client_id="a" * 24)

        with pytest.raises(ValidationError):
            # Special characters
            MQTTConfig(client_id="client@001")

    def test_hostname_validation(self):
        """Test hostname validation."""
        # Valid hostnames
        valid_hostnames = [
            "localhost",
            "mqtt.example.com",
            "192.168.1.1",
            "broker-01",
        ]

        for hostname in valid_hostnames:
            config = MQTTConfig(hostname=hostname)
            assert config.hostname == hostname

        # Empty hostname should fail
        with pytest.raises(ValidationError):
            MQTTConfig(hostname="")

    def test_json_schema_examples(self):
        """Test that JSON schema examples are valid."""
        examples = MQTTConfig.model_config["json_schema_extra"]["examples"]

        for example in examples:
            config = MQTTConfig(**example)
            assert config is not None

    def test_serialization(self):
        """Test model serialization."""
        config = MQTTConfig(
            hostname="test.broker.com",
            port=1883,
            username="user",
            password="pass",
            topic_prefix="live",
        )

        # Serialize to dict
        data = config.model_dump()

        assert data["hostname"] == "test.broker.com"
        assert data["port"] == 1883
        assert data["username"] == "user"
        assert data["password"] == "pass"
        assert data["topic_prefix"] == "live"

        # Serialize to JSON
        json_str = config.model_dump_json()
        assert "test.broker.com" in json_str
        assert "live" in json_str

    def test_deserialization(self):
        """Test model deserialization."""
        data = {
            "hostname": "mqtt.example.com",
            "port": 8883,
            "username": "test",
            "qos": 2,
        }

        config = MQTTConfig(**data)

        assert config.hostname == "mqtt.example.com"
        assert config.port == 8883
        assert config.username == "test"
        assert config.qos == 2
        # Defaults should still apply
        assert config.topic_prefix == "live"
        assert config.keepalive == 60

    def test_tls_configuration(self):
        """Test TLS/SSL configuration."""
        # Non-TLS (default)
        config = MQTTConfig()
        assert config.use_tls is False
        assert config.port == 1883

        # TLS enabled
        config = MQTTConfig(use_tls=True, port=8883)
        assert config.use_tls is True
        assert config.port == 8883

    def test_authentication_configuration(self):
        """Test authentication configuration."""
        # No authentication
        config = MQTTConfig()
        assert config.username is None
        assert config.password is None

        # With authentication
        config = MQTTConfig(username="admin", password="secret123")
        assert config.username == "admin"
        assert config.password == "secret123"

    def test_session_configuration(self):
        """Test session configuration."""
        # Clean session (default)
        config = MQTTConfig()
        assert config.clean_session is True

        # Persistent session
        config = MQTTConfig(clean_session=False)
        assert config.clean_session is False

    def test_complex_configuration(self):
        """Test complex production-like configuration."""
        config = MQTTConfig(
            hostname="prod.mqtt.internal",
            port=8883,
            username="live-crew-prod",
            password="very-secure-password",
            topic_prefix="production/live-crew",
            qos=2,
            keepalive=300,
            timeout=60,
            client_id="live-crew-01",
            clean_session=False,
            use_tls=True,
        )

        # Verify all settings
        assert config.hostname == "prod.mqtt.internal"
        assert config.port == 8883
        assert config.username == "live-crew-prod"
        assert config.password == "very-secure-password"
        assert config.topic_prefix == "production/live-crew"
        assert config.qos == 2
        assert config.keepalive == 300
        assert config.timeout == 60
        assert config.client_id == "live-crew-01"
        assert config.clean_session is False
        assert config.use_tls is True
