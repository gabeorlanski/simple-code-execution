"""Unit tests for server configuration."""

from unittest.mock import patch

import pytest

from code_execution.server.config import DEFAULT_HOST
from code_execution.server.config import DEFAULT_LOG_LEVEL
from code_execution.server.config import DEFAULT_MAX_CONCURRENCY
from code_execution.server.config import DEFAULT_PORT
from code_execution.server.config import ServerConfig
from code_execution.server.config import parse_server_args


class TestServerConfig:
    """Test cases for ServerConfig model."""

    def test_server_config_defaults(self):
        """Test ServerConfig with default values."""
        config = ServerConfig()

        assert config.host == DEFAULT_HOST
        assert config.port == DEFAULT_PORT
        assert config.max_concurrency == DEFAULT_MAX_CONCURRENCY
        assert config.log_level == DEFAULT_LOG_LEVEL
        assert config.reload is False

    def test_server_config_custom_values(self):
        """Test ServerConfig with custom values."""
        config = ServerConfig(
            host="127.0.0.1",
            port=9000,
            max_concurrency=5,
            log_level="debug",
            reload=True,
        )

        assert config.host == "127.0.0.1"
        assert config.port == 9000
        assert config.max_concurrency == 5
        assert config.log_level == "debug"
        assert config.reload is True

    def test_server_config_validation_port_range(self):
        """Test port validation."""
        with pytest.raises(ValueError):
            ServerConfig(port=0)

        with pytest.raises(ValueError):
            ServerConfig(port=65536)

    def test_server_config_validation_concurrency_range(self):
        """Test max_concurrency validation."""
        with pytest.raises(ValueError):
            ServerConfig(max_concurrency=0)

        with pytest.raises(ValueError):
            ServerConfig(max_concurrency=101)


class TestParseServerArgs:
    """Test cases for CLI argument parsing."""

    @patch("sys.argv", ["script"])
    def test_parse_server_args_defaults(self):
        """Test parsing with default arguments."""
        config = parse_server_args()

        assert config.host == DEFAULT_HOST
        assert config.port == DEFAULT_PORT
        assert config.max_concurrency == DEFAULT_MAX_CONCURRENCY
        assert config.log_level == DEFAULT_LOG_LEVEL
        assert config.reload is False

    @patch(
        "sys.argv",
        [
            "script",
            "--host",
            "127.0.0.1",
            "--port",
            "9000",
            "--max-concurrency",
            "5",
            "--log-level",
            "debug",
            "--reload",
        ],
    )
    def test_parse_server_args_custom(self):
        """Test parsing with custom arguments."""
        config = parse_server_args()

        assert config.host == "127.0.0.1"
        assert config.port == 9000
        assert config.max_concurrency == 5
        assert config.log_level == "debug"
        assert config.reload is True

    @patch("sys.argv", ["script", "--max-concurrency", "0"])
    def test_parse_server_args_invalid_concurrency_low(self):
        """Test parsing with invalid low max-concurrency."""
        with pytest.raises(SystemExit):
            parse_server_args()

    @patch("sys.argv", ["script", "--max-concurrency", "101"])
    def test_parse_server_args_invalid_concurrency_high(self):
        """Test parsing with invalid high max-concurrency."""
        with pytest.raises(SystemExit):
            parse_server_args()

    @patch("sys.argv", ["script", "--port", "0"])
    def test_parse_server_args_invalid_port_low(self):
        """Test parsing with invalid low port."""
        with pytest.raises(SystemExit):
            parse_server_args()

    @patch("sys.argv", ["script", "--port", "65536"])
    def test_parse_server_args_invalid_port_high(self):
        """Test parsing with invalid high port."""
        with pytest.raises(SystemExit):
            parse_server_args()

    @patch("sys.argv", ["script", "--log-level", "invalid"])
    def test_parse_server_args_invalid_log_level(self):
        """Test parsing with invalid log level."""
        with pytest.raises(SystemExit):
            parse_server_args()

    @patch("sys.argv", ["script", "--help"])
    def test_parse_server_args_help(self):
        """Test that help flag works."""
        with pytest.raises(SystemExit) as exc_info:
            parse_server_args()
        # argparse exits with 0 for help
        assert exc_info.value.code == 0

    @patch(
        "sys.argv",
        [
            "script",
            "--host",
            "localhost",
            "--port",
            "8080",
            "--max-concurrency",
            "20",
        ],
    )
    def test_parse_server_args_valid_edge_cases(self):
        """Test parsing with valid edge case values."""
        config = parse_server_args()

        assert config.host == "localhost"
        assert config.port == 8080
        assert config.max_concurrency == 20

    @patch("sys.argv", ["script", "--max-concurrency", "1"])  # minimum valid
    def test_parse_server_args_min_concurrency(self):
        """Test parsing with minimum valid concurrency."""
        config = parse_server_args()

        assert config.max_concurrency == 1

    @patch("sys.argv", ["script", "--max-concurrency", "100"])  # maximum valid
    def test_parse_server_args_max_concurrency(self):
        """Test parsing with maximum valid concurrency."""
        config = parse_server_args()

        assert config.max_concurrency == 100
