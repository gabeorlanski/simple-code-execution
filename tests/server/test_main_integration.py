"""Integration tests for main server functionality with CLI config."""

from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest

from code_execution.server.config import ServerConfig
from code_execution.server.main import create_app


class TestCreateApp:
    """Test cases for create_app function."""

    def test_create_app_default_config(self):
        """Test creating app with default configuration."""
        app = create_app()
        
        # Should have default config in state
        assert hasattr(app.state, 'config')
        config = app.state.config
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.max_concurrency == 10

    def test_create_app_custom_config(self):
        """Test creating app with custom configuration."""
        custom_config = ServerConfig(
            host="127.0.0.1",
            port=9000,
            max_concurrency=5,
            log_level="debug",
            reload=True
        )
        
        app = create_app(custom_config)
        
        # Should have custom config in state
        assert hasattr(app.state, 'config')
        config = app.state.config
        assert config.host == "127.0.0.1"
        assert config.port == 9000
        assert config.max_concurrency == 5
        assert config.log_level == "debug"
        assert config.reload is True

    @patch('code_execution.server.main.get_concurrency_limiter')
    @pytest.mark.asyncio
    async def test_lifespan_initializes_concurrency(self, mock_get_limiter):
        """Test that lifespan initializes concurrency limiter."""
        # Setup mock
        mock_limiter = AsyncMock()
        mock_get_limiter.return_value = mock_limiter
        
        # Create app with custom config
        custom_config = ServerConfig(max_concurrency=15)
        app = create_app(custom_config)
        
        # Simulate lifespan startup
        async with app.router.lifespan_context(app):
            pass
        
        # Verify limiter was configured
        mock_limiter.set_max_concurrency.assert_called_once_with(15)

    @patch('sys.argv', ['script', '--host', '127.0.0.1', '--port', '9000'])
    @patch('uvicorn.run')
    def test_main_function_parses_args(self, mock_uvicorn_run):
        """Test that main function parses CLI args and passes to uvicorn."""
        from code_execution.server.main import main
        
        main()
        
        # Verify uvicorn.run was called with parsed config
        mock_uvicorn_run.assert_called_once()
        args, kwargs = mock_uvicorn_run.call_args
        
        # First argument should be the configured app
        configured_app = args[0]
        assert hasattr(configured_app.state, 'config')
        
        # Check that uvicorn was called with correct parameters
        assert kwargs['host'] == '127.0.0.1'
        assert kwargs['port'] == 9000
        assert kwargs['log_level'] == 'info'  # default
        assert kwargs['reload'] is False  # default

    @patch('sys.argv', [
        'script', 
        '--host', 'localhost',
        '--port', '8080',
        '--max-concurrency', '20',
        '--log-level', 'debug',
        '--reload'
    ])
    @patch('uvicorn.run')
    def test_main_function_all_args(self, mock_uvicorn_run):
        """Test main function with all CLI arguments."""
        from code_execution.server.main import main
        
        main()
        
        # Verify uvicorn.run was called with all parsed config
        mock_uvicorn_run.assert_called_once()
        args, kwargs = mock_uvicorn_run.call_args
        
        # Check app configuration
        configured_app = args[0]
        config = configured_app.state.config
        assert config.host == 'localhost'
        assert config.port == 8080
        assert config.max_concurrency == 20
        assert config.log_level == 'debug'
        assert config.reload is True
        
        # Check uvicorn parameters
        assert kwargs['host'] == 'localhost'
        assert kwargs['port'] == 8080
        assert kwargs['log_level'] == 'debug'
        assert kwargs['reload'] is True