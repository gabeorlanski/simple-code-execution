"""Configuration models and CLI parsing for the FastAPI server."""

import argparse
from typing import Optional

from pydantic import BaseModel
from pydantic import Field


class ServerConfig(BaseModel):
    """Configuration settings for the server."""

    host: str = Field(
        default="0.0.0.0",
        description="Host address to bind the server to"
    )
    port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="Port number to bind the server to"
    )
    max_concurrency: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of concurrent executions allowed"
    )
    log_level: str = Field(
        default="info",
        description="Logging level (debug, info, warning, error)"
    )
    reload: bool = Field(
        default=False,
        description="Enable auto-reload for development"
    )


# Global server settings
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
DEFAULT_MAX_CONCURRENCY = 10
DEFAULT_LOG_LEVEL = "info"
MIN_CONCURRENCY = 1
MAX_CONCURRENCY = 100


def parse_server_args() -> ServerConfig:
    """
    Parse command line arguments for server configuration.
    
    Returns:
        ServerConfig object with parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Simple Code Execution Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_HOST,
        help="Host address to bind the server to"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Port number to bind the server to"
    )
    
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=DEFAULT_MAX_CONCURRENCY,
        help=f"Maximum concurrent executions ({MIN_CONCURRENCY}-{MAX_CONCURRENCY})"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default=DEFAULT_LOG_LEVEL,
        choices=["debug", "info", "warning", "error"],
        help="Logging level"
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    args = parser.parse_args()
    
    # Validate max_concurrency range
    if not (MIN_CONCURRENCY <= args.max_concurrency <= MAX_CONCURRENCY):
        parser.error(
            f"max-concurrency must be between {MIN_CONCURRENCY} and {MAX_CONCURRENCY}"
        )
    
    # Validate port range
    if not (1 <= args.port <= 65535):
        parser.error("port must be between 1 and 65535")
    
    return ServerConfig(
        host=args.host,
        port=args.port,
        max_concurrency=args.max_concurrency,
        log_level=args.log_level,
        reload=args.reload
    )