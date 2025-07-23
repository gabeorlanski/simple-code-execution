"""FastAPI server for code execution."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, Optional

import structlog
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import status
from fastapi.responses import JSONResponse

from code_execution.server.concurrency import get_concurrency_limiter
from code_execution.server.config import ServerConfig
from code_execution.server.config import parse_server_args
from code_execution.server.models import ApiResponse
from code_execution.server.models import ErrorResponse
from code_execution.server.models import ExecuteRequest
from code_execution.server.models import ExecuteResponse
from code_execution.server.service import ExecutionService

# Configure logging
logger = structlog.get_logger()

# Server configuration
SERVER_TITLE = "Simple Code Execution Server"
SERVER_DESCRIPTION = (
    "A FastAPI server for executing code through various runners"
)
SERVER_VERSION = "0.1.0"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.

    Args:
        app: The FastAPI application instance
    """
    # Get server config from app state
    config: ServerConfig = getattr(app.state, "config", ServerConfig())

    logger.info(
        "Starting Simple Code Execution Server",
        host=config.host,
        port=config.port,
        max_concurrency=config.max_concurrency,
    )

    # Initialize concurrency limiter with configured limit
    limiter = get_concurrency_limiter()
    await limiter.set_max_concurrency(config.max_concurrency)

    yield

    logger.info("Shutting down Simple Code Execution Server")


# Create FastAPI application
app = FastAPI(
    title=SERVER_TITLE,
    description=SERVER_DESCRIPTION,
    version=SERVER_VERSION,
    lifespan=lifespan,
)


@app.exception_handler(ValueError)
async def value_error_handler(_request, exc: ValueError) -> JSONResponse:
    """
    Handle ValueError exceptions.

    Args:
        _request: The HTTP request (unused)
        exc: The ValueError exception

    Returns:
        JSON error response
    """
    logger.warning("Validation error", error=str(exc))
    error_response = ErrorResponse(error="Validation Error", details=str(exc))
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=error_response.model_dump(),
    )


@app.exception_handler(RuntimeError)
async def runtime_error_handler(_request, exc: RuntimeError) -> JSONResponse:
    """
    Handle RuntimeError exceptions.

    Args:
        _request: The HTTP request (unused)
        exc: The RuntimeError exception

    Returns:
        JSON error response
    """
    logger.error("Runtime error", error=str(exc))
    error_response = ErrorResponse(error="Execution Error", details=str(exc))
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.model_dump(),
    )


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint.

    Returns:
        Health status
    """
    return {"status": "healthy"}


@app.get("/queue/status")
async def get_queue_status() -> Dict[str, int]:
    """
    Get current execution queue status.

    Returns:
        Queue status information
    """
    limiter = get_concurrency_limiter()
    return limiter.get_queue_info()


@app.post("/execute", response_model=ApiResponse)
async def execute_code(request: ExecuteRequest) -> ExecuteResponse:
    """
    Execute code through the specified runner.

    Args:
        request: The execution request containing files, commands, and options

    Returns:
        The execution result

    Raises:
        HTTPException: If execution fails
    """
    logger.info(
        "Received execution request",
        files=list(request.files.keys()),
        num_commands=len(request.commands),
        executable_type=request.executable_type,
        early_stopping=request.early_stopping,
    )

    try:
        # Execute the request through the service
        result, metadata = await ExecutionService.execute_request(request)

        # Return successful response with execution metadata
        response = ExecuteResponse(result=result)
        logger.debug(
            "Returning successful response",
            execution_id=metadata.get("execution_id"),
        )
        return response

    except ValueError as e:
        # Validation errors are handled by the exception handler
        raise e

    except RuntimeError as e:
        # Runtime errors are handled by the exception handler
        raise e

    except Exception as e:
        # Catch-all for unexpected errors
        logger.error("Unexpected error during execution", error=str(e))
        error_response = ErrorResponse(
            error="Internal Server Error",
            details="An unexpected error occurred",
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_response.model_dump(),
        ) from e


# Application entry point for uvicorn
def create_app(config: Optional[ServerConfig] = None) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        config: Server configuration (if None, creates default)

    Returns:
        The configured FastAPI application
    """
    if config is None:
        config = ServerConfig()

    # Store config in app state for lifespan access
    app.state.config = config

    return app


def main() -> None:
    """
    Main entry point for running the server with CLI arguments.
    """
    import uvicorn

    # Parse command line arguments
    config = parse_server_args()

    # Create app with configuration
    configured_app = create_app(config)

    logger.info(
        "Starting server with configuration",
        host=config.host,
        port=config.port,
        max_concurrency=config.max_concurrency,
        log_level=config.log_level,
        reload=config.reload,
    )

    # Run the server
    uvicorn.run(
        configured_app,
        host=config.host,
        port=config.port,
        log_level=config.log_level,
        reload=config.reload,
    )


if __name__ == "__main__":
    main()
