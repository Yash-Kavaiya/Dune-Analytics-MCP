"""
Dune Analytics MCP Server
A comprehensive Model Context Protocol server for interacting with Dune Analytics API
Built with FastMCP 2.x - Updated for deployment compatibility
"""

import os
import asyncio
import logging
import sys
import argparse
from typing import Dict, List, Optional, Union, Literal, Annotated
from datetime import datetime, timedelta
import json
import aiohttp
from pydantic import BaseModel, Field
from dataclasses import dataclass

from fastmcp import FastMCP, Context
from fastmcp.tools.tool import ToolResult
from fastmcp.exceptions import ToolError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP(
    name="Dune Analytics MCP Server",
    description="Comprehensive MCP server for interacting with Dune Analytics API"
)

# Pydantic models for structured data
class QueryParameter(BaseModel):
    """Query parameter for Dune queries"""
    name: str = Field(description="Parameter name")
    value: Union[str, int, float, bool] = Field(description="Parameter value")
    type: Literal["text", "number", "date", "enum"] = Field(description="Parameter type")

class ExecutionResult(BaseModel):
    """Execution result from Dune query"""
    execution_id: str
    query_id: int
    is_execution_finished: bool
    state: str
    submitted_at: datetime
    expires_at: Optional[datetime] = None
    execution_started_at: Optional[datetime] = None
    execution_ended_at: Optional[datetime] = None

class QueryResult(BaseModel):
    """Query result data"""
    rows: List[Dict]
    metadata: Dict
    next_uri: Optional[str] = None
    next_offset: Optional[int] = None

@dataclass
class DuneConfig:
    """Configuration for Dune API"""
    api_key: str
    base_url: str = "https://api.dune.com/api/v1"
    timeout: int = 300

class DuneClient:
    """Async client for Dune Analytics API"""

    def __init__(self, config: DuneConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers={"X-DUNE-API-KEY": self.config.api_key},
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def execute_query(
        self, 
        query_id: int, 
        parameters: Optional[List[QueryParameter]] = None,
        performance: Literal["medium", "large"] = "medium"
    ) -> ExecutionResult:
        """Execute a Dune query"""
        if not self.session:
            raise ToolError("Client session not initialized")

        url = f"{self.config.base_url}/query/{query_id}/execute"

        payload = {
            "performance": performance
        }

        if parameters:
            payload["query_parameters"] = [
                {
                    "name": p.name,
                    "value": str(p.value),
                    "type": p.type
                }
                for p in parameters
            ]

        async with self.session.post(url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise ToolError(f"Failed to execute query {query_id}: {error_text}")

            data = await response.json()
            return ExecutionResult(**data)

    async def get_execution_status(self, execution_id: str) -> ExecutionResult:
        """Get execution status"""
        if not self.session:
            raise ToolError("Client session not initialized")

        url = f"{self.config.base_url}/execution/{execution_id}/status"

        async with self.session.get(url) as response:
            if response.status != 200:
                error_text = await response.text()
                raise ToolError(f"Failed to get execution status: {error_text}")

            data = await response.json()
            return ExecutionResult(**data)

    async def get_execution_results(
        self, 
        execution_id: str,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> QueryResult:
        """Get execution results"""
        if not self.session:
            raise ToolError("Client session not initialized")

        url = f"{self.config.base_url}/execution/{execution_id}/results"
        params = {}

        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset

        async with self.session.get(url, params=params) as response:
            if response.status != 200:
                error_text = await response.text()
                raise ToolError(f"Failed to get execution results: {error_text}")

            data = await response.json()
            return QueryResult(
                rows=data.get("result", {}).get("rows", []),
                metadata=data.get("result", {}).get("metadata", {}),
                next_uri=data.get("next_uri"),
                next_offset=data.get("next_offset")
            )

    async def get_latest_result(
        self, 
        query_id: int,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> QueryResult:
        """Get latest query result without executing"""
        if not self.session:
            raise ToolError("Client session not initialized")

        url = f"{self.config.base_url}/query/{query_id}/results"
        params = {}

        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset

        async with self.session.get(url, params=params) as response:
            if response.status != 200:
                error_text = await response.text()
                raise ToolError(f"Failed to get latest result: {error_text}")

            data = await response.json()
            return QueryResult(
                rows=data.get("result", {}).get("rows", []),
                metadata=data.get("result", {}).get("metadata", {}),
                next_uri=data.get("next_uri"),
                next_offset=data.get("next_offset")
            )

    async def wait_for_execution(
        self, 
        execution_id: str, 
        max_attempts: int = 60, 
        delay: int = 5
    ) -> ExecutionResult:
        """Wait for execution to complete"""
        for attempt in range(max_attempts):
            status = await self.get_execution_status(execution_id)

            if status.is_execution_finished:
                return status

            if status.state in ["QUERY_STATE_FAILED", "QUERY_STATE_CANCELLED"]:
                raise ToolError(f"Query execution failed with state: {status.state}")

            await asyncio.sleep(delay)

        raise ToolError(f"Execution timeout after {max_attempts * delay} seconds")

def get_dune_client() -> DuneClient:
    """Get configured Dune client"""
    api_key = os.getenv("DUNE_API_KEY")
    if not api_key:
        raise ToolError("DUNE_API_KEY environment variable not set")

    base_url = os.getenv("DUNE_API_BASE_URL", "https://api.dune.com/api/v1")
    timeout = int(os.getenv("DUNE_API_TIMEOUT", "300"))

    config = DuneConfig(api_key=api_key, base_url=base_url, timeout=timeout)
    return DuneClient(config)

# MCP Tools
@mcp.tool(
    description="Execute a Dune Analytics query and return results",
    annotations={
        "title": "Execute Dune Query",
        "readOnlyHint": False,
        "openWorldHint": True
    }
)
async def execute_dune_query(
    query_id: Annotated[int, Field(description="Dune query ID to execute", gt=0)],
    parameters: Annotated[
        Optional[List[Dict[str, Union[str, int, float]]]], 
        Field(description="Query parameters as list of dicts with 'name', 'value', 'type' keys")
    ] = None,
    performance: Annotated[
        Literal["medium", "large"], 
        Field(description="Execution performance tier")
    ] = "medium",
    wait_for_completion: Annotated[
        bool, 
        Field(description="Whether to wait for execution completion")
    ] = True,
    ctx: Context = None
) -> ExecutionResult:
    """Execute a Dune Analytics query with optional parameters"""

    if ctx:
        await ctx.info(f"Executing Dune query {query_id}")

    # Parse parameters
    query_params = []
    if parameters:
        for param in parameters:
            if not all(k in param for k in ["name", "value", "type"]):
                raise ToolError("Parameters must have 'name', 'value', and 'type' keys")
            query_params.append(QueryParameter(**param))

    async with get_dune_client() as client:
        # Execute query
        execution = await client.execute_query(query_id, query_params, performance)

        if ctx:
            await ctx.info(f"Query execution started: {execution.execution_id}")

        if wait_for_completion:
            if ctx:
                await ctx.info("Waiting for execution to complete...")
            execution = await client.wait_for_execution(execution.execution_id)

            if ctx:
                await ctx.info(f"Execution completed with state: {execution.state}")

        return execution

@mcp.tool(
    description="Get the results of a Dune query execution",
    annotations={
        "title": "Get Query Results",
        "readOnlyHint": True,
        "openWorldHint": True
    }
)
async def get_query_results(
    execution_id: Annotated[str, Field(description="Execution ID from query execution")],
    limit: Annotated[
        Optional[int], 
        Field(description="Maximum number of rows to return", ge=1, le=100000)
    ] = None,
    offset: Annotated[
        Optional[int], 
        Field(description="Number of rows to skip", ge=0)
    ] = None,
    ctx: Context = None
) -> QueryResult:
    """Get results from a Dune query execution"""

    if ctx:
        await ctx.info(f"Fetching results for execution {execution_id}")

    async with get_dune_client() as client:
        results = await client.get_execution_results(execution_id, limit, offset)

        if ctx:
            await ctx.info(f"Retrieved {len(results.rows)} rows")

        return results

@mcp.tool(
    description="Get the latest cached results of a Dune query without executing it",
    annotations={
        "title": "Get Latest Results",
        "readOnlyHint": True,
        "openWorldHint": True
    }
)
async def get_latest_results(
    query_id: Annotated[int, Field(description="Dune query ID", gt=0)],
    limit: Annotated[
        Optional[int], 
        Field(description="Maximum number of rows to return", ge=1, le=100000)
    ] = None,
    offset: Annotated[
        Optional[int], 
        Field(description="Number of rows to skip", ge=0)
    ] = None,
    ctx: Context = None
) -> QueryResult:
    """Get the latest cached results of a Dune query without executing it"""

    if ctx:
        await ctx.info(f"Fetching latest results for query {query_id}")

    async with get_dune_client() as client:
        results = await client.get_latest_result(query_id, limit, offset)

        if ctx:
            await ctx.info(f"Retrieved {len(results.rows)} cached rows")

        return results

@mcp.tool(
    description="Execute a Dune query and get results in one operation",
    annotations={
        "title": "Run Query Complete",
        "readOnlyHint": False,
        "openWorldHint": True
    }
)
async def run_query_complete(
    query_id: Annotated[int, Field(description="Dune query ID to execute", gt=0)],
    parameters: Annotated[
        Optional[List[Dict[str, Union[str, int, float]]]], 
        Field(description="Query parameters as list of dicts")
    ] = None,
    performance: Annotated[
        Literal["medium", "large"], 
        Field(description="Execution performance tier")
    ] = "medium",
    limit: Annotated[
        Optional[int], 
        Field(description="Maximum number of rows to return", ge=1, le=100000)
    ] = None,
    offset: Annotated[
        Optional[int], 
        Field(description="Number of rows to skip", ge=0)
    ] = None,
    ctx: Context = None
) -> ToolResult:
    """Execute a Dune query and return results in one operation"""

    if ctx:
        await ctx.info(f"Running complete query execution for {query_id}")
        await ctx.report_progress(0, 100)

    # Parse parameters
    query_params = []
    if parameters:
        for param in parameters:
            if not all(k in param for k in ["name", "value", "type"]):
                raise ToolError("Parameters must have 'name', 'value', and 'type' keys")
            query_params.append(QueryParameter(**param))

    async with get_dune_client() as client:
        # Execute query
        if ctx:
            await ctx.report_progress(20, 100)
        execution = await client.execute_query(query_id, query_params, performance)

        if ctx:
            await ctx.info(f"Execution started: {execution.execution_id}")
            await ctx.report_progress(40, 100)

        # Wait for completion
        execution = await client.wait_for_execution(execution.execution_id)

        if ctx:
            await ctx.report_progress(80, 100)

        # Get results
        results = await client.get_execution_results(execution.execution_id, limit, offset)

        if ctx:
            await ctx.info(f"Query completed successfully with {len(results.rows)} rows")
            await ctx.report_progress(100, 100)

        # Return structured result with both execution info and data
        return ToolResult(
            content=[
                {
                    "type": "text",
                    "text": f"Query {query_id} executed successfully.\n"
                           f"Execution ID: {execution.execution_id}\n"
                           f"State: {execution.state}\n"
                           f"Rows returned: {len(results.rows)}\n"
                           f"Execution time: {execution.execution_started_at} to {execution.execution_ended_at}"
                }
            ],
            structured_content={
                "execution": execution.model_dump(),
                "results": results.model_dump()
            }
        )

@mcp.tool(
    description="Check the execution status of a Dune query",
    annotations={
        "title": "Check Execution Status",
        "readOnlyHint": True,
        "openWorldHint": True
    }
)
async def check_execution_status(
    execution_id: Annotated[str, Field(description="Execution ID to check")],
    ctx: Context = None
) -> ExecutionResult:
    """Check the execution status of a Dune query"""

    if ctx:
        await ctx.info(f"Checking status for execution {execution_id}")

    async with get_dune_client() as client:
        status = await client.get_execution_status(execution_id)

        if ctx:
            await ctx.info(f"Execution status: {status.state}")

        return status

@mcp.tool(
    description="Cancel a running Dune query execution",
    annotations={
        "title": "Cancel Execution",
        "readOnlyHint": False,
        "destructiveHint": True,
        "openWorldHint": True
    }
)
async def cancel_execution(
    execution_id: Annotated[str, Field(description="Execution ID to cancel")],
    ctx: Context = None
) -> Dict[str, str]:
    """Cancel a running Dune query execution"""

    if ctx:
        await ctx.info(f"Cancelling execution {execution_id}")

    async with get_dune_client() as client:
        url = f"{client.config.base_url}/execution/{execution_id}/cancel"

        async with client.session.post(url) as response:
            if response.status != 200:
                error_text = await response.text()
                raise ToolError(f"Failed to cancel execution: {error_text}")

            if ctx:
                await ctx.info(f"Execution {execution_id} cancelled successfully")

            return {"status": "cancelled", "execution_id": execution_id}

@mcp.tool(
    description="Get query information and metadata",
    annotations={
        "title": "Get Query Info",
        "readOnlyHint": True,
        "openWorldHint": True
    }
)
async def get_query_info(
    query_id: Annotated[int, Field(description="Dune query ID", gt=0)],
    ctx: Context = None
) -> Dict:
    """Get information and metadata about a Dune query"""

    if ctx:
        await ctx.info(f"Fetching info for query {query_id}")

    async with get_dune_client() as client:
        url = f"{client.config.base_url}/query/{query_id}"

        async with client.session.get(url) as response:
            if response.status != 200:
                error_text = await response.text()
                raise ToolError(f"Failed to get query info: {error_text}")

            data = await response.json()

            if ctx:
                await ctx.info(f"Retrieved query info for '{data.get('name', 'Unknown')}'")

            return data

@mcp.tool(
    description="List available query parameters for a Dune query",
    annotations={
        "title": "List Query Parameters",
        "readOnlyHint": True,
        "openWorldHint": True
    }
)
async def list_query_parameters(
    query_id: Annotated[int, Field(description="Dune query ID", gt=0)],
    ctx: Context = None
) -> List[Dict]:
    """List available parameters for a Dune query"""

    if ctx:
        await ctx.info(f"Listing parameters for query {query_id}")

    # Get query info first
    query_info = await get_query_info(query_id, ctx)

    # Extract parameters from query
    parameters = query_info.get("parameters", [])

    if ctx:
        await ctx.info(f"Found {len(parameters)} parameters")

    return parameters

# Resources
@mcp.resource(
    uri="dune://queries/{query_id}",
    description="Get Dune query information by ID",
    name="Query Information"
)
async def query_resource(query_id: str) -> str:
    """Resource to get query information"""
    try:
        query_id_int = int(query_id)
        query_info = await get_query_info(query_id_int)
        return json.dumps(query_info, indent=2, default=str)
    except Exception as e:
        return f"Error fetching query {query_id}: {str(e)}"

@mcp.resource(
    uri="dune://results/{execution_id}",
    description="Get execution results by execution ID",
    name="Execution Results"
)
async def results_resource(execution_id: str) -> str:
    """Resource to get execution results"""
    try:
        async with get_dune_client() as client:
            results = await client.get_execution_results(execution_id)
            return json.dumps(results.model_dump(), indent=2, default=str)
    except Exception as e:
        return f"Error fetching results for {execution_id}: {str(e)}"

# Health check endpoint for deployments
@mcp.tool(
    description="Health check endpoint for deployment monitoring",
    annotations={
        "title": "Health Check",
        "readOnlyHint": True,
        "openWorldHint": False
    }
)
async def health_check() -> Dict[str, str]:
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Dune Analytics MCP Server",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Dune Analytics MCP Server")
    parser.add_argument(
        "--transport", 
        choices=["stdio", "http", "sse"], 
        default="stdio",
        help="Transport protocol to use"
    )
    parser.add_argument(
        "--host", 
        default="127.0.0.1",
        help="Host to bind to (for HTTP transport)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000,
        help="Port to bind to (for HTTP transport)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    return parser.parse_args()

async def main_async():
    """Async main function for deployment scenarios"""
    args = parse_args()

    # Configure logging
    log_level = getattr(logging, args.log_level)
    logging.getLogger().setLevel(log_level)

    logger.info(f"Starting Dune Analytics MCP Server")
    logger.info(f"Transport: {args.transport}")

    if args.transport in ["http", "sse"]:
        logger.info(f"Host: {args.host}:{args.port}")

    # Check for required environment variables
    if not os.getenv("DUNE_API_KEY"):
        logger.error("DUNE_API_KEY environment variable not set")
        logger.error("Please set your Dune API key before starting the server")
        sys.exit(1)

    try:
        # Run server with specified transport
        await mcp.run_async(
            transport=args.transport,
            host=args.host if args.transport in ["http", "sse"] else None,
            port=args.port if args.transport in ["http", "sse"] else None
        )
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)

def main():
    """Main entry point"""
    # Check if we're running in an async context already
    try:
        loop = asyncio.get_running_loop()
        # If we're already in an async context, run the async main
        loop.create_task(main_async())
    except RuntimeError:
        # Not in an async context, run normally
        asyncio.run(main_async())

# Server configuration and startup
if __name__ == "__main__":
    # Handle both deployment scenarios and local development
    if len(sys.argv) == 1:
        # No arguments provided, use default stdio transport
        mcp.run(transport="stdio")
    else:
        # Arguments provided, use main function with argument parsing
        main()
