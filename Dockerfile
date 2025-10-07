FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install FastMCP and dependencies directly
RUN pip install --no-cache-dir \
    fastmcp>=2.12.0 \
    aiohttp>=3.9.0 \
    pydantic>=2.0.0 \
    python-dotenv>=1.0.0

# Copy only the essential server file
COPY dune_mcp_server.py ./

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash mcp \
    && chown -R mcp:mcp /app
USER mcp

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV FASTMCP_LOG_LEVEL=INFO

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Run server with HTTP transport for cloud deployment
CMD ["python", "dune_mcp_server.py", "--transport", "http", "--host", "0.0.0.0", "--port", "8000"]
