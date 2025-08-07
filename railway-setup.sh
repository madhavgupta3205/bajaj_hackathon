#!/bin/bash
# Railway startup script

# Set Python environment
export PYTHONPATH="${PYTHONPATH}:/app"
export PYTHONUNBUFFERED=1

# Create necessary directories
mkdir -p /app/data/chroma_db
mkdir -p /app/data/temp
mkdir -p /app/data/documents

# Set proper permissions
chmod -R 755 /app/data

echo "Environment setup complete"
