#!/bin/bash
# Render startup script for Image Classification API

echo "🚀 Starting Image Classification API on Render..."
echo "📦 Model: google/vit-base-patch16-224"

# Start the FastAPI application with uvicorn
#!/usr/bin/env bash
uvicorn main:app --host 0.0.0.0 --port $PORT
