#!/bin/bash
# Setup script for VibeShift on Unix-like systems (macOS/Linux)

set -e

echo "=========================================="
echo "🎵 VibeShift - Automated Setup"
echo "=========================================="

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
else
    echo "✅ uv is already installed"
fi

# Install Python 3.10
echo "🐍 Ensuring Python 3.10 is available..."
uv python install 3.10

# Create virtual environment and install dependencies
echo "📚 Installing dependencies..."
uv sync

echo ""
echo "=========================================="
echo "🎉 Setup completed successfully!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source .venv/bin/activate"
echo ""
echo "2. Start the application:"
echo "   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
