#!/bin/bash
# Start the unified viewer backend

echo "🎾 Starting Tennis Unified Viewer Backend..."
echo ""

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "Error: app.py not found. Please run from unified_viewer directory."
    exit 1
fi

# Activate virtual environment if it exists
if [ -d "../venv" ]; then
    echo "Activating virtual environment..."
    source ../venv/bin/activate
fi

# Start backend
echo "Starting FastAPI backend on http://localhost:8000"
python app.py


