#!/bin/bash
# Start the unified viewer frontend (development mode)

echo "🎾 Starting Tennis Unified Viewer Frontend..."
echo ""

# Check if we're in the right directory
if [ ! -d "frontend" ]; then
    echo "Error: frontend directory not found."
    exit 1
fi

cd frontend

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

# Start frontend dev server
echo "Starting React dev server on http://localhost:3000"
npm start


