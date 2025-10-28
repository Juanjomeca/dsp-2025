#!/bin/bash
# Script to activate the DSP virtual environment
# Usage: source activate_env.sh

echo "🔧 Activating DSP virtual environment..."

# Deactivate any existing virtual environment
if [[ -n "$VIRTUAL_ENV" ]]; then
    echo "🔄 Deactivating previous virtual environment..."
    deactivate
fi

# Activate the current project's virtual environment
source .venv/bin/activate

echo "✅ Virtual environment activated!"
echo "📦 Python location: $(which python || echo 'python not found, try python3')"
echo "📦 Python3 location: $(which python3)"
echo "📦 Pip location: $(which pip)"
echo "📦 Virtual environment: $VIRTUAL_ENV"
echo ""
echo "To deactivate, run: deactivate"
