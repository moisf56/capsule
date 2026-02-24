#!/bin/bash
# start_vision.sh — Launch llama-server for MedGemma vision inference
# Port 8081 (HAPI FHIR=8080, MCP=8082)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

LLAMA_SERVER="$PROJECT_DIR/ml-models/llama.cpp/build/bin/llama-server"
MODEL="$PROJECT_DIR/ml-models/gguf/medgemma-1.5-4b-it-Q4_K_M.gguf"
MMPROJ="$PROJECT_DIR/ml-models/gguf/medgemma-1.5-4b-it-mmproj.gguf"

# Validate files exist
if [ ! -f "$LLAMA_SERVER" ]; then
    echo "ERROR: llama-server not found at $LLAMA_SERVER"
    exit 1
fi
if [ ! -f "$MODEL" ]; then
    echo "ERROR: Model GGUF not found at $MODEL"
    exit 1
fi
if [ ! -f "$MMPROJ" ]; then
    echo "ERROR: mmproj GGUF not found at $MMPROJ"
    exit 1
fi

echo "Starting MedGemma Vision Server on port 8081..."
echo "Model: $MODEL"
echo "mmproj: $MMPROJ"

exec "$LLAMA_SERVER" \
    -m "$MODEL" \
    --mmproj "$MMPROJ" \
    --port 8081 \
    --host 0.0.0.0 \
    -ngl 99 \
    -c 4096 \
    -np 1 \
    --temp 0.3 \
    --no-webui
