#!/bin/zsh

set -e

echo "===================================="
echo "      Installing Ollama"
echo "===================================="

# Install Ollama (macOS + Linux)
curl --http1.1 -fsSL https://ollama.com/install.sh | sh

echo "===================================="
echo "      Starting Ollama Service"
echo "===================================="

# Start Ollama in the background
ollama serve > /dev/null 2>&1 &!

sleep 3

echo "===================================="
echo "   Downloading CPU Embedding Model"
echo "===================================="

# Best model for CPU machines
MODEL="nomic-embed-text"

ollama pull $MODEL

echo "===================================="
echo "       Testing the Model"
echo "===================================="

TEST='{"prompt":"hello world"}'
echo $TEST | ollama run $MODEL

echo "===================================="
echo "   Ollama Embedding Model Ready!"
echo "===================================="
