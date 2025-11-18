# Geothermal RAG v5.0

## Quick Start

### Prerequisites
* Python 3.10+
* Ollama
* 8GB RAM

### Steps
1. Install Ollama from https://ollama.ai
2. Pull models:
   ollama pull qwen2.5:7b
   ollama pull qwen2.5:3b
   ollama pull nomic-embed-text

3. Start Ollama:
   ollama serve

4. Run launcher:
   Windows: start.bat
   Linux/Mac: ./start.sh

5. Open http://127.0.0.1:7860

## Features
* Ensemble validation (85-95% accuracy)
* Enhanced trajectory extraction
* Multi-turn conversations
* Exact citations

## Troubleshooting
* Ollama not running: ollama serve
* Module error: python test_setup.py
* Crash: start_manual.bat

Built for geothermal engineering
