# Geothermal RAG v5.0 - Quick Start Guide

This guide will help you get the Geothermal RAG system up and running quickly.

## Prerequisites

Before you begin, ensure you have:
- **Python 3.10+** installed
- **8GB RAM** minimum
- **~10GB free disk space** (for models)
- **Internet connection** (for initial setup)

## Installation Steps

### 1. Install Ollama

Ollama is required to run the AI models locally.

```bash
# Visit https://ollama.ai and download the installer
# Or use the command line:
curl -fsSL https://ollama.ai/install.sh | sh
```

### 2. Pull Required Models

Download the AI models needed by the system:

```bash
# Main chat model (7B parameters, ~4.7GB)
ollama pull qwen2.5:7b

# Extraction and judge model (3B parameters, ~2GB)
ollama pull qwen2.5:3b

# Embedding model for semantic search (~274MB)
ollama pull nomic-embed-text
```

**Note:** This will download approximately 7GB of models. First-time download may take 10-30 minutes depending on your internet speed.

### 3. Install Python Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

This installs:
- Gradio (web interface)
- ChromaDB (vector database)
- PyMuPDF & pdfplumber (PDF processing)
- And other supporting libraries

### 4. Verify Installation

Run the evaluation script to ensure everything is set up correctly:

```bash
python evaluate_system.py
```

You should see:
```
✅ ALL TESTS PASSED! System is ready for deployment.
```

## Running the System

### Option 1: Start Ollama and Application Separately

```bash
# Terminal 1: Start Ollama server
ollama serve

# Terminal 2: Start the application
python app_working.py
```

### Option 2: Quick Start (Recommended)

The system includes start scripts for convenience:

**Windows:**
```cmd
start.bat
```

**Linux/Mac:**
```bash
./start.sh
```

### 3. Access the Web Interface

Once started, open your browser and navigate to:
```
http://127.0.0.1:7860
```

You should see the Geothermal RAG v5.0 interface.

## Using the System

### Step 1: Upload Documents

1. Click **"Select PDF Files"** in the left panel
2. Choose one or more geothermal well report PDFs
3. Click **"Index Files"** button
4. Wait 60-120 seconds for processing
5. You'll see a success message with indexed wells

### Step 2: Ask Questions

Three query modes are automatically detected:

**Factual Q&A:**
```
Q: What is the total depth of HAG-GT-01?
```

**Summary Mode:**
```
Q: Give a summary of NLW-GT-02-S1
```

**Trajectory Extraction:**
```
Q: Extract trajectory of HAG-GT-01
```

### Step 3: Multi-Turn Conversations

The system remembers your previous questions:

```
Q1: What is the depth of HAG-GT-01?
Q2: What about NLW-GT-02-S1?
Q3: Which well was deeper?  # Uses context from Q1 & Q2
```

### Step 4: Nodal Analysis (Optional)

After extracting a trajectory:

1. Click the **"Run Nodal Analysis"** button that appears
2. The system runs production analysis
3. Results are displayed below

## Example Queries

### For Testing

Try these queries with sample geothermal documents:

1. **Depth Information:**
   ```
   What is the total depth of [WELL_NAME]?
   What is the measured depth vs true vertical depth?
   ```

2. **Drilling Details:**
   ```
   When was [WELL_NAME] drilled?
   What formations were encountered?
   ```

3. **Technical Data:**
   ```
   Extract trajectory table from [WELL_NAME]
   What is the casing design?
   What were the drilling problems?
   ```

4. **Summary:**
   ```
   Give me a summary of [WELL_NAME]
   Summarize the completion details
   ```

## Troubleshooting

### Ollama Not Running
```
Error: Connection refused to localhost:11434
```
**Solution:** Start Ollama with `ollama serve`

### Module Import Error
```
ModuleNotFoundError: No module named 'gradio'
```
**Solution:** Run `pip install -r requirements.txt`

### Out of Memory
```
Error: Out of memory
```
**Solution:** Close other applications or use a smaller model configuration

### Slow Response Times
- First query is always slower (model loading)
- Subsequent queries should be 5-30 seconds
- Check if Ollama is using GPU (much faster)

### No Documents Found
```
Error: Please upload documents first
```
**Solution:** Index at least one PDF document before querying

## System Performance

### Expected Performance
- **Document Indexing:** 60-120 seconds for small PDFs
- **Factual Q&A:** 5-15 seconds per query
- **Summary Generation:** 15-30 seconds
- **Trajectory Extraction:** 10-20 seconds

### Hardware Recommendations
- **Minimum:** 8GB RAM, CPU-only
- **Recommended:** 16GB RAM, GPU with 6GB+ VRAM
- **Optimal:** 32GB RAM, GPU with 12GB+ VRAM

## Advanced Configuration

The system can be customized by editing `config/config_v5.yaml`:

### Common Adjustments

**Change chunk sizes:**
```yaml
embedding_strategies:
  factual_qa:
    chunk_size: 350  # Increase for more context
    chunk_overlap: 120
```

**Adjust confidence thresholds:**
```yaml
agents:
  judge:
    min_confidence: 0.70  # Lower for more lenient validation
```

**Modify top-K retrieval:**
```yaml
agents:
  rag_retrieval:
    top_k_factual: 10  # Increase for more context chunks
```

## Running Evaluation Scripts

### System Evaluation
Tests all core functionality:
```bash
python evaluate_system.py
```

### System Demonstration
Shows capabilities without Ollama:
```bash
python demo_system.py
```

## Getting Help

### Check Logs
The system provides real-time logging in the web interface. Check the **"System Activity"** panel for detailed agent logs.

### Common Issues

1. **Timeout errors:** Increase `timeout` in config
2. **Low confidence warnings:** Normal for ambiguous questions
3. **No trajectory found:** Document may not contain trajectory tables

### Documentation

- **Full Evaluation Report:** See `EVALUATION_REPORT.md`
- **README:** See `README.md`
- **Configuration:** See `config/config_v5.yaml`

## Next Steps

Once the system is running:

1. Upload your geothermal well reports
2. Experiment with different query types
3. Use multi-turn conversations for complex analysis
4. Extract trajectories for nodal analysis
5. Review validation confidence scores

## System Status Check

To verify everything is working:

```bash
# Check Python environment
python --version  # Should be 3.10+

# Check Ollama is running
curl http://localhost:11434/api/tags  # Should list models

# Check models are available
ollama list  # Should show qwen2.5:7b, qwen2.5:3b, nomic-embed-text

# Run evaluation
python evaluate_system.py  # Should pass all tests
```

---

**System Ready! 🎉**

The Geothermal RAG v5.0 system is now ready to process your geothermal engineering documents with advanced AI capabilities.

For questions or issues, refer to the evaluation report or check the agent logs in the web interface.
