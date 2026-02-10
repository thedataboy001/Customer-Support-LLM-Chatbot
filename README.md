# Customer Service Chatbot

A customer-service chatbot application powered by locally fine-tuned Llama 3 models, with both Gradio and Streamlit interfaces.

## Project Overview

**Purpose:** Demonstrates local model-based inference with production-ready UI frameworks for customer service conversations using LLMs.

**Key Components:**
- [main.py](main.py) — Gradio-based chatbot interface
- [streamlit_app.py](streamlit_app.py) — Streamlit interface with LangGraph state management
- [src/model.py](src/model.py) — Model loading and inference logic (`TrainingModel` class)
- [src/agent.py](src/agent.py) — Agent utilities for extended functionality
- [notebook/notebook.ipynb](notebook.ipynb) — Exploratory data analysis and model inspection
- [llama3_model_v1/](llama3_model_v1/) — Fine-tuned model checkpoints and artifacts

## Model Artifacts

The project uses fine-tuned Llama 3 adapter weights stored in [llama3_model_v1/meta-llama/](llama3_model_v1/meta-llama/):

**Available Checkpoints:**
- `checkpoint-500` — Early training checkpoint
- `checkpoint-1000` — Mid-training checkpoint
- `checkpoint-1500` — Default checkpoint (used in UIs)
- `checkpoint-2000` — Advanced checkpoint
- `checkpoint-2500` — Performance optimized
- `checkpoint-2520` — Final checkpoint (latest)

Each checkpoint contains:
- `adapter_model.safetensors` — Fine-tuned adapter weights (LoRA)
- `adapter_config.json` — Adapter configuration
- `tokenizer.json`, `tokenizer_config.json` — Tokenizer files
- Training artifacts (`optimizer.pt`, `scheduler.pt`, etc.)

**Switch Checkpoints:** Update the `MODEL_PATH_DEFAULT` variable in the app scripts or pass a custom path to `TrainingModel()`.

## Running the Application

### Prerequisites

Ensure dependencies are installed:

```powershell
pip install -e .
```

Or install manually:

```powershell
pip install torch transformers peft langchain-core langgraph streamlit gradio datasets
```

### Gradio Interface (main.py)

Lightweight web UI for quick testing:

```powershell
python main.py
```

Access at `http://127.0.0.1:7860`

### Streamlit Interface (streamlit_app.py)

Full-featured interface with conversation history and agent logic:

```powershell
streamlit run streamlit_app.py
```

Access at `http://localhost:8501`

## Project Structure

```
├── main.py                           # Gradio chatbot interface
├── streamlit_app.py                  # Streamlit chatbot with LangGraph
├── notebook/
│   ├── notebook.ipynb                # Exploratory analysis Dependencies
├── pyproject.toml                    # Dependencies & project config
├── src/
│   ├── model.py                      # TrainingModel class
│   ├── agent.py                      # Agent utilities
│   └── __pycache__/
└── llama3_model_v1/
    └── meta-llama/
        ├── checkpoint-500/           # Early checkpoint
        ├── checkpoint-1000/          # Mid checkpoint
        ├── checkpoint-1500/          # Default (recommended)
        ├── checkpoint-2000/          # Advanced
        ├── checkpoint-2500/          # Performance optimized
        ├── checkpoint-2520/          # Latest
        └── runs/                     # TensorBoard training logs
```

## Development & Notes

- **Model Loading:** Uses `transformers` + `peft` (LoRA) for efficient adapter loading
- **Inference:** Powered by `torch` with configurable generation parameters
- **State Management:** Streamlit app uses LangGraph for multi-turn conversation history
- **Analysis:** Use [notebook.ipynb](notebook/notebook.ipynb) to inspect training logs and model behavior

## Dependencies

Key dependencies (see [pyproject.toml](pyproject.toml)):
- `torch>=2.10.0` — PyTorch for model inference
- `transformers>=5.1.0` — Hugging Face model loading
- `peft>=0.18.1` — Parameter-efficient fine-tuning (LoRA)
- `gradio>=6.5.1` — Gradio web UI
- `streamlit>=1.54.0` — Streamlit web framework
- `langgraph>=1.0.8` — State graph for multi-turn conversations
- `langchain-core>=1.2.9` — LangChain utilities
- `datasets>=4.5.0` — Dataset handling
- `trl>=0.27.2` — Training utilities

## Getting Started

1. **Clone & Install:**
   ```powershell
   cd customer_service_chatbot
   pip install -e .
   ```

2. **Run the UI:**
   ```powershell
   # Option 1: Gradio (simpler)
   python main.py
   
   # Option 2: Streamlit (more features)
   streamlit run streamlit_app.py
   ```

3. **Explore the Model:**
   - Open [notebook.ipynb](notebook/notebook.ipynb) in Jupyter to inspect training logs and model performance

## Troubleshooting

- **Model not found:** Verify `llama3_model_v1/meta-llama/checkpoint-1500/` exists
- **OOM errors:** Reduce batch size or use a smaller checkpoint
- **Slow inference:** Ensure GPU is available (`torch.cuda.is_available()`)

## Contact

For questions or issues, open an issue in this repository or contact the project maintainer.

