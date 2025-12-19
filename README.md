Customer Service Chatbot — Mini Project

**Project Overview**
- **Purpose:** A small customer-service chatbot project that demonstrates local model-based inference and a minimal app wrapper for testing conversational flows.
- **Core files:** [app.py](app.py) (main app), [app_v2.py](app_v2.py) (alternate/updated runner), [EDA.ipynb](EDA.ipynb) (exploratory analysis), and the model folder [llama3_model_v1/](llama3_model_v1/).


**Model artifacts**
- The project stores adapter/checkpoint artifacts under [llama3_model_v1/meta-llama/](llama3_model_v1/meta-llama/) with multiple checkpoints (e.g., `checkpoint-2500`, `checkpoint-2520`). Fine-tuned adapter weights and tokenizer/config files. To switch checkpoints, point your loading code to the desired subfolder.

**Running the app**
- Quick local run (example):

```powershell
python app.py
```

- The app scripts will load model/config from the `llama3_model_v1` folder. If the scripts expect a particular path or environment variable, update the script or set an env var before running.

**Development & Notes**
- Use [EDA.ipynb](EDA.ipynb) for exploratory data analysis and to inspect any training logs under `llama3_model_v1/meta-llama/runs/`.

**Contact / Maintainer**
- For questions, contact the project owner or open an issue in this repository.
