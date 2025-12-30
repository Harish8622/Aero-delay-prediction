# Aero Delay Prediction

End-to-end ML pipeline for predicting flight delays, plus an interactive agent for querying results.

## Architecture Diagram

[Place architecture diagram here]

## Repository Structure (High Level)

- `ml_pipelines/` preprocessing, training, and full pipeline runners
- `data/` raw and cached datasets (local only)
- `models/` trained artifacts and evaluation outputs
- `agent/` interactive agent interface
- `src/` shared utilities and core code
- `notebooks/` exploration and experiments
- `project_report.md` detailed training results and analysis

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root with required credentials:

```bash
# .env
BASE_PATH=/Users/harish/Desktop/Aero-delay-prediction
OPENAI_API_KEY=your_key_here
VISUAL_CROSSING_KEY=your_key_here
```

## Data

1. Download flight delay and cancellation dataset:
   ```python
   download_data_path = kagglehub.dataset_download("patrickzel/flight-delay-and-cancellation-dataset-2019-2023")
   ```
2. Download airport metadata from OpenFlight.
3. Save raw data under `data/` (for example, `data/raw/`), then keep derived caches under `data/cache/`.

## Run the Pipeline

From the project root:

```bash
python -m ml_pipelines.run_preprocessing
python -m ml_pipelines.run_training
```

To run the full pipeline end-to-end:

```bash
python -m ml_pipelines.full_training_pipeline
```

## Training Outputs

- Models and evaluation artifacts are written under `models/`.
- Use `project_report.md` for all training results, metrics, and analysis details.

## Run the Agent

From the project root:

```bash
python -m agent.customer_agent
```

## Run the Web App (Localhost)

From the project root:

```bash
pip install flask
python -m agent.web_app
```

Then open `http://127.0.0.1:8000` in your browser.

## Next Steps

- Move shared core functions into `src/` and rename agent core to `agent_core/`.
- Add CI pipeline.
- Add tests (unit + pipeline smoke test).
