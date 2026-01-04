# Aero Delay Prediction

Passengers are often caught off guard by delays. This project uses years of
historical flight data to build an end-to-end delay prediction pipeline and pairs
it with a LangGraph conversational agent. The agent combines the delay model with
supporting tools and built in DeepEval evaluation to answer questions about upcoming flights, including whether a
flight is predicted to be delayed, so passengers can plan with fewer surprises.


## Architecture Diagram

### Preprocessing

```mermaid
flowchart LR
  A[Flight Delay Data CSV] --> C[Load Data]
  B[Airport Data CSV] --> C
  C --> D[Filter and Clean Data]
  D --> E[Enrich with Meteostat Weather Data]
  E --> F[Feature Engineering]
  F --> G[Save Processed Data]
```

### Training

```mermaid
flowchart LR
  A[Processed Dataset] --> B[Time-Based Train/Test Split]
  B --> C[Train CatBoost Model]
  C --> D[Evaluate Metrics]
  D --> E[Save Model Artifacts]
```

### Agent

```mermaid
flowchart LR
  A[User Input] --> B[Agent Node: LLM with Tools]
  B --> C{Tool Calls?}
  C -- Yes --> D[Tools Node]
  D --> B
  C -- No --> E[Agent Response]
  E --> F[Evaluation Loop up to 3 retries]
  F -- Pass --> G[Return Response]
  F -- Fail --> B
```

**Tools Available** 
route confirmation, distance calculation, temporal feature, weather information, delay model inference.


Example Agent Flow:
- User asks: "Will my flight from Orlando to San Diego be delayed?"
- Agent asks for missing details (airline and time).
- User provides: "Delta, tomorrow afternoon."
- Agent calls tools to build features for inference.
- Agent calls the delay model inference tool to get a delay prediction.
- Agent generates a response.
- Evaluation checks relevance and bias; if it fails, the agent regenerates.

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
BASE_PATH=/YOURSYSTEMPATH/Aero-delay-prediction
OPENAI_API_KEY=your_key_here
VISUAL_CROSSING_KEY=your_key_here
```

## Data

1. Download flight delay and cancellation dataset:
   ```python
   download_data_path = kagglehub.dataset_download("patrickzel/flight-delay-and-cancellation-dataset-2019-2023")
   ```
2. Download airport metadata (`airports.dat`) from [OpenFlights](https://openflights.org/data) and save as CSV.
3. Save raw data under `data/` (for example, `data/raw/`), then keep derived caches under `data/cache/`.
4. Confirm the input paths in `ml_pipelines/params/preprocessing/preprocessing_params.py` match the filenames you saved (for example, `data/raw/flights_sample_3m.csv` and `data/raw/airports.csv`).

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
python -m agent.web_app
```

Then open `http://127.0.0.1:8000` in your browser.

## Linting and Checks (Local)

Run the same checks as CI:

```bash
bash scripts/ci_check.sh
```

## Limitations

### Current Model Performance:
    - The CatBoost classifier achieves ~0.67 accuracy with a 30-minute delay threshold.
  The model prioritises recall (≈0.61) to capture a majority of delayed flights, at the cost of higher false positives (precision ≈0.34), reflecting the asymmetric cost of missed delays for passengers.
    - This is a known limitation and classification performance will be improved upon in future iterations
- The prediction model is only trained on US flight data so is not applicable to other countries.
- The agent evaluation only uses Bias and Relevancy metrics, it does not consider other metrics such as faithfulness regarding tool calls.


## Next Steps


- Add more robust prompts to limit scope of responses
- improve evaluations to encompass faithfulness with respect to tool calls
- add unit tests with a focus on feature engineering logic
- improve prediction model by exploring feature and threshold tuning, dataset balancing and different model architectures

✍️ Author

Built by Harish Kumaravel for MLE portfolio, with emphasis on agentic workflows

🚀 License

MIT — feel free to fork and build upon.
