# Aero Delay Prediction

A minimal machine learning pipeline to predict flight delays.

## Quickstart

1. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2. **Download datasets:**
    - Flight delay and cancellation data:
      ```python
      download_data_path = kagglehub.dataset_download("patrickzel/flight-delay-and-cancellation-dataset-2019-2023")
      ```
    - Airport data from OpenFlight.

3. **Run preprocessing:**
    From the project root, execute:
    ```bash
    python -m ml_pipelines.run_preprocessing
    ```
    - You can select the cache directory (`True` or `False`).

4. **Run training:**
    ```bash
    python -m ml_pipelines.run_training
    ```

5. **Run full training pipeline:**
    ```bash
    python -m ml_pipelines.full_training_pipeline
    ```



Fix any reported issues before committing your code.
## Development Tasks

- Add unit tests.
- Fix CI pipeline.
- Add more print statements for debugging.
- Ensure all column names are lowercase.
- Proceed to model training.

## Model Training

### Grid Search Example

```python
param_dist = {
     "n_estimators": [600, 900, 1200, 1800, 2400, 3000],
     "learning_rate": np.logspace(np.log10(0.015), np.log10(0.12), 8),
     "max_depth": [4, 5, 6, 7, 8],
     "min_child_weight": [1.0, 2.0, 3.0, 5.0, 8.0],
     "subsample": [0.65, 0.75, 0.85, 0.95],
     "colsample_bytree": [0.65, 0.75, 0.85, 0.95],
     "reg_lambda": [0.5, 1.0, 2.0, 3.0, 5.0, 8.0],
     "reg_alpha": [0.0, 0.05, 0.1, 0.25, 0.5],
     "gamma": [0.0, 0.2, 0.5, 1.0],
     # keep scale_pos_weight fixed from y_train
}
```

## Classification Report

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.867     | 0.682  | 0.763    | 220,983 |
| 1     | 0.340     | 0.609  | 0.436    | 59,353  |

- **Accuracy:** 0.667 (280,336 samples)
- **Macro avg:** Precision 0.603, Recall 0.646, F1-score 0.600
- **Weighted avg:** Precision 0.755, Recall 0.667, F1-score 0.694

- Delay threshold: 15 minutes, wind threshold: 25
- Now using threshold: 30 minutes, windspeed: 30, data from 2022-01-01 onwards

## Testing

- Test end-to-end pipeline.
- Ensure unit tests are sufficient.

## Next Steps

python -m agent.customer_agent run from root


move into agent core the core functionalities