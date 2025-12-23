# Aero Delay Prediction Project Report

## Overview

Aero Delay Prediction is a minimal machine learning pipeline developed to predict flight delays using historical flight and weather data. The project currently covers data acquisition, preprocessing, model training, and initial evaluation.

## Progress Summary

### Data Acquisition

- **Flight Delay Data:** Downloaded from Kaggle using `kagglehub.dataset_download("patrickzel/flight-delay-and-cancellation-dataset-2019-2023")`.
- **Airport Data:** Sourced from OpenFlight.

### Preprocessing

- Implemented a preprocessing pipeline (`python -m ml_pipelines.preprocessing.run_preprocessing`), configurable with cache options.
- Ensured column names are standardized to lowercase.
- Considering dropping rows with missing weather data for improved model reliability.

### Model Training

- Performed grid search for hyperparameter optimization:
    - Parameters tuned include `n_estimators`, `learning_rate`, `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`, `reg_lambda`, `reg_alpha`, and `gamma`.
    - Used classification metrics to evaluate model performance under different delay and wind speed thresholds.

## Training Results

### Grid Search Parameter Distribution

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

### Classification Report

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.867     | 0.682  | 0.763    | 220,983 |
| 1     | 0.340     | 0.609  | 0.436    | 59,353  |

- **Accuracy:** 0.667 (280,336 samples)
- **Macro avg:** Precision 0.603, Recall 0.646, F1-score 0.600
- **Weighted avg:** Precision 0.755, Recall 0.667, F1-score 0.694
- Delay threshold: 15 minutes, wind threshold: 25
- Updated setting: 30-minute delay, windspeed 30, data from 2022-01-01 onwards

### Evaluation

- **Initial Results (Delay threshold: 15, Wind threshold: 25):**
    - Accuracy: 0.667
    - Decent recall, capturing many delays but with notable false positives.
- **Updated Evaluation (Delay threshold: 30, Wind speed: 30, data from 2022 onward):**
    - Working on cautious metrics and further threshold tuning.

## Next Steps

- Add unit tests for reliability.
- Fix and improve CI pipeline.
- Enhance logging and print statements for better traceability.
- Refine training process and add simple deployment.
- Develop an agent for automation.

## Notes

- The project is focused on reproducibility and extensibility.
- Contributions and suggestions are welcome.

## License

MIT License.
