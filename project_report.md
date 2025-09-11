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
