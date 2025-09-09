# Aero Delay Prediction
Minimal ML pipeline to predict flight delays.

## Quickstart
pip install -r requirements.txt

to run pre processing

run

download data
path = kagglehub.dataset_download("patrickzel/flight-delay-and-cancellation-dataset-2019-2023")




and airport data

from openflight

python -m ml_pipelines.preprocessing.run_preprocessing 
run from root

cache dir can be selected true or false

add unit tests next

fix ci pipeline

add more print statements

make sure all columns are .lower



then move on to training


grid search 1:

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

Classification report:
               precision    recall  f1-score   support

           0      0.867     0.682     0.763    220983
           1      0.340     0.609     0.436     59353

    accuracy                          0.667    280336
   macro avg      0.603     0.646     0.600    280336
weighted avg      0.755     0.667     0.694    280336
the delay threshold was 15 and wind threshold was 25


now using threshold 30 with windspseed 30 and data from 01-01-2022 onwardss

add a cautious metric

maybe can drop rows where weather is na

decent recall so it is capturing a lot of the delays but it is getting a lot of false positives


to do 

ensure training is good

add deployment very simple

then create agent