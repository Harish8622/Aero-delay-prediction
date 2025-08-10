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