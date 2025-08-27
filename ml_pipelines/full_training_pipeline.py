# full training pipeline, runs preprocessing then training
import subprocess, sys
from ml_pipelines.run_preprocessing import main as run_preprocessing
from ml_pipelines.run_training import main as run_training 

def main()::
    run_preprocessing()
    run_training()  