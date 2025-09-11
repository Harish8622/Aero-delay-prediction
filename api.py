from fastapi import FastAPI
from ml_pipelines.full_training_pipeline import main as run_full_pipeline

app = FastAPI()

@app.post("/run_pipeline/")
def run_pipeline():
    run_full_pipeline()
    return {"status": "Pipeline completed"}
