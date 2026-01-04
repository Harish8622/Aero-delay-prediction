import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class TrainingParams:
    def __init__(self):
        # need to define all args from catboost.py here
        base_path = os.getenv("BASE_PATH")
        if not base_path:
            raise ValueError("BASE_PATH not set in .env")
        self.data = os.path.join(base_path, "data/processed/preprocessed_data.csv")
        self.target = "DELAY_FLAG_30"
        self.model_dir = os.path.join(base_path, "models")
        self.seed = 42
        self.iterations = 800
        self.learning_rate = 0.06
        self.depth = 8
        self.l2_leaf_reg = 3.0

        # Ensure required directories exist
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.data), exist_ok=True)
