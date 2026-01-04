# ml_pipelines/preprocessing/run_preprocessing.py
import argparse
import subprocess
import sys
from ml_pipelines.params.preprocessing.preprocessing_params import PreprocessingParams


def main():
    p = PreprocessingParams()

    parser = argparse.ArgumentParser(description="Run preprocessing")
    parser.add_argument("--input_flights_table", default=p.input_flights_table)
    parser.add_argument("--input_airport_table", default=p.input_airport_table)
    parser.add_argument("--output_path", "--out", default=p.output_path)
    parser.add_argument("--cache_dir", default=p.cache_dir)
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "-m",
        "src.preprocessing",  # run as module so 'from src...' works
        "--input_flights_table",
        args.input_flights_table,
        "--input_airport_table",
        args.input_airport_table,
        "--output_path",
        args.output_path,
        "--cache_dir",
        args.cache_dir,
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
