# ml_pipelines/preprocessing/run_traininf=g.py
import subprocess, sys, argparse
from ml_pipelines.params.training_params.training_params import TrainingParams


def main():
    p = TrainingParams()

    parser = argparse.ArgumentParser(description="Run training")
    parser.add_argument("--data", default=p.data)
    parser.add_argument("--target", default=p.target)
    parser.add_argument("--model_dir", default=p.model_dir)
    parser.add_argument("--seed", type=int, default=p.seed)
    parser.add_argument("--iterations", type=int, default=p.iterations)
    parser.add_argument("--learning_rate", type=float, default=p.learning_rate)
    parser.add_argument("--depth", type=int, default=p.depth)
    parser.add_argument("--l2_leaf_reg", type=float, default=p.l2_leaf_reg
    )

 
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "-m",
        "src.training.catboost",  # run as module so 'from ml_pipelines...' works
        "--data", args.data,
        "--target", args.target,
        "--model_dir", args.model_dir,
        "--seed", str(args.seed),
        "--iterations", str(args.iterations),
        "--learning_rate", str(args.learning_rate),
        "--depth", str(args.depth),
        "--l2_leaf_reg", str(args.l2_leaf_reg),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
