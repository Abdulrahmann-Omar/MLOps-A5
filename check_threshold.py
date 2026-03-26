import argparse
import os
import sys
import mlflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate MLflow run accuracy against a threshold.")
    parser.add_argument(
        "--model-info-path",
        default="model_info.txt",
        help="Path to file containing the run id written during training.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Minimum accuracy required to deploy.",
    )
    return parser.parse_args()


def read_run_id(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def main() -> None:
    args = parse_args()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or "file:./mlruns"
    mlflow.set_tracking_uri(tracking_uri)

    run_id = read_run_id(args.model_info_path)
    if not run_id:
        print("No run id found in model_info.txt", file=sys.stderr)
        sys.exit(1)

    run = mlflow.get_run(run_id)
    accuracy = run.data.metrics.get("accuracy")
    if accuracy is None:
        print("No accuracy metric logged for run; failing deployment", file=sys.stderr)
        sys.exit(1)

    print(f"Run ID: {run_id}")
    print(f"Accuracy: {accuracy}")
    print(f"Threshold: {args.threshold}")

    if accuracy < args.threshold:
        print("Accuracy below threshold. Deployment halted.", file=sys.stderr)
        sys.exit(1)

    print("Accuracy meets threshold. Proceeding to deployment.")


if __name__ == "__main__":
    main()