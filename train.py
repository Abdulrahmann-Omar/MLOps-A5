import argparse
import os
import numpy as np
import mlflow
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple classifier and log to MLflow.")
    parser.add_argument(
        "--model-info-path",
        default="model_info.txt",
        help="Path to write the MLflow run id for downstream jobs.",
    )
    parser.add_argument(
        "--label-noise",
        type=float,
        default=0.0,
        help="Probability of flipping a training label to create a low-accuracy run when needed.",
    )
    return parser.parse_args()


def maybe_add_noise(y: np.ndarray, noise: float) -> np.ndarray:
    if noise <= 0:
        return y
    rng = np.random.default_rng(42)
    mask = rng.random(len(y)) < noise
    noisy_labels = y.copy()
    noisy_labels[mask] = rng.permutation(y)[mask]
    return noisy_labels


def main() -> None:
    args = parse_args()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or "file:./mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("assignment-5")

    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )
    y_train_noisy = maybe_add_noise(y_train, args.label_noise)

    with mlflow.start_run(run_name="iris-logreg") as run:
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train_noisy)

        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)

        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("max_iter", 200)
        mlflow.log_param("label_noise", args.label_noise)
        mlflow.log_metric("accuracy", float(accuracy))

        run_id = run.info.run_id
        with open(args.model_info_path, "w", encoding="utf-8") as f:
            f.write(run_id)

        print(f"Run ID: {run_id}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Tracking URI: {tracking_uri}")


if __name__ == "__main__":
    main()