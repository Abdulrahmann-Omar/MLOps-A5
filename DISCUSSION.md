# Assignment 5 — Discussion Notes

**Abdulrahman Omar | MLOps | 3-min Presentation**

---

## 🎯 What's the Big Idea?

The assignment simulates a real MLOps gate: **"Don't deploy unless the model is good enough."**

Instead of one job doing everything, I split it into two independent GitHub runners:

- **`validate`** → train + log → pass the Run ID forward
- **`deploy`** → check quality gate → simulate Docker build

This mirrors how real teams work: training runs on one machine, deployment happens on another.

---

## 🔗 The Core Problem: How Do You Pass Data Between Jobs?

> "Two jobs = two machines. They don't share a filesystem."

The answer is **GitHub Artifacts**:

- Job 1 calls `actions/upload-artifact@v4` to upload `model_info.txt` + `mlruns/`
- Job 2 calls `actions/download-artifact@v4` to restore them

`model_info.txt` contains just one thing: the **MLflow Run ID** (a UUID). That's what links the two jobs.

---

## 🧠 Why Bundle `mlruns/` with the Artifact?

MLflow stores run data locally in `./mlruns/`. When `check_threshold.py` does `mlflow.get_run(run_id)`, it needs to query the **same MLflow backend** that `train.py` used.

Since both jobs use `file:./mlruns` (no external server), I have to carry the entire `mlruns/` folder over as part of the artifact. Otherwise job 2 has no idea what that Run ID refers to.

---

## ⚙️ How the Quality Gate Works

```python
# check_threshold.py — the key logic
run = mlflow.get_run(run_id)
accuracy = run.data.metrics.get("accuracy")
if accuracy < 0.85:
    sys.exit(1)   # ← this fails the GitHub Actions step
```

`sys.exit(1)` returns a non-zero exit code → GitHub Actions marks the step as failed → the whole `deploy` job fails → no Docker build runs.

---

## 🐳 The Dockerfile

```dockerfile
FROM python:3.10-slim
ARG RUN_ID          # build-time argument
ENV RUN_ID=${RUN_ID}
RUN echo "Fetching model for run ${RUN_ID}"
CMD ["python", "-c", "print('Container ready for run ' + '${RUN_ID}')"]
```

`ARG` vs `ENV`: `ARG` is only available at **build time**, `ENV` makes it available at **runtime** too. I use both so the value is accessible throughout the container lifecycle.

In a real scenario you'd do:

```bash
docker build --build-arg RUN_ID=$(cat model_info.txt) -t mymodel .
```

---

## 📊 Triggering Fail vs. Success

I added a `--label-noise` parameter to `train.py` that randomly flips training labels, degrading accuracy:

| `label_noise` | Accuracy | Pipeline |
|---|---|---|
| `0.0` | ~97% | ✅ Deploy runs |
| `0.9` | ~37% | ❌ Deploy halted |

The `workflow_dispatch` trigger lets me pass this as input, so I can produce both run scenarios on demand.

---

## ✅ Results

| Run | Noise | Accuracy | Outcome |
|---|---|---|---|
| Run #1 | 0.0 | 0.967 | Both jobs ✅ — Mock build executed |
| Run #2 | 0.9 | 0.367 | `validate` ✅ / `deploy` ❌ — Halted |

---

## 💬 One-liner Summary

> "I built a GitHub Actions pipeline that trains a model, logs it to MLflow, passes the Run ID between jobs using artifacts, checks the accuracy threshold, and only proceeds to Docker build if the model is good enough — fully automated, zero manual steps."
