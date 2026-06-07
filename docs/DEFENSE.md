# Defense pack - Bloc 4 (Solutions d'IA)

Project: **GreenGuard Dandelion Classifier** (the vision and MLOps platform for an
autonomous chemical-free weeding robot). This is your Q&A preparation for the oral
(5 min talk + 10 min Q&A). It is written in English to match the repository
documentation; if your oral is in French, ask and these can be translated.

The jury will probe the highest-weight criteria first (ML model 25%, CI/CD 20%), so
those sections are deepest. The most likely sharp question is "your F1 is 1.0, isn't
this too easy?" - that answer is prepared below and you should be ready to give it
confidently.

---

## 30-second opening pitch

"GreenGuard is the vision and MLOps platform behind an autonomous robot that removes
weeds mechanically instead of spraying herbicide. The model classifies a plant
image as dandelion (weed) or grass (crop), but the real deliverable is the closed
loop around it: a data pipeline, experiment tracking with MLflow, a FastAPI serving
endpoint, CI/CD, production monitoring with data-drift detection, and retraining that
is triggered automatically when drift crosses a threshold. The driver is real:
herbicide bans and a farm labour shortage mean the robot has to decide reliably, in
the field, forever. My focus was building that reliability backbone, not just a
model."

---

## The model (highest weight, prepare this hardest)

**Q: Your validation F1 is 1.0. Isn't this task trivial?**
A: On this dataset, yes, and I am deliberate about that. Dandelion versus grass on a
small clean dataset is an easy benchmark, and I chose it precisely so I could build
the full MLOps loop end to end rather than spend the whole bloc fighting the model.
The honest framing is in the model card: 1.0 reflects an easy, clean dataset, not
field conditions. In the real robot the same two classes get hard fast: low light,
motion blur, occlusion, growth stage, wet leaves, and species that look like grass.
That gap between benchmark and field is the entire reason the project is built around
drift detection and automated retraining. The model is the easy part; keeping it
correct in production is the engineering.

**Q: Why ResNet18 and transfer learning rather than training from scratch?**
A: With a small dataset, transfer learning from ImageNet gives strong features for
free and avoids overfitting; training a deep net from scratch would need far more
data. I kept a small CNN as a baseline in the model factory to show the comparison,
and the chosen architecture is saved in the checkpoint so any served model is
reproducible. The selection rationale is in `notebooks/02_model_selection`.

**Q: How do you evaluate the model beyond accuracy?**
A: Precision, recall, and F1, plus a confusion matrix, written to
`metrics_summary.json` and logged with the run. For a weeding robot the asymmetry
matters: calling a crop a weed (false positive) means destroying a plant, while
missing a weed is recoverable on the next pass. So per-class recall and the
confusion matrix matter more than a single accuracy number, and that is what I track.

**Q: How is training reproducible?**
A: A fixed random seed, pinned dependencies, the architecture stored in the
checkpoint, and every run tracked in MLflow (parameters, metrics, artifacts). Given a
run you can recover exactly what produced a model.

## CI/CD (second highest weight)

**Q: Describe your CI/CD pipeline.**
A: On every push, GitHub Actions runs the test suite with coverage, then builds the
Docker image and pushes it to GHCR, then validates the Kubernetes manifests with
kubeconform. It is the path from commit to a deployable, validated image.

**Q: Why validate manifests instead of doing a live deploy in CI?**
A: I originally deployed to Minikube in the runner, but that rollout was flaky on
hosted runners for months, which is worse than no signal because it trains you to
ignore red. I replaced it with offline manifest validation, which is fast and
deterministic, and I deploy to Minikube locally for the demo. Reliable CI you trust
beats an impressive step that is red half the time.

**Q: How is the model delivered to the API?**
A: The image is built and pushed by CI. The model artifact lives in object storage
(MinIO/S3); the API loads it by path at startup and exposes its version on
`/version`, so the running model is always identifiable.

## Serving API

**Q: Walk me through the serving path.**
A: A FastAPI service with `/predict` (image in, class plus confidence out),
`/health`, `/version`, and `/metrics` for Prometheus. There is also a Streamlit UI
for interactive use during the demo. The API records prediction counts and latency
as metrics so serving is observable.

**Q: How would this run on the actual robot?**
A: The robot needs on-device, low-latency inference, so the production step is to
export the model (ONNX or TorchScript), quantise it, and run it on the robot's edge
accelerator, with the cloud platform handling training, drift, and retraining. The
current FastAPI service is the cloud and development serving surface; the edge export
is the next increment.

## Retraining and monitoring

**Q: When and how does the model retrain?**
A: Retraining is drift-gated, not on a fixed timer. The monitoring side computes a
Population Stability Index on incoming data; when drift crosses the threshold, the
Airflow retrain DAG triggers a new training run. Retraining only when the data has
actually shifted saves compute and avoids needless model churn.

**Q: What exactly do you monitor in production?**
A: Two levels. Operational: request rate, latency, and error counts in Prometheus
and Grafana. Model: data drift via PSI plus an Evidently report, surfaced as gauges
on `/metrics` and a dedicated Grafana drift dashboard. Drift is the early warning
that field conditions have moved away from the training distribution.

**Q: How do you know retraining actually improved the model?**
A: A retrained model is evaluated on the held-out set and its metrics are logged to
MLflow before it is promoted, so you compare candidate against current. The honest
next step is formal model-registry stage transitions (staging to production) gated on
those metrics.

## Business relevance

**Q: Why does this matter commercially?**
A: Regulations like France's zero-herbicide rules and a shrinking agricultural
workforce make mechanical, autonomous weeding valuable. The model's reliability maps
directly to business outcomes: fewer crops destroyed, fewer weeds missed, less manual
labour. The cahier des charges has a business-outcome to ML-metric to
monitoring-signal table that makes that chain explicit.

## Reflection

**Q: What was the hardest part?**
A: Not the model, the operational reality. Getting a corporate proxy to let the
container download pretrained weights, getting the full eleven-service stack to run
end to end, and making CI trustworthy were where the real time went. That mirrors
real MLOps: the model is a small part of the system.

**Q: What would you do differently or next?**
A: Use a larger, field-realistic dataset (the current 1.0 is not representative),
add more classes for real crops, export and quantise the model for the edge, add an
active-learning loop that feeds the robot's mispredictions back into training, and
add model-registry promotion gates. The loop is built; these make it field-grade.

**Q: If the model is so simple, what is the actual contribution?**
A: The contribution is the system, not the classifier. A model is easy to train and
hard to keep correct in production. What I built is the part that keeps it correct:
versioned data and experiments, automated and validated delivery, drift detection,
and retraining that fires on real change. That is what an AI solution is, as opposed
to a notebook.
