# JADD — Jacobian-Aware Drift Distance

A drift-monitoring module for object detectors deployed in digital twins. It measures distribution shift in the **prediction space of the detector head** rather than in raw feature space, so it only fires on changes that actually affect the deployed model and stays silent on visual changes the detector tolerates.

The metric, **JADD (Jacobian-Aware Drift Distance)**, is label-free, closed-form, and reuses the features the detector already computes. A classical detector-agnostic aggregate (PSI / KL / JS / KS / Wasserstein) is provided alongside as a baseline. Control is exposed through a FastAPI REST API, with monitoring via Prometheus and Grafana.

> JADD is developed at the CTLab, ITMO University.

## Features

- **Prediction-space drift.** Feature-distribution shift is projected through the detector-head Jacobian, weighting every feature direction by its effect on class logits and box coordinates.
- **Label-free and closed-form.** No ground-truth labels, auxiliary models, or stochastic forward passes — only the first two moments of the feature distribution and the head Jacobian.
- **Mean + shape.** Captures both the mean shift of the feature cloud and changes in distribution shape (covariance), so it detects new scene modes and emerging object classes that leave the mean nearly unchanged.
- **Photometric sensitivity.** Reacts to exposure, noise, and contrast changes that alter head-relevant feature statistics — exactly the corruptions that degrade detector quality.
- **Closed-loop maintenance.** Drift frames are exported for annotation; retraining is triggered once a sufficient volume of drifted data has accumulated.
- Processes video files, ZIP archives of frames, and RTSP streams.
- Object detection with a pretrained YOLO11l, filtered by COCO classes; optional precise segmentation via SAM.
- Wall-clock schedule with timezone (weekdays, time windows, months).
- Metrics in Prometheus with a ready-made Grafana dashboard; one-command launch via Docker Compose.

## Pipeline Architecture

```
Video / archive / RTSP
        │
        ▼
┌──────────────────────────────────────────────┐
│  ObjectDetector  (YOLO11l, class filter)     │
└──────────────────────────────────────────────┘
        │  detections + bbox
        ▼
┌──────────────────────────────────────────────┐
│  ObjectSegmenter (SAM, optional)            │
│  → object crops                              │
└──────────────────────────────────────────────┘
        │  per-object features f (backbone, pre-head)
        ▼
┌──────────────────────────────┬──────────────────────────────┐
│  JADDCalculator (JADD)       │  DriftAnalyzer (baseline)     │
│  project through head Jac.   │  raw-feature-space statistics │
│  → prediction-space drift    │  → detector-agnostic aggregate│
└──────────────────────────────┴──────────────────────────────┘
        │
        ▼
   threshold alert  →  frame export (CVAT)  →  retraining  →  Prometheus/Grafana
```

The main working route is the `JADDCalculator` (`jadd.py`), which unifies detection and drift computation into a single job (`jadd_job.py`) for `video` / `archive` / `rtsp` sources.

## The JADD Metric

Let \(f\) be the internal feature vector of a detected object, produced by the detector backbone immediately before the prediction head. Let \(P\) be the reference (baseline) distribution of \(f\) and \(Q\) the distribution in the current sliding window, with means \(\mu_P, \mu_Q\) and covariances \(\Sigma_P, \Sigma_Q\). Let \(g\) be the detector head mapping features to class logits and box coordinates, and

\[
J = \left.\frac{\partial g}{\partial f}\right|_{f=\mu_P}
\]

its Jacobian evaluated at the reference mean — a matrix encoding how sensitive each head output is to perturbations in each feature dimension. JADD is defined as

\[
\mathrm{JADD}^{\,2}(P,Q) = \bigl\|\, J(\mu_Q - \mu_P)\,\bigr\|_2^{\,2} + \lambda\; d_B^{\,2}(C_P, C_Q),
\]

where \(C_P = J\,\Sigma_P\,J^\top\), \(C_Q = J\,\Sigma_Q\,J^\top\) are the feature covariances pushed forward through the head, and \(d_B\) is the Bures distance between two positive-semidefinite matrices:

\[
d_B^{\,2}(C_P, C_Q) = \operatorname{tr}\!\bigl(C_P + C_Q - 2\,(C_P^{1/2}\, C_Q\, C_P^{1/2})^{1/2}\bigr).
\]

The reported value is \(\sqrt{\mathrm{JADD}^2}\).

### Intuition

The Jacobian \(J\) acts as a **sensitivity lens**. Feature-space shifts lying in directions to which the head is insensitive — the null space or small-singular-value subspace of \(J\) — are projected to near zero in prediction space and do not trigger the metric. Shifts along prediction-relevant directions are amplified.

- The **first term** tracks how far the feature mean, as seen by the head, has moved.
- The **second term** (weighted by \(\lambda \ge 0\)) tracks changes in the shape of the distribution (variance growth, emerging scene modes) that leave the mean nearly unchanged.
- \(\lambda = 0\) reduces the metric to the mean-shift-only ablation \(\mathrm{JADD} = \|J(\mu_Q - \mu_P)\|_2\).
- Under the local Gaussian approximation, \(\lambda = 1\) recovers the squared Wasserstein-2 distance between the two feature distributions in prediction space.

### Sensitivity to photometric distortions

JADD is by construction **not invariant** to photometric changes that alter head-relevant feature statistics. Additive zero-mean noise leaves the mean almost unchanged but makes the Bures term strictly positive; a global contrast gain moves features along the most sensitive direction of the head; over/under-exposure moves features along prediction-relevant directions. Since these corruptions also degrade detector quality, this response is, at least in part, the desired behavior: JADD signals prediction-relevant drift, not a semantic detector of "dangerous" changes.

## Baseline Aggregate (detector-agnostic)

As a detector-agnostic baseline, the module also computes classical drift statistics between the same reference \(P\) and current window \(Q\), but in **raw feature space**: Population Stability Index (PSI), KL divergence, Jensen–Shannon (JS) divergence, Kolmogorov–Smirnov (KS) statistic, and Wasserstein distance. They are combined into a single normalized aggregate with weights:

| Metric | Weight |
|---|---|
| `psi_mean` | 0.25 |
| `kl_mean` | 0.25 |
| `js_divergence` | 0.20 |
| `ks_statistic` | 0.15 |
| `wasserstein_distance` | 0.15 |

The best variant (`v5`) rescales the aggregate by the mean detector confidence per class. The baseline consumes exactly the same per-object features, sliding window, and reference set as JADD; the two monitors differ only in the space in which the divergence is measured. Unlike JADD, the baseline does not project through the detector-head Jacobian and is therefore susceptible to false alarms from benign visual changes that do not affect detector predictions.

A **Page-Hinkley** test is additionally maintained for online change detection.

## Repository Structure

```
.
├── README.md
├── .gitignore
├── drift_detection/                # FastAPI application and modules
│   ├── api.py                     #  REST API, Prometheus metrics, jobs
│   ├── drift_detector.py          #  ObjectDriftDetector (YOLO + SAM + analyzer)
│   ├── drift_analyzer.py          #  baseline aggregate: PSI/KL/JS/KS/Wasserstein, Page-Hinkley
│   ├── object_detector.py         #  YOLO11
│   ├── object_segmentation.py     #  SAM (Segment Anything)
│   ├── jadd.py                    #  JADDCalculator — JADD metric (head-Jacobian projection)
│   ├── jadd_job.py                #  unified job for video/archive/rtsp
│   ├── drift_frames_export.py     #  drift frame export (CVAT, pending queue)
│   ├── drift_schedule.py          #  wall-clock schedule with timezone
│   ├── cvat_loader.py             #  CVAT archive loading, YOLO annotation parsing
│   ├── model_trainer.py           #  YOLO training / retraining on baseline
│   ├── metrics.py                 #  optional standalone metrics module
│   ├── Dockerfile
│   ├── .dockerignore
│   ├── docker-compose.yml         #  api + prometheus + grafana
│   └── requirements.txt
└── monitoring/
    ├── prometheus/prometheus.yml
    └── grafana/
        ├── dashboards/object_drift_dashboard.json
        └── provisioning/{datasources,dashboards}/
```

## Quick Start (Docker Compose)

Brings up the API (port 8000), Prometheus (9090), and Grafana (3000):

```bash
cd drift_detection
docker compose up -d --build
```

- API: http://localhost:8000/docs — interactive Swagger documentation.
- Grafana: http://localhost:3000 (default credentials `admin` / `admin`; the `Object Drift` dashboard is imported automatically).

> For a reverse proxy with a path prefix, set `ROOT_PATH=/drift-api` (see the root `docker-compose.yml` of the full stack with auto-labeling, single entry point `:11502`).

## Local Development

```bash
cd drift_detection
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

The `yolo11l.pt` and `sam_b.pt` model weights are downloaded automatically on first run.

## Main API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/process_video_pretrained_jadd` | Run drift detection (video / archive / rtsp) |
| `GET` | `/video_jobs/{job_id}` | Job status |
| `POST` | `/video_jobs/{job_id}/stop` | Stop a job |
| `GET` | `/drift_frames/pending` | Fetch drift frames for annotation |
| `POST` | `/upload_baseline` | Upload baseline images |
| `POST` | `/train_model` | Train YOLO on a baseline (CVAT archive) |
| `POST` | `/process_video_pretrained` | Process video with a pretrained model |
| `POST` | `/process_archive_pretrained` | Process a ZIP archive of frames |
| `GET` | `/video_jobs/{job_id}/download` | Download a result |
| `GET` | `/metrics` | Prometheus metrics |

### The `schedule` parameter

Time format is `HH:MM` (24h), e.g. `09:00`, `15:30`. Overnight ranges are allowed: `{"start":"22:00","end":"06:00"}`. Weekdays: `0=Mon … 6=Sun` (or `mon`/`пн`). Default timezone: `Europe/Moscow`.

```json
{
  "timezone": "Europe/Moscow",
  "enabled": true,
  "rules": [
    {
      "weekdays": [0, 1, 2, 3, 4],
      "time_ranges": [{"start": "09:00", "end": "18:00"}]
    }
  ]
}
```

Rules are combined with OR; fields within a rule with AND. No `schedule` / `enabled != false` / empty `rules` ⇒ detection is always active.

### The `since` parameter

Seconds since `1970-01-01 UTC` (unix time), e.g. `1711929600` (= 2024-04-01 00:00:00 UTC).

## Monitoring

Metrics are exported to Prometheus (`/metrics`):

- `object_drift_detections_total` — drift detection counter;
- `object_drift_jadd` — the JADD metric;
- `object_drift_psi`, `object_drift_kl_divergence`, `object_drift_js_divergence`;
- `object_drift_ks_statistic`, `object_drift_ks_pvalue`;
- `object_drift_wasserstein` — Wasserstein distance over brightness.

A ready-made Grafana dashboard ships at `monitoring/grafana/dashboards/object_drift_dashboard.json` and is wired in via provisioning.

## Tech Stack

- Python 3.12, FastAPI, Uvicorn
- PyTorch, TorchVision (ResNet50 for features)
- Ultralytics YOLO11, Segment Anything (SAM)
- OpenCV, NumPy, SciPy
- Prometheus Client, Grafana, Docker Compose

## License

Developed at [CTLab, ITMO](https://github.com/CTLab-ITMO). See the repository maintainers for details.
