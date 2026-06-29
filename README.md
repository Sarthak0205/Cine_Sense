# 🎬 CineSense

> Hybrid Anime Recommendation System powered by Semantic Retrieval and Graph-Based Reranking

![Version](https://img.shields.io/badge/version-v1.0.0-blue)
![Python](https://img.shields.io/badge/Python-3.13.5-green)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688)
![React](https://img.shields.io/badge/React-TypeScript-61DAFB)
![License](https://img.shields.io/badge/license-MIT-orange)

---

# Overview

CineSense is a production-oriented anime recommendation platform that combines semantic retrieval with graph-based reranking to generate high-quality recommendations.

The system first retrieves semantically similar anime using embedding-based retrieval and then applies a lightweight collaborative graph-reranking layer to improve recommendation quality without retraining embeddings.

Through extensive benchmarking and evaluation, CineSense achieved:

* **+72.8% improvement in NDCG@10**
* **+39.3% improvement in MRR**
* **+85.4% improvement in Precision@10**

compared to the semantic-only baseline.

---

# Live Demo

## Frontend

https://cine-sense-g2a7.vercel.app

## Backend API

https://cine-sense-94ry.onrender.com

## Swagger Documentation

https://cine-sense-94ry.onrender.com/docs

---

# Key Features

## Recommendation Engine

* Semantic similarity retrieval
* Graph-based reranking
* Multi-seed recommendations
* Franchise-aware filtering
* Duplicate prevention
* Diversity preservation
* Explainable recommendations

## Backend

* FastAPI REST API
* Production-ready architecture
* Health monitoring
* Runtime telemetry
* Configuration management
* Graceful degradation
* Safe deployment validation

## Frontend

* React + TypeScript UI
* Anime search
* Multi-seed selection
* Recommendation visualization
* Recommendation detail modal
* Responsive design

## Evaluation Framework

* Gold-standard benchmark evaluation
* NDCG@10
* MRR
* Precision@10
* Multi-seed validation
* Automated regression testing

---

# Architecture

```text
User Seeds
     │
     ▼
Semantic Retrieval
(Top 100 Candidates)
     │
     ▼
Graph-Based Reranking
(Jaccard + Distance)
     │
     ▼
Franchise Filtering
     │
     ▼
Diversity Enforcement
     │
     ▼
Final Recommendations
```

---

# Research Datasets

The original research datasets (`archive/` and `archive-2/`) are intentionally excluded from version control due to their size.

Training, evaluation, and audit scripts that depend on these datasets require local copies to be placed in the expected directories.

The production recommendation service does **not** require these datasets and runs entirely from the exported model artifacts located in:

```text
cinesense/models/twostage_v1/
```

This allows the deployed application to remain lightweight while preserving reproducibility for local research workflows.


# Recommendation Pipeline

## Stage 1 — Semantic Retrieval

The system retrieves candidate anime using semantic embeddings generated from metadata and content information.

Primary signals:

* Synopsis similarity
* Genre similarity
* Theme similarity
* Content embeddings

Output:

```text
Top 100 semantic candidates
```

---

## Stage 2 — Graph-Based Reranking

Candidates are reranked using collaborative watch patterns extracted from user viewing behavior.

Signals:

### Jaccard Similarity

Measures overlap between user watch communities.

### Graph Distance

Captures collaborative proximity in the user-watch graph.

### Semantic Score

Preserves semantic relevance.

Final reranking formula:

```text
Final Score =
Semantic Score
+ (Jaccard Weight × Jaccard Similarity)
+ (Distance Weight × Distance Signal)
```

---

# Benchmark Results

| Model             | NDCG@10 | MRR    | Precision@10 |
| ----------------- | ------- | ------ | ------------ |
| Semantic Baseline | 0.1301  | 0.3905 | 9.71%        |
| Graph Reranking   | 0.2249  | 0.5438 | 18.00%       |

---

# Performance Improvements

| Metric       | Improvement |
| ------------ | ----------- |
| NDCG@10      | +72.8%      |
| MRR          | +39.3%      |
| Precision@10 | +85.4%      |

---

# Evaluation Methodology

Evaluation was performed using a manually curated gold-standard benchmark dataset.

Metrics:

* NDCG@10
* Mean Reciprocal Rank (MRR)
* Precision@10
* Discovery Rate
* Franchise Diversity

Example validation commands:

```bash
python evaluation/holdout_benchmark_eval.py
python evaluation/diversity_audit.py
python evaluation/stability_audit.py
```

---

# Tech Stack

## Backend

* Python
* FastAPI
* NumPy
* Pandas
* PyArrow
* Pydantic

## Machine Learning

* Sentence Transformers
* Semantic Retrieval
* Graph-Based Reranking

## Frontend

* React
* TypeScript
* Vite

## Evaluation

* NDCG
* MRR
* Precision@K
* Offline Benchmark Framework

---

 # Project Structure

```text
CineSense
│
├── api/
│   ├── tests/
│   └── main.py
│
├── cinesense/
│   ├── config/
│   ├── models/
│   ├── ranking/
│   ├── recommenders/
│   ├── retrieval/
│   ├── services/
│   ├── tests/
│   └── utils/
│
├── docs/
│   ├── audits/
│   ├── benchmark_summary.md
│   ├── release_notes.md
│   └── ui_specification.md
│
├── evaluation/
│
├── frontend/
│   ├── public/
│   └── src/
│
├── research/
│   └── notebooks/
│
├── requirements.txt
├── requirements-dev.txt
├── render.yaml
├── runtime.txt
└── README.md
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/Sarthak0205/Cine_Sense.git

cd Cine_Sense
```

---

## Create Virtual Environment

```bash
python -m venv venv

source venv/bin/activate
```

---

## Production Dependencies

```bash
pip install -r requirements.txt
```

---

## Development & Research Dependencies

Includes:

* Sentence Transformers
* Torch
* Scikit-learn
* Evaluation tooling
* Offline experiments
* Embedding training scripts

```bash
pip install -r requirements-dev.txt
```

---

# Running Backend

```bash
uvicorn api.main:app --reload
```

API:

```text
http://localhost:8000
```

Swagger Docs:

```text
http://localhost:8000/docs
```

---

# Running Frontend

```bash
cd frontend

npm install

npm run dev
```

Frontend:

```text
http://localhost:5173
```

---

# Production Deployment

## Backend (Render)

Python Version:

```text
3.13.5
```

Build Command:

```bash
pip install -r requirements.txt
```

Start Command:

```bash
uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

Environment Variables:

```text
CINESENSE_MODEL_DIR=cinesense/models/twostage_v1

ALLOWED_ORIGINS=https://cine-sense-g2a7.vercel.app
```

Production API:

```text
https://cine-sense-94ry.onrender.com
```

---

## Frontend (Vercel)

Framework:

```text
Vite
```

Root Directory:

```text
frontend
```

Build Command:

```bash
npm run build
```

Environment Variables:

```text
VITE_API_URL=https://cine-sense-94ry.onrender.com
```

Production Frontend:

```text
https://cine-sense-g2a7.vercel.app
```

---

# API Endpoints

## Get Recommendations

```http
POST /recommend
```

Example Request:

```json
{
  "anime_ids": [20],
  "ratings": {
    "20": 9
  },
  "top_k": 5
}
```

---

## Search Anime

```http
GET /anime/search?q=death
```

---

## Health Check

```bash
curl https://cine-sense-94ry.onrender.com/health
```

Example Response:

```json
{
  "status": "ok",
  "model_version": "twostage_v1",
  "graph_available": true
}
```

---

# Quality Assurance

## Unit Tests

```bash
python -m unittest discover -s cinesense/tests
```

---

## API Tests

```bash
python -m unittest discover -s api/tests
```

---

## Frontend Build Verification

```bash
npm run build --prefix frontend
```

---

# Production Features

* Runtime model validation
* Health monitoring
* Environment-based CORS configuration
* Graph asset validation
* Configuration validation
* Graceful degradation
* Safe deployment verification
* Benchmark regression detection

---

# Release Information

Current Release:

```text
v1.0.0
```

Status:

```text
Production Deployment Complete
```

---

# Future Roadmap

## v1.1

* User accounts
* Favorites
* Recommendation history
* Recommendation feedback
* Enhanced explanations

---

## v2.0

* Anime + Movies unified catalog
* Personalized recommendations
* User embeddings
* Hybrid recommendation engine

---

# Author

**Sarthak Chaudhari**

GitHub:

https://github.com/Sarthak0205

Project:

CineSense

---

# License

MIT License

Copyright (c) 2026 Sarthak Chaudhari
