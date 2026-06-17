# 🎬 CineSense

> Hybrid Anime Recommendation System powered by Semantic Retrieval and Graph-Based Reranking

![Version](https://img.shields.io/badge/version-v1.0.0--rc1-blue)
![Python](https://img.shields.io/badge/Python-3.11+-green)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688)
![React](https://img.shields.io/badge/React-TypeScript-61DAFB)
![License](https://img.shields.io/badge/license-MIT-orange)

---

# Overview

CineSense is a production-oriented anime recommendation platform that combines modern semantic search with collaborative graph-based reranking to generate high-quality recommendations.

The system first retrieves semantically similar anime using embedding-based retrieval and then applies a lightweight graph-reranking layer using collaborative watch behavior to improve recommendation quality without retraining embeddings.

Through extensive benchmarking and evaluation, CineSense achieved:

* **+72.8% improvement in NDCG@10**
* **+39.3% improvement in MRR**
* **+85.4% improvement in Precision@10**

compared to the semantic-only baseline.

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
* A/B testing support
* Rollback controls
* Runtime telemetry
* Configuration management

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
* Production monitoring
* Automated regression testing

---

# Architecture

```text
User Seeds
     │
     ▼
Sentence Transformer Embeddings
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

Validation scripts:

```bash
python evaluation/compare_rerank_methods.py
python evaluation/production_monitor.py
python evaluation/multi_seed_validation.py
```

---

# Tech Stack

## Backend

* Python
* FastAPI
* NumPy
* Pandas
* NetworkX
* Pydantic

## Machine Learning

* Sentence Transformers
* Cosine Similarity Retrieval
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
│   ├── routers/
│   ├── schemas/
│   └── main.py
│
├── cinesense/
│   ├── config/
│   ├── models/
│   ├── recommenders/
│   ├── retrieval/
│   ├── ranking/
│   ├── services/
│   ├── tests/
│   └── utils/
│
├── evaluation/
│   ├── benchmark.py
│   ├── compare_rerank_methods.py
│   ├── production_monitor.py
│   └── multi_seed_validation.py
│
├── frontend/
│   ├── src/
│   └── public/
│
└── README.md
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/Sarthak0205/Cine_Sense.git
cd Cine_Sense
```

## Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
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

# API Endpoints

## Get Recommendations

```http
POST /recommend
```

Example Request:

```json
{
  "anime_ids": [1535, 5114],
  "mode": "discover",
  "top_k": 10
}
```

---

## Search Anime

```http
GET /anime/search?q=death
```

---

## Anime Details

```http
GET /anime/{anime_id}
```

---

# Quality Assurance

## Unit Tests

```bash
python -m unittest discover cinesense/tests
```

## Integration Tests

```bash
python -m unittest discover api/tests
```

---

# Production Features

* Graph asset validation
* Runtime telemetry
* Health monitoring
* Safe rollback mechanism
* A/B testing support
* Configuration validation
* Graceful degradation
* Benchmark regression detection

---

# Release Information

Current Release:

```text
v1.0.0-rc1
```

Status:

```text
Release Candidate
```

---

# Future Roadmap

## RC2

* User accounts
* Favorites
* Recommendation history
* Recommendation feedback collection
* Enhanced explainability

## RC3

* Anime + Movies unified catalog
* Personalized recommendations
* User embeddings
* Hybrid recommendation engine

---

# Screenshots

Add screenshots here:

```text
docs/homepage.png
docs/search.png
docs/recommendations.png
```

---

# Author

**Sarthak Deshmukh**

* GitHub: https://github.com/Sarthak0205
* Project: CineSense

---

# License

MIT License
Copyright (c) 2026 Sarthak Chaudhari.
