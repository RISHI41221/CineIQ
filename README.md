# CineIQ

Production-ready machine learning movie recommendation service built with FastAPI, hybrid ranking, sentiment-aware reranking, and explainable recommendations.

## 🎬 Overview

Modern recommendation systems have to solve a familiar product problem: how do you keep users engaged without trapping them in a narrow recommendation loop?

Pure content-based systems often overfit to metadata similarity. Pure collaborative systems can struggle with cold-start cases or sparse user behavior. Pure matrix factorization models personalize well, but can feel opaque and miss context from rich movie attributes.

**CineIQ** solves this by combining multiple recommendation signals into a single production-friendly web service:

- **Content-based ranking** uses TF-IDF over movie metadata and descriptive features to find semantically similar titles.
- **Collaborative filtering** uses a precomputed Pearson correlation matrix to capture patterns from user rating behavior.
- **Matrix factorization** uses a trained `scikit-surprise` SVD model to personalize results for a given user.
- **Sentiment reranking** adjusts recommendation strength based on user-generated movie tags with VADER sentiment analysis.
- **Explainable AI** generates human-readable reasons for each recommendation so the output is not just accurate, but interpretable.

The result is a hybrid recommendation API that balances relevance, personalization, sentiment context, and trust.

## ✨ Features

- **Hybrid Recommendation Engine** that blends TF-IDF content similarity, Pearson collaborative filtering, and SVD-based matrix factorization.
- **Sentiment Reranker** that applies VADER-based post-processing on user tag signals to boost or dampen recommendation scores.
- **Explainability Layer** that adds rule-based natural language explanations describing why each movie was recommended.
- **FastAPI Backend** with startup-time model loading and production-oriented API structure.
- **Dockerized Deployment** for consistent local development, demo environments, and cloud-ready packaging.

## 🧠 How It Works

1. A client sends a recommendation request with `userId`, `movie_title`, and `top_n`.
2. CineIQ generates candidate movies from both content-based and collaborative recommenders.
3. The SVD model predicts personalized preference strength for each candidate.
4. The hybrid engine combines the signals into a final ranking score.
5. The sentiment reranker adjusts scores using VADER sentiment on user tags.
6. The explainability module adds a human-readable justification for each result.
7. FastAPI returns the final recommendation list as JSON.

## 🛠️ Tech Stack

- **Python**
- **FastAPI**
- **Pandas**
- **Scikit-Learn**
- **Scikit-Surprise**
- **Docker**

## 📁 Project Highlights

- `main.py` - FastAPI application and API lifecycle management
- `cineiq_hybrid_backend.py` - hybrid recommendation engine
- `cineiq_sentiment_reranker.py` - VADER-based reranking logic
- `cineiq_explainability.py` - rule-based XAI module
- `build_pearson.py` - one-off script to precompute the Pearson matrix
- `cleaned_data/movies_master.csv` - master movie metadata
- `model_artifacts/svd_model.pkl` - trained SVD artifact
- `model_artifacts/pearson_matrix.pkl` - precomputed collaborative filtering matrix

## 🚀 API Snapshot

Primary endpoint:

```http
POST /api/v1/recommend
```

Example request body:

```json
{
  "userId": 1,
  "movie_title": "Toy Story",
  "top_n": 10
}
```

Example response fields:

- `title`
- `final_hybrid_score`
- `sentiment_adjusted_score`
- `explanation`

## 💻 How to Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the FastAPI development server:

```bash
uvicorn main:app --reload
```

Once the server is running, the API will be available at:

- `http://127.0.0.1:8000`
- Swagger UI: `http://127.0.0.1:8000/docs`

## 🐳 How to Run via Docker

Build the image:

```bash
docker build -t cineiq-api .
```

Run the container:

```bash
docker run -p 8000:8000 cineiq-api
```

Once the container is running, the API will be available at:

- `http://127.0.0.1:8000`
- Swagger UI: `http://127.0.0.1:8000/docs`

## 📈 Why This Project Matters

CineIQ is designed as more than a notebook experiment. It demonstrates how to take a machine learning recommendation workflow and package it as a deployable service with:

- precomputed collaborative artifacts for faster startup
- model lifecycle management in FastAPI
- interpretable recommendation outputs
- containerized deployment for reproducibility

This makes it a strong applied ML portfolio project at the intersection of recommender systems, MLOps, and production API engineering.
