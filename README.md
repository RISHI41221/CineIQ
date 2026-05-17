# CineIQ

Local-first machine learning movie recommendation service built with FastAPI, hybrid ranking, sentiment-aware reranking, explainable recommendations, and Dockerized delivery.

## 🎬 Overview

Recommendation systems face a classic product challenge: deliver results that feel personal and relevant without trapping users in a narrow recommendation loop.

Content-only systems can become too metadata-driven. Collaborative systems can struggle with sparsity and cold-start cases. Matrix factorization models personalize well, but often hide their reasoning and require heavier infrastructure to serve reliably.

**CineIQ** solves this with a hybrid recommendation pipeline exposed through a production-grade FastAPI backend:

- **Content-based retrieval** uses TF-IDF over curated movie metadata and feature text.
- **Collaborative filtering** uses a precomputed Pearson correlation matrix derived from user ratings.
- **Matrix factorization** uses a trained `scikit-surprise` SVD model to personalize ranking for a specific user.
- **Sentiment-aware reranking** uses VADER on user tag text to boost or dampen candidate scores.
- **Explainable AI** adds a user-facing explanation so recommendations are not only accurate, but interpretable.

The result is a recommendation service that balances similarity, personalization, sentiment context, and transparency in one API.

## 📺 Demo Video

[Watch the CineIQ Demo on YouTube/Loom here](link)

## 🧠 The Hardware & MLOps Challenge

CineIQ originally targeted low-cost cloud deployment, but the production behavior of the ML stack made that tradeoff impractical. Loading a trained `scikit-surprise` SVD artifact alongside the TF-IDF vectorizer and associated recommendation assets creates a memory profile that exceeds the limits of many free-tier platforms.

In particular, free-tier instances capped at **512MB RAM** can trigger **Out-of-Memory (OOM) kills** during application startup or inference, especially when matrix factorization artifacts and vectorized text features are loaded concurrently. That makes the service unreliable, even when the code itself is correct.

Rather than force the project into an unstable hosting model, CineIQ adopts a **local-first Docker deployment strategy**. This decision keeps the environment reproducible, preserves the full recommendation quality of the hybrid stack, and ensures the service runs consistently on developer machines without cloud memory constraints becoming the bottleneck.

## ✨ Features

- **Hybrid Recommendation Engine** that blends TF-IDF content similarity, Pearson collaborative filtering, and SVD-based matrix factorization.
- **Sentiment Reranker** that applies VADER-based post-processing on user tag signals to boost or dampen recommendation scores.
- **Explainability Layer** that adds rule-based natural language explanations describing why each movie was recommended.
- **FastAPI Backend** with startup-time model loading and production-oriented API structure.
- **Local-First Docker Deployment** for stable, reproducible execution of memory-intensive ML assets.

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

- `main.py` - FastAPI application and lifecycle-managed model loading
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

## 🐳 How to Run

Build the Docker image:

```bash
docker build -t cineiq-api .
```

Run the container locally:

```bash
docker run -p 8000:8000 cineiq-api
```

Once the container is running, the API will be available at:

- `http://127.0.0.1:8000`
- Swagger UI: `http://127.0.0.1:8000/docs`

## 📈 Why This Project Matters

CineIQ is designed as more than a notebook experiment. It demonstrates how to turn a recommendation workflow into a reproducible ML service with:

- precomputed collaborative artifacts for faster, lighter startup
- lifecycle-managed model loading in FastAPI
- interpretable recommendation outputs
- Docker-based packaging for stable local execution

This makes it a strong applied ML portfolio project at the intersection of recommender systems, MLOps, and production API engineering.
