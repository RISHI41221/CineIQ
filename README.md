# CineIQ: Hybrid Explainable Movie Recommendation Engine

An open, transparent, and highly personalized movie recommendation system built to break the "recommendation loop" of modern streaming platforms. 

---

## 🛑 Problem Statement
Content discovery on modern streaming platforms is opaque, biased toward promoted titles, and often traps users in recommendation loops. There is a critical need for an open, explainable movie recommendation engine that combines multiple Machine Learning strategies to deliver personalized, interpretable suggestions that evolve with user taste over time.

### Project Deliverables & Requirements
* **Hybrid Recommendation Engine:** Combines collaborative filtering (Pearson), content-based filtering (TF-IDF + cosine similarity), and matrix factorization (SVD) via a weighted ensemble.
* **Sentiment-Aware Re-Ranker:** Uses VADER sentiment analysis on user reviews to adjust rankings based on real audience reception.
* **Explainability Layer:** Every recommendation surfaces a dynamic, human-readable reason detailing exactly *why* the AI chose it (e.g., matching specific genres, collaborative taste patterns, or high sentiment).
* **User Interface:** A Streamlit dashboard to interact with the engine.

---

## 🛠️ Prerequisites
Assuming you are starting with a clean system and only have **VS Code** installed, you will need to install the following software to run this project:

1. **Git:** To clone this repository. ([Download Git](https://git-scm.com/downloads))
2. **Docker Desktop:** Required to run the isolated backend API and heavy ML models. ([Download Docker](https://www.docker.com/products/docker-desktop/))
3. **Anaconda (or Miniconda):** Required to manage the Python environment and run the data pipeline. ([Download Anaconda](https://www.anaconda.com/download))

---

## 🚀 Installation & Setup

### Step 1: Clone the Repository
Open an Anaconda Prompt or a VS Code terminal and run the following commands to download the code to your `D:` drive:
```bash
git clone <https://github.com/RISHI41221/CineIQ.git>
cd CineIQ
