# CineIQ: Hybrid Explainable Movie Recommendation Engine

## Problem Statement

Streaming platforms often trap users in an opaque "recommendation loop" where movies are suggested without any clear reason. CineIQ solves this by combining a **hybrid recommendation engine** with **sentiment reranking** and **Explainable AI (XAI)** so users can receive recommendations that are both personalized and understandable.

The recommendation engine combines:

- **SVD** for personalized recommendations
- **Pearson correlation** for collaborative filtering
- **TF-IDF** for content-based similarity
- **Sentiment reranking** to refine recommendation quality
- **Explainable AI** to show why each movie was recommended

This project is built as a microservices application with:

- A **FastAPI backend** running in **Docker**
- A **Streamlit frontend** running locally

---

## Prerequisites

Before you begin, install the following tools on your computer:

- **Git**
- **Docker Desktop**
- **Anaconda**

If you only have VS Code installed right now, install the tools above first.

---

## Installation & Setup (Step-by-Step)

### Step 1: Clone the Repository

Open a terminal in **VS Code** and run:

```bash
git clone https://github.com/your-username/CineIQ.git
cd CineIQ
```

---

### Step 2: Create the Python Environment

Create a Conda environment:

```bash
conda create -n cineiq_env python=3.10 -y
```

Activate the environment:

```bash
conda activate cineiq_env
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

### Step 3: Download the Raw Data

The raw dataset files are too large to store directly in GitHub. You must download them separately from Google Drive and place them into a folder named `raw_data/` in the project root.

Download link:

[INSERT_GOOGLE_DRIVE_LINK_HERE](https://example.com)

After downloading, your project should contain a folder like this:

```bash
CineIQ/
  raw_data/
```

---

### Step 4: Run the Data Pipeline

Before starting the app, you must generate the cleaned datasets and model artifacts.

1. Open `CineIQ.ipynb` in VS Code.
2. Run **all cells** in the notebook.

This notebook will generate:

- Cleaned data files
- The trained **SVD model**

After the notebook finishes, return to the terminal in the project root and run:

```bash
python build_pearson.py
```

This command generates the remaining artifacts needed by the recommendation system.

---

## Running the Application (Two Terminals)

You must use **two separate terminals**.

### Terminal 1: Start the FastAPI Backend

From the project root, build the Docker image:

```bash
docker build -t cineiq-api .
```

Then run the backend container:

```bash
docker run -p 8000:8000 cineiq-api
```

Keep this terminal open while using the frontend.

### Terminal 2: Start the Streamlit Frontend

Open a **new terminal** in VS Code and run:

```bash
cd frontend
conda activate cineiq_env
streamlit run app.py
```

This will start the Streamlit frontend locally.

---

## Project Resources

- **Deployment Video:** [INSERT_DEPLOYMENT_VIDEO_LINK_HERE](https://example.com)
- **Full Project Report:** [INSERT_PROJECT_REPORT_LINK_HERE](https://example.com)
