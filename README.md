# CineIQ: Hybrid Explainable Movie Recommendation Engine

## Problem Statement

Modern streaming platforms often trap users inside an opaque "recommendation loop" where movies are suggested without clearly explaining why they were chosen. **CineIQ** solves that problem by combining a **hybrid recommendation engine** with **sentiment reranking** and **Explainable AI (XAI)** so recommendations are both personalized and understandable.

The recommendation engine combines:

- **SVD** for personalized recommendations
- **Pearson correlation** for collaborative filtering
- **TF-IDF** for content-based similarity
- **VADER sentiment reranking** to refine recommendation quality
- **Explainable AI** to generate a human-readable reason for each recommendation

This project uses:

- A **FastAPI backend** running in **Docker**
- A **Streamlit frontend** running locally on Windows

---

## Prerequisites

Before you begin, make sure these tools are installed:

- **Git**  To clone this repository. ([Download Git](https://git-scm.com/downloads))
- **Docker Desktop**  Required to run the isolated backend API and heavy ML models. ([Download Docker](https://www.docker.com/products/docker-desktop/))
- **Anaconda**  Required to manage the Python environment and run the data pipeline. ([Download Anaconda](https://www.anaconda.com/download))

If you only have VS Code installed right now, install the three tools above first.

---

## Installation & Setup

### Step 1: Open Anaconda Prompt and Clone the Repository

On Windows, open **Anaconda Prompt** first.

Before cloning the project, move to the drive where you want to save it. For example, if you want to work on your `D:` drive, type:

```bash
D:
```

If you want to work on your `C:` drive instead, type:

```bash
C:
```

After moving to the correct drive, run:

```bash
git clone https://github.com/RISHI41221/CineIQ.git
cd CineIQ
```

This downloads the repository and moves you into the project folder.

---

### Step 2: Create and Activate the Environment

In the same **Anaconda Prompt**, run:

```bash
conda create -n cineiq_env python=3.10 -y
conda activate cineiq_env
pip install -r requirements.txt
```

This creates a dedicated Python environment for the project and installs the required dependencies.

---

### Step 3: Download the Raw Data

The raw data files are too large to store directly on GitHub. This project depends on raw datasets such as:

- **MovieLens**
- **TMDB**
- **IMDB**

Download the zip file from:

[INSERT_DRIVE_LINK_HERE](https://example.com)

After downloading it, extract the folders directly into a `raw_data/` directory inside the project root.

Your project should look like this afterward:

```bash
CineIQ/
  raw_data/
  frontend/
  CineIQ.ipynb
  build_pearson.py
  README.md
```

---

### Step 4: Run the Data Pipeline

Before running the app, you must generate the cleaned data files and trained artifacts.

1. Open **VS Code**.
2. Open the file `CineIQ.ipynb`.
3. In the **top-right corner** of the notebook, select the **`cineiq_env`** kernel.
4. Run **all cells** from top to bottom.

The notebook will:

- Clean the raw data
- Generate processed datasets
- Train and save the **SVD model**

After the notebook finishes, go back to the **Anaconda Prompt** and run:

```bash
python build_pearson.py
```

This creates the final recommendation artifacts needed by the backend.

---

## Running the Application (Two Anaconda Prompts)

You must use **two separate Anaconda Prompts** to run this project:

- **Anaconda Prompt 1** runs the FastAPI backend
- **Anaconda Prompt 2** runs the Streamlit frontend

Do not close the backend prompt while using the frontend.

---

### Terminal 1 (Backend)

Open an **Anaconda Prompt** and navigate to the project root.

If needed, first switch to the drive where the project is stored:

```bash
D:
```

Then move into the project folder:

```bash
cd path\to\CineIQ
```

Build the Docker image:

```bash
docker build -t cineiq-api .
```

Run the backend container:

```bash
docker run -p 8000:8000 cineiq-api
```

Wait for the backend startup message before opening the frontend. Once it starts successfully, keep this prompt open.

---

### Terminal 2 (Frontend)

Open a **brand new Anaconda Prompt**.

If needed, switch to the correct drive first:

```bash
D:
```

Then navigate directly to the frontend folder:

```bash
cd path\to\CineIQ\frontend
```

Activate the environment:

```bash
conda activate cineiq_env
```

Start the Streamlit frontend:

```bash
streamlit run app.py
```

This launches the frontend locally so it can connect to the FastAPI backend.

---

## Project Resources

- **Deployment Video:** [INSERT_DEPLOYMENT_VIDEO_LINK_HERE](https://example.com)
- **Full Project Report:** [INSERT_PROJECT_REPORT_LINK_HERE](https://example.com)
