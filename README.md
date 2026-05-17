#### Part A: Open and Run the Notebook

1. Open `CineIQ.ipynb` in VS Code.
2. When prompted for a kernel, select **`cineiq_env`**.
3. Run **all cells** from top to bottom.

The notebook is responsible for:

- Cleaning and preparing the raw data
- Creating processed datasets
- Training and saving the **SVD model**

Wait until the notebook finishes completely before moving to the next step.

#### Part B: Generate the Pearson Artifacts

After the notebook has finished, return to the terminal in the project root and run:

```bash
python build_pearson.py
```

This step generates the additional collaborative-filtering artifacts required by the recommendation engine.

At the end of this step, the project should have the data and model files needed by the backend API.

---

## Running the Application

This project requires **two separate terminals**:

- **Terminal 1** runs the FastAPI backend inside Docker
- **Terminal 2** runs the Streamlit frontend locally

Do not close Terminal 1 while using the frontend, because the frontend depends on the backend API being available.

---

### Terminal 1 (Backend)

Open a terminal in the project root folder and build the Docker image:

```bash
docker build -t cineiq-api .
```

Then run the backend container:

```bash
docker run -p 8000:8000 cineiq-api
```

Wait for the backend startup to complete before opening the frontend. Once the API is running successfully, it should be available at:

```bash
http://127.0.0.1:8000
```

Keep this terminal open.

---

### Terminal 2 (Frontend)

Open a **new terminal** in VS Code.

Move into the frontend folder:

```bash
cd frontend
```

Activate the Conda environment again:

```bash
conda activate cineiq_env
```

Start the Streamlit app:

```bash
streamlit run app.py
```

If the command succeeds, Streamlit will usually open automatically in your browser. If it does not, copy the local URL shown in the terminal and open it manually.

---

## How Explainability Works

CineIQ does not just return a list of movie recommendations. It also generates dynamic explanation text so the user understands why a specific title appeared.

The explanation engine can describe recommendations in several ways:

- **TF-IDF:** Explains that a movie was selected because it shares similar themes or genres with the searched movie.
- **Pearson:** Explains that users with similar viewing patterns or rating behavior also liked the recommended movie.
- **SVD:** Explains that the model predicts the movie is a strong fit for the user's latent taste profile.
- **VADER sentiment:** Appends extra context when audience sentiment is especially strong, such as highlighting highly positive reception.

This makes the recommendation process more transparent and easier to trust.

---

## Project Resources

- **Deployment Video:** [INSERT_DEPLOYMENT_VIDEO_LINK_HERE](https://example.com)
- **Full Project Report:** [INSERT_PROJECT_REPORT_LINK_HERE](https://example.com)

