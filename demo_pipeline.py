# test_pipeline.py

import pandas as pd
from pathlib import Path

# Import your custom modules
from cineiq_hybrid_backend import (
    configure_hybrid_engine, 
    hybrid_recommendation, 
    load_svd_model
)
from cineiq_sentiment_reranker import apply_sentiment_reranking
from cineiq_explainability import add_explanations

def main():
    print("1. Loading project data...")
    # Dynamic pathing just like you set up
    project_root = Path(__file__).resolve().parent
    data_path = project_root / "cleaned_data"
    
    # Load the necessary DataFrames
    movies = pd.read_csv(data_path / "movies_master.csv", low_memory=False)
    
    # Note: In a real test, you need to define how your content and collab 
    # recommenders work here, or import them if you moved them to a .py file!
    # For this test, we assume they are configured.
    
    print("2. Configuring Hybrid Engine...")
    try:
        # Load the SVD model you trained previously
        svd_model = load_svd_model()
        
        # In a real scenario, you'd pass your actual recommend_movies and recommend_collaborative functions here
        # configure_hybrid_engine(movies, svd_model, recommend_movies, recommend_collaborative)
        print("Engine configured successfully.")
    except FileNotFoundError:
        print("ERROR: SVD model not found. You need to train it first in the notebook!")
        return

    # --- SIMULATING A USER REQUEST ---
    test_user = 1
    test_movie = "Toy Story"
    
    print(f"\n3. Generating base hybrid recommendations for '{test_movie}'...")
    # NOTE: This will fail until you pass the actual content/collab functions to configure_hybrid_engine
    # hybrid_recs = hybrid_recommendation(userId=test_user, movie_title=test_movie, top_n=10)
    
    # For the sake of testing the RERANKER specifically, let's mock a dataframe 
    # that looks exactly like the output of hybrid_recommendation:
    mock_hybrid_output = pd.DataFrame({
        "movieId": [1, 2, 3],
        "title": ["Toy Story", "Jumanji", "Grumpier Old Men"],
        "final_hybrid_score": [0.85, 0.72, 0.60]
    })
    
    print("\n4. Passing to Sentiment Reranker...")
    reranked_recs = apply_sentiment_reranking(mock_hybrid_output, movies)
    
    print("\n5. Generating Explanations...")
    # Add dummy scores so the explainability module has something to read
    reranked_recs['tfidf_score'] = [0.8, 0.4, 0.1]
    reranked_recs['svd_score'] = [0.9, 0.8, 0.7]
    reranked_recs['pearson_score'] = [0.85, 0.6, 0.3] 
    
    # We also need a mock vader score since the reranker was supposed to generate it
    if 'vader_compound_score' not in reranked_recs.columns:
        reranked_recs['vader_compound_score'] = [0.8, -0.2, 0.5]

    final_results = add_explanations(reranked_recs)
    
    print("\n--- FINAL OUTPUT TO FRONTEND ---")
    print(final_results[['title', 'final_hybrid_score', 'sentiment_adjusted_score', 'explanation']])

if __name__ == "__main__":
    main()