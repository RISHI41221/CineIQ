"""Rule-based explainability utilities for CINEIQ recommendations.

This module builds dynamic explanation strings from the strongest
recommendation signal, shared metadata, and sentiment context.

Typical usage
-------------
```python
from cineiq_explainability import add_explanations

final_df = add_explanations(
    reranked_df,
    search_query="Toy Story",
    search_movie_genres="Adventure|Animation|Children|Comedy|Fantasy",
)
```
"""

from __future__ import annotations

from typing import Any

import pandas as pd


POSITIVE_SENTIMENT_THRESHOLD = 0.6
NEGATIVE_SENTIMENT_THRESHOLD = -0.4


def add_explanations(
    recommendations_df: pd.DataFrame,
    *,
    search_query: str = "",
    search_movie_genres: Any = None,
) -> pd.DataFrame:
    """Attach a dynamic explanation string to each recommendation row.

    Parameters
    ----------
    recommendations_df:
        Recommendation DataFrame containing at least:
        - `tfidf_score`
        - `pearson_score`
        - `svd_score`
        - `vader_compound_score`
    search_query:
        The original movie title the user searched for.
    search_movie_genres:
        The genre string for the searched movie, if available.

    Returns
    -------
    pd.DataFrame
        A copy of the input DataFrame with a new `explanation` column.
    """

    _validate_recommendation_schema(recommendations_df)

    if recommendations_df.empty:
        empty_result = recommendations_df.copy()
        empty_result["explanation"] = pd.Series(dtype=str)
        return empty_result

    result = recommendations_df.copy()

    score_columns = ["tfidf_score", "pearson_score", "svd_score", "vader_compound_score"]
    for column in score_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    # Missing scores are treated as neutral/absent signals so the row can still
    # receive an explanation instead of failing halfway through the pipeline.
    result["tfidf_score"] = result["tfidf_score"].fillna(0.0)
    result["pearson_score"] = result["pearson_score"].fillna(0.0)
    result["svd_score"] = result["svd_score"].fillna(0.0)
    result["vader_compound_score"] = result["vader_compound_score"].fillna(0.0)

    normalized_search_query = _normalize_search_query(search_query)
    result["explanation"] = result.apply(
        lambda row: _build_explanation_for_row(
            row,
            search_query=normalized_search_query,
            search_movie_genres=search_movie_genres,
        ),
        axis=1,
    )
    return result


def _validate_recommendation_schema(recommendations_df: pd.DataFrame) -> None:
    """Verify that the DataFrame contains the columns required for explanations."""

    if not isinstance(recommendations_df, pd.DataFrame):
        raise TypeError("recommendations_df must be a pandas DataFrame.")

    required_columns = {
        "tfidf_score",
        "pearson_score",
        "svd_score",
        "vader_compound_score",
    }
    missing_columns = required_columns.difference(recommendations_df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            "recommendations_df is missing required columns: "
            f"{missing}"
        )


def _build_explanation_for_row(
    row: pd.Series,
    *,
    search_query: str,
    search_movie_genres: Any,
) -> str:
    """Create one explanation string for a recommendation row."""

    primary_driver = _select_primary_driver(row)
    explanation_parts: list[str] = []

    if primary_driver == "pearson" and search_query:
        explanation_parts.append(
            f"Fans of {search_query} also highly rated this film."
        )

    genre_explanation = _genre_match_explanation(
        recommendation_genres=row.get("genres"),
        search_movie_genres=search_movie_genres,
    )
    if genre_explanation:
        explanation_parts.append(genre_explanation)

    sentiment_explanation = _sentiment_suffix(row.get("vader_compound_score", 0.0))
    if sentiment_explanation:
        explanation_parts.append(sentiment_explanation)

    if not explanation_parts:
        explanation_parts.append(_driver_fallback_explanation(primary_driver, row))

    return " ".join(explanation_parts)


def _select_primary_driver(row: pd.Series) -> str:
    """Find the dominant recommendation signal for a row.

    Tie handling is deterministic: if two or more scores are equal, the first
    signal in the order TF-IDF -> Pearson -> SVD wins.
    """

    scored_drivers = [
        ("tfidf", _safe_float(row.get("tfidf_score", 0.0))),
        ("pearson", _safe_float(row.get("pearson_score", 0.0))),
        ("svd", _safe_float(row.get("svd_score", 0.0))),
    ]
    return max(scored_drivers, key=lambda item: item[1])[0]


def _driver_fallback_explanation(driver_name: str, row: pd.Series) -> str:
    """Build a dynamic fallback explanation for rows without richer context."""

    if driver_name == "tfidf":
        top_genres = _top_genres(row.get("genres"), limit=2)
        if top_genres:
            return (
                "Matched through similar content signals, especially around "
                f"{_join_with_and(top_genres)}."
            )

        tfidf_score = _safe_float(row.get("tfidf_score", 0.0))
        return f"Matched through similar content signals with a TF-IDF score of {tfidf_score:.2f}."

    if driver_name == "pearson":
        pearson_score = _safe_float(row.get("pearson_score", 0.0))
        return (
            "Viewers with similar taste patterns drove this pick with a "
            f"Pearson score of {pearson_score:.2f}."
        )

    if driver_name == "svd":
        predicted_rating = row.get("svd_predicted_rating")
        if predicted_rating is not None and not pd.isna(predicted_rating):
            return (
                "Predicted to be a strong fit for you with an estimated rating of "
                f"{_safe_float(predicted_rating):.2f}."
            )

        svd_score = _safe_float(row.get("svd_score", 0.0))
        return f"Predicted to be a strong fit for you with a personalized score of {svd_score:.2f}."

    raise ValueError(f"Unsupported explanation driver: {driver_name}")


def _genre_match_explanation(
    recommendation_genres: Any,
    search_movie_genres: Any,
) -> str:
    """Describe the strongest shared genre overlap, if one exists."""

    recommended_genres = _parse_genres(recommendation_genres)
    if not recommended_genres:
        return ""

    search_genre_keys = {
        genre.casefold()
        for genre in _parse_genres(search_movie_genres)
    }
    if not search_genre_keys:
        return ""

    shared_genres = [
        genre
        for genre in recommended_genres
        if genre.casefold() in search_genre_keys
    ][:2]
    if not shared_genres:
        return ""

    return (
        "A strong match based on your interest in "
        f"{_join_with_and(shared_genres)}."
    )


def _sentiment_suffix(vader_compound_score: Any) -> str:
    """Return the optional VADER-based explanation suffix."""

    compound_score = _safe_float(vader_compound_score)

    if compound_score > POSITIVE_SENTIMENT_THRESHOLD:
        return (
            "\U0001F525 Audiences are currently raving about it with a "
            f"sentiment score of {compound_score:.2f}!"
        )
    if compound_score < NEGATIVE_SENTIMENT_THRESHOLD:
        return (
            "Audience reactions are more mixed right now with a sentiment "
            f"score of {compound_score:.2f}."
        )
    return ""


def _top_genres(raw_genres: Any, limit: int = 2) -> list[str]:
    """Return the first genres listed for a movie, preserving source order."""

    return _parse_genres(raw_genres)[:max(limit, 0)]


def _parse_genres(raw_genres: Any) -> list[str]:
    """Parse a MovieLens-style pipe-delimited genre string into clean labels."""

    if raw_genres is None or pd.isna(raw_genres):
        return []

    cleaned_genres = str(raw_genres).strip()
    if not cleaned_genres or cleaned_genres.lower() == "nan":
        return []
    if cleaned_genres == "(no genres listed)":
        return []

    deduped_genres: list[str] = []
    seen_genres: set[str] = set()
    for genre in cleaned_genres.split("|"):
        normalized_genre = genre.strip()
        if not normalized_genre:
            continue

        lowered_genre = normalized_genre.casefold()
        if lowered_genre in seen_genres:
            continue

        seen_genres.add(lowered_genre)
        deduped_genres.append(normalized_genre)

    return deduped_genres


def _join_with_and(values: list[str]) -> str:
    """Join one or two short labels into a natural-language phrase."""

    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    return f"{values[0]} and {values[1]}"


def _normalize_search_query(search_query: Any) -> str:
    """Normalize the user-entered search query without changing its casing."""

    if search_query is None or pd.isna(search_query):
        return ""

    cleaned_query = str(search_query).strip()
    if cleaned_query.lower() == "nan":
        return ""

    return cleaned_query


def _safe_float(value: Any) -> float:
    """Convert a value into float without raising on null-like inputs."""

    if value is None or pd.isna(value):
        return 0.0

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
