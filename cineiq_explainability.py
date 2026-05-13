"""Rule-based explainability utilities for CINEIQ recommendations.

This module adds a lightweight explanation string to each recommendation based
on the strongest recommendation signal and optional sentiment context.

Typical usage
-------------
```python
from cineiq_explainability import add_explanations

final_df = add_explanations(reranked_df)
```
"""

from __future__ import annotations

from typing import Any

import pandas as pd


TFIDF_EXPLANATION = (
    "Recommended because it shares similar genres, themes, or cast members "
    "with your search."
)
PEARSON_EXPLANATION = (
    "Recommended because users who share your specific movie tastes rated "
    "this highly."
)
SVD_EXPLANATION = (
    "Recommended because our personalized prediction model strongly matches "
    "this to your profile."
)
POSITIVE_SENTIMENT_SUFFIX = " 🔥 Audiences are currently raving about it!"
NEGATIVE_SENTIMENT_SUFFIX = " ⚠️ Note: General audience reception is highly mixed."


def add_explanations(recommendations_df: pd.DataFrame) -> pd.DataFrame:
    """Attach a rule-based explanation string to each recommendation row.

    Parameters
    ----------
    recommendations_df:
        Recommendation DataFrame containing at least:
        - `tfidf_score`
        - `pearson_score`
        - `svd_score`
        - `vader_compound_score`

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

    result["explanation"] = result.apply(_build_explanation_for_row, axis=1)
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


def _build_explanation_for_row(row: pd.Series) -> str:
    """Create one explanation string for a recommendation row."""

    primary_driver = _select_primary_driver(row)
    base_explanation = _base_explanation_from_driver(primary_driver)
    sentiment_suffix = _sentiment_suffix(row.get("vader_compound_score", 0.0))
    return f"{base_explanation}{sentiment_suffix}"


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


def _base_explanation_from_driver(driver_name: str) -> str:
    """Map the strongest model signal to a user-facing explanation template."""

    if driver_name == "tfidf":
        return TFIDF_EXPLANATION
    if driver_name == "pearson":
        return PEARSON_EXPLANATION
    if driver_name == "svd":
        return SVD_EXPLANATION

    raise ValueError(f"Unsupported explanation driver: {driver_name}")


def _sentiment_suffix(vader_compound_score: Any) -> str:
    """Return the optional VADER-based explanation suffix."""

    compound_score = _safe_float(vader_compound_score)

    if compound_score > 0.6:
        return POSITIVE_SENTIMENT_SUFFIX
    if compound_score < -0.4:
        return NEGATIVE_SENTIMENT_SUFFIX
    return ""


def _safe_float(value: Any) -> float:
    """Convert a value into float without raising on null-like inputs."""

    if value is None or pd.isna(value):
        return 0.0

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
