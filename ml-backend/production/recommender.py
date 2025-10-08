"""
recommender.py
--------------
Purpose:
    Combine signals from data_match, bio_match, interaction weights, and Elo ratings
    to generate a final ranked recommendation feed for a user.

Integration Order:
    Runs AFTER:
        - data_match.py
        - bio_match.py
        - reject_superlike_like.py
        - elo_update.py
"""

import pandas as pd
from functools import lru_cache

from production.data_match import get_top_matches
from production.bio_match import top_similar_bios
from production.reject_superlike_like import adjust_candidate_scores
from production.elo_update import get_elo_scores
from production.logger import get_logger
from production.config_loader import load_config

# -------------------- INIT --------------------
logger = get_logger(__name__)
config = load_config()

TOP_N = config.get("recommender_top_n", 10)


# -------------------- CORE FUNCTION --------------------
def get_recommendations(user_id: str, top_n: int = TOP_N) -> pd.DataFrame:
    """
    Generate final ranked recommendations for a user by combining:
        - Structured feature similarity (data_match)
        - Bio semantic similarity (bio_match)
        - Interaction weights (reject_superlike_like)
        - Elo ranking (elo_update)

    Returns:
        DataFrame with columns: ['user_id', 'name', 'gender', 'final_score']
    """

    # ------------------ 1. Base matches (structured features) ------------------
    users = pd.read_csv(config["paths"]["users_csv"])
    if user_id not in users["user_id"].values:
        raise ValueError(f"User {user_id} not found in users.csv")

    user_index = users.index[users["user_id"] == user_id][0]
    base_matches = get_top_matches(user_index, top_n=50, use_swipe_logs=False)
    base_matches = base_matches[["user_id", "name", "gender", "score"]].copy()
    base_matches.rename(columns={"score": "base_score"}, inplace=True)

    # ------------------ 2. Bio similarity ------------------
    bio_matches = top_similar_bios(config["paths"]["users_csv"], user_id, top_n=50)
    bio_matches = bio_matches[["user_id", "similarity"]].copy()
    bio_matches.rename(columns={"similarity": "bio_score"}, inplace=True)

    # Merge base + bio
    merged = pd.merge(base_matches, bio_matches, on="user_id", how="left")
    merged["bio_score"].fillna(0.0, inplace=True)

    # ------------------ 3. Interaction weights ------------------
    merged = adjust_candidate_scores(user_id, merged)
    # merged now has column 'score' updated with interaction weights
    # We'll combine base_score, bio_score, and weighted score later

    # ------------------ 4. Elo scores ------------------
    elo_df = get_elo_scores()
    merged = pd.merge(merged, elo_df, on="user_id", how="left")
    merged["elo"].fillna(1500, inplace=True)  # default Elo if missing

    # ------------------ 5. Compute final score ------------------
    # Weighted sum: can be configured in config.yaml
    WEIGHTS = config.get("recommender_weights", {
        "base": 0.4,
        "bio": 0.3,
        "interaction": 0.2,
        "elo": 0.1
    })

    # Normalize Elo to 0-1
    elo_min, elo_max = merged["elo"].min(), merged["elo"].max()
    merged["elo_norm"] = (merged["elo"] - elo_min) / max(elo_max - elo_min, 1)

    merged["final_score"] = (
        WEIGHTS["base"] * merged["base_score"] +
        WEIGHTS["bio"] * merged["bio_score"] +
        WEIGHTS["interaction"] * merged["score"] +
        WEIGHTS["elo"] * merged["elo_norm"]
    )

    # ------------------ 6. Return top N ------------------
    merged_sorted = merged.sort_values(by="final_score", ascending=False)
    top_feed = merged_sorted.head(top_n)[
        ["user_id", "name", "gender", "final_score"]
    ].reset_index(drop=True)

    logger.info(f"Generated top {top_n} recommendations for user {user_id}")
    return top_feed


# -------------------- STANDALONE TEST --------------------
if __name__ == "__main__":
    test_user_id = "U001"
    recommendations = get_recommendations(test_user_id)
    print(recommendations)
