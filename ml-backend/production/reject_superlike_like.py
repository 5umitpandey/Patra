"""
reject_superlike_like.py
------------------------
Purpose:
    Adjusts recommendation candidate scores based on past swipe interactions
    (like / superlike / reject). This helps the recommender personalize the
    feed using learned user preferences.

Integration Order:
    Runs AFTER:
        - data_match.py
        - bio_match.py
    BEFORE:
        - elo_update.py
        - recommender.py
"""

import pandas as pd
import numpy as np
from functools import lru_cache
from production.logger import get_logger
from production.config_loader import load_config

# -------------------- INIT --------------------
logger = get_logger(__name__)
config = load_config()

USERS_CSV = config["paths"]["users_csv"]
SWIPE_LOG_CSV = config["paths"]["swipe_log_csv"]

# Default interaction weight configuration
ACTION_WEIGHTS = {
    "superlike": 3.0,
    "like": 2.0,
    "reject": 0.0
}

DEFAULT_WEIGHT = 1.0


# -------------------- LOADERS --------------------
@lru_cache(maxsize=1)
def load_users() -> pd.DataFrame:
    """Load user profiles."""
    users = pd.read_csv(USERS_CSV)
    if "user_id" not in users.columns:
        raise ValueError("users.csv must contain 'user_id'")
    return users


def load_swipe_logs() -> pd.DataFrame:
    """Load swipe interaction logs."""
    try:
        logs = pd.read_csv(SWIPE_LOG_CSV)
        if not {"user_id", "target_user_id", "action"}.issubset(logs.columns):
            raise ValueError("swipe_log.csv must contain ['user_id', 'target_user_id', 'action']")
        return logs
    except FileNotFoundError:
        logger.warning("Swipe log not found, creating an empty log.")
        return pd.DataFrame(columns=["user_id", "target_user_id", "action"])


# -------------------- CORE FUNCTIONS --------------------
def get_interaction_weight(action: str) -> float:
    """Return the weight multiplier for a given action."""
    return ACTION_WEIGHTS.get(action.lower(), DEFAULT_WEIGHT)


def adjust_candidate_scores(user_id: str, candidates_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adjusts candidate scores based on user’s past interactions.

    Args:
        user_id: The active user's ID.
        candidates_df: DataFrame containing columns ['user_id', 'score'] from recommender/data_match.

    Returns:
        DataFrame with added column ['interaction_weight'] and adjusted 'score'.
    """
    users = load_users()
    swipe_logs = load_swipe_logs()

    if user_id not in users["user_id"].values:
        logger.warning(f"User {user_id} not found in users.csv.")
        return candidates_df

    user_swipes = swipe_logs[swipe_logs["user_id"] == user_id]

    # Initialize weights
    candidates_df["interaction_weight"] = DEFAULT_WEIGHT

    for idx, row in candidates_df.iterrows():
        target_id = row["user_id"]
        previous_action = user_swipes[user_swipes["target_user_id"] == target_id]

        if not previous_action.empty:
            action = previous_action.iloc[0]["action"]
            weight = get_interaction_weight(action)
            candidates_df.at[idx, "interaction_weight"] = weight

            # Optional: downweight rejects entirely
            if action.lower() == "reject":
                candidates_df.at[idx, "score"] = 0.0
            else:
                candidates_df.at[idx, "score"] *= weight

    logger.info(f"Adjusted candidate scores for user {user_id} based on swipe logs.")
    return candidates_df.sort_values(by="score", ascending=False).reset_index(drop=True)


# -------------------- UPDATE LOGS --------------------
def update_user_interactions(user_id: str, target_user_id: str, action: str):
    """Log a new interaction (like/superlike/reject)."""
    swipe_logs = load_swipe_logs()
    new_entry = pd.DataFrame([{
        "user_id": user_id,
        "target_user_id": target_user_id,
        "action": action.lower()
    }])
    updated_logs = pd.concat([swipe_logs, new_entry], ignore_index=True)
    updated_logs.to_csv(SWIPE_LOG_CSV, index=False)
    logger.info(f"User {user_id} performed '{action}' on {target_user_id}.")


# -------------------- STANDALONE TEST --------------------
if __name__ == "__main__":
    test_user_id = "U001"
    # Dummy candidate scores (normally output of recommender)
    dummy_candidates = pd.DataFrame({
        "user_id": ["U002", "U003", "U004"],
        "score": [0.75, 0.60, 0.55]
    })

    print("Before adjustment:\n", dummy_candidates)
    adjusted = adjust_candidate_scores(test_user_id, dummy_candidates)
    print("\nAfter adjustment:\n", adjusted)
