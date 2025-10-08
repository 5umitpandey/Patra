"""
elo_update.py
-------------
Purpose:
    Maintain and update dynamic Elo ratings for users based on interactions
    (like / superlike / reject). Works in tandem with recommender and
    interaction modules.

Integration Order:
    Runs AFTER:
        - reject_superlike_like.py (user actions)
    BEFORE:
        - recommender.py (to influence ranking)
"""

import pandas as pd
from functools import lru_cache
from production.config_loader import load_config
from production.logger import get_logger

# -------------------- INIT --------------------
logger = get_logger(__name__)
config = load_config()

USERS_CSV = config["paths"]["users_csv"]
K = config.get("elo_k", 32)

# Default action scoring
ACTION_SCORES = {
    "like": (1, 0),
    "superlike": (1, 0),
    "reject": (0, 1)
}


# -------------------- HELPER FUNCTIONS --------------------
def expected_score(rating_a: float, rating_b: float) -> float:
    """Elo expected score of player A vs B."""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def update_elo_score(rating_a: float, rating_b: float, score_a: float, score_b: float, k: float = K):
    """Compute new Elo ratings for two users."""
    exp_a = expected_score(rating_a, rating_b)
    exp_b = expected_score(rating_b, rating_a)
    new_a = rating_a + k * (score_a - exp_a)
    new_b = rating_b + k * (score_b - exp_b)
    return new_a, new_b


# -------------------- DATA LOADING --------------------
@lru_cache(maxsize=1)
def load_users() -> pd.DataFrame:
    """Load users and ensure Elo column exists."""
    users = pd.read_csv(USERS_CSV)
    if "elo" not in users.columns:
        users["elo"] = 1500  # default Elo
    if "user_id" not in users.columns:
        raise ValueError("users.csv must contain 'user_id' column.")
    return users


# -------------------- CORE FUNCTIONS --------------------
def process_interaction(user_id: str, target_id: str, action: str):
    """
    Update Elo ratings for two users based on a single interaction.

    Args:
        user_id: Active user performing the action.
        target_id: Target user receiving the action.
        action: Action type (like, superlike, reject).
    """
    users = load_users()

    # Validate users exist
    if user_id not in users["user_id"].values:
        logger.warning(f"User {user_id} not found.")
        return
    if target_id not in users["user_id"].values:
        logger.warning(f"Target {target_id} not found.")
        return

    # Get current ratings
    ua = users.loc[users["user_id"] == user_id].iloc[0]
    ub = users.loc[users["user_id"] == target_id].iloc[0]

    Ra, Rb = ua["elo"], ub["elo"]

    # Determine scores
    Sa, Sb = ACTION_SCORES.get(action.lower(), (0, 0))

    # Update Elo
    new_a, new_b = update_elo_score(Ra, Rb, Sa, Sb)
    users.loc[users["user_id"] == user_id, "elo"] = new_a
    users.loc[users["user_id"] == target_id, "elo"] = new_b

    # Save updated Elo ratings
    users.to_csv(USERS_CSV, index=False)
    logger.info(f"Updated Elo → {user_id}: {new_a:.1f}, {target_id}: {new_b:.1f}")


def get_elo_scores() -> pd.DataFrame:
    """Return current Elo scores for all users (for recommender)."""
    users = load_users()
    return users[["user_id", "elo"]].copy()


# -------------------- STANDALONE TEST --------------------
if __name__ == "__main__":
    process_interaction("U001", "U002", "like")
    print(get_elo_scores().head(10))
