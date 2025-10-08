"""
main.py
-------
Purpose:
    Central orchestrator for Patra ML backend.
    Calls all modules in sequence to generate recommendations,
    apply interactions, update Elo, and provide final feed.
"""

import argparse
import pandas as pd

from production.logger import get_logger
from production.config_loader import load_config

from production.data_match import get_top_matches
from production.bio_match import top_similar_bios
from production.reject_superlike_like import adjust_candidate_scores, update_user_interactions
from production.elo_update import process_interaction, get_elo_scores
from production.recommender import get_recommendations

# -------------------- INIT --------------------
logger = get_logger(__name__)
config = load_config()


# -------------------- CORE FUNCTIONS --------------------
def generate_user_feed(user_id: str, top_n: int = None) -> pd.DataFrame:
    """
    Generate top recommendations for a given user.
    Combines base features, bio similarity, interaction weights, and Elo.
    """
    if top_n is None:
        top_n = config.get("recommender_top_n", 10)

    feed = get_recommendations(user_id, top_n=top_n)
    logger.info(f"Generated feed for user {user_id} (top {top_n})")
    return feed


def record_interaction(user_id: str, target_id: str, action: str):
    """
    Record a user interaction and update Elo ratings accordingly.
    """
    # Update Elo first
    process_interaction(user_id, target_id, action)
    
    # Update swipe logs / adjust candidate weights
    update_user_interactions(user_id, target_id, action)
    
    logger.info(f"Recorded interaction: {user_id} -> {target_id} [{action}]")


# -------------------- CLI / RUNNER --------------------
def main():
    parser = argparse.ArgumentParser(description="Patra ML Backend Main Orchestrator")
    parser.add_argument("--user_id", type=str, required=True, help="Active user ID")
    parser.add_argument("--top_n", type=int, default=None, help="Number of recommendations")
    parser.add_argument("--interact", nargs=2, metavar=("TARGET_ID", "ACTION"),
                        help="Optional: record an interaction (e.g., U002 like)")

    args = parser.parse_args()

    user_id = args.user_id

    # Record interaction if provided
    if args.interact:
        target_id, action = args.interact
        record_interaction(user_id, target_id, action)

    # Generate final feed
    feed = generate_user_feed(user_id, top_n=args.top_n)
    print("\nTop Recommendations:")
    print(feed.to_string(index=False))


# -------------------- ENTRY POINT --------------------
if __name__ == "__main__":
    main()
