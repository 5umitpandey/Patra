"""
production/__init__.py
----------------------
Marks the production folder as a Python package.
Optionally exposes key functions for simpler imports.
"""

# Expose main modules
from .data_match import get_top_matches
from .bio_match import top_similar_bios
from .reject_superlike_like import get_sbert_model, adjust_candidate_scores, update_user_interactions
from .elo_update import process_interaction, get_elo_scores
from .recommender import get_recommendations
