import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from functools import lru_cache

# -------------------- CONFIG --------------------
USERS_CSV = r"D:\SM\Projects\Patra_ML\data\profiles\users.csv"
SWIPE_LOG_CSV = r"D:\SM\Projects\Patra_ML\data\profiles\swipe_log.csv"

WEIGHTS = {
    "age": 0.22,
    "location": 0.15,
    "hobbies": 0.20,
    "looking_for": 0.10,
    "bio": 0.23,
    "profession": 0.10,
}

ACTION_WEIGHTS = {"superlike": 3, "like": 2, "reject": 0}


def get_sbert_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# -------------------- DATA LOADING --------------------
def load_users(users_csv=USERS_CSV):
    users = pd.read_csv(users_csv)
    users.reset_index(inplace=True)  # keep original index for bio mapping
    return users


def load_swipe_logs(swipe_log_csv=SWIPE_LOG_CSV):
    try:
        return pd.read_csv(swipe_log_csv)
    except FileNotFoundError:
        return pd.DataFrame(columns=["user_id", "target_user_id", "action"])


# -------------------- SIMILARITY HELPERS --------------------
def jaccard_score(list1, list2):
    s1, s2 = set(list1), set(list2)
    return len(s1 & s2) / len(s1 | s2) if s1 | s2 else 0


def compute_bio_embeddings(users):
    model = get_sbert_model()
    return model.encode(users["bio"].fillna("").tolist(), convert_to_numpy=True)


# -------------------- MATCH SCORE --------------------
def match_score(u1, u2, bio_embeddings):
    """Weighted multi-factor score between two users."""
    # Age
    age_score = max(0, 1 - abs(u1.age - u2.age) / 10)

    # Location
    if u1.city == u2.city:
        loc_score = 1
    elif u1.state == u2.state:
        loc_score = 0.5
    else:
        loc_score = 0

    # Hobbies & looking_for
    hobby_score = jaccard_score(u1.hobbies.split(";"), u2.hobbies.split(";"))
    lf_score = jaccard_score(u1.looking_for.split(";"), u2.looking_for.split(";"))

    # Bio
    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    bio_score = cosine_sim(bio_embeddings[u1["index"]], bio_embeddings[u2["index"]])


    # Profession
    prof_score = 1 if u1.profession == u2.profession else 0.5

    # Final weighted score
    total = (
        WEIGHTS["age"] * age_score
        + WEIGHTS["location"] * loc_score
        + WEIGHTS["hobbies"] * hobby_score
        + WEIGHTS["looking_for"] * lf_score
        + WEIGHTS["bio"] * bio_score
        + WEIGHTS["profession"] * prof_score
    )
    return total


def action_weight(action):
    return ACTION_WEIGHTS.get(action, 1)


# -------------------- MATCH ENGINE --------------------
def get_top_matches(user_index, top_n=10, use_swipe_logs=True):
    users = load_users()
    swipe_logs = load_swipe_logs()
    bio_embeddings = compute_bio_embeddings(users)

    u1 = users.iloc[user_index]

    candidates = []
    for i, u2 in users.iterrows():
        if i == user_index:
            continue
        score = match_score(u1, u2, bio_embeddings)
        candidates.append((u2.user_id, u2.name, u2.gender, score))

    swiped, new_candidates = [], []

    # Check swipe history
    for c in candidates:
        swipe = swipe_logs[
            (swipe_logs.user_id == u1.user_id)
            & (swipe_logs.target_user_id == c[0])
        ]
        if not swipe.empty:
            act = swipe.iloc[0]["action"]
            if act == "reject":
                continue
            elif act in ["like", "superlike"]:
                weighted_score = c[3] * action_weight(act)
                swiped.append((c[0], c[1], c[2], weighted_score))
        else:
            new_candidates.append(c)

    # Orientation handling
    if u1.gender in ["Gay", "Lesbian"]:
        same = sorted(
            [c for c in new_candidates if c[2] == u1.gender],
            key=lambda x: x[3],
            reverse=True,
        )
        other = sorted(
            [c for c in new_candidates if c[2] != u1.gender],
            key=lambda x: x[3],
            reverse=True,
        )
        n_same = min(len(same), top_n // 2)
        n_other = top_n - n_same
        mixed_new = sorted(same[:n_same] + other[:n_other], key=lambda x: x[3], reverse=True)
    else:
        mixed_new = sorted(new_candidates, key=lambda x: x[3], reverse=True)

    # Final ranking
    if use_swipe_logs:
        swiped_sorted = sorted(swiped, key=lambda x: x[3], reverse=True)
        remaining_slots = top_n - len(swiped_sorted)
        if remaining_slots <= 0:
            final_list = swiped_sorted[:top_n]
        else:
            final_list = swiped_sorted + mixed_new[:remaining_slots]
    else:
        final_list = mixed_new[:top_n]

    # Return DataFrame for consistency
    return pd.DataFrame(final_list, columns=["user_id", "name", "gender", "score"])
