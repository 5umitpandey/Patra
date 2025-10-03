import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache

# -------------------- MODEL LOADING --------------------
@lru_cache(maxsize=1)
def get_model(model_name: str = "all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)


# -------------------- CORE FUNCTIONS --------------------
def bio_similarity(bio_a: str, bio_b: str, model=None) -> float:
    if model is None:
        model = get_model()
    embeddings = model.encode([bio_a, bio_b])
    return float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])


def top_similar_bios(users_csv: str, target_user_id: str, top_n: int = 10, model=None) -> pd.DataFrame:
    if model is None:
        model = get_model()

    # Load users
    users = pd.read_csv(users_csv)
    if "bio" not in users.columns or "user_id" not in users.columns:
        raise ValueError("CSV must contain 'user_id' and 'bio' columns.")

    # Encode all bios
    bio_embeddings = model.encode(users["bio"].tolist(), convert_to_numpy=True)

    # Get target
    if target_user_id not in users["user_id"].values:
        raise ValueError(f"User {target_user_id} not found in CSV.")

    target_idx = users[users["user_id"] == target_user_id].index[0]
    target_emb = bio_embeddings[target_idx]

    # Compute similarities
    similarities = cosine_similarity([target_emb], bio_embeddings)[0]
    similarities[target_idx] = -1  # exclude self

    # Pick top N
    top_indices = similarities.argsort()[::-1][:top_n]

    # Prepare results as DataFrame (easier for integration/logging)
    results = users.iloc[top_indices].copy()
    results["similarity"] = similarities[top_indices]

    return results[["user_id", "bio", "similarity"]]
