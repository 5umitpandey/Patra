import pandas as pd
import numpy as np
from production.logger import get_logger
from production.utils import load_csv_safe

logger = get_logger()

# -----------------------------------------
# Jaccard similarity between list-like fields
# -----------------------------------------
def jaccard_score(list1, list2):
    s1, s2 = set(list1), set(list2)
    return len(s1 & s2) / len(s1 | s2) if s1 | s2 else 0


# -----------------------------------------
# Compute basic match score (no bio similarity)
# -----------------------------------------
def match_score(u1, u2):
    try:
        # Age score
        age_diff = abs(u1.age - u2.age)
        age_score = max(0, 1 - (age_diff / 10))

        # Location score
        if u1.city == u2.city:
            loc_score = 1
        elif u1.state == u2.state:
            loc_score = 0.5
        else:
            loc_score = 0

        # Hobbies score
        hobby_score = jaccard_score(str(u1.hobbies).split(";"), str(u2.hobbies).split(";"))

        # Looking-for score
        lf_score = jaccard_score(str(u1.looking_for).split(";"), str(u2.looking_for).split(";"))

        # Profession score
        prof_score = 1 if u1.profession == u2.profession else 0.5

        # Weighted sum (bio removed here)
        final_score = (
            0.30 * age_score +
            0.15 * loc_score +
            0.25 * hobby_score +
            0.20 * lf_score +
            0.10 * prof_score
        )
        return final_score

    except Exception as e:
        logger.error(f"Error computing match score: {e}")
        return 0.0


# -----------------------------------------
# Prepare top N candidates for a given user
# -----------------------------------------
def get_top_matches(user_index, users_df, top_n=10):
    u1 = users_df.iloc[user_index]
    scores = []

    for i, u2 in users_df.iterrows():
        if i == user_index:
            continue

        score = match_score(u1, u2)

        # Soft orientation boost
        if u1.gender in ["Gay", "Lesbian"] and u2.gender == u1.gender:
            score *= 1.2

        scores.append((u2.user_id, u2.name, u2.gender, score))

    # Sort all candidates by score descending
    scores = sorted(scores, key=lambda x: x[3], reverse=True)

    if u1.gender in ["Gay", "Lesbian"]:
        # Separate same-orientation and others
        same = [s for s in scores if s[2] == u1.gender]
        other = [s for s in scores if s[2] != u1.gender]

        half = top_n // 2
        top_matches = same[:half] + other[:top_n - half]
        top_matches = sorted(top_matches, key=lambda x: x[3], reverse=True)
    else:
        top_matches = scores[:top_n]

    results_df = pd.DataFrame(top_matches, columns=["match_user_id", "match_name", "match_gender", "base_score"])
    results_df["source_user_id"] = u1.user_id
    results_df["source_name"] = u1.name

    return results_df


# -----------------------------------------
# Main pipeline wrapper (used by main.py)
# -----------------------------------------
def prepare_candidate_pool(config):
    try:
        users_path = config["paths"]["users"]
        logger.info(f"Loading users from {users_path}")

        users_df = load_csv_safe(users_path)
        users_df.reset_index(drop=True, inplace=True)

        logger.info("Computing base candidate matches...")
        all_matches = []

        for idx in range(len(users_df)):
            top_df = get_top_matches(idx, users_df, top_n=config["matching"]["top_n"])
            all_matches.append(top_df)

        final_df = pd.concat(all_matches, ignore_index=True)
        logger.info(f"Candidate pool prepared: {len(final_df)} rows")

        return final_df

    except Exception as e:
        logger.error(f"Failed to prepare candidate pool: {e}")
        return pd.DataFrame()


# -----------------------------------------
# Standalone test mode
# -----------------------------------------
if __name__ == "__main__":
    from production.config_loader import load_config
    config = load_config()

    df = prepare_candidate_pool(config)
    print(df.head(10))
