import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load user data
users = pd.read_csv(r"D:\SM\Projects\Patra_ML\data\profiles\users.csv")

# Add an index column so we can map bios
users.reset_index(inplace=True)

# Vectorize bios
vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
bio_embeddings = vectorizer.fit_transform(users["bio"])

def jaccard_score(list1, list2):
    s1, s2 = set(list1), set(list2)
    return len(s1 & s2) / len(s1 | s2) if s1 | s2 else 0

def match_score(u1, u2):
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
    hobby_score = jaccard_score(u1.hobbies.split(";"), u2.hobbies.split(";"))
    
    # Looking-for score
    lf_score = jaccard_score(u1.looking_for.split(";"), u2.looking_for.split(";"))
    
    # Bio score
    bio_score = cosine_similarity(
        bio_embeddings[u1["index"]], bio_embeddings[u2["index"]]
    )[0][0]
    
    # Profession score
    prof_score = 1 if u1.profession == u2.profession else 0.5
    
    # Weighted sum
    final_score = (
        0.25 * age_score +
        0.15 * loc_score +
        0.25 * hobby_score +
        0.10 * lf_score +
        0.15 * bio_score +
        0.10 * prof_score
    )
    return final_score

def get_top_matches(user_index, top_n=10):
    u1 = users.iloc[user_index]
    scores = []

    for i, u2 in users.iterrows():
        if i == user_index:
            continue
        
        score = match_score(u1, u2)
        
        # Soft orientation boost: +20% score if same orientation
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
        top_matches = []

        # Take up to half from same-orientation
        top_matches.extend(same[:half])

        # Fill remaining slots from others, but maintain score order
        remaining_slots = top_n - len(top_matches)
        top_matches.extend(other[:remaining_slots])

        # Final sort of top_matches by original score
        top_matches = sorted(top_matches, key=lambda x: x[3], reverse=True)
        return top_matches
    else:
        return scores[:top_n]


if __name__ == "__main__":
    # Default: first user
    user_index = 0
    user = users.iloc[user_index]
    print(f"Finding matches for {user.user_id} ({user.name})...\n")
    
    top_matches = get_top_matches(user_index, top_n=10)
    for m in top_matches:
        print(f"{m[0]} ({m[1]}) [{m[2]}] → Match: {m[3]*100:.1f}%")
