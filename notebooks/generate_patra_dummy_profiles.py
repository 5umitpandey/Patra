import os
import random
import shutil
import csv
from datetime import datetime, timedelta
from pathlib import Path

# ---------- CONFIG ----------
ROOT = Path(r"D:\SM\Projects\Patra_ML")
SRC_IMAGES = ROOT / "data" / "images"
TARGET_IMAGES = ROOT / "data" / "images" / "assigned"
PROFILES_DIR = ROOT / "data" / "profiles"
BIOS_DIR = ROOT / "data" / "text" / "bios"

# Required counts per category
COUNTS = {
    "male": 300,
    "female": 300,
    "gay": 200,
    "lesbian": 200
}

# The "as-of" date to calculate ages
REFERENCE_DATE = datetime(2025, 9, 17)

# Lists for random generation (India-focused)
CITIES_STATES = [
    ("Mumbai","Maharashtra"), ("Delhi","Delhi"), ("Bengaluru","Karnataka"),
    ("Chennai","Tamil Nadu"), ("Kolkata","West Bengal"), ("Hyderabad","Telangana"),
    ("Pune","Maharashtra"), ("Ahmedabad","Gujarat"), ("Surat","Gujarat"),
    ("Jaipur","Rajasthan"), ("Lucknow","Uttar Pradesh"), ("Kanpur","Uttar Pradesh"),
    ("Nagpur","Maharashtra"), ("Indore","Madhya Pradesh"), ("Bhopal","Madhya Pradesh"),
    ("Vishakhapatnam","Andhra Pradesh"), ("Coimbatore","Tamil Nadu"), ("Patna","Bihar"),
    ("Vadodara","Gujarat"), ("Ludhiana","Punjab"), ("Agra","Uttar Pradesh"),
    ("Nashik","Maharashtra"), ("Faridabad","Haryana"), ("Meerut","Uttar Pradesh"),
    ("Rajkot","Gujarat"), ("Kalyan","Maharashtra"), ("Vijayawada","Andhra Pradesh"),
    ("Madurai","Tamil Nadu"), ("Gwalior","Madhya Pradesh"), ("Srinagar","Jammu & Kashmir")
]

PROFESSIONS = ["Student", "Working Professional"]
LOOKING_FOR_CHOICES = ["friendship", "relationship", "fun", "casual"]
HOBBIES_LIST = ["music","sports","reading","travel","art","gaming","cooking","dancing",
                "movies","fitness","tech","pets","photography","fashion","outdoor",
                "writing","yoga","games"]

# Small name lists (extend if you like)
MALE_NAMES = ["Arjun","Rohan","Vikram","Sahil","Amit","Karan","Rahul","Aakash","Nikhil","Manish","Simardeep", "Ashir", "Deepak","Saurav","Ravi","Tanmay","Aditya","Harsh","Yash","Aniket","Siddharth","Pranav","Vivek","Sameer","Kunal","Naveen","Ishaan","Gautam","Abhishek","Rishabh","Shivam","Kartik","Rajat","Arnav","Tejas","Lokesh","Mayank","Bhavesh","Hemant","Dhruv","Om","Nitin","Sanjay","Puneet","Mukesh","Aman","Ritesh","Suyash","Shaurya","Ishan"]
FEMALE_NAMES = ["Aisha","Priya","Ananya","Sneha","Pooja","Ritu","Neha","Sakshi","Divya","Kavya","Simran","Riya","Nisha","Shruti","Anjali","Meera","Shreya","Kriti","Jyoti","Tanvi","Isha","Maya","Bhavya","Charu","Sonia","Namrata","Pallavi","Swati","Kiran","Nandini","Yamini","Smita","Rachna","Gauri","Aparna","Tanya","Esha","Zara","Rekha","Vidya", "Lisa"]

# Bio templates (each will produce up to ~2 short lines, keep <100 chars)
BIO_TEMPLATES = [
    "Explorer at heart - coffee, books and weekend hikes.\nAlways learning, always curious.",
    "Designer by day, amateur chef by night.\nLooking for good conversations and good food.",
    "Travel junkie who loves mountains and street food.\nLet's find the next hidden gem together.",
    "Tech enthusiast, fitness lover and a movie buff.\nWill trade playlists for travel stories.",
    "Bookworm who enjoys spontaneous trips.\nAsk me about the best book I read this year.",
    "Music, photography and late-night coding sessions.\nI care about kindness and curiosity.",
    "Yoga in the morning, coffee in the evening.\nLet's explore the city together.",
    "Food lover with a soft spot for home-cooked meals.\nOpen to new adventures and new friends.",
    "Runner, pet-lover and aspiring home-chef.\nLooking for meaningful connections.",
    "Artsy, outdoorsy, and always planning the next trip.\nCan beat you in a movie-quote duel."
]

# ----------------- FUNCTIONS -----------------
def ensure_dirs():
    for p in [TARGET_IMAGES, PROFILES_DIR, BIOS_DIR]:
        p.mkdir(parents=True, exist_ok=True)

def list_images_for_category(cat):
    folder = SRC_IMAGES / cat
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Image folder missing: {folder}")
    images = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in [".jpg",".jpeg",".png"]]
    return images

def random_dob(min_age=18, max_age=60, ref=REFERENCE_DATE):
    # pick an integer age
    age = random.randint(min_age, max_age)
    # pick a random day within that year range
    start_birth = ref.replace(year=ref.year - age) - timedelta(days=365)
    # to allow variation, offset by up to 365 days
    birth = start_birth + timedelta(days=random.randint(0, 365))
    return birth.date()

def calculate_age(dob, ref=REFERENCE_DATE):
    # dob is a date object
    born = dob
    refd = ref.date()
    years = refd.year - born.year - ((refd.month, refd.day) < (born.month, born.day))
    return years

def sample_multiselect(options, min_k=1, max_k=3):
    k = random.randint(min_k, min(max_k, len(options)))
    return sorted(random.sample(options, k))

# ----------------- MAIN -----------------
def main():
    random.seed(42)  # deterministic-ish
    ensure_dirs()

    # validate image availability
    images_pool = {}
    for cat, cnt in COUNTS.items():
        imgs = list_images_for_category(cat)
        if len(imgs) < cnt:
            raise RuntimeError(f"Need at least {cnt} images in {SRC_IMAGES / cat}, but found {len(imgs)}.")
        images_pool[cat] = imgs

    rows = []
    next_id = 1

    # process each category
    for cat, required in COUNTS.items():
        for i in range(required):
            user_id = f"user_{next_id:04d}"
            # choose a name
            if cat in ("male", "gay"):
                name = random.choice(MALE_NAMES)
            else:
                name = random.choice(FEMALE_NAMES)

            # DOB and age
            dob = random_dob(min_age=18, max_age=60)
            age = calculate_age(dob)

            # location
            city, state = random.choice(CITIES_STATES)

            # profession: younger are more likely students
            if age <= 23:
                profession = random.choices(PROFESSIONS, weights=[0.7, 0.3])[0]
            else:
                profession = random.choices(PROFESSIONS, weights=[0.15, 0.85])[0]

            # select looking_for and hobbies
            looking_for = sample_multiselect(LOOKING_FOR_CHOICES, 1, 3)
            hobbies = sample_multiselect(HOBBIES_LIST, 2, 5)

            # pick an image (and remove from pool to avoid reuse)
            img_path = images_pool[cat].pop(random.randrange(len(images_pool[cat])))
            # copy to target with deterministic name
            target_img_name = f"{user_id}{img_path.suffix.lower()}"
            target_img_path = TARGET_IMAGES / target_img_name
            shutil.copy(img_path, target_img_path)

            # bio (choose a template and slightly vary)
            bio_template = random.choice(BIO_TEMPLATES)
            # Make sure bio <= 100 chars; if longer, truncate safely
            bio = bio_template
            if len(bio) > 100:
                bio = bio[:97] + "..."

            # build row
            row = {
                "user_id": user_id,
                "name": name,
                "dob": dob.isoformat(),
                "age": age,
                "city": city,
                "state": state,
                "profession": profession,
                "photo_path": str(Path("data") / "images" / "assigned" / target_img_name),
                "gender": cat.capitalize(),   # Male/Female/Gay/Lesbian
                "bio": bio.replace("\n", " \\n "),  # keep linebreak marker but single CSV field
                "looking_for": ",".join(looking_for),
                "hobbies": ",".join(hobbies)
            }
            rows.append(row)

            # write bio file (2 lines)
            bio_lines = bio.split("\n")
            bios_text = "\n".join(bio_lines[:2])  # ensure max 2 lines
            with open(BIOS_DIR / f"{user_id}.txt", "w", encoding="utf-8") as bf:
                bf.write(bios_text)

            next_id += 1

    # write CSV
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = PROFILES_DIR / "users.csv"
    fieldnames = ["user_id","name","dob","age","city","state","profession","photo_path","gender","bio","looking_for","hobbies"]
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print("Done.")
    print(f"Created {len(rows)} users.")
    print(f"CSV: {csv_path}")
    print(f"Copied images to: {TARGET_IMAGES}")
    print(f"Bio files: {BIOS_DIR}")

if __name__ == "__main__":
    main()
