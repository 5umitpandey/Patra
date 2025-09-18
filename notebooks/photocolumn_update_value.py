import pandas as pd

# Path to CSV
csv_path = r"D:\SM\Projects\Patra_ML\data\profiles\users.csv"

# Load CSV
df = pd.read_csv(csv_path)

# Update photo_path column
df["photo_path"] = df["photo_path"].str.replace(
    r"(user_\d+)(\.jpg|\.png|\.jpeg)", r"\1@thapar.edu\2", regex=True
)

# Save updated CSV
output_path = r"D:\SM\Projects\Patra_ML\data\profiles\users_updated.csv"
df.to_csv(output_path, index=False)

print(f"✅ Updated CSV saved at: {output_path}")
