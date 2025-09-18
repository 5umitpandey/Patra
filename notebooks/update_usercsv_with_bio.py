import os
import pandas as pd

# Paths
csv_path = r"D:\SM\Projects\Patra_ML\data\profiles\users_with_bios.csv"
bios_folder = r"D:\SM\Projects\Patra_ML\data\text\bios"

# Load users.csv
df = pd.read_csv(csv_path)

# Dictionary to hold bios
bios = {}

# Read each .txt file and map to user_id
for file in os.listdir(bios_folder):
    if file.endswith(".txt"):
        user_id = file.replace(".txt", "")  # e.g. "user_0001@thapar.edu"
        with open(os.path.join(bios_folder, file), "r", encoding="utf-8") as f:
            bios[user_id] = f.read().strip()  # Remove extra spaces/newlines

# Add new column 'bio' by mapping user_id
df["bio"] = df["user_id"].map(bios)

# Save new CSV
output_path = r"D:\SM\Projects\Patra_ML\data\profiles\users_with_bios__.csv"
df.to_csv(output_path, index=False)

print(f"✅ New CSV saved at: {output_path}")
