import os
import pandas as pd

# Paths
csv_path = r"D:\SM\Projects\Patra_ML\data\profiles\users.csv"
img_folder = r"D:\SM\Projects\Patra_ML\data\images\assigned"

# Load CSV
df = pd.read_csv(csv_path)

# Get user_ids (first column)
user_ids = df.iloc[:, 0].tolist()   # ['user_0001@thapar.edu', ...]

# List all images (sorted to match order)
images = sorted(os.listdir(img_folder))

# Rename one by one
for old_name, new_id in zip(images, user_ids):
    old_path = os.path.join(img_folder, old_name)
    
    # Keep extension
    ext = os.path.splitext(old_name)[1]
    new_name = new_id + ext
    new_path = os.path.join(img_folder, new_name)

    os.rename(old_path, new_path)
    print(f"Renamed: {old_name} -> {new_name}")

print("✅ All images renamed successfully!")
