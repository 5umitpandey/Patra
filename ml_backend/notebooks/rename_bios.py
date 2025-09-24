import os

# Path to folder
folder = r"D:\SM\Projects\Patra_ML\data\text\bios"

# List all files
files = sorted(os.listdir(folder))

# Rename each file
for old_name in files:
    if old_name.endswith(".txt") and old_name.startswith("user_"):
        old_path = os.path.join(folder, old_name)

        # Remove ".txt" then add "@thapar.edu.txt"
        base = os.path.splitext(old_name)[0]  # e.g. user_0001
        new_name = base + "@thapar.edu.txt"
        new_path = os.path.join(folder, new_name)

        os.rename(old_path, new_path)
        print(f"Renamed: {old_name} -> {new_name}")

print("✅ All bios renamed successfully!")