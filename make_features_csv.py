import cv2
import pandas as pd
import os

BASE_DIR = "."          # MLOPS folder
SPLITS = ["train", "valid", "test"]

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

def extract_features_from_split(split_name):
    split_dir = os.path.join(BASE_DIR, split_name)

    if not os.path.exists(split_dir):
        print(f"❌ Folder not found: {split_dir}")
        return None

    rows = []

    for file_name in os.listdir(split_dir):
        if not file_name.lower().endswith(IMAGE_EXTENSIONS):
            continue  # skip CSV and other files

        img_path = os.path.join(split_dir, file_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        rows.append({
            "image_name": file_name,
            "brightness_mean": float(gray.mean()),
            "contrast": float(gray.std()),
            "image_width": w,
            "image_height": h,
            "split": split_name
        })

    return pd.DataFrame(rows)


all_dfs = []

for split in SPLITS:
    print(f"📊 Processing {split} images...")
    df_split = extract_features_from_split(split)

    if df_split is not None and not df_split.empty:
        df_split.to_csv(f"{split}_features.csv", index=False)
        all_dfs.append(df_split)
        print(f"✅ {split}_features.csv created")
    else:
        print(f"⚠️ No images found in {split}")

# Combine all splits
if all_dfs:
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df.to_csv("features_all.csv", index=False)
    print("✅ features_all.csv created")
