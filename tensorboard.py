import pandas as pd
from torch.utils.tensorboard import SummaryWriter

# Load features
df = pd.read_csv("features_all.csv")

writer = SummaryWriter(log_dir="runs/feature_monitoring")

step = 0

for split in ["train", "valid", "test"]:
    split_df = df[df["split"] == split]

    if split_df.empty:
        continue

    # ✅ Convert Pandas Series → NumPy arrays
    writer.add_histogram(
        f"{split}/brightness_mean",
        split_df["brightness_mean"].to_numpy(),
        step
    )

    writer.add_histogram(
        f"{split}/contrast",
        split_df["contrast"].to_numpy(),
        step
    )

    writer.add_histogram(
        f"{split}/image_width",
        split_df["image_width"].to_numpy(),
        step
    )

    writer.add_histogram(
        f"{split}/image_height",
        split_df["image_height"].to_numpy(),
        step
    )

    # Scalars (mean values)
    writer.add_scalar(
        f"{split}/brightness_mean_avg",
        split_df["brightness_mean"].mean(),
        step
    )

    writer.add_scalar(
        f"{split}/contrast_avg",
        split_df["contrast"].mean(),
        step
    )

    step += 1

writer.close()
print("✅ TensorBoard logs created successfully")
