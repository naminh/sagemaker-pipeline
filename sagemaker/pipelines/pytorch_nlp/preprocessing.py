"""Script used by Data Processing Step."""

import pandas as pd


if __name__ == "__main__":
    base_dir = "/opt/ml/processing"

    train_df = pd.read_csv(
        f"{base_dir}/input/train.csv",
    )

    train_df = train_df.head(100)
    train_df["text"] = train_df["text"].astype(str)
    train_df["selected_text"] = train_df["selected_text"].astype(str)

    pd.DataFrame(train_df).to_csv(f"{base_dir}/train/train.csv", index=False)
