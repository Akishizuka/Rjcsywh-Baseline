import os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def prepare_mydata(csv_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    df = df.sort_values("timestamp").reset_index(drop=True)

    feature_cols = [
        c for c in df.columns if c not in ["timestamp", "fault_type", "target_service"]
    ]
    labels = (df["fault_type"] != "normal").astype(int).to_numpy()

    train_df = df[df["fault_type"] == "normal"].copy()
    train_x = train_df[feature_cols].to_numpy(dtype=np.float32)
    test_x = df[feature_cols].to_numpy(dtype=np.float32)
    test_y = labels.astype(np.int32)

    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x).astype(np.float32)
    test_x = scaler.transform(test_x).astype(np.float32)

    train_path = os.path.join(out_dir, "mydata_train.npy")
    test_path = os.path.join(out_dir, "mydata_test.npy")
    label_path = os.path.join(out_dir, "mydata_test_label.npy")

    np.save(train_path, train_x)
    np.save(test_path, test_x)
    np.save(label_path, test_y)

    print(f"Saved: {train_path}")
    print(f"Saved: {test_path}")
    print(f"Saved: {label_path}")
    print(f"train shape: {train_x.shape}")
    print(f"test shape:  {test_x.shape}")
    print(f"label shape: {test_y.shape}")
    print(f"feature count: {len(feature_cols)}")
    print(f"anomaly ratio: {test_y.mean():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path",
        default=os.path.join("data", "mydata", "final_dataset_for_algorithm.csv"),
        type=str,
    )
    parser.add_argument(
        "--out_dir",
        default=os.path.join("data", "mydata"),
        type=str,
    )
    args = parser.parse_args()
    prepare_mydata(args.csv_path, args.out_dir)
