import json
import os

from tqdm import tqdm
from sklearn.model_selection import train_test_split

test_file_names = [x.strip() for x in open("../data/test_file_name.txt").readlines()]

main_folder_path = ("/media/dhanachandra/New Volume/HCN_from_2022/2025/Confidence_detection_for_medical_code/Confidence_detection_for_medical_code/org/data/Sentara_UI_JSON-20250915T130259Z-1-001/Sentara_UI_JSON")

folder_name_2_filepath = {}
for batch_name in os.listdir(main_folder_path):
    batch_dir_path = os.path.join(main_folder_path, batch_name)
    # Skip if not a directory
    if not os.path.isdir(batch_dir_path):
        continue
    # print(batch_name)
    for folder_name in os.listdir(batch_dir_path):
        if folder_name in test_file_names:
            continue
        doc_folder_path = os.path.join(batch_dir_path, folder_name)
        if not os.path.isdir(doc_folder_path):
            continue
        file_path = os.path.join(doc_folder_path, "final_merged_result.json")
        if os.path.exists(file_path):
            folder_name_2_filepath[folder_name] = file_path

# Extract document names and paths as lists
doc_names = list(folder_name_2_filepath.keys())
file_paths = list(folder_name_2_filepath.values())

# Split the data
train_names, val_names, train_paths, val_paths = train_test_split(
    doc_names, file_paths,
    test_size=0.2,
    random_state=2025,
    shuffle=True
)

print(len(train_names))
print(len(val_names))

import pandas as pd

# Prepare dataframe
train_df = pd.DataFrame({
    "file_name": train_names,
    "file_path": train_paths
})
val_df = pd.DataFrame({
    "file_name": val_names,
    "file_path": val_paths
})

# Save to CSV
train_df.to_csv("../data/train_split.csv", index=False)
val_df.to_csv("../data/val_split.csv", index=False)