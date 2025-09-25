import os
import shutil
import random

# Source and target folders
src_folder = r"LPRNet_Pytorch\data\test2"
dst_parent = r"LPRNet_Pytorch\data\split"

# Create destination subfolders
train_folder = os.path.join(dst_parent, "train")
val_folder = os.path.join(dst_parent, "val")
test_folder = os.path.join(dst_parent, "test")

for folder in [train_folder, val_folder, test_folder]:
    os.makedirs(folder, exist_ok=True)

# Collect all images
files = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]

# Shuffle to randomize split
random.shuffle(files)

# Split
train_files = files[:800]
val_files = files[800:900]
test_files = files[900:1000]

# Helper function
def move_files(file_list, target_folder):
    for f in file_list:
        src_path = os.path.join(src_folder, f)
        dst_path = os.path.join(target_folder, f)
        shutil.copy(src_path, dst_path)  # use copy to keep original; use move if you want them removed
    print(f"Copied {len(file_list)} files to {target_folder}")

# Move files
move_files(train_files, train_folder)
move_files(val_files, val_folder)
move_files(test_files, test_folder)
