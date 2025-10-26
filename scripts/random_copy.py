#!/usr/bin/env python3

import os
import random
import shutil

# source and destination directories
src_dir = "evidence_cache"
dst_dir = "evidence_cache_small"

# make sure destination exists
os.makedirs(dst_dir, exist_ok=True)

# list all files (ignore subfolders)
files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]

# pick 15 random files
selected = random.sample(files, 15)

# copy them
for f in selected:
    shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_dir, f))

print(f"Copied {len(selected)} files successfully!")
