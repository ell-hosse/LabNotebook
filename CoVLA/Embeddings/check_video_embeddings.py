import json
import numpy as np
from pathlib import Path

MMAP_PATH = Path(r"D:\hf\CoVLA-metadata\clip_vision.f16.mmap")
META_PATH = Path(r"D:\hf\CoVLA-metadata\clip_vision_meta.json")

with META_PATH.open("r", encoding="utf-8") as f:
    meta = json.load(f)

shape = tuple(meta["shape"]) # [N, 60, 512]
dtype = np.float16

vision = np.memmap(MMAP_PATH, dtype=dtype, mode="r", shape=shape)

print("Loaded memmap:")
print("Shape:", vision.shape)
print("Dtype:", vision.dtype)

clip_id = 0

clip_feats = vision[clip_id] # shape: [60, 512]

print("Clip 0 shape:", clip_feats.shape)
print("First frame feature (first 10 dims):")
print(clip_feats[0][:10])


import matplotlib.pyplot as plt

# Track cosine similarity between consecutive frames
cos_sims = []

for t in range(59):
    a = clip_feats[t].astype(np.float32)
    b = clip_feats[t+1].astype(np.float32)
    cos = np.dot(a, b)
    cos_sims.append(cos)

plt.plot(cos_sims)
plt.title("Consecutive frame cosine similarity")
plt.show()

print("Similarity between two videos =", np.dot(vision[0, 0], vision[100, 0]))

