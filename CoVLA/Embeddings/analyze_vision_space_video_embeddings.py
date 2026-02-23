import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

MMAP_PATH = Path(r"D:\hf\CoVLA-metadata\clip_vision.f16.mmap")
META_PATH = Path(r"D:\hf\CoVLA-metadata\clip_vision_meta.json")

def l2norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, eps, None)

# load memmap
with META_PATH.open("r", encoding="utf-8") as f:
    meta = json.load(f)

N, T, D = meta["shape"]
vision = np.memmap(MMAP_PATH, dtype=np.float16, mode="r", shape=(N, T, D))

# pool: [N, 60, 512] -> [N, 512]
X = vision.astype(np.float32).mean(axis=1) # mean over frames
X = l2norm(X) # normalize pooled vectors

print("Pooled X:", X.shape, X.dtype)

# PCA (SVD-based)
Xc = X - X.mean(axis=0, keepdims=True)
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
Z2 = Xc @ Vt[:2].T # [N,2]

print("Explained variance ratio (approx):",
      (S[:2]**2 / np.sum(S**2)).tolist())

plt.figure()
plt.scatter(Z2[:, 0], Z2[:, 1], s=8)
plt.title("PCA of clip-level CLIP embeddings (mean pooled)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

import matplotlib.pyplot as plt
import numpy as np

sim = X @ X.T
plt.figure()
plt.imshow(sim, aspect="auto")
plt.title("Cosine similarity matrix (clip-level)")
plt.colorbar()
plt.show()

# detect all-zero clips
zero_mask = np.all(vision == 0, axis=(1,2))
print("Zero clips:", int(zero_mask.sum()))

# detect low temporal variation
# if a clip's frames are nearly identical, std over time will be tiny
clip_std = vision.astype(np.float32).std(axis=1).mean(axis=1)  # [N]
bad = np.argsort(clip_std)[:20]
print("Lowest temporal-variation clips:", bad.tolist())
print("Their std values:", clip_std[bad].tolist())

import json
import numpy as np
from pathlib import Path

MMAP_PATH = Path(r"D:\hf\CoVLA-metadata\clip_vision.f16.mmap")
META_PATH = Path(r"D:\hf\CoVLA-metadata\clip_vision_meta.json")
CLIP_INDEX_PATH = Path(r"D:\hf\CoVLA-metadata\clip_index.jsonl")

# Load metadata
with META_PATH.open("r", encoding="utf-8") as f:
    meta = json.load(f)

N, T, D = meta["shape"]
vision = np.memmap(MMAP_PATH, dtype=np.float16, mode="r", shape=(N, T, D))

# Load clip_index to map row -> video_id
row_to_vid = {}
with CLIP_INDEX_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        row_to_vid[int(rec["row"])] = rec["video_id"]

# Compute temporal variation
temporal_variation = []

for i in range(N):
    clip = vision[i].astype(np.float32) # [60,512]

    # cosine between consecutive frames
    cos = np.sum(clip[:-1] * clip[1:], axis=1) # shape [59]
    variation = np.mean(1.0 - cos)

    temporal_variation.append(variation)

temporal_variation = np.array(temporal_variation)

# Top 20 highest
top20_idx = np.argsort(-temporal_variation)[:20]

print("\nTop 20 highest temporal variation clips:\n")
for rank, idx in enumerate(top20_idx, 1):
    print(
        f"{rank:2d}. Row={idx:4d} | "
        f"VideoID={row_to_vid.get(idx)} | "
        f"Variation={temporal_variation[idx]:.6f}"
    )