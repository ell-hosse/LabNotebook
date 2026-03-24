from residual_adapter import ResidualAdapterFusionTextMain
import torch
import numpy as np
from pathlib import Path
import json

CAPTION_MMAP = Path(r"D:\hf\CoVLA-metadata\clip_text.f16.mmap")
CAPTION_META = Path(r"D:\hf\CoVLA-metadata\clip_text_meta.json")
VIDEO_MMAP = Path(r"D:\hf\CoVLA-metadata\clip_vision.f16.mmap")
VIDEO_META = Path(r"D:\hf\CoVLA-metadata\clip_vision_meta.json")

with CAPTION_META.open("r", encoding="utf-8") as f:
    meta = json.load(f)

N, T, D = meta["shape"]
assert T == 60 and D == 512, f"Unexpected shape: {(N, T, D)}"

emb_captions = np.memmap(CAPTION_MMAP, dtype=np.float16, mode="r", shape=(N, T, D))
emb_videos = np.memmap(VIDEO_MMAP, dtype=np.float16, mode="r", shape=(N, T, D))

emb_captions = torch.from_numpy(np.asarray(emb_captions)).float()
emb_videos = torch.from_numpy(np.asarray(emb_videos)).float()

fusion = ResidualAdapterFusionTextMain(embed_dim=512, adapter_dim=128)
fused = fusion(emb_videos, emb_captions)

print(fused.shape)