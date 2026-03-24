import torch
import torch.nn as nn
from pathlib import Path
import json
import numpy as np


class ResidualAdapterFusionTextMain(nn.Module):
    def __init__(self, embed_dim=512, adapter_dim=128, dropout=0.1):
        super().__init__()

        self.pre_norm = nn.LayerNorm(embed_dim * 2)

        self.down = nn.Linear(embed_dim * 2, adapter_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(adapter_dim, embed_dim)

        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )

        self.out_norm = nn.LayerNorm(embed_dim)


    def forward(self, frame_emb, text_emb):
        x = torch.cat([text_emb, frame_emb], dim=-1)
        x = self.pre_norm(x)

        delta = self.down(x)
        delta = self.act(delta)
        delta = self.dropout(delta)
        delta = self.up(delta)

        gate = self.gate(x)

        fused = text_emb + gate * delta
        fused = self.out_norm(fused)

        return fused

    def creat_fused_embeddings(self):
        CAPTION_MMAP = Path(r"D:\hf\CoVLA-metadata\clip_text.f16.mmap")
        CAPTION_META = Path(r"D:\hf\CoVLA-metadata\clip_text_meta.json")
        VIDEO_MMAP = Path(r"D:\hf\CoVLA-metadata\clip_vision.f16.mmap")

        with CAPTION_META.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        N, T, D = meta["shape"]
        assert T == 60 and D == 512, f"Unexpected shape: {(N, T, D)}"

        emb_captions = np.memmap(CAPTION_MMAP, dtype=np.float16, mode="r", shape=(N, T, D))
        emb_videos = np.memmap(VIDEO_MMAP, dtype=np.float16, mode="r", shape=(N, T, D))

        emb_captions = torch.from_numpy(np.asarray(emb_captions)).float()
        emb_videos = torch.from_numpy(np.asarray(emb_videos)).float()

        device = next(self.parameters()).device
        emb_captions = emb_captions.to(device)
        emb_videos = emb_videos.to(device)

        self.eval()
        with torch.no_grad():
            fused = self(emb_videos, emb_captions)

        return fused