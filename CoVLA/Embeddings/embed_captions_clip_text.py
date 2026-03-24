import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch

try:
    from transformers import CLIPModel, CLIPProcessor
except ImportError as e:
    raise ImportError("transformers is required. Install with: pip install transformers") from e


CLIP_INDEX_PATH = Path(r"D:\hf\CoVLA-metadata\clip_index.jsonl")

CAPTIONS_PATH = Path(r"D:\hf\CoVLA-metadata\captions10_per_video.jsonl")

OUT_TEXT_MMAP = Path(r"D:\hf\CoVLA-metadata\clip_text.f16.mmap")
OUT_TEXT_META_JSON = Path(r"D:\hf\CoVLA-metadata\clip_text_meta.json")

N_FRAMES_PER_CLIP = 600
DOWNSAMPLED_FRAMES = 60
FRAME_STEP = N_FRAMES_PER_CLIP // DOWNSAMPLED_FRAMES # 10

CLIP_NAME = "openai/clip-vit-base-patch16" # text embedding dim=512

TEXT_BATCH_SIZE = 64
USE_FP16 = True


def get_downsampled_frame_indices() -> List[int]:
    return list(range(0, N_FRAMES_PER_CLIP, FRAME_STEP))[:DOWNSAMPLED_FRAMES]


def load_clip_index(index_path: Path) -> List[Tuple[str, int]]:
    assert index_path.exists(), f"clip_index not found: {index_path}"
    pairs = []
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            vid = rec["video_id"]
            row = int(rec["row"])
            pairs.append((vid, row))
    pairs.sort(key=lambda x: x[1])

    for i, (_, r) in enumerate(pairs):
        if r != i:
            raise ValueError(f"clip_index rows not contiguous at i={i}, found row={r}")
    print(f"Found {len(pairs)} clips", pairs[0])
    return pairs
    return pairs


def _extract_captions_list(rec: dict) -> Optional[List[str]]:
    # Most likely:
    if isinstance(rec.get("captions"), list):
        caps = rec["captions"]
        # ensure strings
        caps = [str(c) for c in caps]
        return caps

    # Sometimes: "caption" as list or dict
    if isinstance(rec.get("caption"), list):
        return [str(c) for c in rec["caption"]]


    # Sometimes: caption_0..caption_9
    caps = []
    for i in range(10):
        k = f"caption_{i}"
        if k in rec:
            caps.append(str(rec[k]))
    if len(caps) > 0:
        return caps

    # Sometimes: captions10
    if isinstance(rec.get("captions10"), list):
        return [str(c) for c in rec["captions10"]]

    return None


def load_captions_map(captions_path: Path) -> Dict[str, List[str]]:
    """
    captions10_per_video.jsonl format (your sample):
      {"caption_id": "...", "caption_idx": 0..9, "rich_caption": "..."}
    Returns:
      caption_id -> list[str] length 10 aligned by caption_idx
    """
    assert captions_path.exists(), f"captions file not found: {captions_path}"

    temp: Dict[str, List[Optional[str]]] = {}

    with captions_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)

            vid = rec.get("caption_id")  # <-- key fix
            idx = rec.get("caption_idx")
            cap = rec.get("rich_caption") or rec.get("caption")

            if vid is None or idx is None or cap is None:
                continue

            vid = str(vid)
            idx = int(idx)
            if not (0 <= idx < 10):
                continue

            if vid not in temp:
                temp[vid] = [None] * 10

            temp[vid][idx] = str(cap)

    # finalize: fill missing idx values by carrying forward last available (or empty)
    out: Dict[str, List[str]] = {}
    for vid, caps in temp.items():
        fixed: List[str] = []
        last = ""
        for c in caps:
            if c is None:
                fixed.append(last)   # carry forward
            else:
                fixed.append(c)
                last = c
        out[vid] = fixed

    return out

def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def main():
    clip_pairs = load_clip_index(CLIP_INDEX_PATH)
    N = len(clip_pairs)
    print(f"Clips in index: {N}")

    captions_map = load_captions_map(CAPTIONS_PATH)
    print(f"Loaded captions for {len(captions_map)} videos")

    # Prepare output memmap: [N, 60, 512]
    OUT_TEXT_MMAP.parent.mkdir(parents=True, exist_ok=True)
    text_mmap = np.memmap(
        OUT_TEXT_MMAP, dtype=np.float16, mode="w+",
        shape=(N, DOWNSAMPLED_FRAMES, 512)
    )

    # Load CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    processor = CLIPProcessor.from_pretrained(CLIP_NAME)
    model = CLIPModel.from_pretrained(CLIP_NAME, use_safetensors=True).to(device)
    model.eval()

    use_amp = (device == "cuda" and USE_FP16)

    idx_600 = get_downsampled_frame_indices()  # [0,10,...,590]
    # per sampled frame time-step, which caption index (0..9)
    caption_ids = [i // 60 for i in idx_600]   # length 60, values 0..9

    missing_caption_videos = []
    processed = 0

    with torch.no_grad():
        for video_id, row in clip_pairs:
            caps = captions_map.get(video_id)

            if not caps:
                # no captions found => fill zeros to keep alignment
                missing_caption_videos.append(video_id)
                text_mmap[row, :, :] = 0
                continue

            # Ensure we have at least 10 captions (pad/truncate)
            # Because your rule assumes caption_idx in 0..9
            if len(caps) < 10:
                # pad by repeating last caption (or empty)
                last = caps[-1] if len(caps) > 0 else ""
                caps = caps + [last] * (10 - len(caps))
            elif len(caps) > 10:
                caps = caps[:10]

            # Encode the 10 captions (one batch)
            # get_text_features returns [B, 512] (no .pooler_output)
            inputs = processor(text=caps, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            if use_amp:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    txt_feats = model.get_text_features(input_ids=input_ids, attention_mask=attention_mask).pooler_output
            else:
                txt_feats = model.get_text_features(input_ids=input_ids, attention_mask=attention_mask).pooler_output

            txt_feats = l2_normalize(txt_feats).detach().cpu().numpy().astype(np.float16)  # [10,512]

            # Expand to per-frame (60) using caption_ids (idx_600//60)
            # Each time step t picks one of the 10 caption embeddings
            per_frame = txt_feats[np.array(caption_ids, dtype=np.int64)]  # [60,512]

            text_mmap[row, :, :] = per_frame
            processed += 1

            if processed % 100 == 0:
                text_mmap.flush()
                print(f"Processed {processed}/{N} caption-videos")

    text_mmap.flush()

    meta = {
        "clip_index": str(CLIP_INDEX_PATH),
        "captions_path": str(CAPTIONS_PATH),
        "clip_model": CLIP_NAME,
        "dtype": "float16",
        "shape": [N, DOWNSAMPLED_FRAMES, 512],
        "frame_indices_600": idx_600,
        "caption_index_rule": "caption_idx = frame_index // 60 (frame_index in 0..599 space)",
        "caption_ids_for_downsampled_frames": caption_ids,
        "missing_caption_videos_count": len(missing_caption_videos),
        "missing_caption_videos_example": missing_caption_videos[:20],
        "note": "Saved per-frame caption embeddings aligned with clip_index row order and 60 downsampled time steps.",
    }
    with OUT_TEXT_META_JSON.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print("Text embeddings:", OUT_TEXT_MMAP)
    print("Meta:", OUT_TEXT_META_JSON)
    print("Missing caption videos:", len(missing_caption_videos))


if __name__ == "__main__":
    main()