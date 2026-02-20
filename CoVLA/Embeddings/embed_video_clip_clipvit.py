import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch

try:
    import cv2
except ImportError as e:
    raise ImportError(
        "opencv-python is required for video reading. Install with: pip install opencv-python"
    ) from e

try:
    from transformers import CLIPModel, CLIPProcessor
except ImportError as e:
    raise ImportError(
        "transformers is required. Install with: pip install transformers"
    ) from e


VIDEOS_DIR = Path(r"D:\hf\hub\datasets--turing-motors--CoVLA-Dataset\snapshots\0a6d39e41659903a26dde957744e70dbc360bb6d\videos")
MANIFEST_PATH = Path(r"D:\hf\CoVLA-metadata\manifest.jsonl")
CLIP_INDEX_PATH = Path(r"D:\hf\CoVLA-metadata\clip_index.jsonl")

OUT_VISION_MMAP = Path(r"D:\hf\CoVLA-metadata\clip_vision.f16.mmap")
OUT_META_JSON = Path(r"D:\hf\CoVLA-metadata\clip_vision_meta.json")

N_FRAMES_PER_CLIP = 600
DOWNSAMPLED_FRAMES = 60
FRAME_STEP = N_FRAMES_PER_CLIP // DOWNSAMPLED_FRAMES  # 10

CLIP_NAME = "openai/clip-vit-base-patch16"  # ViT-B/16, outputs 512-d image features

BATCH_SIZE = 32
USE_FP16 = True  # embeddings saved as float16 anyway

VIDEO_EXTS = [".mp4", ".webm", ".mkv", ".avi", ".mov"]


def get_downsampled_frame_indices() -> List[int]:
    step = N_FRAMES_PER_CLIP // DOWNSAMPLED_FRAMES  # 600//60 = 10
    return list(range(0, N_FRAMES_PER_CLIP, step))[:DOWNSAMPLED_FRAMES]  # 0..590


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
    # sanity: rows should be 0..N-1
    for i, (_, r) in enumerate(pairs):
        if r != i:
            raise ValueError(f"clip_index rows not contiguous at i={i}, found row={r}")
    return pairs


def load_manifest_video_paths(manifest_path: Path) -> Dict[str, str]:
    if not manifest_path.exists():
        return {}
    out = {}
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            vid = rec.get("video_id")
            vpath = rec.get("video_path")
            if vid and vpath:
                out[vid] = vpath
    return out


def find_video_path(video_id: str, manifest_map: Dict[str, str]) -> Optional[Path]:
    if video_id in manifest_map:
        p = Path(manifest_map[video_id])
        if p.exists():
            return p

    for ext in VIDEO_EXTS:
        candidate = VIDEOS_DIR / f"{video_id}{ext}"
        if candidate.exists():
            return candidate

    return None


def map_indices_to_actual(total_frames: int, idx_600: List[int]) -> List[int]:
    if total_frames <= 0:
        return idx_600
    if total_frames == N_FRAMES_PER_CLIP:
        return idx_600
    # scale 0..599 -> 0..(total_frames-1)
    mapped = []
    for i in idx_600:
        j = int(round(i * (total_frames - 1) / (N_FRAMES_PER_CLIP - 1)))
        mapped.append(max(0, min(total_frames - 1, j)))
    return mapped


def read_frames_opencv(video_path: Path, frame_numbers: List[int]) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames = []
    for fn in frame_numbers:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fn))
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()
    return frames


def main():
    assert CLIP_INDEX_PATH.exists(), f"clip_index not found: {CLIP_INDEX_PATH}"
    assert VIDEOS_DIR.exists(), f"VIDEOS_DIR not found: {VIDEOS_DIR}"

    clip_pairs = load_clip_index(CLIP_INDEX_PATH)  # [(video_id,row)] sorted by row
    N = len(clip_pairs)
    print(f"Clips in index: {N}")

    manifest_map = load_manifest_video_paths(MANIFEST_PATH)

    # Prepare output memmap
    OUT_VISION_MMAP.parent.mkdir(parents=True, exist_ok=True)
    # shape [N, 60, 512]
    vision = np.memmap(
        OUT_VISION_MMAP, dtype=np.float16, mode="w+",
        shape=(N, DOWNSAMPLED_FRAMES, 512)
    )

    # Load CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    processor = CLIPProcessor.from_pretrained(CLIP_NAME)
    model = CLIPModel.from_pretrained(CLIP_NAME, use_safetensors=True).to(device)

    model.eval()

    # Use autocast fp16 on CUDA if desired
    use_amp = (device == "cuda" and USE_FP16)

    idx_600 = get_downsampled_frame_indices()  # [0,10,...,590]

    missing_videos = []
    processed = 0

    with torch.no_grad():
        for video_id, row in clip_pairs:
            vpath = find_video_path(video_id, manifest_map)
            if vpath is None:
                missing_videos.append(video_id)
                # fill zeros to keep alignment
                vision[row, :, :] = 0
                continue

            # get actual frame count
            cap = cv2.VideoCapture(str(vpath))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            mapped_frames = map_indices_to_actual(total_frames, idx_600)

            # read frames
            frames_rgb = read_frames_opencv(vpath, mapped_frames)

            # batched CLIP encoding
            feats_list = []
            for b in range(0, len(frames_rgb), BATCH_SIZE):
                batch_imgs = frames_rgb[b:b + BATCH_SIZE]
                inputs = processor(images=batch_imgs, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(device)

                if use_amp:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        img_feats = model.get_image_features(pixel_values=pixel_values)
                else:
                    img_feats = model.get_image_features(pixel_values=pixel_values)

                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)

                feats_list.append(img_feats.detach().cpu())

            feats = torch.cat(feats_list, dim=0).numpy().astype(np.float16)  # [60,512]
            if feats.shape != (DOWNSAMPLED_FRAMES, 512):
                # safety: force shape if something weird happened
                tmp = np.zeros((DOWNSAMPLED_FRAMES, 512), dtype=np.float16)
                m = min(tmp.shape[0], feats.shape[0])
                tmp[:m] = feats[:m]
                feats = tmp

            vision[row, :, :] = feats
            processed += 1

            if processed % 25 == 0:
                vision.flush()
                print(f"Processed {processed}/{N} videos")

    vision.flush()

    meta = {
        "clip_index": str(CLIP_INDEX_PATH),
        "manifest": str(MANIFEST_PATH),
        "videos_dir": str(VIDEOS_DIR),
        "clip_model": CLIP_NAME,
        "dtype": "float16",
        "shape": [N, DOWNSAMPLED_FRAMES, 512],
        "frame_indices_600": idx_600,
        "note": "frame_indices_600 are mapped to actual video frame numbers if CAP_PROP_FRAME_COUNT != 600",
        "missing_videos_count": len(missing_videos),
        "missing_videos_example": missing_videos[:20],
    }
    with OUT_META_JSON.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print("Vision embeddings:", OUT_VISION_MMAP)
    print("Meta:", OUT_META_JSON)
    print("Missing videos:", len(missing_videos))


if __name__ == "__main__":
    main()
