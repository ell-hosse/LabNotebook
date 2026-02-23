import json
from pathlib import Path
import numpy as np

CAPTION_MMAP = Path(r"D:\hf\CoVLA-metadata\clip_text.f16.mmap")
CAPTION_META = Path(r"D:\hf\CoVLA-metadata\clip_text_meta.json")
CLIP_INDEX_PATH = Path(r"D:\hf\CoVLA-metadata\clip_index.jsonl")


def load_row_to_video_id(clip_index_path: Path) -> dict[int, str]:
    row_to_vid = {}
    with clip_index_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            row = int(rec["row"])
            row_to_vid[row] = rec["video_id"]
    return row_to_vid


def main():
    assert CAPTION_MMAP.exists(), f"Missing: {CAPTION_MMAP}"
    assert CAPTION_META.exists(), f"Missing: {CAPTION_META}"
    assert CLIP_INDEX_PATH.exists(), f"Missing: {CLIP_INDEX_PATH}"

    with CAPTION_META.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    N, T, D = meta["shape"]  # expected [N,60,512]
    assert T == 60 and D == 512, f"Unexpected shape: {(N, T, D)}"

    captions = np.memmap(CAPTION_MMAP, dtype=np.float16, mode="r", shape=(N, T, D))
    row_to_vid = load_row_to_video_id(CLIP_INDEX_PATH)

    variation = np.zeros((N,), dtype=np.float32)
    zero_rows = []

    for i in range(N):
        clip = captions[i].astype(np.float32)  # [60,512]

        # Detect all-zeros (missing captions filled with zeros)
        if np.all(clip == 0):
            variation[i] = np.nan
            zero_rows.append(i)
            continue

        # Because these were L2-normalized when saved:
        # cosine(f_t, f_{t+1}) = dot product
        cos = np.sum(clip[:-1] * clip[1:], axis=1)  # [59]
        variation[i] = float(np.mean(1.0 - cos))

    valid_mask = ~np.isnan(variation)
    valid_idx = np.where(valid_mask)[0]

    if len(valid_idx) == 0:
        print("No valid rows found (all NaN). Check if the caption memmap is all zeros.")
        return

    # Top 20 highest / lowest among valid
    top20 = valid_idx[np.argsort(-variation[valid_idx])[:20]]
    bot20 = valid_idx[np.argsort(variation[valid_idx])[:20]]

    print("\n==============================")
    print("Caption temporal variation")
    print("==============================")
    print(f"Total clips: {N}")
    print(f"Valid clips: {len(valid_idx)}")
    print(f"All-zero (missing) clips: {len(zero_rows)}")

    v = variation[valid_idx]
    print(f"\nStats (valid only): min={v.min():.6f}  mean={v.mean():.6f}  max={v.max():.6f}")

    print("\n--- Top 20 HIGHEST variation ---")
    for rank, i in enumerate(top20, 1):
        vid = row_to_vid.get(int(i), "UNKNOWN")
        print(f"{rank:2d}. row={int(i):4d}  video_id={vid}  variation={variation[i]:.6f}")

    print("\n--- Top 20 LOWEST variation ---")
    for rank, i in enumerate(bot20, 1):
        vid = row_to_vid.get(int(i), "UNKNOWN")
        print(f"{rank:2d}. row={int(i):4d}  video_id={vid}  variation={variation[i]:.6f}")


if __name__ == "__main__":
    main()