import json
from pathlib import Path


MANIFEST_PATH = Path(r"D:\hf\CoVLA-metadata\manifest.jsonl")
OUT_INDEX_JSONL = Path(r"D:\hf\CoVLA-metadata\clip_index.jsonl")


def get_last_index(path: Path) -> int:
    if not path.exists():
        return -1  # means start from 0

    last_row = -1
    with open(path, 'r') as file:
        last_row = sum(1 for _ in file) - 1

    return last_row


def main():
    assert MANIFEST_PATH.exists(), f"Manifest not found: {MANIFEST_PATH}"
    OUT_INDEX_JSONL.parent.mkdir(parents=True, exist_ok=True)

    last_index = get_last_index(OUT_INDEX_JSONL)
    start_index = last_index + 1

    print(f"Starting from index: {start_index}")

    video_ids = []
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            vid = rec.get("video_id")
            if vid:
                video_ids.append(vid)

    video_ids = sorted(set(video_ids))

    with OUT_INDEX_JSONL.open("a", encoding="utf-8") as f:
        for offset, vid in enumerate(video_ids):
            row = start_index + offset
            f.write(json.dumps({"video_id": vid, "row": row}, ensure_ascii=False) + "\n")

    print("Done.")
    print("Index file:", OUT_INDEX_JSONL)
    print("Total new clips indexed:", len(video_ids))


if __name__ == "__main__":
    main()