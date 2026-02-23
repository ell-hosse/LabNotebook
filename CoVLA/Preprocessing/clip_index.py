import json
from pathlib import Path


MANIFEST_PATH = Path(r"D:\hf\CoVLA-metadata\manifest.jsonl")
OUT_INDEX_JSONL = Path(r"D:\hf\CoVLA-metadata\clip_index.jsonl")

def main():
    assert MANIFEST_PATH.exists(), f"Manifest not found: {MANIFEST_PATH}"
    OUT_INDEX_JSONL.parent.mkdir(parents=True, exist_ok=True)

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

    with OUT_INDEX_JSONL.open("w", encoding="utf-8") as f:
        for row, vid in enumerate(video_ids):
            f.write(json.dumps({"video_id": vid, "row": row}, ensure_ascii=False) + "\n")

    print("Done.")
    print("Index file:", OUT_INDEX_JSONL)
    print("Total clips indexed:", len(video_ids))


if __name__ == "__main__":
    main()
