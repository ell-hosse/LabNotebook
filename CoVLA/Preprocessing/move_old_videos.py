import json
import shutil
from pathlib import Path

VIDEO_DIR = Path(r"D:\hf\hub\datasets--turing-motors--CoVLA-Dataset\snapshots\0a6d39e41659903a26dde957744e70dbc360bb6d\videos")
INDEX_FILE = Path(r"D:\hf\CoVLA-metadata\clip_index.jsonl")  # update path
TARGET_DIR = VIDEO_DIR / "old_batch"
TARGET_DIR.mkdir(exist_ok=True)

with INDEX_FILE.open() as f:
    for line in f:
        vid = json.loads(line)["video_id"]
        for ext in [".mp4"]:
            src = VIDEO_DIR / f"{vid}{ext}"
            if src.exists():
                shutil.move(src, TARGET_DIR / src.name)
                break

