import json
from pathlib import Path
import numpy as np


def traj_jsonl_2_num():
    CLIP_INDEX = Path(r"D:\hf\CoVLA-metadata\clip_index.jsonl")
    TRAJ_JSONL = Path(r"D:\hf\CoVLA-metadata\traj_video_60f_traj10.jsonl")

    T = 60
    P = 10
    C = 3

    video_to_row = {}
    with CLIP_INDEX.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            video_to_row[item["video_id"]] = item["row"]

    N = len(video_to_row)

    traj = np.zeros((N, T, P, C), dtype=np.float32)

    with TRAJ_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)

            video_id = item["video_id"]
            frame_idx = item["frame_idx"] // 10
            trajectory = np.asarray(item["trajectory"], dtype=np.float32)

            row = video_to_row[video_id]
            traj[row, frame_idx] = trajectory

    return traj