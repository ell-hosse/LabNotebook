import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

MANIFEST_PATH = Path(r"D:\hf\CoVLA-metadata\manifest.jsonl")
OUT_TRAJ_JSONL = Path(r"D:\hf\CoVLA-metadata\traj_video_60f_traj10.jsonl")
OUT_CAPS_JSONL = Path(r"D:\hf\CoVLA-metadata\captions10_per_video.jsonl")

N_FRAMES_PER_CLIP = 600  # 30s @ 20Hz
DOWNSAMPLED_FRAMES = 60 # keep 60 anchors
CAPTION_WINDOW = 60 # unique caption per 60 frames => 10 per clip
TRAJ_LEN = 60 # trajectory list length per frame
TRAJ_SUBSAMPLE = 10 # choose 10 of 60


def get_downsampled_frame_indices() -> List[int]:
    step = N_FRAMES_PER_CLIP // DOWNSAMPLED_FRAMES  # 600//60 = 10
    return list(range(0, N_FRAMES_PER_CLIP, step))[:DOWNSAMPLED_FRAMES]


# Uniformly pick 10 indices out of 60: 0..59
def get_traj_subsample_indices() -> List[int]:
    idxs = []
    for k in range(TRAJ_SUBSAMPLE):
        t = int(round(k * (TRAJ_LEN - 1) / (TRAJ_SUBSAMPLE - 1)))
        idxs.append(t)
    idxs = sorted(set(idxs))
    while len(idxs) < TRAJ_SUBSAMPLE:
        for cand in range(TRAJ_LEN):
            if cand not in idxs:
                idxs.append(cand)
                if len(idxs) == TRAJ_SUBSAMPLE:
                    break
        idxs = sorted(idxs)
    return idxs


def read_manifest(manifest_path: Path) -> List[Dict[str, Any]]:
    rows = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def parse_single_key_jsonl_line(obj: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    if not isinstance(obj, dict) or len(obj) != 1:
        raise ValueError("Expected a single-key dict per JSONL line.")
    (k, v), = obj.items()
    return int(k), v


def get_uniform_indices(n: int, k: int = 10) -> List[int]:
    if n < k:
        raise ValueError(f"n must be >= k, got n={n}, k={k}")
    if k == 1:
        return [0]
    return [int(round(i * (n - 1) / (k - 1))) for i in range(k)]


def extract_selected_trajectories(states_path: Path, wanted_frames: set,) -> Dict[int, List[List[float]]]:
    out: Dict[int, List[List[float]]] = {}
    with states_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            frame_idx, frame = parse_single_key_jsonl_line(json.loads(line))
            if frame_idx not in wanted_frames:
                continue

            traj = frame.get("trajectory")
            if not isinstance(traj, list) or len(traj) < TRAJ_SUBSAMPLE:
                # accept only if we can output 10 points
                continue

            keep_idxs = get_uniform_indices(len(traj), TRAJ_SUBSAMPLE)
            traj_sub = [traj[i] for i in keep_idxs]

            out[frame_idx] = traj_sub
            if len(out) == len(wanted_frames):
                break
    return out


def extract_unique_rich_captions(captions_path: Path, window_starts: List[int]) -> Dict[int, str]:
    starts_set = set(window_starts)
    out: Dict[int, str] = {}

    with captions_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            frame_idx, frame = parse_single_key_jsonl_line(json.loads(line))
            if frame_idx not in starts_set:
                continue

            rich = frame.get("rich_caption")
            if not isinstance(rich, str):
                rich = "" if rich is None else str(rich)

            caption_idx = frame_idx // CAPTION_WINDOW # 0..9
            out[caption_idx] = rich

            if len(out) == len(window_starts):
                break

    return out


def main():
    assert MANIFEST_PATH.exists(), f"Manifest not found: {MANIFEST_PATH}"

    manifest = read_manifest(MANIFEST_PATH)
    print(f"Loaded {len(manifest)} manifest rows")

    OUT_TRAJ_JSONL.parent.mkdir(parents=True, exist_ok=True)
    OUT_CAPS_JSONL.parent.mkdir(parents=True, exist_ok=True)

    frame_indices_60 = get_downsampled_frame_indices() # 60 anchors: 0..590 step 10
    traj_keep_idxs_10 = get_traj_subsample_indices() # 10 out of 60
    caption_window_starts = list(range(0, N_FRAMES_PER_CLIP, CAPTION_WINDOW))[:10] # 0..540 step 60

    print("Downsampled frame indices (60):", frame_indices_60[:10], "...", frame_indices_60[-3:])
    print("Trajectory keep indices (10 of 60):", traj_keep_idxs_10)
    print("Caption window starts (10):", caption_window_starts)

    n_traj_rows = 0
    n_cap_rows = 0
    skipped = 0

    with OUT_TRAJ_JSONL.open("w", encoding="utf-8") as f_traj, OUT_CAPS_JSONL.open("w", encoding="utf-8") as f_caps:
        for i, rec in enumerate(manifest, start=1):
            video_id = rec.get("video_id")
            states_path = rec.get("states_path")
            captions_path = rec.get("captions_path")

            if not (video_id and states_path and captions_path):
                skipped += 1
                continue

            states_path = Path(states_path)
            captions_path = Path(captions_path)
            if not (states_path.exists() and captions_path.exists()):
                skipped += 1
                continue

            wanted_frames = set(frame_indices_60)
            traj_map = extract_selected_trajectories(states_path, wanted_frames)

            # Trajectory rows (no video_path; use key "trajectory")
            for frame_idx in frame_indices_60:
                traj_sub = traj_map.get(frame_idx)
                if traj_sub is None:
                    continue
                out_row = {
                    "video_id": video_id,
                    "frame_idx": frame_idx, # still 0..599 indexing
                    "trajectory": traj_sub # 10 points, each is [x,y,z] (as stored)
                }
                f_traj.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                n_traj_rows += 1

            # Caption rows (no video_id)
            cap_map = extract_unique_rich_captions(captions_path, caption_window_starts)
            caption_id = Path(captions_path).stem

            for caption_idx in range(10):
                caption = cap_map.get(caption_idx, "")
                out_row = {
                    "caption_id": caption_id,
                    "caption_idx": caption_idx, # 0..9
                    "rich_caption": caption
                }
                f_caps.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                n_cap_rows += 1

            if i % 50 == 0:
                print(f"Processed {i}/{len(manifest)} clips | traj_rows={n_traj_rows} cap_rows={n_cap_rows}")

    print("\nDone.")
    print("Traj JSONL:", OUT_TRAJ_JSONL)
    print("Captions JSONL:", OUT_CAPS_JSONL)
    print(f"Trajectory rows written: {n_traj_rows}")
    print(f"Caption rows written:    {n_cap_rows}")
    print(f"Skipped manifest rows:   {skipped}")


if __name__ == "__main__":
    main()