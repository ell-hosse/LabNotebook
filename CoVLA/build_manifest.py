import os
import json
from pathlib import Path

VIDEO_DIR = Path(r"D:\hf\hub\datasets--turing-motors--CoVLA-Dataset\snapshots\0a6d39e41659903a26dde957744e70dbc360bb6d\videos")
CAPTIONS_DIR = Path(r"D:\hf\CoVLA-metadata\captions")
STATES_DIR = Path(r"D:\hf\CoVLA-metadata\states")

OUT_PATH = Path(r"D:\hf\CoVLA-metadata\manifest.jsonl")
REPORT_PATH = Path(r"D:\hf\CoVLA-metadata\manifest_report.json")

# It will only include clips that exist in VIDEO_DIR (downloaded portion).
ONLY_LOCAL_VIDEOS = True

# If None, it includes all matched items found.
FRACTION = None

VIDEO_EXTS = {".mp4", ".webm", ".mkv"}


def base_id(p: Path) -> str:
    return p.stem


def count_jsonl_frames(jsonl_path: Path) -> int:
    """
    Counts number of lines (frames) in a JSONL file in a streaming-safe way.
    Each line corresponds to one frame entry like {"0": {...}}.
    """
    n = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n


def main():
    assert VIDEO_DIR.exists(), f"VIDEO_DIR not found: {VIDEO_DIR}"
    assert CAPTIONS_DIR.exists(), f"CAPTIONS_DIR not found: {CAPTIONS_DIR}"
    assert STATES_DIR.exists(), f"STATES_DIR not found: {STATES_DIR}"

    video_files = [p for p in VIDEO_DIR.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    video_ids = {base_id(p): p for p in video_files}

    caption_files = [p for p in CAPTIONS_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".jsonl"]
    caption_ids = {base_id(p): p for p in caption_files}

    state_files = [p for p in STATES_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".jsonl"]
    state_ids = {base_id(p): p for p in state_files}

    if ONLY_LOCAL_VIDEOS:
        candidate_ids = set(video_ids.keys())
    else:
        candidate_ids = set(video_ids.keys()) | set(caption_ids.keys()) | set(state_ids.keys())

    matched = []
    missing_video = []
    missing_caption = []
    missing_state = []

    for vid in sorted(candidate_ids):
        v = video_ids.get(vid)
        c = caption_ids.get(vid)
        s = state_ids.get(vid)

        if v is None:
            missing_video.append(vid)
            continue
        if c is None:
            missing_caption.append(vid)
            continue
        if s is None:
            missing_state.append(vid)
            continue

        matched.append(vid)

    if FRACTION is not None:
        assert 0 < FRACTION <= 1.0
        n_keep = max(1, int(len(matched) * FRACTION))
        matched = matched[:n_keep]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    with OUT_PATH.open("w", encoding="utf-8") as out:
        for vid in matched:
            vpath = str(video_ids[vid])
            cpath = str(caption_ids[vid])
            spath = str(state_ids[vid])

            if not (Path(vpath).exists() and Path(cpath).exists() and Path(spath).exists()):
                continue

            rec = {
                "video_id": vid,
                "video_path": vpath,
                "states_path": spath,
                "captions_path": cpath,
            }

            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            rows_written += 1

    report = {
        "video_dir": str(VIDEO_DIR),
        "captions_dir": str(CAPTIONS_DIR),
        "states_dir": str(STATES_DIR),
        "only_local_videos": ONLY_LOCAL_VIDEOS,
        "fraction": FRACTION,
        "n_videos_found": len(video_ids),
        "n_captions_found": len(caption_ids),
        "n_states_found": len(state_ids),
        "n_matched_written": rows_written,
        "n_missing_caption": len(missing_caption),
        "n_missing_state": len(missing_state),
        "n_missing_video": len(missing_video),
        "example_missing_caption": missing_caption[:10],
        "example_missing_state": missing_state[:10],
        "example_missing_video": missing_video[:10],
        "out_manifest": str(OUT_PATH),
    }

    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(f"Manifest: {OUT_PATH}")
    print(f"Report:   {REPORT_PATH}")
    print(f"Matched rows written: {rows_written}")


if __name__ == "__main__":
    main()
