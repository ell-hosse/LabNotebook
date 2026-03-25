from pathlib import Path
import json

CLIP_INDEX = Path(r"D:\hf\CoVLA-metadata\clip_index.jsonl")
STATES_JSONL = Path(r"D:\hf\CoVLA-metadata\states")
OUT_VELOCITY_JSONL = Path(r"D:\hf\CoVLA-metadata\velocity_60_frames.jsonl")

N_FRAMES_PER_CLIP = 600
DOWNSAMPLED_FRAMES = 60

def get_downsampled_frame_indices():
    N_FRAMES_PER_CLIP = 600
    DOWNSAMPLED_FRAMES = 60

    step = N_FRAMES_PER_CLIP // DOWNSAMPLED_FRAMES
    return list(range(0, N_FRAMES_PER_CLIP, step))[:DOWNSAMPLED_FRAMES]


def extract_velocity():
    frame_indices = get_downsampled_frame_indices()
    velocities = []

    with open(CLIP_INDEX, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            video_id = item["video_id"]
            state_file = STATES_JSONL / f"{video_id}.jsonl"

            clip_velocities = [0.0] * len(frame_indices)

            with open(state_file, "r", encoding="utf-8") as sf:
                for i, line in enumerate(sf):
                    if i >= N_FRAMES_PER_CLIP:
                        break
                    if i in frame_indices:
                        state = json.loads(line)
                        clip_velocities[frame_indices.index(i)] = state[f"{i}"]

            velocities.append(clip_velocities)

    with open(OUT_VELOCITY_JSONL, "w", encoding="utf-8") as f:
        for row in velocities:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    extract_velocity()