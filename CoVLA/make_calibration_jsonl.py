import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import math

IN_MANIFEST = Path(r"D:\hf\CoVLA-metadata\manifest.jsonl")
OUT_CALIB = Path(r"D:\hf\CoVLA-metadata\calibration_by_video.jsonl")
OUT_REPORT = Path(r"D:\hf\CoVLA-metadata\calibration_by_video_report.json")

# Tolerances (in case of tiny float noise)
ATOL = 1e-7
RTOL = 1e-6


def is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def allclose_nested(a: Any, b: Any, atol: float, rtol: float) -> bool:
    """Compare nested lists of numbers without numpy."""
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(allclose_nested(x, y, atol, rtol) for x, y in zip(a, b))
    if is_number(a) and is_number(b):
        return math.isclose(float(a), float(b), rel_tol=rtol, abs_tol=atol)
    return a == b


def parse_single_key_line(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Each states jsonl line looks like {"0": {...}} or {"123": {...}}
    Return the inner dict {...}
    """
    if not isinstance(obj, dict) or len(obj) != 1:
        raise ValueError("Expected a single-key dict per JSONL line.")
    (_, frame_dict), = obj.items()
    if not isinstance(frame_dict, dict):
        raise ValueError("Expected frame payload to be a dict.")
    return frame_dict


def check_constant_matrices(states_path: Path) -> Tuple[bool, bool, Optional[list], Optional[list], int]:
    """
    Returns:
      (intrinsic_constant, extrinsic_constant, intrinsic_matrix_if_constant, extrinsic_matrix_if_constant, frames_scanned)
    """
    intrinsic_ref = None
    extrinsic_ref = None
    intrinsic_constant = True
    extrinsic_constant = True
    frames_scanned = 0

    with states_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            frame = parse_single_key_line(json.loads(line))
            intrinsic = frame.get("intrinsic_matrix")
            extrinsic = frame.get("extrinsic_matrix")

            if intrinsic is None:
                intrinsic_constant = False
            if extrinsic is None:
                extrinsic_constant = False

            if intrinsic_ref is None and intrinsic is not None:
                intrinsic_ref = intrinsic
            if extrinsic_ref is None and extrinsic is not None:
                extrinsic_ref = extrinsic

            if intrinsic_constant and intrinsic_ref is not None and intrinsic is not None:
                if not allclose_nested(intrinsic_ref, intrinsic, ATOL, RTOL):
                    intrinsic_constant = False

            if extrinsic_constant and extrinsic_ref is not None and extrinsic is not None:
                if not allclose_nested(extrinsic_ref, extrinsic, ATOL, RTOL):
                    extrinsic_constant = False

            frames_scanned += 1

            # early stop if both already failed
            if not intrinsic_constant and not extrinsic_constant:
                break

    return (
        intrinsic_constant,
        extrinsic_constant,
        intrinsic_ref if intrinsic_constant else None,
        extrinsic_ref if extrinsic_constant else None,
        frames_scanned,
    )


def main():
    assert IN_MANIFEST.exists(), f"Manifest not found: {IN_MANIFEST}"

    OUT_CALIB.parent.mkdir(parents=True, exist_ok=True)

    n_total = 0
    n_written = 0
    missing_states = 0
    not_constant = 0
    examples_not_constant = []
    examples_missing_states = []

    with IN_MANIFEST.open("r", encoding="utf-8") as fin, OUT_CALIB.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_total += 1

            rec = json.loads(line)
            video_id = rec.get("video_id")
            states_path_str = rec.get("states_path")

            if not video_id or not states_path_str:
                continue

            states_path = Path(states_path_str)
            if not states_path.exists():
                missing_states += 1
                if len(examples_missing_states) < 10:
                    examples_missing_states.append(video_id)
                continue

            intrinsic_const, extrinsic_const, K, E, frames_scanned = check_constant_matrices(states_path)

            # Write ONLY if both are constant and present
            if intrinsic_const and extrinsic_const and K is not None and E is not None:
                out_rec = {
                    "video_id": video_id,
                    "intrinsic_matrix": K,
                    "extrinsic_matrix": E,
                }
                fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                n_written += 1
            else:
                not_constant += 1
                if len(examples_not_constant) < 10:
                    examples_not_constant.append({
                        "video_id": video_id,
                        "intrinsic_constant": intrinsic_const,
                        "extrinsic_constant": extrinsic_const,
                        "frames_scanned": frames_scanned
                    })

    report = {
        "in_manifest": str(IN_MANIFEST),
        "out_calibration_jsonl": str(OUT_CALIB),
        "total_manifest_rows_read": n_total,
        "written_rows": n_written,
        "missing_states_files": missing_states,
        "not_written_due_to_nonconstant_or_missing_calib": not_constant,
        "example_missing_states_video_ids": examples_missing_states,
        "example_nonconstant_cases": examples_not_constant,
        "atol": ATOL,
        "rtol": RTOL,
    }

    with OUT_REPORT.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("Done.")
    print("Calibration JSONL:", OUT_CALIB)
    print("Report:", OUT_REPORT)
    print(f"Read {n_total} manifest rows, wrote {n_written} calibration rows.")
    print(f"Missing states files: {missing_states}")
    print(f"Not written (non-constant / missing calib): {not_constant}")


if __name__ == "__main__":
    main()
