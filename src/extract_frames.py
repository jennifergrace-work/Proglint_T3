"""Dump every frame of a video to JPEGs.

Usage:
    python -m src.extract_frames --video video_20260910_161600.mp4 --out frames
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent


def extract(video_path: Path, out_dir: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(str(out_dir / f"f{i:05d}.jpg"), frame)
        i += 1
    cap.release()
    return i


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract every frame of a video to JPEGs.")
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="frames")
    args = ap.parse_args()

    n = extract(ROOT / args.video, ROOT / args.out)
    print(f"extracted {n} frames -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
