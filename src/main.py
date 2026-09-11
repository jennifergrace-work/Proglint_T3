from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

STATE_COLOR = {
    "placed": (0, 220, 220),
    "picking": (0, 165, 255),
    "picked": (0, 220, 0),
    "placing": (255, 165, 0),
}

BAG_BOX_COLOR = (0, 255, 0)
PERSON_BOX_COLOR = (255, 0, 0)
WRIST_COLOR = (0, 0, 255)


def load_config() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def get_boxes(result, class_ids: set, conf_min: float) -> list[tuple[int, int, int, int, float]]:
    boxes = result.boxes
    if boxes is None or boxes.cls is None:
        return []
    out = []
    for (x1, y1, x2, y2), cls, conf in zip(
        boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist()
    ):
        if int(cls) in class_ids and conf >= conf_min:
            out.append((int(x1), int(y1), int(x2), int(y2), conf))
    return out


def get_wrists(result, l_idx: int, r_idx: int, conf_min: float) -> list[tuple[int, int, float, str]]:
    kp = result.keypoints
    if kp is None or kp.conf is None:
        return []
    out = []
    for xy, conf in zip(kp.xy.tolist(), kp.conf.tolist()):
        if conf[l_idx] >= conf_min:
            out.append((int(xy[l_idx][0]), int(xy[l_idx][1]), conf[l_idx], "L"))
        if conf[r_idx] >= conf_min:
            out.append((int(xy[r_idx][0]), int(xy[r_idx][1]), conf[r_idx], "R"))
    return out


def draw_boxes(frame, boxes, color, label) -> None:
    for x1, y1, x2, y2, conf in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f} ({x1},{y1})", (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def draw_wrists(frame, wrists) -> None:
    for x, y, conf, side in wrists:
        cv2.circle(frame, (x, y), 6, WRIST_COLOR, -1)
        cv2.putText(frame, f"{side} wrist {conf:.2f} ({x},{y})", (x + 8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WRIST_COLOR, 1, cv2.LINE_AA)


def draw_state_top_center(frame, state: str) -> None:
    h, w = frame.shape[:2]
    text = state.upper()
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
    x, y = (w - tw) // 2, th + 20
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                STATE_COLOR[state], 3, cv2.LINE_AA)


ROI_COLOR = (0, 255, 255)


def draw_roi(frame, zone: tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = zone
    cv2.rectangle(frame, (x1, y1), (x2, y2), ROI_COLOR, 2)
    cv2.putText(frame, "ROI (place zone)", (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, ROI_COLOR, 1, cv2.LINE_AA)


def near_bag_zone(x1: int, y1: int, x2: int, y2: int, margin: float) -> tuple[int, int, int, int]:
    mx, my = int((x2 - x1) * margin), int((y2 - y1) * margin)
    return x1 - mx, y1 - my, x2 + mx, y2 + my


def wrist_near_zone(zone: tuple[int, int, int, int] | None, wrists, margin: float) -> bool:
    if zone is None:
        return False
    ex1, ey1, ex2, ey2 = near_bag_zone(*zone, margin)
    return any(ex1 <= wx <= ex2 and ey1 <= wy <= ey2 for wx, wy, _, _ in wrists)


def next_state(state: str, bag: bool, wrist: bool, was_together: bool) -> str:

    together = bag and wrist
    absent = not bag and not wrist
    just_touched = together and not was_together  # rising edge, not lingering contact

    if state == "placed" and just_touched:
        return "picking"

    if state == "picking" and absent:
        return "picked"

    if state == "picked" and just_touched:
        return "placing"

    if state == "placing":
        return "placed"

    return state


def run(cfg: dict, video_path: Path, out_path: Path) -> None:
    det_model = YOLO(str(ROOT / cfg["model"]["detection"]))
    pose_model = YOLO(str(ROOT / cfg["model"]["pose"]))

    bag_classes = set(cfg["detection"]["bag_classes"])
    person_class = {cfg["detection"]["person_class"]}
    bag_conf_min = cfg["detection"]["bag_conf_min"]
    kp_conf_min = cfg["detection"]["kp_conf_min"]
    l_wrist = cfg["pose"]["left_wrist"]
    r_wrist = cfg["pose"]["right_wrist"]
    wrist_near_margin = cfg["detection"]["wrist_near_margin"]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    state = "placed"
    frame_idx = 0
    place_zone: tuple[int, int, int, int] | None = None
    was_together = False

    # every state must stay on screen at least this long to be readable,
    # even one (like "placing") the real state only holds for a single frame
    min_hold_frames = max(1, int(cfg["display"]["min_label_s"] * fps))
    display_state = state
    display_hold = min_hold_frames
    display_queue: list[str] = []

    while True:  # while all the frames are run
        ok, frame = cap.read()
        if not ok:
            break

        # #2 detections
        det = det_model(frame, verbose=False)[0]
        pose = pose_model(frame, verbose=False)[0]

        bag_boxes = get_boxes(det, bag_classes, bag_conf_min)
        person_boxes = get_boxes(det, person_class, bag_conf_min)
        wrists = get_wrists(pose, l_wrist, r_wrist, kp_conf_min)

        if place_zone is None and bag_boxes:
            place_zone = bag_boxes[0][:4]  # freeze on the bag's resting position

        bagbox = bool(bag_boxes)
        personbox = bool(person_boxes)
        wrist_detect = wrist_near_zone(place_zone, wrists, wrist_near_margin)

        prev_state = state
        state = next_state(state, bagbox, wrist_detect, was_together)
        was_together = bagbox and wrist_detect
        if state != prev_state:
            display_queue.append(state)

        if display_hold > 0:
            display_hold -= 1
        elif display_queue:
            display_state = display_queue.pop(0)
            display_hold = min_hold_frames - 1

        if place_zone:
            ex1, ey1, ex2, ey2 = near_bag_zone(*place_zone, wrist_near_margin)
            draw_roi(frame, (ex1, ey1, ex2, ey2))  # constant, never moves
        draw_boxes(frame, bag_boxes, BAG_BOX_COLOR, "bag")  # live, follows the bag
        draw_boxes(frame, person_boxes, PERSON_BOX_COLOR, "person")
        draw_wrists(frame, wrists)
        draw_state_top_center(frame, display_state)
        writer.write(frame)

        print(f"frame {frame_idx:5d}  bag={bagbox!s:5}  person={personbox!s:5}  "
              f"wrist={wrist_detect!s:5}  state={state}  display={display_state}")
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"\nOutput -> {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Placed/Picking/Picked/Placing bag-action detector.")
    ap.add_argument("--video", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config()
    video_path = ROOT / (args.video or cfg["video"])
    out_dir = ROOT / cfg["output_dir"]
    out_path = Path(args.out) if args.out else out_dir / f"labeled_{video_path.stem}.mp4"

    run(cfg, video_path, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
