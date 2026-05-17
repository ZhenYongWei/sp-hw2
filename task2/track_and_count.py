import cv2
import os
import json
from pathlib import Path
import numpy as np
from ultralytics import YOLO

TASK_DIR = Path(__file__).resolve().parent
REPO_ROOT = TASK_DIR.parent
DATA_DIR = REPO_ROOT / "data"
MODEL_PATH = str(TASK_DIR / "runs" / "yolov8n_vehicle" / "weights" / "best.pt")
OUTPUT_DIR = str(TASK_DIR / "output")
CONF_THRES = 0.3
IOU_THRES = 0.5

def find_test_video():
    video_dir = str(TASK_DIR)
    preferred = ["test_video.mp4", "clip_0.mp4", "traffic.mp4", "sample_5s.mp4"]
    for name in preferred:
        path = os.path.join(video_dir, name)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return path
    for f in os.listdir(video_dir):
        if f.endswith((".mp4", ".avi", ".mov", ".mkv")):
            return os.path.join(video_dir, f)
    traffic_dir = str(DATA_DIR / "trafic_data")
    for root, dirs, files in os.walk(traffic_dir):
        for f in files:
            if f.endswith((".mp4", ".avi", ".mov", ".mkv")):
                return os.path.join(root, f)
    return None


def run_tracking(model_path, video_path):
    model = YOLO(model_path)
    results = model.track(
        source=video_path,
        persist=True,
        tracker="bytetrack.yaml",
        conf=CONF_THRES,
        iou=IOU_THRES,
        verbose=False,
    )

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    return list(results), fps, w, h, total_frames


def draw_result(frame, result):
    annotated = frame.copy()
    if result.boxes.id is None:
        return annotated

    boxes = result.boxes
    names = result.names
    for i in range(len(boxes)):
        track_id = int(boxes.id[i])
        cls_id = int(boxes.cls[i])
        conf = float(boxes.conf[i])
        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
        label = f"ID:{track_id} {names[cls_id]} {conf:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
    return annotated


def save_tracking_video(video_path, results_list, fps, w, h, output_path=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "tracked_video.mp4")

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx >= len(results_list):
            break
        writer.write(draw_result(frame, results_list[frame_idx]))
        frame_idx += 1

    cap.release()
    writer.release()

    print(f"Tracking video saved to {output_path}")
    return output_path


def select_occlusion_window(results_list, window_size=4):
    if not results_list:
        return []

    counts = []
    for result in results_list:
        if result.boxes.id is None:
            counts.append(0)
        else:
            counts.append(len(result.boxes))

    best_start = 0
    best_score = -1
    for start in range(max(1, len(counts) - window_size + 1)):
        score = sum(counts[start:start + window_size])
        if score > best_score:
            best_score = score
            best_start = start

    return list(range(best_start, min(best_start + window_size, len(results_list))))


def choose_counting_line(results_list, frame_height):
    candidates = [int(frame_height * ratio) for ratio in (0.3, 0.4, 0.5, 0.6, 0.7)]
    best_line = candidates[0]
    best_count = -1

    for line_y in candidates:
        crossed_ids = set()
        previous_positions = {}
        for result in results_list:
            if result.boxes.id is None:
                continue
            boxes = result.boxes
            for i in range(len(boxes)):
                track_id = int(boxes.id[i])
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                cy = (y1 + y2) / 2
                if track_id in previous_positions:
                    prev_cy = previous_positions[track_id]
                    if (prev_cy < line_y and cy >= line_y) or (prev_cy > line_y and cy <= line_y):
                        crossed_ids.add(track_id)
                previous_positions[track_id] = cy

        if len(crossed_ids) > best_count:
            best_count = len(crossed_ids)
            best_line = line_y

    return best_line


def extract_occlusion_frames(video_path, results_list, frame_indices, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, "occlusion_frames")
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    saved = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            path = os.path.join(output_dir, f"frame_{idx}.jpg")
            annotated = draw_result(frame, results_list[idx])
            cv2.putText(
                annotated,
                f"Frame {idx}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )
            cv2.imwrite(path, annotated)
            saved.append(path)
    cap.release()
    return saved


def count_crossing_line(video_path, results_list, line_y=None):
    if line_y is None:
        cap = cv2.VideoCapture(video_path)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        line_y = choose_counting_line(results_list, h)

    crossed_ids = set()
    previous_positions = {}

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    output_path = os.path.join(OUTPUT_DIR, "counting_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < len(results_list):
            result = results_list[frame_idx]
            frame = draw_result(frame, result)
            if result.boxes.id is not None:
                boxes = result.boxes
                for i in range(len(boxes)):
                    track_id = int(boxes.id[i])
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    cy = (y1 + y2) / 2

                    if track_id in previous_positions:
                        prev_cy = previous_positions[track_id]
                        if (prev_cy < line_y and cy >= line_y) or (prev_cy > line_y and cy <= line_y):
                            crossed_ids.add(track_id)

                    previous_positions[track_id] = cy

        cv2.line(frame, (0, line_y), (w, line_y), (0, 0, 255), 2)
        cv2.putText(frame, f"Crossed: {len(crossed_ids)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Total objects crossed line: {len(crossed_ids)}")
    print(f"Counting video saved to {output_path}")
    return len(crossed_ids), line_y


def save_summary(video_path, fps, width, height, total_frames, occlusion_frames, line_y, crossing_count):
    summary = {
        "video_path": video_path,
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "occlusion_frame_indices": occlusion_frames,
        "counting_line_y": line_y,
        "crossing_count": crossing_count,
    }
    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")
    return summary_path


if __name__ == "__main__":
    video_path = find_test_video()

    if video_path:
        print(f"Found video: {video_path}")
        results, fps, w, h, total_frames = run_tracking(MODEL_PATH, video_path)
        print(f"Video: {video_path}, FPS={fps}, Size={w}x{h}, Frames={total_frames}")

        print("=" * 60)
        print("Step 1: Video tracking")
        print("=" * 60)
        tracked_video = save_tracking_video(video_path, results, fps, w, h)
        print(f"Tracked video saved: {tracked_video}")

        print("=" * 60)
        print("Step 2: Extract occlusion frames")
        print("=" * 60)
        frame_indices = select_occlusion_window(results, window_size=4)
        saved = extract_occlusion_frames(video_path, results, frame_indices)
        print(f"Selected frame indices: {frame_indices}")
        print(f"Occlusion frames saved: {saved}")

        print("=" * 60)
        print("Step 3: Line crossing counting")
        print("=" * 60)
        count, line_y = count_crossing_line(video_path, results)
        print(f"Crossing count: {count}")

        print("=" * 60)
        print("Step 4: Save summary")
        print("=" * 60)
        summary_path = save_summary(video_path, fps, w, h, total_frames, frame_indices, line_y, count)
        print(f"Summary path: {summary_path}")
    else:
        print("No test video found. Please provide a video for Task 2.")
        print("You can download a traffic video or record one yourself.")
