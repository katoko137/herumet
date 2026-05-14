from __future__ import annotations

import argparse
import time

import cv2

from hardhat_detector import HardhatDetector, PersonHelmetResult


def update_no_helmet_alerts(
    persons: list[PersonHelmetResult],
    no_helmet_counts: dict[int, int],
    alerted_ids: set[int],
    alert_frames: int,
):
    """未装着が一定フレーム続いた人物だけ、ターミナルへ一度警告する。"""
    if alert_frames <= 0:
        return

    current_person_ids = {person.person_id for person in persons}

    for person in persons:
        person_id = person.person_id
        if person.has_helmet:
            no_helmet_counts.pop(person_id, None)
            alerted_ids.discard(person_id)
            continue

        # ByteTrackが維持する人物IDごとに、未装着の連続フレーム数を数える。
        no_helmet_counts[person_id] = no_helmet_counts.get(person_id, 0) + 1

        if no_helmet_counts[person_id] >= alert_frames and person_id not in alerted_ids:
            print(
                f"[警告] Person ID {person_id} がヘルメット未装着で "
                f"{no_helmet_counts[person_id]} フレーム連続検出されました。",
                flush=True,
            )
            alerted_ids.add(person_id)

    # 画面外に消えた人物のカウントを残さない。
    for person_id in list(no_helmet_counts):
        if person_id not in current_person_ids:
            no_helmet_counts.pop(person_id, None)
            alerted_ids.discard(person_id)


def draw_fps(frame, fps):
    """推論速度を画面左上に表示する。"""
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )


def run_camera(camera_index=0, width=1280, height=720, conf=0.25, device=None, alert_frames=0):
    """カメラから1フレームずつ読み、検出結果をリアルタイム表示する。"""
    detector = HardhatDetector(conf=conf, device=device, verbose=False)

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    window_name = "Hard Hat Detection - Camera"
    previous_time = time.time()

    # no_helmet_counts: {人物ID: ヘルメット未装着で連続検出されたフレーム数}
    # alerted_ids: すでにターミナル警告を出した人物ID
    no_helmet_counts: dict[int, int] = {}
    alerted_ids: set[int] = set()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Could not read a frame from the camera.")

            result = detector.process_frame(frame, persist=True)
            update_no_helmet_alerts(
                result.persons,
                no_helmet_counts,
                alerted_ids,
                alert_frames,
            )

            output_frame = result.annotated_frame
            current_time = time.time()
            fps = 1.0 / max(current_time - previous_time, 1e-6)
            previous_time = current_time
            draw_fps(output_frame, fps)

            cv2.imshow(window_name, output_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


def parse_args():
    parser = argparse.ArgumentParser(description="Hard hat realtime camera detection")
    parser.add_argument("--camera", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--camera-index", type=int, default=0, help="camera index")
    parser.add_argument("--width", type=int, default=1280, help="camera capture width")
    parser.add_argument("--height", type=int, default=720, help="camera capture height")
    parser.add_argument("--conf", type=float, default=0.25, help="detection confidence threshold")
    parser.add_argument("--device", default=None, help="Ultralytics device, for example 0, cuda, or cpu")
    parser.add_argument(
        "--alert-frames",
        type=int,
        default=0,
        help="print a terminal warning after this many consecutive no-helmet frames; 0 disables it",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_camera(
        camera_index=args.camera_index,
        width=args.width,
        height=args.height,
        conf=args.conf,
        device=args.device,
        alert_frames=args.alert_frames,
    )


if __name__ == "__main__":
    main()
