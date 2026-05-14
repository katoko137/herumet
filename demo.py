from trackers import HardhatTracker, PersonTracker
from utils import save_picture, read_picture, read_video, save_video


HARDHAT_MODEL_PATH = "models/last_hardhat_200_epochs.pt"
PERSON_MODEL_PATH = "yolov8x.pt"


def run_demo():
    # デモ用の静止画と動画を処理する。リアルタイム処理は main.py に分離している。
    hardhat_tracker = HardhatTracker(model_path=HARDHAT_MODEL_PATH)
    person_tracker = PersonTracker(PERSON_MODEL_PATH)

    process_picture(
        hardhat_tracker,
        person_tracker,
        "input_files/hardhat_input_picture_1.jpg",
        "output_files/hardhat_output_picture_1.jpg",
    )
    process_picture(
        hardhat_tracker,
        person_tracker,
        "input_files/hardhat_input_picture_2.jpg",
        "output_files/hardhat_output_picture_2.jpg",
    )
    process_video(
        hardhat_tracker,
        person_tracker,
        "input_files/hardhat_input_video.avi",
        "output_files/hardhat_output_video.avi",
    )


def process_picture(hardhat_tracker, person_tracker, input_path, output_path):
    # 画像1枚を読み込み、ヘルメットと人物の検出枠を描画して保存する。
    picture = read_picture(input_path)

    hardhat_detection = hardhat_tracker.detect_frame(input_path)
    person_detection = person_tracker.detect_frame(input_path)

    output_picture = hardhat_tracker.draw_frame_bboxes(picture, hardhat_detection)
    output_picture = person_tracker.draw_frame_bboxes(
        output_picture,
        person_detection,
        hardhat_detection,
    )
    save_picture(output_picture, output_path)


def process_video(hardhat_tracker, person_tracker, input_path, output_path):
    # 動画をまとめて読み込み、全フレームに検出枠を描画して保存する。
    video = read_video(input_path)

    hardhat_detections = hardhat_tracker.detect_frames(video)
    hardhat_detections = hardhat_tracker.interpolate_hardhat_positions(hardhat_detections)
    person_detections = person_tracker.detect_frames(video)

    output_video_frames = hardhat_tracker.draw_video_bboxes(video, hardhat_detections)
    output_video_frames = person_tracker.draw_video_bboxes(
        output_video_frames,
        person_detections,
        hardhat_detections,
    )
    save_video(output_video_frames, output_path)


if __name__ == "__main__":
    run_demo()
