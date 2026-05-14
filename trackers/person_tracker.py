from ultralytics import YOLO
import cv2
import sys
from utils.bbox_utils import hardhat_is_on
sys.path.append('utils')


class PersonTracker:
    def __init__(self, model_path, conf=0.25, device=None, verbose=False):
        self.model = YOLO(model_path)
        self.conf = conf
        self.device = device
        self.verbose = verbose

    def detect_frames(self, frames, persist=False):
        # 複数フレームを順番に人物検出する。
        person_detections = []
        for frame in frames:
            person_dict = self.detect_frame(frame, persist=persist)
            person_detections.append(person_dict)
        return person_detections

    def detect_frame(self, frame, persist=False):
        # 1フレームから人物の矩形座標を取り出す。
        results = self.model.track(
            frame,
            classes=[0],
            conf=self.conf,
            device=self.device,
            persist=persist,
            verbose=self.verbose,
        )[0]
        id_name_dict = results.names
        person_dict = {}
        for index, box in enumerate(results.boxes, start=1):
            track_id = index
            if box.id is not None:
                track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = int(box.cls.tolist()[0])
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == "person":
                person_dict[track_id] = result
        return person_dict
    
    def draw_video_bboxes(self, video_frames, person_detections, hardhat_detections=None):
        # 複数フレームに人物の検出枠を描画する。
        output_video_frames = []
        if hardhat_detections is None:
            hardhat_detections = [None] * len(video_frames)
        for frame, person_dict, hardhat_dict in zip(video_frames, person_detections, hardhat_detections):
            frame = self.draw_frame_bboxes(frame, person_dict, hardhat_dict)
            output_video_frames.append(frame)
        
        return output_video_frames

    def draw_frame_bboxes(self, frame, person_detections, hardhat_detections=None):
        # ヘルメット装着者は緑、未装着者は赤で人物枠を描画する。
        for track_id, bbox in person_detections.items():
            color = (0, 0, 255)
            x1, y1, x2, y2 = bbox
            if hardhat_detections is not None:
                for bbox_hh in hardhat_detections.values():
                    if hardhat_is_on(bbox, bbox_hh, frame):
                        color = (0, 255, 0)
                        break
            cv2.putText(frame, f"Person ID: {track_id}", (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        return frame
