from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from utils.bbox_utils import hardhat_is_on


HARDHAT_MODEL_PATH = "models/last_hardhat_200_epochs.pt"
PERSON_MODEL_PATH = "yolov8x.pt"

BBox = list[float]


@dataclass(frozen=True)
class PersonHelmetResult:
    """人物1人分のbboxとヘルメット着用判定をまとめた結果。"""

    person_id: int
    person_bbox: BBox
    has_helmet: bool
    helmet_id: Optional[int] = None
    helmet_bbox: Optional[BBox] = None


@dataclass(frozen=True)
class FrameDetectionResult:
    """1フレーム分の描画済み画像と検出・判定結果。"""

    annotated_frame: np.ndarray
    persons: list[PersonHelmetResult]
    hardhats: dict[int, BBox]


class HardhatDetector:
    """カメラ生フレームを受け取り、検出・判定・bbox描画をまとめて行うクラス。"""

    HELMET_COLOR = (0, 255, 0)
    PERSON_WITH_HELMET_COLOR = (0, 255, 0)
    PERSON_WITHOUT_HELMET_COLOR = (0, 0, 255)
    TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(
        self,
        hardhat_model_path=HARDHAT_MODEL_PATH,
        person_model_path=PERSON_MODEL_PATH,
        conf=0.25,
        device=None,
        verbose=False,
    ):
        """YOLOモデルを一度だけ読み込み、以降のフレーム処理で再利用する。"""
        from trackers import HardhatTracker, PersonTracker

        self.hardhat_tracker = HardhatTracker(
            model_path=hardhat_model_path,
            conf=conf,
            device=device,
            verbose=verbose,
        )
        self.person_tracker = PersonTracker(
            person_model_path,
            conf=conf,
            device=device,
            verbose=verbose,
        )

    def process_frame(self, frame: np.ndarray, persist=True) -> FrameDetectionResult:
        """OpenCVのBGRフレームを処理し、元画像を壊さず描画済みフレームを返す。"""
        # 検出: helmetとpersonを同じ入力フレームから取得する。
        hardhat_detections = self._copy_detections(self.hardhat_tracker.detect_frame(frame))
        person_detections = self._copy_detections(
            self.person_tracker.detect_frame(frame, persist=persist)
        )

        # 判定: 人物ごとに、頭部付近へ最初に入ったhelmetを対応付ける。
        persons = self._build_person_results(
            person_detections,
            hardhat_detections,
            frame,
        )

        # 描画: 呼び出し元が生フレームを再利用できるよう、コピーへ描画する。
        annotated_frame = frame.copy()
        self._draw_hardhats(annotated_frame, hardhat_detections)
        self._draw_persons(annotated_frame, persons)

        return FrameDetectionResult(
            annotated_frame=annotated_frame,
            persons=persons,
            hardhats=hardhat_detections,
        )

    def _build_person_results(
        self,
        person_detections: dict[int, BBox],
        hardhat_detections: dict[int, BBox],
        frame: np.ndarray,
    ) -> list[PersonHelmetResult]:
        persons = []
        for person_id, person_bbox in person_detections.items():
            helmet_id, helmet_bbox = self._find_matching_hardhat(
                person_bbox,
                hardhat_detections,
                frame,
            )
            persons.append(
                PersonHelmetResult(
                    person_id=person_id,
                    person_bbox=person_bbox,
                    has_helmet=helmet_id is not None,
                    helmet_id=helmet_id,
                    helmet_bbox=helmet_bbox,
                )
            )
        return persons

    @staticmethod
    def _find_matching_hardhat(
        person_bbox: BBox,
        hardhat_detections: dict[int, BBox],
        frame: np.ndarray,
    ) -> tuple[Optional[int], Optional[BBox]]:
        for helmet_id, helmet_bbox in hardhat_detections.items():
            if hardhat_is_on(person_bbox, helmet_bbox, frame):
                return helmet_id, helmet_bbox
        return None, None

    def _draw_hardhats(self, frame: np.ndarray, hardhat_detections: dict[int, BBox]):
        for helmet_id, bbox in hardhat_detections.items():
            self._draw_bbox(
                frame,
                bbox,
                f"Helmet ID: {helmet_id}",
                self.HELMET_COLOR,
            )

    def _draw_persons(self, frame: np.ndarray, persons: list[PersonHelmetResult]):
        for person in persons:
            color = (
                self.PERSON_WITH_HELMET_COLOR
                if person.has_helmet
                else self.PERSON_WITHOUT_HELMET_COLOR
            )
            status = "helmet" if person.has_helmet else "no helmet"
            self._draw_bbox(
                frame,
                person.person_bbox,
                f"Person ID: {person.person_id} ({status})",
                color,
            )

    @classmethod
    def _draw_bbox(cls, frame: np.ndarray, bbox: BBox, label: str, color):
        x1, y1, x2, y2 = bbox
        label_y = max(int(y1) - 10, 15)
        cv2.putText(
            frame,
            label,
            (int(x1), label_y),
            cls.TEXT_FONT,
            0.5,
            color,
            2,
        )
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    @staticmethod
    def _copy_detections(detections: dict[int, BBox]) -> dict[int, BBox]:
        return {int(object_id): list(bbox) for object_id, bbox in detections.items()}
