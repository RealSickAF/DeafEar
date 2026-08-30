#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Background hand-gesture recognizer: camera -> MediaPipe -> keypoint/motion classifiers -> stable label callback."""
import csv
import threading
import time
from collections import Counter, deque

import cv2 as cv
import mediapipe as mp

from engine.gesture_zones import label_matches_zone, load_label_zones
from engine.handtracking import (
    assign_hand_histories,
    body_anchor_from_detection,
    calc_landmark_list,
    flatten_point_history,
    hand_openness,
    has_enough_motion,
    normalize_point_to_body,
    pre_process_landmark,
)
from model import KeyPointClassifier, PointHistoryClassifier

DEFAULT_LABELS_PATH = 'model/keypoint_classifier/keypoint_classifier_label.csv'
DEFAULT_MODEL_PATH = 'model/keypoint_classifier/keypoint_classifier.tflite'
DEFAULT_MOTION_LABELS_PATH = 'model/point_history_classifier/point_history_classifier_label.csv'
DEFAULT_MOTION_MODEL_PATH = 'model/point_history_classifier/point_history_classifier.tflite'
DEFAULT_ZONES_PATH = 'model/point_history_classifier/label_zones.csv'
MOTION_HISTORY_LENGTH = 45
MOTION_NONE_CLASS_ID = 0
VALID_MODES = ('letters', 'words', 'both')


def list_camera_indices(max_probe=6):
    """Probes camera indices 0..max_probe-1 and returns the ones that produce frames."""
    available = []
    for index in range(max_probe):
        cap = cv.VideoCapture(index)
        if cap is not None and cap.isOpened():
            ok, _ = cap.read()
            if ok:
                available.append(index)
        cap.release()
    return available


class GestureRecognizer:
    """Runs hand tracking + classification on its own thread and reports stable gestures."""

    def __init__(
        self,
        camera_index=0,
        width=640,
        height=480,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
        stable_votes=5,
        history_len=8,
        labels_path=DEFAULT_LABELS_PATH,
        model_path=DEFAULT_MODEL_PATH,
        motion_labels_path=DEFAULT_MOTION_LABELS_PATH,
        motion_model_path=DEFAULT_MOTION_MODEL_PATH,
        motion_stable_votes=4,
        zones_path=DEFAULT_ZONES_PATH,
        mode='words',
        on_gesture=None,
        on_frame=None,
    ):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.stable_votes = stable_votes
        self.history_len = history_len
        self.labels_path = labels_path
        self.model_path = model_path
        self.motion_labels_path = motion_labels_path
        self.motion_model_path = motion_model_path
        self.motion_stable_votes = motion_stable_votes
        self.zones_path = zones_path
        self.mode = mode if mode in VALID_MODES else 'both'
        self.on_gesture = on_gesture
        self.on_frame = on_frame

        self._thread = None
        self._stop_event = threading.Event()
        self._last_stable_label = None

    def set_mode(self, mode):
        if mode in VALID_MODES:
            self.mode = mode

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    @property
    def is_running(self):
        return bool(self._thread and self._thread.is_alive())

    def _load_labels(self):
        with open(self.labels_path, encoding='utf-8-sig') as f:
            return [row[0].strip() for row in csv.reader(f) if row and row[0].strip()]

    def _load_motion_labels(self):
        with open(self.motion_labels_path, encoding='utf-8-sig') as f:
            return [row[0].strip() for row in csv.reader(f) if row and row[0].strip()]

    def _load_zones(self):
        try:
            return load_label_zones(self.zones_path)
        except (FileNotFoundError, OSError):
            return {}

    def _run(self):
        labels = self._load_labels()
        classifier = KeyPointClassifier(model_path=self.model_path)
        motion_labels = self._load_motion_labels()
        motion_classifier = PointHistoryClassifier(model_path=self.motion_model_path)
        zones = self._load_zones()

        cap = cv.VideoCapture(self.camera_index)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.height)

        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )

        history = deque(maxlen=self.history_len)
        point_history_right = deque(maxlen=MOTION_HISTORY_LENGTH)
        point_history_left = deque(maxlen=MOTION_HISTORY_LENGTH)
        motion_history = deque(maxlen=self.motion_stable_votes)
        last_anchor = None

        try:
            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.02)
                    continue

                frame = cv.flip(frame, 1)
                rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = hands.process(rgb)
                rgb.flags.writeable = True

                current_label = None
                if results.multi_hand_landmarks:
                    hand_landmark_lists = [calc_landmark_list(frame, hl) for hl in results.multi_hand_landmarks]

                    processed = pre_process_landmark(hand_landmark_lists[0])
                    class_id = classifier(processed)
                    if 0 <= class_id < len(labels):
                        current_label = labels[class_id]
                    history.append(current_label)

                    face_results = face_detector.process(rgb)
                    if face_results.detections:
                        last_anchor = body_anchor_from_detection(
                            face_results.detections[0], frame.shape[1], frame.shape[0]
                        )

                    hand_features = []
                    for landmark_list in hand_landmark_lists:
                        rel_x, rel_y = normalize_point_to_body(landmark_list[0], last_anchor)
                        hand_features.append([rel_x, rel_y, hand_openness(landmark_list)])

                    assign_hand_histories(
                        hand_features, results.multi_handedness, point_history_right, point_history_left
                    )
                else:
                    history.clear()
                    point_history_right.append([0.0, 0.0, 0.0])
                    point_history_left.append([0.0, 0.0, 0.0])

                if self.on_frame:
                    self.on_frame(frame, results.multi_hand_landmarks)

                mode = self.mode

                stable_label = None
                if mode != 'words' and history:
                    top_label, top_count = Counter(history).most_common(1)[0]
                    if top_label and top_count >= self.stable_votes:
                        stable_label = top_label

                stable_motion_label = None
                if mode != 'letters':
                    both_ready = (
                        len(point_history_right) == MOTION_HISTORY_LENGTH
                        and len(point_history_left) == MOTION_HISTORY_LENGTH
                    )
                    enough_motion = both_ready and (
                        has_enough_motion(point_history_right) or has_enough_motion(point_history_left)
                    )
                    if enough_motion:
                        processed_history = (
                            flatten_point_history(point_history_right)
                            + flatten_point_history(point_history_left)
                        )
                        motion_id = motion_classifier(processed_history)
                        motion_history.append(motion_id)
                        top_motion_id, top_motion_count = Counter(motion_history).most_common(1)[0]
                        if (
                            top_motion_id != MOTION_NONE_CLASS_ID
                            and top_motion_count >= self.motion_stable_votes
                            and 0 <= top_motion_id < len(motion_labels)
                            and label_matches_zone(
                                motion_labels[top_motion_id], point_history_right, point_history_left, zones
                            )
                        ):
                            stable_motion_label = motion_labels[top_motion_id]
                    else:
                        motion_history.append(MOTION_NONE_CLASS_ID)

                if mode == 'letters':
                    final_label = stable_label
                elif mode == 'words':
                    final_label = stable_motion_label
                else:
                    # 'both': a recognized motion sign takes priority over a static hand-shape label.
                    final_label = stable_motion_label or stable_label

                if final_label and final_label != self._last_stable_label:
                    self._last_stable_label = final_label
                    if self.on_gesture:
                        self.on_gesture(final_label)
                elif final_label is None:
                    self._last_stable_label = None
        finally:
            hands.close()
            face_detector.close()
            cap.release()
