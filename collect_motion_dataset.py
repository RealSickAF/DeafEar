#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collects motion-trajectory samples (dynamic signs like SALUT) for the point-history classifier."""
import argparse
import csv
from collections import deque

import cv2 as cv
import mediapipe as mp

from engine.handtracking import (
    assign_hand_histories,
    body_anchor_from_detection,
    calc_landmark_list,
    flatten_point_history,
    hand_openness,
    normalize_point_to_body,
)

LABELS_PATH = 'model/point_history_classifier/point_history_classifier_label.csv'
DATASET_PATH = 'model/point_history_classifier/point_history.csv'
HISTORY_LENGTH = 45


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0')
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--height', type=int, default=540)
    parser.add_argument('--classes', type=int, default=None,
                         help='only cycle through the first N labels (handy for a quick test)')
    return parser.parse_args()


def load_labels():
    with open(LABELS_PATH, encoding='utf-8-sig') as f:
        return [row[0].strip() for row in csv.reader(f) if row and row[0].strip()]


def append_sample(class_id, processed_history):
    with open(DATASET_PATH, 'a', newline='') as f:
        csv.writer(f).writerow([class_id, *processed_history])


def remove_last_rows(count):
    if count <= 0:
        return
    with open(DATASET_PATH) as f:
        lines = f.readlines()
    with open(DATASET_PATH, 'w', newline='') as f:
        f.writelines(lines[:-count] if count < len(lines) else [])


def main():
    args = get_args()
    labels = load_labels()
    if args.classes:
        labels = labels[:args.classes]
    if not labels:
        raise RuntimeError(f'No labels found in {LABELS_PATH}')

    cap_source = int(args.device) if args.device.isdigit() else args.device
    cap = cv.VideoCapture(cap_source)
    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))

    actual_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv.CAP_PROP_FPS)
    fourcc_int = int(cap.get(cv.CAP_PROP_FOURCC))
    actual_fourcc = ''.join(chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)).strip('\x00') or 'unknown'
    print(f'[camera] requested={args.width}x{args.height} actual={actual_width}x{actual_height}'
          f'@{actual_fps:.1f}fps codec={actual_fourcc}')

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    face_detector = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    class_index = 0
    capturing = False
    frames_captured = 0
    flash_message = ''
    flash_frames_left = 0
    sample_counts = [0] * len(labels)
    point_history_right = deque(maxlen=HISTORY_LENGTH)
    point_history_left = deque(maxlen=HISTORY_LENGTH)
    last_anchor = None

    print('Controls: [ / ] change sign, SPACE capture one repetition, U undo last capture, ESC quit')
    print('Press SPACE right as you START the motion - the next 45 frames (~1.5s) are captured')
    print('as ONE sample, so each press = exactly one clean repetition, no overlap ambiguity.')
    print('For two-handed signs, keep both hands visible in frame throughout the motion.')
    print('Keep your face visible throughout - the magenta dot is the body reference point.')

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv.flip(frame, 1)
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = hands.process(rgb)
        rgb.flags.writeable = True

        face_results = face_detector.process(rgb)
        if face_results.detections:
            last_anchor = body_anchor_from_detection(face_results.detections[0], frame.shape[1], frame.shape[0])

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            hand_landmark_lists = [calc_landmark_list(frame, hl) for hl in results.multi_hand_landmarks]

            hand_features = []
            for landmark_list in hand_landmark_lists:
                rel_x, rel_y = normalize_point_to_body(landmark_list[0], last_anchor)
                hand_features.append([rel_x, rel_y, hand_openness(landmark_list)])

            assign_hand_histories(hand_features, results.multi_handedness, point_history_right, point_history_left)
        else:
            point_history_right.append([0.0, 0.0, 0.0])
            point_history_left.append([0.0, 0.0, 0.0])

        if last_anchor is not None:
            cv.circle(frame, (int(last_anchor[0]), int(last_anchor[1])), 4, (255, 0, 255), -1)

        if capturing:
            frames_captured += 1
            if frames_captured >= HISTORY_LENGTH:
                processed = flatten_point_history(point_history_right) + flatten_point_history(point_history_left)
                append_sample(class_index, processed)
                sample_counts[class_index] += 1
                capturing = False
                flash_message = f'Captured! (total: {sample_counts[class_index]})'
                flash_frames_left = 30

        status = f'CAPTURING {frames_captured}/{HISTORY_LENGTH}' if capturing else 'ready (press SPACE)'
        cv.putText(frame, f'Sign: {labels[class_index]} ({class_index + 1}/{len(labels)})',
                   (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv.putText(frame, f'Samples this sign: {sample_counts[class_index]} [{status}]',
                   (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if flash_frames_left > 0:
            cv.putText(frame, flash_message, (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            flash_frames_left -= 1
        cv.imshow('Collect RSL Motion Samples', frame)

        key = cv.waitKey(1)
        if key == 27:
            break
        elif key == ord(' ') and not capturing:
            point_history_right.clear()
            point_history_left.clear()
            capturing = True
            frames_captured = 0
        elif key in (ord('u'), ord('U')):
            capturing = False
            if sample_counts[class_index] > 0:
                remove_last_rows(1)
                sample_counts[class_index] -= 1
                flash_message = f'Undone (total: {sample_counts[class_index]})'
                flash_frames_left = 30
        elif key == ord(']'):
            class_index = (class_index + 1) % len(labels)
            capturing = False
        elif key == ord('['):
            class_index = (class_index - 1) % len(labels)
            capturing = False

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
