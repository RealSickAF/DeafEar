#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collects labeled keypoint samples for the full RSL vocabulary (not limited to 10 classes)."""
import argparse
import csv

import cv2 as cv
import mediapipe as mp

from engine.handtracking import calc_landmark_list, pre_process_landmark

LABELS_PATH = 'model/keypoint_classifier/keypoint_classifier_label.csv'
DATASET_PATH = 'model/keypoint_classifier/keypoint.csv'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0')
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--height', type=int, default=540)
    return parser.parse_args()


def load_labels():
    with open(LABELS_PATH, encoding='utf-8-sig') as f:
        return [row[0].strip() for row in csv.reader(f) if row and row[0].strip()]


def append_sample(class_id, processed_landmarks):
    with open(DATASET_PATH, 'a', newline='') as f:
        csv.writer(f).writerow([class_id, *processed_landmarks])


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
    if not labels:
        raise RuntimeError(f'No labels found in {LABELS_PATH}')

    cap_source = int(args.device) if args.device.isdigit() else args.device
    cap = cv.VideoCapture(cap_source, cv.CAP_DSHOW) if isinstance(cap_source, int) else cv.VideoCapture(cap_source)
    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)

    actual_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    print(f'[camera] requested={args.width}x{args.height} actual={actual_width}x{actual_height}')

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    class_index = 0
    recording = False
    sample_counts = [0] * len(labels)
    streak_count = 0  # samples appended since recording was last turned on; undoable with 'u'

    print('Controls: [ / ] change class, SPACE toggle recording, U undo last streak, ESC quit')

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv.flip(frame, 1)
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = hands.process(rgb)
        rgb.flags.writeable = True

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if recording:
                landmark_list = calc_landmark_list(frame, hand_landmarks)
                processed = pre_process_landmark(landmark_list)
                append_sample(class_index, processed)
                sample_counts[class_index] += 1
                streak_count += 1

        status = 'RECORDING' if recording else 'paused'
        cv.putText(frame, f'Class: {labels[class_index]} ({class_index + 1}/{len(labels)})',
                   (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv.putText(frame, f'Samples this class: {sample_counts[class_index]} [{status}]',
                   (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if streak_count > 0:
            cv.putText(frame, f'Press U to undo last {streak_count} samples',
                       (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        cv.imshow('Collect RSL Samples', frame)

        key = cv.waitKey(1)
        if key == 27:
            break
        elif key == ord(' '):
            recording = not recording
            if recording:
                streak_count = 0
        elif key in (ord('u'), ord('U')):
            recording = False
            remove_last_rows(streak_count)
            sample_counts[class_index] -= streak_count
            streak_count = 0
        elif key == ord(']'):
            class_index = (class_index + 1) % len(labels)
            recording = False
            streak_count = 0
        elif key == ord('['):
            class_index = (class_index - 1) % len(labels)
            recording = False
            streak_count = 0

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
