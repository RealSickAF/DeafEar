#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Local test window: shows the camera feed with hand landmarks, the recognized gesture, and speaks it out loud."""
import argparse
import csv
import queue
import threading
from collections import Counter, deque

import cv2 as cv
import mediapipe as mp

from audio.output_router import list_output_devices, play
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
from service.dictionary import build_gender_pairs, build_index, closest_word, inflect_for_gender, load_wordlist
from service.fingerspell_buffer import DEFAULT_CONFIRM_DELAY, DEFAULT_FINALIZE_TIMEOUT, FingerspellBuffer
from service.translator_service import label_to_phrase, load_phrase_map
from tts.speech_engine import synthesize

LABELS_PATH = 'model/keypoint_classifier/keypoint_classifier_label.csv'
MOTION_LABELS_PATH = 'model/point_history_classifier/point_history_classifier_label.csv'
MOTION_HISTORY_LENGTH = 45
MOTION_NONE_CLASS_ID = 0
MODES = ('words', 'letters', 'both')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', help='camera index or stream URL')
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--height', type=int, default=540)
    parser.add_argument('--voice', choices=['male', 'female'], default='female')
    parser.add_argument('--output_device', default=None, help='playback device index (default = system speakers)')
    parser.add_argument('--stable_votes', type=int, default=5)
    parser.add_argument('--history_len', type=int, default=8)
    parser.add_argument('--motion_stable_votes', type=int, default=4)
    parser.add_argument('--letter_hold_delay', type=float, default=DEFAULT_CONFIRM_DELAY,
                        help='seconds a letter must stay stable before being added to the spelled word')
    parser.add_argument('--word_pause', type=float, default=DEFAULT_FINALIZE_TIMEOUT,
                        help='seconds of silence after the last letter before resolving the word')
    return parser.parse_args()


def load_labels():
    with open(LABELS_PATH, encoding='utf-8-sig') as f:
        return [row[0].strip() for row in csv.reader(f) if row and row[0].strip()]


def load_motion_labels():
    with open(MOTION_LABELS_PATH, encoding='utf-8-sig') as f:
        return [row[0].strip() for row in csv.reader(f) if row and row[0].strip()]


def start_speech_worker(voice, output_device_index):
    speech_queue = queue.Queue()

    def worker():
        while True:
            phrase = speech_queue.get()
            try:
                pcm, rate = synthesize(phrase, gender=voice)
                play(pcm, rate, device_index=output_device_index)
            except Exception as exc:
                print(f'[speech error] {exc}')

    threading.Thread(target=worker, daemon=True).start()
    return speech_queue


def main():
    args = get_args()
    labels = load_labels()
    motion_labels = load_motion_labels()
    phrase_map = load_phrase_map()
    output_device_index = int(args.output_device) if args.output_device is not None else None

    print('Available output devices:')
    for device in list_output_devices():
        print(f"  [{device['index']}] {device['name']}")

    speech_queue = start_speech_worker(args.voice, output_device_index)

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
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    face_detector = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    classifier = KeyPointClassifier()
    motion_classifier = PointHistoryClassifier()
    try:
        zones = load_label_zones()
    except (FileNotFoundError, OSError):
        zones = {}
    history = deque(maxlen=args.history_len)
    point_history_right = deque(maxlen=MOTION_HISTORY_LENGTH)
    point_history_left = deque(maxlen=MOTION_HISTORY_LENGTH)
    motion_history = deque(maxlen=args.motion_stable_votes)
    last_stable_label = None
    mode_index = 0
    last_anchor = None
    spelling_status = ''

    wordlist = load_wordlist()
    word_index = build_index(wordlist)
    gender_pairs = build_gender_pairs(wordlist)

    def on_letter_progress(spelled_so_far):
        nonlocal spelling_status
        spelling_status = f'Spelling: {spelled_so_far}'

    def on_word_resolved(spelled, word):
        nonlocal spelling_status
        spelling_status = f'Word: {spelled} -> "{word}"'
        speech_queue.put(word)

    def resolve_word(spelled):
        word = closest_word(spelled, word_index) or spelled.lower()
        return inflect_for_gender(word, args.voice, gender_pairs)

    fingerspell = FingerspellBuffer(
        resolve_word_fn=resolve_word,
        on_word=on_word_resolved,
        on_letter=on_letter_progress,
        confirm_delay=args.letter_hold_delay,
        finalize_timeout=args.word_pause,
    )

    print('Press ESC to quit, M to switch mode (words / letters / both).')

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        mode = MODES[mode_index]
        frame = cv.flip(frame, 1)
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = hands.process(rgb)
        rgb.flags.writeable = True

        raw_label = None
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            hand_landmark_lists = [calc_landmark_list(frame, hl) for hl in results.multi_hand_landmarks]

            processed = pre_process_landmark(hand_landmark_lists[0])
            class_id = classifier(processed)
            if 0 <= class_id < len(labels):
                raw_label = labels[class_id]
            history.append(raw_label)

            face_results = face_detector.process(rgb)
            if face_results.detections:
                last_anchor = body_anchor_from_detection(face_results.detections[0], frame.shape[1], frame.shape[0])

            hand_features = []
            for landmark_list in hand_landmark_lists:
                rel_x, rel_y = normalize_point_to_body(landmark_list[0], last_anchor)
                hand_features.append([rel_x, rel_y, hand_openness(landmark_list)])

            assign_hand_histories(hand_features, results.multi_handedness, point_history_right, point_history_left)
        else:
            history.clear()
            point_history_right.append([0.0, 0.0, 0.0])
            point_history_left.append([0.0, 0.0, 0.0])

        if last_anchor is not None:
            cv.circle(frame, (int(last_anchor[0]), int(last_anchor[1])), 4, (255, 0, 255), -1)

        stable_label = None
        if mode != 'words' and history:
            top_label, top_count = Counter(history).most_common(1)[0]
            if top_label and top_count >= args.stable_votes:
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
                    and top_motion_count >= args.motion_stable_votes
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
            final_label = stable_motion_label or stable_label

        if mode == 'letters':
            if final_label and final_label != last_stable_label:
                last_stable_label = final_label
                fingerspell.feed(final_label)
            elif final_label is None:
                last_stable_label = None
        else:
            if final_label and final_label != last_stable_label:
                last_stable_label = final_label
                speech_queue.put(label_to_phrase(final_label, phrase_map))
            elif final_label is None:
                last_stable_label = None

        cv.putText(frame, f'Raw: {raw_label or "-"}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv.putText(frame, f'Stable: {final_label or "-"}', (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv.putText(frame, f'Mode: {mode} (press M to switch)', (10, 120), cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        if mode == 'letters' and spelling_status:
            cv.putText(frame, spelling_status, (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        elif final_label:
            cv.putText(frame, f'Speaking: "{label_to_phrase(final_label, phrase_map)}"',
                       (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        face_scale_text = f'{last_anchor[2]:.0f}px' if last_anchor is not None else '-'
        right_xy = point_history_right[-1] if point_history_right else [0.0, 0.0, 0.0]
        left_xy = point_history_left[-1] if point_history_left else [0.0, 0.0, 0.0]
        cv.putText(frame, f'Face size: {face_scale_text} (changes with distance)',
                   (10, 150), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
        cv.putText(frame, f'R hand: ({right_xy[0]:+.2f}, {right_xy[1]:+.2f}) face-widths',
                   (10, 172), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
        cv.putText(frame, f'L hand: ({left_xy[0]:+.2f}, {left_xy[1]:+.2f}) face-widths (should stay steady as you move)',
                   (10, 194), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)

        cv.imshow('DeafEar - Test Preview', frame)
        key = cv.waitKey(1)
        if key == 27:
            break
        elif key in (ord('m'), ord('M')):
            mode_index = (mode_index + 1) % len(MODES)
            history.clear()
            motion_history.clear()
            last_stable_label = None
            fingerspell.reset()
            spelling_status = ''

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
