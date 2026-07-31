#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import copy
import csv
import itertools
import time
from collections import Counter, deque

import cv2 as cv
import mediapipe as mp
import numpy as np

from model import KeyPointClassifier, PointHistoryClassifier
from utils import CvFpsCalc


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        default='0',
        help='camera device index (e.g. 0) or stream URL (e.g. http://<ip>:<port>/video)',
    )
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)
    parser.add_argument("--fps", help='requested camera fps (0 to leave driver default)', type=float, default=60.0)
    parser.add_argument("--process_scale", help='inference scale (0.4-1.0)', type=float, default=0.7)
    parser.add_argument("--inference_interval", help='run MediaPipe every N frames (1=every frame)', type=int, default=1)
    parser.add_argument("--model_complexity", help='mediapipe hand model complexity (0=fast,1=accurate)', type=int, default=0)
    parser.add_argument("--max_num_hands", help='max hands to track', type=int, default=2)
    parser.add_argument("--show_debug_details", action='store_true', help='show raw/corrected hand-sign debug details')
    parser.add_argument("--show_perf_overlay", action='store_true', help='show capture/inference/post timings overlay')
    parser.add_argument("--fast_visuals", action='store_true', help='simplify overlays for higher FPS')
    parser.add_argument("--normalize_lighting", action='store_true', help='normalize frame lighting before hand detection')

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.5)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=float,
                        default=0.3)

    args = parser.parse_args()
    return args


def parse_capture_source(device_arg):
    device_value = str(device_arg).strip()
    return int(device_value) if device_value.isdigit() else device_value


def fourcc_to_string(value):
    value = int(value)
    raw = ''.join([chr((value >> 8 * i) & 0xFF) for i in range(4)])
    cleaned = ''.join(ch for ch in raw if 32 <= ord(ch) <= 126)
    return cleaned


def _gamma_lut(gamma):
    gamma = max(0.1, float(gamma))
    table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(256)], dtype=np.uint8)
    return table


def normalize_frame_lighting(image):
    ycrcb = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
    y_channel = ycrcb[:, :, 0]

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y_channel = clahe.apply(y_channel)

    mean_luma = float(np.mean(y_channel))
    if mean_luma > 155.0:
        y_channel = cv.LUT(y_channel, _gamma_lut(1.22))
    elif mean_luma < 85.0:
        y_channel = cv.LUT(y_channel, _gamma_lut(0.82))

    ycrcb[:, :, 0] = y_channel
    return cv.cvtColor(ycrcb, cv.COLOR_YCrCb2BGR)


def get_label_text(index, labels, fallback=''):
    if index is None:
        return fallback

    try:
        index = int(index)
    except (TypeError, ValueError):
        return fallback

    if 0 <= index < len(labels):
        return labels[index]
    return fallback


def resolve_hand_sign_text(hand_sign_id, labels, is_right_hand):
    text = get_label_text(hand_sign_id, labels)
    if text == 'Semiopen':
        return 'sima dumoku' if is_right_hand else 'sima dumortu'
    return text


def dist(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def hand_bbox(landmark_list):
    xs = [point[0] for point in landmark_list]
    ys = [point[1] for point in landmark_list]
    return [min(xs), min(ys), max(xs), max(ys)]


def hand_scale(landmark_list):
    bbox = hand_bbox(landmark_list)
    width = max(1.0, float(bbox[2] - bbox[0]))
    height = max(1.0, float(bbox[3] - bbox[1]))
    return max(width, height)


def is_plausible_hand_shape(landmark_list):
    bbox = hand_bbox(landmark_list)
    width = max(1.0, float(bbox[2] - bbox[0]))
    height = max(1.0, float(bbox[3] - bbox[1]))
    scale = max(width, height)
    aspect = max(width / height, height / width)

    # Reject tiny/noisy blobs and extreme elongated shapes that often come from body false positives.
    return scale >= 45.0 and aspect <= 3.0


def hand_center(landmark_list):
    bbox = hand_bbox(landmark_list)
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def bbox_iou(bbox_a, bbox_b):
    inter_w = max(0.0, min(bbox_a[2], bbox_b[2]) - max(bbox_a[0], bbox_b[0]))
    inter_h = max(0.0, min(bbox_a[3], bbox_b[3]) - max(bbox_a[1], bbox_b[1]))
    inter_area = inter_w * inter_h

    area_a = max(1.0, float((bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])))
    area_b = max(1.0, float((bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])))
    union_area = max(1.0, area_a + area_b - inter_area)
    return inter_area / union_area


def finger_extension_count(landmark_list):
    wrist = landmark_list[0]
    tip_ids = [8, 12, 16, 20]
    mcp_ids = [5, 9, 13, 17]
    count = 0
    for tip_id, mcp_id in zip(tip_ids, mcp_ids):
        if dist(landmark_list[tip_id], wrist) > dist(landmark_list[mcp_id], wrist) * 1.10:
            count += 1
    return count


def is_closed_fist(landmark_list):
    wrist = landmark_list[0]
    tip_ids = [8, 12, 16, 20]
    mcp_ids = [5, 9, 13, 17]

    avg_tip_dist = np.mean([dist(landmark_list[idx], wrist) for idx in tip_ids])
    avg_mcp_dist = np.mean([dist(landmark_list[idx], wrist) for idx in mcp_ids])

    thumb_tip = landmark_list[4]
    thumb_mcp = landmark_list[2]
    thumb_folded = dist(thumb_tip, wrist) < dist(thumb_mcp, wrist) * 1.05

    extensions = finger_extension_count(landmark_list)
    return avg_tip_dist < avg_mcp_dist * 1.2 and thumb_folded and extensions <= 1


def looks_like_compact_fist(landmark_list):
    wrist = landmark_list[0]
    tip_ids = [8, 12, 16, 20]
    mcp_ids = [5, 9, 13, 17]

    avg_tip_dist = np.mean([dist(landmark_list[idx], wrist) for idx in tip_ids])
    avg_mcp_dist = np.mean([dist(landmark_list[idx], wrist) for idx in mcp_ids])
    extensions = finger_extension_count(landmark_list)

    return avg_tip_dist < avg_mcp_dist * 1.32 and extensions <= 2


def closed_fist_anchor_point(landmark_list):
    # Knuckle center is more stable than wrist when fists are near face/shoulders.
    knuckle_ids = [5, 9, 13, 17]
    x = int(np.mean([landmark_list[idx][0] for idx in knuckle_ids]))
    y = int(np.mean([landmark_list[idx][1] for idx in knuckle_ids]))
    return [x, y]


def looks_like_semiopen(landmark_list):
    if is_closed_fist(landmark_list):
        return False
    extensions = finger_extension_count(landmark_list)
    return 1 <= extensions <= 3


def looks_like_pointer_pose(landmark_list):
    wrist = landmark_list[0]
    index_tip = landmark_list[8]
    index_mcp = landmark_list[5]
    middle_tip = landmark_list[12]
    ring_tip = landmark_list[16]
    pinky_tip = landmark_list[20]
    middle_mcp = landmark_list[9]
    ring_mcp = landmark_list[13]
    pinky_mcp = landmark_list[17]

    index_extended = dist(index_tip, wrist) > dist(index_mcp, wrist) * 1.18
    other_folded = (
        dist(middle_tip, wrist) < dist(middle_mcp, wrist) * 1.08 and
        dist(ring_tip, wrist) < dist(ring_mcp, wrist) * 1.08 and
        dist(pinky_tip, wrist) < dist(pinky_mcp, wrist) * 1.08
    )
    return index_extended and other_folded


def looks_like_ok_sign(landmark_list):
    wrist = landmark_list[0]
    thumb_tip = landmark_list[4]
    index_tip = landmark_list[8]
    middle_tip = landmark_list[12]
    ring_tip = landmark_list[16]
    pinky_tip = landmark_list[20]
    middle_mcp = landmark_list[9]
    ring_mcp = landmark_list[13]
    pinky_mcp = landmark_list[17]

    size = hand_scale(landmark_list)
    thumb_index_touch = dist(thumb_tip, index_tip) < (0.28 * size)
    middle_extended = dist(middle_tip, wrist) > dist(middle_mcp, wrist) * 1.15
    ring_extended = dist(ring_tip, wrist) > dist(ring_mcp, wrist) * 1.12
    pinky_extended = dist(pinky_tip, wrist) > dist(pinky_mcp, wrist) * 1.10

    return thumb_index_touch and middle_extended and ring_extended and pinky_extended


def motion_stats(point_history, recent_len=12):
    points = [point for point in point_history if point[0] != 0 and point[1] != 0]
    if len(points) < 3:
        return 0.0, 0.0, 0.0, np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    recent_points = points[-recent_len:]
    xs = np.array([point[0] for point in recent_points], dtype=np.float32)
    ys = np.array([point[1] for point in recent_points], dtype=np.float32)

    dx = np.diff(xs)
    dy = np.diff(ys)
    total_abs_motion = float(np.sum(np.abs(dx)) + np.sum(np.abs(dy)))
    range_x = float(np.max(xs) - np.min(xs))
    range_y = float(np.max(ys) - np.min(ys))
    return total_abs_motion, range_x, range_y, dx, dy


def has_meaningful_recent_motion(point_history, min_total=22.0, min_span=6.0):
    total_abs_motion, range_x, range_y, _, _ = motion_stats(point_history)
    return total_abs_motion >= float(min_total) and max(range_x, range_y) >= float(min_span)


def looks_like_stroking_motion(point_history):
    total_abs_motion, range_x, range_y, dx, dy = motion_stats(point_history)
    if total_abs_motion < 26.0:
        return False

    sum_abs_dx = float(np.sum(np.abs(dx)))
    sum_abs_dy = float(np.sum(np.abs(dy)))
    dominant_velocity = dx if sum_abs_dx >= sum_abs_dy else dy
    dominant_span = range_x if sum_abs_dx >= sum_abs_dy else range_y

    if float(np.sum(np.abs(dominant_velocity))) < 20.0 or dominant_span < 7.0:
        return False

    meaningful = dominant_velocity[np.abs(dominant_velocity) > 1.5]
    if len(meaningful) < 4:
        return False

    signs = np.sign(meaningful)
    sign_changes = int(np.sum(signs[1:] * signs[:-1] < 0))
    return sign_changes >= 1


def detect_two_hand_andrei_cojoc(hand_a_landmark_list, hand_b_landmark_list):
    if not hand_a_landmark_list or not hand_b_landmark_list:
        return False

    hand_a_bbox = hand_bbox(hand_a_landmark_list)
    hand_b_bbox = hand_bbox(hand_b_landmark_list)

    hand_a_middle_tip = hand_a_landmark_list[12]
    hand_a_middle_mcp = hand_a_landmark_list[9]
    hand_b_middle_tip = hand_b_landmark_list[12]
    hand_b_middle_mcp = hand_b_landmark_list[9]

    hand_a_middle_length = dist(hand_a_middle_tip, hand_a_middle_mcp)
    hand_b_middle_length = dist(hand_b_middle_tip, hand_b_middle_mcp)

    hand_a_extensions = finger_extension_count(hand_a_landmark_list)
    hand_b_extensions = finger_extension_count(hand_b_landmark_list)

    hand_a_size = hand_scale(hand_a_landmark_list)
    hand_b_size = hand_scale(hand_b_landmark_list)
    hand_scale_max = max(hand_a_size, hand_b_size)

    hand_a_padding = max(25.0, 0.18 * hand_a_size)
    hand_b_padding = max(25.0, 0.18 * hand_b_size)

    hand_b_tip_inside_hand_a = (
        hand_a_bbox[0] - hand_a_padding <= hand_b_middle_tip[0] <= hand_a_bbox[2] + hand_a_padding and
        hand_a_bbox[1] - hand_a_padding <= hand_b_middle_tip[1] <= hand_a_bbox[3] + hand_a_padding
    )

    hand_a_tip_inside_hand_b = (
        hand_b_bbox[0] - hand_b_padding <= hand_a_middle_tip[0] <= hand_b_bbox[2] + hand_b_padding and
        hand_b_bbox[1] - hand_b_padding <= hand_a_middle_tip[1] <= hand_b_bbox[3] + hand_b_padding
    )

    hand_a_center = hand_center(hand_a_landmark_list)
    hand_b_center = hand_center(hand_b_landmark_list)
    wrist_distance = dist(hand_a_landmark_list[0], hand_b_landmark_list[0])
    iou = bbox_iou(hand_a_bbox, hand_b_bbox)
    center_distance = dist(hand_a_center, hand_b_center)

    middle_tips_close = dist(hand_a_middle_tip, hand_b_middle_tip) < (0.95 * hand_scale_max)

    middle_fingers_extended = (
        hand_a_middle_length > (0.18 * hand_a_size) and
        hand_b_middle_length > (0.18 * hand_b_size)
    )

    both_hands_active = hand_a_extensions >= 2 and hand_b_extensions >= 2
    distinct_hands = (
        center_distance > (0.22 * hand_scale_max) and
        (iou < 0.85 or wrist_distance > (0.30 * hand_scale_max))
    )

    return (
        both_hands_active and
        middle_fingers_extended and
        middle_tips_close and
        distinct_hands and
        (hand_b_tip_inside_hand_a or hand_a_tip_inside_hand_b or center_distance < (1.4 * hand_scale_max))
    )


def looks_like_vertical_stroking_motion(point_history):
    total_abs_motion, range_x, range_y, dx, dy = motion_stats(point_history)
    if total_abs_motion < 26.0:
        return False

    sum_abs_dx = float(np.sum(np.abs(dx)))
    sum_abs_dy = float(np.sum(np.abs(dy)))
    if sum_abs_dy < 20.0 or range_y < 8.0:
        return False

    if sum_abs_dx > sum_abs_dy * 0.90 or range_x > range_y * 1.15:
        return False

    meaningful = dy[np.abs(dy) > 1.5]
    if len(meaningful) < 4:
        return False

    signs = np.sign(meaningful)
    sign_changes = int(np.sum(signs[1:] * signs[:-1] < 0))
    return sign_changes >= 1


def detect_two_hand_paunescu_stroke(right_hand_landmark_list, left_hand_landmark_list, point_history_right, point_history_left, image_shape):
    if right_hand_landmark_list is None or left_hand_landmark_list is None:
        return False

    if not (is_plausible_hand_shape(right_hand_landmark_list) and is_plausible_hand_shape(left_hand_landmark_list)):
        return False

    right_vertical = looks_like_vertical_stroking_motion(point_history_right)
    left_vertical = looks_like_vertical_stroking_motion(point_history_left)
    right_stroking = looks_like_stroking_motion(point_history_right)
    left_stroking = looks_like_stroking_motion(point_history_left)

    # At least one clearly vertical hand and both hands in some stroking motion.
    if not ((right_vertical or left_vertical) and right_stroking and left_stroking):
        return False

    right_motion_ok = has_meaningful_recent_motion(point_history_right, min_total=20.0, min_span=6.0)
    left_motion_ok = has_meaningful_recent_motion(point_history_left, min_total=20.0, min_span=6.0)
    if not (right_motion_ok and left_motion_ok):
        return False

    right_center = hand_center(right_hand_landmark_list)
    left_center = hand_center(left_hand_landmark_list)

    horizontal_span = abs(right_center[0] - left_center[0])
    vertical_gap = abs(right_center[1] - left_center[1])

    # Long-motion trigger: both hands moving together, not tied to a specific face/neck zone.
    hands_together = (0.05 * image_shape[1]) <= horizontal_span <= (0.85 * image_shape[1])
    similar_height = vertical_gap <= (0.35 * image_shape[0])

    return hands_together and similar_height


def detect_two_hand_paunescu_mouth_pose(right_hand_landmark_list, left_hand_landmark_list, image_shape):
    if right_hand_landmark_list is None or left_hand_landmark_list is None:
        return False

    image_h, image_w = image_shape[0], image_shape[1]
    right_center = hand_center(right_hand_landmark_list)
    left_center = hand_center(left_hand_landmark_list)

    horizontal_span = abs(right_center[0] - left_center[0])
    vertical_gap = abs(right_center[1] - left_center[1])

    # Neck-only trigger: both hands must independently sit in the upper-neck band.
    right_in_neck_band = (0.18 * image_h) <= right_center[1] <= (0.52 * image_h)
    left_in_neck_band = (0.18 * image_h) <= left_center[1] <= (0.52 * image_h)
    close_enough_side_by_side = (0.10 * image_w) <= horizontal_span <= (0.62 * image_w)
    similar_height = vertical_gap <= (0.22 * image_h)

    return right_in_neck_band and left_in_neck_band and close_enough_side_by_side and similar_height


def detect_single_hand_outward_sweep(landmark_list, point_history, image_shape):
    if landmark_list is None or point_history is None:
        return False

    if not is_plausible_hand_shape(landmark_list):
        return False

    points = [point for point in point_history if point[0] != 0 and point[1] != 0]
    if len(points) < 4:
        return False

    image_h, image_w = image_shape[0], image_shape[1]
    hand_x, hand_y = hand_center(landmark_list)

    # Start low and sweep outward from the torso/groin area.
    in_lower_torso_band = (0.40 * image_h) <= hand_y <= (0.95 * image_h)
    if not in_lower_torso_band:
        return False

    total_abs_motion, range_x, range_y, dx, dy = motion_stats(point_history)
    if total_abs_motion < 8.0:
        return False

    sum_abs_dx = float(np.sum(np.abs(dx)))
    sum_abs_dy = float(np.sum(np.abs(dy)))
    if sum_abs_dx < sum_abs_dy * 0.55:
        return False

    if range_x < 6.0 or range_y > range_x * 1.50:
        return False

    recent_points = points[-4:]
    xs = np.array([point[0] for point in recent_points], dtype=np.float32)
    start_x = float(xs[0])
    end_x = float(xs[-1])
    mid_x = image_w * 0.5

    if hand_x < mid_x:
        outward = end_x <= start_x - 3.0 or float(np.max(xs)) <= start_x + 10.0
    else:
        outward = end_x >= start_x + 3.0 or float(np.min(xs)) >= start_x - 10.0

    return outward


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    max_value = max(list(map(abs, temp_landmark_list))) if temp_landmark_list else 1

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list


def pre_process_point_history(image, point_history, mirror_x=False):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        normalized_x = (temp_point_history[index][0] - base_x) / image_width
        if mirror_x:
            normalized_x = -normalized_x
        temp_point_history[index][0] = normalized_x
        temp_point_history[index][1] = (temp_point_history[index][1] - base_y) / image_height

    temp_point_history = list(itertools.chain.from_iterable(temp_point_history))
    return temp_point_history


def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        return
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])


def append_training_sample(class_id, landmark_list):
    if class_id is None:
        return
    csv_path = 'model/keypoint_classifier/keypoint.csv'
    with open(csv_path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([class_id, *landmark_list])


def draw_landmarks(image, landmark_point, fast=False):
    if fast:
        if len(landmark_point) > 0:
            fast_edges = [
                (0, 5), (5, 9), (9, 13), (13, 17),
                (0, 1), (1, 2), (2, 3), (3, 4),
                (5, 6), (6, 7), (7, 8),
                (9, 10), (10, 11), (11, 12),
                (13, 14), (14, 15), (15, 16),
                (17, 18), (18, 19), (19, 20),
            ]
            for start_index, end_index in fast_edges:
                cv.line(image, tuple(landmark_point[start_index]), tuple(landmark_point[end_index]), (220, 220, 220), 1, cv.LINE_AA)

            for index in (0, 4, 8, 12, 16, 20):
                landmark = landmark_point[index]
                cv.circle(image, (landmark[0], landmark[1]), 3, (255, 255, 255), -1, cv.LINE_AA)
        return image

    if len(landmark_point) > 0:
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (255, 255, 255), 2)

        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (255, 255, 255), 2)

        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (255, 255, 255), 2)

        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (255, 255, 255), 2)

        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (255, 255, 255), 2)

        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (255, 255, 255), 2)

    for index, landmark in enumerate(landmark_point):
        if index == 0:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
    return image


def draw_info_text(image, brect, handedness, hand_sign_text, finger_gesture_text, debug_text='', show_handedness=True, show_finger_text=True, landmark_list=None, simplify_effects=False):
    info_text = handedness.classification[0].label[0:] if show_handedness else ''
    if hand_sign_text != "":
        info_text = hand_sign_text if info_text == '' else info_text + ': ' + hand_sign_text

    is_cheery_text = hand_sign_text != '' and 'SanatateNumaiBile' in hand_sign_text
    is_fire_text = (
        hand_sign_text != '' and (
            'sima dumoku' in hand_sign_text or
            'sima dumortu' in hand_sign_text or
            'Semiopen' in hand_sign_text
        )
    )

    text_x = max(10, brect[0] + 6)
    text_y = max(30, brect[1] - 8)
    if is_cheery_text or is_fire_text:
        text_y = min(image.shape[0] - 20, brect[3] + 28)
    if is_fire_text:
        text_y = max(26, brect[1] - 22)

    if simplify_effects:
        cv.putText(image, info_text, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 5, cv.LINE_AA)
        cv.putText(image, info_text, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv.LINE_AA)
    elif is_cheery_text:
        draw_arch_text(
            image,
            info_text,
            (text_x + 10, text_y + 10),
            radius=max(72, min(140, len(info_text) * 7)),
            font_scale=1.22,
            thickness=4,
            outline_thickness=9,
            colors=cheery_colors(),
            phase_offset=0.3,
            sparkle=True,
            glitter=False,
        )

    elif is_fire_text:
        draw_fire_text(image, info_text, (text_x, text_y), font_scale=1.18, thickness=4, outline_thickness=9)
        if landmark_list is not None:
            tip_ids = [4, 8, 12, 16, 20]
            palette = [
                (0, 0, 110),
                (0, 20, 180),
                (0, 60, 230),
                (0, 120, 255),
            ]
            for index, tip_id in enumerate(tip_ids):
                tip_x, tip_y = landmark_list[tip_id]
                for flame_index, scale in enumerate((1.0, 0.75, 0.5)):
                    plume_height = int((28 + flame_index * 10) * scale)
                    plume_width = int((10 + flame_index * 4) * scale)
                    top_y = max(0, tip_y - plume_height)
                    left_x = max(0, tip_x - plume_width // 2)
                    right_x = min(image.shape[1] - 1, tip_x + plume_width // 2)
                    color = palette[(index + flame_index) % len(palette)]
                    cv.ellipse(image, (tip_x, max(0, tip_y - 8 - flame_index * 2)), (max(3, plume_width // 2), max(10, plume_height // 2)), 0, 0, 180, color, -1, cv.LINE_AA)
                    cv.line(image, (tip_x, tip_y), (tip_x, top_y), color, 2, cv.LINE_AA)
                    cv.line(image, (left_x, tip_y - 2), (tip_x, top_y), (0, 0, 255), 1, cv.LINE_AA)
                    cv.line(image, (right_x, tip_y - 2), (tip_x, top_y), (0, 60, 255), 1, cv.LINE_AA)
    else:
        cv.putText(image, info_text, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 8, cv.LINE_AA)
        cv.putText(image, info_text, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv.LINE_AA)

    if show_finger_text and finger_gesture_text != "":
        finger_text_x = max(10, brect[0] + 2)
        finger_text_y = max(30, brect[1] - 18)

        if finger_gesture_text == 'PAUNESCU' and not simplify_effects:
            text_size, _ = cv.getTextSize('PAUNESCU', cv.FONT_HERSHEY_SIMPLEX, 1.55, 6)
            text_width = text_size[0]
            finger_text_x = max(10, min(image.shape[1] - text_width - 10, brect[0]))
            draw_slime_text(image, 'PAUNESCU', (finger_text_x, finger_text_y), font_scale=1.55, thickness=6, outline_thickness=13)
            motion_y = min(image.shape[0] - 10, finger_text_y + 28)
            cv.putText(image, 'motion', (finger_text_x + 4, motion_y), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 5, cv.LINE_AA)
            cv.putText(image, 'motion', (finger_text_x + 4, motion_y), cv.FONT_HERSHEY_SIMPLEX, 0.55, (190, 255, 190), 1, cv.LINE_AA)
        else:
            cv.putText(image, "Finger: " + finger_gesture_text, (finger_text_x, finger_text_y), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 7, cv.LINE_AA)
            cv.putText(image, "Finger: " + finger_gesture_text, (finger_text_x, finger_text_y), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv.LINE_AA)

    if debug_text != '':
        debug_y = min(image.shape[0] - 8, brect[3] + 18)
        cv.putText(image, debug_text, (text_x, debug_y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, debug_text, (text_x, debug_y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)

    return image


def rainbow_colors():
    return [
        (255, 0, 0),
        (255, 128, 0),
        (255, 255, 0),
        (0, 200, 0),
        (0, 180, 255),
        (0, 0, 255),
        (180, 0, 255),
    ]


def draw_rainbow_text(image, text, origin, font_scale=1.2, thickness=4, outline_thickness=9):
    x, y = origin
    colors = rainbow_colors()

    for index, character in enumerate(text):
        char_width, _ = cv.getTextSize(character, cv.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        color = colors[index % len(colors)]

        cv.putText(image, character, (x, y), cv.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), outline_thickness, cv.LINE_AA)
        cv.putText(image, character, (x, y), cv.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv.LINE_AA)

        x += char_width + 1

    return image


def fire_colors():
    return [
        (0, 0, 120),
        (0, 0, 170),
        (0, 20, 210),
        (0, 45, 235),
        (0, 70, 255),
    ]


def draw_fire_text(image, text, origin, font_scale=1.15, thickness=4, outline_thickness=9):
    x, y = origin
    colors = fire_colors()
    phase = time.time() * 12.0
    flicker = 1.0 + (0.06 * np.sin(phase))
    drift_y = int(1 * np.sin(phase * 1.7))

    for index, character in enumerate(text):
        char_scale = font_scale * (flicker + 0.02 * np.sin(phase + index))
        (char_width, char_height), _ = cv.getTextSize(character, cv.FONT_HERSHEY_SIMPLEX, char_scale, thickness)
        color = colors[(index + int(abs(np.sin(phase)) * 2)) % len(colors)]

        cv.putText(image, character, (x, y + drift_y), cv.FONT_HERSHEY_SIMPLEX, char_scale, (0, 0, 0), outline_thickness, cv.LINE_AA)
        cv.putText(image, character, (x + 1, y + drift_y - 1), cv.FONT_HERSHEY_SIMPLEX, char_scale, (0, 35, 220), outline_thickness // 2, cv.LINE_AA)
        cv.putText(image, character, (x, y + drift_y), cv.FONT_HERSHEY_SIMPLEX, char_scale, color, thickness, cv.LINE_AA)

        flame_x = x + max(4, char_width // 2)
        flame_y = y + drift_y - char_height - 4 - int(4 * abs(np.sin(phase + index * 0.7)))
        cv.line(image, (flame_x, flame_y + 10), (flame_x, flame_y), (0, 0, 255), 2, cv.LINE_AA)
        cv.line(image, (flame_x, flame_y + 7), (flame_x - 2, flame_y + 1), (0, 60, 255), 1, cv.LINE_AA)
        cv.line(image, (flame_x, flame_y + 7), (flame_x + 2, flame_y + 1), (0, 60, 255), 1, cv.LINE_AA)

        x += char_width + 1

    return image


def slime_colors():
    return [
        (0, 90, 0),
        (0, 130, 0),
        (40, 170, 40),
        (70, 210, 70),
        (120, 255, 120),
    ]


def draw_slime_text(image, text, origin, font_scale=1.45, thickness=5, outline_thickness=11):
    x, y = origin
    colors = slime_colors()
    phase = time.time() * 4.0

    for index, character in enumerate(text):
        char_scale = font_scale * (1.0 + 0.03 * np.sin(phase + index * 0.35))
        (char_width, char_height), _ = cv.getTextSize(character, cv.FONT_HERSHEY_SIMPLEX, char_scale, thickness)
        color = colors[(index + int(abs(np.sin(phase)) * 2)) % len(colors)]

        wobble_y = int(3 * np.sin(phase + index * 0.5))
        cv.putText(image, character, (x, y + wobble_y), cv.FONT_HERSHEY_SIMPLEX, char_scale, (0, 0, 0), outline_thickness, cv.LINE_AA)
        cv.putText(image, character, (x + 1, y + wobble_y - 1), cv.FONT_HERSHEY_SIMPLEX, char_scale, (10, 40, 10), outline_thickness // 2, cv.LINE_AA)
        cv.putText(image, character, (x, y + wobble_y), cv.FONT_HERSHEY_SIMPLEX, char_scale, color, thickness, cv.LINE_AA)

        drip_top = y + wobble_y + int(char_height * 0.15)
        drip_bottom = drip_top + 12 + int(10 * abs(np.sin(phase + index)))
        drip_x = x + char_width // 2
        cv.line(image, (drip_x, drip_top), (drip_x, drip_bottom), (0, 70, 0), 4, cv.LINE_AA)
        cv.circle(image, (drip_x, drip_bottom), 4, (0, 130, 0), -1, cv.LINE_AA)
        cv.circle(image, (drip_x + 5, drip_bottom + 2), 2, (120, 255, 120), -1, cv.LINE_AA)

        slime_bubble_x = x + char_width - 4
        slime_bubble_y = y + wobble_y - char_height - 6
        cv.circle(image, (slime_bubble_x, slime_bubble_y), 3, (120, 255, 120), -1, cv.LINE_AA)
        cv.circle(image, (slime_bubble_x - 8, slime_bubble_y + 4), 2, (70, 210, 70), -1, cv.LINE_AA)

        x += char_width + 2

    return image


def draw_glitter_text(image, text, origin, colors, font_scale=1.18, thickness=4, outline_thickness=10, sparkle=True, confetti=False):
    x, y = origin
    phase = time.time() * 5.0

    for index, character in enumerate(text):
        char_scale = font_scale * (1.0 + 0.02 * np.sin(phase + index * 0.35))
        (char_width, char_height), _ = cv.getTextSize(character, cv.FONT_HERSHEY_SIMPLEX, char_scale, thickness)
        color = colors[(index + int(abs(np.sin(phase)) * len(colors))) % len(colors)]

        cv.putText(image, character, (x, y), cv.FONT_HERSHEY_SIMPLEX, char_scale, (0, 0, 0), outline_thickness, cv.LINE_AA)
        cv.putText(image, character, (x + 1, y - 1), cv.FONT_HERSHEY_SIMPLEX, char_scale, (255, 255, 255), outline_thickness // 3, cv.LINE_AA)
        cv.putText(image, character, (x, y), cv.FONT_HERSHEY_SIMPLEX, char_scale, color, thickness, cv.LINE_AA)

        if sparkle:
            spark_y = y - char_height - 6
            cv.circle(image, (x + char_width // 2 + 8, spark_y), 2, (255, 255, 255), -1, cv.LINE_AA)
            cv.circle(image, (x + char_width // 2 - 6, spark_y + 4), 1, (255, 255, 255), -1, cv.LINE_AA)

        if confetti:
            confetti_phase = phase + index * 0.45
            for offset_index, radius in enumerate((1, 2, 1)):
                confetti_x = x + char_width // 2 + int(np.cos(confetti_phase + offset_index) * (8 + offset_index * 4))
                confetti_y = y - char_height - int(6 + offset_index * 3 + 3 * np.sin(confetti_phase + offset_index))
                confetti_color = colors[(index + offset_index) % len(colors)]
                cv.circle(image, (confetti_x, confetti_y), radius, confetti_color, -1, cv.LINE_AA)

        x += char_width + 2

    return image


def cheery_colors():
    return [
        (0, 255, 255),
        (0, 255, 180),
        (120, 255, 120),
        (255, 255, 120),
        (255, 210, 120),
        (255, 170, 220),
    ]


def draw_cheery_text(image, text, origin, font_scale=1.12, thickness=4, outline_thickness=9):
    x, y = origin
    colors = cheery_colors()
    phase = time.time() * 6.0
    bounce = int(4 * np.sin(phase))
    pulse = 1.0 + (0.04 * abs(np.sin(phase * 1.4)))

    for index, character in enumerate(text):
        char_scale = font_scale * (pulse + 0.015 * np.sin(phase + index * 0.5))
        (char_width, char_height), _ = cv.getTextSize(character, cv.FONT_HERSHEY_SIMPLEX, char_scale, thickness)
        color = colors[(index + int(abs(np.sin(phase)) * 3)) % len(colors)]

        cv.putText(image, character, (x, y + bounce), cv.FONT_HERSHEY_SIMPLEX, char_scale, (0, 0, 0), outline_thickness, cv.LINE_AA)
        cv.putText(image, character, (x + 1, y + bounce - 1), cv.FONT_HERSHEY_SIMPLEX, char_scale, (180, 180, 180), outline_thickness // 2, cv.LINE_AA)
        cv.putText(image, character, (x, y + bounce), cv.FONT_HERSHEY_SIMPLEX, char_scale, color, thickness, cv.LINE_AA)

        spark_x = x + max(4, char_width // 2)
        spark_y = y + bounce - char_height - 5
        cv.circle(image, (spark_x, spark_y), 2, (255, 255, 255), -1, cv.LINE_AA)
        cv.circle(image, (spark_x, spark_y), 5, (255, 255, 180), 1, cv.LINE_AA)

        confetti_phase = phase + index * 0.4
        for offset_index, radius in enumerate((1, 2, 1)):
            confetti_x = spark_x + int(np.cos(confetti_phase + offset_index) * (8 + offset_index * 4))
            confetti_y = spark_y - int(6 + offset_index * 3 + 3 * np.sin(confetti_phase + offset_index))
            confetti_color = colors[(index + offset_index) % len(colors)]
            cv.circle(image, (confetti_x, confetti_y), radius, confetti_color, -1, cv.LINE_AA)

        x += char_width + 1

    return image


def pink_glitter_colors():
    return [
        (255, 105, 180),
        (255, 128, 193),
        (255, 182, 214),
        (255, 192, 203),
        (255, 160, 220),
    ]


def draw_arch_text(image, text, center, radius=70, font_scale=1.12, thickness=4, outline_thickness=9, colors=None, phase_offset=0.0, sparkle=False, glitter=False):
    if text == "":
        return image

    colors = colors or pink_glitter_colors()
    cx, cy = center
    count = len(text)
    char_widths = [cv.getTextSize(character, cv.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0] for character in text]
    total_width = sum(char_widths) + max(0, count - 1) * int(font_scale * 6)
    effective_radius = max(radius, int(total_width * 0.75))
    arc_span = min(np.pi - 0.25, max(1.6, total_width / max(1.0, float(effective_radius)) * 1.2))
    if count == 1:
        angles = [np.pi / 2]
    else:
        angles = np.linspace((np.pi / 2) + (arc_span / 2.0), (np.pi / 2) - (arc_span / 2.0), count)

    phase = time.time() * 5.0 + phase_offset

    for index, character in enumerate(text):
        angle = float(angles[index])
        wobble = 4.0 * np.sin(phase + index * 0.45)
        char_x = int(cx + np.cos(angle) * (effective_radius + wobble))
        char_y = int(cy - np.sin(angle) * (effective_radius + wobble))
        char_scale = font_scale * (1.0 + 0.02 * np.sin(phase + index * 0.6))
        color = colors[(index + int(abs(np.sin(phase)) * len(colors))) % len(colors)]

        cv.putText(image, character, (char_x, char_y), cv.FONT_HERSHEY_SIMPLEX, char_scale, (0, 0, 0), outline_thickness, cv.LINE_AA)
        cv.putText(image, character, (char_x, char_y), cv.FONT_HERSHEY_SIMPLEX, char_scale, color, thickness, cv.LINE_AA)

        if sparkle:
            spark_r = 2 + int(abs(np.sin(phase + index)) * 2)
            cv.circle(image, (char_x + 8, char_y - 10), spark_r, (255, 255, 255), -1, cv.LINE_AA)
            cv.circle(image, (char_x - 7, char_y + 4), 1, (255, 255, 255), -1, cv.LINE_AA)

        if glitter:
            glitter_phase = phase + index * 0.8
            for offset_index, radius_offset in enumerate((1, 2, 1)):
                gx = char_x + int(np.cos(glitter_phase + offset_index) * (7 + offset_index * 3))
                gy = char_y - int(7 + offset_index * 2 + 3 * np.sin(glitter_phase + offset_index))
                gcolor = colors[(index + offset_index) % len(colors)]
                cv.circle(image, (gx, gy), radius_offset, gcolor, -1, cv.LINE_AA)

    return image


def draw_shared_text(image, text, hand_draw_infos=None, simplify_effects=False):
    if text == "":
        return image

    text_x = 12
    text_y = 120
    if hand_draw_infos:
        left = min(info[0][0] for info in hand_draw_infos)
        top = min(info[0][1] for info in hand_draw_infos)
        right = max(info[0][2] for info in hand_draw_infos)
        bottom = max(info[0][3] for info in hand_draw_infos)
        center_x = (left + right) // 2
        text_x = max(10, center_x)
        text_y = min(image.shape[0] - 20, max(40, bottom + 24))

    if simplify_effects:
        cv.putText(image, text, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 6, cv.LINE_AA)
        cv.putText(image, text, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)
        return image

    if text == 'PAUNESCU':
        if hand_draw_infos:
            left = min(info[0][0] for info in hand_draw_infos)
            top = min(info[0][1] for info in hand_draw_infos)
            right = max(info[0][2] for info in hand_draw_infos)
            bottom = max(info[0][3] for info in hand_draw_infos)
            center_x = (left + right) // 2
            center_y = (top + bottom) // 2
            text_size, _ = cv.getTextSize('PAUNESCU', cv.FONT_HERSHEY_SIMPLEX, 1.45, 6)
            text_x = max(10, min(image.shape[1] - text_size[0] - 10, center_x - text_size[0] // 2))
            text_y = min(image.shape[0] - 18, max(44, center_y + 14))

        draw_slime_text(image, 'PAUNESCU', (text_x, text_y), font_scale=1.45, thickness=6, outline_thickness=13)
        return image

    text_size, _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 1.18, 4)
    text_width = text_size[0]
    text_x = max(10, text_x - text_width // 2)

    draw_glitter_text(
        image,
        text,
        (text_x, text_y),
        pink_glitter_colors(),
        font_scale=1.18,
        thickness=4,
        outline_thickness=10,
        sparkle=True,
        confetti=True,
    )
    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2), (152, 251, 152), -1)
    return image


def draw_info(image, fps, mode, number, perf_text=''):
    cv.putText(image, "FPS: " + str(fps), (12, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 6, cv.LINE_AA)
    cv.putText(image, "FPS: " + str(fps), (12, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv.LINE_AA)
    if perf_text:
        cv.putText(image, perf_text, (12, 52), cv.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, perf_text, (12, 52), cv.FONT_HERSHEY_SIMPLEX, 0.58, (173, 255, 47), 1, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Motion Record (Point History)']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE: " + mode_string[mode - 1], (12, 78), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "CLASS: " + str(number), (12, 100), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(image, "J: handedness on/off", (12, image.shape[0] - 12), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "J: handedness on/off", (12, image.shape[0] - 12), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv.LINE_AA)
    cv.putText(image, "F: finger text on/off", (12, image.shape[0] - 34), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "F: finger text on/off", (12, image.shape[0] - 34), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv.LINE_AA)
    return image


def main():
    args = get_args()

    cap_device = parse_capture_source(args.device)
    cap_width = args.width
    cap_height = args.height
    cap_fps = max(0.0, float(args.fps))

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence
    process_scale = min(max(args.process_scale, 0.4), 1.0)
    inference_interval = max(1, int(args.inference_interval))
    model_complexity = 0 if args.model_complexity <= 0 else 1
    max_num_hands = max(1, min(args.max_num_hands, 2))
    show_debug_details = args.show_debug_details
    show_perf_overlay = args.show_perf_overlay
    fast_visuals = args.fast_visuals
    lighting_normalization_enabled = args.normalize_lighting

    use_brect = True

    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
    if cap_fps > 0.0:
        cap.set(cv.CAP_PROP_FPS, cap_fps)

    if cap is None or not cap.isOpened():
        raise RuntimeError(
            f'Unable to open camera source {cap_device!r}. '
            'Use a valid webcam index (0, 1, 2, ...) or a reachable stream URL.'
        )

    startup_ok = False
    for _ in range(30):
        ret, _ = cap.read()
        if ret:
            startup_ok = True
            break
        time.sleep(0.05)

    if not startup_ok:
        raise RuntimeError(
            f'Camera source {cap_device!r} opened but returned no frames. '
            'If this is a virtual camera, open its desktop app and select a live source. '
            'Otherwise try another index (for this PC, 2 is a working physical camera).'
        )

    actual_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    actual_fps = float(cap.get(cv.CAP_PROP_FPS))
    actual_fourcc = fourcc_to_string(cap.get(cv.CAP_PROP_FOURCC)).strip('\x00')
    print(
        f'[camera] requested={cap_width}x{cap_height}@{cap_fps:.2f}fps '
        f'actual={actual_width}x{actual_height}@{actual_fps:.2f}fps codec={actual_fourcc or "unknown"}'
    )

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=max_num_hands,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier(score_th=0.20)

    with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        keypoint_classifier_labels = [row[0] for row in csv.reader(f)]
    with open('model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
        point_history_classifier_labels = [row[0] for row in csv.reader(f)]

    cvFpsCalc = CvFpsCalc(buffer_len=10)

    history_length = 16
    point_history_right = deque(maxlen=history_length)
    point_history_left = deque(maxlen=history_length)
    finger_gesture_history_right = deque(maxlen=history_length)
    finger_gesture_history_left = deque(maxlen=history_length)
    hand_sign_history_right = deque(maxlen=5)
    hand_sign_history_left = deque(maxlen=5)
    handedness_history = deque(maxlen=2)
    left_hand_landmark_list = None
    right_hand_landmark_list = None
    mode = 0
    number = -1
    record_custom_gesture = False
    record_key_pressed = False
    shared_two_hand_text = ''
    two_hand_match_streak = 0
    paunescu_two_hand_streak = 0
    read_failures = 0
    max_read_failures = 30
    show_handedness = True
    show_finger_text = True
    no_hand_frames = 0
    max_no_hand_grace_frames = 14
    last_landmark_list = None
    last_brect = None
    right_missing_frames = 0
    left_missing_frames = 0
    max_single_hand_grace_frames = 20
    max_right_hand_grace_frames = 24
    max_left_hand_grace_frames = 30
    recent_hand_context_frames = 8
    right_closed_fist_hold = 0
    left_closed_fist_hold = 0
    max_closed_fist_hold_frames = 12
    right_anchor_prev = None
    left_anchor_prev = None
    max_anchor_jump_scale = 2.20
    right_track_center = None
    left_track_center = None
    paunescu_display_hold_frames = 0
    max_paunescu_display_hold_frames = 9
    paunescu_anchor_hand_draw_infos = None
    inference_frame_index = 0
    cached_results = None
    perf_capture_ms = deque(maxlen=15)
    perf_inference_ms = deque(maxlen=15)
    perf_post_ms = deque(maxlen=15)
    perf_total_ms = deque(maxlen=15)

    while True:
        fps = cvFpsCalc.get()

        key = cv.waitKey(1)
        if key == 27:
            break
        if key in (ord('j'), ord('J')):
            show_handedness = not show_handedness
            continue
        if key in (ord('f'), ord('F')):
            show_finger_text = not show_finger_text
            continue
        if key == 2490368:
            record_key_pressed = True
            record_custom_gesture = True
            number = -1
        elif key == -1:
            if record_key_pressed:
                record_custom_gesture = True
            else:
                record_custom_gesture = False
        else:
            number, mode = select_mode(key, mode)
            if record_key_pressed:
                record_custom_gesture = True

        frame_start = time.perf_counter()

        capture_start = time.perf_counter()
        ret, image = cap.read()
        capture_ms = (time.perf_counter() - capture_start) * 1000.0
        if not ret:
            read_failures += 1
            if read_failures >= max_read_failures:
                raise RuntimeError(
                    f'Lost frames from camera source {cap_device!r}. '
                    'Check camera app/source availability and then restart.'
                )
            time.sleep(0.02)
            continue
        read_failures = 0

        image = cv.flip(image, 1)
        debug_image = image.copy()

        inference_start = time.perf_counter()
        run_inference = (inference_frame_index % inference_interval == 0) or (cached_results is None)
        if run_inference:
            process_image = image
            if process_scale < 0.99:
                process_image = cv.resize(image, None, fx=process_scale, fy=process_scale, interpolation=cv.INTER_LINEAR)
            if lighting_normalization_enabled:
                process_image = normalize_frame_lighting(process_image)

            image_rgb = cv.cvtColor(process_image, cv.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = hands.process(image_rgb)
            image_rgb.flags.writeable = True
            cached_results = results
            inference_ms = (time.perf_counter() - inference_start) * 1000.0
        else:
            results = cached_results
            inference_ms = 0.0
        inference_frame_index += 1

        post_start = time.perf_counter()

        hand_draw_infos = []
        shared_two_hand_text = ''
        paunescu_motion_match = False
        paunescu_anchor_hand_draw_infos = None

        if results.multi_hand_landmarks is not None:
            no_hand_frames = 0
            handedness_history.clear()
            custom_gesture_sample_ready = False
            detected_hand_landmark_lists = []
            assigned_slots = set()
            seen_right = False
            seen_left = False

            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                last_landmark_list = landmark_list
                last_brect = brect

                handedness_history.append(handedness.classification[0].label[0:])

                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                predicted_is_right_hand = handedness.classification[0].label[0:] == 'Right'
                current_center = hand_center(landmark_list)
                current_scale = hand_scale(landmark_list)

                is_right_hand = predicted_is_right_hand
                if right_track_center is not None and left_track_center is not None:
                    distance_to_right = dist(current_center, right_track_center)
                    distance_to_left = dist(current_center, left_track_center)
                    margin = max(8.0, 0.18 * current_scale)
                    if distance_to_right + margin < distance_to_left:
                        is_right_hand = True
                    elif distance_to_left + margin < distance_to_right:
                        is_right_hand = False
                elif right_track_center is not None and left_track_center is None:
                    if dist(current_center, right_track_center) < (2.2 * current_scale):
                        is_right_hand = True
                elif left_track_center is not None and right_track_center is None:
                    if dist(current_center, left_track_center) < (2.2 * current_scale):
                        is_right_hand = False

                # Enforce unique lane ownership per frame to avoid both detections collapsing into one side.
                if is_right_hand and 'right' in assigned_slots and 'left' not in assigned_slots:
                    is_right_hand = False
                elif (not is_right_hand) and 'left' in assigned_slots and 'right' not in assigned_slots:
                    is_right_hand = True

                current_point_history = point_history_right if is_right_hand else point_history_left
                current_gesture_history = finger_gesture_history_right if is_right_hand else finger_gesture_history_left
                current_sign_history = hand_sign_history_right if is_right_hand else hand_sign_history_left

                if is_right_hand:
                    assigned_slots.add('right')
                    seen_right = True
                    right_missing_frames = 0
                    right_hand_landmark_list = landmark_list
                    right_track_center = current_center
                else:
                    assigned_slots.add('left')
                    seen_left = True
                    left_missing_frames = 0
                    left_hand_landmark_list = landmark_list
                    left_track_center = current_center
                detected_hand_landmark_lists.append(landmark_list)

                if mode == 1:
                    if 0 <= number < len(keypoint_classifier_labels):
                        append_training_sample(number, pre_processed_landmark_list)
                    elif record_custom_gesture and not custom_gesture_sample_ready:
                        if len(detected_hand_landmark_lists) >= 2 and detect_two_hand_andrei_cojoc(
                            detected_hand_landmark_lists[0],
                            detected_hand_landmark_lists[1],
                        ):
                            append_training_sample(7, pre_processed_landmark_list)
                            custom_gesture_sample_ready = True

                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                raw_sign_id = hand_sign_id
                raw_label = get_label_text(raw_sign_id, keypoint_classifier_labels)
                extension_count = finger_extension_count(landmark_list)

                # Class 7 is a dedicated two-hand gesture; block single-hand predictions.
                if hand_sign_id == 7:
                    hand_sign_id = -1

                predicted_label = get_label_text(hand_sign_id, keypoint_classifier_labels)
                if predicted_label == 'OK' and not looks_like_ok_sign(landmark_list):
                    if is_closed_fist(landmark_list):
                        hand_sign_id = 1
                        predicted_label = get_label_text(hand_sign_id, keypoint_classifier_labels)

                closed_fist_now = is_closed_fist(landmark_list)
                compact_fist_now = looks_like_compact_fist(landmark_list)
                if is_right_hand:
                    if closed_fist_now or compact_fist_now:
                        right_closed_fist_hold = max_closed_fist_hold_frames
                    else:
                        right_closed_fist_hold = max(0, right_closed_fist_hold - 1)
                    closed_fist_stable = right_closed_fist_hold > 0
                else:
                    if closed_fist_now or compact_fist_now:
                        left_closed_fist_hold = max_closed_fist_hold_frames
                    else:
                        left_closed_fist_hold = max(0, left_closed_fist_hold - 1)
                    closed_fist_stable = left_closed_fist_hold > 0

                # Keep post-processing minimal to avoid suppressing valid right-hand classes.

                other_point_history = point_history_left if is_right_hand else point_history_right
                both_hands_context = (
                    right_hand_landmark_list is not None and
                    left_hand_landmark_list is not None
                )
                other_hand_motion_ok = has_meaningful_recent_motion(other_point_history, min_total=18.0, min_span=5.0)
                paunescu_context_ok = both_hands_context and other_hand_motion_ok

                if hand_sign_id != -1:
                    current_sign_history.append(hand_sign_id)
                    top_id, top_count = Counter(current_sign_history).most_common(1)[0]
                    required_votes = 2
                    if top_count >= required_votes:
                        hand_sign_id = top_id

                final_label = get_label_text(hand_sign_id, keypoint_classifier_labels)
                debug_sign_text = ''
                if show_debug_details:
                    debug_sign_text = f'raw:{raw_label} final:{final_label} ext:{extension_count}'

                hand_sign_text = ''
                if hand_sign_id != -1:
                    hand_sign_text = resolve_hand_sign_text(
                        hand_sign_id,
                        keypoint_classifier_labels,
                        is_right_hand,
                    )

                # Keep motion history active in motion mode and for pointer/semi-closed poses.
                # For closed fists, fingertip landmarks are often unstable; use wrist as a robust anchor.
                motion_anchor_name = 'none'
                if mode == 2 or hand_sign_id == 2 or looks_like_semiopen(landmark_list):
                    use_wrist_anchor = closed_fist_stable if mode == 2 else closed_fist_now
                    motion_anchor_name = 'knuckle' if use_wrist_anchor else 'tip'
                    motion_anchor_point = closed_fist_anchor_point(landmark_list) if use_wrist_anchor else landmark_list[8]

                    prev_anchor = right_anchor_prev if is_right_hand else left_anchor_prev
                    anchor_jump_limit = hand_scale(landmark_list) * max_anchor_jump_scale
                    if prev_anchor is not None and dist(motion_anchor_point, prev_anchor) > anchor_jump_limit:
                        motion_anchor_point = prev_anchor
                        motion_anchor_name = f'{motion_anchor_name}h'

                    if is_right_hand:
                        right_anchor_prev = motion_anchor_point
                    else:
                        left_anchor_prev = motion_anchor_point
                    current_point_history.append(motion_anchor_point)
                else:
                    current_point_history.append([0, 0])
                    if is_right_hand:
                        right_anchor_prev = None
                    else:
                        left_anchor_prev = None

                pre_processed_point_history_list = pre_process_point_history(
                    debug_image,
                    current_point_history,
                    mirror_x=False,
                )
                logging_csv(number, mode, pre_processed_landmark_list, pre_processed_point_history_list)

                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(pre_processed_point_history_list)

                current_gesture_history.append(finger_gesture_id)
                recent_window = 4 if not is_right_hand else 6
                recent_gesture_ids = list(current_gesture_history)[-recent_window:]
                smoothed_finger_gesture_id = 0
                recent_paunescu_votes = recent_gesture_ids.count(4)
                paunescu_pose_ok = closed_fist_stable or looks_like_semiopen(landmark_list)
                paunescu_motion_ok = has_meaningful_recent_motion(current_point_history, min_total=22.0, min_span=5.5)
                if recent_paunescu_votes >= 2 and paunescu_pose_ok and paunescu_motion_ok and paunescu_context_ok:
                    smoothed_finger_gesture_id = 4
                else:
                    non_zero_recent_ids = [gesture_id for gesture_id in recent_gesture_ids if gesture_id != 0]
                    if non_zero_recent_ids:
                        smoothed_finger_gesture_id = Counter(non_zero_recent_ids).most_common(1)[0][0]

                if not is_right_hand and finger_gesture_id != 0:
                    left_raw_recent = list(current_gesture_history)[-3:]
                    if left_raw_recent.count(finger_gesture_id) >= 2:
                        smoothed_finger_gesture_id = finger_gesture_id

                # Fallback: if raw predictions include class 4 recently, surface PAUNESCU.
                raw_recent_window = 6 if not is_right_hand else 10
                raw_recent_gesture_ids = list(current_gesture_history)[-raw_recent_window:]
                if (
                    smoothed_finger_gesture_id == 0
                    and raw_recent_gesture_ids.count(4) >= 1
                    and paunescu_pose_ok
                    and paunescu_motion_ok
                    and paunescu_context_ok
                ):
                    smoothed_finger_gesture_id = 4

                # If the model collapses to generic Move (3), promote to PAUNESCU on stroke-like dynamics.
                if smoothed_finger_gesture_id == 3 and looks_like_stroking_motion(current_point_history) and paunescu_context_ok:
                    smoothed_finger_gesture_id = 4

                finger_gesture_text = get_label_text(
                    smoothed_finger_gesture_id,
                    point_history_classifier_labels,
                )

                single_hand_outward_match = mode == 2 and detect_single_hand_outward_sweep(
                    landmark_list,
                    current_point_history,
                    debug_image.shape,
                )
                paunescu_motion_fallback = single_hand_outward_match
                if single_hand_outward_match:
                    smoothed_finger_gesture_id = 4
                    finger_gesture_text = get_label_text(4, point_history_classifier_labels)
                    paunescu_motion_match = True

                if show_debug_details:
                    motion_raw_label = get_label_text(finger_gesture_id, point_history_classifier_labels)
                    fist_hold = right_closed_fist_hold if is_right_hand else left_closed_fist_hold
                    debug_sign_text = f'{debug_sign_text} mraw:{motion_raw_label} mph:{1 if paunescu_motion_fallback else 0} ma:{motion_anchor_name} cfh:{fist_hold}'

                current_draw_info = (brect, handedness, hand_sign_text, finger_gesture_text, debug_sign_text, landmark_list)
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list, fast=fast_visuals)
                hand_draw_infos.append(current_draw_info)
                if single_hand_outward_match:
                    paunescu_anchor_hand_draw_infos = [current_draw_info]

            if not seen_right:
                right_missing_frames += 1
                if right_missing_frames > max_right_hand_grace_frames:
                    right_hand_landmark_list = None
                    right_track_center = None
                elif right_hand_landmark_list is not None:
                    if mode == 2:
                        point_history_right.append(point_history_right[-1] if len(point_history_right) > 0 else [0, 0])
                    right_bbox = hand_bbox(right_hand_landmark_list)
                    debug_image = draw_bounding_rect(use_brect, debug_image, right_bbox)
                    debug_image = draw_landmarks(debug_image, right_hand_landmark_list, fast=fast_visuals)

            if not seen_left:
                left_missing_frames += 1
                if left_missing_frames > max_left_hand_grace_frames:
                    left_hand_landmark_list = None
                    left_track_center = None
                elif left_hand_landmark_list is not None:
                    if mode == 2:
                        point_history_left.append(point_history_left[-1] if len(point_history_left) > 0 else [0, 0])
                    left_bbox = hand_bbox(left_hand_landmark_list)
                    debug_image = draw_bounding_rect(use_brect, debug_image, left_bbox)
                    debug_image = draw_landmarks(debug_image, left_hand_landmark_list, fast=fast_visuals)

            custom_match = (
                len(detected_hand_landmark_lists) >= 2 and
                detect_two_hand_andrei_cojoc(
                    detected_hand_landmark_lists[0],
                    detected_hand_landmark_lists[1],
                )
            )

            if custom_match:
                two_hand_match_streak = min(two_hand_match_streak + 1, 8)
            else:
                two_hand_match_streak = max(two_hand_match_streak - 1, 0)

            if paunescu_motion_match:
                paunescu_two_hand_streak = min(paunescu_two_hand_streak + 1, 8)
            else:
                paunescu_two_hand_streak = max(paunescu_two_hand_streak - 1, 0)

            if paunescu_two_hand_streak >= 1:
                paunescu_display_hold_frames = max_paunescu_display_hold_frames
                shared_two_hand_text = 'PAUNESCU'
            elif paunescu_display_hold_frames > 0:
                paunescu_display_hold_frames -= 1
                shared_two_hand_text = 'PAUNESCU'
            elif two_hand_match_streak >= 4:
                shared_two_hand_text = 'Andrei.slattttttt'
            else:
                paunescu_display_hold_frames = max(0, paunescu_display_hold_frames - 1)
        else:
            no_hand_frames += 1
            two_hand_match_streak = max(two_hand_match_streak - 1, 0)
            paunescu_two_hand_streak = max(paunescu_two_hand_streak - 1, 0)
            paunescu_display_hold_frames = max(0, paunescu_display_hold_frames - 1)
            right_missing_frames = min(max_right_hand_grace_frames + 1, right_missing_frames + 1)
            left_missing_frames = min(max_left_hand_grace_frames + 1, left_missing_frames + 1)

            if no_hand_frames <= max_no_hand_grace_frames:
                # Keep short continuity through brief tracker drops.
                if len(point_history_right) > 0 and point_history_right[-1] != [0, 0]:
                    point_history_right.append(point_history_right[-1])
                else:
                    point_history_right.append([0, 0])

                if len(point_history_left) > 0 and point_history_left[-1] != [0, 0]:
                    point_history_left.append(point_history_left[-1])
                else:
                    point_history_left.append([0, 0])
            else:
                point_history_right.append([0, 0])
                point_history_left.append([0, 0])
                hand_sign_history_right.clear()
                hand_sign_history_left.clear()

            if no_hand_frames <= max_no_hand_grace_frames and last_landmark_list is not None and last_brect is not None:
                debug_image = draw_bounding_rect(use_brect, debug_image, last_brect)
                debug_image = draw_landmarks(debug_image, last_landmark_list, fast=fast_visuals)

        if mode != 2 and results.multi_hand_landmarks is not None and len(handedness_history) == 1:
            if 'Right' not in handedness_history:
                point_history_right.clear()
                for _ in range(history_length):
                    point_history_right.append([0, 0])
                finger_gesture_history_right.clear()
                hand_sign_history_right.clear()
            if 'Left' not in handedness_history:
                point_history_left.clear()
                for _ in range(history_length):
                    point_history_left.append([0, 0])
                finger_gesture_history_left.clear()
                hand_sign_history_left.clear()

        if not fast_visuals:
            debug_image = draw_point_history(debug_image, point_history_right)
            debug_image = draw_point_history(debug_image, point_history_left)
        if shared_two_hand_text != '':
            debug_image = draw_shared_text(
                debug_image,
                shared_two_hand_text,
                paunescu_anchor_hand_draw_infos or hand_draw_infos,
                simplify_effects=fast_visuals,
            )
        else:
            for brect, handedness, hand_sign_text, finger_gesture_text, debug_sign_text, landmark_list in hand_draw_infos:
                # In mouth-pose mode, avoid lone-hand PAUNESCU labels when two-hand trigger is not active.
                if finger_gesture_text == 'PAUNESCU':
                    finger_gesture_text = ''
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    hand_sign_text,
                    finger_gesture_text,
                    debug_sign_text,
                    show_handedness,
                    show_finger_text,
                    landmark_list,
                    simplify_effects=fast_visuals,
                )

        post_ms = (time.perf_counter() - post_start) * 1000.0
        total_ms = (time.perf_counter() - frame_start) * 1000.0
        perf_capture_ms.append(capture_ms)
        perf_inference_ms.append(inference_ms)
        perf_post_ms.append(post_ms)
        perf_total_ms.append(total_ms)

        perf_text = ''
        if show_perf_overlay and len(perf_total_ms) > 0:
            avg_capture = sum(perf_capture_ms) / len(perf_capture_ms)
            avg_inference = sum(perf_inference_ms) / len(perf_inference_ms)
            avg_post = sum(perf_post_ms) / len(perf_post_ms)
            avg_total = sum(perf_total_ms) / len(perf_total_ms)
            avg_budget_fps = 1000.0 / max(1.0, avg_total)
            perf_text = (
                f'cap:{avg_capture:.1f}ms inf:{avg_inference:.1f}ms '
                f'post:{avg_post:.1f}ms total:{avg_total:.1f}ms ({avg_budget_fps:.1f}fps) '
                f'infN:{inference_interval}'
            )

        debug_image = draw_info(debug_image, fps, mode, number, perf_text)

        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
