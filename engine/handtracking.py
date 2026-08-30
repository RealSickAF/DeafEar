#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared MediaPipe landmark preprocessing used by the recognizer and data collection tool."""
import copy
import itertools


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for landmark in landmarks.landmark:
        x = min(int(landmark.x * image_width), image_width - 1)
        y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([x, y])
    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    base_x, base_y = temp_landmark_list[0]
    for point in temp_landmark_list:
        point[0] -= base_x
        point[1] -= base_y

    flat = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max((abs(value) for value in flat), default=0) or 1
    return [value / max_value for value in flat]


def body_anchor_from_detection(detection, image_width, image_height):
    """Extracts (anchor_x, anchor_y, scale) in pixels from a MediaPipe face detection.

    anchor is the face center; scale is the face box size, used so "how far the hand moved
    from the face" stays consistent regardless of how close the person is to the camera.
    """
    bbox = detection.location_data.relative_bounding_box
    anchor_x = (bbox.xmin + bbox.width / 2) * image_width
    anchor_y = (bbox.ymin + bbox.height / 2) * image_height
    scale = max(bbox.width * image_width, bbox.height * image_height)
    return anchor_x, anchor_y, scale


def normalize_point_to_body(point, anchor):
    """Expresses a pixel point as (x, y) face-widths away from the body anchor (anchor_x, anchor_y, scale)."""
    if anchor is None:
        return [0.0, 0.0]
    anchor_x, anchor_y, scale = anchor
    if scale <= 1e-6:
        return [0.0, 0.0]
    return [(point[0] - anchor_x) / scale, (point[1] - anchor_y) / scale]


def hand_openness(landmark_list):
    """Returns 0..1: fraction of the 4 non-thumb fingers that are extended (captures hand-shape change)."""
    wrist = landmark_list[0]
    tip_ids = [8, 12, 16, 20]
    mcp_ids = [5, 9, 13, 17]

    def dist(a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    extended = 0
    for tip_id, mcp_id in zip(tip_ids, mcp_ids):
        if dist(landmark_list[tip_id], wrist) > dist(landmark_list[mcp_id], wrist) * 1.1:
            extended += 1
    return extended / 4.0


def flatten_point_history(point_history):
    """Flattens a deque of already body-relative [x, y, openness] samples into one flat list."""
    return list(itertools.chain.from_iterable(point_history))


def has_enough_motion(point_history, min_total=0.4):
    """Rejects near-static histories (e.g. a held letter pose) before running the motion classifier.

    Expects body-relative [x, y, openness] points (see normalize_point_to_body); only x, y are used.
    """
    points = [point for point in point_history if not (point[0] == 0 and point[1] == 0)]
    if len(points) < 2:
        return False

    total = 0.0
    for previous, current in zip(points, points[1:]):
        total += abs(current[0] - previous[0]) + abs(current[1] - previous[1])
    return total >= min_total


def assign_hand_histories(hand_features_list, handedness_list, right_history, left_history):
    """Routes each detected hand's [x, y, openness] feature into its own history, padding the other hand."""
    seen_right = False
    seen_left = False
    for feature, handedness in zip(hand_features_list, handedness_list):
        if handedness.classification[0].label[0:] == 'Right':
            right_history.append(feature)
            seen_right = True
        else:
            left_history.append(feature)
            seen_left = True
    if not seen_right:
        right_history.append([0.0, 0.0, 0.0])
    if not seen_left:
        left_history.append([0.0, 0.0, 0.0])
