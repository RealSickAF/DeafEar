#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Runtime gate that rejects a predicted gesture label if the currently tracked hand(s) don't
match how that label was actually recorded (which hand(s), and roughly where relative to the
face). Built from model/point_history_classifier/label_zones.csv (see build_label_zones.py).

This exists because the classifier alone will happily score a full 101-way softmax from
whatever it's given, including a one-handed motion that partially resembles a two-handed sign,
or a chest-height sign performed near the head - this adds a hard, data-derived sanity check
on top of that softmax result.
"""
import csv

ZONES_PATH = 'model/point_history_classifier/label_zones.csv'
PRESENCE_RATIO_THRESHOLD = 0.5
BOUNDS_MARGIN_RATIO = 0.35
MIN_BOUNDS_MARGIN = 0.3


def _bounds(row, side):
    keys = (f'{side}_x_min', f'{side}_x_max', f'{side}_y_min', f'{side}_y_max')
    if any(row[key] == '' for key in keys):
        return None
    return tuple(float(row[key]) for key in keys)


def load_label_zones(path=ZONES_PATH):
    zones = {}
    with open(path, encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            zones[row['label']] = {
                'right_required': row['right_required'] == '1',
                'right_bounds': _bounds(row, 'right'),
                'left_required': row['left_required'] == '1',
                'left_bounds': _bounds(row, 'left'),
            }
    return zones


def _hand_ok(hand_history, required, bounds):
    present = [(x, y) for x, y, _openness in hand_history if not (x == 0.0 and y == 0.0)]
    is_present = bool(hand_history) and len(present) / len(hand_history) >= PRESENCE_RATIO_THRESHOLD

    if is_present != required:
        return False
    if not required or not present or not bounds:
        return True

    x_min, x_max, y_min, y_max = bounds
    pad_x = max((x_max - x_min) * BOUNDS_MARGIN_RATIO, MIN_BOUNDS_MARGIN)
    pad_y = max((y_max - y_min) * BOUNDS_MARGIN_RATIO, MIN_BOUNDS_MARGIN)
    avg_x = sum(p[0] for p in present) / len(present)
    avg_y = sum(p[1] for p in present) / len(present)
    return (x_min - pad_x) <= avg_x <= (x_max + pad_x) and (y_min - pad_y) <= avg_y <= (y_max + pad_y)


def label_matches_zone(label, point_history_right, point_history_left, zones):
    """True if the current hand presence/position is consistent with how `label` was recorded."""
    zone = zones.get(label)
    if not zone:
        return True
    return (
        _hand_ok(point_history_right, zone['right_required'], zone['right_bounds'])
        and _hand_ok(point_history_left, zone['left_required'], zone['left_bounds'])
    )
