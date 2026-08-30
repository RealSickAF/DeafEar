#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Derives, per gesture label, which hand(s) it actually needs and where (relative to the face
anchor) that hand stays during the recorded motion. Written to a CSV consumed by
engine/gesture_zones.py, so live recognition can reject a predicted label whose current hand
presence/position doesn't match how it was actually recorded (e.g. a one-handed sign getting
misread as a two-handed one, or a chest-height sign firing near the head).

Re-run this any time point_history.csv changes (after collecting more/different samples).
"""
import csv

DATASET_PATH = 'model/point_history_classifier/point_history.csv'
LABELS_PATH = 'model/point_history_classifier/point_history_classifier_label.csv'
ZONES_PATH = 'model/point_history_classifier/label_zones.csv'
FRAME_COUNT = 45
PRESENCE_REQUIRED_RATIO = 0.4  # a hand counts as "needed" for this label if present this often


def load_labels():
    with open(LABELS_PATH, encoding='utf-8-sig') as f:
        return [row[0].strip() for row in csv.reader(f) if row and row[0].strip()]


def _present_points(flat_hand_history):
    """Yields (x, y) for frames where the hand was actually tracked (not zero-padded)."""
    for i in range(0, len(flat_hand_history), 3):
        x, y = flat_hand_history[i], flat_hand_history[i + 1]
        if not (x == 0.0 and y == 0.0):
            yield x, y


def _percentile(values, pct):
    values = sorted(values)
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, round(pct / 100 * (len(values) - 1))))
    return values[index]


def main():
    labels = load_labels()
    per_class = {i: {'right_present': 0, 'left_present': 0, 'total': 0,
                      'right_x': [], 'right_y': [], 'left_x': [], 'left_y': []} for i in range(len(labels))}

    with open(DATASET_PATH) as f:
        for row in csv.reader(f):
            class_id = int(row[0])
            if class_id not in per_class:
                continue
            values = [float(v) for v in row[1:]]
            right_hand, left_hand = values[:FRAME_COUNT * 3], values[FRAME_COUNT * 3:FRAME_COUNT * 6]
            stats = per_class[class_id]
            stats['total'] += 1

            right_points = list(_present_points(right_hand))
            left_points = list(_present_points(left_hand))
            if len(right_points) / FRAME_COUNT >= 0.5:
                stats['right_present'] += 1
                stats['right_x'] += [p[0] for p in right_points]
                stats['right_y'] += [p[1] for p in right_points]
            if len(left_points) / FRAME_COUNT >= 0.5:
                stats['left_present'] += 1
                stats['left_x'] += [p[0] for p in left_points]
                stats['left_y'] += [p[1] for p in left_points]

    rows = []
    for class_id, label in enumerate(labels):
        stats = per_class[class_id]
        total = max(1, stats['total'])
        row = {'label': label}
        for side in ('right', 'left'):
            ratio = stats[f'{side}_present'] / total
            required = ratio >= PRESENCE_REQUIRED_RATIO
            row[f'{side}_required'] = int(required)
            if required and stats[f'{side}_x']:
                row[f'{side}_x_min'] = round(_percentile(stats[f'{side}_x'], 2), 3)
                row[f'{side}_x_max'] = round(_percentile(stats[f'{side}_x'], 98), 3)
                row[f'{side}_y_min'] = round(_percentile(stats[f'{side}_y'], 2), 3)
                row[f'{side}_y_max'] = round(_percentile(stats[f'{side}_y'], 98), 3)
            else:
                row[f'{side}_x_min'] = row[f'{side}_x_max'] = ''
                row[f'{side}_y_min'] = row[f'{side}_y_max'] = ''
        rows.append(row)

    fieldnames = ['label', 'right_required', 'right_x_min', 'right_x_max', 'right_y_min', 'right_y_max',
                  'left_required', 'left_x_min', 'left_x_max', 'left_y_min', 'left_y_max']
    with open(ZONES_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'Wrote {len(rows)} label zones to {ZONES_PATH}')


if __name__ == '__main__':
    main()
