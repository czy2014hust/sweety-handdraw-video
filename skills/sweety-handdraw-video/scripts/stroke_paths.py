"""Extract and measure raster stroke paths without depending on a project checkout.

Smoothing and weak-bridge splits follow srt-whiteboard-animation (MIT,
Copyright (c) 2026 江哥是老登啊). Samples stay on the ink and at most 1.5px
apart, tighter than that project's 2.5px resampling.
"""
import math
import numpy as np

# Upstream resamples at 2.5px. This pipeline rejects a stroke step above 1.5px.
SAMPLE_SPACING = 1.25
MAX_STEP = 1.5


def thin_ink(binary_ink):
    """Zhang-Suen thinning with the Lü-Wang (1986) bound: a pixel needs 3+ neighbours to go, so a line that thins to a
    2-pixel diagonal staircase keeps its end instead of being eaten back to its junction (whole 45-degree marks vanished)."""
    skeleton = np.pad(binary_ink.astype(np.uint8), 1)
    while True:
        deleted_count = 0
        for subiteration in range(2):
            center = skeleton[1:-1, 1:-1]
            neighbors = [skeleton[:-2, 1:-1], skeleton[:-2, 2:], skeleton[1:-1, 2:],
                         skeleton[2:, 2:], skeleton[2:, 1:-1], skeleton[2:, :-2],
                         skeleton[1:-1, :-2], skeleton[:-2, :-2]]
            neighbor_count = sum(neighbors)
            transition_count = sum((neighbors[index] == 0) & (neighbors[(index + 1) % 8] == 1)
                                   for index in range(8))
            removable = (center == 1) & (neighbor_count >= 3) & (neighbor_count <= 6) & (transition_count == 1)
            if subiteration == 0:
                removable &= (neighbors[0] * neighbors[2] * neighbors[4] == 0)
                removable &= (neighbors[2] * neighbors[4] * neighbors[6] == 0)
            else:
                removable &= (neighbors[0] * neighbors[2] * neighbors[6] == 0)
                removable &= (neighbors[0] * neighbors[4] * neighbors[6] == 0)
            deleted_count += int(removable.sum())
            center[removable] = 0
        if deleted_count == 0:
            return skeleton[1:-1, 1:-1].astype(bool)


def trace_strokes(skeleton):
    pixel_set = set(zip(*np.where(skeleton)))
    adjacency = {}
    for y, x in pixel_set:
        adjacent = []
        for delta_y, delta_x in [(-1, 0), (0, 1), (1, 0), (0, -1), (-1, -1), (-1, 1), (1, 1), (1, -1)]:
            neighbor = (y + delta_y, x + delta_x)
            if neighbor not in pixel_set:
                continue
            if delta_x and delta_y and ((y, x + delta_x) in pixel_set or (y + delta_y, x) in pixel_set):
                continue
            adjacent.append(neighbor)
        adjacency[(y, x)] = adjacent
    visited_edges = set()
    strokes = []

    def follow_chain(start, neighbor):
        chain = [start]
        previous, current = start, neighbor
        visited_edges.add(tuple(sorted((previous, current))))
        while True:
            chain.append(current)
            candidates = [point for point in adjacency[current]
                          if tuple(sorted((point, current))) not in visited_edges]
            if len(adjacency[current]) != 2 or not candidates:
                break
            previous, current = current, candidates[0]
            visited_edges.add(tuple(sorted((previous, current))))
        return [[float(x), float(y)] for y, x in chain]

    for start in sorted(pixel_set):
        if len(adjacency[start]) == 2:
            continue
        for neighbor in adjacency[start]:
            if tuple(sorted((start, neighbor))) not in visited_edges:
                strokes.append(follow_chain(start, neighbor))
    for start in sorted(pixel_set):
        for neighbor in adjacency[start]:
            if tuple(sorted((start, neighbor))) not in visited_edges:
                strokes.append(follow_chain(start, neighbor))
    return [stroke for stroke in strokes if len(stroke) > 4]


def calculate_lengths(points):
    coordinates = np.array(points, dtype=float)
    distances = np.linalg.norm(np.diff(coordinates, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(distances)))


def _dedupe(points):
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return points
    keep = np.ones(len(points), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-6
    return points[keep]


def chaikin(points, iterations=1):
    """Corner-cut smoothing. Endpoints stay put."""
    points = _dedupe(points)
    for _ in range(max(0, iterations)):
        if len(points) < 3:
            break
        start, end = points[:-1], points[1:]
        inserted = np.empty((len(start) * 2, 2))
        inserted[0::2] = start * 0.75 + end * 0.25
        inserted[1::2] = start * 0.25 + end * 0.75
        points = np.vstack((points[:1], inserted, points[-1:]))
    return points


def resample_stroke(points, spacing=SAMPLE_SPACING):
    """Even arc-length samples. Spacing never exceeds MAX_STEP."""
    points = _dedupe(points)
    if len(points) < 2:
        return points
    lengths = calculate_lengths(points)
    total = float(lengths[-1])
    if total <= spacing:
        return points[[0, -1]]
    count = max(2, math.ceil(total / min(spacing, MAX_STEP)))
    targets = np.linspace(0, total, count + 1)
    return np.stack((np.interp(targets, lengths, points[:, 0]), np.interp(targets, lengths, points[:, 1])), axis=1)


def smooth_stroke(points, spacing=SAMPLE_SPACING):
    """One Chaikin pass, then resample. Falls back when a step would exceed 1.5px."""
    smoothed = resample_stroke(chaikin(points), spacing)
    if len(smoothed) < 2:
        return np.asarray(points, dtype=float)
    if np.linalg.norm(np.diff(smoothed, axis=0), axis=1).max() > MAX_STEP:
        return np.asarray(points, dtype=float)
    return smoothed


def split_bridge_points(ys, xs, min_side=20):
    """Split a component whose two masses meet through a thin vertical gap.

    Column counts use the median ink column, so a one-pixel ground line between
    two outlines is a valley without splitting a solid silhouette.
    """
    ys = np.asarray(ys, dtype=int)
    xs = np.asarray(xs, dtype=int)
    if len(ys) < min_side * 2:
        return [(ys, xs)]
    min_x, max_x = int(xs.min()), int(xs.max())
    height = int(ys.max() - ys.min() + 1)
    width = max_x - min_x + 1
    if width < 16 or height < 10:
        return [(ys, xs)]
    counts = np.bincount(xs, minlength=max_x + 1)
    band = counts[min_x:max_x + 1]
    typical = float(np.median(band[band > 0])) if np.any(band > 0) else 0
    valley_limit = max(2, int(math.ceil(typical * 0.30)))
    edge_guard = 4
    valleys = []
    start = None
    for col in range(min_x, max_x + 2):
        low = col <= max_x and counts[col] <= valley_limit
        if low and start is None:
            start = col
        elif not low and start is not None:
            end = col - 1
            if end - start + 1 >= 2 and start > min_x + edge_guard and end < max_x - edge_guard:
                valleys.append((start, end))
            start = None
    if not valleys:
        return [(ys, xs)]
    start, end = max(valleys, key=lambda band: (band[1] - band[0], -band[0]))
    cut = (start + end) // 2
    left = xs <= cut
    if int(left.sum()) < min_side or int((~left).sum()) < min_side:
        return [(ys, xs)]
    return split_bridge_points(ys[left], xs[left], min_side) + split_bridge_points(ys[~left], xs[~left], min_side)
