"""
Speed Estimator — Centroid tracking + pixel-displacement to km/h.

Tracks vehicles across frames using nearest-centroid matching and estimates
speed from pixel displacement per frame, converted using a calibration factor
(pixels_per_meter) and the video's FPS.
"""

import numpy as np
from collections import defaultdict, deque


class CentroidTracker:
    """
    Assigns persistent integer IDs to detections by matching centroids
    between consecutive frames using greedy nearest-neighbour matching.
    """

    def __init__(self, max_disappeared: int = 25, max_distance: int = 90):
        self.next_id = 0
        self.objects: dict[int, tuple] = {}        # obj_id -> (cx, cy)
        self.disappeared: defaultdict[int, int] = defaultdict(int)
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    # ------------------------------------------------------------------
    def _register(self, centroid: tuple) -> int:
        obj_id = self.next_id
        self.objects[obj_id] = centroid
        self.next_id += 1
        return obj_id

    def _deregister(self, obj_id: int) -> None:
        del self.objects[obj_id]
        if obj_id in self.disappeared:
            del self.disappeared[obj_id]

    # ------------------------------------------------------------------
    def update(self, detections: list[tuple]) -> dict[int, tuple]:
        """
        Parameters
        ----------
        detections : list of (cx, cy) centroids for this frame

        Returns
        -------
        dict mapping obj_id -> (cx, cy) for all currently tracked objects
        """
        # Case 1: nothing detected — age-out existing objects
        if len(detections) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self._deregister(obj_id)
            return dict(self.objects)

        # Case 2: no existing objects — register all detections
        if len(self.objects) == 0:
            for c in detections:
                self._register(c)
            return dict(self.objects)

        # Case 3: match existing objects to new detections
        obj_ids = list(self.objects.keys())
        obj_cents = list(self.objects.values())

        # Pairwise L2-distance matrix  (n_objects × n_detections)
        D = np.linalg.norm(
            np.array(obj_cents)[:, None, :] - np.array(detections)[None, :, :],
            axis=2
        )

        # Greedy matching: sort by minimum distance in each row
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.max_distance:
                continue
            obj_id = obj_ids[row]
            self.objects[obj_id] = detections[col]
            self.disappeared[obj_id] = 0
            used_rows.add(row)
            used_cols.add(col)

        # Age-out unmatched existing objects
        for row in set(range(D.shape[0])) - used_rows:
            obj_id = obj_ids[row]
            self.disappeared[obj_id] += 1
            if self.disappeared[obj_id] > self.max_disappeared:
                self._deregister(obj_id)

        # Register unmatched new detections
        for col in set(range(D.shape[1])) - used_cols:
            self._register(detections[col])

        return dict(self.objects)

    def reset(self):
        self.next_id = 0
        self.objects.clear()
        self.disappeared.clear()


# ──────────────────────────────────────────────────────────────────────────────


class SpeedEstimator:
    """
    Wraps CentroidTracker to estimate per-vehicle speed in km/h.

    Speed formula (per frame):
        speed (m/s) = pixel_displacement / pixels_per_meter * fps
        speed (km/h) = speed (m/s) * 3.6

    A rolling average over ``smooth_window`` frames reduces jitter.
    """

    def __init__(
        self,
        pixels_per_meter: float = 20.0,
        fps: float = 25.0,
        smooth_window: int = 12,
    ):
        self.pixels_per_meter = pixels_per_meter
        self.fps = fps
        self.smooth_window = smooth_window

        self.tracker = CentroidTracker()
        self._prev_positions: dict[int, tuple] = {}
        self._speed_history: defaultdict[int, deque] = defaultdict(
            lambda: deque(maxlen=smooth_window)
        )
        self.current_speeds: dict[int, float] = {}

    # ------------------------------------------------------------------
    def update_calibration(self, pixels_per_meter: float, fps: float) -> None:
        self.pixels_per_meter = pixels_per_meter
        self.fps = fps

    # ------------------------------------------------------------------
    def update(self, detections: list[tuple]) -> dict[int, float]:
        """
        Parameters
        ----------
        detections : list of (cx, cy) centroids detected in the current frame

        Returns
        -------
        dict  obj_id -> smoothed speed in km/h
        """
        tracked = self.tracker.update(detections)

        for obj_id, centroid in tracked.items():
            if obj_id in self._prev_positions:
                prev = self._prev_positions[obj_id]
                pixel_dist = float(np.linalg.norm(
                    np.array(centroid, dtype=float) - np.array(prev, dtype=float)
                ))
                speed_ms = (pixel_dist / self.pixels_per_meter) * self.fps
                speed_kmh = speed_ms * 3.6
                self._speed_history[obj_id].append(speed_kmh)
                self.current_speeds[obj_id] = float(np.mean(self._speed_history[obj_id]))
            else:
                self.current_speeds[obj_id] = 0.0

            self._prev_positions[obj_id] = centroid

        # Purge speeds for deregistered vehicles
        for obj_id in list(self.current_speeds.keys()):
            if obj_id not in tracked:
                self.current_speeds.pop(obj_id, None)
                self._prev_positions.pop(obj_id, None)

        return dict(self.current_speeds)

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.tracker.reset()
        self._prev_positions.clear()
        self._speed_history.clear()
        self.current_speeds.clear()
