"""
====================================================
SmartTrafficAI - Automated Module Tests
====================================================
Tests TrafficDensity and DrowsinessDetector end-to-end
using synthetic video + mocked ML backends (so this
runs without a real YOLO model, camera, or GPU).

Run with:
    python3 tests/test_pipeline.py

Mocks used (only because this test environment has no
network access to install them / no real model weights):
    - ultralytics.YOLO      -> returns controlled fake boxes
    - mediapipe.solutions   -> returns controlled fake landmarks

On a real machine with ultralytics + mediapipe installed
and a real YOLO .pt file, these mocks are not needed —
this script only mocks the ML backends, never the
SmartTrafficAI code itself.
"""

import os
import sys
import csv
import types
import shutil
import numpy as np
import cv2
from unittest.mock import MagicMock

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASSED = []
FAILED = []


def check(name, condition, detail=""):
    if condition:
        PASSED.append(name)
        print(f"  [PASS] {name}")
    else:
        FAILED.append(name)
        print(f"  [FAIL] {name}  {detail}")


def make_test_video(path, n_frames, size=(320, 240), fps=10):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for _ in range(n_frames):
        writer.write(np.zeros((size[1], size[0], 3), dtype="uint8"))
    writer.release()


# ============================================================
# Mock ultralytics.YOLO with a controllable vehicle-count sequence
# ============================================================

def install_fake_ultralytics(counts_sequence):

    fake_ultra = types.ModuleType("ultralytics")

    class FakeBox:
        def __init__(self, cls_id, conf, xyxy):
            self.cls = [cls_id]
            self.conf = [conf]
            self.xyxy = [np.array(xyxy)]

    class FakeBoxes:
        def __init__(self, boxes):
            self._boxes = boxes

        def __iter__(self):
            return iter(self._boxes)

    class FakeResult:
        def __init__(self, boxes):
            self.boxes = FakeBoxes(boxes)

    class_cycle = ["car", "motorcycle", "bus", "truck"]
    name_to_id = {"car": 0, "motorcycle": 1, "bus": 2, "truck": 3}
    state = {"i": 0}

    class FakeYOLO:
        def __init__(self, path):
            self.names = {0: "car", 1: "motorcycle", 2: "bus", 3: "truck"}

        def predict(self, frame, conf, verbose):
            n = counts_sequence[state["i"] % len(counts_sequence)]
            state["i"] += 1
            boxes = []
            for k in range(n):
                cls_name = class_cycle[k % 4]
                boxes.append(
                    FakeBox(name_to_id[cls_name], 0.85, (10 + k * 5, 10, 40 + k * 5, 40))
                )
            return [FakeResult(boxes)]

    fake_ultra.YOLO = FakeYOLO
    sys.modules["ultralytics"] = fake_ultra


# ============================================================
# Mock mediapipe.solutions.face_mesh with a controllable eye-state sequence
# ============================================================

def install_fake_mediapipe(state_sequence):

    import mediapipe as mp

    class FakeLM:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    class FakeFace:
        def __init__(self, state):
            lm = [FakeLM(0.0, 0.0) for _ in range(478)]
            gap = 0.04 if state == "open" else 0.002

            # LEFT_EYE = [33, 160, 158, 133, 153, 144]
            lm[33] = FakeLM(0.30, 0.50)
            lm[133] = FakeLM(0.40, 0.50)
            lm[160] = FakeLM(0.33, 0.50 - gap / 2)
            lm[144] = FakeLM(0.33, 0.50 + gap / 2)
            lm[158] = FakeLM(0.37, 0.50 - gap / 2)
            lm[153] = FakeLM(0.37, 0.50 + gap / 2)

            # RIGHT_EYE = [362, 385, 387, 263, 373, 380]
            lm[362] = FakeLM(0.60, 0.50)
            lm[263] = FakeLM(0.70, 0.50)
            lm[385] = FakeLM(0.63, 0.50 - gap / 2)
            lm[380] = FakeLM(0.63, 0.50 + gap / 2)
            lm[387] = FakeLM(0.67, 0.50 - gap / 2)
            lm[373] = FakeLM(0.67, 0.50 + gap / 2)

            self.landmark = lm

    class FakeFMResult:
        def __init__(self, state):
            self.multi_face_landmarks = [FakeFace(state)]

    idx = {"i": 0}

    class FakeFaceMesh:
        def __init__(self, **kwargs):
            pass

        def process(self, rgb):
            state = state_sequence[idx["i"] % len(state_sequence)]
            idx["i"] += 1
            return FakeFMResult(state)

    mp.solutions = MagicMock()
    mp.solutions.face_mesh.FaceMesh = FakeFaceMesh
    mp.solutions.drawing_utils = MagicMock()


# ============================================================
# Test 1: TrafficDensity
# ============================================================

def test_traffic_density():

    print("\n=== TrafficDensity ===")

    counts_sequence = [2, 2, 2, 8, 8, 8, 18, 18, 18, 2, 2, 2]
    install_fake_ultralytics(counts_sequence)

    import config
    from modules.traffic_density import TrafficDensity

    test_video = os.path.join(config.BASE_DIR, "input", "traffic_test.mp4")
    make_test_video(test_video, n_frames=len(counts_sequence))

    detector = TrafficDensity()
    out_path = detector.process_video_headless(test_video)

    check("processes all frames", detector.total_frames == len(counts_sequence),
          f"got {detector.total_frames}")

    check("total vehicle count matches", detector.total_vehicles == sum(counts_sequence),
          f"got {detector.total_vehicles}, expected {sum(counts_sequence)}")

    check("max_density latches at HIGH", detector.max_density == "HIGH",
          f"got {detector.max_density}")

    check("output video file was created", os.path.isfile(out_path) and os.path.getsize(out_path) > 0)

    check("PDF report was created", os.path.isfile(detector.report.filename) and
          os.path.getsize(detector.report.filename) > 0)

    with open(detector.logger.filepath) as f:
        rows = list(csv.reader(f))

    check("CSV has header + one row per frame",
          len(rows) == len(counts_sequence) + 1,
          f"got {len(rows)} rows")

    densities = [r[2] for r in rows[1:]]
    expected_densities = ["LOW"] * 3 + ["MEDIUM"] * 3 + ["HIGH"] * 3 + ["LOW"] * 3

    check("density sequence matches vehicle counts exactly",
          densities == expected_densities,
          f"got {densities}")

    dummy_frame = np.zeros((240, 320, 3), dtype="uint8")
    shot_path = detector.capture(dummy_frame)

    check("screenshot capture creates a file", os.path.isfile(shot_path))

    detector.reset()

    check("reset() clears all counters",
          detector.total_frames == 0 and detector.total_vehicles == 0
          and detector.max_density == "LOW" and detector.writer is None)


# ============================================================
# Test 2: DrowsinessDetector
# ============================================================

def test_drowsiness():

    print("\n=== DrowsinessDetector ===")

    # 5 open, 3 closed (short blink), 5 open, 25 closed (sustained -> alert), 5 open
    sequence = ["open"] * 5 + ["closed"] * 3 + ["open"] * 5 + ["closed"] * 25 + ["open"] * 5
    install_fake_mediapipe(sequence)

    import config
    from modules.drowsiness import DrowsinessDetector

    test_video = os.path.join(config.BASE_DIR, "input", "drowsiness_test.mp4")
    make_test_video(test_video, n_frames=len(sequence))

    detector = DrowsinessDetector()
    out_path = detector.process_video_headless(test_video)

    check("processes all frames", detector.total_frames == len(sequence),
          f"got {detector.total_frames}")

    check("at least one blink was registered", detector.blink_count >= 1,
          f"got {detector.blink_count}")

    check("at least one drowsy alert fired during sustained closure",
          detector.alert_count >= 1, f"got {detector.alert_count}")

    check("output video file was created", os.path.isfile(out_path) and os.path.getsize(out_path) > 0)

    check("PDF report was created", os.path.isfile(detector.report.filename) and
          os.path.getsize(detector.report.filename) > 0)

    with open(detector.logger.filepath) as f:
        rows = list(csv.reader(f))

    check("CSV has at least one row per frame processed this run",
          len(rows) - 1 >= len(sequence),
          f"got {len(rows) - 1} data rows for {len(sequence)} frames "
          f"(logger appends across runs, so this may be a multiple)")

    statuses = [r[4] for r in rows[1:]][-len(sequence):]

    check("DROWSY appears in the tail of the CSV matching this run",
          "DROWSY" in statuses)

    check("run ends AWAKE (sequence ends on open frames)",
          statuses[-1] == "AWAKE", f"got {statuses[-1]}")

    dummy_frame = np.zeros((240, 320, 3), dtype="uint8")
    shot_path = detector.capture(dummy_frame)

    check("screenshot capture creates a file", os.path.isfile(shot_path))

    detector.reset()

    check("reset() clears all counters",
          detector.total_frames == 0 and detector.blink_count == 0
          and detector.alert_count == 0 and detector.writer is None)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    test_traffic_density()
    test_drowsiness()

    print("\n" + "=" * 50)
    print(f"RESULTS: {len(PASSED)} passed, {len(FAILED)} failed")
    print("=" * 50)

    if FAILED:
        print("Failed checks:")
        for name in FAILED:
            print(f"  - {name}")
        sys.exit(1)
    else:
        print("All checks passed.")
        sys.exit(0)