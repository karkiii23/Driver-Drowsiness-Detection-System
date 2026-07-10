import cv2
import os
import time
from datetime import datetime

from config import (
    BOX_COLOR,
    TEXT_COLOR,
    WARNING_COLOR,
    BACKGROUND_COLOR,
    FONT,
    FONT_SCALE,
    THICKNESS,
    LOW_DENSITY,
    MEDIUM_DENSITY,
)


# ==========================================================
# FPS Counter
# ==========================================================

class FPSCounter:

    def __init__(self):
        self.prev_time = time.time()

    def update(self):

        current_time = time.time()

        fps = 1 / (current_time - self.prev_time + 1e-6)

        self.prev_time = current_time

        return fps


# ==========================================================
# Timestamp
# ==========================================================

def get_timestamp():

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ==========================================================
# Filename Timestamp
# ==========================================================

def filename_timestamp():

    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ==========================================================
# Draw Rectangle
# ==========================================================

def draw_box(frame, box, color=BOX_COLOR, thickness=2):

    x1, y1, x2, y2 = map(int, box)

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        thickness
    )


# ==========================================================
# Draw Filled Label
# ==========================================================

def draw_label(frame, text, x, y, color=BOX_COLOR):

    (w, h), baseline = cv2.getTextSize(
        text,
        FONT,
        FONT_SCALE,
        THICKNESS
    )

    cv2.rectangle(
        frame,
        (x, y - h - 10),
        (x + w + 8, y),
        color,
        -1
    )

    cv2.putText(
        frame,
        text,
        (x + 4, y - 5),
        FONT,
        FONT_SCALE,
        TEXT_COLOR,
        THICKNESS
    )


# ==========================================================
# Draw Text
# ==========================================================

def draw_text(frame, text, position, color=TEXT_COLOR):

    cv2.putText(
        frame,
        text,
        position,
        FONT,
        FONT_SCALE,
        color,
        THICKNESS
    )


# ==========================================================
# Draw Text Background
# ==========================================================

def draw_text_background(
        frame,
        text,
        x,
        y,
        bg_color=BACKGROUND_COLOR,
        text_color=TEXT_COLOR
):

    (w, h), baseline = cv2.getTextSize(
        text,
        FONT,
        FONT_SCALE,
        THICKNESS
    )

    cv2.rectangle(
        frame,
        (x, y - h - 10),
        (x + w + 8, y + baseline),
        bg_color,
        -1
    )

    cv2.putText(
        frame,
        text,
        (x + 4, y - 5),
        FONT,
        FONT_SCALE,
        text_color,
        THICKNESS
    )


# ==========================================================
# Density Level
# ==========================================================

def traffic_density(vehicle_count):

    if vehicle_count <= LOW_DENSITY:
        return "LOW"

    elif vehicle_count <= MEDIUM_DENSITY:
        return "MEDIUM"

    return "HIGH"


# ==========================================================
# Density Color
# ==========================================================

def density_color(level):

    if level == "LOW":
        return (0, 255, 0)

    if level == "MEDIUM":
        return (0, 255, 255)

    return WARNING_COLOR


# ==========================================================
# Draw Density
# ==========================================================

def draw_density(frame, vehicle_count):

    level = traffic_density(vehicle_count)

    color = density_color(level)

    draw_text_background(
        frame,
        f"Density : {level}",
        20,
        80,
        color,
        (255, 255, 255)
    )


# ==========================================================
# Draw FPS
# ==========================================================

def draw_fps(frame, fps):

    draw_text_background(
        frame,
        f"FPS : {fps:.2f}",
        20,
        40
    )


# ==========================================================
# Draw Vehicle Count
# ==========================================================

def draw_vehicle_count(frame, count):

    draw_text_background(
        frame,
        f"Vehicles : {count}",
        20,
        120
    )


# ==========================================================
# Draw Alert
# ==========================================================

def draw_alert(frame, message):

    h, w = frame.shape[:2]

    cv2.rectangle(
        frame,
        (0, 0),
        (w, 60),
        WARNING_COLOR,
        -1
    )

    cv2.putText(
        frame,
        message,
        (20, 40),
        FONT,
        1.0,
        (255, 255, 255),
        2
    )


# ==========================================================
# Save Screenshot
# ==========================================================

def save_screenshot(frame, folder):

    os.makedirs(folder, exist_ok=True)

    filename = os.path.join(
        folder,
        filename_timestamp() + ".jpg"
    )

    cv2.imwrite(filename, frame)

    return filename


# ==========================================================
# Resize Frame
# ==========================================================

def resize_frame(frame, width=1280):

    h, w = frame.shape[:2]

    ratio = width / w

    height = int(h * ratio)

    return cv2.resize(
        frame,
        (width, height)
    )


# ==========================================================
# Center Point
# ==========================================================

def center(box):

    x1, y1, x2, y2 = map(int, box)

    cx = (x1 + x2) // 2

    cy = (y1 + y2) // 2

    return cx, cy


# ==========================================================
# Crop ROI
# ==========================================================

def crop_roi(frame, roi):

    if roi is None:
        return frame

    x1, y1, x2, y2 = roi

    return frame[y1:y2, x1:x2]


# ==========================================================
# Create Video Writer
# ==========================================================
# Tries codec/backend combinations that produce H.264-in-mp4 output
# natively (playable straight in browsers), and only falls back to
# OpenCV's always-available 'mp4v' (which browsers generally CANNOT
# play inline) if nothing better works on this machine. The caller
# (TrafficDensity) checks writer.isOpened() and can post-process with
# ffmpeg if we had to fall back.

def create_video_writer(path, width, height, fps=30):

    candidates = [
        # (fourcc, apiPreference) — avc1/H264 via Media Foundation is
        # natively available on most Windows installs with no extra
        # setup, and gives browser-playable output directly.
        ("avc1", cv2.CAP_MSMF),
        ("H264", cv2.CAP_MSMF),
        ("avc1", cv2.CAP_ANY),
        ("H264", cv2.CAP_ANY),
    ]

    for fourcc_str, api in candidates:

        try:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)

            writer = cv2.VideoWriter(
                path,
                api,
                fourcc,
                fps,
                (width, height)
            )

            if writer.isOpened():
                print(f"Video writer using codec '{fourcc_str}' (browser-playable).")
                return writer

            writer.release()

        except Exception:
            continue

    # Nothing browser-friendly worked — fall back to mp4v, which
    # OpenCV can always write, but which most browsers can't play
    # inline. The caller is expected to transcode this afterwards.
    print(
        "WARNING: could not open a browser-playable video codec "
        "(avc1/H264). Falling back to 'mp4v' — this file will need "
        "to be transcoded with ffmpeg to play in a browser."
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        path,
        fourcc,
        fps,
        (width, height)
    )

    return writer


# ==========================================================
# Draw ROI
# ==========================================================

def draw_roi(frame, roi):

    if roi is None:
        return

    x1, y1, x2, y2 = roi

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (255, 0, 255),
        2
    )


# ==========================================================
# Draw Line
# ==========================================================

def draw_line(frame, pt1, pt2, color=(255, 255, 0), thickness=2):

    cv2.line(
        frame,
        pt1,
        pt2,
        color,
        thickness
    )


# ==========================================================
# Draw Circle
# ==========================================================

def draw_circle(frame, point, radius=4, color=(0, 255, 255)):

    cv2.circle(
        frame,
        point,
        radius,
        color,
        -1
    )


# ==========================================================
# Put Multi-line Text
# ==========================================================

def draw_multiline(frame, lines, x=20, y=170):

    offset = 0

    for line in lines:

        draw_text_background(
            frame,
            line,
            x,
            y + offset
        )

        offset += 35


# ==========================================================
# Safe Image Display
# ==========================================================

def show(window, frame):

    cv2.imshow(window, frame)

    key = cv2.waitKey(1)

    return key


# ==========================================================
# Exit Key
# ==========================================================

def should_exit():

    key = cv2.waitKey(1)

    return key & 0xFF == ord("q")