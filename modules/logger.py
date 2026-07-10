import os
import csv
from datetime import datetime


# ==========================================================
# Traffic Logger
# ==========================================================

class TrafficLogger:
    """
    Logs per-frame traffic density data to a CSV file.
    Row format:
        Timestamp, Vehicle Count, Density, Cars, Motorcycles, Buses, Trucks
    """

    HEADERS = [
        "Timestamp",
        "Vehicle Count",
        "Density",
        "Cars",
        "Motorcycles",
        "Buses",
        "Trucks"
    ]

    def __init__(self, filepath):

        self.filepath = filepath

        self._init_file()

    def _init_file(self):

        folder = os.path.dirname(self.filepath)

        if folder:
            os.makedirs(folder, exist_ok=True)

        file_exists = os.path.isfile(self.filepath)

        if not file_exists:

            with open(self.filepath, mode="w", newline="") as f:

                writer = csv.writer(f)

                writer.writerow(self.HEADERS)

    def save(
        self,
        vehicle_count,
        density,
        cars,
        motorcycles,
        buses,
        trucks
    ):

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(self.filepath, mode="a", newline="") as f:

            writer = csv.writer(f)

            writer.writerow([
                timestamp,
                vehicle_count,
                density,
                cars,
                motorcycles,
                buses,
                trucks
            ])


# ==========================================================
# Drowsiness Logger
# ==========================================================

class DrowsinessLogger:
    """
    Logs per-frame drowsiness data to a CSV file.
    Row format:
        Timestamp, EAR, Blink Count, Closed Frames, Status
    """

    HEADERS = [
        "Timestamp",
        "EAR",
        "Blink Count",
        "Closed Frames",
        "Status"
    ]

    def __init__(self, filepath):

        self.filepath = filepath

        self._init_file()

    def _init_file(self):

        folder = os.path.dirname(self.filepath)

        if folder:
            os.makedirs(folder, exist_ok=True)

        file_exists = os.path.isfile(self.filepath)

        if not file_exists:

            with open(self.filepath, mode="w", newline="") as f:

                writer = csv.writer(f)

                writer.writerow(self.HEADERS)

    def save(
        self,
        ear,
        blink_count,
        closed_frames,
        status
    ):

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(self.filepath, mode="a", newline="") as f:

            writer = csv.writer(f)

            writer.writerow([
                timestamp,
                round(float(ear), 3),
                blink_count,
                closed_frames,
                status
            ])