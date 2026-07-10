import os

from config import (
    DROWSINESS_VIDEO,
    DROWSINESS_CSV,
    DROWSINESS_REPORT,
    DROWSINESS_OUTPUT_VIDEO,
)
from modules.drowsiness import DrowsinessDetector


def main():

    detector = DrowsinessDetector()

    print("=" * 60)
    print("SMART TRAFFIC AI - DRIVER DROWSINESS TEST")
    print("=" * 60)

    is_webcam = isinstance(DROWSINESS_VIDEO, int)

    print(f"Video : {'webcam index ' + str(DROWSINESS_VIDEO) if is_webcam else DROWSINESS_VIDEO}")

    if not is_webcam and not os.path.exists(DROWSINESS_VIDEO):
        print("\nERROR: Video not found!")
        print("Place your video here:")
        print(DROWSINESS_VIDEO)
        return

    try:

        detector.process_video(DROWSINESS_VIDEO)

        print("\n")
        print("=" * 60)
        print("TEST COMPLETED SUCCESSFULLY")
        print("=" * 60)

        print(f"Frames Processed : {detector.total_frames}")
        print(f"Blink Count     : {detector.blink_count}")
        print(f"Alerts          : {detector.alert_count}")
        print(f"Last EAR        : {detector.ear:.3f}")

        print("\nOutput Files")

        print(f"CSV  : {DROWSINESS_CSV}")
        print(f"PDF  : {DROWSINESS_REPORT}")
        print(f"VIDEO: {DROWSINESS_OUTPUT_VIDEO}")

    except Exception as e:

        print("\nERROR")
        print(e)


if __name__ == "__main__":
    main()