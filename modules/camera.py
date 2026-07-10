import cv2

from config import (
    FRAME_WIDTH,
    FRAME_HEIGHT
)


class Camera:
    """
    Camera / Video Handler
    Supports:
        - Webcam
        - Video file
        - Resize
        - FPS
    """

    def __init__(self, source=0):

        self.source = source

        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Unable to open source: {source}"
            )

        # Webcam settings
        if isinstance(source, int):

            self.cap.set(
                cv2.CAP_PROP_FRAME_WIDTH,
                FRAME_WIDTH
            )

            self.cap.set(
                cv2.CAP_PROP_FRAME_HEIGHT,
                FRAME_HEIGHT
            )

    # ---------------------------------------------
    # Read Frame
    # ---------------------------------------------
    def read(self):

        success, frame = self.cap.read()

        if not success:
            return None

        return frame

    # ---------------------------------------------
    # Has Next Frame
    # ---------------------------------------------
    def is_open(self):

        return self.cap.isOpened()

    # ---------------------------------------------
    # Frame Width
    # ---------------------------------------------
    def width(self):

        return int(
            self.cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

    # ---------------------------------------------
    # Frame Height
    # ---------------------------------------------
    def height(self):

        return int(
            self.cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

    # ---------------------------------------------
    # FPS
    # ---------------------------------------------
    def fps(self):

        fps = self.cap.get(
            cv2.CAP_PROP_FPS
        )

        if fps == 0:
            fps = 30

        return fps

    # ---------------------------------------------
    # Total Frames
    # ---------------------------------------------
    def total_frames(self):

        return int(
            self.cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

    # ---------------------------------------------
    # Current Frame Number
    # ---------------------------------------------
    def current_frame(self):

        return int(
            self.cap.get(
                cv2.CAP_PROP_POS_FRAMES
            )
        )

    # ---------------------------------------------
    # Video Duration
    # ---------------------------------------------
    def duration(self):

        fps = self.fps()

        frames = self.total_frames()

        if fps == 0:
            return 0

        return frames / fps

    # ---------------------------------------------
    # Restart Video
    # ---------------------------------------------
    def restart(self):

        self.cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            0
        )

    # ---------------------------------------------
    # Resize Frame
    # ---------------------------------------------
    @staticmethod
    def resize(frame, width=1280):

        h, w = frame.shape[:2]

        ratio = width / w

        height = int(h * ratio)

        return cv2.resize(
            frame,
            (width, height)
        )

    # ---------------------------------------------
    # Release
    # ---------------------------------------------
    def release(self):

        if self.cap:

            self.cap.release()

    # ---------------------------------------------
    # Close Everything
    # ---------------------------------------------
    @staticmethod
    def destroy():

        cv2.destroyAllWindows()