"""Background camera readers that always hold the newest frame."""
from __future__ import annotations

import threading
import time

import cv2


class Camera:
    def __init__(self, index: int, width: int = 640, height: int = 480, mjpg: bool = False, name: str = ""):
        self.name = name or f"cam{index}"
        self.size = (width, height)
        self.cap = self._open(index, width, height, mjpg)
        self.frame, self.t = None, 0.0
        self.lock = threading.Lock()
        self.stop = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()
        if self.fresh(timeout=4.0) is None:
            # Some UVC boards stall after a format change on macOS: reopen with the camera's defaults.
            print(f"[{self.name}] no frames at {width}x{height}; reopening with default settings")
            self.cap.release()
            self.cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
            if self.fresh(timeout=4.0) is None:
                raise RuntimeError(f"camera {index} ({self.name}) gives no frames")

    @staticmethod
    def _open(index, width, height, mjpg):
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        if mjpg:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        return cap

    def _loop(self):
        while not self.stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            if (frame.shape[1], frame.shape[0]) != self.size:
                frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
            with self.lock:
                self.frame, self.t = frame, time.monotonic()

    def fresh(self, timeout: float = 1.0):
        """A frame captured after this call, so it shows the arm where it is now."""
        asked = time.monotonic()
        while time.monotonic() - asked < timeout:
            with self.lock:
                if self.frame is not None and self.t > asked:
                    return self.frame.copy()
            time.sleep(0.005)
        return None

    def save(self, path, timeout: float = 1.0) -> bool:
        frame = self.fresh(timeout)
        if frame is None:
            return False
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return True

    def close(self):
        self.stop.set()
        time.sleep(0.05)
        self.cap.release()
