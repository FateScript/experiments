#!/usr/bin/env python3

import contextlib
import time
from imgcat import imgcat

import cv2


@contextlib.contextmanager
def hide_cursor():
    "Hide the cursor in the context."
    print("\033[?25l", end="", flush=True)
    try:
        yield
    except Exception:
        pass
    finally:
        print("\033[?25h", end="", flush=True)


def save_cursor_position():
    "Save the cursor position"
    print("\033[s", end="", flush=True)


def restore_cursor_position():
    "Reset the cursor to the saved position"
    print("\033[u", end="", flush=True)


def clear_after_position():
    "Clear everything after the cursor position."
    print("\033[J", end="", flush=True)


def video_as_stream(
    video_path: str,
    height: int = None,
    num_frames: int = None,
):
    if num_frames is None:
        num_frames = float("inf")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    frame_count = 0
    while cap.isOpened() and frame_count < num_frames:
        ret, frame = cap.read()
        if height:
            frame_scale = height / frame.shape[0]
            frame = cv2.resize(frame, None, fx=frame_scale, fy=frame_scale)

        if not ret:  # "reached the end of the video."
            break
        yield frame
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


def cat_video(video_path: str, height: int = None, num_frames: int = None):
    flow = video_as_stream(video_path, height, num_frames)
    with hide_cursor():
        for frame in flow:
            save_cursor_position()
            imgcat(frame)
            time.sleep(0.02)
            restore_cursor_position()


if __name__ == "__main__":
    import fire
    fire.Fire(cat_video)
