import re
from typing import List
from glob import glob
import cv2
import numpy as np


def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval


def natural_keys(text):
    return [atof(c) for c in re.split(r"[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)", text)]


def load_images(prefix: str, postfix: str = "npy") -> List[str]:
    paths = glob(f"{prefix}/*.{postfix}")
    paths.sort(key=natural_keys)
    return paths


def images_to_video(img_paths: List[str], video_path: str) -> None:
    img = np.load(img_paths[0])
    height, width, layers = img.shape

    # mp4 video codec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(video_path, fourcc, 15, (width, height))

    for img_path in img_paths:
        video.write(cv2.cvtColor(np.load(img_path), cv2.COLOR_RGBA2RGB))

    video.release()


def main() -> None:
    img_prefix = "/home/martin/Dev/zed_ws/records/camera"
    img_paths = load_images(img_prefix)
    video_path = "/home/martin/Dev/zed_ws/records/camera.mp4"
    images_to_video(img_paths, video_path)


if __name__ == "__main__":
    main()
