import numpy as np

from roboreg.detector import OpenCVDetector


def test_opencv_detector() -> None:
    detector = OpenCVDetector(buffer_size=3)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    points, labels = detector.detect(img)
    print(points, labels)


if __name__ == "__main__":
    test_opencv_detector()
