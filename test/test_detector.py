import numpy as np

from roboreg.detector import OpenCVDetector


def test_opencv_detector() -> None:
    detector = OpenCVDetector(n_positive_samples=3, n_negative_samples=3)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    samples, labels = detector.detect(img)
    print("Samples: ", samples)
    print("Labels: ", labels)


def test_sample_parser_mixin() -> None:
    detector = OpenCVDetector()
    detector.positive_samples = [[1, 2], [3, 4]]
    detector.negative_samples = [[5, 6]]
    csv_file = "test/data/samples.csv"
    detector.write(csv_file, samples=detector.samples, labels=detector.labels)

    detector.clear()
    samples, labels = detector.read(csv_file)
    print("Samples: ", samples)
    print("Labels: ", labels)


if __name__ == "__main__":
    # test_opencv_detector()
    test_sample_parser_mixin()
