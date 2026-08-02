import cv2
import numpy as np
import pytest

from roboreg.util import (
    mask_dilate_with_kernel,
    mask_distance_transform,
    mask_erode_with_kernel,
    mask_exponential_decay,
    mask_extract_boundary,
    mask_extract_extended_boundary,
)


def _square_mask(
    *,
    size: int = 7,
    start: int = 2,
    end: int = 5,
    dtype: np.dtype = np.uint8,
    foreground_value: int | bool = 255,
) -> np.ndarray:
    mask = np.zeros((size, size), dtype=dtype)
    mask[start:end, start:end] = foreground_value
    return mask


@pytest.mark.parametrize(
    ("dtype", "foreground_value"),
    [
        (np.bool_, True),
        (np.uint8, 1),
        (np.uint8, 255),
    ],
)
def test_dilate_with_kernel(
    dtype: np.dtype,
    foreground_value: int | bool,
) -> None:
    mask = _square_mask(
        dtype=dtype,
        foreground_value=foreground_value,
    )
    kernel = np.ones((3, 3), dtype=np.uint8)

    result = mask_dilate_with_kernel(mask, kernel)

    expected = np.zeros((7, 7), dtype=np.uint8)
    expected[1:6, 1:6] = 255

    np.testing.assert_array_equal(result, expected)
    assert result.dtype == np.uint8


@pytest.mark.parametrize(
    ("dtype", "foreground_value"),
    [
        (np.bool_, True),
        (np.uint8, 1),
        (np.uint8, 255),
    ],
)
def test_erode_with_kernel(
    dtype: np.dtype,
    foreground_value: int | bool,
) -> None:
    mask = _square_mask(
        start=1,
        end=6,
        dtype=dtype,
        foreground_value=foreground_value,
    )
    kernel = np.ones((3, 3), dtype=np.uint8)

    result = mask_erode_with_kernel(mask, kernel)

    expected = np.zeros((7, 7), dtype=np.uint8)
    expected[2:5, 2:5] = 255

    np.testing.assert_array_equal(result, expected)
    assert result.dtype == np.uint8


def test_distance_transform_single_foreground_pixel() -> None:
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[2, 2] = 255

    result = mask_distance_transform(mask)

    expected = np.zeros((5, 5), dtype=np.float32)
    expected[2, 2] = 1.0

    np.testing.assert_allclose(result, expected, atol=1e-6)
    assert result.dtype == np.float32


def test_distance_transform_accepts_bool() -> None:
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True

    bool_result = mask_distance_transform(mask)
    uint8_result = mask_distance_transform(mask.astype(np.uint8))

    np.testing.assert_allclose(bool_result, uint8_result)


def test_exponential_decay() -> None:
    mask = np.zeros((7, 7), dtype=np.uint8)
    mask[3, 3] = 255

    sigma = 2.0
    result = mask_exponential_decay(mask, sigma=sigma)

    assert result.shape == mask.shape
    assert result.dtype == np.float32
    assert np.all(np.isfinite(result))
    assert np.all((result >= 0.0) & (result <= 1.0))

    # Inside the original mask, inverse-mask distance is zero.
    assert result[3, 3] == pytest.approx(1.0)

    # The response should decay with distance from the mask.
    assert result[3, 2] > result[3, 1]
    assert result[3, 1] > result[3, 0]


def test_exponential_decay_rejects_invalid_sigma() -> None:
    mask = np.zeros((5, 5), dtype=np.uint8)

    with pytest.raises(ValueError, match="sigma must be positive"):
        mask_exponential_decay(mask, sigma=0.0)


def test_extract_boundary() -> None:
    mask = _square_mask(
        size=7,
        start=1,
        end=6,
    )
    kernel = np.ones((3, 3), dtype=np.uint8)

    result = mask_extract_boundary(
        mask,
        erosion_kernel=kernel,
    )

    expected = np.zeros((7, 7), dtype=np.uint8)
    expected[1:6, 1:6] = 255
    expected[2:5, 2:5] = 0

    np.testing.assert_array_equal(result, expected)


def test_extract_extended_boundary() -> None:
    mask = _square_mask(
        size=9,
        start=3,
        end=6,
    )
    kernel = np.ones((3, 3), dtype=np.uint8)

    result = mask_extract_extended_boundary(
        mask,
        dilation_kernel=kernel,
        erosion_kernel=kernel,
    )

    dilated = cv2.dilate(mask, kernel)
    eroded = cv2.erode(mask, kernel)
    expected = cv2.subtract(dilated, eroded)

    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "function",
    [
        mask_dilate_with_kernel,
        mask_distance_transform,
        mask_erode_with_kernel,
        mask_extract_boundary,
        mask_extract_extended_boundary,
    ],
)
def test_mask_functions_reject_non_2d_masks(function) -> None:
    mask = np.zeros((4, 4, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="Expected a 2D mask"):
        function(mask)
