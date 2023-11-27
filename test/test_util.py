import cv2

from roboreg.util import extend_mask, mask_boundary, overlay_mask, shrink_mask


def test_extend_mask() -> None:
    idx = 1
    mask = cv2.imread(f"test/data/high_res/mask_{idx}.png", cv2.IMREAD_GRAYSCALE)
    extended_mask = extend_mask(mask)
    cv2.imshow("mask", mask)
    cv2.imshow("extended_mask", extended_mask)
    cv2.waitKey()


def test_mask_boundary() -> None:
    idx = 1
    img = cv2.imread(f"test/data/high_res/img_{idx}.png")
    mask = cv2.imread(f"test/data/high_res/mask_{idx}.png", cv2.IMREAD_GRAYSCALE)
    boundary_mask = mask_boundary(mask)
    overlay = overlay_mask(img, boundary_mask, mode="b", alpha=1.0, scale=1.0)
    cv2.imshow("mask", mask)
    cv2.imshow("boundary_mask", boundary_mask)
    cv2.imshow("overlay", overlay)
    cv2.waitKey()


def test_shrink_mask() -> None:
    idx = 1
    mask = cv2.imread(f"test/data/high_res/mask_{idx}.png", cv2.IMREAD_GRAYSCALE)
    shrinked_mask = shrink_mask(mask)
    cv2.imshow("mask", mask)
    cv2.imshow("shrinked_mask", shrinked_mask)
    cv2.waitKey()


if __name__ == "__main__":
    # test_extend_mask()
    test_mask_boundary()
    # test_shrink_mask()
