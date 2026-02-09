"""
Tests for base image generation.
"""

import logging
from configs import DirectoryConfig
from tensor_utils import img_tensor_mae, blur
from io_utils import load_image
from fixtures_images import BASE_IMAGE_1, BASE_IMAGE_2


def test_base_image_matches_reference(base_image, test_dirs: DirectoryConfig):
    """
    Verify generated base images match reference images.
    This is just to check if the checkpoint and generation pipeline are as expected for the tests dependent on their behavior.
    """
    logger = logging.getLogger("test_base_image_matches_reference")
    image, _, _ = base_image
    test_image_dir = test_dirs.test_images
    im1 = image[0:1]
    im2 = image[1:2]

    test_im1 = load_image(test_image_dir / BASE_IMAGE_1)
    test_im2 = load_image(test_image_dir / BASE_IMAGE_2)

    # Reduce high-frequency noise differences with gaussian blur. Using perceptual metrics are probably overkill.
    diff1 = img_tensor_mae(blur(im1), blur(test_im1))
    diff2 = img_tensor_mae(blur(im2), blur(test_im2))
    logger.info(f"Base Image Diff1: {diff1}, Diff2: {diff2}")
    assert diff1 < 0.01, "Image 1 does not match its test image."
    assert diff2 < 0.01, "Image 2 does not match its test image."
