"""
Test for other settings included in the upscaling nodes.
"""

import logging
import pathlib
import pytest
import torch
from contextlib import nullcontext

from tensor_utils import img_tensor_mae, blur
from io_utils import save_image, load_image, image_name_format
from configs import DirectoryConfig
from fixtures_images import EXT

# Image file names
CATEGORY = pathlib.Path(pathlib.Path(__file__).stem.removeprefix("test_"))


@pytest.mark.parametrize("batch_size", [1, 2])
def test_minimal_tile_sizes(
    base_image,
    loaded_checkpoint,
    node_classes,
    seed,
    batch_size,
    test_dirs: DirectoryConfig,
):
    """Test upscaling with minimal tile sizes."""
    image, positive, negative = base_image
    image = image[0:1]  # 1 image for simplicity
    model, clip, vae = loaded_checkpoint

    with torch.inference_mode():
        with pytest.raises(AssertionError) if batch_size > 1 else nullcontext():
            usdu = node_classes["UltimateSDUpscale"]
            (upscaled,) = usdu().upscale(
                image=image,
                model=model,
                positive=positive,
                negative=negative,
                vae=vae,
                upscale_by=1.5,
                seed=seed,
                steps=5,
                cfg=8,
                sampler_name="euler",
                scheduler="normal",
                denoise=0.6,
                upscale_model=None,
                mode_type="Chess",
                tile_width=512,
                tile_height=512,
                mask_blur=8,
                tile_padding=8,
                seam_fix_mode="None",
                seam_fix_denoise=1.0,
                seam_fix_width=16,
                seam_fix_mask_blur=8,
                seam_fix_padding=4,
                force_uniform_tiles=False,  # This should trigger the assertion for batch_size > 1
                tiled_decode=False,
                batch_size=batch_size,
            )

        if batch_size > 1:
            return  # Test passed if assertion was raised

    # Save and reload sample image
    sample_dir = test_dirs.sample_images
    filename = CATEGORY / image_name_format("non_uniform_tiles", EXT, batch_size)
    save_image(upscaled[0], sample_dir / filename)
    upscaled = load_image(sample_dir / filename)

    # Compare with reference
    test_image_dir = test_dirs.test_images
    test_image = load_image(test_image_dir / filename)
    logger = logging.getLogger(__name__)
    diff = img_tensor_mae(blur(upscaled), blur(test_image))
    logger.info(f"{filename} MAE: {diff}")
    assert diff < 0.02, f"{filename} does not match reference (MAE {diff})"
