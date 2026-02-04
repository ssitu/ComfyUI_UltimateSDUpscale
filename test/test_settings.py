"""
Test for other settings included in the upscaling nodes.
"""

import logging
import pathlib
import pytest
import torch

from tensor_utils import img_tensor_mae, blur
from io_utils import save_image, load_image
from configs import DirectoryConfig
from fixtures_images import EXT

# Image file names
CATEGORY = pathlib.Path(pathlib.Path(__file__).stem.removeprefix("test_"))


def test_minimal_tile_sizes(
    base_image, loaded_checkpoint, node_classes, seed, test_dirs: DirectoryConfig
):
    """Test upscaling with minimal tile sizes."""
    filename = "non_uniform_tiles"
    image, positive, negative = base_image
    model, clip, vae = loaded_checkpoint

    with torch.inference_mode():
        usdu = node_classes["UltimateSDUpscale"]
        (upscaled,) = usdu().upscale(
            image=image[0:1],
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
            denoise=0.15,
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
            force_uniform_tiles=False,
            tiled_decode=False,
        )

    # Save and reload sample image
    sample_dir = test_dirs.sample_images
    filename_path = CATEGORY / (filename + EXT)
    save_image(upscaled[0], sample_dir / filename_path)
    upscaled = load_image(sample_dir / filename_path)

    # Compare with reference
    test_image_dir = test_dirs.test_images
    test_image = load_image(test_image_dir / filename_path)
    diff = img_tensor_mae(blur(upscaled), blur(test_image))
    logger = logging.getLogger(__name__)
    logger.info(f"{filename} MAE: {diff}")
    assert diff < 0.05, f"{filename} output doesn't match reference"
