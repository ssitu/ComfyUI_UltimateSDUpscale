"""
Test using controlnet in the upscaling workflow.
"""

import logging
import pathlib
import pytest
import torch

from setup_utils import execute
from tensor_utils import img_tensor_mae, blur
from io_utils import save_image, load_image, image_name_format
from configs import DirectoryConfig
from fixtures_images import EXT

CATEGORY = pathlib.Path(pathlib.Path(__file__).stem.removeprefix("test_"))
TEST_CONTROLNET_TILE_MODEL = "control_v11f1e_sd15_tile.pth"


@pytest.mark.parametrize("batch_size", [1, 2])
class TestControlNet:
    """Integration tests for the upscaling workflow with ControlNet."""

    def test_controlnet_tile(
        self,
        base_image,
        loaded_checkpoint,
        node_classes,
        seed,
        batch_size,
        test_dirs: DirectoryConfig,
    ):
        """Generate upscaled images using ControlNet."""
        image, positive, negative = base_image
        model, clip, vae = loaded_checkpoint
        image = image[0:1]

        (controlnet_tile_model,) = execute(
            node_classes["ControlNetLoader"], TEST_CONTROLNET_TILE_MODEL
        )
        (positive,) = execute(
            node_classes["ControlNetApply"], positive, controlnet_tile_model, image, 1.0
        )

        with torch.inference_mode():
            # Run upscale with ControlNet
            usdu = node_classes["UltimateSDUpscale"]
            (upscaled,) = usdu().upscale(
                image=image,
                model=model,
                positive=positive,
                negative=negative,
                vae=vae,
                upscale_by=2.0,
                seed=seed,
                steps=5,
                cfg=8,
                sampler_name="euler",
                scheduler="normal",
                denoise=1.0,
                upscale_model=None,
                mode_type="Chess",
                tile_width=512,
                tile_height=512,
                mask_blur=8,
                tile_padding=32,
                seam_fix_mode="None",
                seam_fix_denoise=1.0,
                seam_fix_width=64,
                seam_fix_mask_blur=8,
                seam_fix_padding=16,
                force_uniform_tiles=True,
                tiled_decode=False,
                batch_size=batch_size,
            )
        # Save and reload sample image
        sample_dir = test_dirs.sample_images
        filename = CATEGORY / image_name_format("controlnet_tile", EXT, batch_size)
        save_image(upscaled[0], sample_dir / filename)
        upscaled = load_image(sample_dir / filename)

        # Verify against reference image
        logger = logging.getLogger("test_controlnet_tile")
        test_img_dir = test_dirs.test_images
        test_img = load_image(test_img_dir / filename)

        # Reduce high-frequency noise differences with gaussian blur
        diff = img_tensor_mae(blur(upscaled), blur(test_img))
        logger.info(f"ControlNet Upscaled Image Diff: {diff}")
        assert diff < 0.01, "ControlNet upscaled image does not match its test image."
