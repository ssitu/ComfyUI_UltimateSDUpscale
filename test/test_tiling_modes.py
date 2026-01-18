"""
Tests for different upscaling modes and seam fix modes.
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


def image_name_format(prefix: str, mode: str) -> str:
    """Helper for the image name format for the tests below."""
    return f"{prefix}_{mode.lower().replace(' ', '_')}{EXT}"


class TestTilingModes:
    def _test_upscale_variant(
        self,
        base_image,
        loaded_checkpoint,
        node_classes,
        seed,
        test_dirs: DirectoryConfig,
        mode_type,
        seam_fix_mode,
        seam_fix_denoise,
        filename_prefix,
    ):
        """Helper method to test upscale variants with different parameters."""
        logger = logging.getLogger(f"test_{filename_prefix}")
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
                upscale_by=2.0,
                seed=seed,
                steps=3,
                cfg=8,
                sampler_name="euler",
                scheduler="normal",
                denoise=0.2,
                upscale_model=None,
                mode_type=mode_type,
                tile_width=512,
                tile_height=512,
                mask_blur=8,
                tile_padding=32,
                seam_fix_mode=seam_fix_mode,
                seam_fix_denoise=seam_fix_denoise,
                seam_fix_width=64,
                seam_fix_mask_blur=8,
                seam_fix_padding=16,
                force_uniform_tiles=True,
                tiled_decode=False,
            )

        # Save and reload sample image
        sample_dir = test_dirs.sample_images
        filename = CATEGORY / filename_prefix
        save_image(upscaled[0], sample_dir / filename)
        upscaled = load_image(sample_dir / filename)

        # Compare with reference
        test_image_dir = test_dirs.test_images
        test_image = load_image(test_image_dir / filename)
        diff = img_tensor_mae(blur(upscaled), blur(test_image))
        logger.info(f"{filename_prefix} MAE: {diff}")
        assert diff < 0.05, f"{filename_prefix} output doesn't match reference"

    # "Chess" is tested in the main workflow test
    @pytest.mark.parametrize("mode_type", ["Linear", "None"])
    def test_mode_types(
        self,
        base_image,
        loaded_checkpoint,
        node_classes,
        seed,
        mode_type,
        test_dirs: DirectoryConfig,
    ):
        """Test different tiling mode types."""
        filename = image_name_format("mode", mode_type)
        self._test_upscale_variant(
            base_image,
            loaded_checkpoint,
            node_classes,
            seed,
            test_dirs,
            mode_type=mode_type,
            seam_fix_mode="None",
            seam_fix_denoise=1.0,
            filename_prefix=filename,
        )

    @pytest.mark.parametrize(
        "seam_fix_mode", ["None", "Band Pass", "Half Tile", "Half Tile + Intersections"]
    )
    def test_seam_fix_modes(
        self,
        base_image,
        loaded_checkpoint,
        node_classes,
        seed,
        seam_fix_mode,
        test_dirs: DirectoryConfig,
    ):
        """Test different seam fix modes."""
        filename = image_name_format("seamfix", seam_fix_mode)
        self._test_upscale_variant(
            base_image,
            loaded_checkpoint,
            node_classes,
            seed,
            test_dirs,
            mode_type="None",
            seam_fix_mode=seam_fix_mode,
            seam_fix_denoise=0.5,
            filename_prefix=filename,
        )
