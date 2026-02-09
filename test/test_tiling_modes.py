"""
Tests for different upscaling modes and seam fix modes.
"""

import logging
import pathlib
import pytest
import torch

from tensor_utils import img_tensor_mae, blur
from io_utils import save_image, load_image, image_name_format
from configs import DirectoryConfig
from fixtures_images import EXT

# Image file names
CATEGORY = pathlib.Path(pathlib.Path(__file__).stem.removeprefix("test_"))


@pytest.mark.parametrize("batch_size", [1, 2])
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
        batch_size,
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
                denoise=0.9,
                upscale_model=None,
                mode_type=mode_type,
                tile_width=512,
                tile_height=512,
                mask_blur=8,
                tile_padding=32,
                seam_fix_mode=seam_fix_mode,
                seam_fix_denoise=seam_fix_denoise,
                seam_fix_width=256,
                seam_fix_mask_blur=8,
                seam_fix_padding=16,
                force_uniform_tiles=True,
                tiled_decode=False,
                batch_size=batch_size,
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
        logger.info(f"{filename} MAE: {diff}")
        assert diff < 0.01, f"{filename} output doesn't match reference"

    # "Chess" is tested in the main workflow test
    @pytest.mark.parametrize("mode_type", ["Linear", "None"])
    def test_mode_types(
        self,
        base_image,
        loaded_checkpoint,
        node_classes,
        seed,
        mode_type,
        batch_size,
        test_dirs: DirectoryConfig,
    ):
        """Test different tiling mode types."""
        filename = image_name_format("mode_" + mode_type.lower(), EXT, batch_size)
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
            batch_size=batch_size,
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
        batch_size,
        test_dirs: DirectoryConfig,
    ):
        """Test different seam fix modes."""
        filename = image_name_format(
            "seamfix_" + seam_fix_mode.lower().replace(" ", "_"), EXT, batch_size
        )
        self._test_upscale_variant(
            base_image,
            loaded_checkpoint,
            node_classes,
            seed,
            test_dirs,
            mode_type="None",
            seam_fix_mode=seam_fix_mode,
            seam_fix_denoise=0.6,
            filename_prefix=filename,
            batch_size=batch_size,
        )
