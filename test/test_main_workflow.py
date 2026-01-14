"""
Tests a common workflow for UltimateSDUpscale.
"""

import logging
import pathlib
import pytest
import torch

from setup_utils import execute
from tensor_utils import img_tensor_mae, blur
from io_utils import save_image, load_image
from configs import DirectoryConfig
from fixtures_images import base_image

# Image file names
EXT = ".jpg"
CATEGORY = pathlib.Path("main_workflow")
UPSCALED_IMAGE_1_NAME = "main1_sd15_upscaled" + EXT
UPSCALED_IMAGE_2_NAME = "main2_sd15_upscaled" + EXT

# Prepend category path
UPSCALED_IMAGE_1 = CATEGORY / UPSCALED_IMAGE_1_NAME
UPSCALED_IMAGE_2 = CATEGORY / UPSCALED_IMAGE_2_NAME


class TestMainWorkflow:
    """Integration tests for the main upscaling workflow."""

    @pytest.fixture(scope="class")
    def upscaled_image(
        self,
        base_image,
        loaded_checkpoint,
        upscale_model,
        node_classes,
        seed,
        test_dirs,
    ):
        """Generate upscaled images using custom sampler."""
        image, positive, negative = base_image
        model, clip, vae = loaded_checkpoint

        with torch.inference_mode():
            # Setup custom scheduler and sampler
            custom_scheduler = node_classes["KarrasScheduler"]
            (sigmas,) = execute(custom_scheduler, 20, 14.614642, 0.0291675, 7.0)
            (_, sigmas) = execute(node_classes["SplitSigmasDenoise"], sigmas, 0.15)

            custom_sampler = node_classes["KSamplerSelect"]
            (sampler,) = execute(custom_sampler, "dpmpp_2m")

            # Run upscale
            usdu = node_classes["UltimateSDUpscaleCustomSample"]
            (upscaled,) = usdu().upscale(
                image=image,
                model=model,
                positive=positive,
                negative=negative,
                vae=vae,
                upscale_by=2.00000004,  # Test small float difference doesn't add extra tiles
                seed=seed,
                steps=10,
                cfg=8,
                sampler_name="euler",
                scheduler="normal",
                denoise=0.2,
                upscale_model=upscale_model,
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
                custom_sampler=sampler,
                custom_sigmas=sigmas,
            )
        # Save images
        sample_dir = test_dirs.sample_images
        upscaled_img1_path = sample_dir / UPSCALED_IMAGE_1
        upscaled_img2_path = sample_dir / UPSCALED_IMAGE_2
        save_image(upscaled[0], upscaled_img1_path)
        save_image(upscaled[1], upscaled_img2_path)
        # Load
        upscaled = torch.cat(
            [load_image(upscaled_img1_path), load_image(upscaled_img2_path)]
        )
        return upscaled

    def test_upscale_with_custom_sampler(
        self, upscaled_image, test_dirs: DirectoryConfig
    ):
        """Test upscaling with custom sampler and sigmas."""
        logger = logging.getLogger("test_upscale_with_custom_sampler")
        # Verify results
        test_image_dir = test_dirs.test_images
        im1_upscaled = upscaled_image[0]
        im2_upscaled = upscaled_image[1]

        test_im1_upscaled = load_image(test_image_dir / UPSCALED_IMAGE_1)
        test_im2_upscaled = load_image(test_image_dir / UPSCALED_IMAGE_2)

        diff1 = img_tensor_mae(blur(im1_upscaled), blur(test_im1_upscaled))
        diff2 = img_tensor_mae(blur(im2_upscaled), blur(test_im2_upscaled))

        # This tolerance is enough to handle both cpu and gpu as the device, as well as jpg compression differences.
        logger.info(f"Diff1: {diff1}, Diff2: {diff2}")
        assert diff1 < 0.05, "Upscaled Image 1 doesn't match its test image."
        assert diff2 < 0.05, "Upscaled Image 2 doesn't match its test image."

    def test_save_sample_images(self, upscaled_image, test_dirs: DirectoryConfig):
        """Save sample images for visual inspection (optional utility test)."""
        sample_dir = test_dirs.sample_images

        # Save upscaled images
        save_image(upscaled_image[0], sample_dir / UPSCALED_IMAGE_1)
        save_image(upscaled_image[1], sample_dir / UPSCALED_IMAGE_2)

    def _test_upscale_variant(
        self, base_image, loaded_checkpoint, upscale_model, node_classes, seed, test_dirs,
        mode_type, seam_fix_mode, seam_fix_denoise, filename_prefix
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

    @pytest.mark.parametrize("mode_type", ["Linear", "None"])
    def test_mode_types(
        self, base_image, loaded_checkpoint, upscale_model, node_classes, seed, mode_type, test_dirs
    ):
        """Test different tiling mode types."""
        filename = f"mode_{mode_type.lower().replace(' ', '_')}{EXT}"
        self._test_upscale_variant(
            base_image, loaded_checkpoint, upscale_model, node_classes, seed, test_dirs,
            mode_type=mode_type,
            seam_fix_mode="None",
            seam_fix_denoise=1.0,
            filename_prefix=filename
        )

    @pytest.mark.parametrize("seam_fix_mode", ["None", "Band Pass", "Half Tile", "Half Tile + Intersections"])
    def test_seam_fix_modes(
        self, base_image, loaded_checkpoint, upscale_model, node_classes, seed, seam_fix_mode, test_dirs
    ):
        """Test different seam fix modes."""
        filename = f"seamfix_{seam_fix_mode.lower().replace(' ', '_')}{EXT}"
        self._test_upscale_variant(
            base_image, loaded_checkpoint, upscale_model, node_classes, seed, test_dirs,
            mode_type="None",
            seam_fix_mode=seam_fix_mode,
            seam_fix_denoise=0.5,
            filename_prefix=filename
        )
