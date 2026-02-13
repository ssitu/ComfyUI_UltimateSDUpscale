"""
Tests a common workflow for UltimateSDUpscale.
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

# Image file names
CATEGORY = pathlib.Path(pathlib.Path(__file__).stem.removeprefix("test_"))


@pytest.mark.parametrize("batch_size", [1, 2])
class TestMainWorkflow:
    """Integration tests for the main upscaling workflow."""

    def test_upscale(
        self,
        base_image,
        loaded_checkpoint,
        upscale_model,
        node_classes,
        seed,
        batch_size,
        test_dirs: DirectoryConfig,
    ):
        """Generate upscaled images using standard workflow."""
        image, positive, negative = base_image
        model, clip, vae = loaded_checkpoint

        with torch.inference_mode():
            usdu = node_classes["UltimateSDUpscale"]
            (upscaled,) = usdu().upscale(
                image=image,
                model=model,
                positive=positive,
                negative=negative,
                vae=vae,
                upscale_by=2.00000004,  # Test small float difference doesn't add extra tiles
                seed=seed,
                steps=5,
                cfg=8,
                sampler_name="euler",
                scheduler="normal",
                denoise=0.7,
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
                batch_size=batch_size,
            )
        # Save images
        im1_filename = image_name_format("upscaled_image1", EXT, batch_size)
        im2_filename = image_name_format("upscaled_image2", EXT, batch_size)
        sample_dir = test_dirs.sample_images
        upscaled_img1_path = sample_dir / CATEGORY / im1_filename
        upscaled_img2_path = sample_dir / CATEGORY / im2_filename
        save_image(upscaled[0], upscaled_img1_path)
        save_image(upscaled[1], upscaled_img2_path)
        # Load to account for compression
        upscaled = torch.cat(
            [load_image(upscaled_img1_path), load_image(upscaled_img2_path)]
        )
        # Verify results
        logger = logging.getLogger("test_upscale")
        test_image_dir = test_dirs.test_images
        im1_upscaled = upscaled[0]
        im2_upscaled = upscaled[1]

        test_im1 = load_image(test_image_dir / CATEGORY / im1_filename)
        test_im2 = load_image(test_image_dir / CATEGORY / im2_filename)

        diff1 = img_tensor_mae(blur(im1_upscaled), blur(test_im1))
        diff2 = img_tensor_mae(blur(im2_upscaled), blur(test_im2))
        # This tolerance is enough to handle both cpu and gpu as the device, as well as jpg compression differences.
        logger.info(f"Diff1: {diff1}, Diff2: {diff2}")
        assert diff1 < 0.01, "Upscaled Image 1 doesn't match its test image."
        assert diff2 < 0.01, "Upscaled Image 2 doesn't match its test image."

    def test_upscale_no_upscale(
        self,
        base_image,
        loaded_checkpoint,
        node_classes,
        seed,
        batch_size,
        test_dirs: DirectoryConfig,
    ):
        """Generate upscaled images using standard workflow using the no upscale node."""
        image, positive, negative = base_image
        model, clip, vae = loaded_checkpoint
        (image,) = execute(
            node_classes["ImageScaleBy"],
            image=image,
            upscale_method="lanczos",
            scale_by=2.0,
        )

        with torch.inference_mode():
            usdu = node_classes["UltimateSDUpscaleNoUpscale"]
            (upscaled,) = usdu().upscale(
                upscaled_image=image,
                model=model,
                positive=positive,
                negative=negative,
                vae=vae,
                seed=seed,
                steps=5,
                cfg=8,
                sampler_name="euler",
                scheduler="normal",
                denoise=0.7,
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
        # Save images
        im1_filename = image_name_format("no_upscale_image_1", EXT, batch_size)
        im2_filename = image_name_format("no_upscale_image_2", EXT, batch_size)
        sample_dir = test_dirs.sample_images
        upscaled_img1_path = sample_dir / CATEGORY / im1_filename
        upscaled_img2_path = sample_dir / CATEGORY / im2_filename
        save_image(upscaled[0], upscaled_img1_path)
        save_image(upscaled[1], upscaled_img2_path)
        # Load to account for compression
        upscaled = torch.cat(
            [load_image(upscaled_img1_path), load_image(upscaled_img2_path)]
        )
        # Verify results
        logger = logging.getLogger("test_upscale_no_upscale")
        test_image_dir = test_dirs.test_images
        im1_upscaled = upscaled[0]
        im2_upscaled = upscaled[1]

        test_im1 = load_image(test_image_dir / CATEGORY / im1_filename)
        test_im2 = load_image(test_image_dir / CATEGORY / im2_filename)

        diff1 = img_tensor_mae(blur(im1_upscaled), blur(test_im1))
        diff2 = img_tensor_mae(blur(im2_upscaled), blur(test_im2))
        # This tolerance is enough to handle both cpu and gpu as the device, as well as jpg compression differences.
        logger.info(f"Diff1: {diff1}, Diff2: {diff2}")
        assert diff1 < 0.01, f"{im1_filename} doesn't match its test image."
        assert diff2 < 0.01, f"{im2_filename} doesn't match its test image."

    def test_upscale_with_custom_sampler(
        self,
        base_image,
        loaded_checkpoint,
        upscale_model,
        node_classes,
        seed,
        batch_size,
        test_dirs: DirectoryConfig,
    ):
        """Generate upscaled images using standard workflow using the custom sampler node."""
        image, positive, negative = base_image
        model, clip, vae = loaded_checkpoint

        with torch.inference_mode():
            # Setup custom scheduler and sampler
            custom_scheduler = node_classes["KarrasScheduler"]
            (sigmas,) = execute(custom_scheduler, 10, 14.614642, 0.0291675, 7.0)
            (_, sigmas) = execute(node_classes["SplitSigmasDenoise"], sigmas, 0.7)

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
                upscale_by=2.0,
                seed=seed,
                steps=10,
                cfg=8,
                sampler_name="euler",
                scheduler="normal",
                denoise=1.0,
                upscale_model=upscale_model,
                mode_type="Chess",
                tile_width=512,
                tile_height=512,
                mask_blur=8,
                tile_padding=32,
                seam_fix_mode="None",
                seam_fix_denoise=0.5,
                seam_fix_width=64,
                seam_fix_mask_blur=8,
                seam_fix_padding=16,
                force_uniform_tiles=True,
                tiled_decode=False,
                batch_size=batch_size,
                custom_sampler=sampler,
                custom_sigmas=sigmas,
            )
        # Save images
        im1_filename = image_name_format("custom_sampler1", EXT, batch_size)
        im2_filename = image_name_format("custom_sampler2", EXT, batch_size)
        sample_dir = test_dirs.sample_images
        upscaled_img1_path = sample_dir / CATEGORY / im1_filename
        upscaled_img2_path = sample_dir / CATEGORY / im2_filename
        save_image(upscaled[0], upscaled_img1_path)
        save_image(upscaled[1], upscaled_img2_path)
        # Load to account for compression
        upscaled = torch.cat(
            [load_image(upscaled_img1_path), load_image(upscaled_img2_path)]
        )
        # Verify results
        logger = logging.getLogger("test_upscale_with_custom_sampler")
        test_image_dir = test_dirs.test_images
        im1_upscaled = upscaled[0]
        im2_upscaled = upscaled[1]
        test_im1 = load_image(test_image_dir / CATEGORY / im1_filename)
        test_im2 = load_image(test_image_dir / CATEGORY / im2_filename)

        diff1 = img_tensor_mae(blur(im1_upscaled), blur(test_im1))
        diff2 = img_tensor_mae(blur(im2_upscaled), blur(test_im2))

        # This tolerance is enough to handle both cpu and gpu as the device, as well as jpg compression differences.
        logger.info(f"Diff1: {diff1}, Diff2: {diff2}")
        assert diff1 < 0.011, f"{im1_filename} doesn't match its test image."
        assert diff2 < 0.011, f"{im2_filename} doesn't match its test image."
