"""
Tests a common workflow for UltimateSDUpscale.
"""

import pytest
import torch
from PIL import Image

import usdu_utils
from test_utils import execute
from configs import DirectoryConfig

BASE_IMAGE_1 = "main1_sd15.jpg"
BASE_IMAGE_2 = "main2_sd15.jpg"
UPSCALED_IMAGE_1 = "main1_sd15_upscaled.jpg"
UPSCALED_IMAGE_2 = "main2_sd15_upscaled.jpg"


class TestMainWorkflow:
    """Integration tests for the main upscaling workflow."""

    @pytest.fixture(scope="class")
    def base_image(self, loaded_checkpoint, seed, node_classes):
        """Generate a base image for upscaling tests."""
        EmptyLatentImage = node_classes["EmptyLatentImage"]
        CLIPTextEncode = node_classes["CLIPTextEncode"]
        KSampler = node_classes["KSampler"]
        VAEDecode = node_classes["VAEDecode"]

        model, clip, vae = loaded_checkpoint

        with torch.inference_mode():
            (empty_latent,) = execute(
                EmptyLatentImage, width=512, height=512, batch_size=2
            )

            (positive,) = execute(
                CLIPTextEncode,
                text="beautiful scenery nature glass bottle landscape, , purple galaxy bottle,",
                clip=clip,
            )

            (negative,) = execute(CLIPTextEncode, text="text, watermark", clip=clip)

            (samples,) = execute(
                KSampler,
                model=model,
                positive=positive,
                negative=negative,
                latent_image=empty_latent,
                seed=seed,
                steps=10,
                cfg=8,
                sampler_name="dpmpp_2m",
                scheduler="karras",
                denoise=1.0,
            )

            (image,) = execute(VAEDecode, samples=samples, vae=vae)

        return image, positive, negative

    def test_base_image_matches_reference(self, base_image, test_dirs: DirectoryConfig):
        """
        Verify generated base images match reference images.
        This is just to check if the checkpoint and generation pipeline are as expected for the tests dependent on their behavior.
        """
        image, _, _ = base_image
        test_image_dir = test_dirs.test_images

        im1 = image[0]
        im2 = image[1]

        test_im1 = usdu_utils.pil_to_tensor(Image.open(test_image_dir / BASE_IMAGE_1))
        test_im2 = usdu_utils.pil_to_tensor(Image.open(test_image_dir / BASE_IMAGE_2))

        diff1 = (im1 - test_im1).abs().mean().item()
        diff2 = (im2 - test_im2).abs().mean().item()

        assert diff1 < 0.015, f"Image 1 does not match test image. Diff: {diff1}"
        assert diff2 < 0.015, f"Image 2 does not match test image. Diff: {diff2}"

    @pytest.fixture(scope="class")
    def upscaled_image(
        self,
        base_image,
        loaded_checkpoint,
        upscale_model,
        node_classes,
        seed,
    ):
        """Generate upscaled images using custom sampler."""
        image, positive, negative = base_image
        model, clip, vae = loaded_checkpoint

        with torch.inference_mode():
            # Setup custom scheduler and sampler
            custom_scheduler = node_classes["KarrasScheduler"]
            (sigmas,) = execute(custom_scheduler, 20, 14.614642, 0.0291675, 7.0)
            (_, sigmas) = execute(node_classes["SplitSigmasDenoise"], sigmas, 0.2)

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

        return upscaled

    def test_upscale_with_custom_sampler(self, upscaled_image, test_dirs: DirectoryConfig):
        """Test upscaling with custom sampler and sigmas."""
        # Verify results
        test_image_dir = test_dirs.test_images
        im1_upscaled = upscaled_image[0]
        im2_upscaled = upscaled_image[1]

        test_im1_upscaled = usdu_utils.pil_to_tensor(
            Image.open(test_image_dir / UPSCALED_IMAGE_1)
        )
        test_im2_upscaled = usdu_utils.pil_to_tensor(
            Image.open(test_image_dir / UPSCALED_IMAGE_2)
        )

        diff1 = (im1_upscaled - test_im1_upscaled).abs().mean().item()
        diff2 = (im2_upscaled - test_im2_upscaled).abs().mean().item()

        # This tolerance is enough to handle both cpu and gpu as the device, as well as jpg compression differences.
        assert diff1 < 0.015, f"Upscaled Image 1 doesn't match. Diff: {diff1}"
        assert diff2 < 0.015, f"Upscaled Image 2 doesn't match. Diff: {diff2}"

    def test_save_sample_images(self, base_image, upscaled_image, test_dirs: DirectoryConfig):
        """Save sample images for visual inspection (optional utility test)."""
        image, _, _ = base_image
        sample_dir = test_dirs.sample_images
        sample_dir.mkdir(exist_ok=True)

        # Save base images
        usdu_utils.tensor_to_pil(image).save(sample_dir / BASE_IMAGE_1)
        usdu_utils.tensor_to_pil(image, 1).save(sample_dir / BASE_IMAGE_2)

        # Save upscaled images
        usdu_utils.tensor_to_pil(upscaled_image).save(sample_dir / UPSCALED_IMAGE_1)
        usdu_utils.tensor_to_pil(upscaled_image, 1).save(sample_dir / UPSCALED_IMAGE_2)


# Allow running directly for debugging
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
