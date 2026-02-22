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


@pytest.mark.parametrize("batch_size", [1, 2])
class TestZImageFunControlNet:
    """Integration tests for the upscaling workflow with Z-Image's Fun Controlnet."""

    @pytest.fixture(scope="function")
    def upscaled(
        self,
        base_image,
        node_classes,
        seed,
        batch_size,
        test_dirs: DirectoryConfig,
    ):
        # TODO: Fixtures for z-image if more tests are needed for this model
        with torch.inference_mode():
            image, _, _ = base_image
            # (image,) = execute(
            #     node_classes["ImageScale"],
            #     image=image,
            #     upscale_method="lanczos",
            #     width=512,
            #     height=512,
            #     crop="center",
            # )
            (model,) = execute(
                node_classes["UNETLoader"],
                "z-image-turbo_fp8_scaled_e4m3fn_KJ.safetensors",
                weight_dtype="fp8_e4m3fn",
            )
            (clip,) = execute(
                node_classes["CLIPLoader"], "qwen_3_4b.safetensors", type="lumina2"
            )
            (vae,) = execute(node_classes["VAELoader"], "ae.safetensors")
            prompt = "beautiful scenery nature glass bottle landscape, , purple galaxy bottle,"
            (pos,) = execute(node_classes["CLIPTextEncode"], clip, prompt)
            (neg,) = execute(node_classes["ConditioningZeroOut"], pos)
            depth_hint = load_image(test_dirs.test_images / CATEGORY / "depth2.png")
            canny_hint = load_image(test_dirs.test_images / CATEGORY / "canny2.png")
            canny_hint = canny_hint.repeat(1, 1, 1, 3)  # Convert from grayscale to RGB
            (hint,) = execute(
                node_classes["ImageBlend"], depth_hint, canny_hint, 1.0, "overlay"
            )
            hint = hint[..., :3]  # Blend and drop alpha
            # Both the image and depth hint are 512x512
            (hint,) = execute(
                node_classes["ImageScaleBy"],
                image=hint,
                upscale_method="lanczos",
                scale_by=2.0,
            )
            save_image(hint, test_dirs.sample_images / CATEGORY / "blended_hint.png")
            (image,) = execute(
                node_classes["ImageScaleBy"],
                image=image,
                upscale_method="lanczos",
                scale_by=2.0,
            )
            (model_patch,) = execute(
                node_classes["ModelPatchLoader"],
                "Z-Image-Turbo-Fun-Controlnet-Tile-2.1-2601-8steps.safetensors",
            )
            (model,) = execute(
                node_classes["ZImageFunControlnet"],
                model,
                model_patch,
                vae,
                hint,
                strength=1.0,
            )

            usdu = node_classes["UltimateSDUpscaleNoUpscale"]
            (upscaled,) = usdu().upscale(
                upscaled_image=image,
                model=model,
                positive=pos,
                negative=neg,
                vae=vae,
                seed=seed,
                steps=5,
                cfg=1,
                sampler_name="euler",
                scheduler="normal",
                denoise=0.8,
                mode_type="Chess",
                tile_width=512,
                tile_height=512,
                mask_blur=16,
                tile_padding=128,
                seam_fix_mode="None",
                seam_fix_denoise=1.0,
                seam_fix_width=64,
                seam_fix_mask_blur=8,
                seam_fix_padding=16,
                force_uniform_tiles=True,
                tiled_decode=False,
                batch_size=batch_size,
            )

        return upscaled

    def _verify_match(self, upscaled, i, batch_size, test_dirs, threshold):
        # Verify reference image match
        logger = logging.getLogger(TestZImageFunControlNet.__name__)
        test_img_dir = test_dirs.test_images
        batch_str = f"_batch{batch_size}" if batch_size > 1 else ""
        filename = CATEGORY / (f"controlnet_zimage_fun_{i + 1}{batch_str}" + EXT)
        sample_dir = test_dirs.sample_images
        save_image(upscaled, sample_dir / filename)
        upscaled = load_image(sample_dir / filename)
        test_img = load_image(test_img_dir / filename)

        # Reduce high-frequency noise differences with gaussian blur
        diff = img_tensor_mae(blur(upscaled), blur(test_img))
        logger.info(f"Diff: {diff}")
        assert diff < threshold, (
            f"{filename} does not match its test image. Diff: {diff}"
        )

    def test_image1(self, upscaled, batch_size, test_dirs):
        self._verify_match(upscaled[0], 0, batch_size, test_dirs, threshold=0.01)

    def test_image2(self, upscaled, batch_size, test_dirs):
        self._verify_match(upscaled[1], 1, batch_size, test_dirs, threshold=0.01)
