"""
Fixtures for base images.
"""

import pathlib
import pytest
import torch

from setup_utils import execute
from io_utils import save_image, load_image

# Image file names
EXT = ".jpg"
CATEGORY = pathlib.Path("main_workflow")
BASE_IMAGE_1_NAME = "main1_sd15" + EXT
BASE_IMAGE_2_NAME = "main2_sd15" + EXT

# Prepend category path
BASE_IMAGE_1 = CATEGORY / BASE_IMAGE_1_NAME
BASE_IMAGE_2 = CATEGORY / BASE_IMAGE_2_NAME


@pytest.fixture(scope="session")
def base_image(loaded_checkpoint, seed, test_dirs, node_classes):
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

    # Save base images
    sample_dir = test_dirs.sample_images
    base_img1_path = sample_dir / BASE_IMAGE_1
    base_img2_path = sample_dir / BASE_IMAGE_2
    save_image(image[0:1], base_img1_path)
    save_image(image[1:2], base_img2_path)

    # Load images back as tensors to account for compression
    image = torch.cat([load_image(base_img1_path), load_image(base_img2_path)])
    return image, positive, negative
