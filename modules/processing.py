# Patched classes to adapt from A111 webui for ComfyUI
from nodes import common_ksampler, VAEEncodeTiled, VAEDecodeTiled, ConditioningSetMask
from utils import pil_to_tensor, tensor_to_pil
import modules.shared as shared
import numpy as np
import torch
from PIL import Image


class StableDiffusionProcessing:

    def __init__(self, init_img, model, positive, negative, vae, seed, steps, cfg, sampler_name, scheduler, denoise):
        # Variables used by the upscaler script
        self.init_images = [init_img]
        self.image_mask = None

        # ComfyUI Sampler inputs
        self.model = model
        self.positive = positive
        self.negative = negative
        self.vae = vae
        self.seed = seed
        self.steps = steps
        self.cfg = cfg
        self.sampler_name = sampler_name
        self.scheduler = scheduler
        self.denoise = denoise

        # Other required A1111 variables for the upscaler script that is currently unused in this script
        self.extra_generation_params = {}


class Processed:

    def __init__(self, p: StableDiffusionProcessing, images: list, seed: int, info: str):
        self.images = images
        self.seed = seed
        self.info = info

    def infotext(self, p: StableDiffusionProcessing, index):
        return None


def fix_seed(p: StableDiffusionProcessing):
    pass

def process_images(p: StableDiffusionProcessing) -> Processed:
    # Where the main image generation happens in A1111

    # Convert the PIL images to a torch tensor
    init_images = p.init_images
    image_tensor = pil_to_tensor(init_images[0])

    # Encode the image
    vae_encoder = VAEEncodeTiled()
    (encoded,) = vae_encoder.encode(p.vae, image_tensor)
    print(encoded["samples"].shape)

    # Convert the black and white mask to a torch tensor
    mask_pil = p.image_mask
    mask_pil_mono = mask_pil.convert("L")
    mask = np.array(mask_pil_mono).astype(np.float32) / 255.0
    mask = torch.from_numpy(mask)

    # Add the mask to the conditioning
    conditioning_set_mask = ConditioningSetMask()
    (masked_positive,) = conditioning_set_mask.append(p.positive, mask, "mask bounds", 1)
    (masked_negative,) = conditioning_set_mask.append(p.negative, mask, "mask bounds", 1)

    # Generate samples
    (samples,) = common_ksampler(p.model, p.seed, p.steps, p.cfg, p.sampler_name,
                                 p.scheduler, masked_positive, masked_negative, encoded, denoise=p.denoise)
    
    # Decode the sample
    vae_decoder = VAEDecodeTiled()
    (decoded,) = vae_decoder.decode(p.vae, samples)
    
    # Convert the sample to a PIL image
    image = tensor_to_pil(decoded)

    # Because ComfyUI noises the masked parts of the image as well, the image must be assembled elsewhere
    if shared.tiled_image is None:
        shared.tiled_image = image
    else:
        # Add the tile to the tiled image using the mask
        shared.tiled_image = Image.composite(image, shared.tiled_image, mask_pil_mono)

    # Return the original image instead of the generated image because the masked parts of the image are noised
    processed = Processed(p, init_images, p.seed, None)
    return processed
