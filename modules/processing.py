# Patched classes to adapt from A1111 webui for ComfyUI
from nodes import common_ksampler, VAEEncode, VAEDecode
from utils import pil_to_tensor, tensor_to_pil, get_crop_region, expand_crop, crop_cond
from PIL import Image, ImageFilter

if (not hasattr(Image, 'Resampling')):  # For older versions of Pillow
    Image.Resampling = Image


class StableDiffusionProcessing:

    def __init__(self, init_img, model, positive, negative, vae, seed, steps, cfg, sampler_name, scheduler, denoise):
        # Variables used by the USDU script
        self.init_images = [init_img]
        self.image_mask = None
        self.mask_blur = 0
        self.inpaint_full_res_padding = 0
        self.width = init_img.width
        self.height = init_img.height

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

        # Variables used only by this script
        self.init_size = init_img.width, init_img.height

        # Other required A1111 variables for the USDU script that is currently unused in this script
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

    # Setup
    image_mask = p.image_mask.convert('L')
    init_image = p.init_images[0]

    # Blur the mask
    if p.mask_blur > 0:
        image_mask = image_mask.filter(ImageFilter.GaussianBlur(p.mask_blur))

    # Locate the white region of the mask outlining the tile and add padding
    crop_region = get_crop_region(image_mask, p.inpaint_full_res_padding)
    crop_region, (p.width, p.height) = expand_crop(crop_region, image_mask.width, image_mask.height)

    # Crop the init_image to get the tile that will be used for generation
    tile = init_image.crop(crop_region)
    initial_tile_size = tile.size
    if tile.size != (p.width, p.height):
        tile = tile.resize((p.width, p.height), Image.Resampling.LANCZOS)

    # Crop conditioning
    positive_cropped = crop_cond(p.positive, crop_region, p.init_size, init_image.size, (p.width, p.height))
    negative_cropped = crop_cond(p.negative, crop_region, p.init_size, init_image.size, (p.width, p.height))

    # Encode the image
    vae_encoder = VAEEncode()
    (latent,) = vae_encoder.encode(p.vae, pil_to_tensor(tile))

    # Generate samples
    (samples,) = common_ksampler(p.model, p.seed, p.steps, p.cfg, p.sampler_name,
                                 p.scheduler, positive_cropped, negative_cropped, latent, denoise=p.denoise)

    # Decode the sample
    vae_decoder = VAEDecode()
    (decoded,) = vae_decoder.decode(p.vae, samples)

    # Convert the sample to a PIL image
    tile_sampled = tensor_to_pil(decoded)

    # Resize back to the original size
    if tile.size != (p.width, p.height):
        tile_sampled = tile_sampled.resize(initial_tile_size, Image.Resampling.LANCZOS)

    # Put the tile into position
    image_tile_only = Image.new('RGBA', init_image.size)
    image_tile_only.paste(tile_sampled, crop_region[:2])

    # Add the mask as an alpha channel
    # Must make a copy due to the possibility of an edge becoming black
    temp = image_tile_only.copy()
    temp.putalpha(image_mask)
    image_tile_only.paste(temp, image_tile_only)

    # Add back the tile to the initial image according to the mask in the alpha channel
    result = init_image.convert('RGBA')
    result.alpha_composite(image_tile_only)

    # Convert back to RGB
    result = result.convert('RGB')

    processed = Processed(p, [result], p.seed, None)
    return processed
