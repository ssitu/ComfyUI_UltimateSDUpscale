# Patched classes to adapt from A111 webui for ComfyUI

class StableDiffusionProcessing:
    seed = 0
    extra_generation_params = {}

    def __init__(self, init_img):
        self.init_images = [init_img]


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
    # Return original images for now
    processed = Processed(p, p.init_images, p.seed, None)
    return processed
