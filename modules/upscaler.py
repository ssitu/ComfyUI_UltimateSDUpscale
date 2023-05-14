import PIL
from utils import tensor_to_pil, pil_to_tensor
from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel
from modules import shared


class Upscaler:
    def upscale(self, img: PIL.Image, scale, selected_model: str = None):
        tensor = pil_to_tensor(img)
        image_upscale_node = ImageUpscaleWithModel()
        (upscaled,) = image_upscale_node.upscale(
            shared.actual_upscaler, tensor)
        return tensor_to_pil(upscaled)


class UpscalerData:
    name = ""
    data_path = ""

    def __init__(self):
        self.scaler = Upscaler()
