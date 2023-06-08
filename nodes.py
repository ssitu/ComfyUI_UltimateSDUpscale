# ComfyUI Node for Ultimate SD Upscale by Coyote-A: https://github.com/Coyote-A/ultimate-upscale-for-automatic1111

import os
import sys
import comfy
from .repositories import ultimate_upscale as ult
from .utils import tensor_to_pil, pil_to_tensor
from modules.processing import StableDiffusionProcessing
import modules.shared as shared
from modules.upscaler import UpscalerData
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "comfy"))


MAX_RESOLUTION = 8192
# The modes available for Ultimate SD Upscale
MODES = {
    "Linear": ult.USDUMode.LINEAR,
    "Chess": ult.USDUMode.CHESS,
    "None": ult.USDUMode.NONE,
}
# The seam fix modes
SEAM_FIX_MODES = {
    "None": ult.USDUSFMode.NONE,
    "Band Pass": ult.USDUSFMode.BAND_PASS,
    "Half Tile": ult.USDUSFMode.HALF_TILE,
    "Half Tile + Intersections": ult.USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS,
}


def USDU_base_inputs():
    return [
        ("image", ("IMAGE",)),
        # Sampling Params
        ("model", ("MODEL",)),
        ("positive", ("CONDITIONING",)),
        ("negative", ("CONDITIONING",)),
        ("vae", ("VAE",)),
        ("upscale_by", ("FLOAT", {"default": 2, "min": 0.05, "max": 4, "step": 0.05})),
        ("seed", ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})),
        ("steps", ("INT", {"default": 20, "min": 1, "max": 10000, "step": 1})),
        ("cfg", ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0})),
        ("sampler_name", (comfy.samplers.KSampler.SAMPLERS,)),
        ("scheduler", (comfy.samplers.KSampler.SCHEDULERS,)),
        ("denoise", ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})),
        # Upscale Params
        ("upscale_model", ("UPSCALE_MODEL",)),
        ("mode_type", (list(MODES.keys()),)),
        ("tile_width", ("INT", {"default": 512, "min": 8, "max": MAX_RESOLUTION, "step": 8})),
        ("tile_height", ("INT", {"default": 512, "min": 8, "max": MAX_RESOLUTION, "step": 8})),
        ("mask_blur", ("INT", {"default": 8, "min": 0, "max": 64, "step": 1})),
        ("tile_padding", ("INT", {"default": 32, "min": 0, "max": 128, "step": 8})),
        # Seam fix params
        ("seam_fix_mode", (list(SEAM_FIX_MODES.keys()),)),
        ("seam_fix_denoise", ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})),
        ("seam_fix_width", ("INT", {"default": 64, "min": 0, "max": 128, "step": 8})),
        ("seam_fix_mask_blur", ("INT", {"default": 8, "min": 0, "max": 64, "step": 1})),
        ("seam_fix_padding", ("INT", {"default": 16, "min": 0, "max": 128, "step": 8})),
    ]


def prepare_inputs(required: list, optional: list = None):
    inputs = {}
    if required:
        inputs["required"] = {}
        for name, type in required:
            inputs["required"][name] = type
    if optional:
        inputs["optional"] = {}
        for name, type in optional:
            inputs["optional"][name] = type
    return inputs


def remove_input(inputs: list, input_name: str):
    for i, (n, _) in enumerate(inputs):
        if n == input_name:
            del inputs[i]
            break


def rename_input(inputs: list, old_name: str, new_name: str):
    for i, (n, t) in enumerate(inputs):
        if n == old_name:
            inputs[i] = (new_name, t)
            break


class UltimateSDUpscale:
    @classmethod
    def INPUT_TYPES(s):
        return prepare_inputs(USDU_base_inputs())

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"

    def upscale(self, image, model, positive, negative, vae, upscale_by, seed,
                steps, cfg, sampler_name, scheduler, denoise, upscale_model,
                mode_type, tile_width, tile_height, mask_blur, tile_padding,
                seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                seam_fix_width, seam_fix_padding):
        #
        # Set up A1111 patches
        #

        # Upscaler
        # An object that the script works with
        shared.sd_upscalers[0] = UpscalerData()
        # Where the actual upscaler is stored, will be used when the script upscales using the Upscaler in UpscalerData
        shared.actual_upscaler = upscale_model

        # Processing
        sdprocessing = StableDiffusionProcessing(
            tensor_to_pil(image), model, positive, negative, vae, seed, steps, cfg, sampler_name, scheduler, denoise
        )

        #
        # Running the script
        #
        script = ult.Script()
        processed = script.run(p=sdprocessing, _=None, tile_width=tile_width, tile_height=tile_height,
                               mask_blur=mask_blur, padding=tile_padding, seams_fix_width=seam_fix_width,
                               seams_fix_denoise=seam_fix_denoise, seams_fix_padding=seam_fix_padding,
                               upscaler_index=0, save_upscaled_image=False, redraw_mode=MODES[mode_type],
                               save_seams_fix_image=False, seams_fix_mask_blur=seam_fix_mask_blur,
                               seams_fix_type=SEAM_FIX_MODES[seam_fix_mode], target_size_type=2,
                               custom_width=None, custom_height=None, custom_scale=upscale_by)

        # Return the resulting image
        upscaled_image = pil_to_tensor(processed.images[0])
        return (upscaled_image,)


class UltimateSDUpscaleNoUpscale:
    @classmethod
    def INPUT_TYPES(s):
        required = USDU_base_inputs()
        remove_input(required, "upscale_model")
        remove_input(required, "upscale_by")
        rename_input(required, "image", "upscaled_image")
        return prepare_inputs(required)

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"

    def upscale(self, upscaled_image, model, positive, negative, vae, seed,
                steps, cfg, sampler_name, scheduler, denoise,
                mode_type, tile_width, tile_height, mask_blur, tile_padding,
                seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                seam_fix_width, seam_fix_padding):
        
        shared.sd_upscalers[0] = UpscalerData()
        shared.actual_upscaler = None
        sdprocessing = StableDiffusionProcessing(
            tensor_to_pil(upscaled_image), model, positive, negative, vae, seed, steps, cfg, sampler_name, scheduler, denoise
        )

        script = ult.Script()
        processed = script.run(p=sdprocessing, _=None, tile_width=tile_width, tile_height=tile_height,
                               mask_blur=mask_blur, padding=tile_padding, seams_fix_width=seam_fix_width,
                               seams_fix_denoise=seam_fix_denoise, seams_fix_padding=seam_fix_padding,
                               upscaler_index=0, save_upscaled_image=False, redraw_mode=MODES[mode_type],
                               save_seams_fix_image=False, seams_fix_mask_blur=seam_fix_mask_blur,
                               seams_fix_type=SEAM_FIX_MODES[seam_fix_mode], target_size_type=2,
                               custom_width=None, custom_height=None, custom_scale=1)

        upscaled_image = pil_to_tensor(processed.images[0])
        return (upscaled_image,)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "UltimateSDUpscale": UltimateSDUpscale,
    "UltimateSDUpscaleNoUpscale": UltimateSDUpscaleNoUpscale
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "UltimateSDUpscale": "Ultimate SD Upscale",
    "UltimateSDUpscaleNoUpscale": "Ultimate SD Upscale (No Upscale)"
}
