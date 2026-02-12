"""
Refactored USD Upscaler batch processing patch.

Preserves original behavior but:
- Organizes imports and helpers
- Replaces prints with logging
- Factors duplicated logic (tile preparation, batching, decoding)
- Uses functools.wraps when monkey-patching methods
- Adds type hints and docstrings for clarity
"""

from __future__ import annotations

import logging
import math
import numpy as np
import torch

from functools import wraps
from typing import Tuple, List, Iterable
from PIL import Image, ImageFilter, ImageDraw
from comfy_extras.nodes_custom_sampler import SamplerCustom

import modules.shared as shared
from nodes import common_ksampler, VAEEncode, VAEDecode, VAEDecodeTiled
from repositories import ultimate_upscale as usdu
import usdu_utils

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


# Compatibility for older Pillow versions
try:
    Image.Resampling  # type: ignore
except Exception:
    Image.Resampling = Image  # type: ignore


# -------------------------
# Utility helpers
# -------------------------
def round_length(length: int, multiple: int = 8) -> int:
    """Round length to nearest multiple (default 8)."""
    return round(length / multiple) * multiple




def _sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise, custom_sampler, custom_sigmas):
    """Sampling wrapper that supports a custom sampler or falls back to common_ksampler."""
    if custom_sampler is not None and custom_sigmas is not None:
        kwargs = dict(
            model=model,
            add_noise=True,
            noise_seed=seed,
            cfg=cfg,
            positive=positive,
            negative=negative,
            sampler=custom_sampler,
            sigmas=custom_sigmas,
            latent_image=latent
        )
        if hasattr(SamplerCustom, "execute"):
            (samples, _) = SamplerCustom.execute(**kwargs)
        else:
            custom_sample = SamplerCustom()
            (samples, _) = getattr(custom_sample, custom_sample.FUNCTION)(**kwargs)
        return samples

    (samples,) = common_ksampler(model, seed, steps, cfg, sampler_name,
                                 scheduler, positive, negative, latent, denoise=denoise)
    return samples


# -------------------------
# Monkey patches for USDUpscaler sizing / redraw / seams fix
# -------------------------
def patch_usdu_upscaler_init():
    """Patch USDUpscaler.__init__ to round upscaler p.width/p.height to multiples."""
    old_init = usdu.USDUpscaler.__init__

    @wraps(old_init)
    def new_init(self, p, image, upscaler_index, save_redraw, save_seams_fix, tile_width, tile_height):
        p.width = round_length(image.width * p.upscale_by)
        p.height = round_length(image.height * p.upscale_by)
        return old_init(self, p, image, upscaler_index, save_redraw, save_seams_fix, tile_width, tile_height)

    usdu.USDUpscaler.__init__ = new_init


def patch_usdu_redraw_init():
    """Patch USDURedraw.init_draw to round tile size used for redraw."""
    old_init_draw = usdu.USDURedraw.init_draw

    @wraps(old_init_draw)
    def new_init_draw(self, p, width, height):
        mask, draw = old_init_draw(self, p, width, height)
        p.width = round_length(self.tile_width + self.padding)
        p.height = round_length(self.tile_height + self.padding)
        return mask, draw

    usdu.USDURedraw.init_draw = new_init_draw


def patch_usdu_seams_fix_init():
    old_init = usdu.USDUSeamsFix.init_draw

    @wraps(old_init)
    def new_init(self, p):
        old_init(self, p)
        p.width = round_length(self.tile_width + self.padding)
        p.height = round_length(self.tile_height + self.padding)

    usdu.USDUSeamsFix.init_draw = new_init


def patch_usdu_upscale_method():
    """Patch USDUpscaler.upscale to keep shared.batch resized to p.width/p.height."""
    old_upscale = usdu.USDUpscaler.upscale

    @wraps(old_upscale)
    def new_upscale(self):
        old_upscale(self)
        # Keep shared.batch consistent with the upscaling width/height for subsequent processing.
        shared.batch = [self.image] + [
            img.resize((self.p.width, self.p.height), resample=Image.LANCZOS)
            for img in shared.batch[1:]
        ]

    usdu.USDUpscaler.upscale = new_upscale


# Apply patches
patch_usdu_upscaler_init()
patch_usdu_redraw_init()
patch_usdu_seams_fix_init()
patch_usdu_upscale_method()


# -------------------------
# Patched script.run replacement
# -------------------------
def patched_script_run(self, p, _, tile_width, tile_height, mask_blur, padding, seams_fix_width, seams_fix_denoise, seams_fix_padding,
                      upscaler_index, save_upscaled_image, redraw_mode, save_seams_fix_image, seams_fix_mask_blur,
                      seams_fix_type, target_size_type, custom_width, custom_height, custom_scale):
    """
    Replacement for usdu.Script.run that preserves the original batch_size
    and delegates to the (patched) USDUpscaler and redraw pipeline.
    """
    preserved_batch_size = getattr(p, 'batch_size', 1)
    logger.info("[USDU Batch Debug] Patched script.run() preserving batch_size=%s", preserved_batch_size)

    # Init (matching original code)
    usdu.processing.fix_seed(p)
    usdu.devices.torch_gc()

    # Keep original file-saving flags as in original code
    p.do_not_save_grid = True
    p.do_not_save_samples = True
    p.inpaint_full_res = False

    p.inpainting_fill = 1
    p.n_iter = 1
    p.batch_size = preserved_batch_size

    seed = p.seed

    # Init image
    init_img = p.init_images[0]
    if init_img is None:
        return usdu.processing.Processed(p, [], seed, "Empty image")
    init_img = usdu.images.flatten(init_img, usdu.shared.opts.img2img_background_color)

    # Override size by user choice
    if target_size_type == 1:
        p.width = custom_width
        p.height = custom_height
    elif target_size_type == 2:
        p.width = math.ceil((init_img.width * custom_scale) / 64) * 64
        p.height = math.ceil((init_img.height * custom_scale) / 64) * 64

    # Create and run upscaler
    upscaler = usdu.USDUpscaler(p, init_img, upscaler_index, save_upscaled_image, save_seams_fix_image, tile_width, tile_height)
    upscaler.upscale()

    # Drawing & seams fix setup
    upscaler.setup_redraw(redraw_mode, padding, mask_blur)
    upscaler.setup_seams_fix(seams_fix_padding, seams_fix_denoise, seams_fix_mask_blur, seams_fix_width, seams_fix_type)
    upscaler.print_info()
    upscaler.add_extra_info()
    upscaler.process()
    result_images = upscaler.result_images

    logger.info("[USDU Batch Debug] Patched script.run() complete, batch_size=%s", p.batch_size)
    return usdu.processing.Processed(p, result_images, seed, upscaler.initial_info or "")


# Replace the original script.run with patched version
usdu.Script.run = patched_script_run


# -------------------------
# Batch processing helpers shared between linear and chess modes
# -------------------------
def _prepare_tile_for_batch(calc_rectangle_fn, current_image: Image.Image, tx: int, ty: int, p) -> Tuple[Image.Image, Tuple[int, int, int, int], Image.Image, Tuple[int, int]]:
    """
    Prepare cropped/resized tile, mask, crop-region and tile-size for encoding.
    Returns: (cropped_tile, initial_tile_size, tile_mask, tile_size)
    """
    tile_mask = Image.new("L", (current_image.width, current_image.height), "black")
    tile_draw = ImageDraw.Draw(tile_mask)
    tile_draw.rectangle(calc_rectangle_fn(tx, ty), fill="white")

    crop_region = usdu_utils.get_crop_region(tile_mask, p.inpaint_full_res_padding)

    if p.uniform_tile_mode:
        x1, y1, x2, y2 = crop_region
        crop_w = x2 - x1
        crop_h = y2 - y1
        crop_ratio = crop_w / crop_h if crop_h != 0 else 1.0
        p_ratio = p.width / p.height if p.height != 0 else 1.0
        if crop_ratio > p_ratio:
            target_w = crop_w
            target_h = round(crop_w / p_ratio)
        else:
            target_w = round(crop_h * p_ratio)
            target_h = crop_h
        crop_region, _ = usdu_utils.expand_crop(crop_region, tile_mask.width, tile_mask.height, target_w, target_h)
        tile_size = (p.width, p.height)
    else:
        x1, y1, x2, y2 = crop_region
        crop_w = x2 - x1
        crop_h = y2 - y1
        target_w = math.ceil(crop_w / 8) * 8
        target_h = math.ceil(crop_h / 8) * 8
        crop_region, tile_size = usdu_utils.expand_crop(crop_region, tile_mask.width, tile_mask.height, target_w, target_h)

    # Optional blur
    if getattr(p, "mask_blur", 0) > 0:
        tile_mask = tile_mask.filter(ImageFilter.GaussianBlur(p.mask_blur))

    cropped_tile = current_image.crop(crop_region)
    initial_tile_size = cropped_tile.size
    if cropped_tile.size != tile_size:
        cropped_tile = cropped_tile.resize(tile_size, Image.Resampling.LANCZOS)

    return cropped_tile, initial_tile_size, tile_mask, crop_region, tile_size


def _process_batch_tiles(p,
                         tiles_coords: Iterable[Tuple[int, int]],
                         current_image: Image.Image,
                         calc_rectangle_fn,
                         vae_encoder: VAEEncode,
                         vae_decoder: VAEDecode,
                         vae_decoder_tiled: VAEDecodeTiled) -> Image.Image:
    """Encode, sample and decode a batch of tiles and composite them into current_image."""
    if not tiles_coords:
        return current_image

    batch_tiles = []
    batch_masks = []
    batch_crop_regions = []
    batch_tile_sizes = []

    for tx, ty in tiles_coords:
        cropped_tile, initial_tile_size, tile_mask, crop_region, tile_size = _prepare_tile_for_batch(calc_rectangle_fn, current_image, tx, ty, p)
        batch_tiles.append((cropped_tile, initial_tile_size))
        batch_masks.append(tile_mask)
        batch_crop_regions.append(crop_region)
        batch_tile_sizes.append(tile_size)

    # Encode tiles -> latent
    batched_tensors = torch.cat([usdu_utils.pil_to_tensor(tile) for tile, _ in batch_tiles], dim=0)
    (latent,) = vae_encoder.encode(p.vae, batched_tensors)

    # Condition from first tile (assume same)
    first_tile_size = batch_tile_sizes[0]
    positive_cropped = usdu_utils.crop_cond(p.positive, batch_crop_regions, p.init_size, current_image.size, first_tile_size)
    negative_cropped = usdu_utils.crop_cond(p.negative, batch_crop_regions, p.init_size, current_image.size, first_tile_size)

    # Sampling
    samples = _sample(p.model, p.seed, p.steps, p.cfg, p.sampler_name, p.scheduler,
                      positive_cropped, negative_cropped, latent, p.denoise,
                      p.custom_sampler, p.custom_sigmas)

    # Update progress bar if present
    if getattr(p, "progress_bar_enabled", False) and getattr(p, "pbar", None) is not None:
        p.pbar.update(len(list(tiles_coords)))

    # Decode
    if not getattr(p, "tiled_decode", False):
        (decoded,) = vae_decoder.decode(p.vae, samples)
    else:
        (decoded,) = vae_decoder_tiled.decode(p.vae, samples, 512)

    # Composite tiles back
    result_img = current_image
    for idx, (tx, ty) in enumerate(tiles_coords):
        tile_sampled = usdu_utils.tensor_to_pil(decoded, idx)
        initial_tile_size = batch_tiles[idx][1]
        crop_region = batch_crop_regions[idx]
        tile_mask = batch_masks[idx]

        if tile_sampled.size != initial_tile_size:
            tile_sampled = tile_sampled.resize(initial_tile_size, Image.Resampling.LANCZOS)

        image_tile_only = Image.new('RGBA', result_img.size)
        image_tile_only.paste(tile_sampled, crop_region[:2])

        # Add mask as alpha and composite
        temp = image_tile_only.copy()
        temp.putalpha(tile_mask)
        image_tile_only.paste(temp, image_tile_only)

        result = result_img.convert('RGBA')
        result.alpha_composite(image_tile_only)
        result_img = result.convert('RGB')

    return result_img


# -------------------------
# Replace USDURedraw.linear_process and chess_process with batched variants
# -------------------------
def patch_usdu_linear_and_chess_process():
    old_linear = usdu.USDURedraw.linear_process
    old_chess = usdu.USDURedraw.chess_process

    @wraps(old_linear)
    def new_linear_process(self, p, image, rows, cols):
        batch_size = getattr(p, 'batch_size', 1)
        logger.info("[USDU Batch Debug] linear_process called batch_size=%s rows=%s cols=%s total_tiles=%s", batch_size, rows, cols, rows * cols)

        if batch_size <= 1:
            logger.info("[USDU Batch Debug] Using original single-tile processing (batch_size=%s)", batch_size)
            return old_linear(self, p, image, rows, cols)

        # Batch mode
        vae_encoder = VAEEncode()
        vae_decoder = VAEDecode()
        vae_decoder_tiled = VAEDecodeTiled()

        mask_template, draw_template = self.init_draw(p, image.width, image.height)
        tiles_to_process: List[Tuple[int, int]] = []
        batch_count = 0

        for yi in range(rows):
            for xi in range(cols):
                if shared.state.interrupted:
                    break

                tiles_to_process.append((xi, yi))

                if len(tiles_to_process) >= batch_size or (yi == rows - 1 and xi == cols - 1):
                    batch_count += 1
                    logger.info("[USDU Batch Debug] Processing batch #%s with %s tiles: %s", batch_count, len(tiles_to_process), tiles_to_process)
                    image = _process_batch_tiles(p, tiles_to_process, image, self.calc_rectangle, vae_encoder, vae_decoder, vae_decoder_tiled)
                    tiles_to_process = []

        logger.info("[USDU Batch Debug] Linear processing complete. Processed %s batches total.", batch_count)

        # Update shared.batch[0] with the processed image so it can be retrieved later
        shared.batch[0] = image

        p.width = image.width
        p.height = image.height
        return image

    @wraps(old_chess)
    def new_chess_process(self, p, image, rows, cols):
        batch_size = getattr(p, 'batch_size', 1)
        if batch_size <= 1:
            return old_chess(self, p, image, rows, cols)

        vae_encoder = VAEEncode()
        vae_decoder = VAEDecode()
        vae_decoder_tiled = VAEDecodeTiled()

        mask_template, draw_template = self.init_draw(p, image.width, image.height)

        # Determine tile "white/black" order
        tile_colors = []
        for yi in range(rows):
            row_colors = []
            for xi in range(cols):
                color = xi % 2 == 0
                if yi > 0 and yi % 2 != 0:
                    color = not color
                row_colors.append(color)
            tile_colors.append(row_colors)

        # Helper to iterate tiles in chess order: white first, then black
        def chess_order_iter(white: bool):
            for yi in range(rows):
                for xi in range(cols):
                    if tile_colors[yi][xi] == white:
                        yield (xi, yi)

        # Process white tiles then black tiles
        for color in (True, False):
            tiles_to_process: List[Tuple[int, int]] = []
            for tx, ty in chess_order_iter(color):
                if shared.state.interrupted:
                    break
                tiles_to_process.append((tx, ty))
                if len(tiles_to_process) >= batch_size:
                    image = _process_batch_tiles(p, tiles_to_process, image, self.calc_rectangle, vae_encoder, vae_decoder, vae_decoder_tiled)
                    tiles_to_process = []
            if tiles_to_process:
                image = _process_batch_tiles(p, tiles_to_process, image, self.calc_rectangle, vae_encoder, vae_decoder, vae_decoder_tiled)

        # Update shared.batch[0] with the processed image so it can be retrieved later
        shared.batch[0] = image

        p.width = image.width
        p.height = image.height
        return image

    usdu.USDURedraw.linear_process = new_linear_process
    usdu.USDURedraw.chess_process = new_chess_process


patch_usdu_linear_and_chess_process()
logger.info("USDU batch patches applied successfully.")
