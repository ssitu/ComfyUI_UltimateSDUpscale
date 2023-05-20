import numpy as np
from PIL import Image
import torch
import math


def tensor_to_pil(img_tensor):
    # Takes a batch of 1 rgb image and returns an RGB PIL image
    i = 255. * img_tensor.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8).squeeze())
    return img


def pil_to_tensor(img):
    # Takes a 3 channel PIL image and returns a tensor of shape [1, height, width, 3]
    image = img.convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None, ]
    return image


def controlnet_hint_to_pil(tensor):
    return tensor_to_pil(tensor.movedim(1, -1))


def pil_to_controlnet_hint(img):
    return pil_to_tensor(img).movedim(-1, 1)


def crop_cond(cond, region, p_size, image_size):
    cropped = []
    for emb, x in cond:
        n = [emb, x.copy()]
        if "control" in n[1]:
            cnet = n[1]["control"]
            im = controlnet_hint_to_pil(cnet.cond_hint_original)
            init_size = im.size
            im = im.resize(image_size, Image.Resampling.NEAREST)
            im = im.crop(region)
            im = im.resize(init_size, Image.Resampling.NEAREST)
            if p_size != im.size:
                im = im.resize(p_size, Image.Resampling.LANCZOS)
            cnet.cond_hint = pil_to_controlnet_hint(im).to(cnet.device)
        cropped.append(n)
    return cropped


def get_mask_region(mask, pad=0):
    # Takes a black and white PIL image in 'L' mode and returns the coordinates of the white rectangular mask region
    # Should be equivalent to the get_crop_region function from https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/masking.py
    coordinates = mask.getbbox()
    if coordinates is not None:
        x1, y1, x2, y2 = coordinates
    else:
        x1, y1, x2, y2 = mask.width, mask.height, 0, 0
    return (
        int(max(x1 - pad, 0)),
        int(max(y1-pad, 0)),
        int(min(x2 + pad, mask.width)),
        int(min(y2 + pad, mask.height))
    )


def expand(region, width, height):
    # Expand the crop region to a multiple of 8 for encoding
    x1, y1, x2, y2 = region
    actual_width = x2 - x1
    actual_height = y2 - y1
    p_width = math.ceil(actual_width/8)*8
    p_height = math.ceil(actual_height/8)*8

    # Try to expand region to the right of half the difference
    width_diff = p_width - actual_width
    x2 = min(x2 + width_diff//2, width)
    # Expand region to the left of the difference including the pixels that could not be expanded to the right
    width_diff = p_width - (x2 - x1)
    x1 = max(x1 - width_diff, 0)
    # Try the right again
    width_diff = p_width - (x2 - x1)
    x2 = min(x2 + width_diff, width)

    # Try to expand region to the bottom of half the difference
    height_diff = p_height - actual_height
    y2 = min(y2 + height_diff//2, height)
    # Expand region to the top of the difference including the pixels that could not be expanded to the bottom
    height_diff = p_height - (y2 - y1)
    y1 = max(y1 - height_diff, 0)
    # Try the bottom again
    height_diff = p_height - (y2 - y1)
    y2 = min(y2 + height_diff, height)

    # Width and height should be the same as p_width and p_height
    return (x1, y1, x2, y2), (p_width, p_height)


def expand_crop_region(crop_region, processing_width, processing_height, image_width, image_height):
    # From https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/masking.py
    """expands crop region get_crop_region() to match the ratio of the image the region will processed in; returns expanded region
    for example, if user drew mask in a 128x32 region, and the dimensions for processing are 512x512, the region will be expanded to 128x128."""

    x1, y1, x2, y2 = crop_region

    ratio_crop_region = (x2 - x1) / (y2 - y1)
    ratio_processing = processing_width / processing_height

    if ratio_crop_region > ratio_processing:
        desired_height = (x2 - x1) / ratio_processing
        desired_height_diff = int(desired_height - (y2-y1))
        y1 -= desired_height_diff//2
        y2 += desired_height_diff - desired_height_diff//2
        if y2 >= image_height:
            diff = y2 - image_height
            y2 -= diff
            y1 -= diff
        if y1 < 0:
            y2 -= y1
            y1 -= y1
        if y2 >= image_height:
            y2 = image_height
    else:
        desired_width = (y2 - y1) * ratio_processing
        desired_width_diff = int(desired_width - (x2-x1))
        x1 -= desired_width_diff//2
        x2 += desired_width_diff - desired_width_diff//2
        if x2 >= image_width:
            diff = x2 - image_width
            x2 -= diff
            x1 -= diff
        if x1 < 0:
            x2 -= x1
            x1 -= x1
        if x2 >= image_width:
            x2 = image_width

    return x1, y1, x2, y2


def resize_image(im, width, height):
    # From https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/images.py
    """
    Resizes im to fit within the specified width and height, maintaining the aspect ratio, and then center the image within the dimensions, filling empty with data from image.
    :param im: The image to resize.
    :param width: The width to resize the image to.
    :param height: The height to resize the image to.
    """
    ratio = width / height
    src_ratio = im.width / im.height

    src_w = width if ratio < src_ratio else im.width * height // im.height
    src_h = height if ratio >= src_ratio else im.height * width // im.width

    resized = im.resize((src_w, src_h), resample=Image.Resampling.LANCZOS)
    res = Image.new("RGB", (width, height))
    res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

    if ratio < src_ratio:
        fill_height = height // 2 - src_h // 2
        res.paste(resized.resize((width, fill_height),
                  box=(0, 0, width, 0)), box=(0, 0))
        res.paste(resized.resize((width, fill_height), box=(
            0, resized.height, width, resized.height)), box=(0, fill_height + src_h))
    elif ratio > src_ratio:
        fill_width = width // 2 - src_w // 2
        res.paste(resized.resize((fill_width, height),
                  box=(0, 0, 0, height)), box=(0, 0))
        res.paste(resized.resize((fill_width, height), box=(
            resized.width, 0, resized.width, height)), box=(fill_width + src_w, 0))

    return res
