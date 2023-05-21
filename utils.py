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


def get_crop_region(mask, pad=0):
    # Takes a black and white PIL image in 'L' mode and returns the coordinates of the white rectangular mask region
    # Should be equivalent to the get_crop_region function from https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/masking.py
    coordinates = mask.getbbox()
    if coordinates is not None:
        x1, y1, x2, y2 = coordinates
    else:
        x1, y1, x2, y2 = mask.width, mask.height, 0, 0
    return (
        int(max(x1 - pad, 0)),
        int(max(y1 - pad, 0)),
        int(min(x2 + pad, mask.width)),
        int(min(y2 + pad, mask.height))
    )


def expand_crop(region, width, height):
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


def resize_crop(region, init_size, resize_size):
    # Resize a crop region so that it fits an image that was resized to the given width and height
    x1, y1, x2, y2 = region
    init_width, init_height = init_size
    resize_width, resize_height = resize_size
    x1 = math.floor(x1 * resize_width / init_width)
    x2 = math.ceil(x2 * resize_width / init_width)
    y1 = math.floor(y1 * resize_height / init_height)
    y2 = math.ceil(y2 * resize_height / init_height)
    return (x1, y1, x2, y2)


def crop_cond(cond, region, p_size, image_size):
    cropped = []
    for emb, x in cond:
        n = [emb, x.copy()]
        if "control" in n[1]:
            cnet = n[1]["control"]
            im = controlnet_hint_to_pil(cnet.cond_hint_original)
            resized_crop = resize_crop(region, image_size, im.size)
            im = im.crop(resized_crop)
            im = im.resize(p_size, Image.Resampling.NEAREST)
            cnet.cond_hint = pil_to_controlnet_hint(im).to(cnet.device)
        cropped.append(n)
    return cropped
