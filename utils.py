import numpy as np
from PIL import Image
import torch
import math


def tensor_to_pil(img_tensor):
    # Takes a batch of 1 image in the form of a tensor of shape [1, channels, height, width]
    # and returns an PIL Image with the corresponding mode deduced by the number of channels
    i = 255. * img_tensor.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8).squeeze())
    return img


def pil_to_tensor(image):
    # Takes a PIL image and returns a tensor of shape [1, height, width, channels]
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image).unsqueeze(0)
    if len(image.shape) == 3:  # If the image is grayscale, add a channel dimension
        image = image.unsqueeze(-1)
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
    # Apply padding
    x1 = max(x1 - pad, 0)
    y1 = max(y1 - pad, 0)
    x2 = min(x2 + pad, mask.width)
    y2 = min(y2 + pad, mask.height)
    return fix_crop_region((x1, y1, x2, y2), (mask.width, mask.height))


def fix_crop_region(region, image_size):
    # Remove the extra pixel added by the get_crop_region function
    image_width, image_height = image_size
    x1, y1, x2, y2 = region
    if x2 < image_width:
        x2 -= 1
    if y2 < image_height:
        y2 -= 1
    return x1, y1, x2, y2


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


def resize_region(region, init_size, resize_size):
    # Resize a crop so that it fits an image that was resized to the given width and height
    x1, y1, x2, y2 = region
    init_width, init_height = init_size
    resize_width, resize_height = resize_size
    x1 = math.floor(x1 * resize_width / init_width)
    x2 = math.ceil(x2 * resize_width / init_width)
    y1 = math.floor(y1 * resize_height / init_height)
    y2 = math.ceil(y2 * resize_height / init_height)
    return (x1, y1, x2, y2)


def crop_controlnet(cond_dict, region, init_size, canvas_size, tile_size):
    if "control" not in cond_dict:
        return
    controlnet = cond_dict["control"]
    im = controlnet_hint_to_pil(controlnet.cond_hint_original)
    resized_crop = resize_region(region, canvas_size, im.size)
    im = im.crop(resized_crop)
    im = im.resize(tile_size, Image.Resampling.NEAREST)
    if hasattr(controlnet, "channels_in") and controlnet.channels_in == 1:  # For t2i adapter
        im = im.convert("L")
    hint = pil_to_controlnet_hint(im)
    controlnet.cond_hint = hint.to(controlnet.device)


def region_intersection(region1, region2):
    """
    Returns the coordinates of the intersection of two rectangular regions.
    :param region1: A tuple of the form (x1, y1, x2, y2) denoting the upper left and the lower right points 
        of the first rectangular region. Expected to have x2 > x1 and y2 > y1.
    :param region2: The second rectangular region with the same format as the first.
    :return: A tuple of the form (x1, y1, x2, y2) denoting the rectangular intersection. 
        None if there is no intersection.
    """
    x1, y1, x2, y2 = region1
    x1_, y1_, x2_, y2_ = region2
    x1 = max(x1, x1_)
    y1 = max(y1, y1_)
    x2 = min(x2, x2_)
    y2 = min(y2, y2_)
    if x1 >= x2 or y1 >= y2:
        return None
    return (x1, y1, x2, y2)


def crop_gligen(cond_dict, region, init_size, canvas_size, tile_size):
    if "gligen" not in cond_dict:
        return
    type, model, cond = cond_dict["gligen"]
    if type != "position":
        from warnings import warn
        warn(f"Unknown gligen type {type}")
        return
    cropped = []
    for c in cond:
        emb, h, w, y, x = c
        # Get the coordinates of the box in the upscaled image
        x1 = x * 8
        y1 = y * 8
        x2 = x1 + w * 8
        y2 = y1 + h * 8
        gligen_upscaled_box = resize_region((x1, y1, x2, y2), init_size, canvas_size)

        # Calculate the intersection of the gligen box and the region
        intersection = region_intersection(gligen_upscaled_box, region)
        if intersection is None:
            continue
        x1, y1, x2, y2 = intersection

        # Offset the gligen box so that the origin is at the top left of the tile region
        x1 -= region[0]
        y1 -= region[1]
        x2 -= region[0]
        y2 -= region[1]

        # Set the new position params
        h = (y2 - y1) // 8
        w = (x2 - x1) // 8
        x = x1 // 8
        y = y1 // 8
        cropped.append((emb, h, w, y, x))
    cond_dict["gligen"] = (type, model, cropped)


def crop_area(cond_dict, region, init_size, canvas_size, tile_size):
    if "area" not in cond_dict:
        return
    # Resize the area conditioning to the canvas size and confine it to the tile region
    h, w, y, x = cond_dict["area"]
    w, h, x, y = 8 * w, 8 * h, 8 * x, 8 * y
    x1, y1, x2, y2 = resize_region((x, y, x + w, y + h), init_size, canvas_size)
    intersection = region_intersection((x1, y1, x2, y2), region)
    if intersection is None:
        del cond_dict["area"]
        del cond_dict["strength"]
        return
    x1, y1, x2, y2 = intersection
    # Offset origin to the top left of the tile
    x1 -= region[0]
    y1 -= region[1]
    x2 -= region[0]
    y2 -= region[1]
    # Set the params for tile
    w, h = (x2 - x1) // 8, (y2 - y1) // 8
    x, y = x1 // 8, y1 // 8
    cond_dict["area"] = (h, w, y, x)


def crop_mask(cond_dict, region, init_size, canvas_size, tile_size):
    if "mask" not in cond_dict:
        return
    mask = cond_dict["mask"]  # (1, H, W)
    # Convert to PIL image
    mask = tensor_to_pil(mask)  # W x H
    # Resize the mask to the canvas size
    mask = mask.resize(canvas_size, Image.Resampling.BICUBIC)
    # Crop the mask to the region
    mask = mask.crop(region)
    # Resize the mask to the tile size
    if tile_size != mask.size:
        mask = mask.resize(tile_size, Image.Resampling.BICUBIC)
    # Remove mask if it is all white
    mask_bbox = mask.getbbox()
    if mask_bbox is not None:
        # Check if mask is completely contains the tile
        if region_intersection(region, mask_bbox) == region:
            del cond_dict["mask"]
            del cond_dict["mask_strength"]
            return
    # Convert back to tensor
    mask = pil_to_tensor(mask)  # (1, H, W, 1)
    mask = mask.squeeze(-1)  # (1, H, W)
    cond_dict["mask"] = mask


def crop_cond(cond, region, init_size, canvas_size, tile_size):
    cropped = []
    for emb, x in cond:
        cond_dict = x.copy()
        n = [emb, cond_dict]
        crop_controlnet(cond_dict, region, init_size, canvas_size, tile_size)
        crop_gligen(cond_dict, region, init_size, canvas_size, tile_size)
        crop_area(cond_dict, region, init_size, canvas_size, tile_size)
        crop_mask(cond_dict, region, init_size, canvas_size, tile_size)
        cropped.append(n)
    return cropped
