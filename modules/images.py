from PIL import Image


def flatten(img, bgcolor):
    # From https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/images.py
    """replaces transparency with bgcolor (example: "#ffffff"), returning an RGB mode image with no transparency"""

    if img.mode == "RGBA":
        background = Image.new('RGBA', img.size, bgcolor)
        background.paste(img, mask=img)
        img = background

    return img.convert('RGB')
