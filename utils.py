import numpy as np
import PIL.Image as Image
import torch

def tensor_to_pil(img_tensor):
    # Takes a batch of 1 rgb image and returns an RGB PIL image
    i = 255. * img_tensor.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8).squeeze())
    return img

def pil_to_tensor(img):
    # Takes a 3 channel PIL image and returns a tensor of shape [1, height, width, 3]
    image = img.convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,]
    return image
