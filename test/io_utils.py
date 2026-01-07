import pathlib
from PIL import Image
import usdu_utils


def save_image(tensor, path: pathlib.Path):
    """The goto function to save a tensor image to the sampled images directory."""
    assert tensor.ndim == 3 or (tensor.ndim == 4 and tensor.shape[0] == 1), (
        f"Expected a 3D tensor (H, W, C) or (1, H, W, C), got {tensor.ndim=}"
    )
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    image = usdu_utils.tensor_to_pil(tensor.cpu())
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=75, optimize=True)


def load_image(path: pathlib.Path, device=None):
    """Load an image from disk and convert it to a tensor."""
    return usdu_utils.pil_to_tensor(Image.open(path)).to(device=device)
