import torchvision.transforms.functional as TF


def img_tensor_mae(tensor1, tensor2):
    """Calculate the mean absolute difference between two image tensors."""
    # Remove batch dimensions if present
    tensor1 = tensor1.squeeze(0).cpu()
    tensor2 = tensor2.squeeze(0).cpu()
    if tensor1.shape != tensor2.shape:
        raise ValueError(
            f"Tensors must have the same shape for comparison. Got {tensor1.shape=} and {tensor2.shape=}."
        )
    return (tensor1 - tensor2).abs().mean().item()


def blur(tensor, kernel_size=9, sigma=None):
    """Apply Gaussian blur to an image tensor."""
    # [1, H, W, C] -> [1, C, H, W]
    if tensor.ndim == 4:
        tensor = tensor.permute(0, 3, 1, 2)
    elif tensor.ndim == 3:
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    else:
        raise ValueError(f"Expected a 3D or 4D tensor, got {tensor.ndim=}")
    return TF.gaussian_blur(tensor, kernel_size=kernel_size, sigma=sigma).permute(0, 2, 3, 1)  # type: ignore
