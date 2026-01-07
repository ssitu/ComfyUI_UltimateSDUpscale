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
