import logging


class SilenceLogs:
    """Context manager to temporarily silence logging."""

    def __enter__(self):
        logging.disable(logging.CRITICAL)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        logging.disable(logging.NOTSET)


def execute(node, *args, **kwargs):
    """Execute a ComfyUI node, handling both V3 and legacy schemas."""
    if hasattr(node, "execute"):
        return node.execute(*args, **kwargs)
    else:
        return getattr(node(), node.FUNCTION)(*args, **kwargs)

def img_tensor_diff(tensor1, tensor2):
    """Calculate the mean absolute difference between two image tensors."""
    # Remove batch dimensions if present
    tensor1 = tensor1.squeeze(0).cpu()
    tensor2 = tensor2.squeeze(0).cpu()
    if tensor1.shape != tensor2.shape:
        raise ValueError(f"Tensors must have the same shape for comparison. Got {tensor1.shape=} and {tensor2.shape=}.")
    return (tensor1 - tensor2).abs().mean().item()
