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
