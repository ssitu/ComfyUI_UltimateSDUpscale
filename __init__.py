import sys
import os
repo_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, repo_dir)
modules = sys.modules.copy()
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# Clean up imports
sys.path.remove(repo_dir)
modules_to_remove = []
for module in sys.modules:
    if module not in modules:
        modules_to_remove.append(module)
for module in modules_to_remove:
    del sys.modules[module]
