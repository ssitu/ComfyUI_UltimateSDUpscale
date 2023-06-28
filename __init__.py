import sys
import os
from itertools import dropwhile
repo_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, repo_dir)
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# Clean up imports
sys.path.remove(repo_dir)
repo_name = os.path.basename(repo_dir)
imported_modules = [k for k, v in dropwhile(lambda item: item[0] != repo_name, sys.modules.items())]
for m in imported_modules[1:]:
    del sys.modules[m]
