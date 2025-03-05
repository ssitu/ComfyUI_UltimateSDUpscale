import sys
import os
# Check for original USDU script
current_dir = os.path.dirname(os.path.realpath(__file__))
repos_dir = os.path.join(current_dir, "repositories")
usdu_dir = os.path.join(repos_dir, "ultimate_sd_upscale")
if len(os.listdir(usdu_dir)) == 0:
    print("[USDU] Original USDU script not found, downloading it from https://github.com/Coyote-A/ultimate-upscale-for-automatic1111")
    import urllib.request
    import zipfile
    import shutil

    # Download the repo
    url = "https://github.com/Coyote-A/ultimate-upscale-for-automatic1111/archive/refs/heads/master.zip"
    zip_path = os.path.join(repos_dir, "usdu.zip")
    urllib.request.urlretrieve(url, zip_path)

    # Extract the repo
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(repos_dir)
    
    # Move the repo to the correct location
    extracted_dir = os.path.join(repos_dir, "ultimate-upscale-for-automatic1111-master")
    shutil.move(extracted_dir, usdu_dir)

    # Clean up
    os.remove(zip_path)
    
    print("[USDU] Original USDU script downloaded successfully")

# Remove other custom_node paths from sys.path to avoid conflicts
custom_node_paths = [path for path in sys.path if "custom_node" in path]
original_sys_path = sys.path.copy()
for path in custom_node_paths:
    sys.path.remove(path)

# Add this repository's path to sys.path for third-party imports
repo_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, repo_dir)
original_modules = sys.modules.copy()

# Place aside potentially conflicting modules
modules_used = [
    "modules",
    "modules.devices",
    "modules.images",
    "modules.processing",
    "modules.scripts",
    "modules.shared",
    "modules.upscaler",
    "utils",
]
original_imported_modules = {}
for module in modules_used:
    if module in sys.modules:
        original_imported_modules[module] = sys.modules.pop(module)

# Proceed with node setup
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# Clean up imports
# Remove any new modules
modules_to_remove = []
for module in sys.modules:
    if module not in original_modules:
        modules_to_remove.append(module)
for module in modules_to_remove:
    del sys.modules[module]

# Restore original modules
sys.modules.update(original_imported_modules)

# Restore original sys.path
sys.path = original_sys_path
