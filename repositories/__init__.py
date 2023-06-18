import os
import sys
import importlib.util

repos_path = os.path.dirname(os.path.realpath(__file__))

# Add the root directory to the path
root_path = os.path.dirname(repos_path)
sys.path.insert(0, root_path)

# Import the script
script_name = "scripts/ultimate-upscale"
repo_name = "ultimate_sd_upscale"
script_path = f"{repos_path}/{repo_name}/{script_name}.py"
spec = importlib.util.spec_from_file_location(script_name, script_path)
ultimate_upscale = importlib.util.module_from_spec(spec)
sys.modules[script_name] = ultimate_upscale
spec.loader.exec_module(ultimate_upscale)
