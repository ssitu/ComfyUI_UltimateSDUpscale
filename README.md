# ComfyUI_UltimateSDUpscale

[ComfyUI](https://github.com/comfyanonymous/ComfyUI) nodes for performing the image-to-image diffusion process on large images in tiles. This approach improves the details that is commonly found on upscaled images while reducing hardware requirements and maintaining an image size that the diffusion model is trained on.

## Installation


### Using Git
1. Git must be installed on your system. Verify by running `git -v` in a terminal.
2. Enter the following command from the terminal starting in ComfyUI/custom_nodes/
    ```
    git clone https://github.com/ssitu/ComfyUI_UltimateSDUpscale
    ```

### ComfyUI Manager
1. [ComfyUI Manager](https://github.com/Comfy-Org/ComfyUI-Manager) must be installed.
2. After launching ComfyUI, open ComfyUI Manager and select the "Custom Nodes Manager" option.
3. Search for "UltimateSDUpscale" and install the node. Select latest for the most up-to-date version.
4. Follow any prompts to restart ComfyUI.

### comfy-cli

1. [comfy-cli](https://github.com/Comfy-Org/comfy-cli) must be installed.
2. Run this command from the terminal: `comfy node install comfyui_ultimatesdupscale`

### Manual Download
1. Download the zip file from https://registry.comfy.org/nodes/comfyui_ultimatesdupscale to select the version you want, or obtain the current nightly version by clicking the green "Code" button on the GitHub repository page and selecting "Download ZIP".
2. Create a new folder in the `ComfyUI/custom_nodes/` directory to hold the extracted files (e.g. `ComfyUI/custom_nodes/ComfyUI_UltimateSDUpscale`).
3. Extract the contents of the zip file into the `ComfyUI/custom_nodes/ComfyUI_UltimateSDUpscale` folder.


## Usage

Nodes can be found in the node menu under `image/upscaling`.

Documentation for the nodes can be found in the [`js/docs/`](js/docs/) folder, or viewed within the application by right-clicking the relevant node and selecting the info icon.

Details about most of the parameters can be found [here](https://github.com/Coyote-A/ultimate-upscale-for-automatic1111/wiki/FAQ#parameters-descriptions).

Example workflows can be found in the [`example_workflows/`](example_workflows/) folder. You can also find them in the ComfyUI application under the Templates menu, scroll down the left sidebar to find the Extensions section, then selecting this repository.

## References
* Ultimate Stable Diffusion Upscale script for the Automatic1111 Web UI: https://github.com/Coyote-A/ultimate-upscale-for-automatic1111
* ComfyUI: https://github.com/comfyanonymous/ComfyUI