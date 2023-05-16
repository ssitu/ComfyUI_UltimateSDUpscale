# ComfyUI_UltimateSDUpscale
 ComfyUI nodes for the Ultimate Stable Diffusion Upscale script by Coyote-A.
 Uses the same script used in the A1111 extension to hopefully replicate images generated in the A1111 webui.

## Installation
Enter the following commands from the commandline starting in ComfyUI/custom_nodes/
```
git clone https://github.com/ssitu/ComfyUI_UltimateSDUpscale.git
cd ComfyUI_UltimateSDUpscale
git submodule update --init --recursive
```

## Usage
Available nodes:
```
image > upscaling > UltimateSDUpscale
```

## Todo
- Make the encoding faster between tiles.
- Add the seam fixes.
- More accurate replication of images made using the Coyote-A upscale extension in A1111.
