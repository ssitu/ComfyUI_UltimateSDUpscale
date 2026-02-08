`Ultimate SD Upscale (No Upscale)` applies tiled image-to-image proc5. **Performance Optimization**
    - Enable `tiled_decode` if you're running out of VRAM during decoding, and want to skip the default behavior of attempting normal decoding.
    - Use the largest tile size that the model and VRAM can handle to reduce the number of tiles needed.
    - Disable `force_uniform_tiles` to only denoise what will be visible after pasting back the tile. This can save processing time, but the model used may not be trained for the resulting tile sizes, and the model will be missing the context around the tile that may otherwise be available with this option enabled.ng to an already upscaled image to enhance details and fix seams, without performing the initial upscaling step with an upscale model.

This variant of the Ultimate SD Upscale node is designed for situations where you already have an upscaled image and only want to apply the tiled redraw and seam fix steps. This is useful when you've upscaled an image using a different method or upscaler and want to use USDU's tiled refinement capabilities to add details and remove artifacts.

The image goes through the redraw step if the tiling order is not set to "None". A tile is selected from the image, defined by the tiling order and tile parameters from the node widgets. The tile is used as input for an image-to-image process, using the sampling-related parameters given by the node widgets. The tile is then pasted back onto the image at the appropriate position. This continues until all tiles have been processed.

After the redraw step, the seam fix step is applied if enabled. There are various strategies for fixing seams, defined by the `seam_fix_mode` parameter from the node widgets. The seam fix step uses the same image-to-image process as the redraw step, but applied to areas between tiles from the redraw step.

## Inputs

| Parameter | Data Type | Input Method | Default | Range | Description |
|-----------|-----------|--------------|---------|--------|-------------|
| `upscaled_image` | IMAGE | Image Input | None | - | The already upscaled image to refine with tiled processing. |
| `model` | MODEL | Model Selection | None | - | The model to use for image-to-image processing on each tile. |
| `positive` | CONDITIONING | Conditioning Input | None | - | The positive conditioning for each tile during the redraw step. |
| `negative` | CONDITIONING | Conditioning Input | None | - | The negative conditioning for each tile during the redraw step. |
| `vae` | VAE | Model Selection | None | - | The VAE model to use for encoding and decoding tiles. |
| `seed` | INT | Number Input | 0 | 0-18446744073709551615 | The seed to use for image-to-image processing, ensuring reproducible results. |
| `steps` | INT | Number Input | 20 | 1-10000 | The number of sampling steps to use for each tile during the redraw step and seam fix step. |
| `cfg` | FLOAT | Slider | 8.0 | 0.0-100.0 | The CFG (Classifier Free Guidance) scale to use for each tile. Higher values make the output follow the prompt more closely. The recommended values depend on the model. |
| `sampler_name` | COMBO | Dropdown | - | Available samplers | The sampler to use for each tile during the image-to-image process. |
| `scheduler` | COMBO | Dropdown | - | Available schedulers | The scheduler to use for each tile during the sampling process. |
| `denoise` | FLOAT | Slider | 0.2 | 0.0-1.0 (step 0.01) | The denoising strength to use for each tile. Higher values allow more creative changes, but more chance of seams. |
| `mode_type` | COMBO | Dropdown | - | Linear, Chess, None | The tiling order to use for the redraw step. Linear processes tiles row by row, Chess uses a checkerboard pattern, and None skips the redraw step. |
| `tile_width` | INT | Number Input | 512 | 64-8192 (step 8) | The base width of each tile during the redraw step. |
| `tile_height` | INT | Number Input | 512 | 64-8192 (step 8) | The base height of each tile during the redraw step. |
| `mask_blur` | INT | Number Input | 8 | 0-64 | The blur radius for the mask applied to tiles, helping blend tiles seamlessly. A higher value means more of the original image is retained near the seams when pasting the refined tiles back on the upscaled image. |
| `tile_padding` | INT | Number Input | 32 | 0-8192 (step 8) | The padding to apply to tiles, providing more context for better blending. Adds to tile size (e.g. (`tile_width` + `tile_padding`)x(`tile_height` + `tile_padding`)). |
| `seam_fix_mode` | COMBO | Dropdown | - | None, Band Pass, Half Tile, Half Tile + Intersections | The seam fix mode to use. Different modes apply different strategies to fix visible seams between tiles. |
| `seam_fix_denoise` | FLOAT | Slider | 1.0 | 0.0-1.0 (step 0.01) | The denoising strength to use for the seam fix step. |
| `seam_fix_width` | INT | Number Input | 64 | 0-8192 (step 8) | The width of the bands used for the Band Pass seam fix mode. |
| `seam_fix_mask_blur` | INT | Number Input | 8 | 0-64 | The blur radius for the seam fix mask, ensuring smooth blending. |
| `seam_fix_padding` | INT | Number Input | 16 | 0-8192 (step 8) | The padding to apply for the seam fix step. Adds to tile size. |
| `force_uniform_tiles` | BOOLEAN | Toggle | True | True/False | If enabled, tiles that would be cut off by the edges of the image will expand using context around the tile to keep the same tile size determined by `tile_width`, `tile_height`, and `tile_padding`. This is what happens in the A1111 Web UI. If disabled, the minimal size for tiles will be used, which may make the sampling faster but may cause artifacts due to irregular tile sizes. |
| `tiled_decode` | BOOLEAN | Toggle | False | True/False | Whether to use tiled decoding when decoding tiles. Useful when you know the ComfyUI engine will attempt a normal decode and run into an Out Of Memory error, and resorts to tiled decoding anyway. |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `IMAGE` | IMAGE | The final refined image. |

## Usage Tips

1. **When to Use This Node**
    - You've already upscaled an image with a different upscaler and want to add details.
    - You want to fix seams or artifacts in an existing high-resolution image.
    - You want more control by separating the upscaling and refinement steps.
    - You want to skip the use of an upscale model, and do a simple upscale with an algorithm like Lanczos or Nearest Neighbor beforehand.

2. **Basic Usage**
    - Typical tile sizes are based on the model resolutions that it is trained on, such as 512x512 for SD1.5 models. If you can generate a coherent image at that resolution, then it is probably a good choice for the tile size.

3. **Tiling Modes**
    - **Linear**: Processes tiles sequentially row by row.
    - **Chess**: Uses a checkerboard pattern, processing every other tile first. Can help reduce visible seams.
    - **None**: Skips the redraw step entirely. Useful if you only want to use the seam fix step to fix visible seams without adding new details.

4. **Denoise Settings**
    - Use a lower denoise (0.05-0.2) to refine the upscaled image to be less blurry, while avoiding seams and hallucinations.
    - Higher denoise values are only usable when using something like a ControlNet tile model to avoid tiles and seams.

5. **Seam Fix Modes**
    - **None**: No seam fixing applied
    - **Band Pass**: Applies processing to band-like areas between tiles
    - **Half Tile**: Processes half-tile overlapping regions
    - **Half Tile + Intersections**: Most thorough, processes half-tiles and their intersections

6. **Performance Optimization**
    - Enable `tiled_decode` if you're running out of VRAM during decoding, and want to skip the default behavior of attempting normal decoding.
    - Use the largest tile size that the model and VRAM can handle to reduce the number of tiles needed.
    - Disable force_uniform_tiles to only denoise what will be visible after pasting back the tile. This can save processing time, but the model used may not be trained for the resulting tile sizes, and the model will be missing the context around the tile that may otherwise be available with this option enabled.

7. **Important Notes**
    - This node does not perform any upscaling; it expects an already upscaled image as input
    - The input image size determines the output size (no scaling is applied)
    - The seam fix step significantly increases processing time. If seams are a problem, it may be better to reduce the denoise or increase tile size instead to avoid the increase in processing time.
