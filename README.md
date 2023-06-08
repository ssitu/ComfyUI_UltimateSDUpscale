# ComfyUI_UltimateSDUpscale

 [ComfyUI](https://github.com/comfyanonymous/ComfyUI) nodes for the [Ultimate Stable Diffusion Upscale script by Coyote-A](https://github.com/Coyote-A/ultimate-upscale-for-automatic1111).
 Uses the same script used in the A1111 extension to hopefully replicate images generated using the A1111 webui.

## Installation

Enter the following command from the commandline starting in ComfyUI/custom_nodes/
```
git clone https://github.com/ssitu/ComfyUI_UltimateSDUpscale --recursive
```

## Usage

Nodes can be found in the node menu under `image/upscaling`:

|Node|Description|
| --- | --- |
| Ultimate SD Upscale | The primary node that has the most of the inputs as the original extension script. |
| Ultimate SD Upscale <br>(No Upscale) | Same as the primary node, but without the upscale inputs and assumes that the input image is already upscaled. |

---

Details about the parameters can be found [here](https://github.com/Coyote-A/ultimate-upscale-for-automatic1111/wiki/FAQ#parameters-descriptions).

## Examples

#### Using the ControlNet tile model:

![image](https://github.com/ssitu/ComfyUI_UltimateSDUpscale/assets/57548627/64f8d3b2-10ae-45ee-9f8a-40b798a51655)
