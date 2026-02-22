from contextlib import contextmanager
import torch

from usdu_utils import resize_region


@contextmanager
def crop_model_cond(
    model, crop_regions, init_size, canvas_size, tile_size, latent_crop=False
):
    """
    Context manager to crop model patches that may contain controlnet hints.

    Usage:
        with crop_model_cond(model, ...) as patched_model:
            # Use patched_model here
            ...
    """
    patched_model = model.clone()
    patches = patched_model.model_options.get("transformer_options", {}).get(
        "patches", {}
    )
    patched = set()
    for module, module_patches in patches.items():
        for patch in module_patches:
            if id(patch) in patched:
                # Avoid cropping the same patch multiple times if it appears in multiple modules
                continue
            if type(patch).__name__ in ("DiffSynthCnetPatch", "ZImageControlPatch"):
                crop_control_patch(patch, crop_regions, canvas_size, latent_crop)
            patched.add(id(patch))
    try:
        yield patched_model
    finally:
        # Restore original model
        # Not needed for now; model_options is a deepcopy when model is cloned.
        pass


def crop_control_patch(patch, crop_regions, canvas_size, latent_crop=True):
    """
    Crop controlnet patch images and latents.

    Args:
        patch: The controlnet patch (ZImageControlPatch or DiffSynthCnetPatch)
        crop_regions: List of (x1, y1, x2, y2) crop coordinates for each tile in the batch
        canvas_size: (width, height) of the canvas
        latent_crop: If True, crop the encoded latent directly without re-encoding.
                     If False, crop pixel image and re-encode via VAE.
    """
    patch_class = type(patch).__name__
    required_attrs = (
        "image",
        "model_patch",
        "vae",
        "strength",
        "encoded_image",
        "encoded_image_size",
    )
    missing_attrs = [attr for attr in required_attrs if not hasattr(patch, attr)]
    assert not missing_attrs, (
        f"{patch_class} is missing required attributes: {', '.join(missing_attrs)}"
    )

    # Normalize to list of regions
    if not isinstance(crop_regions, list):
        crop_regions = [crop_regions]

    # Crop the pixel space image
    assert len(patch.image.shape) == 4, (
        f"Expected image to have 4 dimensions (b, h, w, c), got {patch.image.shape}"
    )

    # Calculate crop region relative to image size (image is [b, h, w, c])
    image_size = (patch.image.shape[2], patch.image.shape[1])  # (w, h)

    # Crop and collect for each region
    cropped_images = []
    for crop_region in crop_regions:
        resized_crop = resize_region(crop_region, canvas_size, image_size)
        x1, y1, x2, y2 = resized_crop
        cropped_image = patch.image[:, y1:y2, x1:x2, :]
        cropped_images.append(cropped_image)

    # Concatenate all cropped images along the batch dimension

    concatenated_image = torch.cat(cropped_images, dim=0)
    print(
        f"Cropped {patch_class} image from {patch.image.shape} to {concatenated_image.shape}"
    )
    patch.image = concatenated_image
    patch.encoded_image_size = (
        concatenated_image.shape[1],
        concatenated_image.shape[2],
    )

    if latent_crop:
        # Crop the encoded latent directly without re-encoding
        downscale_ratio = patch.vae.spacial_compression_encode()
        # encoded_image is [b, c, h, w] and encoded_image_size is (h, w) in pixel space
        assert len(patch.encoded_image.shape) == 4, (
            f"Expected encoded_image to have 4 dimensions (b, c, h, w), got {patch.encoded_image.shape}"
        )

        # Crop and collect latents for each region
        cropped_latents = []
        for crop_region in crop_regions:
            resized_crop = resize_region(crop_region, canvas_size, image_size)
            # Convert pixel crop to latent space crop
            x1, y1, x2, y2 = tuple(x // downscale_ratio for x in resized_crop)
            cropped_latent = patch.encoded_image[:, :, y1:y2, x1:x2]
            cropped_latents.append(cropped_latent)

        # Concatenate all cropped latents along the batch dimension
        # and update the patch with cropped latent
        patch.encoded_image = torch.cat(cropped_latents, dim=0)
    else:
        # Re-encode the cropped image by calling __init__
        # This will encode the cropped_image and update encoded_image/encoded_image_size
        # ZImageControlPatch supports inpaint_image, may have to account for that in the future
        patch.__init__(
            patch.model_patch,
            patch.vae,
            concatenated_image,
            patch.strength,
            # inpaint_image=patch.inpaint_image,
            mask=patch.mask,
        )
