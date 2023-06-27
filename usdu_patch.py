# Make some patches to the script
from repositories import ultimate_upscale as usdu
import modules.shared as shared
import math
from PIL import Image


if (not hasattr(Image, 'Resampling')):  # For older versions of Pillow
    Image.Resampling = Image

#
# Instead of using multiples of 64, use multiples of 8
#

# Upscaler
old_init = usdu.USDUpscaler.__init__


def new_init(self, p, image, upscaler_index, save_redraw, save_seams_fix, tile_width, tile_height):
    p.width = math.ceil((image.width * p.upscale_by) / 8) * 8
    p.height = math.ceil((image.height * p.upscale_by) / 8) * 8
    old_init(self, p, image, upscaler_index, save_redraw, save_seams_fix, tile_width, tile_height)


usdu.USDUpscaler.__init__ = new_init

# Redraw
old_setup_redraw = usdu.USDURedraw.init_draw


def new_setup_redraw(self, p, width, height):
    mask, draw = old_setup_redraw(self, p, width, height)
    p.width = math.ceil((self.tile_width + self.padding) / 8) * 8
    p.height = math.ceil((self.tile_height + self.padding) / 8) * 8
    return mask, draw


usdu.USDURedraw.init_draw = new_setup_redraw

# Seams fix
old_setup_seams_fix = usdu.USDUSeamsFix.init_draw


def new_setup_seams_fix(self, p):
    old_setup_seams_fix(self, p)
    p.width = math.ceil((self.tile_width + self.padding) / 8) * 8
    p.height = math.ceil((self.tile_height + self.padding) / 8) * 8


usdu.USDUSeamsFix.init_draw = new_setup_seams_fix


#
# Make the script upscale on a batch of images instead of one image
#

old_upscale = usdu.USDUpscaler.upscale


def new_upscale(self):
    old_upscale(self)
    shared.batch = [self.image] + \
        [img.resize((self.p.width, self.p.height), resample=Image.LANCZOS) for img in shared.batch[1:]]


usdu.USDUpscaler.upscale = new_upscale
