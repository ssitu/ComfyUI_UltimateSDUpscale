import pathlib


class DirectoryConfig:
    """Helper class for test directories."""

    def __init__(self, test_images: pathlib.Path, sample_images: pathlib.Path):
        self.test_images = test_images
        self.sample_images = sample_images
