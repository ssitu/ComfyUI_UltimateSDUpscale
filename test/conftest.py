"""
Setup for the ComfyUI engine and shared test fixtures.
"""

import os
import sys
from pathlib import Path

# Because of manipulations to sys.path, delayed imports are used to avoid issues

# conftest.py is in repo_root/test/ directory
REPO_ROOT = Path(__file__).parent.parent.resolve()
COMFYUI_ROOT = REPO_ROOT.parent.parent.resolve()

import pytest
import asyncio
import logging

from helpers import SilenceLogs, execute
from configs import DirectoryConfig


def pytest_configure(config):
    """Called before test collection begins."""
    # Ensure submodule root is in path for test imports
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    # Ensure ComfyUI path is set up
    if str(COMFYUI_ROOT) not in sys.path:
        sys.path.insert(0, str(COMFYUI_ROOT))

    # from comfy.cli_args import args
    # args.cpu = True  # Force CPU mode for tests
    # args.force_fp16 = True  # Force float16 mode for tests


#
# # Configuration
#
TEST_CHECKPOINT = "v1-5-pruned-emaonly-fp16.safetensors"
TEST_UPSCALE_MODEL = "4x-UltraSharp.pth"


#
# # Path Setup
#
def _setup_comfyui_paths():
    """Configure ComfyUI folder paths for testing."""
    # Ensure modules containing a utils.py are NOT in sys.path
    # The comfy directory must be removed to prevent comfy/utils.py from shadowing
    # ComfyUI's utils/ package directory when we import utils.extra_config
    to_remove = [
        str(COMFYUI_ROOT / "comfy"),
    ]
    for path_to_remove in to_remove:
        while path_to_remove in sys.path:
            sys.path.remove(path_to_remove)

    # Ensure ComfyUI is in path
    if str(COMFYUI_ROOT) not in sys.path:
        sys.path.insert(0, str(COMFYUI_ROOT))

    # Apply custom paths
    # main.py will trigger a warning that torch was already imported, probably by pytest. Shouldn't be a problem as far as I know.
    from main import apply_custom_paths

    apply_custom_paths()


def _disable_other_custom_nodes():
    """Disable other custom nodes by adding .disabled to their directories. Ideally, a fresh ComfyUI install is used instead."""
    custom_nodes_root = COMFYUI_ROOT / "custom_nodes"
    renamed_directories = []
    for node_dir in custom_nodes_root.iterdir():
        # Ignore non-directories, this repository, already disabled nodes, and __pycache__
        if (
            node_dir.is_dir()
            and node_dir != REPO_ROOT
            and node_dir.suffix != ".disabled"
            and not node_dir.name.startswith("__")
        ):
            try:
                disabled_dir = node_dir.with_suffix(".disabled")
                os.rename(node_dir, disabled_dir)
                logging.info(f"Disabled custom node: {node_dir.name}")
                renamed_directories.append(disabled_dir)
            except Exception as e:
                logging.warning(f"Failed to disable custom node {node_dir.name}: {e}")
    return renamed_directories


def _enable_other_custom_nodes(renamed_directories):
    """Re-enable other custom nodes by removing .disabled from their directories"""
    for node_dir in renamed_directories:
        try:
            os.rename(node_dir, node_dir.with_suffix(""))
            logging.info(f"Re-enabled custom node: {node_dir.name}")
        except Exception as e:
            logging.warning(f"Failed to re-enable custom node {node_dir.name}: {e}")


#
# # Fixtures
#
@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def comfyui_initialized():
    """Initialize ComfyUI nodes once per test session."""
    from nodes import init_extra_nodes

    renamed_directories = _disable_other_custom_nodes()
    _setup_comfyui_paths()

    async def _init():
        with SilenceLogs():
            await init_extra_nodes(init_api_nodes=False)

    asyncio.run(_init())

    yield True

    # Teardown: Re-enable the disabled custom nodes after all tests complete
    _enable_other_custom_nodes(renamed_directories)


@pytest.fixture(scope="session")
def node_classes(comfyui_initialized):
    """Get ComfyUI node class mappings."""
    from nodes import NODE_CLASS_MAPPINGS

    return NODE_CLASS_MAPPINGS


@pytest.fixture(scope="session")
def test_checkpoint():
    """Find and return a valid test checkpoint."""
    import folder_paths

    checkpoints = folder_paths.get_filename_list("checkpoints")
    # TODO: Should probably use a hash instead of matching the filename
    if TEST_CHECKPOINT not in checkpoints:
        pytest.skip(f"No test checkpoint found. Please add {TEST_CHECKPOINT}")

    return TEST_CHECKPOINT


@pytest.fixture(scope="session")
def loaded_checkpoint(comfyui_initialized, test_checkpoint, node_classes):
    """Load checkpoint and return (model, clip, vae) tuple."""
    import torch

    with torch.inference_mode():
        CheckpointLoaderSimple = node_classes["CheckpointLoaderSimple"]
        model, clip, vae = execute(CheckpointLoaderSimple, test_checkpoint)

    return model, clip, vae


@pytest.fixture(scope="session")
def upscale_model(comfyui_initialized, node_classes):
    """Load the first available upscale model."""
    import torch
    import folder_paths

    UpscaleModelLoader = node_classes["UpscaleModelLoader"]

    upscale_models = folder_paths.get_filename_list("upscale_models")
    # TODO: Should probably use a hash instead of matching the filename
    if TEST_UPSCALE_MODEL not in upscale_models:
        pytest.skip("No upscale models found")

    model_name = upscale_models[0]
    with torch.inference_mode():
        (model,) = execute(UpscaleModelLoader, model_name)

    return model


@pytest.fixture
def test_dirs():
    """Return paths to test and sample image directories."""
    test_dir = REPO_ROOT / "test"
    return DirectoryConfig(
        test_images=test_dir / "test_images",
        sample_images=test_dir / "sample_images",
    )


@pytest.fixture(scope="class")
def seed():
    """Default seed for reproducible tests."""
    return 1
