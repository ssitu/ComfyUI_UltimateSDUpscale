# Running Tests

This directory contains tests for ComfyUI_UltimateSDUpscale.

## Prerequisites

- These tests assume that ComfyUI is installed using a virtual environment
- Activate the ComfyUI virtual environment before running tests
- The checkpoint `v1-5-pruned-emaonly-fp16.safetensors` is available
- The upscale model `4x-UltraSharp.pth` is available

## Running Tests

### Using the convenience scripts (works from repo root or test directory):

**Linux/Mac (Bash):**
```bash
./test/run_tests.sh       # From repo root
./run_tests.sh            # From test directory
```
run_tests.sh will forward all arguments into pytest.

### Using pytest directly (must be in test directory):

```bash
cd test
pytest          # Run all tests
pytest -v       # Verbose
```

### Common pytest options:

- `-v` - Verbose output
- `-s` - Show print statements
- `--log-cli-level=INFO` - Show info-level logs
- `-k PATTERN` - Run tests matching pattern
- `--lf` - Run last failed tests

## Test Structure

- `conftest.py` - Pytest configuration, fixtures, and path setup
- `sample_images/` - Generated test images for visual inspection
- `test_images/` - Reference images used as inputs or expected outputs

## Troubleshooting

If you encounter import errors:
1. Make sure you're running from the `test/` directory
2. Verify the virtual environment is activated
