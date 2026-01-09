import logging
import re
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO)


def _fetch_hf_html(repo_id: str, folder_path: str) -> str:
    """Fetch HTML from HuggingFace tree page."""
    url = f"https://huggingface.co/datasets/{repo_id}/tree/main/{folder_path}"
    with urllib.request.urlopen(url) as response:
        return response.read().decode("utf-8")


def list_hf_subfolders(repo_id: str, folder_path: str) -> list[str]:
    """List subfolders in a HuggingFace dataset folder."""
    try:
        html = _fetch_hf_html(repo_id, folder_path)
        pattern = rf'/datasets/{repo_id}/tree/main/({folder_path}/[^"/?]+)'
        return sorted(set(re.findall(pattern, html)))
    except Exception as e:
        logging.error(f"Failed to list subfolders in {folder_path}: {e}")
        return []


def list_hf_files(
    repo_id: str,
    folder_path: str,
    extensions: tuple = (".jpg", ".jpeg", ".png", ".webp"),
) -> list[str]:
    """List image files in a HuggingFace dataset folder."""
    try:
        html = _fetch_hf_html(repo_id, folder_path)
        pattern = rf'/datasets/{repo_id}/blob/main/({folder_path}/[^"]+?({"|".join(e for e in extensions)}))'
        return [match[0] for match in re.findall(pattern, html)]
    except Exception as e:
        logging.error(f"Failed to list files in {folder_path}: {e}")
        return []


def download_test_images(save_dir: str, repo_folder: str, repo_id: str) -> str:
    """Download the test_images/ folder from the HF test dataset repo"""
    # Discover all subfolders and collect files
    subfolders = list_hf_subfolders(repo_id, repo_folder)
    if not subfolders:
        logging.warning(f"No subfolders found in {repo_folder}")
        return str(save_dir)

    all_files = [f for folder in subfolders for f in list_hf_files(repo_id, folder)]
    if not all_files:
        logging.warning(f"No image files found in {repo_folder}")
        return str(save_dir)

    logging.info(f"Found {len(all_files)} files from {len(subfolders)} folders")
    # Download files, preserving folder structure
    save_dir_path = Path(save_dir)
    downloaded = 0
    skipped = 0
    for file_path in all_files:
        relative_path = Path(file_path).relative_to(repo_folder)
        save_path = save_dir_path / relative_path
        if save_path.exists():
            logging.info(f"Skipping {relative_path} (already exists)")
            skipped += 1
            continue

        save_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{file_path}"
        logging.info(f"Downloading {relative_path}...")
        urllib.request.urlretrieve(url, save_path)
        downloaded += 1

    logging.info(f"Downloaded {downloaded} files, skipped {skipped} existing files")
    return str(save_dir_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_test_images(
        repo_id="ssitu/ultimatesdupscale_test",
        save_dir="./test/test_images/",
        repo_folder="test_images",
    )
