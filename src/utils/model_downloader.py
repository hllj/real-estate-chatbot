"""
Model downloader utility for fetching ML models from GitHub releases.

Downloads the trained XGBoost models if they are not present locally.
"""

import requests
from pathlib import Path
from typing import Tuple, Optional, Callable

# Model configurations
MODELS = {
    "s_with_features_best_model.pkl": {
        "url": "https://storage.googleapis.com/real_estate_trained_models/s_with_features_best_model.pkl",
        "description": "Sell mode model",
    },
    "u_with_features_best_model.pkl": {
        "url": "https://storage.googleapis.com/real_estate_trained_models/u_with_features_best_model.pkl",
        "description": "Rent mode model",
    },
}

# Default models directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"


def get_model_path(model_name: str, model_dir: Optional[Path] = None) -> Path:
    """Get the full path for a model file."""
    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR
    return model_dir / model_name


def check_models_exist(model_dir: Optional[Path] = None) -> dict:
    """
    Check which models exist locally.

    Returns:
        Dictionary with model names as keys and (exists, path) as values
    """
    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR

    result = {}
    for model_name in MODELS:
        path = model_dir / model_name
        result[model_name] = {
            "exists": path.exists(),
            "path": path,
            "size_mb": path.stat().st_size / (1024 * 1024) if path.exists() else 0,
        }
    return result


def download_file(
    url: str,
    destination: Path,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    chunk_size: int = 8192,
) -> Tuple[bool, str]:
    """
    Download a file from URL with progress tracking.

    Args:
        url: URL to download from
        destination: Path to save the file
        progress_callback: Optional callback(downloaded_bytes, total_bytes)
        chunk_size: Size of chunks to download

    Returns:
        Tuple of (success, message)
    """
    try:
        # Ensure parent directory exists
        destination.parent.mkdir(parents=True, exist_ok=True)

        # Create a temporary file for downloading
        temp_path = destination.with_suffix(".downloading")

        # Start download with streaming
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(temp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback:
                        progress_callback(downloaded, total_size)

        # Rename temp file to final destination
        temp_path.rename(destination)

        size_mb = downloaded / (1024 * 1024)
        return True, f"Downloaded successfully ({size_mb:.1f} MB)"

    except requests.exceptions.Timeout:
        return False, "Download timed out. Please check your internet connection."
    except requests.exceptions.ConnectionError:
        return False, "Connection error. Please check your internet connection."
    except requests.exceptions.HTTPError as e:
        return False, f"HTTP error: {e.response.status_code}"
    except Exception as e:
        # Clean up partial download
        temp_path = destination.with_suffix(".downloading")
        if temp_path.exists():
            temp_path.unlink()
        return False, f"Download failed: {str(e)}"


def download_models_if_missing(
    model_dir: Optional[Path] = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> Tuple[bool, dict]:
    """
    Check for model files and download from GitHub releases if missing.

    Args:
        model_dir: Directory for models (defaults to PROJECT_ROOT/models)
        progress_callback: Optional callback(model_name, downloaded_bytes, total_bytes)

    Returns:
        Tuple of (all_success, results_dict)
    """
    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR

    # Ensure models directory exists
    model_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    all_success = True

    for model_name, config in MODELS.items():
        model_path = model_dir / model_name

        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            results[model_name] = {
                "status": "exists",
                "message": f"Already exists ({size_mb:.1f} MB)",
                "path": str(model_path),
            }
            continue

        # Download the model
        def model_progress(downloaded: int, total: int):
            if progress_callback:
                progress_callback(model_name, downloaded, total)

        success, message = download_file(
            url=config["url"],
            destination=model_path,
            progress_callback=model_progress,
        )

        results[model_name] = {
            "status": "downloaded" if success else "failed",
            "message": message,
            "path": str(model_path) if success else None,
        }

        if not success:
            all_success = False

    return all_success, results


def ensure_models_exist(model_dir: Optional[Path] = None) -> Tuple[bool, str]:
    """
    Simple function to ensure all models exist, downloading if necessary.

    Returns:
        Tuple of (success, summary_message)
    """
    success, results = download_models_if_missing(model_dir)

    messages = []
    for model_name, result in results.items():
        status = result["status"]
        if status == "exists":
            messages.append(f"  {model_name}: already present")
        elif status == "downloaded":
            messages.append(f"  {model_name}: downloaded successfully")
        else:
            messages.append(f"  {model_name}: FAILED - {result['message']}")

    summary = "\n".join(messages)
    return success, summary
