"""
Data downloader utility for fetching data files from Google Cloud Storage.

Downloads the required CSV data files if they are not present locally.
"""

import requests
from pathlib import Path
from typing import Tuple, Optional, Callable, Dict

# Data file configurations
DATA_FILES = {
    "hcm_amenities.csv": {
        "url": "https://storage.googleapis.com/listing_data_real_estate/hcm_amenities.csv",
        "description": "HCM City amenities data for feature engineering",
    },
    "s_listing_ingestion.csv": {
        "url": "https://storage.googleapis.com/listing_data_real_estate/s_listing_ingestion.csv",
        "description": "Sell listings data from nhatot.com",
    },
    "u_listing_ingestion.csv": {
        "url": "https://storage.googleapis.com/listing_data_real_estate/u_listing_ingestion.csv",
        "description": "Rent listings data from nhatot.com",
    },
}

# Default data directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


def get_data_path(file_name: str, data_dir: Optional[Path] = None) -> Path:
    """Get the full path for a data file."""
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    return data_dir / file_name


def check_data_files_exist(data_dir: Optional[Path] = None) -> Dict[str, dict]:
    """
    Check which data files exist locally.

    Returns:
        Dictionary with file names as keys and info dict as values
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    result = {}
    for file_name in DATA_FILES:
        path = data_dir / file_name
        result[file_name] = {
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
        print(f"[DataDownloader] Starting download: {url}")

        # Ensure parent directory exists
        destination.parent.mkdir(parents=True, exist_ok=True)

        # Create a temporary file for downloading
        temp_path = destination.with_suffix(".downloading")

        # Start download with streaming
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0

        print(f"[DataDownloader] Total size: {total_size / (1024*1024):.1f} MB")

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
        print(f"[DataDownloader] Downloaded successfully: {destination.name} ({size_mb:.1f} MB)")
        return True, f"Downloaded successfully ({size_mb:.1f} MB)"

    except requests.exceptions.Timeout:
        print(f"[DataDownloader] ERROR: Download timed out")
        return False, "Download timed out. Please check your internet connection."
    except requests.exceptions.ConnectionError:
        print(f"[DataDownloader] ERROR: Connection error")
        return False, "Connection error. Please check your internet connection."
    except requests.exceptions.HTTPError as e:
        print(f"[DataDownloader] ERROR: HTTP error {e.response.status_code}")
        return False, f"HTTP error: {e.response.status_code}"
    except Exception as e:
        print(f"[DataDownloader] ERROR: {str(e)}")
        # Clean up partial download
        temp_path = destination.with_suffix(".downloading")
        if temp_path.exists():
            temp_path.unlink()
        return False, f"Download failed: {str(e)}"


def download_data_if_missing(
    data_dir: Optional[Path] = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> Tuple[bool, dict]:
    """
    Check for data files and download from GCS if missing.

    Args:
        data_dir: Directory for data files (defaults to PROJECT_ROOT/data)
        progress_callback: Optional callback(file_name, downloaded_bytes, total_bytes)

    Returns:
        Tuple of (all_success, results_dict)
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    # Ensure data directory exists
    data_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    all_success = True

    for file_name, config in DATA_FILES.items():
        file_path = data_dir / file_name

        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            results[file_name] = {
                "status": "exists",
                "message": f"Already exists ({size_mb:.1f} MB)",
                "path": str(file_path),
            }
            print(f"[DataDownloader] {file_name}: Already exists ({size_mb:.1f} MB)")
            continue

        print(f"[DataDownloader] Downloading {file_name}: {config['description']}")

        # Download the file
        def file_progress(downloaded: int, total: int):
            if progress_callback:
                progress_callback(file_name, downloaded, total)

        success, message = download_file(
            url=config["url"],
            destination=file_path,
            progress_callback=file_progress,
        )

        results[file_name] = {
            "status": "downloaded" if success else "failed",
            "message": message,
            "path": str(file_path) if success else None,
        }

        if not success:
            all_success = False

    return all_success, results


def ensure_data_exists(data_dir: Optional[Path] = None) -> Tuple[bool, str]:
    """
    Simple function to ensure all data files exist, downloading if necessary.

    Returns:
        Tuple of (success, summary_message)
    """
    print("[DataDownloader] Checking data files...")
    success, results = download_data_if_missing(data_dir)

    messages = []
    for file_name, result in results.items():
        status = result["status"]
        if status == "exists":
            messages.append(f"  {file_name}: already present")
        elif status == "downloaded":
            messages.append(f"  {file_name}: downloaded successfully")
        else:
            messages.append(f"  {file_name}: FAILED - {result['message']}")

    summary = "\n".join(messages)
    print(f"[DataDownloader] Summary:\n{summary}")
    return success, summary


if __name__ == "__main__":
    # Allow running directly for testing
    print("Checking and downloading data files...")
    success, summary = ensure_data_exists()
    print(f"\nResult: {'Success' if success else 'Failed'}")
    print(summary)
