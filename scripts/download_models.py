"""Utility script to download and verify required computer-vision model weights.

License and Provenance:
- face_landmarker.task: Google MediaPipe (Apache 2.0 License)
- yolov8n-pose.pt: Ultralytics YOLOv8 (AGPL-3.0 License)
- yolov8n.pt: Ultralytics YOLOv8 (AGPL-3.0 License)
"""

import os
import sys
import urllib.request
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lecture_analyzer.utils.paths import resolve_path

MODELS = {
    "face_landmarker.task": {
        "url": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
        "min_size_bytes": 3_000_000,
        "description": "Google MediaPipe 3D FaceLandmarker Task Model",
        "license": "Apache 2.0",
    },
    "yolov8n-pose.pt": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.pt",
        "min_size_bytes": 6_000_000,
        "description": "Ultralytics YOLOv8n Human Pose Estimation Model",
        "license": "AGPL-3.0",
    },
    "yolov8n.pt": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
        "min_size_bytes": 6_000_000,
        "description": "Ultralytics YOLOv8n Object Detection Model",
        "license": "AGPL-3.0",
    },
}


def download_file(url: str, dest_path: Path, description: str) -> bool:
    print(f"[*] Downloading {description}...")
    print(f"    Source: {url}")
    print(f"    Destination: {dest_path}")

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")

    def progress(blocks, block_size, total_size):
        if total_size > 0:
            downloaded = blocks * block_size
            pct = min(100.0, (downloaded / total_size) * 100.0)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(f"\r    Progress: {pct:5.1f}% ({mb:5.1f}/{total_mb:5.1f} MB)", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, str(temp_path), reporthook=progress)
        print()  # newline
        if temp_path.exists():
            if dest_path.exists():
                dest_path.unlink()
            temp_path.rename(dest_path)
            print(f"[+] Successfully saved {dest_path.name} ({dest_path.stat().st_size:,} bytes)")
            return True
    except Exception as e:
        print(f"\n[-] Download failed for {url}: {e}")
        if temp_path.exists():
            temp_path.unlink()
        return False
    return False


def verify_models(models_dir: Path) -> dict:
    status = {}
    print(f"\n=== Model Verification (Directory: {models_dir}) ===")
    for filename, info in MODELS.items():
        file_path = models_dir / filename
        if file_path.exists():
            size = file_path.stat().st_size
            if size >= info["min_size_bytes"]:
                status[filename] = True
                print(f" [PASS] {filename} ({size:,} bytes) - {info['description']} [{info['license']}]")
            else:
                status[filename] = False
                print(f" [FAIL] {filename} is too small ({size:,} bytes, expected >= {info['min_size_bytes']:,})")
        else:
            status[filename] = False
            print(f" [MISSING] {filename} - {info['description']}")
    return status


def main():
    models_dir = resolve_path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    status = verify_models(models_dir)
    missing = [fn for fn, ok in status.items() if not ok]

    if not missing:
        print("\nAll required model weights are present and verified.")
        return 0

    print(f"\n{len(missing)} model(s) missing or incomplete. Initiating download...")
    for fn in missing:
        info = MODELS[fn]
        dest = models_dir / fn
        success = download_file(info["url"], dest, info["description"])
        if not success:
            print(f"[-] Error: Could not download {fn}. Please check network connection.")
            return 1

    post_status = verify_models(models_dir)
    if all(post_status.values()):
        print("\n[SUCCESS] All models downloaded and verified.")
        return 0
    else:
        print("\n[ERROR] Some models failed verification after download.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
