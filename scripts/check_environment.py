"""Environment and hardware diagnostics tool.

Performs verification of:
1. Python version and architecture
2. Required Python package imports
3. OpenCV, MediaPipe, Ultralytics, and PyTorch runtimes
4. GPU acceleration status (CUDA)
5. Offline model weight files
6. Configuration file parsing
7. Local SQLite database accessibility
8. Physical camera probing (optional via --probe-camera flag)
"""

import argparse
import os
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))


def check_python_version() -> bool:
    print(f"[*] Python Version: {sys.version.split()[0]} ({sys.platform}, {sys.maxsize > 2**32 and '64-bit' or '32-bit'})")
    if sys.version_info < (3, 10):
        print("    [FAIL] Python 3.10+ required.")
        return False
    print("    [PASS] Python version compatible.")
    return True


def check_imports() -> bool:
    packages = [
        ("cv2", "OpenCV"),
        ("mediapipe", "MediaPipe"),
        ("ultralytics", "Ultralytics YOLO"),
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("scipy", "SciPy"),
        ("sklearn", "Scikit-Learn"),
        ("yaml", "PyYAML"),
        ("PySide6", "PySide6 (Qt)"),
    ]
    all_ok = True
    print("\n[*] Verifying Package Imports:")
    for mod_name, label in packages:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "unknown")
            print(f"    [PASS] {label:20s} (v{ver})")
        except ImportError as e:
            print(f"    [FAIL] {label:20s} (Missing: {e})")
            all_ok = False
    return all_ok


def check_gpu() -> None:
    print("\n[*] Hardware Acceleration / GPU Status:")
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        if cuda_avail:
            device_name = torch.cuda.get_device_name(0)
            vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            print(f"    [PASS] CUDA Available: True | Device: {device_name} ({vram_mb:.0f} MB VRAM)")
        else:
            print("    [INFO] CUDA Not Available in PyTorch build. Execution will proceed on CPU (supported).")
    except Exception as e:
        print(f"    [WARN] Unable to probe GPU: {e}")


def check_models() -> bool:
    print("\n[*] Model Files Verification:")
    models_dir = PROJECT_ROOT / "models"
    from scripts.download_models import verify_models
    res = verify_models(models_dir)
    return all(res.values())


def check_configuration() -> bool:
    print("\n[*] Configuration Verification:")
    try:
        from lecture_analyzer.config import load_config
        cfg = load_config()
        print(f"    [PASS] Configuration valid: app_name='{cfg.system.app_name}', camera_idx={cfg.camera.index}")
        return True
    except Exception as e:
        print(f"    [FAIL] Configuration error: {e}")
        return False


def check_database() -> bool:
    print("\n[*] Database Initialization Check:")
    try:
        db_path = PROJECT_ROOT / "data" / "database" / "test_diag.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE IF NOT EXISTS _diag_test (id INTEGER PRIMARY KEY, ts TEXT)")
        conn.execute("INSERT INTO _diag_test (ts) VALUES (datetime('now'))")
        conn.commit()
        conn.close()
        db_path.unlink()
        print("    [PASS] SQLite read/write operations verified.")
        return True
    except Exception as e:
        print(f"    [FAIL] Database check failed: {e}")
        return False


def probe_physical_camera(cam_idx: int = 0) -> bool:
    print(f"\n[*] Physical Camera Probe (index {cam_idx}):")
    try:
        import cv2
        cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(cam_idx)
        if not cap.isOpened():
            print(f"    [INFO] Camera index {cam_idx} cannot be opened. (Normal if no camera connected or in background mode)")
            return False
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            print(f"    [PASS] Camera index {cam_idx} captured test frame ({w}x{h}).")
            return True
        else:
            print(f"    [WARN] Camera opened but failed to capture frame.")
            return False
    except Exception as e:
        print(f"    [WARN] Exception while probing camera: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Diagnostics for Lecture Engagement Pattern Analyzer")
    parser.add_argument("--probe-camera", action="store_true", help="Attempt to probe physical webcam")
    args = parser.parse_args()

    print("======================================================================")
    print(" Lecture Engagement Pattern Analyzer - Environment Diagnostics")
    print("======================================================================")

    py_ok = check_python_version()
    imp_ok = check_imports()
    check_gpu()
    model_ok = check_models()
    cfg_ok = check_configuration()
    db_ok = check_database()

    cam_tested = False
    cam_ok = False
    if args.probe_camera:
        cam_tested = True
        cam_ok = probe_physical_camera(0)
    else:
        print("\n[*] Camera Probe: Skipped (pass --probe-camera to probe physical hardware).")

    all_passed = py_ok and imp_ok and model_ok and cfg_ok and db_ok

    print("\n======================================================================")
    if all_passed:
        print(" [SUMMARY] Environment is READY.")
    else:
        print(" [SUMMARY] Environment has ISSUES that need attention.")
    print("======================================================================")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
