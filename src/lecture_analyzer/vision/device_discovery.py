"""Camera device enumeration, discovery, and hardware classification for Windows.

Identifies physical laptop webcams vs. mobile phone virtual cameras (e.g. Motorola Smart Connect),
allowing automatic selection of the built-in laptop camera and user selection in the UI.
"""

from dataclasses import dataclass
import json
import logging
import platform
import subprocess
from typing import Dict, List, Optional, Tuple

import cv2

logger = logging.getLogger("lecture_analyzer.vision.device_discovery")

# Keywords identifying physical laptop / integrated webcams
LAPTOP_CAMERA_KEYWORDS = [
    "asus", "integrated", "built-in", "webcam", "fhd", "hd", "facetime",
    "internal", "user facing", "front", "camera sensor"
]

# Keywords identifying mobile phone / virtual streaming cameras
VIRTUAL_PHONE_KEYWORDS = [
    "smart connect", "readyfor", "droidcam", "iriun", "epoccam",
    "camo", "virtual", "obs", "ndi", "mobile"
]


@dataclass
class CameraDeviceInfo:
    """Metadata describing an enumerated camera device."""
    index: int
    name: str
    manufacturer: str
    device_id: str
    is_laptop_webcam: bool
    is_virtual_phone: bool
    width: int
    height: int
    backend: str

    @property
    def display_name(self) -> str:
        """Formatted label for UI dropdowns."""
        if self.is_laptop_webcam:
            return f"Camera {self.index}: {self.name} (Laptop Built-in Webcam)"
        elif self.is_virtual_phone:
            return f"Camera {self.index}: {self.name} (Mobile Phone / Virtual)"
        else:
            return f"Camera {self.index}: {self.name}"


def _query_windows_pnp_cameras() -> List[Dict[str, str]]:
    """Queries Windows PnP subsystem for active video/camera entities."""
    if platform.system() != "Windows":
        return []

    try:
        cmd = [
            "powershell", "-NoProfile", "-NonInteractive", "-Command",
            "Get-CimInstance Win32_PnPEntity | Where-Object { $_.PNPClass -eq 'Camera' } | "
            "Select-Object Name, DeviceID, Manufacturer, Present | ConvertTo-Json"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=4)
        if res.returncode == 0 and res.stdout.strip():
            data = json.loads(res.stdout)
            if isinstance(data, dict):
                data = [data]
            return [d for d in data if d.get("Present", True)]
    except Exception as e:
        logger.debug(f"PnP camera query error: {e}")
    return []


def discover_cameras(max_indices: int = 4) -> List[CameraDeviceInfo]:
    """Enumerates available physical and virtual cameras, identifying laptop webcams."""
    pnp_devices = _query_windows_pnp_cameras()
    discovered: List[CameraDeviceInfo] = []

    for idx in range(max_indices):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        backend = "DSHOW"
        if not cap.isOpened():
            cap = cv2.VideoCapture(idx)
            backend = "DEFAULT"

        if cap.isOpened():
            ret, frame = cap.read()
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
            cap.release()

            if ret and frame is not None and frame.size > 0:
                # Correlate with PnP metadata if available
                dev_meta = pnp_devices[idx] if idx < len(pnp_devices) else {}
                name = dev_meta.get("Name", f"Camera Device {idx}")
                mfg = dev_meta.get("Manufacturer", "Generic")
                dev_id = dev_meta.get("DeviceID", "")

                name_lower = name.lower()
                dev_id_lower = dev_id.lower()

                # Detect if device is a virtual phone camera
                is_phone = any(k in name_lower for k in VIRTUAL_PHONE_KEYWORDS) or "root\\camera" in dev_id_lower

                # Detect if device is a laptop built-in webcam
                is_laptop = (not is_phone) and (
                    any(k in name_lower for k in LAPTOP_CAMERA_KEYWORDS)
                    or "usb\\vid" in dev_id_lower
                    or "pci\\" in dev_id_lower
                )

                info = CameraDeviceInfo(
                    index=idx,
                    name=name,
                    manufacturer=mfg,
                    device_id=dev_id,
                    is_laptop_webcam=is_laptop,
                    is_virtual_phone=is_phone,
                    width=w,
                    height=h,
                    backend=backend,
                )
                discovered.append(info)
                logger.info(
                    f"Discovered {info.display_name} [{w}x{h} via {backend}], "
                    f"is_laptop={is_laptop}, is_phone={is_phone}"
                )

    return discovered


def get_recommended_camera_index(discovered: Optional[List[CameraDeviceInfo]] = None) -> int:
    """Returns the index of the laptop's built-in webcam, falling back to 0 if none found."""
    cams = discovered if discovered is not None else discover_cameras()
    if not cams:
        return 0

    # Prioritize laptop built-in webcams
    for cam in cams:
        if cam.is_laptop_webcam:
            return cam.index

    # Second choice: any non-virtual camera
    for cam in cams:
        if not cam.is_virtual_phone:
            return cam.index

    return cams[0].index
