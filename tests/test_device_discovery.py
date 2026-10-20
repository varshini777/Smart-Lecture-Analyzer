"""Tests for camera hardware discovery, PnP device enumeration, and laptop webcam detection."""

from unittest.mock import MagicMock, patch

from lecture_analyzer.vision.device_discovery import (
    CameraDeviceInfo,
    discover_cameras,
    get_recommended_camera_index,
)


def test_camera_device_info_properties():
    """Verify display name generation for laptop webcams vs phone apps."""
    laptop_cam = CameraDeviceInfo(
        index=1,
        name="ASUS FHD webcam",
        manufacturer="Microsoft",
        device_id="USB\\VID_13D3&PID_52B8",
        is_laptop_webcam=True,
        is_virtual_phone=False,
        width=1280,
        height=720,
        backend="DSHOW",
    )
    assert "Laptop Built-in Webcam" in laptop_cam.display_name
    assert "Camera 1" in laptop_cam.display_name

    phone_cam = CameraDeviceInfo(
        index=0,
        name="Smart Connect Camera",
        manufacturer="Motorola",
        device_id="ROOT\\CAMERA\\0000",
        is_laptop_webcam=False,
        is_virtual_phone=True,
        width=1280,
        height=720,
        backend="DSHOW",
    )
    assert "Mobile Phone / Virtual" in phone_cam.display_name
    assert "Camera 0" in phone_cam.display_name


def test_get_recommended_camera_index_prefers_laptop_webcam():
    """Ensure recommender selects the built-in laptop webcam over phone cameras."""
    cams = [
        CameraDeviceInfo(
            index=0,
            name="Smart Connect Camera",
            manufacturer="Motorola",
            device_id="ROOT\\CAMERA\\0000",
            is_laptop_webcam=False,
            is_virtual_phone=True,
            width=1280,
            height=720,
            backend="DSHOW",
        ),
        CameraDeviceInfo(
            index=1,
            name="ASUS FHD webcam",
            manufacturer="Microsoft",
            device_id="USB\\VID_13D3&PID_52B8",
            is_laptop_webcam=True,
            is_virtual_phone=False,
            width=1280,
            height=720,
            backend="DSHOW",
        ),
    ]
    recommended = get_recommended_camera_index(cams)
    assert recommended == 1


def test_get_recommended_camera_index_fallback_empty():
    """Fallback to 0 when no cameras detected."""
    assert get_recommended_camera_index([]) == 0


def test_discover_cameras_with_mocked_cv2_and_pnp():
    """Test full discovery with mocked OpenCV VideoCapture and PnP subsystem."""
    mock_pnp = [
        {"Name": "Smart Connect Camera", "DeviceID": "ROOT\\CAMERA\\0000", "Manufacturer": "Motorola", "Present": True},
        {"Name": "ASUS FHD webcam", "DeviceID": "USB\\VID_13D3", "Manufacturer": "Microsoft", "Present": True},
    ]

    mock_cap0 = MagicMock()
    mock_cap0.isOpened.return_value = True
    mock_cap0.read.return_value = (True, MagicMock(size=100))
    mock_cap0.get.side_effect = lambda prop: 1280 if prop == 3 else 720

    mock_cap1 = MagicMock()
    mock_cap1.isOpened.return_value = True
    mock_cap1.read.return_value = (True, MagicMock(size=100))
    mock_cap1.get.side_effect = lambda prop: 1280 if prop == 3 else 720

    mock_cap_closed = MagicMock()
    mock_cap_closed.isOpened.return_value = False

    with patch("lecture_analyzer.vision.device_discovery._query_windows_pnp_cameras", return_value=mock_pnp):
        with patch("cv2.VideoCapture", side_effect=[mock_cap0, mock_cap1, mock_cap_closed, mock_cap_closed]):
            discovered = discover_cameras(max_indices=3)
            assert len(discovered) == 2
            assert discovered[0].index == 0
            assert discovered[0].is_virtual_phone is True
            assert discovered[1].index == 1
            assert discovered[1].is_laptop_webcam is True
            assert get_recommended_camera_index(discovered) == 1
