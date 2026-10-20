# Models Directory

This directory stores offline pretrained model weights used by the computer-vision pipeline:

1. `face_landmarker.task`: Google MediaPipe 478-point 3D FaceLandmarker model bundle (Apache-2.0 License).
2. `yolov8n-pose.pt`: Ultralytics YOLOv8n Pose estimation model (AGPL-3.0 License).
3. `yolov8n.pt`: Ultralytics YOLOv8n Object detection model (COCO trained, AGPL-3.0 License).

### Downloading Models
To download the verified models locally, run:
```powershell
python scripts/download_models.py
```

Models are never silently downloaded during application startup. If a model file is missing, the application clearly indicates limited operation and treats unavailable perceptual modalities as unobserved (`UNCERTAIN`).
