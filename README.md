# DoorknobDetector
DoorknobDetector is a small computer-vision demo that detects door handles in images and webcam video using a pre-trained TensorFlow object-detection model loaded through OpenCV DNN.

## What the project does

- Loads a frozen inference graph (`frozen_inference_graph.pb`) and graph config (`graph.pbtxt`).
- Runs object detection on:
  - a batch of test images from `Input/`
  - live frames from a webcam
- Draws bounding boxes around detected door handles and saves processed images to `Output/` (batch mode).

## How it works

1. The model is loaded with `cv2.dnn.readNetFromTensorflow(...)`.
2. Each frame/image is converted into a blob (`300x300`) with `cv2.dnn.blobFromImage(...)`.
3. The network outputs detections in `[class, score, box]` format.

## Repository structure

- `Image_test.py` — runs detection on numbered images in `Input/` and writes results to `Output/`.
- `real_teme_test-с.py` — runs real-time detection from webcam (`Esc` to exit).
- `frozen_inference_graph.pb`, `graph.pbtxt` — model files.
- `Input/`, `Output/` — sample input and generated output images.

## Requirements

- Python 3.8+
- OpenCV with DNN support (`opencv-python`)


## Usage

### 1) Batch detection on images

```bash
python Image_test.py
```

Expected behavior:
- reads images like `Input/0.jpg`, `Input/1.jpg`, ...
- saves processed images to `Output/`

### 2) Real-time webcam detection

```bash
python real_teme_test-с.py
```

Expected behavior:
- opens webcam stream
- draws detection boxes in real time
- press `Esc` to close

## Notes

- The scripts assume model files are in the project root.
