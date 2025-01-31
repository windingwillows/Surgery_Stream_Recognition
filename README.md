Surgical Tool Tracking with XMem and YOLO

This project implements a surgical tool tracking system using a combination of YOLO (for object detection) and XMem (for memory-based segmentation and tracking). The system can track surgical tools across video frames, leveraging memory mechanisms for improved long-term consistency.
Key Features:

    YOLO-based Object Detection: Detects surgical tools in each frame.
    XMem-based Memory Tracking: Tracks objects across frames using both short-term and long-term memory.
    Segmentation & Tracking: Outputs segmentation masks that track tools in the video stream.
    Video Processing: Processes video streams or files and outputs the tracked frames with overlayed segmentation masks.

Dependencies:

    Python 3.x
    PyTorch (with CUDA support if GPU is available)
    OpenCV
    Ultralytics YOLO
    TorchVision

Installation

    Clone this repository:
    git clone https://github.com/your-username/surgical-tool-tracker.git
cd surgical-tool-tracker
