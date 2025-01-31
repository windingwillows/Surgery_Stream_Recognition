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
    
Install the required dependencies:
    
    pip install -r requirements.txt

If you don’t have a requirements.txt file, you can manually install the required packages:

    pip install torch torchvision opencv-python ultralytics

Download the YOLOv8 model (for segmentation) from Ultralytics YOLO or use an existing pre-trained model. Put it in a directory and provide the path to the SurgicalToolTracker class.

 Make sure you have a CUDA-enabled GPU for faster processing, or modify the device setting to "cpu" if not using a GPU.

Usage
======

1. Track Surgical Tools in Video:

You can use the SurgicalToolTracker class to track surgical tools in a video file. The tracker will detect and segment surgical tools across frames and overlay the tracked masks on the video.
Example Usage:

    from surgical_tool_tracker import SurgicalToolTracker

    def main():
        # Initialize tracker with YOLO model path
        tracker = SurgicalToolTracker(
        yolo_model_path="path/to/yolov8x-seg.pt"  # Provide path to YOLOv8 model
        )

        # Process video stream
        tracker.process_video_stream(
            video_path="path/to/surgery_video.mp4",   # Input video
            output_path="output_video.mp4"            # Output video
        )

    if __name__ == "__main__":
        main()
. Video Processing Explanation:

SurgicalToolTracker uses YOLO for initial object detection to create a segmentation mask for surgical tools. The XMem model then tracks these objects by recalling features from previous frames (using both short-term and long-term memory). The resulting segmentation mask is applied to the video with an overlay that highlights the tracked surgical tools.

3. YOLO Model Path:

Make sure to specify the correct path to a YOLOv8 model trained for segmentation. You can download a model trained on your dataset or use a pre-trained one from Ultralytics YOLOv8.
4. Output:

The processed video will have an overlay of the tracked objects, which are highlighted in green. The output video will be saved in the specified output_path.

Code Overview

1. Memory Management (XMem):

 XMem is the core memory model that tracks surgical tools using memory-based segmentation. Features from the current and past frames are stored in both working memory and long-term memory, and the model uses these stored features to track objects across frames. The MemoryReader module reads features from both memories and computes attention to improve tracking accuracy.

2. Feature Extraction:

    ResNet50 is used as the backbone for feature extraction from each video frame.
    The features are then processed with additional layers to adapt them for memory tracking.

3. Segmentation Decoder:

    A decoder is used to generate segmentation masks based on memory features and current frame features.
    The segmentation masks are post-processed and applied to the video frames for visualization.

4. Surgical Tool Tracker:

    The SurgicalToolTracker class integrates YOLO for initial detection and XMem for object tracking.
    The tracker is responsible for processing video frames, managing memory, and outputting the tracked video with segmentation masks.

Example Video

Here is an example of the output video that shows the surgical tool being tracked across frames:

[Insert Example Video Here]
Contribution

Feel free to contribute to this repository! You can submit issues or create pull requests for improvements. Make sure to follow the code style and include proper testing if possible.
License

This project is licensed under the MIT License - see the LICENSE file for details.

Troubleshooting
======

Issue with YOLO model: If YOLO isn’t detecting objects correctly, ensure you’re using a model trained for segmentation and that the model path is correct. Memory issues: If the model runs out of memory, consider reducing the batch size or simplifying the network architecture.
