import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ultralytics import YOLO
from typing import Tuple, Optional, List, Dict
import cv2
from dataclasses import dataclass
import torchvision.models as models

@dataclass
class MemoryElement:
    """Structure for storing memory elements"""
    frame_features: torch.Tensor
    mask: torch.Tensor
    timestamp: int

class FeatureExtractor(nn.Module):
    """ResNet-based feature extractor for XMem"""
    def __init__(self, hidden_dim: int):
        super().__init__()
        # Use ResNet50 as backbone as mentioned in paper
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Additional layers for feature processing
        self.conv_adapt = nn.Conv2d(2048, hidden_dim, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.conv_adapt(x)
        x = self.norm(x)
        return F.relu(x)

class MemoryReader(nn.Module):
    """Memory reading module for XMem"""
    def __init__(self, key_dim: int, value_dim: int):
        super().__init__()
        self.key_dim = key_dim
        self.value_dim = value_dim
        
        # Key and value projectors
        self.key_projector = nn.Conv2d(value_dim, key_dim, kernel_size=1)
        self.value_projector = nn.Conv2d(value_dim, value_dim, kernel_size=1)
        
    def forward(self, 
                query: torch.Tensor,
                keys: torch.Tensor,
                values: torch.Tensor) -> torch.Tensor:
        # Project query and memory
        query_keys = self.key_projector(query)
        memory_keys = self.key_projector(keys)
        memory_values = self.value_projector(values)
        
        # Compute attention
        B, C, H, W = query_keys.shape
        query_keys = query_keys.view(B, C, -1).permute(0, 2, 1)
        memory_keys = memory_keys.view(B, C, -1)
        memory_values = memory_values.view(B, self.value_dim, -1)
        
        attention = torch.bmm(query_keys, memory_keys)
        attention = F.softmax(attention / np.sqrt(C), dim=2)
        
        # Read from memory
        read_values = torch.bmm(memory_values, attention.permute(0, 2, 1))
        return read_values.view(B, self.value_dim, H, W)

class XMem(nn.Module):
    """XMem implementation based on Atkinson-Shiffrin memory model"""
    def __init__(
        self,
        key_dim: int = 64,
        value_dim: int = 512,
        hidden_dim: int = 64,
        update_k: int = 5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.device = device
        self.update_k = update_k
        
        # Components
        self.feature_extractor = FeatureExtractor(hidden_dim).to(device)
        self.memory_reader = MemoryReader(key_dim, value_dim).to(device)
        
        # Decoder for final segmentation
        self.decoder = nn.Sequential(
            nn.Conv2d(value_dim + hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, 1, 1)
        ).to(device)
        
        # Memory stores
        self.working_memory: List[MemoryElement] = []
        self.long_term_memory: List[MemoryElement] = []
        
    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess frame for network input"""
        frame = cv2.resize(frame, (384, 384))  # Size from paper
        frame = torch.from_numpy(frame).float().permute(2, 0, 1)
        frame = frame.unsqueeze(0) / 255.0
        return frame.to(self.device)
    
    def _preprocess_mask(self, mask: np.ndarray) -> torch.Tensor:
        """Preprocess mask for network input"""
        mask = cv2.resize(mask, (384, 384))
        mask = torch.from_numpy(mask).float()
        mask = mask.unsqueeze(0).unsqueeze(0)
        return mask.to(self.device)
    
    def initialize(self, frame: np.ndarray, mask: np.ndarray):
        """Initialize tracking with first frame and mask"""
        frame_tensor = self._preprocess_frame(frame)
        mask_tensor = self._preprocess_mask(mask)
        
        # Extract features
        features = self.feature_extractor(frame_tensor)
        
        # Initialize working memory
        self.working_memory = [
            MemoryElement(features, mask_tensor, 0)
        ]
        
    def _update_memory(self, new_element: MemoryElement):
        """Update memory stores"""
        self.working_memory.append(new_element)
        
        # Transfer to long-term memory if working memory is full
        if len(self.working_memory) > self.update_k:
            old_element = self.working_memory.pop(0)
            self.long_term_memory.append(old_element)
            
            # Limit long-term memory size
            if len(self.long_term_memory) > 50:  # Arbitrary limit
                self.long_term_memory = self.long_term_memory[-50:]
                
    def _get_memory_features(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get concatenated features from both memory stores"""
        all_memories = self.working_memory + self.long_term_memory
        
        if not all_memories:
            return None, None
            
        features = torch.cat([m.frame_features for m in all_memories], dim=0)
        masks = torch.cat([m.mask for m in all_memories], dim=0)
        return features, masks
        
    def track(self, frame: np.ndarray) -> np.ndarray:
        """Track objects in new frame"""
        frame_tensor = self._preprocess_frame(frame)
        
        # Extract features
        current_features = self.feature_extractor(frame_tensor)
        
        # Get memory features
        memory_features, memory_masks = self._get_memory_features()
        
        if memory_features is None:
            return np.zeros(frame.shape[:2], dtype=np.uint8)
            
        # Read from memory
        memory_readout = self.memory_reader(
            current_features,
            memory_features,
            memory_features * memory_masks
        )
        
        # Decode final segmentation
        features_cat = torch.cat([current_features, memory_readout], dim=1)
        logits = self.decoder(features_cat)
        mask_pred = torch.sigmoid(logits) > 0.5
        
        # Update memory
        self._update_memory(MemoryElement(
            current_features,
            mask_pred,
            len(self.working_memory) + len(self.long_term_memory)
        ))
        
        # Return resized mask
        mask_np = mask_pred.squeeze().cpu().numpy().astype(np.uint8)
        return cv2.resize(mask_np, (frame.shape[1], frame.shape[0]))

class SurgicalToolTracker:
    def __init__(
        self,
        yolo_model_path: str = "yolov8x-seg.pt",
        min_area_threshold: float = 0.03,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize the surgical tool tracking pipeline"""
        self.device = device
        self.min_area_threshold = min_area_threshold
        
        # Initialize models
        self.yolo = YOLO(yolo_model_path)
        self.xmem = XMem(
            key_dim=64,
            value_dim=512,
            hidden_dim=64,
            update_k=5,
            device=device
        )
        
        # Tracking state
        self.tracking_initialized = False
        self.frame_count = 0
        
    def _compute_mask_area_ratio(self, mask: np.ndarray) -> float:
        """Compute ratio of mask area to frame area"""
        total_pixels = mask.shape[0] * mask.shape[1]
        mask_pixels = np.sum(mask > 0)
        return mask_pixels / total_pixels
        
    def _get_yolo_prediction(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Get YOLO prediction if it meets area threshold"""
        results = self.yolo.predict(frame, stream=True)
        for result in results:
            if result.masks is not None:
                mask = result.masks[0].cpu().numpy()
                if self._compute_mask_area_ratio(mask) >= self.min_area_threshold:
                    return mask
        return None
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Process a single frame from the video stream"""
        if not self.tracking_initialized:
            # Try to get initial mask from YOLO
            mask = self._get_yolo_prediction(frame)
            
            if mask is not None:
                # Initialize XMem tracking
                self.xmem.initialize(frame, mask)
                self.tracking_initialized = True
                return mask, True
            return np.zeros(frame.shape[:2], dtype=np.uint8), False
            
        else:
            # Use XMem for tracking
            self.frame_count += 1
            return self.xmem.track(frame), True
            
    def process_video_stream(self, video_path: str, output_path: Optional[str] = None):
        """Process a video file or stream"""
        cap = cv2.VideoCapture(video_path)
        
        if output_path:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            writer = cv2.VideoWriter(
                output_path, 
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps, 
                (width, height)
            )
            
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            mask, is_tracking = self.process_frame(frame)
            
            if output_path:
                # Overlay mask on frame
                overlay = frame.copy()
                overlay[mask > 0] = [0, 255, 0]  # Green overlay
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                writer.write(frame)
                
        cap.release()
        if output_path:
            writer.release()

def main():
    # Example usage
    tracker = SurgicalToolTracker(
        yolo_model_path="path/to/yolov8x-seg.pt"
    )
    
    # Process video stream
    tracker.process_video_stream(
        video_path="surgery_video.mp4",
        output_path="output.mp4"
    )

if __name__ == "__main__":
    main()
