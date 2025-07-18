"""
Notification system data models
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class DetectionPoint:
    """Represents a detection point with coordinates and class info."""
    x: float  # Normalized coordinates (0-1)
    y: float  # Normalized coordinates (0-1)
    class_name: str
    confidence: float = 0.0

@dataclass
class DetectionBox:
    """Represents a detection bounding box."""
    x1: int  # Pixel coordinates (0-1024)
    y1: int
    x2: int
    y2: int
    class_name: str
    confidence: float = 0.0
    
    def get_center_point(self) -> DetectionPoint:
        """Get the center point of the bounding box as normalized coordinates."""
        center_x = (self.x1 + self.x2) / 2.0 / 1024.0  # Convert to normalized
        center_y = (self.y1 + self.y2) / 2.0 / 1024.0  # Convert to normalized
        return DetectionPoint(center_x, center_y, self.class_name, self.confidence)

@dataclass
class NotificationTrigger:
    """Represents a notification trigger event."""
    device_id: str
    user_id: str
    class_name: str
    trigger_type: str
    trigger_reason: str
    detection_count: int
    confidence: float
    timestamp: int
    detection_coordinates: List[Tuple[float, float]] = None  # For location-based triggers