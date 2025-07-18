"""
Notification system for Firebase Functions

This module provides notification functionality including:
- Trigger detection and processing
- FCM notification sending
- Location and threshold-based triggers
"""

from .models import DetectionPoint, DetectionBox, NotificationTrigger
from .triggers import (
    trigger_action,
    extract_detections_from_inference, 
    check_count_threshold_trigger,
    check_location_trigger
)
from .sender import send_notification
from .utils import is_point_in_region

__all__ = [
    # Models
    'DetectionPoint',
    'DetectionBox', 
    'NotificationTrigger',
    
    # Main functions
    'trigger_action',
    'send_notification',
    
    # Utility functions
    'extract_detections_from_inference',
    'check_count_threshold_trigger',
    'check_location_trigger',
    'is_point_in_region',
]