"""
Notification trigger logic and detection extraction
"""

import asyncio
import time
from typing import List, Tuple, Dict, Any
from firebase_functions import logger
from firebase_admin import firestore

from .models import DetectionPoint, DetectionBox, NotificationTrigger
from .utils import is_point_in_region
from .sender import send_notification

def extract_detections_from_inference(inference_result: dict, inference_mode: str) -> Tuple[List[DetectionPoint], List[DetectionBox]]:
    """
    Extract detection points and boxes from inference results.
    
    Returns:
        Tuple of (points, boxes) - one or both lists may be empty depending on inference mode
    """
    points = []
    boxes = []
    
    try:
        if inference_mode == "Point":
            point_data = inference_result.get("points", [])
            for point in point_data:
                points.append(DetectionPoint(
                    x=float(point.get("x", 0.5)),
                    y=float(point.get("y", 0.5)),
                    class_name=point.get("class", "unknown"),
                    confidence=float(point.get("score", 0.0))
                ))
                
        elif inference_mode == "Detect":
            object_data = inference_result.get("objects", [])
            for obj in object_data:
                bbox = obj.get("bbox", {})
                box = DetectionBox(
                    x1=int(bbox.get("x1", 0)),
                    y1=int(bbox.get("y1", 0)),
                    x2=int(bbox.get("x2", 1024)),
                    y2=int(bbox.get("y2", 1024)),
                    class_name=obj.get("class", "unknown"),
                    confidence=float(obj.get("score", 0.0))
                )
                boxes.append(box)
                # Also create center points for location checking
                points.append(box.get_center_point())
                
    except Exception as e:
        logger.error(f"Error extracting detections: {str(e)}")
    
    return points, boxes

def check_count_threshold_trigger(class_name: str, detection_count: int, threshold: int) -> bool:
    """
    Check if count threshold trigger should fire.
    
    Args:
        class_name: Name of the detected class
        detection_count: Number of detections for this class
        threshold: Threshold value from settings
        
    Returns:
        True if trigger should fire
    """
    return detection_count >= threshold

def check_location_trigger(class_name: str, detections: List[DetectionPoint], region_data: dict) -> Tuple[bool, List[Tuple[float, float]]]:
    """
    Check if location-based trigger should fire.
    
    Args:
        class_name: Name of the detected class
        detections: List of detection points for this class
        region_data: Region data from notification settings
        
    Returns:
        Tuple of (should_trigger, coordinates_in_region)
    """
    coordinates_in_region = []

    # ADD DEBUGGING
    print(f"DEBUG: Checking {len(detections)} detections for class {class_name}")
    for i, detection in enumerate(detections):
        print(f"DEBUG: Detection {i}: x={detection.x}, y={detection.y}")
    
    print(f"DEBUG: Region data: {region_data}")
    
    try:
        for detection in detections:
            if detection.class_name == class_name and is_point_in_region(detection, region_data):
                coordinates_in_region.append((detection.x, detection.y))
        
        # Trigger if any detections are in the region
        should_trigger = len(coordinates_in_region) > 0
        return should_trigger, coordinates_in_region
        
    except Exception as e:
        logger.error(f"Error checking location trigger for {class_name}: {str(e)}")
        return False, []

def trigger_action(user_id: str, device_id: str, inference_result: dict, class_metrics: dict) -> List[NotificationTrigger]:
    """
    Enhanced trigger_action that checks notification settings and sends appropriate notifications.
    
    Args:
        user_id: User ID
        device_id: Device ID  
        inference_result: Raw inference result from API
        class_metrics: Processed class metrics
        
    Returns:
        List of NotificationTrigger objects that were processed
    """
    triggered_notifications = []
    
    try:
        db = firestore.client()
        current_time = int(time.time() * 1000)
        
        # Get device data and notification settings
        device_ref = db.collection("users").document(user_id).collection("devices").document(device_id)
        device_doc = device_ref.get()
        
        if not device_doc.exists:
            logger.error(f"Device {device_id} not found for notification check")
            return triggered_notifications
        
        device_data = device_doc.to_dict()
        notification_settings = device_data.get("notificationSettings", {})
        inference_mode = device_data.get("inferenceMode", "Detect")
        
        # If no notification settings, nothing to do
        if not notification_settings:
            logger.info(f"No notification settings configured for device {device_id}")
            return triggered_notifications
        
        # Extract detection data
        detection_points, detection_boxes = extract_detections_from_inference(inference_result, inference_mode)
        
        # Get class counts from metrics
        class_counts = class_metrics.get("classCounts", {})
        average_confidence = class_metrics.get("averageConfidence", 0.0)
        
        # Process each class that has notification settings
        for class_name, settings in notification_settings.items():
            try:
                trigger_type = settings.get("triggerType", "none")
                
                # Skip if notifications are disabled
                if trigger_type == "none":
                    continue
                
                # Get count for this class
                detection_count = class_counts.get(class_name, 0)
                
                # Get detections for this class
                class_detections = [d for d in detection_points if d.class_name == class_name]
                class_confidence = sum(d.confidence for d in class_detections) / len(class_detections) if class_detections else 0.0
                
                should_trigger = False
                trigger_reason = ""
                coordinates = None
                
                # Check trigger conditions
                if trigger_type == "count":
                    threshold = settings.get("threshold", 1)
                    should_trigger = check_count_threshold_trigger(class_name, detection_count, threshold)
                    if should_trigger:
                        trigger_reason = f"{detection_count} {class_name} detections (threshold: {threshold})"
                
                elif trigger_type == "location":
                    region_data = settings.get("regionData")
                    if region_data:
                        should_trigger, coordinates = check_location_trigger(class_name, class_detections, region_data)
                        if should_trigger:
                            trigger_reason = f"{len(coordinates)} {class_name} detection(s) in trigger region"
                    else:
                        logger.error(f"Location trigger set for {class_name} but no region data found")
                
                # Send notification if trigger conditions are met
                if should_trigger:
                    trigger = NotificationTrigger(
                        device_id=device_id,
                        user_id=user_id,
                        class_name=class_name,
                        trigger_type=trigger_type,
                        trigger_reason=trigger_reason,
                        detection_count=detection_count,
                        confidence=class_confidence,
                        timestamp=current_time,
                        detection_coordinates=coordinates
                    )
                    
                    # Send notification
                    success = asyncio.run(send_notification(trigger))
                    if success:
                        triggered_notifications.append(trigger)
                        
                        # Log successful trigger to Firestore for history/debugging
                        trigger_log_ref = device_ref.collection("notification_triggers").document(str(current_time))
                        trigger_log_ref.set({
                            "className": class_name,
                            "triggerType": trigger_type,
                            "triggerReason": trigger_reason,
                            "detectionCount": detection_count,
                            "confidence": class_confidence,
                            "timestamp": current_time,
                            # "coordinates": coordinates,
                            "notificationSent": True
                        })
                    else:
                        logger.error(f"Failed to send notification for {class_name} on device {device_id}")
                
            except Exception as e:
                logger.error(f"Error processing notification for class {class_name}: {str(e)}")
                continue
        
        logger.info(f"Processed {len(triggered_notifications)} notification triggers for device {device_id}")
        
    except Exception as e:
        logger.error(f"Error in trigger_action: {str(e)}")
    
    return triggered_notifications