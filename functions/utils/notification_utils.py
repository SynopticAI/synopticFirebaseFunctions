# utils/notification_utils.py

from dataclasses import dataclass
from typing import List, Tuple, Optional, Any, Dict
import math
import asyncio
import time
from firebase_functions import logger
from firebase_admin import firestore, messaging

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

def extract_detections_from_inference(inference_result: dict, inference_mode: str) -> Tuple[List[DetectionPoint], List[DetectionBox]]:
    """Extract detection points and boxes from inference results."""
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

def is_point_in_region(point: DetectionPoint, region_data: dict) -> bool:
    """Check if a detection point is within any defined region."""
    try:
        # DEBUG: Log input data
        print(f"DEBUG: Checking point ({point.x}, {point.y}) against region")
        
        paths_data = region_data.get("paths", [])
        print(f"DEBUG: Region has {len(paths_data)} paths")
        
        if not paths_data:
            print("DEBUG: No paths in region data")
            return False
        
        # STEP 1: Infer canvas size from polygon coordinates
        all_x_coords = []
        all_y_coords = []
        
        for path_info in paths_data:
            points_data = path_info.get("points", [])
            for p in points_data:
                all_x_coords.append(p.get("x", 0))
                all_y_coords.append(p.get("y", 0))
        
        if not all_x_coords or not all_y_coords:
            print("DEBUG: No valid polygon points found")
            return False
        
        # Calculate the bounding box of all polygon points
        min_x, max_x = min(all_x_coords), max(all_x_coords)
        min_y, max_y = min(all_y_coords), max(all_y_coords)
        
        # Infer canvas dimensions with some padding
        canvas_width = max(max_x - min_x, max_x + abs(min_x)) + 20
        canvas_height = max(max_y - min_y, max_y + abs(min_y)) + 20
        
        print(f"DEBUG: Inferred canvas size: {canvas_width}x{canvas_height}")
        print(f"DEBUG: Polygon bounds: X({min_x}, {max_x}), Y({min_y}, {max_y})")
        
        # STEP 2: Convert normalized detection point to pixel coordinates
        pixel_x = point.x * canvas_width
        pixel_y = point.y * canvas_height
        
        print(f"DEBUG: Converted point to pixels: ({pixel_x}, {pixel_y})")
        
        # STEP 3: Check each polygon path
        for path_idx, path_info in enumerate(paths_data):
            operation = path_info.get("operation", "add")
            points_data = path_info.get("points", [])
            
            print(f"DEBUG: Path {path_idx}: operation={operation}, points={len(points_data)}")
            
            if not points_data:
                continue
                
            # Create polygon from points
            polygon_points = []
            for p in points_data:
                px, py = p.get("x", 0), p.get("y", 0)
                polygon_points.append((px, py))
                
            print(f"DEBUG: Polygon points: {polygon_points[:3]}...")  # Show first 3 points
            
            # Check if point is inside polygon using ray casting
            if _point_in_polygon(pixel_x, pixel_y, polygon_points):
                print(f"DEBUG: Point is inside polygon {path_idx}")
                return operation == "add"
                
        print("DEBUG: Point not in any polygon")
        return False
        
    except Exception as e:
        print(f"DEBUG: Error in is_point_in_region: {e}")
        return False

def _point_in_polygon(x: float, y: float, polygon: List[Tuple[float, float]]) -> bool:
    """Ray casting algorithm to determine if a point is inside a polygon."""
    n = len(polygon)
    if n < 3:
        return False
    
    inside = False
    p1x, p1y = polygon[0]
    
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

def check_count_threshold_trigger(class_name: str, detection_count: int, threshold: int) -> bool:
    """Check if count threshold trigger should fire."""
    return detection_count >= threshold

def check_location_trigger(class_name: str, detections: List[DetectionPoint], region_data: dict) -> Tuple[bool, List[Tuple[float, float]]]:
    """Check if location-based trigger should fire."""
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

def send_notification(trigger: NotificationTrigger) -> bool:
    """Send FCM notification only to devices that have enabled notifications for this camera."""
    try:
        db = firestore.client()
        
        # Get device data to check which FCM tokens are enabled
        device_doc = db.collection('users').document(trigger.user_id)\
                      .collection('devices').document(trigger.device_id).get()
        
        if not device_doc.exists:
            logger.error(f"Device {trigger.device_id} not found")
            return False
            
        device_data = device_doc.to_dict()
        enabled_fcm_tokens = device_data.get('enabledFCMTokens', [])
        
        # If no tokens are enabled for this device, skip notification
        if not enabled_fcm_tokens:
            logger.info(f"No FCM tokens enabled for device {trigger.device_id}, skipping notification")
            return False
        
        # Get device name for notification
        device_name = device_data.get('name', 'Unknown Device')
        
        # Build notification message
        title = f"Detection Alert: {trigger.class_name}"
        body = f"{trigger.trigger_reason} on {device_name}"
        
        # Create data payload
        data_payload = {
            'deviceId': trigger.device_id,
            'deviceName': device_name,
            'className': trigger.class_name,
            'triggerType': trigger.trigger_type,
            'triggerReason': trigger.trigger_reason,
            'timestamp': str(trigger.timestamp),
            'confidence': str(trigger.confidence),
            'click_action': 'FLUTTER_NOTIFICATION_CLICK'
        }
        
        # Send notifications to enabled tokens only
        successful_sends = 0
        failed_sends = 0
        
        for token in enabled_fcm_tokens:
            try:
                # Create FCM message
                message = messaging.Message(
                    notification=messaging.Notification(
                        title=title,
                        body=body,
                    ),
                    data=data_payload,
                    token=token,
                    android=messaging.AndroidConfig(
                        priority='high',
                        notification=messaging.AndroidNotification(
                            channel_id='ai_detection_channel',
                            priority='high',
                            default_sound=True,
                            notification_count=1
                        )
                    ),
                    apns=messaging.APNSConfig(
                        payload=messaging.APNSPayload(
                            aps=messaging.Aps(
                                alert=messaging.ApsAlert(
                                    title=title,
                                    body=body
                                ),
                                badge=1,
                                sound='default',
                                category='DETECTION_ALERT'
                            )
                        )
                    )
                )
                
                # Send the message
                response = messaging.send(message)
                logger.info(f"Successfully sent notification: {response}")
                successful_sends += 1
                
            except Exception as send_error:
                logger.error(f"Failed to send notification to token {token[:20]}...: {send_error}")
                failed_sends += 1
                
                # If token is invalid, remove it from device's enabled list
                if "not-registered" in str(send_error).lower() or "invalid" in str(send_error).lower():
                    try:
                        updated_tokens = [t for t in enabled_fcm_tokens if t != token]
                        device_doc.reference.update({
                            'enabledFCMTokens': updated_tokens
                        })
                        logger.info(f"Removed invalid token from device {trigger.device_id}")
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to remove invalid token: {cleanup_error}")
        
        logger.info(f"Notification complete for device {trigger.device_id}: {successful_sends} successful, {failed_sends} failed")
        return successful_sends > 0
        
    except Exception as e:
        logger.error(f"Error in send_notification: {e}")
        return False

def trigger_action(user_id: str, device_id: str, inference_result: dict, class_metrics: dict) -> List[NotificationTrigger]:
    """Enhanced trigger_action that checks notification settings and sends appropriate notifications."""
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
                    success = send_notification(trigger)
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