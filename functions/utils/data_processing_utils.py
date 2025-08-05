# utils/data_processing_utils.py

import re
import base64
import time
import math
import datetime
from typing import Dict, List, Tuple, Optional, Any
from firebase_functions import logger
from firebase_admin import firestore, storage
from flask import Request, jsonify

# Import trigger_action at module level to avoid circular import issues
from .notification_utils import trigger_action

def add_cors_headers(response):
    """Add CORS headers to the response.
    
    Args:
        response: Either a Flask response object or a dictionary
        
    Returns:
        A Flask response object with CORS headers
    """
    # If response is a dict, convert it to a Flask response
    if isinstance(response, dict):
        response = jsonify(response)
    
    # Now add headers to the Flask response
    response.headers.set('Access-Control-Allow-Origin', '*')
    response.headers.set('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    response.headers.set('Access-Control-Allow-Headers', 'Content-Type')
    return response

def create_model_output(user_id: str, device_id: str, output_value: float, resolution: float, max_value: float) -> None:
    """
    Stores inference output in Firestore with multiple time resolutions for visualization.
    
    Args:
        user_id: The ID of the user.
        device_id: The ID of the device.
        output_value: The positive inference output value.
        resolution: The bin size for discretization (e.g., 0.5).
        max_value: The maximum value for binning.
    """
    try:
        db = firestore.client()
        # Validate inputs
        if not isinstance(output_value, (int, float)) or output_value < 0:
            logger.error(f"Invalid output_value: {output_value}")
            return
        if resolution <= 0 or max_value <= 0:
            logger.error(f"Invalid resolution {resolution} or max_value {max_value}")
            return

        # Calculate bin index
        number_of_bins = math.ceil(max_value / resolution)
        bin_index = min(math.floor(output_value / resolution), number_of_bins - 1)

        # Current timestamp in milliseconds
        current_time = int(time.time() * 1000)

        # Device reference
        device_ref = db.collection("users").document(user_id).collection("devices").document(device_id)

        # 1. Store in recent_outputs
        recent_outputs_ref = device_ref.collection("recent_outputs")
        recent_outputs_ref.document(str(current_time)).set({
            "timestamp": current_time,
            "outputValue": output_value
        })

        # 2. Calculate current hour and day timestamps
        current_datetime = datetime.datetime.fromtimestamp(current_time / 1000, tz=datetime.timezone.utc)
        hour_start = current_datetime.replace(minute=0, second=0, microsecond=0)
        day_start = current_datetime.replace(hour=0, minute=0, second=0, microsecond=0)
        hour_timestamp = int(hour_start.timestamp() * 1000)
        day_timestamp = int(day_start.timestamp() * 1000)

        # 3. Update hourly aggregation
        hourly_ref = device_ref.collection("hourly_aggregations").document(str(hour_timestamp))
        hourly_doc = hourly_ref.get()

        # Initialize or update counts
        if hourly_doc.exists:
            hourly_data = hourly_doc.to_dict()
            counts = hourly_data.get("counts", {})
            total_count = hourly_data.get("totalCount", 0)
        else:
            counts = {}
            total_count = 0

        # Increment the count for this bin
        counts[str(bin_index)] = counts.get(str(bin_index), 0) + 1
        total_count += 1

        # Write back the updated hourly aggregation
        hourly_ref.set({
            "startTimestamp": hour_timestamp,
            "counts": counts,
            "totalCount": total_count
        }, merge=True)

        # 4. Update daily aggregation
        daily_ref = device_ref.collection("daily_aggregations").document(str(day_timestamp))
        daily_doc = daily_ref.get()

        # Initialize or update counts
        if daily_doc.exists:
            daily_data = daily_doc.to_dict()
            counts = daily_data.get("counts", {})
            total_count = daily_data.get("totalCount", 0)
        else:
            counts = {}
            total_count = 0

        # Increment the count for this bin
        counts[str(bin_index)] = counts.get(str(bin_index), 0) + 1
        total_count += 1

        # Write back the updated daily aggregation
        daily_ref.set({
            "startTimestamp": day_timestamp,
            "counts": counts,
            "totalCount": total_count
        }, merge=True)

        # 5. Clean up old recent_outputs (older than 1 hour)
        one_hour_ago = current_time - 3600000  # 1 hour in milliseconds
        old_outputs = recent_outputs_ref.where("timestamp", "<", one_hour_ago).stream()
        batch = db.batch()
        for output in old_outputs:
            batch.delete(output.reference)
        batch.commit()

        logger.info(f"Stored output {output_value} for user {user_id}, device {device_id}")

    except Exception as e:
        logger.error(f"Error in create_model_output: {str(e)}")
        raise

def update_inference_aggregations(device_ref, current_time: int, primary_metric: float, class_metrics: dict):
    """
    Update hourly and daily aggregations for dashboard time-series visualization.
    Uses the existing aggregation structure for backward compatibility.
    """
    try:
        # Calculate current hour and day timestamps
        current_datetime = datetime.datetime.fromtimestamp(current_time / 1000, tz=datetime.timezone.utc)
        hour_start = current_datetime.replace(minute=0, second=0, microsecond=0)
        day_start = current_datetime.replace(hour=0, minute=0, second=0, microsecond=0)
        hour_timestamp = int(hour_start.timestamp() * 1000)
        day_timestamp = int(day_start.timestamp() * 1000)
        
        # Update hourly aggregation
        hourly_ref = device_ref.collection("hourly_aggregations").document(str(hour_timestamp))
        hourly_doc = hourly_ref.get()
        
        if hourly_doc.exists:
            hourly_data = hourly_doc.to_dict()
            total_value = hourly_data.get("totalValue", 0)
            total_count = hourly_data.get("totalCount", 0)
            class_totals = hourly_data.get("classTotals", {})
        else:
            total_value = 0
            total_count = 0
            class_totals = {}
        
        # Update totals
        total_value += primary_metric
        total_count += 1
        
        # Update class totals
        if "classCounts" in class_metrics:
            for class_name, count in class_metrics["classCounts"].items():
                class_totals[class_name] = class_totals.get(class_name, 0) + count
        
        # Write back hourly aggregation
        hourly_ref.set({
            "startTimestamp": hour_timestamp,
            "totalValue": total_value,
            "totalCount": total_count,
            "averageValue": total_value / total_count,
            "classTotals": class_totals
        }, merge=True)
        
        # Update daily aggregation (similar structure)
        daily_ref = device_ref.collection("daily_aggregations").document(str(day_timestamp))
        daily_doc = daily_ref.get()
        
        if daily_doc.exists:
            daily_data = daily_doc.to_dict()
            total_value = daily_data.get("totalValue", 0)
            total_count = daily_data.get("totalCount", 0)
            class_totals = daily_data.get("classTotals", {})
        else:
            total_value = 0
            total_count = 0
            class_totals = {}
        
        total_value += primary_metric
        total_count += 1
        
        if "classCounts" in class_metrics:
            for class_name, count in class_metrics["classCounts"].items():
                class_totals[class_name] = class_totals.get(class_name, 0) + count
        
        daily_ref.set({
            "startTimestamp": day_timestamp,
            "totalValue": total_value,
            "totalCount": total_count,
            "averageValue": total_value / total_count,
            "classTotals": class_totals
        }, merge=True)
        
    except Exception as e:
        logger.error(f"Error updating aggregations: {str(e)}")

def cleanup_old_inference_data(device_ref, current_time: int):
    """Clean up old inference data to manage storage costs."""
    try:
        # Clean up raw inference results older than 7 days
        seven_days_ago = current_time - (7 * 24 * 60 * 60 * 1000)
        old_results = device_ref.collection("inference_results").where("timestamp", "<", seven_days_ago).limit(50).stream()
        
        batch = firestore.client().batch()
        count = 0
        for result in old_results:
            batch.delete(result.reference)
            count += 1
        
        if count > 0:
            batch.commit()
            logger.info(f"Cleaned up {count} old inference results")
        
        # Clean up recent_outputs older than 1 hour (keeping existing logic)
        one_hour_ago = current_time - 3600000
        old_outputs = device_ref.collection("recent_outputs").where("timestamp", "<", one_hour_ago).limit(50).stream()
        
        batch = firestore.client().batch()
        count = 0
        for output in old_outputs:
            batch.delete(output.reference)
            count += 1
        
        if count > 0:
            batch.commit()
            logger.info(f"Cleaned up {count} old recent outputs")
            
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")

def extract_metrics_from_inference(inference_result: dict, inference_mode: str) -> tuple[float, dict]:
    """
    Extract numerical metrics from inference results for time-series graphing.
    Enhanced version that provides detailed class information for notifications.
    
    Returns:
        tuple: (primary_metric, class_metrics)
            - primary_metric: Single numerical value for main time-series graph
            - class_metrics: Dict with detailed per-class information for notifications
    """
    class_metrics = {}
    
    if inference_mode == "Point":
        points = inference_result.get("points", [])
        primary_metric = len(points)  # Total points detected
        
        # Count points per class with detailed info
        class_counts = {}
        class_confidences = {}
        total_confidence = 0
        
        for point in points:
            class_name = point.get("class", "unknown")
            confidence = point.get("score", 0.0)
            
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            if class_name not in class_confidences:
                class_confidences[class_name] = []
            class_confidences[class_name].append(confidence)
            total_confidence += confidence
        
        class_metrics = {
            "classCounts": class_counts,
            "classConfidences": class_confidences,
            "averageConfidence": total_confidence / len(points) if points else 0.0,
            "totalDetections": len(points),
            "detectionType": "points"
        }
        
    elif inference_mode == "Detect":
        objects = inference_result.get("objects", [])
        primary_metric = len(objects)  # Total objects detected
        
        # Count objects per class with detailed info
        class_counts = {}
        class_confidences = {}
        total_confidence = 0
        
        for obj in objects:
            class_name = obj.get("class", "unknown")
            confidence = obj.get("score", 0.0)
            
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            if class_name not in class_confidences:
                class_confidences[class_name] = []
            class_confidences[class_name].append(confidence)
            total_confidence += confidence
        
        class_metrics = {
            "classCounts": class_counts,
            "classConfidences": class_confidences,
            "averageConfidence": total_confidence / len(objects) if objects else 0.0,
            "totalDetections": len(objects),
            "detectionType": "objects"
        }
        
    elif inference_mode == "VQA":
        # For VQA, analyze the answer
        result_data = inference_result.get("result", {})
        if isinstance(result_data, dict):
            answer = result_data.get("answer", "")
        else:
            answer = str(result_data)
            
        primary_metric = len(answer.split()) if answer else 0  # Word count
        
        class_metrics = {
            "answerLength": len(answer),
            "wordCount": primary_metric,
            "answer": answer[:200],  # Truncated for storage efficiency
            "detectionType": "text"
        }
        
    elif inference_mode == "Caption":
        # For Caption, analyze the caption
        result_data = inference_result.get("result", {})
        if isinstance(result_data, dict):
            caption = result_data.get("caption", "")
        else:
            caption = str(result_data)
            
        primary_metric = len(caption.split()) if caption else 0  # Word count
        
        class_metrics = {
            "captionLength": len(caption),
            "wordCount": primary_metric,
            "caption": caption[:200],  # Truncated for storage efficiency
            "detectionType": "text"
        }
        
    else:
        # Default case
        primary_metric = 1.0  # Just indicate that inference occurred
        class_metrics = {
            "inferenceCompleted": True,
            "detectionType": "unknown"
        }
    
    return primary_metric, class_metrics

def update_credit_usage(user_id: str, device_id: str, credit_usage: float) -> None:
    """
    Update credit usage for both user and device in Firestore.
    
    Args:
        user_id: The ID of the user
        device_id: The ID of the device
        credit_usage: The amount of credits used for this inference
    """
    try:
        db = firestore.client()
        
        # Update user's total credit usage
        user_ref = db.collection("users").document(user_id)
        user_doc = user_ref.get()
        
        if user_doc.exists:
            current_user_credits = user_doc.to_dict().get("totalCreditsUsed", 0.0)
            new_user_credits = current_user_credits + credit_usage
            
            # Use set with merge=True to ensure field creation
            user_ref.set({
                "totalCreditsUsed": new_user_credits
            }, merge=True)
            
            logger.info(f"Updated user {user_id} credit usage: {current_user_credits} -> {new_user_credits}")
        else:
            # Create user document if it doesn't exist
            user_ref.set({
                "totalCreditsUsed": credit_usage
            }, merge=True)
            logger.info(f"Created user {user_id} with initial credit usage: {credit_usage}")
        
        # Update device's credit usage
        device_ref = db.collection("users").document(user_id).collection("devices").document(device_id)
        device_doc = device_ref.get()
        
        if device_doc.exists:
            current_device_credits = device_doc.to_dict().get("totalCreditsUsed", 0.0)
            new_device_credits = current_device_credits + credit_usage
            
            # Use set with merge=True to ensure field creation
            device_ref.set({
                "totalCreditsUsed": new_device_credits
            }, merge=True)
            
            logger.info(f"Updated device {device_id} credit usage: {current_device_credits} -> {new_device_credits}")
        else:
            logger.info(f"Device {device_id} not found for credit update")
        
        # Force a small delay and verify the update worked
        import time
        time.sleep(0.1)
        
        # Verify user update
        updated_user_doc = user_ref.get()
        if updated_user_doc.exists:
            verified_credits = updated_user_doc.to_dict().get("totalCreditsUsed", 0.0)
            logger.info(f"✅ Verified user {user_id} credits: {verified_credits}")
        else:
            logger.error(f"❌ User document verification failed for {user_id}")
        
        # Verify device update  
        updated_device_doc = device_ref.get()
        if updated_device_doc.exists:
            verified_device_credits = updated_device_doc.to_dict().get("totalCreditsUsed", 0.0)
            logger.info(f"✅ Verified device {device_id} credits: {verified_device_credits}")
        else:
            logger.error(f"❌ Device document verification failed for {device_id}")
        
    except Exception as e:
        logger.error(f"❌ Error updating credit usage for user {user_id}, device {device_id}: {str(e)}")
        # Don't raise the exception to avoid breaking the inference flow
        # Credit tracking failures shouldn't stop the inference from working

def save_inference_output(user_id: str, device_id: str, inference_result: dict, inference_mode: str) -> None:
    """
    Save inference output in a format suitable for dashboard visualization and notifications.
    Enhanced version that properly triggers notifications and tracks credit usage.
    """
    try:
        
        db = firestore.client()
        current_time = int(time.time() * 1000)
        
        # Device reference
        device_ref = db.collection("users").document(user_id).collection("devices").document(device_id)
        
        # Extract and handle credit usage if present
        credit_usage = inference_result.get("credit_usage", 0)
        if credit_usage > 0:
            logger.info(f"Processing credit usage: {credit_usage} for user {user_id}, device {device_id}")
            update_credit_usage(user_id, device_id, credit_usage)
        
        # Create a clean result without credit_usage for storage
        clean_result = {k: v for k, v in inference_result.items() if k != "credit_usage"}
        
        # 1. Store raw inference result for detailed analysis
        raw_result_ref = device_ref.collection("inference_results").document(str(current_time))
        raw_result_data = {
            "timestamp": current_time,
            "inferenceMode": inference_mode,
            "result": clean_result,
            "modelUsed": inference_result.get("modelUsed", "unknown"),
            "creditUsage": credit_usage,  # Store credit usage in the detailed results
            "created": firestore.SERVER_TIMESTAMP
        }
        raw_result_ref.set(raw_result_data)
        
        # 2. Extract numerical metrics based on inference mode
        primary_metric, class_metrics = extract_metrics_from_inference(clean_result, inference_mode)
        
        # 3. Store metrics in the format expected by dashboard
        metrics_ref = device_ref.collection("recent_outputs").document(str(current_time))
        metrics_data = {
            "timestamp": current_time,
            "outputValue": primary_metric,
            "inferenceMode": inference_mode,
            "classMetrics": class_metrics,
            "creditUsage": credit_usage  # Also store in recent outputs for dashboard
        }
        metrics_ref.set(metrics_data)
        
        # 4. Update time-based aggregations for dashboard graphing
        update_inference_aggregations(device_ref, current_time, primary_metric, class_metrics)
        
        # 5. Check for notification triggers (enhanced)
        triggered_notifications = trigger_action(user_id, device_id, clean_result, class_metrics)
        
        # 6. Cleanup old raw results
        cleanup_old_inference_data(device_ref, current_time)
        
        logger.info(f"Saved inference output for device {device_id}: mode={inference_mode}, metric={primary_metric}, notifications={len(triggered_notifications)}, credits={credit_usage}")
        
    except Exception as e:
        logger.error(f"Error saving inference output: {str(e)}")
        raise

def _parse_image_request(request) -> tuple:
    """
    Parse image data from request, handling both JSON and form data.
    
    Returns:
        tuple: (user_id, device_id, image_bytes, content_type)
    """
    try:
        if request.is_json:
            # Handle JSON payload
            data = request.get_json()
            user_id = data.get("userID")
            device_id = data.get("deviceID")
            image_data = data.get("image")
            
            if not all([user_id, device_id, image_data]):
                return None, None, None, None

            if image_data.startswith("data:"):
                # Data URL format
                header, encoded = image_data.split(",", 1)
                match = re.search(r"data:(image/\w+);base64", header)
                content_type = match.group(1) if match else "image/jpeg"
                image_bytes = base64.b64decode(encoded)
            else:
                # Pure base64
                content_type = "image/jpeg"
                image_bytes = base64.b64decode(image_data)
        else:
            # Handle form data
            user_id = request.form.get("userID")
            device_id = request.form.get("deviceID")
            image_file = request.files.get("image")
            
            if not all([user_id, device_id, image_file]):
                return None, None, None, None

            image_bytes = image_file.read()
            content_type = image_file.content_type or "image/jpeg"
        
        return user_id, device_id, image_bytes, content_type
        
    except Exception as e:
        logger.error(f"Error parsing image request: {str(e)}")
        return None, None, None, None
    

def get_device_similarity_settings(device_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get similarity comparison settings for a device with defaults.
    
    Args:
        device_data: Device document data from Firestore
        
    Returns:
        Dict with similarity settings and defaults
    """
    similarity_settings = device_data.get('similaritySettings', {})
    
    default_settings = {
        'enabled': True,
        'threshold': 0.85,
        'algorithm': 'histogram',  # 'histogram', 'perceptual', 'structural', 'fast', 'combined'
        'rolling_window_size': 10,
        'adaptive_threshold': False,
        'target_cache_ratio': 0.3
    }
    
    # Merge with defaults
    final_settings = default_settings.copy()
    final_settings.update(similarity_settings)
    
    # Validate settings
    final_settings['threshold'] = max(0.5, min(0.95, final_settings['threshold']))
    final_settings['rolling_window_size'] = max(5, min(50, final_settings['rolling_window_size']))
    final_settings['target_cache_ratio'] = max(0.1, min(0.8, final_settings['target_cache_ratio']))
    
    return final_settings


def get_device_overlay_settings(device_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get image overlay settings for a device with defaults.
    
    Args:
        device_data: Device document data from Firestore
        
    Returns:
        Dict with overlay settings and defaults
    """
    overlay_settings = device_data.get('overlaySettings', {})
    
    default_settings = {
        'enabled': True,
        'show_confidence_scores': True,
        'show_class_labels': True,
        'overlay_quality': 95,  # JPEG quality for overlay images
        'font_size': 14,
        'line_thickness': 3
    }
    
    # Merge with defaults
    final_settings = default_settings.copy()
    final_settings.update(overlay_settings)
    
    # Validate settings
    final_settings['overlay_quality'] = max(60, min(100, final_settings['overlay_quality']))
    final_settings['font_size'] = max(10, min(24, final_settings['font_size']))
    final_settings['line_thickness'] = max(1, min(6, final_settings['line_thickness']))
    
    return final_settings


def log_similarity_metrics(device_ref, similarity_result: Dict[str, Any], 
                          cache_hit: bool, current_time: int) -> None:
    """
    Log similarity comparison metrics for monitoring and tuning.
    
    Args:
        device_ref: Firestore device document reference
        similarity_result: Result from similarity calculation
        cache_hit: Whether this was a cache hit or miss
        current_time: Current timestamp in milliseconds
    """
    try:
        # Update device-level similarity stats
        similarity_stats_ref = device_ref.collection("similarity_stats")
        
        # Create stats document for current hour
        current_datetime = datetime.datetime.fromtimestamp(current_time / 1000, tz=datetime.timezone.utc)
        hour_start = current_datetime.replace(minute=0, second=0, microsecond=0)
        hour_timestamp = int(hour_start.timestamp() * 1000)
        
        hourly_stats_ref = similarity_stats_ref.document(str(hour_timestamp))
        
        # Get existing stats or initialize
        stats_doc = hourly_stats_ref.get()
        if stats_doc.exists:
            stats = stats_doc.to_dict()
        else:
            stats = {
                'hour_timestamp': hour_timestamp,
                'total_comparisons': 0,
                'cache_hits': 0,
                'cache_misses': 0,
                'total_similarity_score': 0.0,
                'total_processing_time_ms': 0.0,
                'algorithm_used': similarity_result.get('method', ''),
                'similarity_scores': [],  # Keep recent scores for analysis
                'created': firestore.SERVER_TIMESTAMP
            }
        
        # Update stats
        stats['total_comparisons'] += 1
        stats['total_similarity_score'] += similarity_result.get('similarity_score', 0.0)
        stats['total_processing_time_ms'] += similarity_result.get('processing_time_ms', 0.0)
        
        # Keep track of recent similarity scores (last 20)
        similarity_scores = stats.get('similarity_scores', [])
        similarity_scores.append(similarity_result.get('similarity_score', 0.0))
        if len(similarity_scores) > 20:
            similarity_scores = similarity_scores[-20:]
        stats['similarity_scores'] = similarity_scores
        
        if cache_hit:
            stats['cache_hits'] += 1
        else:
            stats['cache_misses'] += 1
        
        # Calculate derived metrics
        stats['cache_hit_ratio'] = stats['cache_hits'] / stats['total_comparisons']
        stats['average_similarity'] = stats['total_similarity_score'] / stats['total_comparisons']
        stats['average_processing_time_ms'] = stats['total_processing_time_ms'] / stats['total_comparisons']
        stats['updated'] = firestore.SERVER_TIMESTAMP
        
        # Save updated stats
        hourly_stats_ref.set(stats)
        
        # Log for monitoring
        logger.info(f"📊 Similarity metrics - Score: {similarity_result.get('similarity_score', 0):.3f}, "
                   f"Cache: {'HIT' if cache_hit else 'MISS'}, "
                   f"Time: {similarity_result.get('processing_time_ms', 0):.1f}ms, "
                   f"Hit ratio: {stats['cache_hit_ratio']:.3f}")
        
    except Exception as e:
        logger.error(f"Error logging similarity metrics: {str(e)}")


def store_similarity_comparison_data(device_ref, current_image_path: str, 
                                   inference_result: Dict[str, Any], 
                                   similarity_result: Optional[Dict[str, Any]],
                                   current_time: int) -> None:
    """
    Store the current image and inference result for next comparison.
    
    Args:
        device_ref: Firestore device document reference
        current_image_path: Path to current image in storage
        inference_result: Result from inference
        similarity_result: Result from similarity comparison (if any)
        current_time: Current timestamp in milliseconds
    """
    try:
        # Prepare clean inference result (without sensitive data)
        clean_inference_result = {k: v for k, v in inference_result.items() 
                                if k not in ['credit_usage', 'error']}
        
        # Store reference to current image and its inference result
        comparison_data = {
            'last_image_path': current_image_path,
            'last_inference_result': clean_inference_result,
            'last_inference_timestamp': current_time,
            'last_update_timestamp': current_time,
            'updated': firestore.SERVER_TIMESTAMP
        }
        
        # Add similarity result if available
        if similarity_result:
            comparison_data['last_similarity_check'] = {
                'similarity_score': similarity_result.get('similarity_score', 0.0),
                'method': similarity_result.get('method', ''),
                'processing_time_ms': similarity_result.get('processing_time_ms', 0.0),
                'timestamp': similarity_result.get('timestamp', current_time)
            }
        
        # Update device document with comparison data
        device_ref.update(comparison_data)
        
        logger.info(f"📁 Stored comparison data for next similarity check")
        
    except Exception as e:
        logger.error(f"Error storing similarity comparison data: {str(e)}")


def get_cache_performance_summary(user_id: str, device_id: str, hours: int = 24) -> Dict[str, Any]:
    """
    Get cache performance metrics for the last N hours.
    
    Args:
        user_id: User ID
        device_id: Device ID
        hours: Number of hours to look back
        
    Returns:
        Dict with performance summary
    """
    try:
        db = firestore.client()
        device_ref = db.collection("users").document(user_id).collection("devices").document(device_id)
        
        end_time = int(time.time() * 1000)
        start_time = end_time - (hours * 60 * 60 * 1000)
        
        # Query similarity stats
        stats_query = device_ref.collection("similarity_stats").where(
            "hour_timestamp", ">=", start_time
        ).where(
            "hour_timestamp", "<=", end_time
        ).order_by("hour_timestamp")
        
        stats_docs = stats_query.get()
        
        if not stats_docs:
            return {
                'cache_hit_ratio': 0.0,
                'total_comparisons': 0,
                'total_cache_hits': 0,
                'estimated_credit_savings': 0,
                'average_similarity_score': 0.0,
                'average_processing_time_ms': 0.0,
                'hours_analyzed': hours
            }
        
        # Aggregate stats
        total_comparisons = 0
        total_hits = 0
        total_similarity = 0.0
        total_processing_time = 0.0
        
        for doc in stats_docs:
            data = doc.to_dict()
            total_comparisons += data.get('total_comparisons', 0)
            total_hits += data.get('cache_hits', 0)
            total_similarity += data.get('total_similarity_score', 0.0)
            total_processing_time += data.get('total_processing_time_ms', 0.0)
        
        # Calculate summary metrics
        cache_hit_ratio = total_hits / total_comparisons if total_comparisons > 0 else 0.0
        avg_similarity = total_similarity / total_comparisons if total_comparisons > 0 else 0.0
        avg_processing_time = total_processing_time / total_comparisons if total_comparisons > 0 else 0.0
        
        # Estimate credit savings (assuming 20 credits per inference)
        estimated_savings = total_hits * 20
        
        return {
            'cache_hit_ratio': round(cache_hit_ratio, 3),
            'total_comparisons': total_comparisons,
            'total_cache_hits': total_hits,
            'total_cache_misses': total_comparisons - total_hits,
            'estimated_credit_savings': estimated_savings,
            'average_similarity_score': round(avg_similarity, 3),
            'average_processing_time_ms': round(avg_processing_time, 2),
            'hours_analyzed': hours,
            'performance_rating': 'excellent' if cache_hit_ratio > 0.4 else 'good' if cache_hit_ratio > 0.2 else 'poor'
        }
        
    except Exception as e:
        logger.error(f"Error getting cache performance summary: {str(e)}")
        return {'error': str(e)}


def log_processing_pipeline_metrics(device_id: str, pipeline_stages: Dict[str, float], 
                                  cache_hit: bool, total_time: float) -> None:
    """
    Log detailed metrics about the entire processing pipeline.
    
    Args:
        device_id: Device ID
        pipeline_stages: Dict of stage_name -> time_ms
        cache_hit: Whether inference was cached
        total_time: Total processing time in milliseconds
    """
    try:
        # Create performance summary
        stages_summary = []
        for stage, time_ms in pipeline_stages.items():
            percentage = (time_ms / total_time * 100) if total_time > 0 else 0
            stages_summary.append(f"{stage}: {time_ms:.1f}ms ({percentage:.1f}%)")
        
        # Log comprehensive pipeline metrics
        logger.info(f"🔄 Pipeline Metrics for {device_id}:")
        logger.info(f"   Cache: {'HIT' if cache_hit else 'MISS'}")
        logger.info(f"   Total: {total_time:.1f}ms")
        logger.info(f"   Stages: {' | '.join(stages_summary)}")
        
        # Log performance classification
        if total_time < 1000:
            performance = "🚀 FAST"
        elif total_time < 3000:
            performance = "⚡ GOOD"
        elif total_time < 5000:
            performance = "⏳ SLOW"
        else:
            performance = "🐌 VERY SLOW"
        
        logger.info(f"   Performance: {performance}")
        
    except Exception as e:
        logger.error(f"Error logging pipeline metrics: {str(e)}")


def cleanup_old_similarity_stats(user_id: str, device_id: str, days_to_keep: int = 7) -> Dict[str, Any]:
    """
    Clean up old similarity statistics to manage storage.
    
    Args:
        user_id: User ID
        device_id: Device ID
        days_to_keep: Number of days of stats to keep
        
    Returns:
        Dict with cleanup results
    """
    try:
        db = firestore.client()
        device_ref = db.collection("users").document(user_id).collection("devices").document(device_id)
        
        # Calculate cutoff timestamp
        cutoff_time = int(time.time() * 1000) - (days_to_keep * 24 * 60 * 60 * 1000)
        
        # Query old stats
        old_stats_query = device_ref.collection("similarity_stats").where(
            "hour_timestamp", "<", cutoff_time
        ).limit(100)  # Process in batches
        
        old_stats = old_stats_query.get()
        
        deleted_count = 0
        
        # Delete old stats documents
        for doc in old_stats:
            try:
                doc.reference.delete()
                deleted_count += 1
            except Exception as delete_error:
                logger.info(f"Error deleting stats doc {doc.id}: {str(delete_error)}")
        
        logger.info(f"🧹 Cleaned up {deleted_count} old similarity stats for device {device_id}")
        
        return {
            'deleted_count': deleted_count,
            'cutoff_days': days_to_keep,
            'success': True
        }
        
    except Exception as e:
        logger.error(f"Error cleaning up similarity stats: {str(e)}")
        return {
            'error': str(e),
            'success': False
        }


def update_device_processing_stats(device_ref, processing_metrics: Dict[str, Any]) -> None:
    """
    Update device-level processing statistics.
    
    Args:
        device_ref: Firestore device document reference
        processing_metrics: Dict with processing metrics
    """
    try:
        # Update device document with latest processing stats
        stats_update = {
            'lastProcessingMetrics': processing_metrics,
            'lastProcessingTimestamp': int(time.time() * 1000),
            'processingStatsUpdated': firestore.SERVER_TIMESTAMP
        }
        
        # Add cumulative stats
        device_doc = device_ref.get()
        if device_doc.exists:
            device_data = device_doc.to_dict()
            cumulative_stats = device_data.get('cumulativeProcessingStats', {})
            
            # Update cumulative counters
            cumulative_stats['totalProcessed'] = cumulative_stats.get('totalProcessed', 0) + 1
            cumulative_stats['totalCacheHits'] = cumulative_stats.get('totalCacheHits', 0) + (1 if processing_metrics.get('cache_hit') else 0)
            cumulative_stats['totalProcessingTimeMs'] = cumulative_stats.get('totalProcessingTimeMs', 0) + processing_metrics.get('total_time_ms', 0)
            
            # Calculate running averages
            if cumulative_stats['totalProcessed'] > 0:
                cumulative_stats['averageProcessingTimeMs'] = cumulative_stats['totalProcessingTimeMs'] / cumulative_stats['totalProcessed']
                cumulative_stats['overallCacheHitRatio'] = cumulative_stats['totalCacheHits'] / cumulative_stats['totalProcessed']
            
            stats_update['cumulativeProcessingStats'] = cumulative_stats
        
        device_ref.update(stats_update)
        
    except Exception as e:
        logger.error(f"Error updating device processing stats: {str(e)}")

