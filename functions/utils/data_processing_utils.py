# utils/data_processing_utils.py - Simplified version

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

def get_inference_similarity_threshold(device_data: Dict[str, Any]) -> float:
    """
    Get simple similarity threshold (0.0-1.0) with default of 0.95.
    Only configurable setting - everything else uses system defaults.
    """
    # threshold = device_data.get('inferenceSimilarityThreshold', 0.95)
    threshold = device_data.get('inferenceSimilarityThreshold', 0.0) # SIMILARITY CHECK DISABLED
    return max(0.0, min(1.0, float(threshold)))

def get_similarity_system_settings() -> Dict[str, Any]:
    """
    Get non-configurable system settings for similarity processing.
    These are fixed values optimized for Firebase Functions environment.
    """
    return {
        'algorithm': 'average_hash',        # Fast, works without heavy dependencies
        'rolling_window_size': 10,          # Keep last 10 images
        'max_comparison_size': (256, 256),  # Resize images for fast comparison
        'enable_caching': True              # Always enabled if threshold > 0
    }

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

def get_basic_processing_stats(user_id: str, device_id: str, hours: int = 24) -> Dict[str, Any]:
    """
    Get basic processing statistics for the last N hours.
    Simplified version that works with the new system.
    """
    try:
        db = firestore.client()
        device_ref = db.collection("users").document(user_id).collection("devices").document(device_id)
        
        # Get device data for current settings
        device_doc = device_ref.get()
        if not device_doc.exists:
            return {'error': 'Device not found'}
        
        device_data = device_doc.to_dict()
        
        # Calculate time range
        end_time = int(time.time() * 1000)
        start_time = end_time - (hours * 60 * 60 * 1000)
        
        # Get recent inference results to calculate cache hits
        recent_results = device_ref.collection("inference_results").where(
            "timestamp", ">=", start_time
        ).where(
            "timestamp", "<=", end_time
        ).order_by("timestamp").get()
        
        # Analyze results
        total_inferences = len(recent_results.docs)
        cache_hits = 0
        total_credits = 0
        
        for doc in recent_results.docs:
            data = doc.to_dict()
            result = data.get('result', {})
            
            if result.get('cache_hit', False):
                cache_hits += 1
            
            credits = data.get('creditUsage', 0)
            total_credits += credits
        
        # Calculate metrics
        cache_hit_ratio = cache_hits / total_inferences if total_inferences > 0 else 0.0
        estimated_savings = cache_hits * 20  # Assume 20 credits per inference
        
        return {
            'similarity_threshold': get_inference_similarity_threshold(device_data),
            'total_inferences': total_inferences,
            'cache_hits': cache_hits,
            'cache_misses': total_inferences - cache_hits,
            'cache_hit_ratio': round(cache_hit_ratio, 3),
            'total_credits_used': total_credits,
            'estimated_credit_savings': estimated_savings,
            'hours_analyzed': hours,
            'performance_rating': 'excellent' if cache_hit_ratio > 0.4 else 'good' if cache_hit_ratio > 0.2 else 'poor'
        }
        
    except Exception as e:
        logger.error(f"Error getting basic processing stats: {str(e)}")
        return {'error': str(e)}