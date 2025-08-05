# the following few lines are just to remind me of what commands I need to deploy
# functions/venv/Scripts/activate
# pip install -r functions/requirements.txt

# firebase deploy --only functions

# firebase deploy --only functions:process_icon_creation,functions:assistant_chat,functions:perform_inference,functions:set_classes
# firebase deploy --only functions:process_icon_creation
# firebase deploy --only functions:assistant_chat
# firebase deploy --only functions:perform_inference
# firebase deploy --only functions:set_classes
# firebase deploy --only functions:device_heartbeat


from firebase_functions import storage_fn, https_fn, logger, pubsub_fn
from firebase_admin import initialize_app, storage, firestore, credentials, messaging
import tempfile
import os
import requests
import json
from google.cloud import storage, pubsub_v1
import datetime
import base64
import time
import gc
from google.cloud import storage as gcs

import re
from firebase_functions import https_fn  # Use Firebase Functions' decorator
from flask import Request, jsonify

from pydantic import BaseModel
import asyncio

import anthropic
import openai
from openai import OpenAI
from PIL import Image
import io
import uuid 
import math

# Own imports
from config.loader import get_openai_api_key, get_anthropic_api_key, get_gemini_api_key

# Import utility functions
from utils.notification_utils import (
    DetectionPoint,
    DetectionBox,
    NotificationTrigger,
    extract_detections_from_inference,
    is_point_in_region,
    check_count_threshold_trigger,
    check_location_trigger,
    send_notification,
    trigger_action
)
from utils.data_processing_utils import (
    add_cors_headers,
    create_model_output,
    update_inference_aggregations,
    cleanup_old_inference_data,
    extract_metrics_from_inference,
    update_credit_usage,
    save_inference_output,
    _parse_image_request
)

#openai.api_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(
  api_key=get_openai_api_key()
)
openai.api_key = get_openai_api_key()

client = anthropic.Anthropic(
    api_key=get_anthropic_api_key(),
)

cred = credentials.Certificate("aimanagerfirebasebackend-firebase-adminsdk-fbsvc-5977b3630b.json")
initialize_app()

HYPEROPT_URL = "https://hyperopt-trainer-723311357828.europe-west4.run.app/optimize"



DIAGONAL_SWAG_URL = "https://diagonalswag-723311357828.europe-west4.run.app/predict"







# =================== END NOTIFICATION SYSTEM ===================










# Updated receive_image function for main.py

# Updated receive_image function to handle credit usage in response
@https_fn.on_request(region="europe-west4")
def receive_image(request):
    """
    Enhanced receive_image function that processes inference for operational devices
    and triggers notifications based on detection results. Now includes credit tracking.
    """
    if request.method == 'OPTIONS':
        response = jsonify({})
        response = add_cors_headers(response)
        return response, 204
    
    try:
        # Parse the image upload (existing logic)
        if request.method != "POST":
            return add_cors_headers(jsonify({"error": "Invalid HTTP method. POST required."})), 405

        # Extract image data and parameters
        user_id, device_id, image_bytes, content_type = _parse_image_request(request)
        
        if not all([user_id, device_id, image_bytes]):
            return add_cors_headers(jsonify({"error": "Missing required parameters"})), 400

        # Store the image with timestamp
        timestamp = str(int(time.time() * 1000))
        file_extension = "jpg" if content_type == "image/jpeg" else "png"
        filename = f"{timestamp}.{file_extension}"
        file_path = f"users/{user_id}/devices/{device_id}/receiving/{filename}"
        
        # Upload to Firebase Storage
        bucket = storage.bucket()
        blob = bucket.blob(file_path)
        blob.upload_from_string(image_bytes, content_type=content_type)
        
        # Force garbage collection after image upload
        gc.collect()

        # Initialize response data
        response_data = {
            "status": "success",
            "file_path": file_path,
            "timestamp": timestamp,
        }
        
        # Check device status for inference processing
        db = firestore.client()
        device_ref = db.collection("users").document(user_id).collection("devices").document(device_id)
        device_doc = device_ref.get()
        
        if device_doc.exists:
            device_data = device_doc.to_dict()
            device_status = device_data.get("status", "Not Configured")
            inference_mode = device_data.get("inferenceMode", "Detect")
            
            logger.info(f"Device {device_id} status: {device_status}, mode: {inference_mode}")
            
            # Only process inference if device is operational
            if device_status == "Operational":
                try:
                    logger.info(f"Processing inference for operational device {device_id}")
                    
                    # Call inference function
                    inference_result = inference(image_bytes, user_id, device_id)
                    
                    # Force garbage collection after inference processing
                    gc.collect()

                    if "error" not in inference_result:
                        # Save inference output for dashboard and process notifications
                        save_inference_output(user_id, device_id, inference_result, inference_mode)
                        
                        # Extract metrics for response
                        clean_result = {k: v for k, v in inference_result.items() if k != "credit_usage"}
                        primary_metric, class_metrics = extract_metrics_from_inference(clean_result, inference_mode)
                        
                        # Add inference data to response
                        response_data.update({
                            "inference_processed": True,
                            "inference_mode": inference_mode,
                            "model_used": inference_result.get("modelUsed", "unknown"),
                            "credit_usage": inference_result.get("credit_usage", 0),
                            "primary_metric": primary_metric,
                            "total_detections": class_metrics.get("totalDetections", 0),
                            "class_counts": class_metrics.get("classCounts", {}),
                            "average_confidence": round(class_metrics.get("averageConfidence", 0.0), 3)
                        })
                        
                        logger.info(f"Successfully processed inference: {primary_metric} detections, "
                                  f"classes: {list(class_metrics.get('classCounts', {}).keys())}, "
                                  f"credits used: {inference_result.get('credit_usage', 0)}")
                        
                    else:
                        logger.error(f"Inference error for device {device_id}: {inference_result['error']}")
                        response_data["inference_error"] = inference_result["error"]
                        
                except Exception as inference_error:
                    logger.error(f"Exception during inference processing for device {device_id}: {str(inference_error)}")
                    response_data["inference_error"] = str(inference_error)
            else:
                logger.info(f"Device {device_id} not operational (status: {device_status}), skipping inference")
                response_data["device_status"] = device_status
                response_data["inference_skipped"] = "Device not operational"
        else:
            logger.error(f"Device {device_id} not found in Firestore")
            response_data["warning"] = "Device not found in database"
        
        # Final garbage collection before returning
        gc.collect()
        return add_cors_headers(jsonify(response_data)), 200

    except Exception as e:
        logger.error(f"Error in receive_image: {str(e)}")
        # Force garbage collection even on error
        gc.collect()
        return add_cors_headers(jsonify({"error": str(e)})), 500

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

@https_fn.on_request(region="europe-west4")
def device_heartbeat(request):
    """
    Handles device heartbeat and settings synchronization.
    Verifies camera ID, stores WiFi signal strength, and can trigger device reset.
    """
    try:
        db = firestore.client()
        
        # Parse request data
        data = request.get_json()
        user_id = data.get("user_id")
        device_id = data.get("device_id")
        received_camera_id = data.get("camera_id")  # MAC address from ESP32
        wifi_signal_strength = data.get("wifi_signal_strength")  # RSSI in dBm

        if not all([user_id, device_id, received_camera_id]):
            logger.error(f"Missing required fields. user_id: {user_id}, device_id: {device_id}, camera_id: {received_camera_id}")
            return {"error": "Missing required fields"}, 400

        # Validate WiFi signal strength (optional field, but validate if present)
        if wifi_signal_strength is not None:
            try:
                wifi_signal_strength = int(wifi_signal_strength)
                # Validate reasonable RSSI range (-100 to 0 dBm)
                if wifi_signal_strength < -100 or wifi_signal_strength > 0:
                    logger.warning(f"WiFi signal strength out of expected range: {wifi_signal_strength} dBm")
            except (ValueError, TypeError):
                logger.warning(f"Invalid WiFi signal strength format: {wifi_signal_strength}")
                wifi_signal_strength = None

        # Normalize received camera ID (remove colons and convert to uppercase)
        normalized_received_id = received_camera_id.replace(":", "").upper()
        logger.info(f"Received camera ID: {received_camera_id} (normalized: {normalized_received_id})")
        
        if wifi_signal_strength is not None:
            logger.info(f"WiFi Signal Strength: {wifi_signal_strength} dBm")

        device_ref = db.collection("users").document(user_id).collection("devices").document(device_id)
        device_doc = device_ref.get()

        if not device_doc.exists:
            logger.error(f"Device not found: {device_id}")
            return {"error": "Device not found"}, 404

        device_data = device_doc.to_dict()
        stored_camera_id = device_data.get('connectedCameraId')
        
        # Prepare update data
        current_timestamp = str(int(datetime.datetime.now().timestamp() * 1000))
        update_data = {
            'last_heartbeat': current_timestamp
        }
        
        # Add WiFi signal strength if provided
        if wifi_signal_strength is not None:
            update_data['wifi_signal_strength'] = wifi_signal_strength
        
        # Check if this is the first heartbeat (no connectedCameraId stored yet)
        if not stored_camera_id:
            logger.info(f"First connection for device {device_id}. Storing camera ID: {received_camera_id}")
            # Store the original format of the camera_id
            update_data['connectedCameraId'] = received_camera_id
            device_ref.update(update_data)
        else:
            # Normalize stored camera ID for comparison
            normalized_stored_id = stored_camera_id.replace(":", "").upper()
            logger.info(f"Comparing camera IDs - Stored: {stored_camera_id} (normalized: {normalized_stored_id})")
            
            # Compare normalized versions
            if normalized_stored_id != normalized_received_id:
                logger.info(
                    f"Camera ID mismatch detected for device {device_id}!\n"
                    f"Expected: {stored_camera_id} ({normalized_stored_id})\n"
                    f"Received: {received_camera_id} ({normalized_received_id})"
                )
                # Uncomment the lines below if you want to enforce camera ID matching
                # return {
                #     "reset": True,
                #     "message": f"Camera ID mismatch. Expected {stored_camera_id}, got {received_camera_id}"
                # }, 200

        # Update device with heartbeat timestamp and WiFi signal strength
        device_ref.update(update_data)
        
        # Return current settings
        settings = {
            "captureIntervalHours": device_data.get("captureIntervalHours", 0),
            "captureIntervalMinutes": device_data.get("captureIntervalMinutes", 1),
            "captureIntervalSeconds": device_data.get("captureIntervalSeconds", 0),
            "motionTriggered": device_data.get("motionTriggered", False),
            "saveImages": device_data.get("saveImages", True)
        }

        logger.info(f"Heartbeat successful for device {device_id}")
        return {"settings": settings}, 200

    except Exception as e:
        logger.error(f"Error in device_heartbeat: {str(e)}")
        return {"error": str(e)}, 500

@pubsub_fn.on_message_published(topic="icon-generation-requests", region="europe-west4")
def process_icon_creation(event: pubsub_fn.CloudEvent) -> None:
    """Process icon creation requests from Pub/Sub."""
    try:
        # Decode the Pub/Sub message
        encoded_data = event.data.message.data
        message_data = json.loads(base64.b64decode(encoded_data).decode('utf-8'))
        
        db = firestore.client()
        
        # Extract the settings
        settings = message_data.get("settings", {})
        user_id = settings.get("userId")
        device_id = settings.get("deviceId")
        description = settings.get("description", "a specific task")
        messageTimestamp = message_data.get("messageTimestamp")

        if messageTimestamp:
            assistant_doc_ref = db.collection("users").document(user_id)\
                .collection("devices").document(device_id)\
                .collection("assistant").document(messageTimestamp)
            
            # Initialize document
            assistant_doc_ref.set({
                "textArray": ["Creating icon..."],
                "progress": "0/1",
                "status": "processing"
            }, merge=True)

        if not user_id or not device_id:
            raise ValueError("Missing userId or deviceId in settings")

        # Build the prompt for image generation
        prompt = (
            f"Create a simple, minimalistic icon with a flat design that represents "
            f"either classification or object detection and the icon is needed for the user so easily recognize specific devices in a long device list, so it needs to be simple but convey the essential task."
            f"here is the description of the device from the chatgpt assistant: {description} "
            f"The icon should use clear lines,modern aesthetics, subtle colors, and be easy to understand at a glance. "
            f"the icon should only be using different shades of grey with at most some parts in orange to highlight them, to create consistent designs."
            f"It should also not have too much fine detail, so that its still clearly visible even when downscaled."
            f"Make there is absolutely NO TEXT in the icon. Make the icon close to the size of the whole image (with rounded corner though) so that there is no dead space."
        )

        # Generate image with DALL-E
        response = openai_client.images.generate(
            prompt=prompt,
            model="dall-e-3",
            n=1,
            size="1024x1024"
        )

        image_data = response.data
        if not image_data or not image_data[0].url:
            raise Exception("No image URL returned from OpenAI")
        
        image_url = image_data[0].url

        # Download the generated image
        image_response = requests.get(image_url)
        if image_response.status_code != 200:
            raise Exception("Failed to download generated image")
        
        image_bytes = image_response.content

        # Upload to Firebase Storage
        bucket = storage.bucket()
        storage_path = f"users/{user_id}/devices/{device_id}/icon.png"
        blob = bucket.blob(storage_path)
        blob.upload_from_string(image_bytes, content_type="image/png")
        blob.make_public()
        icon_url = blob.public_url

        # Update device document
        device_ref = db.collection("users").document(user_id)\
                      .collection("devices").document(device_id)
        device_ref.update({
            # "iconAlreadyCreated": True,
            "imageUrl": icon_url
        })

        if messageTimestamp:
            # First update: Set the image URL
            assistant_doc_ref.update({
                "imageUrls": [icon_url],
                "textArray": ["Creating icon..."],
                "progress": "0/1"
            })

            # Small delay to ensure frontend receives the URL
            time.sleep(1)

            # Second update: Set status to complete
            assistant_doc_ref.update({
                "textArray": ["Icon created successfully"],
                "progress": "1/1",
                "status": "actionComplete"
            })

        logger.info(f"Successfully created icon for device {device_id}")

    except Exception as e:
        logger.error(f"Error in process_icon_creation: {str(e)}")
        if 'messageTimestamp' in locals() and 'assistant_doc_ref' in locals():
            try:
                assistant_doc_ref.update({
                    "status": "error",
                    "error": str(e),
                    "textArray": ["Error creating icon"]
                })
            except Exception as update_error:
                logger.error(f"Error updating error status: {str(update_error)}")
        raise

@https_fn.on_request(region="europe-west4")
def set_classes(request):
    """
    Firebase function to set classes and class descriptions for a device.
    
    Expected request body:
    {
        "user_id": "user123",
        "device_id": "device456",
        "classes": ["class1", "class2"],
        "classDescriptions": ["description1", "description2"],
        "messageTimestamp": "timestamp" (optional)
    }
    """
    if request.method == 'OPTIONS':
        response = jsonify({})
        response = add_cors_headers(response)
        return response, 204
    try:
        # Parse request data
        request_json = request.get_json(silent=True)
        if not request_json:
            return add_cors_headers({"error": "Invalid JSON payload"}), 400
            
        # Extract parameters
        user_id = request_json.get("user_id")
        device_id = request_json.get("device_id")
        classes = request_json.get("classes", [])
        class_descriptions = request_json.get("classDescriptions", [])
        message_timestamp = request_json.get("messageTimestamp")
        
        # Validate required parameters
        if not all([user_id, device_id]):
            return add_cors_headers({"error": "Missing required parameters"}), 400
            
        # Validate classes array
        if not isinstance(classes, list):
            return add_cors_headers({"error": "Classes must be an array"}), 400
            
        # Validate class_descriptions array
        if not isinstance(class_descriptions, list):
            return add_cors_headers({"error": "ClassDescriptions must be an array"}), 400
            
        # Ensure class_descriptions matches classes length
        while len(class_descriptions) < len(classes):
            class_descriptions.append("")
            
        # If messageTimestamp is provided, update the assistant document to show progress
        if message_timestamp:
            db = firestore.client()
            assistant_doc_ref = db.collection("users").document(user_id)\
                .collection("devices").document(device_id)\
                .collection("assistant").document(message_timestamp)
            
            # Initialize document with processing status
            assistant_doc_ref.set({
                "textArray": ["Setting up classes..."],
                "progress": "0/1",
                "status": "processing"
            }, merge=True)
        
        # Update the device document in Firestore
        db = firestore.client()
        device_ref = db.collection('users').document(user_id).collection('devices').document(device_id)
        
        update_data = {
            'classes': classes,
            'classDescriptions': class_descriptions,
            'updated': int(datetime.datetime.now().timestamp() * 1000),  # Use timestamp instead of SERVER_TIMESTAMP
            'setupStage': 1.0,
            'status': 'Operational'
        }
        
        device_ref.update(update_data)
        
        # If messageTimestamp is provided, update the assistant document to show completion
        if message_timestamp:
            # Update document with completion status
            assistant_doc_ref.update({
                "textArray": [f"Successfully defined {len(classes)} classes"],
                "progress": "1/1",
                "status": "actionComplete",
                "classes": classes,
                "classDescriptions": class_descriptions
            })
        
        return add_cors_headers({
            "success": True,
            "message": f"Updated {len(classes)} classes",
            "classes": classes
        })
        
    except Exception as e:
        logger.error(f"Error in set_classes: {str(e)}")
        
        # Update error status in the document if messageTimestamp was provided
        if 'message_timestamp' in locals() and message_timestamp:
            try:
                assistant_doc_ref.update({
                    "textArray": [f"Error setting classes: {str(e)}"],
                    "progress": "0/1",
                    "status": "error"
                })
            except Exception as update_error:
                logger.error(f"Error updating error status: {str(update_error)}")
                
        return add_cors_headers({"error": str(e)}), 500


# ------------------------------------- INFERENCE ---------------------------------------

import io
import base64
import requests
import json
import concurrent.futures
from firebase_functions import https_fn, logger
from firebase_admin import firestore, storage
import datetime
import time

import os
from inference.moondream_inference import run_moondream_inference
from inference.gemini_inference import run_gemini_inference

# Configuration for Moondream API
USE_CLOUD_API = True  # Set to False to use the local API
MOONDREAM_CLOUD_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJrZXlfaWQiOiIwNTYxZTE1Mi01MWU1LTRjZmUtYjY4Mi0wZTc3NTlmNmE3MjYiLCJpYXQiOjE3NDMxMDczMTh9.3zAvYwRYQtQxKqo10ZCBbTzdqG1kZh1eYu4O4C4EVK0"


def inference(image_data, user_id, device_id):
    """
    Core inference function that acts as a switch to dispatch to the correct model
    based on the INFERENCE_MODEL environment variable.
    """
    try:
        # Get device configuration from Firestore
        db = firestore.client()
        device_ref = db.collection('users').document(user_id).collection('devices').document(device_id)
        device_doc = device_ref.get()

        if not device_doc.exists:
            return {"error": "Device not found"}

        device_config = device_doc.to_dict()
        inference_mode = device_config.get('inferenceMode', 'Detect')

        # This is the main switch logic
        inference_model_name = os.getenv("INFERENCE_MODEL", "moondream").lower()
        logger.info(f"Using inference model: {inference_model_name}")

        result = {}
        if inference_mode == 'Detect':
            if inference_model_name == "gemini":
                # Call the unified Gemini function
                result = run_gemini_inference(image_data, device_config)
            else:
                # Default to the Moondream pipeline
                result = run_moondream_inference(image_data, device_config)
        else:
            logger.warning(f"Inference mode '{inference_mode}' is not 'Detect'. Defaulting to Moondream.")
            result = run_moondream_inference(image_data, device_config)

        # Add metadata and return in the expected format
        timestamp = str(int(time.time() * 1000))
        result["timestamp"] = timestamp
        result["inferenceMode"] = inference_mode
        result["modelUsed"] = inference_model_name

        return result

    except Exception as e:
        logger.error(f"Error in top-level inference function: {str(e)}")
        return {"error": str(e)}

@https_fn.on_request(region="europe-west4")
def perform_inference(request):
    """
    Firebase function to perform inference on an image.
    
    Expected request body:
    {
        "user_id": "user123",
        "device_id": "device456",
        "image_path": "path/to/image.jpg"
    }
    """
    if request.method == 'OPTIONS':
        response = jsonify({})
        response = add_cors_headers(response)
        return response, 204
    try:
        logger.info("Starting perform_inference function")
        
        # Parse request data
        request_json = request.get_json(silent=True)
        if not request_json:
            logger.error("Invalid JSON payload")
            response = jsonify({"error": "Invalid JSON payload"})
            return add_cors_headers(response), 400
        
        # Log request data for debugging (be careful with sensitive data)
        logger.info(f"Request data: user_id={request_json.get('user_id')}, device_id={request_json.get('device_id')}")
        logger.info(f"Image path: {request_json.get('image_path')}")
        
        # Extract parameters
        user_id = request_json.get("user_id")
        device_id = request_json.get("device_id")
        image_path = request_json.get("image_path")
        
        # Validate required parameters
        if not all([user_id, device_id, image_path]):
            return add_cors_headers({"error": "Missing required parameters"}), 400
        
        # Download the image from Firebase Storage
        bucket = storage.bucket()
        blob = bucket.blob(image_path)
        
        # Check if image exists
        if not blob.exists():
            return add_cors_headers({"error": f"Image not found: {image_path}"}), 404
        
        # Download image as bytes
        image_data = blob.download_as_bytes()
        
        # Perform inference
        result = inference(image_data, user_id, device_id)
        
        credit_usage = result.get("credit_usage", 0)
        if credit_usage > 0:
            logger.info(f"Processing credit usage: {credit_usage}")
            update_credit_usage(user_id, device_id, credit_usage)

        # Return the result
        if "error" in result:
            return {"error": result["error"]}, 500
        
        clean_result = {k: v for k, v in result.items() if k != "credit_usage"}
        return add_cors_headers(clean_result)
        
    except Exception as e:
        logger.error(f"Error in perform_inference: {str(e)}")
        return add_cors_headers({"error": str(e)}), 500

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
            logger.warning(f"Device {device_id} not found for credit update")
        
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

# -------------------------------- ASSISTANT_CHAT BEGIN  --------------------------------

# Settings that the assistant is not allowed to modify directly
SETTINGS_BLACKLIST = {
    'status', 'connectedCameraId', 'last_heartbeat',
    'iconAlreadyCreated', 'deletionStarted'
}

# Available inference modes with descriptions
INFERENCE_MODES = {
    "Point": "Find exact coordinates of specific features in the image",
    "Detect": "Identify and locate objects with bounding boxes, should be used for counting and therefor detecting classes",
    "VQA": "Answer questions about the image content",
    "Caption": "Generate descriptive captions for the image"
}

SETUP_PROCESS_OVERVIEW = """
Complete Setup Process:

Stage 0 (0.0-0.9): Initial Setup
- Define device name and task description
- Select appropriate inference mode 
- Define classes (Detect and Point can work with a single or multiple classes)
- Create device icon

Stage 1 (1.0-1.9): Testing
- Test the inference with sample images
- Verify results are as expected
- Adjust settings if needed

Stage 2 (2.0): Operational
- Device fully configured and ready for use
- Ongoing support and optimization
"""

# ---------------------------
# 1. Tool Processing Functions
# ---------------------------

def process_create_device_icon(user_id, device_id, icon_request, messageTimestamp):
    """Process createDeviceIcon tool request"""
    try:
        db = firestore.client()
        
        # Prepare payload
        payload = {
            "settings": {
                "userId": user_id,
                "deviceId": device_id,
                "description": icon_request
            },
            "messageTimestamp": messageTimestamp
        }

        # Publish to Pub/Sub topic
        publisher = pubsub_v1.PublisherClient()
        topic_path = publisher.topic_path('aimanagerfirebasebackend', 'icon-generation-requests')
        
        publisher.publish(
            topic_path,
            json.dumps(payload).encode('utf-8')
        )
        
        # Update device document
        device_ref = db.collection("users").document(user_id)\
                    .collection("devices").document(device_id)
        device_ref.update({
            "iconAlreadyCreated": True,
        })
        
        return {
            "status": "success",
            "message": "Icon generation is being processed in the background"
        }
        
    except Exception as e:
        logger.error(f"Exception in createDeviceIcon: {str(e)}")
        return {"status": "error", "message": str(e)}

def process_set_classes(user_id, device_id, classes, class_descriptions=None):
    """Process setClasses tool request"""
    try:
        # Ensure class_descriptions has same length as classes
        if class_descriptions is None:
            class_descriptions = [""] * len(classes)
        
        while len(class_descriptions) < len(classes):
            class_descriptions.append("")
        
        # Call the set_classes function
        set_classes_url = "https://europe-west4-aimanagerfirebasebackend.cloudfunctions.net/set_classes"
        payload = {
            "user_id": user_id,
            "device_id": device_id,
            "classes": classes,
            "classDescriptions": class_descriptions
        }
        
        response = requests.post(set_classes_url, json=payload)
        if response.status_code == 200:
            return {
                "status": "success",
                "message": f"Updated {len(classes)} classes",
                "classes": classes,
                "classDescriptions": class_descriptions
            }
        else:
            return {
                "status": "error",
                "message": f"Failed to update classes: {response.text}",
                "status_code": response.status_code
            }
    except Exception as e:
        logger.error(f"Exception in setClasses: {str(e)}")
        return {"status": "error", "message": str(e)}


# ---------------------------
# 2. Firestore Settings Update Function
# ---------------------------
def update_device_settings(user_id: str, device_id: str, settings: dict) -> bool:
    """Update device settings in Firestore with robust type-safe conversions and validation."""
    db = firestore.client()
    device_ref = db.collection('users').document(user_id).collection('devices').document(device_id)

    
    try:
        device_doc = device_ref.get()
        if not device_doc.exists:
            logger.error("Device document does not exist")
            return False

        current_settings = device_doc.to_dict()
        logger.info(f"Current device settings: {json.dumps(current_settings, indent=2)}")

        # First pass: collect all changes for validation
        proposed_updates = {}
        for key, new_value in settings.items():
            # Skip blacklisted settings
            if key in SETTINGS_BLACKLIST:
                logger.error(f"Skipping blacklisted setting: {key}")
                continue
                
            # Skip non-existent settings
            if key not in current_settings:
                logger.error(f"Skipping non-existent setting: {key}")
                continue
                
            current_value = current_settings[key]
            
            try:
                # Handle type conversions
                if isinstance(current_value, list):
                    # Lists require special handling
                    if isinstance(new_value, str):
                        try:
                            # Try to parse as JSON first
                            parsed_value = json.loads(new_value)
                            if isinstance(parsed_value, list):
                                proposed_updates[key] = parsed_value
                            else:
                                # If JSON parsing succeeds but doesn't yield a list
                                logger.error(f"Expected list for {key}, got {type(parsed_value).__name__}")
                                continue
                        except json.JSONDecodeError:
                            # Fall back to splitting by comma
                            proposed_updates[key] = [x.strip() for x in new_value.split(',')]
                    elif isinstance(new_value, list):
                        proposed_updates[key] = new_value
                    else:
                        logger.error(f"Cannot convert {type(new_value).__name__} to list for {key}")
                        continue
                elif isinstance(current_value, bool):
                    if isinstance(new_value, bool):
                        proposed_updates[key] = new_value
                    else:
                        # Convert string 'true'/'false' to boolean
                        proposed_updates[key] = str(new_value).lower() == 'true'
                elif isinstance(current_value, int):
                    try:
                        proposed_updates[key] = int(float(new_value))
                    except (ValueError, TypeError):
                        logger.error(f"Cannot convert {new_value} to int for {key}")
                        continue
                elif isinstance(current_value, float):
                    try:
                        proposed_updates[key] = float(new_value)
                    except (ValueError, TypeError):
                        logger.error(f"Cannot convert {new_value} to float for {key}")
                        continue
                else:
                    # For strings and other types
                    proposed_updates[key] = new_value
                
                logger.info(f"Proposed conversion for {key}: {new_value} -> {proposed_updates[key]}")
                
            except Exception as e:
                logger.error(f"Error converting setting {key}: {str(e)}")
                continue
                
        if proposed_updates:
            logger.info(f"Updating settings with: {json.dumps(proposed_updates, indent=2)}")
            device_ref.update(proposed_updates)
            return True
        
        logger.error("No valid settings to update after validation")
        return False

    except Exception as e:
        logger.error(f"Error updating settings: {str(e)}")
        return False

# ---------------------------
# 3. Actual Assistant Chat Function
# ---------------------------

# Simplified Gemini-based assistant_chat function
# Replace the existing assistant_chat function in main.py with this

import google.genai as genai
from google.genai import types
import json
import re
from firebase_functions import https_fn, logger
from firebase_admin import firestore, storage
from flask import Request, jsonify
import base64
from PIL import Image
import io

# Simplified tool definitions for Gemini
GEMINI_TOOLS = [
    {
        "function_declarations": [
            {
                "name": "createDeviceIcon",
                "description": "Generate a custom device icon based on task description",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "iconRequest": {
                            "type": "STRING",
                            "description": "Description of the device task for icon generation"
                        }
                    },
                    "required": ["iconRequest"]
                }
            },
            {
                "name": "setClasses",
                "description": "Define classification classes and their descriptions",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "classes": {
                            "type": "ARRAY",
                            "items": {"type": "STRING"},
                            "description": "List of class names"
                        },
                        "classDescriptions": {
                            "type": "ARRAY",
                            "items": {"type": "STRING"},
                            "description": "List of descriptions for each class (optional, defaults to empty descriptions)"
                        }
                    },
                    "required": ["classes"]
                }
            },
            {
                "name": "updateSettings",
                "description": "Update device settings",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "settings": {
                            "type": "OBJECT",
                            "description": "Key-value pairs of settings to update"
                        }
                    },
                    "required": ["settings"]
                }
            }
        ]
    }
]

def build_system_prompt(device_data: dict, setup_stage: float, user_language: str) -> str:
    """Build a concise system prompt for Gemini"""
    
    # Determine current stage guidance
    if setup_stage >= 2:
        stage_guidance = "Device is operational. Help with optimization and questions."
    elif setup_stage >= 1:
        stage_guidance = "Guide user to test the device. When testing is complete, update setupStage to 2.0."
    else:
        stage_guidance = "As soon as you understand what the user wants to monitor, immediately set up the device with appropriate defaults. Only ask questions if the user's intent is unclear."
    
    return f"""You are an AI assistant helping users set up industrial monitoring devices. Respond in {user_language}.

Current device settings: {json.dumps(device_data, indent=2)}
Setup stage: {setup_stage}

IMPORTANT: {stage_guidance}

When setting up devices:
- **Take action immediately** when user intent is clear
- Set smart defaults: device name from task description, "Detect" inference mode for most cases  
- Use **bold** formatting for key terms
- Only ask questions if truly necessary for setup
- Call functions proactively to make progress

Examples of immediate action:
- User says "detect defects": → Set name="Defect Detection", classes=["normal", "defect"], create icon
- User says "count people": → Set name="People Counter", classes=["person"], create icon
- User says "monitor safety": → Ask what specific safety aspect (PPE, restricted areas, etc.)

Available inference modes:
- Detect: Object detection with bounding boxes (use for most cases)
- Point: Find coordinates (for precise positioning tasks)
- VQA: Answer questions about images
- Caption: Generate descriptions

Never modify: status, connectedCameraId, last_heartbeat, iconAlreadyCreated, deletionStarted"""

def execute_function_call(func_name: str, func_args: dict, user_id: str, device_id: str, message_timestamp: str, call_index: int = 1) -> dict:
    """Execute a function call and return the result"""
    
    # Create unique timestamp for this action if there are multiple calls
    action_timestamp = f"{message_timestamp}_{call_index}" if call_index > 1 else message_timestamp
    
    if func_name == "createDeviceIcon":
        return process_create_device_icon(
            user_id, device_id, 
            func_args.get("iconRequest", ""), 
            action_timestamp
        )
    
    elif func_name == "setClasses":
        set_classes_url = "https://europe-west4-aimanagerfirebasebackend.cloudfunctions.net/set_classes"
        payload = {
            "user_id": user_id,
            "device_id": device_id,
            "classes": func_args.get("classes", []),
            "classDescriptions": func_args.get("classDescriptions", []),
            "messageTimestamp": action_timestamp
        }
        
        import requests
        response = requests.post(set_classes_url, json=payload)
        
        if response.status_code == 200:
            return {
                "status": "success",
                "message": f"Updated {len(func_args.get('classes', []))} classes",
                "timestamp": action_timestamp
            }
        else:
            return {
                "status": "error", 
                "message": f"Failed to update classes: {response.text}",
                "timestamp": action_timestamp
            }
    
    elif func_name == "updateSettings":
        success = update_device_settings(user_id, device_id, func_args.get("settings", {}))
        return {
            "status": "success" if success else "error",
            "message": "Settings updated" if success else "Failed to update settings",
            "timestamp": action_timestamp
        }
    
    else:
        return {
            "status": "error", 
            "message": f"Unknown function: {func_name}",
            "timestamp": action_timestamp
        }

@https_fn.on_request(region="europe-west4")
def assistant_chat(request: Request):
    """Simplified Gemini-based assistant chat"""
    
    if request.method == 'OPTIONS':
        response = jsonify({})
        response = add_cors_headers(response)
        return response, 204
    
    try:
        # Parse request
        request_json = request.get_json(silent=True)
        if not request_json:
            return add_cors_headers(jsonify({"error": "Invalid JSON payload"})), 400
        
        # Extract parameters
        prompt_type = request_json.get("prompt_type", "conversational")
        user_language = request_json.get("userLanguage", "english")
        message_timestamp = request_json.get("messageTimestamp", "none")
        user_message = request_json.get("message", "") or "ok"
        user_id = request_json.get("user_id")
        device_id = request_json.get("device_id")
        
        if not user_id or not device_id:
            return add_cors_headers(jsonify({"error": "Missing user_id or device_id"})), 400
        
        # Get device data
        db = firestore.client()
        device_doc = db.collection("users").document(user_id).collection("devices").document(device_id).get()
        
        device_data = device_doc.to_dict() if device_doc.exists else {}
        setup_stage = device_data.get("setupStage", 0)
        
        # Build system prompt
        system_prompt = build_system_prompt(device_data, setup_stage, user_language)
        
        # Handle different prompt types
        if prompt_type == "initial":
            if setup_stage == 0 and not device_data.get('name'):
                # Truly initial setup
                full_prompt = f"""{system_prompt}

This is the initial setup. Greet the user warmly and ask what they want to monitor. Be brief and welcoming.

USER MESSAGE: {user_message}"""
            else:
                # Returning to existing device
                full_prompt = f"""{system_prompt}

The user is returning to configure their device. Acknowledge the current setup and offer to help continue.

USER MESSAGE: {user_message}"""
        else:
            # Regular conversational prompt
            full_prompt = f"""{system_prompt}

USER MESSAGE: {user_message}

Respond to the user's message above. If their intent is clear, take immediate action with function calls."""
        
        # Prepare message content - start with the full prompt
        contents = [full_prompt]
        
        # Handle image attachments
        image_timestamps = request_json.get("imageTimestamps", [])
        if image_timestamps:
            storage_client = storage.bucket()
            
            for timestamp in image_timestamps:
                try:
                    image_path = f"users/{user_id}/devices/{device_id}/assistant/{timestamp}.jpg"
                    blob = storage_client.blob(image_path)
                    image_bytes = blob.download_as_bytes()
                    
                    # Convert to PIL Image for Gemini
                    img = Image.open(io.BytesIO(image_bytes))
                    contents.append(img)
                    logger.info(f"Added image {timestamp} to request")
                    
                except Exception as e:
                    logger.error(f"Error processing image {timestamp}: {str(e)}")
        
        # Create Gemini client and generate response
        client = genai.Client(api_key=get_gemini_api_key())
        
        response = client.models.generate_content(
            model='models/gemini-2.5-flash',
            contents=contents,
            config=types.GenerateContentConfig(
                tools=GEMINI_TOOLS,
                temperature=0.7,
                max_output_tokens=2048,
            )
        )
        
        # Process response
        assistant_reply = ""
        actions = []
        function_call_counter = 0
        
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'text') and part.text:
                assistant_reply += part.text
            elif hasattr(part, 'function_call') and part.function_call:
                function_call_counter += 1
                
                # Execute function call
                func_result = execute_function_call(
                    part.function_call.name,
                    dict(part.function_call.args),
                    user_id,
                    device_id,
                    message_timestamp,
                    function_call_counter
                )
                
                # Create action entry for frontend
                action_timestamp = func_result.get("timestamp", message_timestamp)
                actions.append({
                    "tool": part.function_call.name,
                    "timestamp": action_timestamp,
                    "result": func_result
                })
        
        # Build response
        result = {
            "assistant_reply": assistant_reply.strip(),
        }
        
        if actions:
            result["actions"] = actions
        
        return add_cors_headers(jsonify(result))
        
    except Exception as e:
        logger.error(f"Error in assistant_chat: {str(e)}")
        return add_cors_headers(jsonify({"error": str(e)})), 500

# Keep existing helper functions (these don't need to change)
# - process_create_device_icon()
# - update_device_settings() 
# - add_cors_headers()

# -------------------------------- ASSISTANT_CHAT END  --------------------------------






