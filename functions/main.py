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
from config.loader import get_openai_api_key, get_anthropic_api_key

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
        
        return add_cors_headers(jsonify(response_data)), 200

    except Exception as e:
        logger.error(f"Error in receive_image: {str(e)}")
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

# ---------------------------
# 1. Pydantic Model Definitions
# ---------------------------
class ActivelyAssistUser(BaseModel):
    messageToUser: str
    settingsToChange: list[str]
    newSettingValues: list[str]

class UpdateCommunicationProfile(BaseModel):
    last_conversation_summary: str
    non_device_preferences: str
    device_preferences: str


# ---------------------------
# 2. Batch Tool and Inference Modes
# ---------------------------

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
# 3. Tool Definitions (Single Source of Truth)
# ---------------------------

# Define all tools in a single location to avoid duplication
tools = [
    {
        "name": "createDeviceIcon",
        "description": "Generates a custom device icon. Provide 'iconRequest' with a description of the device task.",
        "input_schema": {
            "type": "object",
            "properties": {
                "iconRequest": {"type": "string", "description": "Description of the device task for icon generation."}
            },
            "required": ["iconRequest"]
        }
    },
    {
        "name": "setClasses",
        "description": "Defines classes and their descriptions for the device. (classDescriptions in ENGLISH!) ",
        "input_schema": {
            "type": "object",
            "properties": {
                "classes": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "List of class names (e.g., ['normal', 'defect'])."
                },
                "classDescriptions": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "List of descriptions for each class."
                }
            },
            "required": ["classes"]
        }
    },
    {
        "name": "batch_tool",
        "description": "Invoke multiple other tool calls simultaneously",
        "input_schema": {
            "type": "object",
            "properties": {
                "invocations": {
                    "type": "array",
                    "description": "The tool calls to invoke",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "The name of the tool to invoke"
                            },
                            "arguments": {
                                "type": "string",
                                "description": "The arguments to the tool as a JSON string"
                            }
                        },
                        "required": ["name", "arguments"]
                    }
                }
            },
            "required": ["invocations"]
        }
    },
    {
        "name": "responseAfterToolUse",
        "description": "Provide a response to be shown to the user after completing tool actions. ALWAYS use this when using the batch_tool to ensure a proper response in the user's language.",
        "input_schema": {
            "type": "object",
            "properties": {
                "response": {
                    "type": "string", 
                    "description": "The response text to show to the user in their language"
                }
            },
            "required": ["response"]
        }
    }
]

# --------------------------------------------------------
#    Stage-specific guidance
#    Return the text for whichever stage is >= the cutoff
# --------------------------------------------------------

def setupStagePromptSegment(setupStage: float) -> str:
    if setupStage >= 2:
        return (
            "Stage 2 - Operational\n"
            "• Device is fully configured and operational.\n"
            "• Help the user with any questions about usage.\n"
            "• Provide guidance on optimizing inference results.\n"
            "• Suggest improvements if needed.\n"
        )
    elif setupStage >= 1:
        return (
            "Stage 1 - Testing\n"
            "• Guide the user to test the inference mode.\n"
            "• Verify the device performs as expected.\n"
            "• Help troubleshoot any issues.\n"
            "• When testing is complete, add 'setupStage' to settingsToChange and set its value to 2.0\n"
        )
    else:
        return (
            "Stage 0 - Initial Setup\n"
            "• Welcome the user and understand their monitoring needs.\n"
            "• Set device name and task description by adding them to settingsToChange.\n"
            "• Make sure to proactively set a new Device Name if the current one is not meaningful.\n"
            "• Set an appropriate inference mode based on the task.\n"
            "• Help define classes and their descriptions.\n"
            "• Create a device icon once task is defined.\n"
            "• Actively Show progress by adding 'setupStage' to settingsToChange with values like 0.3, 0.5, 0.8.\n"
            "• When initial setup is complete, add 'setupStage' to settingsToChange and set its value to 1.0\n"
        )


# ---------------------------
# 4. Tool Processing Functions
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

def process_tool_call(tool_name, tool_input, user_id, device_id, messageTimestamp):
    """Process a single tool call"""
    if tool_name == "createDeviceIcon":
        icon_request = tool_input.get("iconRequest", "")
        return process_create_device_icon(user_id, device_id, icon_request, messageTimestamp)
    
    elif tool_name == "setClasses":
        classes = tool_input.get("classes", [])
        class_descriptions = tool_input.get("classDescriptions", None)
        
        # Call set_classes with the messageTimestamp
        set_classes_url = "https://europe-west4-aimanagerfirebasebackend.cloudfunctions.net/set_classes"
        payload = {
            "user_id": user_id,
            "device_id": device_id,
            "classes": classes,
            "classDescriptions": class_descriptions if class_descriptions else [],
            "messageTimestamp": messageTimestamp
        }
        
        response = requests.post(set_classes_url, json=payload)
        if response.status_code == 200:
            return {
                "status": "success",
                "message": f"Updated {len(classes)} classes",
                "classes": classes,
                "classDescriptions": class_descriptions if class_descriptions else []
            }
        else:
            return {
                "status": "error",
                "message": f"Failed to update classes: {response.text}",
                "status_code": response.status_code
            }
    
    else:
        return {"status": "error", "message": f"Unknown tool: {tool_name}"}


def process_batch_tool(invocations, user_id, device_id, messageTimestamp):
    """Process batch tool invocations"""
    results = []
    custom_response = None  # Store the responseAfterToolUse content
    
    for idx, invocation in enumerate(invocations):
        try:
            tool_name = invocation.get("name")
            arguments_str = invocation.get("arguments", "{}")
            
            if tool_name == "responseAfterToolUse":
                try:
                    arguments = json.loads(arguments_str)
                    custom_response = arguments.get("response", "")
                    logger.info(f"Found responseAfterToolUse tool with response: {custom_response}")
                    continue  # Skip normal processing for this special tool
                except json.JSONDecodeError:
                    logger.error(f"Error parsing responseAfterToolUse arguments: {arguments_str}")
                    continue

            # Parse arguments JSON string
            try:
                arguments = json.loads(arguments_str)
            except json.JSONDecodeError:
                arguments = {}
            
            # Create unique timestamp for each invocation
            invocation_timestamp = messageTimestamp
            if idx > 0:
                invocation_timestamp = f"{messageTimestamp}_{idx+1}"
            
            # Add messageTimestamp to arguments for functions that support it
            if tool_name in ["createDeviceIcon", "setClasses"]:
                # Pass the messageTimestamp to the tool function
                if tool_name == "createDeviceIcon":
                    # Add messageTimestamp to the payload
                    result = process_create_device_icon(user_id, device_id, arguments.get("iconRequest", ""), invocation_timestamp)
                elif tool_name == "setClasses":
                    # For setClasses, add messageTimestamp to the arguments
                    set_classes_url = "https://europe-west4-aimanagerfirebasebackend.cloudfunctions.net/set_classes"
                    payload = {
                        "user_id": user_id,
                        "device_id": device_id,
                        "classes": arguments.get("classes", []),
                        "classDescriptions": arguments.get("classDescriptions", []),
                        "messageTimestamp": invocation_timestamp
                    }
                    
                    # Call the set_classes function
                    response = requests.post(set_classes_url, json=payload)
                    if response.status_code == 200:
                        result = {
                            "status": "success",
                            "message": f"Updated {len(arguments.get('classes', []))} classes",
                            "classes": arguments.get("classes", []),
                            "classDescriptions": arguments.get("classDescriptions", [])
                        }
                    else:
                        result = {
                            "status": "error",
                            "message": f"Failed to update classes: {response.text}",
                            "status_code": response.status_code
                        }
            else:
                # Process other tools that don't need special timestamp handling
                result = process_tool_call(tool_name, arguments, user_id, device_id, invocation_timestamp)
            
            results.append({
                "tool": tool_name,
                "timestamp": invocation_timestamp,
                "result": result
            })
            
        except Exception as e:
            logger.error(f"Error processing invocation {idx}: {str(e)}")
            results.append({
                "tool": invocation.get("name", "unknown"),
                "timestamp": f"{messageTimestamp}_{idx+1}" if idx > 0 else messageTimestamp,
                "result": {"status": "error", "message": str(e)}
            })
    
    return results, custom_response

# ---------------------------
# 5. Firestore Settings Update Function
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
# 6. Prompt Building Functions
# ---------------------------
def build_prompt(prompt_type: str, available_settings: str, user_message: str, userLanguage: str = "english", setupStage: float = 0, conversation_history: list = []) -> str:
    IMAGE_GUIDANCE = """
I can analyze images to better understand your needs:
- Ask for photos of the physical setup
- Request examples of what you want to detect/classify
- Ask for images showing normal vs. abnormal conditions

When images would help, explicitly ask the user to upload images by clicking the image icon.
If user sends images, analyze them carefully to inform recommendations.
"""

    BATCH_TOOL_GUIDANCE = """
Use the batch_tool when you need to perform multiple actions at once. For example:
- Create a device icon AND set up classes
- Make any combination of tool calls that logically belong together

Format for batch_tool:
{
  "invocations": [
    {
      "name": "createDeviceIcon",
      "arguments": "{\"iconRequest\": \"description of device task\"}"
    },
    {
      "name": "setClasses",
      "arguments": "{\"classes\": [\"class1\", \"class2\"], \"classDescriptions\": [\"desc1\", \"desc2\"]}"
    },
    {
      "name": "responseAfterToolUse",
      "arguments": "{\"response\": \"I've set up your classes and created an icon specifically for your device. Next, let's test the detection.\"}"
    }
  ]
}

CRITICAL: ALWAYS include the responseAfterToolUse tool when using batch_tool to ensure a proper response in the user's language.
"""

    # Base prompt with common instructions
    base_prompt = f"""
You are an AI assistant helping users set up a device for industrial image-based monitoring.
Speak as if you were the AI itself - use direct, first-person language: "I can help you..." not "The device will now..."
So that the user feels like he is telling you what to detect or supervise. So talk to them like this for example : "Ok, I will detect ... for you."
Focus on practical applications, communicate in {userLanguage}, and guide the user through:

{SETUP_PROCESS_OVERVIEW}

{IMAGE_GUIDANCE}

Current setup stage: {setupStage}
Stage Task:
{setupStagePromptSegment(setupStage)}

Current device settings:
{available_settings}

Blacklisted settings (do not modify):
{SETTINGS_BLACKLIST}

Conversation history:
{json.dumps(conversation_history, indent=2) if conversation_history else "No prior conversation"}

Available Inference Modes:
{INFERENCE_MODES}

Response Format (JSON):
{{
    "messageToUser": <string>,
    "settingsToChange": <list of strings>,
    "newSettingValues": <list of strings>
}}

To help users quickly grasp key points in your responses, **highlight important words or phrases by surrounding them with double asterisks**.
For example: "I need to set up your **inference mode**." Highlight only the most important 1-3 segments in each message.
This formatting will be rendered as bold, colored text in the user interface.

IMPORTANT: To progress through setup stages, add 'setupStage' to settingsToChange and a new value to newSettingValues.
For example, to move from stage 0 to stage 1:
"settingsToChange": ["setupStage", "name"],
"newSettingValues": ["1.0", "My Device"]

Available Tools:
- createDeviceIcon: Generate a device icon based on task description
- setClasses: Define classes and descriptions (classDescriptions in ENGLISH!)
- batch_tool: Perform multiple actions at once

{BATCH_TOOL_GUIDANCE}

Never leave users wondering what to do next.
End messages with either a clear question or a call to action!
Perform non-critical tasks quietly in the background to speed up the setup process.

**Always** respond with a valid JSON object in a text block:
- Keep "messageToUser" concise and simple, providing only necessary information, moving the setup along and ALWAYS in the users language ( {userLanguage} ).
- Avoid technical details unless the user explicitly asks for them.
- Set setupStage values proactively to reflect real progress (e.g., x.5 for halfway through stage 0, x+1 for next stage).

"""

    # Type-specific segments
    if prompt_type == "initial":
        type_specific = f"""
Greet the user warmly and ask about their monitoring needs (e.g., "Hi! I'm here to help set up your device. What would you like me to monitor?").
If setupStage is larger than 0 and the device seems set up for a specific task already, continue assisting based on context.
Wait for clear intent before suggesting changes. Return a JSON object with a friendly greeting and no changes yet.
"""
    elif prompt_type == "conversational":
        type_specific = f"""
Respond to the user's message: {user_message}
Keep answers conversational and helpful (e.g., "I'll set up your device for defecting machine malfunctions right away" or "I need a bit more information about what you want to monitor").
Be proactive - when the user's intent is clear, suggest appropriate settings and take actions.

When helping with classes, always define classes clearly:
1. Suggest appropriate class names based on the user's task
2. Provide clear simple descriptions for each class that only get more specific if the user requires it eg. if the user just wants to distinguish jackets from something else then the description should be just "jackets", only become more specific when the user wants to do more finegrained differentiation.
3. Use the setClasses tool to save these settings
4. if the user just wants to detect one type of object/situation only one class is needed!

Always return a JSON object.
ABSOLUTELY ALWAYS include messageToUser, also when using tools!
To advance through setup stages, add 'setupStage' to settingsToChange with an appropriate value, especially when using tools.
"""
    else:
        raise ValueError("Invalid prompt_type")

    # Combine base and type-specific parts
    prompt = base_prompt + type_specific

    return prompt.strip()

def build_end_of_conversation_prompt(non_device_preferences: str, device_preferences: str, conversation_history: list = None) -> str:
    conv_history_text = ""
    if conversation_history:
        conv_history_text = "\nConversation History:\n" + "\n".join(
            [f"{msg.get('role')}: {msg.get('content')}" for msg in conversation_history]
        )
    prompt = f"""
Please summarize the conversation to give context for future interactions and adjust the preferences accordingly.

Non-device specific preferences:
{non_device_preferences}

Device specific preferences:
{device_preferences}

Conversation History:
{conv_history_text}

Instructions:
- Provide your response as valid JSON following the UpdateCommunicationProfile format:
  {{
    "last_conversation_summary": <string>,
    "non_device_preferences": <string>,
    "device_preferences": <string>
  }}
- **IMPORTANT**: Respond with only a valid JSON object matching the format above. Do not include any additional text, explanations, or commentary outside the JSON object.
"""
    return prompt.strip()


# ---------------------------
# 7. Main Firebase Function
# --------------------------- 

@https_fn.on_request(region="europe-west4")
def assistant_chat(request: Request):
    """
    Handles chat interaction with the AI assistant, supporting image attachments and various prompt types.
    Implements batch tooling for more efficient tool usage.
    """
    if request.method == 'OPTIONS':
        response = jsonify({})
        response = add_cors_headers(response)
        return response, 204
        
    db = firestore.client()
    max_tokens = 2048
    try:
        # Parse request data
        request_json = request.get_json(silent=True)
        if request_json is None:
            return add_cors_headers(jsonify({"error": "Invalid JSON payload"})), 400

        # Extract common parameters
        prompt_type = request_json.get("prompt_type", "conversational")
        userLanguage = request_json.get("userLanguage", "english")
        messageTimestamp = request_json.get("messageTimestamp", "none")
        user_message = request_json.get("message", "") or "ok"  # Default to "ok" if empty
        user_id = request_json.get("user_id")
        device_id = request_json.get("device_id")

        # Validate required parameters for non-endOfConversation prompts
        if prompt_type != "endOfConversation" and (not user_id or not device_id):
            return add_cors_headers(jsonify({"error": "Missing user_id or device_id"})), 400

        # Get device data and setup stage
        device_data = {}
        setupStage = 0
        
        if prompt_type in ["initial", "conversational"]:
            device_doc_ref = db.collection("users").document(user_id).collection("devices").document(device_id)
            device_doc = device_doc_ref.get()
            
            if device_doc.exists:
                device_data = device_doc.to_dict()
                setupStage = device_data.get("setupStage", 0)
                    
            # Build the appropriate prompt
            system_prompt = build_prompt(
                prompt_type, 
                json.dumps(device_data, indent=2) if device_data else "{}", 
                user_message, 
                userLanguage, 
                setupStage, 
                request_json.get("conversation_history", [])
            )

        elif prompt_type == "endOfConversation":
            # Handle end of conversation prompt
            user_doc = db.collection("users").document(user_id).get()
            non_device_preferences = user_doc.to_dict().get("non_device_preferences", "") if user_doc.exists else ""
            
            device_doc_ref = db.collection("users").document(user_id).collection("devices").document(device_id)
            device_doc = device_doc_ref.get()
            device_data = device_doc.to_dict() if device_doc.exists else {}
            device_preferences = device_data.get("device_preferences", "")
            
            system_prompt = build_end_of_conversation_prompt(
                non_device_preferences, 
                device_preferences, 
                request_json.get("conversation_history", [])
            )
            
        else:
            return add_cors_headers(jsonify({"error": "Invalid prompt_type"})), 400

        # Build message content for Anthropic API
        content_blocks = []
        
        # Process image attachments if present
        if prompt_type in ["initial", "conversational"]:
            image_timestamps = request_json.get("imageTimestamps", [])
            if image_timestamps:
                storage_client = storage.bucket()
                max_tokens = 2000 + 3000 * min(len(image_timestamps), 3) 
                
                for timestamp in image_timestamps:
                    try:
                        # Get the image from Firebase Storage
                        image_path = f"users/{user_id}/devices/{device_id}/assistant/{timestamp}.jpg"
                        blob = storage_client.blob(image_path)
                        
                        # Download the image as bytes
                        image_bytes = blob.download_as_bytes()
                        
                        # Convert to base64
                        base64_encoded = base64.b64encode(image_bytes).decode('utf-8')
                        
                        # Detect media type (assume JPEG if we can't determine)
                        media_type = "image/jpeg"
                        if image_path.lower().endswith('.png'):
                            media_type = "image/png"
                        elif image_path.lower().endswith('.gif'):
                            media_type = "image/gif"
                        
                        # Add image block to content
                        content_blocks.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_encoded
                            }
                        })
                        
                        logger.info(f"Successfully processed image {timestamp}")
                    
                    except Exception as e:
                        logger.error(f"Error processing image {timestamp}: {str(e)}")
        
        # Add text content
        content_blocks.append({
            "type": "text",
            "text": user_message
        })
        
        # Prepare messages for Anthropic
        messages = [
            {
                "role": "user", 
                "content": content_blocks
            }
        ]

        # Prepare API arguments
        api_kwargs = {
            "model": "claude-3-7-sonnet-20250219",
            "system": system_prompt, 
            "messages": messages,
            "max_tokens": max_tokens,
        }
        
        # Add tools for appropriate prompt types
        if prompt_type in ["initial", "conversational"]:
            api_kwargs["tools"] = tools
        
        # Log the API call details for debugging
        logger.info(f"Calling Anthropic API with {len(content_blocks)} content blocks")
        if content_blocks:
            logger.info(f"Content types: {[block['type'] for block in content_blocks]}")

        # Call Anthropic API
        completion = client.messages.create(**api_kwargs)

        # Process response
        text_content = "".join([block.text for block in completion.content if block.type == "text"])
        tool_uses = [block for block in completion.content if block.type == "tool_use"]

        # Parse JSON response
        if prompt_type != "endOfConversation":
            # Handle response for chat prompts
            json_match = re.search(r'\{.*\}', text_content, re.DOTALL)
            if not json_match:
                parsed_response = ActivelyAssistUser(
                    messageToUser="", settingsToChange=[], newSettingValues=[]
                )
            else:
                json_str = json_match.group()
                try:
                    response_json = json.loads(json_str)
                    # Ensure newSettingValues are strings
                    if "newSettingValues" in response_json:
                        response_json["newSettingValues"] = [str(val) for val in response_json["newSettingValues"]]
                    
                    parsed_response = ActivelyAssistUser(**response_json)
                except Exception as e:
                    return add_cors_headers(jsonify({"error": f"Failed to parse JSON: {str(e)}"})), 500
            
            # Handle settings updates
            update_success = None
            if parsed_response.settingsToChange and parsed_response.newSettingValues:
                if len(parsed_response.settingsToChange) == len(parsed_response.newSettingValues):
                    updates = dict(zip(parsed_response.settingsToChange, parsed_response.newSettingValues))
                    update_success = update_device_settings(user_id, device_id, updates)
                else:
                    update_success = False

            # Create base result
            result = {
                "assistant_reply": parsed_response.messageToUser,
                "device_settings_update": update_success if update_success is not None else None
            }

            # Process tool uses (including batch tool)
            if tool_uses:
                all_actions = []
                
                for tool_use in tool_uses:
                    tool_name = tool_use.name
                    tool_input = tool_use.input
                    
                    if tool_name == "batch_tool":
                        # Process batch tool invocations
                        batch_results, custom_response  = process_batch_tool(
                            tool_input.get("invocations", []),
                            user_id,
                            device_id,
                            messageTimestamp
                        )
                        
                        # Add all batch results to actions
                        all_actions.extend(batch_results)
                        
                    else:
                        # Process individual tool
                        action_result = process_tool_call(
                            tool_name,
                            tool_input,
                            user_id,
                            device_id,
                            messageTimestamp
                        )
                        
                        all_actions.append({
                            "tool": tool_name,
                            "timestamp": messageTimestamp,
                            "result": action_result
                        })
                
                # Create response with actions
                result = {
                    "actions": all_actions,
                    "assistant_reply": custom_response or parsed_response.messageToUser,
                }
                
                # Ensure we have a meaningful message to display
                if not result["assistant_reply"] or result["assistant_reply"].strip() == "":
                    # Generate a summary message based on the actions performed
                    action_summary = ""
                    
                    # Count actions by type
                    action_counts = {}
                    for action in all_actions:
                        tool = action.get("tool", "unknown")
                        action_counts[tool] = action_counts.get(tool, 0) + 1
                    
                    # Build a summary message
                    action_parts = []
                    for tool, count in action_counts.items():
                        if tool == "createDeviceIcon":
                            action_parts.append("created a device icon")
                        elif tool == "setClasses":
                            action_parts.append(f"set up {count} classes")
                        else:
                            action_parts.append(f"performed {tool}")
                    
                    if action_parts:
                        action_summary = "I've " + ", and ".join(action_parts) + " for you."
                    
                    result["assistant_reply"] = action_summary or "I've completed the requested actions."
            else:
                # Response without actions
                result = {"assistant_reply": parsed_response.messageToUser}

        else:
            # Handle endOfConversation response
            json_match = re.search(r'\{.*\}', text_content, re.DOTALL)
            if not json_match:
                return add_cors_headers(jsonify({"error": "Invalid response format"})), 500
                
            json_str = json_match.group()
            try:
                response_json = json.loads(json_str)
                parsed_response = UpdateCommunicationProfile(**response_json)
            except Exception as e:
                return add_cors_headers(jsonify({"error": f"Failed to parse JSON: {str(e)}"})), 500

            # Update user and device preferences
            user_doc_ref = db.collection("users").document(user_id)
            device_doc_ref = db.collection("users").document(user_id).collection("devices").document(device_id)
            
            user_doc_ref.set({"non_device_preferences": parsed_response.non_device_preferences}, merge=True)
            device_doc_ref.set({
                "device_preferences": parsed_response.device_preferences,
                "last_conversation_summary": parsed_response.last_conversation_summary
            }, merge=True)
            
            result = {
                "assistant_reply": parsed_response.last_conversation_summary,
                "non_device_preferences": parsed_response.non_device_preferences,
                "device_preferences": parsed_response.device_preferences,
                "preferences_saved": True
            }

        return add_cors_headers(jsonify(result))

    except Exception as e:
        logger.error(f"Error in assistant_chat: {str(e)}")
        return add_cors_headers(jsonify({"error": str(e)})), 500

# -------------------------------- ASSISTANT_CHAT END  --------------------------------






