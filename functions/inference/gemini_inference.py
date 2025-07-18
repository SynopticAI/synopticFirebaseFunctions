# inference/gemini_inference.py

import os
from google import genai
from google.genai import types
from firebase_functions import logger
import json
import re
from PIL import Image
import io
from config.loader import get_gemini_api_key

def run_gemini_inference(image_data, device_config):
    """
    Main function to run Gemini inference for all classes at once.
    Now includes credit usage calculation based on token count and improved coordinate handling.
    """
    classes = device_config.get('classes', [])
    class_descriptions = device_config.get('classDescriptions', [])
    
    if not classes:
        return {"error": "No classes defined for Gemini inference."}

    # Build the class list for the prompt
    class_prompt_list = ""
    for i, class_name in enumerate(classes):
        description = class_descriptions[i].strip() if i < len(class_descriptions) and class_descriptions[i].strip() else class_name
        class_prompt_list += f"- class_name: \"{class_name}\", description: \"{description}\"\n"

    # Updated prompt to request center coordinates + width/height format
    prompt = f"""
    You are a computer vision expert. Analyze the image and detect all instances of the following objects/situations.

    OBJECTS TO DETECT:
    {class_prompt_list}

    Your response MUST be a single raw JSON list of all detected objects.
    Each object in the list must be a dictionary with these exact keys:
    1. "class_name": The string name of the detected class.
    2. "score": A confidence score (float between 0.0 and 1.0).
    3. "center_x": Normalized x-coordinate of the object's center (float between 0.0 and 1.0).
    4. "center_y": Normalized y-coordinate of the object's center (float between 0.0 and 1.0).
    5. "width": Normalized width of the bounding box (float between 0.0 and 1.0).
    6. "height": Normalized height of the bounding box (float between 0.0 and 1.0).

    Important coordinate guidelines:
    - All coordinates must be normalized (between 0.0 and 1.0)
    - center_x and center_y represent the exact center of the object
    - width and height represent the full dimensions of the bounding box
    - Ensure objects are clearly visible and confidence scores are realistic

    Example response:
    [
      {{
        "class_name": "car",
        "score": 0.92,
        "center_x": 0.3,
        "center_y": 0.25,
        "width": 0.2,
        "height": 0.15
      }},
      {{
        "class_name": "person", 
        "score": 0.88,
        "center_x": 0.65,
        "center_y": 0.4,
        "width": 0.1,
        "height": 0.3
      }}
    ]

    If you find no objects, return an empty list: [].
    Only return valid JSON. Do not include any explanations or markdown formatting.
    """

    try:
        # Get API key from config
        api_key = get_gemini_api_key()
        if not api_key:
            raise ValueError("Gemini API key not found in configuration")
        
        client = genai.Client(api_key=api_key)
        img = Image.open(io.BytesIO(image_data))
        
        response = client.models.generate_content(
            model='models/gemini-2.5-flash-lite-preview-06-17',
            contents=[prompt, img],
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=-1) # -1 enables dynamic thinking
            )
        )

        # Calculate credit usage based on token count
        credit_usage = 0
        try:
            # Get usage metadata from response
            usage_metadata = getattr(response, 'usage_metadata', None)
            if usage_metadata:
                total_token_count = getattr(usage_metadata, 'total_token_count', 0)
                credit_usage = round(0.3 * total_token_count, 2)
                logger.info(f"Gemini token usage: {total_token_count}, credit cost: {credit_usage}")
            else:
                # Fallback: estimate based on prompt and response length
                prompt_tokens = len(prompt.split()) * 1.3  # rough estimation
                response_tokens = len(response.text.split()) * 1.3
                estimated_tokens = prompt_tokens + response_tokens
                credit_usage = round(0.3 * estimated_tokens, 2)
                logger.warning(f"Using estimated token count: {estimated_tokens}, credit cost: {credit_usage}")
        except Exception as token_error:
            logger.error(f"Error calculating token usage: {token_error}")
            # Fallback to a fixed cost if token counting fails
            credit_usage = 15.0  # Reasonable fallback for Gemini calls

        # Improved JSON parsing with better error handling
        raw_response = response.text.strip()
        logger.info(f"Raw Gemini response: {raw_response[:200]}...")  # Log first 200 chars for debugging
        
        # Clean up the response text more thoroughly
        # Remove markdown code blocks
        raw_response = re.sub(r'```(?:json)?\s*', '', raw_response)
        raw_response = re.sub(r'```\s*$', '', raw_response)
        
        # Remove any leading/trailing text that's not JSON
        # Find the first '[' and last ']' to extract just the JSON array
        start_idx = raw_response.find('[')
        end_idx = raw_response.rfind(']')
        
        if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
            logger.error(f"No valid JSON array found in response: {raw_response}")
            return {
                'objects': [],
                'credit_usage': credit_usage,
                'error': 'Invalid JSON response format'
            }
        
        json_str = raw_response[start_idx:end_idx + 1]
        
        try:
            detections = json.loads(json_str)
        except json.JSONDecodeError as json_error:
            logger.error(f"JSON parsing failed: {json_error}, raw JSON: {json_str}")
            return {
                'objects': [],
                'credit_usage': credit_usage,
                'error': f'JSON parsing error: {str(json_error)}'
            }

        if not isinstance(detections, list):
            logger.error(f"Expected list, got {type(detections)}: {detections}")
            return {
                'objects': [],
                'credit_usage': credit_usage,
                'error': 'Response is not a JSON array'
            }

        # --- TRANSFORMATION FROM CENTER+WH TO CORNER COORDINATES ---
        formatted_objects = []
        for i, det in enumerate(detections):
            try:
                # Validate required fields
                required_fields = ['class_name', 'score', 'center_x', 'center_y', 'width', 'height']
                missing_fields = [field for field in required_fields if field not in det]
                if missing_fields:
                    logger.warning(f"Detection {i} missing fields: {missing_fields}, skipping")
                    continue

                # Extract and validate center coordinates and dimensions
                center_x = float(det.get('center_x', 0.5))
                center_y = float(det.get('center_y', 0.5))
                width = float(det.get('width', 0.1))
                height = float(det.get('height', 0.1))
                score = float(det.get('score', 0.0))
                class_name = det.get('class_name', 'unknown')

                # Validate coordinate ranges
                if not (0.0 <= center_x <= 1.0 and 0.0 <= center_y <= 1.0):
                    logger.warning(f"Invalid center coordinates for detection {i}: ({center_x}, {center_y}), skipping")
                    continue
                
                if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
                    logger.warning(f"Invalid dimensions for detection {i}: {width}x{height}, skipping")
                    continue

                if not (0.0 <= score <= 1.0):
                    logger.warning(f"Invalid confidence score for detection {i}: {score}, clamping to [0,1]")
                    score = max(0.0, min(1.0, score))

                # Convert center+wh to corner coordinates
                half_width = width / 2.0
                half_height = height / 2.0
                
                x_min = center_x - half_width
                y_min = center_y - half_height
                x_max = center_x + half_width
                y_max = center_y + half_height
                
                # Clamp to valid normalized range
                x_min = max(0.0, min(x_min, 1.0))
                y_min = max(0.0, min(y_min, 1.0))
                x_max = max(0.0, min(x_max, 1.0))
                y_max = max(0.0, min(y_max, 1.0))
                
                # Ensure min <= max (handle edge cases)
                if x_min >= x_max:
                    x_min = max(0.0, center_x - 0.05)
                    x_max = min(1.0, center_x + 0.05)
                if y_min >= y_max:
                    y_min = max(0.0, center_y - 0.05)
                    y_max = min(1.0, center_y + 0.05)

                # Scale to 1024x1024 pixel space for backward compatibility
                x1 = max(0, min(int(x_min * 1024), 1023))
                y1 = max(0, min(int(y_min * 1024), 1023))
                x2 = max(0, min(int(x_max * 1024), 1023))
                y2 = max(0, min(int(y_max * 1024), 1023))

                # Final validation - ensure we have a valid box
                if x1 >= x2 or y1 >= y2:
                    logger.warning(f"Invalid final box coordinates for detection {i}: ({x1},{y1},{x2},{y2}), skipping")
                    continue

                formatted_objects.append({
                    'class': class_name,
                    'score': score,
                    'bbox': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
                })
                
                logger.info(f"Successfully processed detection {i}: {class_name} at ({x1},{y1},{x2},{y2}) with score {score:.2f}")
                
            except (ValueError, TypeError) as e:
                logger.error(f"Error processing detection {i}: {e}, detection data: {det}")
                continue
        
        logger.info(f"Gemini found and formatted {len(formatted_objects)} valid objects out of {len(detections)} detections.")
        
        return {
            'objects': formatted_objects,
            'credit_usage': credit_usage
        }

    except Exception as e:
        logger.error(f"Error in Gemini inference pipeline: {str(e)}")
        return {
            "error": str(e),
            "objects": [],
            "credit_usage": 15.0  # Fallback credit cost for error cases
        }