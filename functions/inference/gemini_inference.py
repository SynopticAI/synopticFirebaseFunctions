# inference/gemini_inference.py

import os
from google import genai
from google.genai import types
from firebase_functions import logger
import json
from PIL import Image
import io
from config.loader import get_gemini_api_key

def run_gemini_inference(image_data, device_config):
    """
    Main function to run Gemini inference using Google's native object detection capabilities.
    Uses Google's recommended [ymin, xmin, ymax, xmax] format normalized to [0, 1000].
    Now leverages Gemini 2.5's built-in object detection training for better accuracy.
    """
    classes = device_config.get('classes', [])
    class_descriptions = device_config.get('classDescriptions', [])
    
    if not classes:
        return {"error": "No classes defined for Gemini inference."}

    # Build the class list for the prompt
    class_prompt_list = ""
    for i, class_name in enumerate(classes):
        description = class_descriptions[i].strip() if i < len(class_descriptions) and class_descriptions[i].strip() else class_name
        class_prompt_list += f"- {class_name}: {description}\n"

    # Use Google's native object detection approach - simpler prompt that lets Gemini use its natural format
    prompt = f"""
Detect all instances of the following objects in the image:

{class_prompt_list}

For each detection, provide the object name from the list above and its bounding box coordinates. The box_2d should be [ymin, xmin, ymax, xmax] normalized to 0-1000.

Return as a JSON array. If no objects are found, return an empty array.
"""

    try:
        # Get API key from config
        api_key = get_gemini_api_key()
        if not api_key:
            raise ValueError("Gemini API key not found in configuration")
        
        client = genai.Client(api_key=api_key)
        img = Image.open(io.BytesIO(image_data))
        
        # Use Google's recommended approach: image first, then text
        response = client.models.generate_content(
            model='models/gemini-2.5-flash-lite',
            # model='models/gemini-2.5-flash',
            # model='models/gemini-2.5-pro',
            contents=[img, prompt],  # Image before text as recommended by Google
            config=types.GenerateContentConfig(
                response_mime_type="application/json",  # Force structured JSON response
                thinking_config=types.ThinkingConfig(thinking_budget=-1)  # Try 0 instead of -1
            )
        )

        # Calculate credit usage based on token count
        credit_usage = 0
        try:
            usage_metadata = getattr(response, 'usage_metadata', None)
            if usage_metadata:
                total_token_count = getattr(usage_metadata, 'total_token_count', 0)
                credit_usage = round(0.3 * total_token_count, 2)
                logger.info(f"Gemini token usage: {total_token_count}, credit cost: {credit_usage}")
            else:
                # Fallback: estimate based on prompt and response length
                prompt_tokens = len(prompt.split()) * 1.3
                response_tokens = len(response.text.split()) * 1.3
                estimated_tokens = prompt_tokens + response_tokens
                credit_usage = round(0.3 * estimated_tokens, 2)
                logger.info(f"Using estimated token count: {estimated_tokens}, credit cost: {credit_usage}")
        except Exception as token_error:
            logger.error(f"Error calculating token usage: {token_error}")
            credit_usage = 15.0  # Reasonable fallback

        # Parse JSON response (should be much cleaner with native JSON response mode)
        logger.info(f"Raw Gemini response: {response.text[:500]}...")
        
        try:
            detections = json.loads(response.text)
        except json.JSONDecodeError as json_error:
            logger.error(f"JSON parsing failed: {json_error}, raw response: {response.text}")
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

        # Debug: Log the structure of the first detection to understand the format
        if detections:
            logger.info(f"First detection structure: {json.dumps(detections[0], indent=2)}")
            logger.info(f"Available fields in first detection: {list(detections[0].keys())}")

        # Convert from Google's native format to our format
        formatted_objects = []
        for i, det in enumerate(detections):
            try:
                # Try multiple possible field names for class
                class_name = None
                for class_field in ['class_name', 'label', 'name', 'object', 'class', 'category']:
                    if class_field in det:
                        class_name = det[class_field]
                        break
                
                if class_name is None:
                    logger.info(f"Detection {i} has no recognizable class field, available fields: {list(det.keys())}")
                    continue

                # Try multiple possible field names for confidence
                confidence = None
                for conf_field in ['confidence', 'score', 'confidence_score', 'probability']:
                    if conf_field in det:
                        confidence = float(det[conf_field])
                        break
                
                if confidence is None:
                    logger.info(f"Detection {i} has no recognizable confidence field, using default 0.8")
                    confidence = 0.8  # Default confidence if not provided

                # Try to get bounding box - this should be more consistent
                box_2d = det.get('box_2d', det.get('bbox', det.get('bounding_box', [])))
                
                if not box_2d:
                    logger.info(f"Detection {i} has no bounding box data, available fields: {list(det.keys())}")
                    continue

                # Validate box_2d format - should be [ymin, xmin, ymax, xmax]
                if not isinstance(box_2d, list) or len(box_2d) != 4:
                    logger.info(f"Invalid box_2d format for detection {i}: {box_2d}, skipping")
                    continue

                # Extract coordinates: [ymin, xmin, ymax, xmax] in [0, 1000] as per Google's format
                ymin, xmin, ymax, xmax = [float(coord) for coord in box_2d]

                # Validate coordinate ranges should be [0, 1000]
                if not all(0 <= coord <= 1000 for coord in [ymin, xmin, ymax, xmax]):
                    logger.info(f"Coordinates out of [0,1000] range for detection {i}: {box_2d}, clamping")
                    ymin = max(0, min(ymin, 1000))
                    xmin = max(0, min(xmin, 1000))
                    ymax = max(0, min(ymax, 1000))
                    xmax = max(0, min(xmax, 1000))

                # Ensure min <= max (basic sanity check)
                if ymin >= ymax or xmin >= xmax:
                    logger.info(f"Invalid box coordinates for detection {i}: ymin={ymin}, ymax={ymax}, xmin={xmin}, xmax={xmax}, skipping")
                    continue

                # Validate confidence score
                if not (0.0 <= confidence <= 1.0):
                    logger.info(f"Invalid confidence score for detection {i}: {confidence}, clamping to [0,1]")
                    confidence = max(0.0, min(1.0, confidence))

                # Convert from Google's [0, 1000] to [0, 1] normalized coordinates
                ymin_norm = ymin / 1000.0
                xmin_norm = xmin / 1000.0
                ymax_norm = ymax / 1000.0
                xmax_norm = xmax / 1000.0

                # Convert to our expected 1024x1024 pixel space format (for backward compatibility)
                x1 = max(0, min(int(xmin_norm * 1024), 1023))
                y1 = max(0, min(int(ymin_norm * 1024), 1023))
                x2 = max(0, min(int(xmax_norm * 1024), 1023))
                y2 = max(0, min(int(ymax_norm * 1024), 1023))

                # Final validation - ensure we have a valid box
                if x1 >= x2 or y1 >= y2:
                    logger.info(f"Invalid final box coordinates for detection {i}: ({x1},{y1},{x2},{y2}), skipping")
                    continue

                formatted_objects.append({
                    'class': class_name,
                    'score': confidence,
                    'bbox': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
                })
                
                logger.info(f"Successfully processed detection {i}: {class_name} at ({x1},{y1},{x2},{y2}) with score {confidence:.2f}")
                
            except (ValueError, TypeError) as e:
                logger.error(f"Error processing detection {i}: {e}, detection data: {det}")
                continue
        
        logger.info(f"Gemini native object detection found {len(formatted_objects)} valid objects out of {len(detections)} detections.")
        
        return {
            'objects': formatted_objects,
            'credit_usage': credit_usage
        }

    except Exception as e:
        logger.error(f"Error in Gemini native object detection pipeline: {str(e)}")
        return {
            "error": str(e),
            "objects": [],
            "credit_usage": 15.0  # Fallback credit cost
        }