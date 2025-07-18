# inference/moondream_inference.py

import concurrent.futures
from firebase_functions import logger
from PIL import Image
import io
import moondream as md
from config.loader import get_moondream_api_key

# Fixed credit cost for Moondream usage
MOONDREAM_CREDIT_COST = 20

def preprocess_image(image_data):
    """Prepares the image as a PIL.Image object for the moondream library."""
    try:
        return Image.open(io.BytesIO(image_data))
    except Exception as e:
        logger.error(f"Error creating PIL.Image for Moondream: {str(e)}")
        raise

def call_moondream_api(pil_image, inference_mode, text_input=None):
    """
    Handles the API request using the official moondream library.
    """
    try:
        # Get API key from config
        api_key = get_moondream_api_key()
        if not api_key:
            raise ValueError("Moondream API key not found in configuration")
        
        # Initialize the cloud model with the API key
        model = md.vl(api_key=api_key)
        
        # Map our inference modes to the library's method names
        inference_type_map = {
            "Detect": "detect",
            "Point": "point",
            "VQA": "query",
            "Caption": "caption"
        }
        inference_method_name = inference_type_map.get(inference_mode, "detect")

        logger.info(f"Calling Moondream Cloud API with method: {inference_method_name}, input: '{text_input}'")

        # Call the appropriate method on the model object
        if inference_method_name == "detect":
            result = model.detect(pil_image, text_input or "object")
        elif inference_method_name == "point":
            result = model.point(pil_image, text_input or "object")
        elif inference_method_name == "query":
            result = model.query(pil_image, text_input)
        elif inference_method_name == "caption":
            result = model.caption(pil_image)
        else:
            return {"error": f"Unknown inference method: {inference_method_name}"}
        
        # Return the result in the same format as the original code
        return {"result": result}

    except Exception as e:
        logger.error(f"Error during Moondream Cloud API call: {str(e)}")
        return {"error": str(e)}

def run_moondream_inference(image_data, device_config):
    """
    Main function to run the full Moondream inference pipeline using the official cloud API.
    Now includes fixed credit usage calculation.
    """
    inference_mode = device_config.get('inferenceMode', 'Detect')
    classes = device_config.get('classes', [])
    class_descriptions = device_config.get('classDescriptions', [])
    
    if not classes:
        return {"error": "No classes defined for Moondream inference."}

    # Preprocess image once to a PIL.Image object
    pil_image = preprocess_image(image_data)
    
    # Ensure class_descriptions has same length as classes
    while len(class_descriptions) < len(classes):
        class_descriptions.append("")
    
    # Results containers
    all_results = []
    class_results = []
    
    # Create prompts using descriptions when available, otherwise use class names
    inference_prompts = []
    for i, class_name in enumerate(classes):
        if i < len(class_descriptions) and class_descriptions[i] and class_descriptions[i].strip():
            # Use the class description if available
            inference_prompts.append((class_name, class_descriptions[i].strip()))
        else:
            # Fall back to class name if no description
            inference_prompts.append((class_name, class_name))
    
    # Log which prompts we're using
    for class_name, prompt in inference_prompts:
        logger.info(f"Using prompt for class '{class_name}': '{prompt}'")
    
    # Calculate credit usage: fixed cost per inference, regardless of number of classes
    # This is because all classes are processed in a single inference session
    credit_usage = MOONDREAM_CREDIT_COST
    logger.info(f"Moondream credit usage: {credit_usage} (fixed cost)")
    
    # Process each class in parallel (using max_workers=1 for cloud API to avoid memory issues)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future_to_class = {
            executor.submit(call_moondream_api, pil_image, inference_mode, prompt): class_name
            for class_name, prompt in inference_prompts
        }
        
        for future in concurrent.futures.as_completed(future_to_class):
            class_name = future_to_class[future]
            try:
                result = future.result()
                # Check for errors in the result
                if "error" in result:
                    logger.error(f"Error from Moondream API for class {class_name}: {result['error']}")
                    class_results.append({
                        'class': class_name,
                        'error': result['error']
                    })
                    continue
                
                # Add class name to result
                result['class_name'] = class_name
                class_results.append({
                    'class': class_name,
                    'result': result
                })
                all_results.append(result)
            except Exception as e:
                logger.error(f"Error processing class {class_name}: {e}")
                class_results.append({
                    'class': class_name,
                    'error': str(e)
                })
    
    # Merge the results based on inference mode (following original logic)
    merged_result = {
        'all_class_results': class_results,
        'credit_usage': credit_usage  # Add credit usage to result
    }
    
    # For Point mode, extract all points with their class
    if inference_mode == 'Point':
        all_points = []
        for i, result in enumerate(all_results):
            # Safely extract points from result
            api_result = result.get('result', [])
            
            # Handle different possible return types from the API
            points = []
            if isinstance(api_result, list):
                # Direct list of points
                points = api_result
            elif isinstance(api_result, dict):
                # Dict with 'points' key
                points = api_result.get('points', [])
            elif isinstance(api_result, str):
                # String response (like "No points detected")
                logger.info(f"Moondream returned string for class {classes[i] if i < len(classes) else 'Unknown'}: {api_result}")
                continue
            
            class_name = classes[i] if i < len(classes) else 'Unknown'
            
            for point in points:
                # Ensure point is a dict before trying to access its properties
                if not isinstance(point, dict):
                    continue
                    
                # Ensure coordinates are correctly formatted
                x = point.get('x', 0.5)
                y = point.get('y', 0.5)
                
                # Create standardized point format
                new_point = {
                    'x': float(x),  # Keep normalized for frontend scaling
                    'y': float(y),  # Keep normalized for frontend scaling
                    'class': class_name,
                    'score': point.get('score', 0.9)
                }
                
                all_points.append(new_point)
        
        merged_result['points'] = all_points
    
    # For Detect mode, extract all objects with their class
    elif inference_mode == 'Detect':
        all_objects = []
        for i, result in enumerate(all_results):
            # Safely extract objects from result
            api_result = result.get('result', [])
            
            # Handle different possible return types from the API
            objects = []
            if isinstance(api_result, list):
                # Direct list of objects
                objects = api_result
            elif isinstance(api_result, dict):
                # Dict with 'objects' key
                objects = api_result.get('objects', [])
            elif isinstance(api_result, str):
                # String response (like "No objects detected")
                logger.info(f"Moondream returned string for class {classes[i] if i < len(classes) else 'Unknown'}: {api_result}")
                continue
            
            class_name = classes[i] if i < len(classes) else 'Unknown'
            
            for obj in objects:
                # Ensure obj is a dict before trying to assign to it
                if not isinstance(obj, dict):
                    continue
                    
                # Extract normalized coordinates (0-1) - handle multiple possible key names
                x_min = float(obj.get('x_min', obj.get('left', obj.get('x1', 0))))
                y_min = float(obj.get('y_min', obj.get('top', obj.get('y1', 0))))
                x_max = float(obj.get('x_max', obj.get('right', obj.get('x2', 1))))
                y_max = float(obj.get('y_max', obj.get('bottom', obj.get('y2', 1))))
                
                # Convert to pixel coordinates (0-1024)
                # Make sure coordinates are in the correct range
                x1 = max(0, min(int(x_min * 1024), 1023))
                y1 = max(0, min(int(y_min * 1024), 1023))
                x2 = max(0, min(int(x_max * 1024), 1023))
                y2 = max(0, min(int(y_max * 1024), 1023))
                
                # Ensure x1 <= x2 and y1 <= y2
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1
                
                # Create a standardized object format
                new_obj = {
                    'class': class_name,
                    'score': obj.get('score', 0.9),
                    'bbox': {
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2
                    }
                }
                
                all_objects.append(new_obj)
        
        merged_result['objects'] = all_objects
    
    # For other modes (VQA, Caption), return the result directly
    else:
        if all_results:
            merged_result['result'] = all_results[0].get('result', {})
    
    return merged_result