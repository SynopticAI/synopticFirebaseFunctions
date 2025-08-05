from PIL import Image, ImageDraw, ImageFont
import io
import math
from typing import List, Dict, Any, Tuple, Optional
from firebase_functions import logger


class ImageOverlayProcessor:
    """
    Handles drawing bounding boxes, points, and labels on images.
    Ports the Flutter BoundingBoxPainter and PointPainter logic to Python PIL.
    """
    
    # Class colors matching the Flutter implementation
    DEFAULT_COLORS = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green  
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta/Purple
        (255, 165, 0),  # Orange
        (0, 255, 255),  # Cyan
        (128, 0, 128),  # Purple
        (255, 192, 203), # Pink
        (0, 128, 0),    # Dark Green
        (255, 20, 147), # Deep Pink
        (0, 191, 255),  # Deep Sky Blue
    ]
    
    @staticmethod
    def get_class_color(class_name: str, class_color_map: Dict[str, Tuple[int, int, int]]) -> Tuple[int, int, int]:
        """Get consistent color for a class name using hash-based selection."""
        if class_name in class_color_map:
            return class_color_map[class_name]
        
        # Assign new color based on hash of class name for consistency
        color_index = abs(hash(class_name)) % len(ImageOverlayProcessor.DEFAULT_COLORS)
        color = ImageOverlayProcessor.DEFAULT_COLORS[color_index]
        class_color_map[class_name] = color
        return color
    
    @staticmethod
    def load_font(size: int = 14) -> ImageFont.ImageFont:
        """Load the best available font for text rendering."""
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 
            "/System/Library/Fonts/Arial.ttf",  # macOS
            "C:/Windows/Fonts/arial.ttf",      # Windows
        ]
        
        for font_path in font_paths:
            try:
                return ImageFont.truetype(font_path, size)
            except (OSError, IOError):
                continue
        
        # Fallback to default font
        try:
            return ImageFont.load_default()
        except:
            return None
    
    @staticmethod
    def draw_bounding_boxes(image_bytes: bytes, objects: List[Dict[str, Any]], 
                           image_size: Tuple[int, int] = (1024, 1024)) -> bytes:
        """
        Draw bounding boxes on image matching the Flutter BoundingBoxPainter implementation.
        
        Args:
            image_bytes: Original image as bytes
            objects: List of detected objects with bbox, class, and score
            image_size: Coordinate system size (default 1024x1024)
            
        Returns:
            Image bytes with bounding boxes drawn
        """
        try:
            if not objects:
                logger.info("No objects to draw, returning original image")
                return image_bytes
                
            # Open and prepare image
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            original_size = image.size
            draw = ImageDraw.Draw(image)
            
            # Load font
            font = ImageOverlayProcessor.load_font(14)
            small_font = ImageOverlayProcessor.load_font(12)
            
            # Class color mapping for consistency
            class_color_map = {}
            
            logger.info(f"Drawing {len(objects)} bounding boxes on {original_size} image")
            
            for i, obj in enumerate(objects):
                try:
                    bbox = obj.get('bbox', {})
                    class_name = obj.get('class', f'Object_{i}')
                    score = obj.get('score', 0.0)
                    
                    # Get coordinates (expected to be in image_size coordinate system)
                    x1 = float(bbox.get('x1', 0))
                    y1 = float(bbox.get('y1', 0))
                    x2 = float(bbox.get('x2', 0))
                    y2 = float(bbox.get('y2', 0))
                    
                    # Validate coordinates
                    if x1 >= x2 or y1 >= y2:
                        logger.info(f"Invalid bounding box coordinates for {class_name}: ({x1},{y1}) to ({x2},{y2})")
                        continue
                    
                    # Scale coordinates to actual image size
                    # Use minimum scaling factor to preserve aspect ratio
                    scale_factor = min(
                        original_size[0] / image_size[0], 
                        original_size[1] / image_size[1]
                    )
                    
                    # Center the content on the image
                    x_offset = (original_size[0] - image_size[0] * scale_factor) / 2
                    y_offset = (original_size[1] - image_size[1] * scale_factor) / 2
                    
                    # Apply scaling and centering
                    scaled_x1 = x1 * scale_factor + x_offset
                    scaled_y1 = y1 * scale_factor + y_offset
                    scaled_x2 = x2 * scale_factor + x_offset
                    scaled_y2 = y2 * scale_factor + y_offset
                    
                    # Ensure coordinates are within image bounds
                    scaled_x1 = max(0, min(original_size[0], scaled_x1))
                    scaled_y1 = max(0, min(original_size[1], scaled_y1))
                    scaled_x2 = max(0, min(original_size[0], scaled_x2))
                    scaled_y2 = max(0, min(original_size[1], scaled_y2))
                    
                    # Get color for this class
                    color = ImageOverlayProcessor.get_class_color(class_name, class_color_map)
                    
                    # Draw bounding box rectangle with thick outline
                    draw.rectangle(
                        [(scaled_x1, scaled_y1), (scaled_x2, scaled_y2)],
                        outline=color,
                        width=3
                    )
                    
                    # Prepare label text
                    label_text = str(class_name)
                    
                    # Get text dimensions
                    if font:
                        bbox_text = draw.textbbox((0, 0), label_text, font=font)
                        text_width = bbox_text[2] - bbox_text[0]
                        text_height = bbox_text[3] - bbox_text[1]
                    else:
                        # Fallback text size estimation
                        text_width = len(label_text) * 8
                        text_height = 14
                    
                    # Ensure label fits within image
                    label_y = max(text_height + 6, scaled_y1 - 2)
                    
                    # Draw label background rectangle
                    label_bg_color = (*color, 179)  # 70% opacity
                    draw.rectangle(
                        [(scaled_x1, label_y - text_height - 6), 
                         (scaled_x1 + text_width + 10, label_y - 2)],
                        fill=color,  # PIL doesn't support alpha in fill, use solid color
                        outline=None
                    )
                    
                    # Draw class label text
                    draw.text(
                        (scaled_x1 + 5, label_y - text_height - 4),
                        label_text,
                        fill=(255, 255, 255),  # White text
                        font=font
                    )
                    
                    # Draw confidence score if available
                    if score > 0:
                        score_text = f"{int(score * 100)}%"
                        
                        if small_font:
                            score_bbox = draw.textbbox((0, 0), score_text, font=small_font)
                            score_width = score_bbox[2] - score_bbox[0]
                            score_height = score_bbox[3] - score_bbox[1]
                        else:
                            score_width = len(score_text) * 6
                            score_height = 12
                        
                        # Position score in bottom-right of bounding box
                        score_x = scaled_x2 - score_width - 5
                        score_y = scaled_y2 - score_height - 2
                        
                        # Draw score background
                        draw.rectangle(
                            [(score_x - 3, score_y - 2),
                             (score_x + score_width + 3, score_y + score_height + 2)],
                            fill=color,
                            outline=None
                        )
                        
                        # Draw score text
                        draw.text(
                            (score_x, score_y),
                            score_text,
                            fill=(255, 255, 255),
                            font=small_font
                        )
                    
                except Exception as obj_error:
                    logger.error(f"Error drawing bounding box for object {i}: {str(obj_error)}")
                    continue
            
            # Convert back to bytes
            output_buffer = io.BytesIO()
            image.save(output_buffer, format='JPEG', quality=95, optimize=True)
            
            logger.info(f"Successfully drew bounding boxes on image")
            return output_buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Error drawing bounding boxes: {str(e)}")
            return image_bytes  # Return original image on error
    
    @staticmethod
    def draw_points(image_bytes: bytes, points: List[Dict[str, Any]], 
                   pulse_factor: float = 1.0) -> bytes:
        """
        Draw detection points on image matching the Flutter PointPainter implementation.
        
        Args:
            image_bytes: Original image as bytes
            points: List of detected points with x, y, class
            pulse_factor: Animation factor for pulsing effect (0.5-1.0)
            
        Returns:
            Image bytes with points drawn
        """
        try:
            if not points:
                logger.info("No points to draw, returning original image")
                return image_bytes
                
            # Open and prepare image
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            draw = ImageDraw.Draw(image)
            
            # Load font
            font = ImageOverlayProcessor.load_font(12)
            
            # Class color mapping
            class_color_map = {}
            
            logger.info(f"Drawing {len(points)} points on {image.size} image")
            
            for i, point in enumerate(points):
                try:
                    # Get normalized coordinates (0-1 range)
                    x = float(point.get('x', 0.5))
                    y = float(point.get('y', 0.5))
                    class_name = point.get('class', f'Point_{i}')
                    score = point.get('score', 0.0)
                    
                    # Validate coordinates
                    if not (0 <= x <= 1 and 0 <= y <= 1):
                        logger.info(f"Point coordinates out of range: ({x}, {y})")
                        x = max(0, min(1, x))
                        y = max(0, min(1, y))
                    
                    # Scale to image size
                    scaled_x = x * image.size[0]
                    scaled_y = y * image.size[1]
                    
                    # Get color for this class
                    color = ImageOverlayProcessor.get_class_color(class_name, class_color_map)
                    
                    # Draw outer pulsing circle (simulating animation)
                    pulse_radius = int(20 * pulse_factor)
                    
                    # Create a semi-transparent effect by drawing multiple circles
                    for radius in range(pulse_radius, 8, -2):
                        alpha_factor = 1.0 - (radius - 8) / (pulse_radius - 8)
                        circle_color = tuple(int(c * alpha_factor + 255 * (1 - alpha_factor)) for c in color)
                        
                        draw.ellipse(
                            [(scaled_x - radius, scaled_y - radius),
                             (scaled_x + radius, scaled_y + radius)],
                            outline=circle_color,
                            width=1
                        )
                    
                    # Draw inner fixed circle
                    inner_radius = 8
                    draw.ellipse(
                        [(scaled_x - inner_radius, scaled_y - inner_radius),
                         (scaled_x + inner_radius, scaled_y + inner_radius)],
                        fill=color,
                        outline=color
                    )
                    
                    # Draw crosshairs
                    line_length = 15
                    line_width = 2
                    
                    # Horizontal line
                    draw.line(
                        [(scaled_x - line_length, scaled_y), (scaled_x + line_length, scaled_y)],
                        fill=color,
                        width=line_width
                    )
                    
                    # Vertical line
                    draw.line(
                        [(scaled_x, scaled_y - line_length), (scaled_x, scaled_y + line_length)],
                        fill=color, 
                        width=line_width
                    )
                    
                    # Draw class label with background
                    label_text = str(class_name)
                    
                    if font:
                        bbox_text = draw.textbbox((0, 0), label_text, font=font)
                        text_width = bbox_text[2] - bbox_text[0]
                        text_height = bbox_text[3] - bbox_text[1]
                    else:
                        text_width = len(label_text) * 7
                        text_height = 12
                    
                    # Position label to avoid going off-screen
                    label_x = scaled_x + 10
                    label_y = scaled_y - 20 - text_height
                    
                    if label_x + text_width > image.size[0]:
                        label_x = scaled_x - text_width - 10
                    if label_y < 0:
                        label_y = scaled_y + 20
                    
                    # Draw label background
                    draw.rectangle(
                        [(label_x - 2, label_y - 2),
                         (label_x + text_width + 4, label_y + text_height + 2)],
                        fill=color,
                        outline=None
                    )
                    
                    # Draw label text
                    draw.text(
                        (label_x, label_y),
                        label_text,
                        fill=(255, 255, 255),
                        font=font
                    )
                    
                    # Add confidence score if available
                    if score > 0:
                        score_text = f"{int(score * 100)}%"
                        if font:
                            score_bbox = draw.textbbox((0, 0), score_text, font=font)
                            score_width = score_bbox[2] - score_bbox[0]
                        else:
                            score_width = len(score_text) * 6
                        
                        draw.text(
                            (label_x, label_y + text_height + 2),
                            score_text,
                            fill=(255, 255, 255),
                            font=font
                        )
                    
                except Exception as point_error:
                    logger.error(f"Error drawing point {i}: {str(point_error)}")
                    continue
            
            # Convert back to bytes
            output_buffer = io.BytesIO()
            image.save(output_buffer, format='JPEG', quality=95, optimize=True)
            
            logger.info(f"Successfully drew points on image")
            return output_buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Error drawing points: {str(e)}")
            return image_bytes
    
    @staticmethod
    def apply_inference_overlay(image_bytes: bytes, inference_result: Dict[str, Any], 
                               inference_mode: str) -> bytes:
        """
        Apply appropriate overlay based on inference mode and results.
        This is the main entry point that delegates to specific drawing methods.
        
        Args:
            image_bytes: Original image as bytes
            inference_result: Inference results from the model
            inference_mode: Type of inference ("Point", "Detect", "VQA", "Caption")
            
        Returns:
            Image bytes with appropriate overlays applied
        """
        try:
            logger.info(f"Applying {inference_mode} overlay to image")
            
            if inference_mode == "Point":
                points = inference_result.get('points', [])
                if points:
                    logger.info(f"Drawing {len(points)} points")
                    return ImageOverlayProcessor.draw_points(image_bytes, points, pulse_factor=0.8)
                else:
                    logger.info("No points found in inference result")
                    
            elif inference_mode == "Detect":
                objects = inference_result.get('objects', [])
                if objects:
                    logger.info(f"Drawing {len(objects)} bounding boxes")
                    return ImageOverlayProcessor.draw_bounding_boxes(image_bytes, objects)
                else:
                    logger.info("No objects found in inference result")
            
            elif inference_mode in ["VQA", "Caption"]:
                # For text-based results, optionally add text overlay
                result_text = ""
                if inference_mode == "VQA":
                    result_data = inference_result.get("result", {})
                    if isinstance(result_data, dict):
                        result_text = result_data.get("answer", "")
                    else:
                        result_text = str(result_data)
                elif inference_mode == "Caption":
                    result_data = inference_result.get("result", {})
                    if isinstance(result_data, dict):
                        result_text = result_data.get("caption", "")
                    else:
                        result_text = str(result_data)
                
                if result_text:
                    return ImageOverlayProcessor.add_text_overlay(image_bytes, result_text)
            
            # Return original image if no overlays needed or no results found
            logger.info("No overlays applied, returning original image")
            return image_bytes
            
        except Exception as e:
            logger.error(f"Error applying inference overlay: {str(e)}")
            return image_bytes
    
    @staticmethod
    def add_text_overlay(image_bytes: bytes, text: str, max_length: int = 100) -> bytes:
        """
        Add text overlay to image for VQA/Caption results.
        
        Args:
            image_bytes: Original image as bytes
            text: Text to overlay
            max_length: Maximum text length to display
            
        Returns:
            Image bytes with text overlay
        """
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            draw = ImageDraw.Draw(image)
            
            # Truncate text if too long
            if len(text) > max_length:
                text = text[:max_length-3] + "..."
            
            # Load font
            font = ImageOverlayProcessor.load_font(16)
            
            # Calculate text size and position
            if font:
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            else:
                text_width = len(text) * 10
                text_height = 16
            
            # Position at bottom of image with padding
            margin = 10
            text_x = margin
            text_y = image.size[1] - text_height - margin * 2
            
            # Draw background rectangle
            bg_padding = 8
            draw.rectangle(
                [(text_x - bg_padding, text_y - bg_padding),
                 (text_x + text_width + bg_padding, text_y + text_height + bg_padding)],
                fill=(0, 0, 0, 200),  # Semi-transparent black
                outline=(255, 255, 255),
                width=2
            )
            
            # Draw text
            draw.text(
                (text_x, text_y),
                text,
                fill=(255, 255, 255),
                font=font
            )
            
            # Convert back to bytes
            output_buffer = io.BytesIO()
            image.save(output_buffer, format='JPEG', quality=95)
            
            return output_buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Error adding text overlay: {str(e)}")
            return image_bytes