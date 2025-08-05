from PIL import Image, ImageDraw, ImageFont
import io
import math
from typing import List, Dict, Any, Tuple, Optional
from firebase_functions import logger

class ConsistentColorManager:
    """
    Manages consistent color assignment matching Flutter frontend exactly.
    Colors should match device_dashboard_page.dart color scheme.
    """
    
    # Match the exact colors from Flutter - converted to RGB
    FLUTTER_DASHBOARD_COLORS = [
        (33, 150, 243),   # Colors.blue
        (244, 67, 54),    # Colors.red  
        (76, 175, 80),    # Colors.green
        (255, 152, 0),    # Colors.orange
        (156, 39, 176),   # Colors.purple
        (0, 150, 136),    # Colors.teal
        (63, 81, 181),    # Colors.indigo
        (233, 30, 99),    # Colors.pink
        (255, 193, 7),    # Colors.amber
        (0, 188, 212),    # Colors.cyan
    ]
    
    # Global class-to-color mapping for consistency across all function calls
    _global_class_colors = {}
    
    @classmethod
    def get_class_color(cls, class_name: str, all_classes: list = None) -> Tuple[int, int, int]:
        """
        Get consistent color for a class name.
        Colors are assigned in the order classes are first encountered,
        matching the Flutter dashboard behavior.
        """
        if class_name in cls._global_class_colors:
            return cls._global_class_colors[class_name]
        
        # If we have a list of all classes, use deterministic ordering
        if all_classes:
            # Sort classes for deterministic color assignment
            sorted_classes = sorted(set(all_classes))
            
            # Assign colors to any new classes in the sorted list
            for i, sorted_class in enumerate(sorted_classes):
                if sorted_class not in cls._global_class_colors:
                    color_index = i % len(cls.FLUTTER_DASHBOARD_COLORS)
                    cls._global_class_colors[sorted_class] = cls.FLUTTER_DASHBOARD_COLORS[color_index]
            
            return cls._global_class_colors.get(class_name, cls.FLUTTER_DASHBOARD_COLORS[0])
        
        # Fallback: assign next available color
        current_color_count = len(cls._global_class_colors)
        color_index = current_color_count % len(cls.FLUTTER_DASHBOARD_COLORS)
        color = cls.FLUTTER_DASHBOARD_COLORS[color_index]
        cls._global_class_colors[class_name] = color
        
        return color
    
    @classmethod
    def reset_colors(cls):
        """Reset color assignments (useful for testing)"""
        cls._global_class_colors.clear()

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
        """Draw bounding boxes with consistent Flutter colors"""
        try:
            if not objects:
                return image_bytes
                
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            draw = ImageDraw.Draw(image)
            font = ImageOverlayProcessor.load_font(14)
            small_font = ImageOverlayProcessor.load_font(12)
            
            # Extract all class names for consistent color assignment
            all_classes = [obj.get('class', f'Object_{i}') for i, obj in enumerate(objects)]
            
            for i, obj in enumerate(objects):
                try:
                    bbox = obj.get('bbox', {})
                    class_name = obj.get('class', f'Object_{i}')
                    score = obj.get('score', 0.0)
                    
                    # Get coordinates and validate
                    x1, y1, x2, y2 = bbox.get('x1', 0), bbox.get('y1', 0), bbox.get('x2', 0), bbox.get('y2', 0)
                    if x1 >= x2 or y1 >= y2:
                        continue
                    
                    # Scale coordinates to image size
                    original_size = image.size
                    scale_factor = min(original_size[0] / image_size[0], original_size[1] / image_size[1])
                    x_offset = (original_size[0] - image_size[0] * scale_factor) / 2
                    y_offset = (original_size[1] - image_size[1] * scale_factor) / 2
                    
                    scaled_x1 = max(0, min(original_size[0], x1 * scale_factor + x_offset))
                    scaled_y1 = max(0, min(original_size[1], y1 * scale_factor + y_offset))
                    scaled_x2 = max(0, min(original_size[0], x2 * scale_factor + x_offset))
                    scaled_y2 = max(0, min(original_size[1], y2 * scale_factor + y_offset))
                    
                    # Get consistent color for this class
                    color = ConsistentColorManager.get_class_color(class_name, all_classes)
                    
                    # Draw bounding box
                    draw.rectangle([(scaled_x1, scaled_y1), (scaled_x2, scaled_y2)], outline=color, width=3)
                    
                    # Draw class label with background
                    if font:
                        bbox_text = draw.textbbox((0, 0), class_name, font=font)
                        text_width = bbox_text[2] - bbox_text[0]
                        text_height = bbox_text[3] - bbox_text[1]
                    else:
                        text_width, text_height = len(class_name) * 8, 14
                    
                    label_y = max(text_height + 6, scaled_y1 - 2)
                    
                    # Label background
                    draw.rectangle([(scaled_x1, label_y - text_height - 6), 
                                   (scaled_x1 + text_width + 10, label_y - 2)], fill=color)
                    
                    # Label text
                    draw.text((scaled_x1 + 5, label_y - text_height - 4), class_name, 
                             fill=(255, 255, 255), font=font)
                    
                    # Confidence score
                    if score > 0:
                        score_text = f"{int(score * 100)}%"
                        if small_font:
                            score_bbox = draw.textbbox((0, 0), score_text, font=small_font)
                            score_width = score_bbox[2] - score_bbox[0]
                            score_height = score_bbox[3] - score_bbox[1]
                        else:
                            score_width, score_height = len(score_text) * 6, 12
                        
                        score_x = scaled_x2 - score_width - 5
                        score_y = scaled_y2 - score_height - 2
                        
                        draw.rectangle([(score_x - 3, score_y - 2),
                                       (score_x + score_width + 3, score_y + score_height + 2)], fill=color)
                        draw.text((score_x, score_y), score_text, fill=(255, 255, 255), font=small_font)
                        
                except Exception as obj_error:
                    logger.error(f"Error drawing object {i}: {obj_error}")
                    continue
            
            # Return processed image
            output_buffer = io.BytesIO()
            image.save(output_buffer, format='JPEG', quality=95, optimize=True)
            return output_buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Error drawing bounding boxes: {e}")
            return image_bytes

    @staticmethod
    def draw_points(image_bytes: bytes, points: List[Dict[str, Any]], pulse_factor: float = 1.0) -> bytes:
        """Draw points with consistent Flutter colors"""
        try:
            if not points:
                return image_bytes
                
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            draw = ImageDraw.Draw(image)
            font = ImageOverlayProcessor.load_font(12)
            
            # Extract all class names for consistent color assignment  
            all_classes = [point.get('class', f'Point_{i}') for i, point in enumerate(points)]
            
            for i, point in enumerate(points):
                try:
                    x = max(0, min(1, float(point.get('x', 0.5))))
                    y = max(0, min(1, float(point.get('y', 0.5))))
                    class_name = point.get('class', f'Point_{i}')
                    
                    # Scale to image size
                    scaled_x = x * image.size[0]
                    scaled_y = y * image.size[1]
                    
                    # Get consistent color
                    color = ConsistentColorManager.get_class_color(class_name, all_classes)
                    
                    # Draw pulsing outer circle
                    pulse_radius = int(20 * pulse_factor)
                    for radius in range(pulse_radius, 8, -2):
                        alpha_factor = 1.0 - (radius - 8) / (pulse_radius - 8)
                        circle_color = tuple(int(c * alpha_factor + 255 * (1 - alpha_factor)) for c in color)
                        draw.ellipse([(scaled_x - radius, scaled_y - radius),
                                     (scaled_x + radius, scaled_y + radius)], outline=circle_color, width=1)
                    
                    # Inner circle
                    inner_radius = 8
                    draw.ellipse([(scaled_x - inner_radius, scaled_y - inner_radius),
                                 (scaled_x + inner_radius, scaled_y + inner_radius)], fill=color, outline=color)
                    
                    # Crosshairs
                    line_length, line_width = 15, 2
                    draw.line([(scaled_x - line_length, scaled_y), (scaled_x + line_length, scaled_y)], 
                             fill=color, width=line_width)
                    draw.line([(scaled_x, scaled_y - line_length), (scaled_x, scaled_y + line_length)], 
                             fill=color, width=line_width)
                    
                    # Class label
                    if font:
                        bbox_text = draw.textbbox((0, 0), class_name, font=font)
                        text_width = bbox_text[2] - bbox_text[0]
                        text_height = bbox_text[3] - bbox_text[1]
                    else:
                        text_width, text_height = len(class_name) * 7, 12
                    
                    label_x = min(scaled_x + 10, image.size[0] - text_width - 10)
                    label_y = max(0, scaled_y - 20 - text_height)
                    
                    draw.rectangle([(label_x - 2, label_y - 2),
                                   (label_x + text_width + 4, label_y + text_height + 2)], fill=color)
                    draw.text((label_x, label_y), class_name, fill=(255, 255, 255), font=font)
                    
                except Exception as point_error:
                    logger.error(f"Error drawing point {i}: {point_error}")
                    continue
            
            output_buffer = io.BytesIO()
            image.save(output_buffer, format='JPEG', quality=95, optimize=True)
            return output_buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Error drawing points: {e}")
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